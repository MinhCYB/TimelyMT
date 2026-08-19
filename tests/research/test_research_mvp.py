from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from timelymt.data.translation_artifacts import RuntimeSourceToken, RuntimeTalk, TranslationHypothesis
from timelymt.research.evaluation import average_lagging, latency_metrics, quality_metrics
from timelymt.research.policy import flatten_state, train_policy
from timelymt.research.pseudo_labels import generate_pseudo_labels
from timelymt.research.streaming import (
    causal_state, fixed_n, fixed_time, learned_rollout, local_agreement_style, lcp_length, prediction_record,
    select_dev_configuration,
)
from timelymt.research.cli import _validate_pseudo_talk_file, train
from timelymt.research.cli import main as cli_main
from timelymt.research.policy_p3_global_runner import rollout_p3
from scripts.validate_demo_trace import validate as validate_demo_trace


def talk(count: int, emits: list[int] | None = None, split: str = "train") -> RuntimeTalk:
    emits = emits or [index * 500 for index in range(count)]
    return RuntimeTalk("talk", split, tuple(RuntimeSourceToken("talk", f"t{index}", index, f"w{index}", emits[index]) for index in range(count)))


class Provider:
    def __init__(self, texts: dict[tuple[int, int], str] | None = None) -> None:
        self.texts = texts or {}
        self.calls: list[tuple[int, int]] = []

    def __call__(self, runtime_talk: RuntimeTalk, start: int, end: int) -> TranslationHypothesis:
        self.calls.append((start, end))
        text = self.texts.get((start, end), " ".join(f"v{i}" for i in range(start, end + 1)))
        return TranslationHypothesis("1.0.0", f"r{start}-{end}", runtime_talk.talk_id, runtime_talk.split, start, end, runtime_talk.tokens[end].emit_ms, " ".join(t.text for t in runtime_talk.tokens[start:end + 1]), text, end-start+1, len(text.split()), "m", "r", "c", "f", "g", None, None, None)


class UnstableProvider(Provider):
    def __call__(self, runtime_talk: RuntimeTalk, start: int, end: int) -> TranslationHypothesis:
        self.texts[(start, end)] = f"changing-{end}"
        return super().__call__(runtime_talk, start, end)


class ThresholdPolicy:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.states = []

    def predict_commit_probability(self, state):
        self.states.append(state)
        return self.probability


class SequencePolicy(ThresholdPolicy):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__(0.0)
        self.probabilities = probabilities

    def predict_commit_probability(self, state):
        self.states.append(state)
        return self.probabilities[len(self.states) - 1]


class StreamingTests(unittest.TestCase):
    def test_fixed_n_boundaries_and_remainder(self):
        commits = fixed_n(talk(10), Provider(), 4)
        self.assertEqual([(c.source_start, c.source_end, c.reason) for c in commits], [(0, 3, "fixed_n"), (4, 7, "fixed_n"), (8, 9, "talk_end")])

    def test_fixed_time_uses_only_current_timestamp(self):
        provider = Provider()
        commits = fixed_time(talk(7, [0, 400, 800, 1200, 1600, 2000, 9000]), provider, 1600)
        self.assertEqual((commits[0].source_start, commits[0].source_end), (0, 4))
        self.assertNotIn((0, 3), provider.calls)

    def test_local_agreement_k2_k3_and_forced_remainder(self):
        texts = {(0, 3): "a b", (0, 4): "a b c", (0, 5): "a b c d", (5, 5): "z"}
        self.assertEqual(local_agreement_style(talk(6), Provider(texts), 2)[0].source_end, 4)
        self.assertEqual(local_agreement_style(talk(6), Provider(texts), 3)[0].source_end, 5)
        self.assertEqual(lcp_length(["A, b", "A, b c"]), 2)

    def test_learned_min_threshold_max_and_own_history(self):
        policy = ThresholdPolicy(1.0)
        commits = learned_rollout(talk(9), Provider(), policy, 0.5)
        self.assertEqual([c.source_token_count for c in commits], [4, 4, 1])
        self.assertEqual(len(policy.states), 3)
        self.assertEqual(policy.states[1]["previous_committed_target_text"], commits[0].translated_text)
        max_commits = learned_rollout(talk(49), Provider(), ThresholdPolicy(0.0), 0.5)
        self.assertEqual([c.source_token_count for c in max_commits], [48, 1])

    def test_prediction_concatenation_is_deterministic(self):
        runtime = talk(5)
        commits = fixed_n(runtime, Provider({(0, 3): "xin chao", (4, 4): "ban"}), 4)
        record = prediction_record("fixed_n_4", runtime, commits)
        self.assertEqual(record["prediction"], "xin chao ban")
        self.assertNotIn("reference", record)

    def test_trace_is_observability_only_and_causal(self):
        runtime = talk(5)
        provider_off, provider_on = Provider(), Provider()
        policy_off, policy_on = SequencePolicy([0.1, 0.9]), SequencePolicy([0.1, 0.9])
        without_trace = learned_rollout(runtime, provider_off, policy_off, 0.5)
        events = []
        with_trace = learned_rollout(runtime, provider_on, policy_on, 0.5, events.append)
        self.assertEqual(with_trace, without_trace)
        self.assertEqual(provider_on.calls, provider_off.calls)
        self.assertEqual(len(policy_on.states), len(policy_off.states))
        self.assertEqual([event["event_index"] for event in events], list(range(5)))
        self.assertEqual([event["source_token_end"] for event in events], list(range(5)))
        self.assertEqual([event["observation_ms"] for event in events], [0, 500, 1000, 1500, 2000])
        waits = events[:3]
        self.assertTrue(all(event["decision"] == "WAIT" for event in waits))
        self.assertTrue(all(event["candidate_translation"] is None and event["p_commit"] is None and event["numeric_features"] is None for event in waits))
        self.assertNotIn((0, 0), provider_on.calls)
        self.assertEqual(len(policy_on.states), 2)
        listen, commit = events[3:]
        self.assertEqual((listen["decision"], listen["decision_reason"]), ("LISTEN", "below_threshold"))
        self.assertIsNotNone(listen["candidate_translation"])
        self.assertLess(listen["p_commit"], listen["threshold"])
        self.assertEqual((commit["decision"], commit["decision_reason"]), ("COMMIT", "policy"))
        self.assertGreaterEqual(commit["p_commit"], commit["threshold"])
        numeric_names = set(commit["numeric_features"])
        self.assertEqual(len(numeric_names), 11)
        self.assertTrue(all(event["candidate_source_end"] == event["source_token_end"] for event in events))
        self.assertTrue(all(f"w{event['source_token_end'] + 1}" not in event["candidate_source_text"].split() for event in events if event["source_token_end"] + 1 < 5))
        trace_commits = [event for event in events if event["decision"] == "COMMIT"]
        self.assertEqual(len(trace_commits), len(with_trace))
        for event, commit_record in zip(trace_commits, with_trace):
            self.assertEqual((event["candidate_source_start"], event["candidate_source_end"]), (commit_record.source_start, commit_record.source_end))
            self.assertEqual(event["committed_target_text"], commit_record.translated_text)
            self.assertEqual(event["observation_ms"], commit_record.observation_emit_ms)
            self.assertEqual(event["decision_reason"], commit_record.reason)

    def test_trace_represents_short_talk_end_and_max_length(self):
        short_events = []
        short_commits = learned_rollout(talk(5), Provider(), ThresholdPolicy(1.0), 0.5, short_events.append)
        self.assertEqual([event["event_index"] for event in short_events], list(range(5)))
        self.assertEqual((short_events[-1]["decision"], short_events[-1]["decision_reason"], short_events[-1]["candidate_source_start"]), ("COMMIT", "talk_end", 4))
        self.assertEqual(short_events[-1]["committed_target_text"], short_commits[-1].translated_text)
        max_events = []
        learned_rollout(talk(49), Provider(), ThresholdPolicy(0.1), 0.5, max_events.append)
        maximum = max_events[47]
        self.assertEqual((maximum["decision"], maximum["decision_reason"], maximum["is_forced"]), ("COMMIT", "max_length", True))
        self.assertEqual(maximum["p_commit"], 0.1)

    def test_trace_source_clock_is_independent_of_policy_decisions(self):
        real_events, zero_events = [], []
        learned_rollout(talk(9), Provider(), ThresholdPolicy(1.0), 0.5, real_events.append)
        learned_rollout(talk(9), Provider(), ThresholdPolicy(0.0), 0.5, zero_events.append)
        self.assertEqual(
            [(event["event_index"], event["source_token_end"], event["observation_ms"]) for event in real_events],
            [(event["event_index"], event["source_token_end"], event["observation_ms"]) for event in zero_events],
        )

    def test_read_only_trace_validator_accepts_synthetic_trace(self):
        events = []
        runtime = talk(5, split="dev")
        learned_rollout(runtime, Provider(), SequencePolicy([0.1, 0.9]), 0.5, events.append)
        validate_demo_trace({
            "artifact_version": "demo-policy-trace-v1", "talk_id": runtime.talk_id, "split": "dev",
            "strategy": "p3_global_0.50", "threshold": 0.5, "prepared_context_mode": "zero",
            "checkpoint_sha256": "synthetic", "source_token_count": len(runtime.tokens),
            "source_final_emit_ms": runtime.tokens[-1].emit_ms,
            "prepared_context": {"prepared_context_effective_embedding_norm": 0.0},
            "events": events,
        })


class P3TraceCliTests(unittest.TestCase):
    def test_trace_rollout_constraints_are_checked_before_model_loading(self):
        output = Path("trace.json")
        for split, thresholds, talk_id, pattern in (
            ("train", [0.60], "talk", "DEV only"),
            ("test", [0.60], "talk", "permits TRAIN/DEV"),
            ("dev", [0.60], None, "talk-id"),
            ("dev", [0.50, 0.60], "talk", "one threshold"),
        ):
            with self.subTest(split=split, thresholds=thresholds, talk_id=talk_id):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    rollout_p3(split, thresholds, talk_id=talk_id, trace_output=output)

    def test_trace_output_is_optional_and_cli_validates_before_rollout(self):
        with patch("timelymt.research.policy_p3_global_runner.rollout_p3") as rollout:
            cli_main(["rollout-p3", "--split", "dev", "--talk-id", "talk", "--thresholds", "0.60"])
        self.assertIsNone(rollout.call_args.kwargs["trace_output"])
        with self.assertRaises(SystemExit):
            cli_main(["rollout-p3", "--split", "test", "--talk-id", "talk", "--thresholds", "0.60", "--trace-output", "trace.json"])
        with self.assertRaises(SystemExit):
            cli_main(["rollout-p3", "--split", "dev", "--thresholds", "0.60", "--trace-output", "trace.json"])
        with self.assertRaises(SystemExit):
            cli_main(["rollout-p3", "--split", "dev", "--talk-id", "talk", "--thresholds", "0.50", "0.60", "--trace-output", "trace.json"])


class PseudoAndFeatureTests(unittest.TestCase):
    def test_earliest_future_stability_and_oracle_exclusion(self):
        texts = {(0, 3): "a b", (0, 4): "a b", (0, 5): "a b", (4, 5): "c"}
        rows = generate_pseudo_labels(talk(6), Provider(texts))
        self.assertEqual(rows[0]["label_reason"], "stability")
        self.assertEqual(rows[0]["state_source_end"], 3)
        self.assertNotIn("future", str(rows[0]["causal"]).lower())
        self.assertNotIn("oracle", str(rows[0]["causal"]).lower())
        self.assertEqual(rows[-1]["label_reason"], "talk_end")

    def test_test_pseudo_labels_forbidden(self):
        with self.assertRaises(ValueError):
            generate_pseudo_labels(talk(4, split="test"), Provider())

    def test_max_length_is_forced(self):
        rows = generate_pseudo_labels(talk(49), UnstableProvider())
        forced = [row for row in rows if row["label_reason"] == "max_length"]
        self.assertEqual(len(forced), 1)
        self.assertEqual(forced[0]["causal"]["numeric"]["source_buffer_token_count"], 48.0)

    def test_variant_history_ablation(self):
        state = causal_state(talk(4), 0, 3, "v", "", [])
        p0, p1, p2 = (flatten_state(state, variant) for variant in ("P0", "P1", "P2"))
        self.assertNotIn("previous_source_text", p0)
        self.assertIn("previous_source_text", p1)
        self.assertNotIn("previous_target_text", p1)
        self.assertIn("previous_target_text", p2)
        self.assertFalse(any("future" in key or "gold" in key for key in p2))

    def test_resumable_validation_checks_feature_names_not_source_text(self):
        row = generate_pseudo_labels(talk(4), Provider())[0]
        row["causal"]["current_source_text"] = "an exciting future"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "talk.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(_validate_pseudo_talk_file(path, "talk", "train"), [row])
            row["causal"]["future_reference"] = "leak"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "leaked"):
                _validate_pseudo_talk_file(path, "talk", "train")


class EvaluationTests(unittest.TestCase):
    def test_average_lagging_hand_cases(self):
        self.assertAlmostEqual(average_lagging(4, [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(average_lagging(4, [2, 2, 4, 4]), 5 / 3)

    def test_latency_metrics_preserves_al(self):
        runtime = talk(4)
        record = prediction_record("fixed_n_4", runtime, fixed_n(runtime, Provider(), 4))
        metrics = latency_metrics([record], {"talk": "r0 r1 r2 r3"})
        self.assertAlmostEqual(metrics["token_level_average_lagging"], 4.0)

    def test_sacrebleu_fixture(self):
        records = [{"talk_id": "x", "prediction": "xin chao"}]
        metrics = quality_metrics(records, {"x": "xin chao"})
        self.assertEqual(metrics["chrF2"], 100.0)
        self.assertIn("version:", metrics["BLEU_signature"])

    def test_dev_selection_rule_and_ties(self):
        metrics = {
            "fixed_n_8": {"token_level_average_lagging": 4.0, "chrF2": 0.0, "BLEU": 0.0},
            "learned_P0_0.30": {"token_level_average_lagging": 3.5, "chrF2": 20.0, "BLEU": 10.0},
            "learned_P1_0.40": {"token_level_average_lagging": 4.0, "chrF2": 21.0, "BLEU": 9.0},
            "learned_P2_0.70": {"token_level_average_lagging": 4.0, "chrF2": 21.0, "BLEU": 9.0},
        }
        self.assertEqual(select_dev_configuration(metrics)["selected_strategy"], "learned_P2_0.70")


class TrainingProtectionTests(unittest.TestCase):
    def test_partial_manifest_is_rejected_even_with_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"artifact_status":"partial","split":"train","dataset_checksum":"6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce","split_checksum":"aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4"}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partial"):
                train(path, "P0", allow_smoke=True)

    def test_checkpoint_round_trip_and_predict_proba(self):
        import joblib
        states = []
        runtime = talk(5)
        for index, label in enumerate(("LISTEN", "COMMIT", "LISTEN", "COMMIT")):
            state = causal_state(runtime, 0, 3, f"v{index}", f"v{index-1}", [])
            states.append({"causal": state, "label": label})
        policy = train_policy(states, "P0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.joblib"
            joblib.dump(policy, path)
            restored = joblib.load(path)
            probability = restored.predict_commit_probability(states[0]["causal"])
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
