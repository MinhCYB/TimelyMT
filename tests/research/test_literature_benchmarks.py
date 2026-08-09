from __future__ import annotations

from dataclasses import asdict
import inspect
from pathlib import Path
import tempfile
import unittest

from timelymt.data.translation_artifacts import RuntimeSourceToken, RuntimeTalk, TranslationHypothesis
from timelymt.research.evaluation import (
    average_lagging,
    latency_metrics,
    length_adaptive_average_lagging,
)
from timelymt.research.meaningful_units import (
    MU_NUMERIC_FEATURES,
    MU_TEXT_FEATURES,
    flatten_mu_state,
    generate_mu_supervision,
    mu_rollout,
    train_mu_policy,
)
from timelymt.research.cli import train_mu
from timelymt.research.streaming import local_agreement_la2, prediction_record


def talk(count: int, split: str = "train") -> RuntimeTalk:
    return RuntimeTalk("talk", split, tuple(
        RuntimeSourceToken("talk", f"t{index}", index, f"w{index}", index * 500)
        for index in range(count)
    ))


class Provider:
    def __init__(self, texts: dict[tuple[int, int], str]) -> None:
        self.texts = texts
        self.calls: list[tuple[int, int]] = []

    def __call__(self, runtime_talk: RuntimeTalk, start: int, end: int) -> TranslationHypothesis:
        if end > runtime_talk.latest_observed_token_index:
            raise AssertionError("future source access")
        self.calls.append((start, end))
        text = self.texts[(start, end)]
        return TranslationHypothesis(
            "1.0.0", f"r{start}-{end}", runtime_talk.talk_id, runtime_talk.split,
            start, end, runtime_talk.tokens[end].emit_ms,
            " ".join(token.text for token in runtime_talk.tokens[start:end + 1]),
            text, end - start + 1, len(text.split()), "m", "r", "c", "f", "g",
            None, None, None,
        )


class LiteratureLocalAgreementTests(unittest.TestCase):
    def test_lcp_emission_revision_and_deterministic_flush(self):
        provider = Provider({
            (0, 3): "Tôi muốn đi đến trường",
            (0, 4): "Tôi muốn đi đến công viên",
            (0, 5): "Tôi muốn đi đến công viên hôm nay",
        })
        commits = local_agreement_la2(talk(6), provider)
        self.assertEqual([commit.translated_text for commit in commits], [
            "Tôi muốn đi đến", "công viên hôm nay",
        ])
        self.assertEqual([commit.reason for commit in commits], ["agreement", "talk_end"])
        self.assertEqual(" ".join(commit.translated_text for commit in commits), "Tôi muốn đi đến công viên hôm nay")
        self.assertEqual(provider.calls, [(0, 3), (0, 4), (0, 5)])
        self.assertEqual(commits[0].observation_token_index, 4)
        self.assertEqual(commits[1].observation_token_index, 5)

    def test_zero_lcp_complete_agreement_and_exact_tokens(self):
        zero = local_agreement_la2(talk(5), Provider({(0, 3): "A, b", (0, 4): "a, b"}))
        self.assertEqual([(item.reason, item.translated_text) for item in zero], [("talk_end", "a, b")])
        complete = local_agreement_la2(talk(6), Provider({
            (0, 3): "Xin Chào!", (0, 4): "Xin Chào!", (0, 5): "Xin Chào! bạn",
        }))
        self.assertEqual([item.translated_text for item in complete], ["Xin Chào!", "bạn"])

    def test_maximum_source_unit_flushes_and_resets(self):
        class EchoProvider:
            def __init__(self):
                self.calls = []

            def __call__(self, runtime_talk, start, end):
                self.calls.append((start, end))
                text = f"unit-{start} through-{end}"
                return TranslationHypothesis(
                    "1.0.0", f"r{start}-{end}", runtime_talk.talk_id, runtime_talk.split,
                    start, end, runtime_talk.tokens[end].emit_ms, "source", text,
                    end - start + 1, 2, "m", "r", "c", "f", "g", None, None, None,
                )

        provider = EchoProvider()
        commits = local_agreement_la2(talk(52), provider)
        self.assertIn((0, 47), provider.calls)
        self.assertEqual(provider.calls[-1], (48, 51))
        self.assertEqual(commits[-2].reason, "max_length")
        self.assertEqual(commits[-1].reason, "talk_end")

    def test_la2_prediction_has_no_reference(self):
        runtime = talk(5)
        commits = local_agreement_la2(runtime, Provider({(0, 3): "a", (0, 4): "a b"}))
        record = prediction_record("local_agreement_la2", runtime, commits)
        self.assertNotIn("reference", record)


class MeaningfulUnitTests(unittest.TestCase):
    def test_independent_oracle_labels_and_causal_features(self):
        provider = Provider({
            (0, 3): "a x", (0, 5): "a b c", (0, 4): "a b", (2, 5): "c d",
        })
        rows = generate_mu_supervision(talk(6), provider)
        self.assertEqual([row["label"] for row in rows[:2]], ["LISTEN", "COMMIT"])
        self.assertEqual(rows[1]["label_reason"], "meaningful_unit")
        self.assertIn("mu_oracle_training_only", rows[0])
        self.assertNotIn("oracle_training_only", rows[0])
        self.assertEqual(set(rows[0]["causal"]), {*MU_TEXT_FEATURES, "numeric"})
        self.assertEqual(set(rows[0]["causal"]["numeric"]), set(MU_NUMERIC_FEATURES))
        flattened = flatten_mu_state(rows[0]["causal"])
        self.assertFalse(any(term in key for key in flattened for term in ("previous", "history", "future", "gold", "oracle")))

    def test_test_supervision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forbidden"):
            generate_mu_supervision(talk(4, "test"), Provider({(0, 3): "a"}))

    def test_final_training_rejects_smoke_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                '{"artifact_type":"mu_zhang2020_supervision","artifact_status":"smoke",'
                '"split":"train","dataset_checksum":"6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce",'
                '"split_checksum":"aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "refuses smoke"):
                train_mu(path)

    def test_fit_reload_and_sequential_causal_rollout(self):
        import joblib

        provider = Provider({
            (0, 3): "a x", (0, 5): "a b c", (0, 4): "a b", (2, 5): "c d",
        })
        rows = generate_mu_supervision(talk(6), provider)
        policy = train_mu_policy(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mu.joblib"
            joblib.dump(policy, path)
            restored = joblib.load(path)
            probability = restored.predict_commit_probability(rows[0]["causal"])
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

        class AlwaysCommit:
            def __init__(self) -> None:
                self.states = []

            def predict_commit_probability(self, state):
                self.states.append(state)
                return 1.0

        runtime_provider = Provider({(0, 3): "a", (4, 5): "b"})
        runtime_policy = AlwaysCommit()
        commits = mu_rollout(talk(6), runtime_provider, runtime_policy)
        self.assertEqual([(item.source_start, item.source_end) for item in commits], [(0, 3), (4, 5)])
        self.assertEqual(runtime_provider.calls, [(0, 3), (4, 5)])
        self.assertFalse(any("previous" in str(state).lower() for state in runtime_policy.states))


class LAALTests(unittest.TestCase):
    def test_hand_computed_length_cases_and_al_unchanged(self):
        emissions = [2, 2, 4, 4]
        self.assertAlmostEqual(length_adaptive_average_lagging(4, emissions, 4), average_lagging(4, emissions))
        self.assertAlmostEqual(average_lagging(4, emissions[:2]), 1.0)
        self.assertAlmostEqual(length_adaptive_average_lagging(4, emissions[:2], 4), 1.5)
        self.assertAlmostEqual(length_adaptive_average_lagging(4, emissions, 2), average_lagging(4, emissions))
        over = [2, 2, 4, 4, 4, 4]
        self.assertAlmostEqual(average_lagging(4, over), 2.0)
        self.assertAlmostEqual(length_adaptive_average_lagging(4, over, 4), 2.0)

    def test_reference_is_evaluator_only_and_corpus_is_token_weighted(self):
        self.assertEqual(list(inspect.signature(length_adaptive_average_lagging).parameters), [
            "source_length", "target_emissions", "reference_length",
        ])
        self.assertNotIn("reference", inspect.signature(local_agreement_la2).parameters)
        self.assertNotIn("reference", inspect.signature(mu_rollout).parameters)
        records = [
            {"talk_id": "a", "source_token_count": 4, "commits": [{
                "source_token_count": 2, "source_clock_duration_ms": 500,
                "observation_token_index": 1, "observation_emit_ms": 500,
                "translated_text": "x", "reason": "agreement",
            }]},
            {"talk_id": "b", "source_token_count": 4, "commits": [{
                "source_token_count": 4, "source_clock_duration_ms": 1500,
                "observation_token_index": 3, "observation_emit_ms": 1500,
                "translated_text": "x y z", "reason": "talk_end",
            }]},
        ]
        metrics = latency_metrics(records, {"a": "r s", "b": "r s"})
        expected = (
            length_adaptive_average_lagging(4, [2], 2)
            + 3 * length_adaptive_average_lagging(4, [4, 4, 4], 2)
        ) / 4
        self.assertAlmostEqual(metrics["token_level_length_adaptive_average_lagging"], expected)
        self.assertIn("token_level_average_lagging", metrics)


if __name__ == "__main__":
    unittest.main()
