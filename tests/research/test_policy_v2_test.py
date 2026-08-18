from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import timelymt.research.policy_v2_test as gate
from timelymt.research.policy_v2 import (
    DATASET_CHECKSUM, ENCODER_REVISION, EXPERIMENT_STATUS, LOCAL_RUNTIME, SPLIT_CHECKSUM,
    TRANSLATOR_FINGERPRINT,
)


def prediction(strategy: str = "v2_P2_0.50", talk_id: str = "test-talk", model_hash: str = "model"):
    return {
        "strategy": strategy, "talk_id": talk_id, "split": "test", "artifact_status": "full",
        "experiment_status": EXPERIMENT_STATUS, "model_sha256": model_hash,
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator_fingerprint": TRANSLATOR_FINGERPRINT, "encoder_revision": ENCODER_REVISION,
        "runtime": LOCAL_RUNTIME, "test_plan_sha256": "plan", "primary_strategy": gate.PRIMARY_STRATEGY,
        "commits": [{}],
    }


class PolicyV2TestPlanTests(unittest.TestCase):
    def test_plan_contains_only_locked_strategies(self):
        plan = gate._validate_test_plan()
        self.assertEqual(plan["primary_strategy"], "v2_P2_0.50")
        self.assertEqual(plan["history_ablation"], ["v2_P0_0.50", "v2_P1_0.50", "v2_P2_0.50"])
        self.assertEqual(
            plan["reference_systems"],
            ["learned_P1_0.60", "local_agreement_la2", "fixed_n_8", "fixed_time_3200"],
        )
        self.assertNotIn("threshold_search", plan["pipeline"])

    def test_missing_frozen_v2_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(gate, "V2_FROZEN", Path(directory) / "missing.json"):
                with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
                    gate._validate_v2()

    def test_wrong_selected_strategy_is_rejected(self):
        frozen = {
            "artifact_status": "v2-dev-frozen-complete", "experiment_status": EXPERIMENT_STATUS,
            "selected_strategy": "v2_P1_0.50", "dataset_checksum": DATASET_CHECKSUM,
            "split_checksum": SPLIT_CHECKSUM, "translator_fingerprint": TRANSLATOR_FINGERPRINT,
            "test_status": "UNTOUCHED", "checkpoint_hashes": gate.FROZEN_V2_HASHES,
        }
        with patch.object(gate, "_load_json", return_value=frozen):
            with self.assertRaisesRegex(RuntimeError, "config identity mismatch"):
                gate._validate_v2()

    def test_checkpoint_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ("P0", "P1", "P2"):
                (root / f"V2{variant}.pt").write_bytes(b"wrong")
                (root / f"V2{variant}.metadata.json").write_text(
                    json.dumps({"checkpoint_sha256": gate.FROZEN_V2_HASHES[variant]}), encoding="utf-8",
                )
            with patch.object(gate, "V2_CHECKPOINTS", root), patch(
                "timelymt.research.policy_v2_runner._valid_checkpoint", return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint hash"):
                    gate._validate_checkpoints()

    def test_test_pseudo_label_artifact_keeps_gate_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            pseudo = Path(directory) / "test"
            pseudo.mkdir()
            with patch.object(gate, "PSEUDO_TEST", pseudo), patch.object(gate, "_validate_v1", return_value={}), patch.object(
                gate, "_validate_v2",
            ), patch.object(gate, "_validate_test_plan"), patch.object(gate, "_validate_checkpoints"), patch.object(
                gate, "_validate_data_and_translator",
            ):
                with self.assertRaisesRegex(RuntimeError, "pseudo-label"):
                    gate.validate_test_gate()

    def test_gate_exposes_no_search_selection_or_training(self):
        source = (Path(__file__).parents[2] / "src/timelymt/research/policy_v2_test.py").read_text(encoding="utf-8")
        self.assertNotIn("select_v2_configuration", source)
        self.assertNotIn("select_dev_configuration", source)
        self.assertNotIn("train_v2", source)
        self.assertNotIn("generate_pseudo_labels", source)
        self.assertNotIn("for threshold in", source)


class PolicyV2TestExecutionSafetyTests(unittest.TestCase):
    def test_valid_prediction_resumes_and_invalid_prediction_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.json"
            path.write_text("{}", encoding="utf-8")
            with patch.object(gate, "_validate_test_prediction"):
                self.assertTrue(gate._resume_prediction(path, "v2_P2_0.50", "test-talk", "model"))
                self.assertTrue(path.exists())
            path.write_text("{}", encoding="utf-8")
            with patch.object(gate, "_validate_test_prediction", side_effect=RuntimeError("invalid")):
                self.assertFalse(gate._resume_prediction(path, "v2_P2_0.50", "test-talk", "model"))
                self.assertFalse(path.exists())

    def test_test_evaluation_requires_full_expected_prediction_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            strategy = "v2_P2_0.50"
            prediction_dir = experiment / "predictions/test" / strategy
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "talk-a.json").write_text(json.dumps(prediction(talk_id="talk-a")), encoding="utf-8")
            with patch.object(gate, "EXPERIMENT", experiment):
                with self.assertRaisesRegex(RuntimeError, "full exact prediction coverage"):
                    gate._prediction_records(strategy, {"talk-a", "talk-b"}, "model")

    def test_complete_metrics_resume_without_loading_references(self):
        required = {
            "BLEU": 1.0, "chrF2": 2.0, "token_level_average_lagging": 3.0,
            "token_level_length_adaptive_average_lagging": 4.0, "number_of_commits": 5.0,
            "commits_per_100_source_tokens": 6.0, "per_talk": [{"talk_id": "test-talk"}],
            "artifact_status": "full", "experiment_status": EXPERIMENT_STATUS,
            "experiment_label": gate.EXPERIMENT_LABEL, "dataset_checksum": DATASET_CHECKSUM,
            "split_checksum": SPLIT_CHECKSUM, "primary_strategy": gate.PRIMARY_STRATEGY,
            "test_plan_sha256": "plan",
        }
        metrics = {strategy: {**required, "model_sha256": "model"} for strategy in gate.TEST_STRATEGIES}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics/test/all.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with patch.object(gate, "EXPERIMENT", Path(directory)), patch.object(
                gate, "_model_hash", return_value="model",
            ), patch.object(gate, "sha256_file", return_value="plan"):
                self.assertEqual(gate._resumable_metrics({"test-talk"}, {}), metrics)

    def test_runner_calls_only_frozen_prediction_evaluation_and_reporting(self):
        metrics = {"v2_P2_0.50": {}}
        with patch.object(gate, "validate_test_gate") as validate, patch.object(
            gate, "_rollout_references",
        ) as references, patch.object(gate, "_rollout_v2") as rollout_v2, patch.object(
            gate, "_evaluate_test", return_value=metrics,
        ) as evaluate, patch.object(gate, "_write_summary") as report:
            gate.run_test()
        validate.assert_called_once_with()
        references.assert_called_once_with()
        self.assertEqual([call.args[0] for call in rollout_v2.call_args_list], list(gate.HISTORY_ABLATION))
        evaluate.assert_called_once_with()
        report.assert_called_once_with(metrics)


if __name__ == "__main__":
    unittest.main()
