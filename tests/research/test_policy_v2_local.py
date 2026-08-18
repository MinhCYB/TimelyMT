from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from timelymt.research.policy_v2 import DATASET_CHECKSUM, SPLIT_CHECKSUM, TRANSLATOR_FINGERPRINT
from timelymt.research.policy_v2_local import (
    ENV_NAME, LocalRunLock, Runner, THRESHOLD_ARGS, gpu_preflight_command,
    orchestration_commands, reconciliation_action, reconcile_v1, research_command, rollout_command,
    train_command,
)
from timelymt.research.policy_v2_runner import _frozen_document_matches


def write_tree(directory: Path, manifest: dict, content: str = "row\n") -> None:
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "talk.jsonl").write_text(content, encoding="utf-8")


class V1LocalReconciliationTests(unittest.TestCase):
    def root(self, base: Path) -> Path:
        source = base / "docs/archive/timelymt-checkpoint/data/policy/pseudo_labels/train"
        write_tree(source, {"artifact_status": "full", "identity": "frozen"}, "frozen\n")
        return base

    def test_partial_train_is_moved_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(Path(directory))
            destination = root / "data/policy/pseudo_labels/train"
            write_tree(destination, {"artifact_status": "partial"}, "partial\n")
            self.assertEqual(reconcile_v1(root), "backup")
            backups = list((root / "data/policy/pseudo_labels/_local_backup").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "talk.jsonl").read_text(encoding="utf-8"), "partial\n")
            self.assertFalse(destination.exists())

    def test_full_mismatch_hard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(Path(directory))
            write_tree(root / "data/policy/pseudo_labels/train", {"artifact_status": "full"}, "different\n")
            with self.assertRaisesRegex(RuntimeError, "claims artifact_status=full"):
                reconcile_v1(root)

    def test_exact_frozen_train_skips_and_missing_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(Path(directory))
            self.assertEqual(reconciliation_action(root), "import")
            source = root / "docs/archive/timelymt-checkpoint/data/policy/pseudo_labels/train"
            destination = root / "data/policy/pseudo_labels/train"
            destination.parent.mkdir(parents=True)
            import shutil
            shutil.copytree(source, destination)
            self.assertEqual(reconciliation_action(root), "skip")

    def test_dry_run_does_not_mutate_partial_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.root(Path(directory))
            destination = root / "data/policy/pseudo_labels/train"
            write_tree(destination, {"artifact_status": "partial"}, "partial\n")
            self.assertEqual(reconcile_v1(root, dry_run=True), "backup")
            self.assertTrue(destination.is_dir())
            self.assertFalse((root / "data/policy/pseudo_labels/_local_backup").exists())


class LocalCommandTests(unittest.TestCase):
    def test_environment_device_grid_and_sequential_order(self):
        python = Path("C:/dedicated/timelymt-v2/python.exe")
        self.assertEqual(train_command("P0", python=python)[0], str(python))
        rollout = rollout_command("P1", python=python)
        self.assertEqual(rollout[0], str(python))
        self.assertEqual(rollout[rollout.index("--encoder-device") + 1], "cpu")
        self.assertEqual(rollout[rollout.index("--policy-device") + 1], "cpu")
        self.assertEqual(rollout[rollout.index("--translator-device") + 1], "cuda")
        self.assertEqual(rollout[rollout.index("--batch-size") + 1], "1")
        start = rollout.index("--thresholds") + 1
        self.assertEqual(tuple(rollout[start:start + len(THRESHOLD_ARGS)]), THRESHOLD_ARGS)
        commands = orchestration_commands(python=python)
        variants = [command[command.index("--variant") + 1] for command in commands if "rollout-v2" in command]
        self.assertEqual(variants, ["P0", "P1", "P2"])
        self.assertFalse(any("multiprocessing" in token or "workers" in token for command in commands for token in command))
        self.assertTrue(all(command[0] == str(python) for command in commands))
        self.assertFalse(any("smart-traffic" in token.lower() for command in commands for token in command))

    def test_gpu_preflight_uses_c_not_module_main(self):
        command = gpu_preflight_command(python=Path("dedicated-python"))
        self.assertEqual(command[1], "-c")
        self.assertNotIn("__main__", " ".join(command))
        self.assertFalse(any(token == "-m" and next_token == "__main__" for token, next_token in zip(command, command[1:])))

    def test_no_test_command_can_be_constructed(self):
        commands = orchestration_commands()
        self.assertFalse(any(token.lower() == "test" for command in commands for token in command))
        self.assertTrue(all("dev" in command for command in commands if "rollout-v2" in command))

    def test_runner_stops_after_failed_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(root=Path(directory))
            with patch("timelymt.research.policy_v2_local.subprocess.Popen") as popen:
                process = popen.return_value
                process.stdout = iter(())
                process.wait.return_value = 7
                with self.assertRaisesRegex(RuntimeError, "return code 7"):
                    runner.run_process("synthetic", research_command("compare-v2"), ENV_NAME)

    def test_train_resume_validation_controls_skip(self):
        with patch("timelymt.research.policy_v2_runner._valid_checkpoint", side_effect=lambda variant: variant != "P2"):
            from timelymt.research.policy_v2_local import train_will_run
            self.assertFalse(train_will_run("P0"))
            self.assertFalse(train_will_run("P1"))
            self.assertTrue(train_will_run("P2"))

    def test_rollout_command_preserves_fine_grained_resume_implementation(self):
        source = (Path(__file__).parents[2] / "src/timelymt/research/policy_v2_runner.py").read_text(encoding="utf-8")
        self.assertIn("validate_prediction_record(_load_json(path)", source)
        self.assertIn("validated resume hit", source)
        self.assertIn("for threshold in thresholds", source)
        self.assertIn("for index, talk_id in enumerate(expected_talks", source)


class LocalLockTests(unittest.TestCase):
    def test_active_lock_refuses_second_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock"
            with LocalRunLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "another policy-v2-local"):
                    LocalRunLock(lock_path).__enter__()

    def test_stale_lock_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock"
            lock_path.mkdir()
            (lock_path / "owner.json").write_text(
                json.dumps({"pid": 2147483647, "process_start_token": -1}), encoding="utf-8",
            )
            with LocalRunLock(lock_path):
                owner = json.loads((lock_path / "owner.json").read_text(encoding="utf-8"))
                self.assertEqual(owner["pid"], os.getpid())
            self.assertFalse(lock_path.exists())


class FreezeResumeValidationTests(unittest.TestCase):
    def test_every_frozen_identity_hash_is_revalidated(self):
        expected = {
            "artifact_status": "v2-dev-frozen-complete", "experiment_status": "post_hoc_exploratory",
            "v1_source_identity": {"commit": "v1"}, "dataset_checksum": DATASET_CHECKSUM,
            "split_checksum": SPLIT_CHECKSUM, "translator_fingerprint": TRANSLATOR_FINGERPRINT,
            "encoder": {"revision": "pinned"}, "runtime": {"encoder_device": "cpu"},
            "checkpoint_hashes": {"P0": "0", "P1": "1", "P2": "2"}, "metrics_sha256": "v2",
            "v1_metrics_sha256": "v1", "selected_strategy": "v2_P2_0.50", "selection_sha256": "selection",
            "train_manifest_checksum": "train", "dev_manifest_checksum": "dev", "test_status": "UNTOUCHED",
        }
        self.assertTrue(_frozen_document_matches(expected, expected))
        for field in (
            "v1_source_identity", "dataset_checksum", "split_checksum", "encoder", "runtime",
            "checkpoint_hashes", "metrics_sha256", "v1_metrics_sha256", "selected_strategy",
            "selection_sha256", "train_manifest_checksum", "dev_manifest_checksum", "experiment_status",
        ):
            altered = dict(expected)
            altered[field] = "changed"
            self.assertFalse(_frozen_document_matches(altered, expected), field)


if __name__ == "__main__":
    unittest.main()
