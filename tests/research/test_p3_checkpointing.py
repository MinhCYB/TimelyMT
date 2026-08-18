from __future__ import annotations

import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import torch

from timelymt.research.p3_checkpointing import (
    P3_CHECKPOINT_ROOT, P3_PACKAGE_SCHEMA_VERSION, _safe_relative,
    discover_p3_candidates, resolve_local_conflict, restore_p3_candidate,
)
from timelymt.research.policy_p3_global import P3_INPUT_DIMENSION
from timelymt.research.policy import NUMERIC_FEATURES
from timelymt.research.policy_v2 import ENCODER_MODEL_ID, ENCODER_REVISION, POOLING_VERSION, NumericScaler, V2MLP, sha256_file
from timelymt.data.translation_artifacts import stable_fingerprint


class P3CheckpointingTests(unittest.TestCase):
    def _package(self, root: Path, name: str, *, created_at: str, compatible: bool = True) -> tuple[Path, Path, Path]:
        package = root / name; checkpoint_root = package / Path(*P3_CHECKPOINT_ROOT.parts); checkpoint_root.mkdir(parents=True)
        manifest = root / "manifest.json"; manifest.write_text("{}", encoding="utf-8")
        config = root / "config.json"; config.write_text(json.dumps({"variant": "P3_GLOBAL"}), encoding="utf-8")
        model = V2MLP(P3_INPUT_DIMENSION); scaler = NumericScaler((0.0,) * 11, (1.0,) * 11)
        payload = {"variant": "P3_GLOBAL", "input_dimension": P3_INPUT_DIMENSION, "embedding_dimension": 384, "scaler": {"mean": list(scaler.mean), "scale": list(scaler.scale), "fitted_split": "train"}, "model_state_dict": model.state_dict()}
        checkpoint = checkpoint_root / "P3_GLOBAL.pt"; torch.save(payload, checkpoint)
        internal = {"variant": "P3_GLOBAL", "input_dimension": P3_INPUT_DIMENSION, "prepared_context_schema_version": "prepared-context-v0", "prepared_representation_version": "prepared-global-v0", "prepared_context_manifest_fingerprint": sha256_file(manifest), "encoder_model_id": ENCODER_MODEL_ID, "encoder_revision": ENCODER_REVISION, "pooling_version": POOLING_VERSION, "embedding_dimension": 384, "numeric_feature_ordering": list(NUMERIC_FEATURES), "numeric_scaler_fit_split": "train", "numeric_scaler_fingerprint": stable_fingerprint(payload["scaler"]), "mlp_architecture": ["Linear(input,256)", "GELU", "Dropout(0.20)", "Linear(256,64)", "GELU", "Dropout(0.10)", "Linear(64,1)"], "training_hyperparameters": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "epochs": 20, "seed": 20260809, "weighted_bce": "TRAIN_LISTEN/TRAIN_COMMIT"}, "checkpoint_sha256": sha256_file(checkpoint)}
        (checkpoint_root / "P3_GLOBAL.metadata.json").write_text(json.dumps(internal), encoding="utf-8")
        outer = {"schema_version": P3_PACKAGE_SCHEMA_VERSION, "experiment": "P3_GLOBAL", "created_at": created_at, "completed_stage": "TRAINED", "repo_commit": "abc", "repo_remote": "https://example.test/repo", "config_fingerprint": stable_fingerprint({"variant": "P3_GLOBAL"}), "prepared_context_manifest_fingerprint": sha256_file(manifest), "checkpoint_sha256": internal["checkpoint_sha256"]}
        if not compatible: outer["experiment"] = "P2"
        (package / "checkpoint-metadata.json").write_text(json.dumps(outer), encoding="utf-8")
        return config, manifest, package

    def test_discovery_orders_only_compatible_expanded_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, manifest, _ = self._package(root, "old", created_at="2026-01-01T00:00:00Z")
            self._package(root, "new", created_at="2026-02-01T00:00:00Z")
            self._package(root, "bad", created_at="2027-01-01T00:00:00Z", compatible=False)
            found = discover_p3_candidates(root, config_path=config, manifest_path=manifest)
            self.assertEqual([item["path"].name for item in found], ["new", "old"])

    def test_path_and_conflict_safety(self):
        with self.assertRaises(RuntimeError): _safe_relative("checkpoints/../TEST/x", (P3_CHECKPOINT_ROOT,))
        with self.assertRaises(RuntimeError): _safe_relative("outputs/test/x", (P3_CHECKPOINT_ROOT,))
        self.assertEqual(resolve_local_conflict(None, {"checkpoint_sha256": "a", "created_at": "2"}), "restore-persistent")
        self.assertEqual(resolve_local_conflict({"checkpoint_sha256": "a", "created_at": "1"}, {"checkpoint_sha256": "a", "created_at": "2"}), "keep-local")
        self.assertEqual(resolve_local_conflict({"checkpoint_sha256": "a", "created_at": "2"}, {"checkpoint_sha256": "b", "created_at": "2"}), "conflict")

    def test_raw_archive_member_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../../TEST/leak"); info.size = 0; handle.addfile(info)
            with self.assertRaises(RuntimeError):
                _safe_relative("../../TEST/leak", (P3_CHECKPOINT_ROOT,))

    def test_discovery_accepts_safe_raw_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, manifest, package = self._package(root, "expanded", created_at="2026-01-01T00:00:00Z")
            archive = root / "p3.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(package.rglob("*")):
                    if path.is_file(): handle.add(path, path.relative_to(package).as_posix())
            shutil_target = root / "expanded"
            import shutil
            shutil.rmtree(shutil_target)
            found = discover_p3_candidates(root, config_path=config, manifest_path=manifest)
            self.assertEqual([item["path"].name for item in found], ["p3.tar.gz"])

    def test_safe_raw_and_expanded_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, manifest, package = self._package(root, "expanded", created_at="2026-01-01T00:00:00Z")
            target = root / "repo"
            restore_p3_candidate(package, target, config_path=config, manifest_path=manifest)
            self.assertTrue((target / P3_CHECKPOINT_ROOT / "P3_GLOBAL.pt").is_file())
            archive = root / "p3.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(package.rglob("*")):
                    if path.is_file(): handle.add(path, path.relative_to(package).as_posix())
            target_archive = root / "repo-archive"
            restore_p3_candidate(archive, target_archive, config_path=config, manifest_path=manifest)
            self.assertTrue((target_archive / P3_CHECKPOINT_ROOT / "P3_GLOBAL.metadata.json").is_file())

    def test_notebook_is_json_and_every_code_cell_compiles(self):
        notebook = json.loads((Path(__file__).parents[2] / "notebooks/kaggle-p3-global.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
