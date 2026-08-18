from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import torch

from timelymt.research.policy import NUMERIC_FEATURES
from timelymt.research.policy_v2 import (
    DATASET_CHECKSUM, ENCODER_REVISION, EXPERIMENT_STATUS, LOCAL_RUNTIME, SPLIT_CHECKSUM,
    TRANSLATOR_FINGERPRINT, V1_SOURCE_COMMIT, EmbeddingCache, FrozenMiniLMEncoder,
    NumericScaler, V2MLP, V2Policy, class_weight, load_v2_checkpoint, restore_v1_artifacts,
    save_v2_checkpoint, select_v2_configuration, state_texts, train_v2_policy, validate_causal_state,
    validate_prediction_record, validate_v1_checkpoint_metadata,
)
from timelymt.research.cli import main
from timelymt.research.streaming import learned_rollout
from timelymt.data.translation_artifacts import RuntimeSourceToken, RuntimeTalk, TranslationHypothesis


def state(previous_source: str = "", previous_target: str = ""):
    return {
        "current_source_text": "current source",
        "previous_committed_source_text": previous_source,
        "previous_committed_target_text": previous_target,
        "numeric": {name: float(index) for index, name in enumerate(NUMERIC_FEATURES)},
    }


class MockEncoder:
    dimension = 384

    def encode(self, texts):
        return np.stack([
            np.full(self.dimension, (sum(text.encode("utf-8")) % 31 + 1) / 31, dtype=np.float32)
            for text in texts
        ])


class RecordingPolicy:
    def __init__(self):
        self.states = []

    def predict_commit_probability(self, state):
        self.states.append(state)
        return 1.0


def runtime_talk():
    tokens = tuple(RuntimeSourceToken("talk", f"token-{index}", index, f"w{index}", index * 100) for index in range(9))
    return RuntimeTalk("talk", "dev", tokens)


def provider(talk, start, end):
    text = " ".join(f"v{index}" for index in range(start, end + 1))
    return TranslationHypothesis(
        "1.0.0", f"request-{start}-{end}", talk.talk_id, talk.split, start, end,
        talk.tokens[end].emit_ms, " ".join(token.text for token in talk.tokens[start:end + 1]),
        text, end - start + 1, len(text.split()), "model", "revision", "config", "fingerprint",
        "generation", None, None, None,
    )


def checkpoint_metadata(checkpoint_hash: str, variant: str = "P0"):
    return {
        "checkpoint_sha256": checkpoint_hash, "experiment_status": EXPERIMENT_STATUS,
        "encoder_revision": ENCODER_REVISION, "variant": variant,
    }


class PolicyV2FeatureTests(unittest.TestCase):
    def test_p0_p1_p2_field_isolation(self):
        value = state("prior source", "he thong dich")
        self.assertEqual(state_texts(value, "P0"), ("current source",))
        self.assertEqual(state_texts(value, "P1"), ("current source", "prior source"))
        self.assertEqual(state_texts(value, "P2"), ("current source", "prior source", "he thong dich"))

    def test_forbidden_and_extra_fields_are_rejected(self):
        value = state()
        value["gold_target"] = "leak"
        with self.assertRaises(RuntimeError):
            validate_causal_state(value)
        value = state()
        value["numeric"]["future_ratio"] = 1.0
        with self.assertRaises(RuntimeError):
            validate_causal_state(value)

    def test_zero_embedding_for_absent_history_and_exact_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory), MockEncoder())
            result = cache.encode(["", "same", "same", "different"])
            np.testing.assert_array_equal(result[0], np.zeros(384, dtype=np.float32))
            np.testing.assert_array_equal(result[1], result[2])
            self.assertFalse(np.array_equal(result[1], result[3]))

    def test_masked_pooling_shape_normalization_and_encoder_freeze_contract(self):
        hidden = torch.tensor([[[3.0, 4.0], [99.0, 99.0]], [[1.0, 0.0], [0.0, 1.0]]])
        mask = torch.tensor([[1, 0], [1, 1]])
        pooled = FrozenMiniLMEncoder.pool(hidden, mask)
        self.assertEqual(tuple(pooled.shape), (2, 2))
        torch.testing.assert_close(torch.linalg.vector_norm(pooled, dim=1), torch.ones(2))
        source = Path(__file__).parents[2] / "src/timelymt/research/policy_v2.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("eval().requires_grad_(False)", text)
        self.assertIn("torch.no_grad()", text)

    def test_scaler_train_only_and_weight_formula(self):
        values = np.asarray([[1.0] * len(NUMERIC_FEATURES), [3.0] * len(NUMERIC_FEATURES)], dtype=np.float32)
        scaler = NumericScaler.fit(values, split_name="train")
        np.testing.assert_allclose(scaler.transform(values).mean(axis=0), 0.0)
        with self.assertRaises(RuntimeError):
            NumericScaler.fit(values, split_name="dev")
        self.assertEqual(class_weight({"LISTEN": 12, "COMMIT": 3}), 4.0)

    def test_sequential_rollout_uses_own_committed_history(self):
        policy = RecordingPolicy()
        commits = learned_rollout(runtime_talk(), provider, policy, 0.5)
        self.assertEqual(policy.states[1]["previous_committed_source_text"], "w0 w1 w2 w3")
        self.assertEqual(policy.states[1]["previous_committed_target_text"], commits[0].translated_text)

    def test_small_offline_mock_encoder_training_smoke(self):
        rows = [
            {"split": "train", "label": label, "causal": state(f"source-{index}", f"target-{index}")}
            for index, label in enumerate(("LISTEN", "COMMIT", "LISTEN", "COMMIT"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory), MockEncoder())
            policy, training = train_v2_policy(rows, "P2", cache, epochs=1, batch_size=2, device="cpu")
            self.assertEqual(training["label_counts"], {"LISTEN": 2, "COMMIT": 2})
            self.assertGreaterEqual(policy.predict_commit_probability(rows[0]["causal"]), 0.0)


class PolicyV2ArtifactTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = EmbeddingCache(root / "cache", MockEncoder())
            policy = V2Policy("P0", V2MLP(395), NumericScaler((0.0,) * 11, (1.0,) * 11), cache, torch.device("cpu"))
            path, metadata_path = root / "V2P0.pt", root / "V2P0.metadata.json"
            digest = save_v2_checkpoint(path, policy)
            metadata_path.write_text(json.dumps(checkpoint_metadata(digest)), encoding="utf-8")
            restored = load_v2_checkpoint(path, metadata_path, cache, device="cpu")
            self.assertAlmostEqual(policy.predict_commit_probability(state()), restored.predict_commit_probability(state()), places=6)
            metadata_path.write_text(json.dumps(checkpoint_metadata("0" * 64)), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                load_v2_checkpoint(path, metadata_path, cache, device="cpu")

    def test_v1_identity_does_not_require_current_v2_commit(self):
        validate_v1_checkpoint_metadata({
            "checkpoint_stage": "dev-frozen-complete", "git_commit": V1_SOURCE_COMMIT,
            "dataset_manifest_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
            "translator_config_fingerprint": TRANSLATOR_FINGERPRINT,
        })
        self.assertNotIn("v2_code_commit", V1_SOURCE_COMMIT)

    def test_prediction_resume_identity(self):
        record = {
            "strategy": "v2_P0_0.30", "talk_id": "dev-talk", "split": "dev",
            "artifact_status": "full", "experiment_status": EXPERIMENT_STATUS,
            "model_sha256": "abc", "dataset_checksum": DATASET_CHECKSUM,
            "encoder_revision": ENCODER_REVISION, "runtime": LOCAL_RUNTIME, "commits": [{}],
        }
        validate_prediction_record(record, strategy="v2_P0_0.30", talk_id="dev-talk", model_hash="abc")
        record["model_sha256"] = "wrong"
        with self.assertRaises(RuntimeError):
            validate_prediction_record(record, strategy="v2_P0_0.30", talk_id="dev-talk", model_hash="abc")

    def test_v2_selection_uses_v1_rule_with_v2_names(self):
        v1 = {"fixed_n_8": {"token_level_average_lagging": 4.0, "chrF2": 0.0, "BLEU": 0.0}}
        v2 = {
            "v2_P0_0.30": {"token_level_average_lagging": 3.0, "chrF2": 20.0, "BLEU": 10.0},
            "v2_P1_0.40": {"token_level_average_lagging": 4.0, "chrF2": 21.0, "BLEU": 9.0},
        }
        self.assertEqual(select_v2_configuration(v1, v2)["selected_strategy"], "v2_P1_0.40")

    def test_cli_rejects_test_for_v2(self):
        with self.assertRaises(SystemExit):
            main(["rollout-v2", "--split", "test", "--variant", "P0"])


class V1RestoreSafetyTests(unittest.TestCase):
    def _files(self):
        manifest = {
            "artifact_status": "full", "publishable": True, "split": "train",
            "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
            "translator": {"config_fingerprint": TRANSLATOR_FINGERPRINT},
            "config": {}, "config_checksum": "config", "state_count": 1,
            "LISTEN": 1, "COMMIT": 0, "talk_ids": ["talk"], "expected_talk_ids": ["talk"],
        }
        row = {"talk_id": "talk", "split": "train", "label": "LISTEN", "causal": state()}
        dev_manifest = {**manifest, "split": "dev"}
        dev_row = {**row, "split": "dev"}
        metadata = {
            "checkpoint_stage": "dev-frozen-complete", "git_commit": V1_SOURCE_COMMIT,
            "dataset_manifest_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
            "translator_config_fingerprint": TRANSLATOR_FINGERPRINT,
        }
        return {
            "checkpoint-metadata.json": json.dumps(metadata).encode(),
            "data/policy/pseudo_labels/train/manifest.json": json.dumps(manifest).encode(),
            "data/policy/pseudo_labels/train/talk.jsonl": (json.dumps(row) + "\n").encode(),
            "data/policy/pseudo_labels/dev/manifest.json": json.dumps(dev_manifest).encode(),
            "data/policy/pseudo_labels/dev/talk.jsonl": (json.dumps(dev_row) + "\n").encode(),
            "outputs/experiments/research-mvp/metrics/dev/all.json": b"{}",
            "outputs/experiments/research-mvp/dev-selection.json": b"{}",
            "outputs/experiments/research-mvp/frozen-eval-config.json": b"{}",
        }

    def _write_tree(self, root: Path):
        for name, content in self._files().items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    def test_expanded_layout_and_no_v1_source_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, destination = base / "download/timelymt-checkpoint", base / "repo"
            source.mkdir(parents=True)
            destination.mkdir()
            self._write_tree(source)
            (destination / ".git").mkdir()
            # Avoid invoking git in this synthetic tree while still proving restore isolation.
            import timelymt.research.policy_v2 as module
            original = module.current_git_commit
            module.current_git_commit = lambda root: "v2-current"
            try:
                restore_v1_artifacts(source.parent, destination)
            finally:
                module.current_git_commit = original
            self.assertTrue((destination / "data/policy/pseudo_labels/train/talk.jsonl").is_file())
            self.assertFalse((destination / "src").exists())
            self.assertFalse((destination / "outputs/experiments/research-mvp").exists())

    def test_raw_tar_and_traversal_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tree = base / "tree"
            tree.mkdir()
            self._write_tree(tree)
            archive = base / "valid.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in tree.rglob("*"):
                    handle.add(path, arcname=f"timelymt-checkpoint/{path.relative_to(tree).as_posix()}")
            destination = base / "repo"
            destination.mkdir()
            import timelymt.research.policy_v2 as module
            original = module.current_git_commit
            module.current_git_commit = lambda root: "v2-current"
            try:
                restore_v1_artifacts(archive, destination)
            finally:
                module.current_git_commit = original
            unsafe = base / "unsafe.tar.gz"
            with tarfile.open(unsafe, "w:gz") as handle:
                info = tarfile.TarInfo("../escape")
                info.size = 1
                handle.addfile(info, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                restore_v1_artifacts(unsafe, destination)
            symlink = base / "symlink.tar.gz"
            with tarfile.open(symlink, "w:gz") as handle:
                info = tarfile.TarInfo("data/policy/pseudo_labels/train/link.jsonl")
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                handle.addfile(info)
            with self.assertRaisesRegex(RuntimeError, "unsafe.*member"):
                restore_v1_artifacts(symlink, destination)


if __name__ == "__main__":
    unittest.main()
