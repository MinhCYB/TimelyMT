from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from timelymt.data.prepared_context import (
    PreparedContextMetadata, PreparedContextPool, PreparedContextSource, source_text_checksum,
)
from timelymt.research.policy import NUMERIC_FEATURES
from timelymt.research.policy_v2 import EmbeddingCache, NumericScaler, V2MLP, input_dimension
from timelymt.research.policy_p3_global import (
    P3_INPUT_DIMENSION, P3_VARIANT, PreparedGlobalEmbedding, build_p3_feature_matrix,
    build_prepared_global_embedding, checkpoint_payload, load_matching_pool, load_p3_checkpoint, make_p3_checkpoint_metadata,
    p3_feature_vector, prepare_p3_text_embeddings, save_p3_checkpoint, train_p3_global_policy,
    validate_p3_checkpoint_metadata, validate_pool_identity,
)
from timelymt.research.policy_p3_global_runner import _strategy, attach_prepared_context_provenance, p3_runtime


class FakeEncoder:
    dimension = 384

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(tuple(texts))
        result = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[sum(text.encode()) % 384] = 1.0
            result.append(vector)
        return np.stack(result)


def source(source_id, text, classification="SAFE_PRETALK_CONFIRMED"):
    return PreparedContextSource(source_id, "paper", text, "https://example.com/source", "en", None,
        "2026-01-01T00:00:00Z", classification == "SAFE_PRETALK_CONFIRMED", classification,
        "prepared", classification == "TRANSCRIPT_DERIVED", classification == "REFERENCE_DERIVED", source_text_checksum(text))


def pool(talk_id="talk", split="train", sources=(), metadata=None):
    return PreparedContextPool("prepared-context-v0", talk_id, split, metadata or PreparedContextMetadata(), tuple(sources))


def state(value=1.0):
    return {"current_source_text": "current", "previous_committed_source_text": "previous", "previous_committed_target_text": "target",
            "numeric": {name: value + index for index, name in enumerate(NUMERIC_FEATURES)}}


class PreparedRepresentationTests(unittest.TestCase):
    def test_empty_pool_is_exact_zero_float32(self):
        result = build_prepared_global_embedding(pool(), FakeEncoder())
        self.assertEqual(result.embedding.shape, (384,))
        self.assertEqual(result.embedding.dtype, np.float32)
        np.testing.assert_array_equal(result.embedding, np.zeros(384, dtype=np.float32))

    def test_one_multiple_order_metadata_and_ineligible_behavior(self):
        encoder = FakeEncoder()
        one = build_prepared_global_embedding(pool(sources=[source("b", "one")]), encoder)
        np.testing.assert_array_equal(one.embedding, encoder.encode(["one"])[0])
        first, second = source("z", "first"), source("a", "second")
        expected_values = encoder.encode(["second", "first"])
        expected = expected_values.mean(axis=0)
        expected /= np.linalg.norm(expected)
        result = build_prepared_global_embedding(pool(sources=[first, second]), encoder)
        reversed_result = build_prepared_global_embedding(pool(sources=[second, first], metadata=PreparedContextMetadata("new", "speaker", "domain")), encoder)
        np.testing.assert_allclose(result.embedding, expected.astype(np.float32))
        np.testing.assert_array_equal(result.embedding, reversed_result.embedding)
        ignored = [source("ignored", "must not encode", kind) for kind in ("SAFE_PRETALK_PLAUSIBLE", "PUBLIC_POST_TALK", "QUESTIONABLE", "TRANSCRIPT_DERIVED", "REFERENCE_DERIVED")]
        unchanged = build_prepared_global_embedding(pool(sources=[first, second, *ignored]), encoder)
        np.testing.assert_array_equal(result.embedding, unchanged.embedding)
        changed = build_prepared_global_embedding(pool(sources=[first, source("a", "changed")]), encoder)
        self.assertFalse(np.array_equal(result.embedding, changed.embedding))

    def test_pool_identity_and_missing_pool_safety(self):
        with self.assertRaises(RuntimeError):
            validate_pool_identity(pool("wrong"), talk_id="talk", split="train")
        with self.assertRaises(RuntimeError):
            validate_pool_identity(pool(split="dev"), talk_id="talk", split="train")
        with self.assertRaises(RuntimeError):
            validate_pool_identity(pool(), talk_id="talk", split="test")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_matching_pool(Path(directory), talk_id="talk", split="train")


class P3FeatureTests(unittest.TestCase):
    def test_feature_order_dtype_and_per_talk_join(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder()
            cache = EmbeddingCache(Path(directory), encoder)
            scaler = NumericScaler((0.0,) * 11, (1.0,) * 11)
            prepared_a = PreparedGlobalEmbedding("a", "train", (), (), np.full(384, 9, dtype=np.float32))
            prepared_b = PreparedGlobalEmbedding("b", "train", (), (), np.full(384, 8, dtype=np.float32))
            feature = p3_feature_vector(state(), prepared_a, cache, scaler)
            self.assertEqual(feature.shape, (P3_INPUT_DIMENSION,))
            self.assertEqual(input_dimension("P2"), 1163)
            self.assertEqual(feature.dtype, np.float32)
            np.testing.assert_array_equal(feature[1152:1536], prepared_a.embedding)
            changed = p3_feature_vector(state(), prepared_b, cache, scaler)
            np.testing.assert_array_equal(feature[:1152], changed[:1152])
            np.testing.assert_array_equal(feature[1536:], changed[1536:])
            rows = [{"talk_id": "a", "split": "train", "label": "LISTEN", "causal": state()}, {"talk_id": "b", "split": "train", "label": "COMMIT", "causal": state(2)}]
            matrix = build_p3_feature_matrix(rows, {"a": prepared_a, "b": prepared_b}, cache, scaler)
            np.testing.assert_array_equal(matrix[0, 1152:1536], prepared_a.embedding)
            np.testing.assert_array_equal(matrix[1, 1152:1536], prepared_b.embedding)
            self.assertEqual([row["label"] for row in rows], ["LISTEN", "COMMIT"])

    def test_batched_preparation_reuses_exact_cache_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder(); cache = EmbeddingCache(Path(directory), encoder)
            rows = [{"talk_id": "a", "split": "train", "label": "LISTEN", "causal": state()}, {"talk_id": "a", "split": "train", "label": "COMMIT", "causal": state()}]
            stats = prepare_p3_text_embeddings(rows, [pool("a", sources=[source("s", "current")])], cache)
            self.assertEqual(stats["unique_texts"], 3)
            self.assertEqual(stats["cache_misses"], 3)
            self.assertEqual(len(encoder.calls), 1)
            second = prepare_p3_text_embeddings(rows, [pool("a", sources=[source("s", "current")])], cache)
            self.assertEqual(second["cache_hits"], 3)
            self.assertEqual(len(encoder.calls), 1)

    def test_batched_encoder_output_is_float32_384d(self):
        encoder = FakeEncoder()
        values = encoder.encode(["one", "two"])
        self.assertEqual(values.shape, (2, 384))
        self.assertEqual(values.dtype, np.float32)

    def test_zero_mode_replaces_only_prepared_block_and_default_is_real(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory), FakeEncoder())
            scaler = NumericScaler((0.0,) * 11, (1.0,) * 11)
            prepared = PreparedGlobalEmbedding("context", "dev", ("source",), ("sha256:" + "0" * 64,), np.full(384, 3.0, dtype=np.float32))
            default = p3_feature_vector(state(), prepared, cache, scaler)
            real = p3_feature_vector(state(), prepared, cache, scaler, prepared_context_mode="real")
            zero = p3_feature_vector(state(), prepared, cache, scaler, prepared_context_mode="zero")
            self.assertEqual(default.shape, (1547,))
            self.assertEqual(default.dtype, np.float32)
            np.testing.assert_array_equal(default, real)
            np.testing.assert_array_equal(default[:1152], zero[:1152])
            np.testing.assert_array_equal(default[1536:], zero[1536:])
            np.testing.assert_array_equal(zero[1152:1536], np.zeros(384, dtype=np.float32))
            self.assertFalse(np.array_equal(default[1152:1536], zero[1152:1536]))

    def test_empty_context_is_exactly_invariant_between_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EmbeddingCache(Path(directory), FakeEncoder())
            scaler = NumericScaler((0.0,) * 11, (1.0,) * 11)
            prepared = PreparedGlobalEmbedding("empty", "dev", (), (), np.zeros(384, dtype=np.float32))
            self.assertEqual(prepared.embedding_norm, 0.0)
            np.testing.assert_array_equal(prepared.effective_embedding("real"), prepared.effective_embedding("zero"))
            np.testing.assert_array_equal(
                p3_feature_vector(state(), prepared, cache, scaler, prepared_context_mode="real"),
                p3_feature_vector(state(), prepared, cache, scaler, prepared_context_mode="zero"),
            )

    def test_context_provenance_distinguishes_eligible_source_from_zero_injection(self):
        prepared = PreparedGlobalEmbedding("context", "dev", ("source",), ("sha256:" + "0" * 64,), np.full(384, 0.5, dtype=np.float32))
        real, zero = prepared.provenance("real"), prepared.provenance("zero")
        self.assertTrue(zero["has_eligible_context"])
        self.assertEqual(zero["eligible_source_ids"], ["source"])
        self.assertGreater(real["prepared_context_effective_embedding_norm"], 0.0)
        self.assertEqual(zero["prepared_context_effective_embedding_norm"], 0.0)
        self.assertEqual(zero["prepared_context_mode"], "zero")


class P3RuntimeTests(unittest.TestCase):
    def _config(self, directory, runtime):
        path = Path(directory) / "p3.json"
        path.write_text(json.dumps({"runtime": runtime}), encoding="utf-8")
        return path

    def test_auto_and_explicit_cpu_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(p3_runtime(self._config(directory, {"encoder_device": "cpu", "policy_device": "cpu", "encoder_batch_size": 8}))["encoder_device"].type, "cpu")
            with patch("timelymt.research.policy_p3_global_runner.torch.cuda.is_available", return_value=False):
                runtime = p3_runtime(self._config(directory, {"encoder_device": "auto", "policy_device": "auto", "encoder_batch_size": 8}))
            self.assertEqual((runtime["encoder_device"].type, runtime["policy_device"].type), ("cpu", "cpu"))

    def test_explicit_cuda_fails_when_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, patch("timelymt.research.policy_p3_global_runner.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "explicitly requested"):
                p3_runtime(self._config(directory, {"encoder_device": "cuda", "policy_device": "cpu"}))

    def test_synthetic_training_uses_selected_cpu_device_and_portable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder(); cache = EmbeddingCache(Path(directory) / "cache", encoder)
            prepared = PreparedGlobalEmbedding("a", "train", (), (), np.zeros(384, dtype=np.float32))
            rows = [{"talk_id": "a", "split": "train", "label": "LISTEN", "causal": state()}, {"talk_id": "a", "split": "train", "label": "COMMIT", "causal": state(2)}]
            policy, _ = train_p3_global_policy(rows, {"a": prepared}, cache, epochs=1, batch_size=2, device="cpu")
            self.assertEqual(next(policy.model.parameters()).device.type, "cpu")
            payload = checkpoint_payload(policy)
            self.assertTrue(all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in payload["model_state_dict"].values()))
            restored = V2MLP(P3_INPUT_DIMENSION); restored.load_state_dict(payload["model_state_dict"], strict=True)
            manifest = Path(directory) / "manifest.json"; manifest.write_text("{}", encoding="utf-8")
            checkpoint = Path(directory) / "p3.pt"; digest = save_p3_checkpoint(checkpoint, policy)
            metadata = make_p3_checkpoint_metadata(checkpoint_hash=digest, prepared_manifest=manifest, train_talk_ids=["a"], training={"label_counts": {"LISTEN": 1, "COMMIT": 1}, "positive_weight": 1.0}, scaler=policy.scaler)
            metadata_path = Path(directory) / "p3.json"; metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            loaded = load_p3_checkpoint(checkpoint, metadata_path, cache, prepared, manifest_path=manifest, device="cpu")
            self.assertEqual(next(loaded.model.parameters()).device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_synthetic_training_uses_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder(); cache = EmbeddingCache(Path(directory) / "cache", encoder)
            prepared = PreparedGlobalEmbedding("a", "train", (), (), np.zeros(384, dtype=np.float32))
            rows = [{"talk_id": "a", "split": "train", "label": "LISTEN", "causal": state()}, {"talk_id": "a", "split": "train", "label": "COMMIT", "causal": state(2)}]
            policy, _ = train_p3_global_policy(rows, {"a": prepared}, cache, epochs=1, batch_size=2, device="cuda")
            self.assertEqual(next(policy.model.parameters()).device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_cuda_mlp_outputs_match_within_tolerance(self):
        torch.manual_seed(20260809)
        cpu_model = V2MLP(P3_INPUT_DIMENSION).eval()
        cuda_model = V2MLP(P3_INPUT_DIMENSION).eval().to("cuda")
        cuda_model.load_state_dict(cpu_model.state_dict())
        values = torch.randn(3, P3_INPUT_DIMENSION, dtype=torch.float32)
        with torch.inference_mode():
            expected = cpu_model(values)
            actual = cuda_model(values.to("cuda")).cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class P3CheckpointTests(unittest.TestCase):
    def test_metadata_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            cache = EmbeddingCache(root / "cache", FakeEncoder())
            prepared = PreparedGlobalEmbedding("talk", "train", (), (), np.zeros(384, dtype=np.float32))
            from timelymt.research.policy_p3_global import P3GlobalPolicy
            policy = P3GlobalPolicy(V2MLP(P3_INPUT_DIMENSION), NumericScaler((0.0,) * 11, (1.0,) * 11), cache, prepared, torch.device("cpu"))
            path = root / "model.pt"
            digest = save_p3_checkpoint(path, policy)
            metadata = make_p3_checkpoint_metadata(checkpoint_hash=digest, prepared_manifest=manifest, train_talk_ids=["talk"], training={"label_counts": {"LISTEN": 1, "COMMIT": 1}, "positive_weight": 1.0}, scaler=policy.scaler)
            payload = checkpoint_payload(policy)
            validate_p3_checkpoint_metadata(metadata, payload, manifest_path=manifest, cache=cache)
            for key, value in (("variant", "P2"), ("input_dimension", 0), ("prepared_representation_version", "wrong"), ("encoder_revision", "wrong")):
                altered = dict(metadata); altered[key] = value
                with self.assertRaises(RuntimeError):
                    validate_p3_checkpoint_metadata(altered, payload, manifest_path=manifest, cache=cache)
            altered = dict(metadata); altered["prepared_context_manifest_fingerprint"] = "wrong"
            with self.assertRaises(RuntimeError):
                validate_p3_checkpoint_metadata(altered, payload, manifest_path=manifest, cache=cache)

    def test_loading_zero_mode_does_not_change_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = root / "manifest.json"; manifest.write_text("{}", encoding="utf-8")
            cache = EmbeddingCache(root / "cache", FakeEncoder())
            prepared = PreparedGlobalEmbedding("talk", "train", (), (), np.zeros(384, dtype=np.float32))
            from timelymt.research.policy_p3_global import P3GlobalPolicy
            policy = P3GlobalPolicy(V2MLP(P3_INPUT_DIMENSION), NumericScaler((0.0,) * 11, (1.0,) * 11), cache, prepared, torch.device("cpu"))
            checkpoint = root / "model.pt"; digest = save_p3_checkpoint(checkpoint, policy)
            metadata = make_p3_checkpoint_metadata(checkpoint_hash=digest, prepared_manifest=manifest, train_talk_ids=["talk"], training={"label_counts": {"LISTEN": 1, "COMMIT": 1}, "positive_weight": 1.0}, scaler=policy.scaler)
            metadata_path = root / "model.json"; metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            before = checkpoint.read_bytes()
            loaded = load_p3_checkpoint(checkpoint, metadata_path, cache, prepared, manifest_path=manifest, prepared_context_mode="zero")
            self.assertEqual(loaded.prepared_context_mode, "zero")
            self.assertEqual(checkpoint.read_bytes(), before)


class P3AblationNamingTests(unittest.TestCase):
    def test_real_preserves_canonical_strategy_and_zero_is_isolated(self):
        self.assertEqual(_strategy(0.50, "real"), "p3_global_0.50")
        self.assertEqual(_strategy(0.50, "zero"), "p3_global_zeroctx_0.50")

    def test_prediction_artifact_provenance_records_zero_condition(self):
        prepared = PreparedGlobalEmbedding("context", "dev", ("source",), ("sha256:" + "0" * 64,), np.ones(384, dtype=np.float32))
        record = {}
        attach_prepared_context_provenance(record, prepared, "zero")
        self.assertEqual(record["prepared_context_mode"], "zero")
        self.assertEqual(record["prepared_context_effective_embedding_norm"], 0.0)
        self.assertEqual(record["prepared_context"]["eligible_source_ids"], ["source"])
        self.assertTrue(record["prepared_context"]["has_eligible_context"])


class CompatibilityTests(unittest.TestCase):
    def test_translator_provider_has_no_prepared_context_input(self):
        source_text = (Path(__file__).parents[2] / "src/timelymt/research/cli.py").read_text(encoding="utf-8")
        provider = source_text[source_text.index("class Provider:"):source_text.index("def _translator")]
        self.assertIn("make_translation_request(observed, start, end", provider)
        self.assertNotIn("prepared", provider.lower())


if __name__ == "__main__":
    unittest.main()
