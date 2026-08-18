from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from timelymt.data.prepared_context import (
    PreparedContextMetadata, PreparedContextPool, PreparedContextSource, source_text_checksum,
)
from timelymt.research.policy import NUMERIC_FEATURES
from timelymt.research.policy_v2 import EmbeddingCache, NumericScaler, V2MLP, input_dimension
from timelymt.research.policy_p3_global import (
    P3_INPUT_DIMENSION, P3_VARIANT, PreparedGlobalEmbedding, build_p3_feature_matrix,
    build_prepared_global_embedding, checkpoint_payload, load_matching_pool, make_p3_checkpoint_metadata,
    p3_feature_vector, save_p3_checkpoint, validate_p3_checkpoint_metadata, validate_pool_identity,
)


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


class CompatibilityTests(unittest.TestCase):
    def test_translator_provider_has_no_prepared_context_input(self):
        source_text = (Path(__file__).parents[2] / "src/timelymt/research/cli.py").read_text(encoding="utf-8")
        provider = source_text[source_text.index("class Provider:"):source_text.index("def _translator")]
        self.assertIn("make_translation_request(observed, start, end", provider)
        self.assertNotIn("prepared", provider.lower())


if __name__ == "__main__":
    unittest.main()
