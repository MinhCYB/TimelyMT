from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.canonical.core import canonical_content_checksum
from timelymt.data.translation_artifacts import (
    DATASET_NAME,
    ArtifactProvenance,
    RuntimeSourceToken,
    TranslationHypothesis,
    TranslationRequest,
    build_artifact_provenance,
    make_translation_request,
    read_artifact_jsonl,
    reconstruct_source_text,
    runtime_talk_from_canonical,
    stable_fingerprint,
    translate_requests,
    translator_identity,
    validate_artifact_manifest,
    validate_translation_hypothesis,
    validate_translation_request,
    write_artifact_jsonl,
)
from timelymt.translator import TranslationCache, TranslationResult, Translator
from timelymt.translator.envit5 import EnViT5Translator, _Runtime, load_config


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs/translator/envit5.json"


def canonical_talk(talk_id: str = "talk-a") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "talk": {"talk_id": talk_id, "source_language": "en", "target_language": "vi"},
        "source": {
            "language": "en",
            "segments": [
                {"segment_id": "en-1", "index": 0, "text": "Hello, WORLD!", "start_ms": 0, "end_ms": 300},
                {"segment_id": "en-2", "index": 1, "text": "Next line.", "start_ms": 300, "end_ms": 500},
            ],
        },
        "target_reference": {
            "language": "vi",
            "segments": [
                {"segment_id": "vi-1", "index": 0, "text": "Xin chao."},
                {"segment_id": "vi-2", "index": 1, "text": "Dong tiep."},
            ],
        },
        "alignments": [
            {"alignment_id": "a-1", "source_segment_ids": ["en-1"], "target_segment_ids": ["vi-1"], "method": "manual"},
            {"alignment_id": "a-2", "source_segment_ids": ["en-2"], "target_segment_ids": ["vi-2"], "method": "manual"},
        ],
        "stream": {
            "timing_mode": "simulated",
            "tokens": [
                {"token_id": "tok-1", "index": 0, "text": "Hello", "source_segment_id": "en-1", "segment_index": 0, "emit_ms": 100},
                {"token_id": "tok-2", "index": 1, "text": "WORLD", "source_segment_id": "en-1", "segment_index": 1, "emit_ms": 300},
                {"token_id": "tok-3", "index": 2, "text": "Next", "source_segment_id": "en-2", "segment_index": 0, "emit_ms": 400},
                {"token_id": "tok-4", "index": 3, "text": "line", "source_segment_id": "en-2", "segment_index": 1, "emit_ms": 500},
            ],
        },
        "provenance": {"processing_version": "test", "processed_at": "2026-08-09T00:00:00Z"},
    }


def split_manifest(*talk_ids: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "split_type": "experimental",
        "dataset_manifest_checksum": "d" * 64,
        "strategy": {},
        "splits": {"train": list(talk_ids), "dev": [], "test": []},
    }


class EchoTranslator(Translator):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def translate(self, text: str) -> TranslationResult:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts):
        batch = list(texts)
        self.batches.append(batch)
        return [
            TranslationResult(
                f"vi:{text}", text, 99, len(text.split()) + 1,
                {"device": "cpu", "dtype": "float32", "cache_hit": False},
            )
            for text in batch
        ]


class TranslationArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical = canonical_talk()
        self.split = split_manifest("talk-a")
        self.identity = translator_identity(load_config(CONFIG_PATH))
        self.runtime = runtime_talk_from_canonical(
            self.canonical,
            split_manifest=self.split,
            observed_through_token_index=2,
        )

    def test_runtime_view_is_causal_and_omits_boundary_and_gold_fields(self) -> None:
        self.assertEqual([token.text for token in self.runtime.tokens], ["Hello", "WORLD", "Next"])
        self.assertEqual(self.runtime.latest_observed_token_index, 2)
        serialized = json.dumps(asdict(self.runtime))
        for forbidden in ("target", "reference", "alignment", "segment_index", "source_segment_id", "line"):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_single_multi_and_nonzero_spans_use_exact_runtime_text(self) -> None:
        single = make_translation_request(self.runtime, 1, 1, translator=self.identity)
        multi = make_translation_request(self.runtime, 1, 2, translator=self.identity)
        self.assertEqual(single.source_text, "WORLD")
        self.assertEqual(single.observation_emit_ms, 300)
        self.assertEqual(multi.source_text, "WORLD Next")
        self.assertEqual(multi.observation_emit_ms, 400)

    def test_reconstruction_is_deterministic_and_does_not_restore_punctuation_or_case(self) -> None:
        tokens = self.runtime.tokens[:2]
        self.assertEqual(reconstruct_source_text(tokens), reconstruct_source_text(tokens))
        self.assertEqual(reconstruct_source_text(tokens), "Hello WORLD")
        self.assertNotEqual(reconstruct_source_text(tokens), self.canonical["source"]["segments"][0]["text"])  # type: ignore[index]
        self.assertFalse(reconstruct_source_text(tokens).endswith((".", "!", "?", ",")))

    def test_invalid_and_future_spans_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be <="):
            make_translation_request(self.runtime, 2, 1, translator=self.identity)
        with self.assertRaisesRegex(ValueError, "future token"):
            make_translation_request(self.runtime, 0, 3, translator=self.identity)
        full_runtime = runtime_talk_from_canonical(
            self.canonical, split_manifest=self.split, observed_through_token_index=3,
        )
        with self.assertRaisesRegex(ValueError, "future token"):
            make_translation_request(self.runtime, 0, full_runtime.latest_observed_token_index, translator=self.identity)

    def test_request_id_is_stable_and_bound_to_span_and_translator(self) -> None:
        first = make_translation_request(self.runtime, 0, 2, translator=self.identity)
        second = make_translation_request(self.runtime, 0, 2, translator=self.identity)
        changed_span = make_translation_request(self.runtime, 1, 2, translator=self.identity)
        changed_identity = replace(self.identity, config_fingerprint="0" * 64)
        changed_translator = make_translation_request(self.runtime, 0, 2, translator=changed_identity)
        self.assertEqual(first.request_id, second.request_id)
        self.assertNotEqual(first.request_id, changed_span.request_id)
        self.assertNotEqual(first.request_id, changed_translator.request_id)

    def test_output_semantic_version_changes_config_and_request_identity_only(self) -> None:
        current_config = load_config(CONFIG_PATH)
        old_config = replace(current_config, config_version="1.0.0")
        current_identity = translator_identity(current_config)
        old_identity = translator_identity(old_config)
        current_request = make_translation_request(self.runtime, 0, 2, translator=current_identity)
        old_request = make_translation_request(self.runtime, 0, 2, translator=old_identity)

        self.assertNotEqual(current_identity.config_fingerprint, old_identity.config_fingerprint)
        self.assertNotEqual(current_request.request_id, old_request.request_id)
        self.assertEqual(current_identity.generation_config_fingerprint, old_identity.generation_config_fingerprint)
        self.assertEqual(current_identity.model_id, old_identity.model_id)
        self.assertEqual(current_identity.model_revision, old_identity.model_revision)

    def test_split_is_inherited_and_missing_assignment_is_rejected(self) -> None:
        request = make_translation_request(self.runtime, 0, 0, translator=self.identity)
        self.assertEqual(request.split, "train")
        with self.assertRaisesRegex(ValueError, "not assigned"):
            runtime_talk_from_canonical(
                canonical_talk("talk-missing"),
                split_manifest=self.split,
                observed_through_token_index=0,
            )

    def test_request_semantic_validation_detects_timestamp_source_and_identity_changes(self) -> None:
        request = make_translation_request(self.runtime, 0, 2, translator=self.identity)
        validate_translation_request(request, runtime_talk=self.runtime, translator=self.identity)
        for changed in (
            replace(request, observation_emit_ms=399),
            replace(request, source_text="Hello WORLD Next."),
            replace(request, split="test"),
            replace(request, request_id="trq-" + "0" * 64),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_translation_request(changed, runtime_talk=self.runtime, translator=self.identity)

    def test_request_and_hypothesis_types_have_no_unsafe_fields(self) -> None:
        names = set(TranslationRequest.__dataclass_fields__) | set(TranslationHypothesis.__dataclass_fields__)
        for fragment in ("reference", "alignment", "bleu", "chrf", "commit", "policy", "future"):
            self.assertFalse(any(fragment in name.lower() for name in names))

    def test_batch_translation_preserves_order_and_uses_canonical_span_counts(self) -> None:
        requests = [
            make_translation_request(self.runtime, 0, end, translator=self.identity)
            for end in range(3)
        ]
        translator = EchoTranslator()
        hypotheses = translate_requests(
            translator, requests, translator_identity=self.identity, batch_size=2,
        )
        self.assertEqual(translator.batches, [["Hello", "Hello WORLD"], ["Hello WORLD Next"]])
        self.assertEqual([row.request_id for row in hypotheses], [row.request_id for row in requests])
        self.assertEqual([row.source_token_count for row in hypotheses], [1, 2, 3])
        self.assertEqual([row.cache_hit for row in hypotheses], [False, False, False])

    def test_duplicates_are_rejected_before_translation(self) -> None:
        request = make_translation_request(self.runtime, 0, 1, translator=self.identity)
        translator = EchoTranslator()
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            translate_requests(
                translator, [request, request], translator_identity=self.identity,
            )
        self.assertEqual(translator.batches, [])

    def test_hypothesis_must_resolve_request_and_preserve_source(self) -> None:
        request = make_translation_request(self.runtime, 0, 1, translator=self.identity)
        hypothesis = translate_requests(
            EchoTranslator(), [request], translator_identity=self.identity,
        )[0]
        validate_translation_hypothesis(hypothesis, request=request, translator=self.identity)
        with self.assertRaisesRegex(ValueError, "exact translation request"):
            validate_translation_hypothesis(
                replace(hypothesis, source_text="changed"), request=request, translator=self.identity,
            )

    def test_jsonl_and_manifest_round_trip_with_provenance(self) -> None:
        requests = [make_translation_request(self.runtime, 0, 1, translator=self.identity)]
        provenance = ArtifactProvenance(
            DATASET_NAME, "1.0.0", "d" * 64, "snapshot.json", "split.json",
            "s" * 64, self.identity.model_id, self.identity.model_revision,
            self.identity.config_version, self.identity.config_fingerprint,
            self.identity.generation_config_fingerprint,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "requests.jsonl"
            manifest_path = Path(directory) / "requests.manifest.json"
            manifest = write_artifact_jsonl(
                artifact_path,
                requests,
                provenance=provenance,
                artifact_type="translation_requests",
                manifest_path=manifest_path,
                created_at="2026-08-09T00:00:00Z",
            )
            loaded = read_artifact_jsonl(artifact_path, TranslationRequest)
            self.assertEqual(loaded, requests)
            validate_artifact_manifest(manifest, records=loaded, artifact_bytes=artifact_path.read_bytes())
            self.assertEqual(manifest.artifact_checksum, hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            self.assertEqual(manifest.source_talk_ids, ("talk-a",))
            self.assertEqual(json.loads(manifest_path.read_text())["provenance"]["dataset_name"], DATASET_NAME)

    def test_frozen_dataset_and_split_provenance_fingerprints(self) -> None:
        split = split_manifest("talk-a")
        snapshot = {
            "dataset_name": DATASET_NAME,
            "snapshot_version": "1.0.0",
            "manifest_checksum": "d" * 64,
            "split_manifest_checksum": stable_fingerprint(split),
        }
        provenance = build_artifact_provenance(
            dataset_snapshot_manifest=snapshot,
            dataset_manifest_path="data/manifests/timelymt-streaming-dataset-v1.json",
            split_manifest=split,
            split_manifest_path="data/splits/experimental.json",
            translator=self.identity,
        )
        self.assertEqual(provenance.split_checksum, stable_fingerprint(split))
        self.assertEqual(provenance.translator_config_fingerprint, stable_fingerprint(asdict(load_config(CONFIG_PATH))))

    def test_dataset_document_is_not_mutated(self) -> None:
        before = json.dumps(self.canonical, ensure_ascii=False, sort_keys=True)
        runtime_talk_from_canonical(
            self.canonical, split_manifest=self.split, observed_through_token_index=1,
        )
        self.assertEqual(json.dumps(self.canonical, ensure_ascii=False, sort_keys=True), before)
        self.assertEqual(canonical_content_checksum(self.canonical), canonical_content_checksum(canonical_talk()))


class CacheCompatibilityTests(unittest.TestCase):
    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            ids = [[index + 1 for index, _ in enumerate(text.split())] + [99] for text in texts]
            if kwargs.get("return_tensors") == "pt":
                return {"input_ids": self.Tensor(ids)}
            return {"input_ids": ids}

        def batch_decode(self, generated, **kwargs):
            return ["ban dich" for _ in generated]

        class Tensor:
            def __init__(self, values): self.values = values
            def to(self, device): return self

    class FakeModel:
        def __init__(self): self.calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return [[1, 2, 0] for _ in kwargs["input_ids"].values]

    class FakeTorch:
        class _Context:
            def __enter__(self): return None
            def __exit__(self, *args): return False

        @staticmethod
        def inference_mode(): return CacheCompatibilityTests.FakeTorch._Context()

    def test_request_translation_reuses_existing_translator_cache(self) -> None:
        identity = translator_identity(load_config(CONFIG_PATH))
        runtime = runtime_talk_from_canonical(
            canonical_talk(), split_manifest=split_manifest("talk-a"), observed_through_token_index=1,
        )
        request = make_translation_request(runtime, 0, 1, translator=identity)
        model = self.FakeModel()

        def loader(config, device):
            return _Runtime(self.FakeTokenizer(), model, self.FakeTorch(), device, "float32", config.model_revision, 1)

        with tempfile.TemporaryDirectory() as directory:
            config = load_config(CONFIG_PATH)
            first = EnViT5Translator(config, cache=TranslationCache(directory), _runtime_loader=loader, _cuda_available=False)
            second = EnViT5Translator(config, cache=TranslationCache(directory), _runtime_loader=loader, _cuda_available=False)
            miss = translate_requests(first, [request], translator_identity=identity)[0]
            hit = translate_requests(second, [request], translator_identity=identity)[0]
            self.assertFalse(miss.cache_hit)
            self.assertTrue(hit.cache_hit)
            self.assertEqual(model.calls, 1)
            self.assertEqual(miss.translated_text, hit.translated_text)


if __name__ == "__main__":
    unittest.main()
