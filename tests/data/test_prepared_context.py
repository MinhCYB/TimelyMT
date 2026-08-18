from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.prepared_context import (
    PreparedContextPool,
    load_prepared_context,
    serialize_prepared_context,
    source_text_checksum,
    validate_prepared_context,
    write_prepared_context,
)


ROOT = Path(__file__).parents[2]


def source(classification: str = "SAFE_PRETALK_CONFIRMED") -> dict[str, object]:
    text = "  Exact source text.\nUnicode: café.\n"
    transcript_used = classification == "TRANSCRIPT_DERIVED"
    reference_used = classification == "REFERENCE_DERIVED"
    available_before_talk = classification not in {"PUBLIC_POST_TALK", "UNAVAILABLE"}
    return {
        "source_id": "source-1",
        "source_type": "paper",
        "text": text,
        "source_uri": "https://example.invalid/paper",
        "language": "en",
        "published_at": "2025-01-02T03:04:05Z",
        "acquired_at": None if classification == "UNAVAILABLE" else "2025-02-03T04:05:06+00:00",
        "available_before_talk": available_before_talk,
        "classification": classification,
        "relationship": "The speaker authored this paper about the talk topic.",
        "transcript_used": transcript_used,
        "reference_used": reference_used,
        "checksum": source_text_checksum(text),
    }


def pool(*sources: dict[str, object], split: str = "train") -> dict[str, object]:
    return {
        "schema_version": "prepared-context-v0",
        "talk_id": "talk-1",
        "split": split,
        "metadata": {"title": "Informational title", "speaker": "Speaker", "domain": "science"},
        "sources": list(sources),
    }


class PreparedContextTests(unittest.TestCase):
    def test_schema_is_closed_and_declares_v0_contract(self) -> None:
        schema = json.loads((ROOT / "schemas/prepared-context-pool.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], "prepared-context-v0")
        self.assertEqual(schema["properties"]["split"]["enum"], ["train", "dev"])
        self.assertFalse(schema["$defs"]["source"]["additionalProperties"])
        self.assertEqual(schema["$defs"]["source"]["properties"]["checksum"]["pattern"], "^sha256:[0-9a-f]{64}$")

    def test_empty_pool_with_metadata_is_valid_and_metadata_is_not_context(self) -> None:
        document = pool()
        validate_prepared_context(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = load_prepared_context(path)
        self.assertEqual(loaded.metadata.title, "Informational title")
        self.assertEqual(loaded.sources, ())
        self.assertEqual(loaded.eligible_sources(), ())
        self.assertFalse(hasattr(loaded, "all_context_text"))

    def test_confirmed_source_is_valid_and_eligible(self) -> None:
        validate_prepared_context(pool(source()))
        loaded = self._load(pool(source()))
        self.assertEqual(loaded.eligible_sources(), loaded.sources)
        self.assertTrue(loaded.sources[0].model_eligible)

    def test_non_derived_unsafe_classifications_are_inventory_only(self) -> None:
        for classification in (
            "SAFE_PRETALK_PLAUSIBLE", "PUBLIC_POST_TALK", "QUESTIONABLE", "UNAVAILABLE",
        ):
            with self.subTest(classification=classification):
                loaded = self._load(pool(source(classification)))
                self.assertEqual(loaded.eligible_sources(), ())

    def test_transcript_and_reference_derived_sources_are_inventory_only(self) -> None:
        for classification in ("TRANSCRIPT_DERIVED", "REFERENCE_DERIVED"):
            with self.subTest(classification=classification):
                loaded = self._load(pool(source(classification)))
                self.assertEqual(loaded.eligible_sources(), ())
                self.assertFalse(loaded.sources[0].model_eligible)

    def test_confirmed_source_with_leaking_or_post_talk_provenance_fails(self) -> None:
        for field, value in (
            ("transcript_used", True),
            ("reference_used", True),
            ("available_before_talk", False),
        ):
            with self.subTest(field=field):
                invalid = source()
                invalid[field] = value
                with self.assertRaisesRegex(ValueError, "contradictory provenance"):
                    validate_prepared_context(pool(invalid))

    def test_derived_classifications_must_disclose_derivation(self) -> None:
        for classification, field in (
            ("TRANSCRIPT_DERIVED", "transcript_used"),
            ("REFERENCE_DERIVED", "reference_used"),
        ):
            with self.subTest(classification=classification):
                invalid = source(classification)
                invalid[field] = False
                with self.assertRaisesRegex(ValueError, "must declare"):
                    validate_prepared_context(pool(invalid))

    def test_wrong_checksum_and_silent_text_normalization_fail(self) -> None:
        invalid = source()
        invalid["checksum"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_prepared_context(pool(invalid))
        changed = source()
        changed["text"] = str(changed["text"]).strip()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_prepared_context(pool(changed))

    def test_acquired_source_requires_nonempty_text(self) -> None:
        invalid = source()
        invalid["text"] = ""
        invalid["checksum"] = source_text_checksum("")
        with self.assertRaisesRegex(ValueError, "text is required"):
            validate_prepared_context(pool(invalid))

    def test_duplicate_source_id_fails(self) -> None:
        duplicate = source("SAFE_PRETALK_PLAUSIBLE")
        with self.assertRaisesRegex(ValueError, "Duplicate.*source_id"):
            validate_prepared_context(pool(source(), duplicate))

    def test_test_split_fails_without_accessing_test_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "train or dev"):
            validate_prepared_context(pool(split="test"))

    def test_unknown_fields_fail_at_pool_metadata_and_source_levels(self) -> None:
        top_level = pool(); top_level["approved"] = True
        with self.assertRaisesRegex(ValueError, "unsupported or missing fields"):
            validate_prepared_context(top_level)
        metadata = pool(); metadata["metadata"]["description"] = "Not an approved source"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "metadata contains unsupported"):
            validate_prepared_context(metadata)
        item = source(); item["approved"] = True
        with self.assertRaisesRegex(ValueError, "source has unsupported"):
            validate_prepared_context(pool(item))

    def test_invalid_timestamps_fail_and_unknown_publication_time_is_valid(self) -> None:
        unknown = source(); unknown["published_at"] = None
        validate_prepared_context(pool(unknown))
        for field, value in (("published_at", "2025-01-01"), ("acquired_at", "not-a-date"), ("acquired_at", None)):
            with self.subTest(field=field, value=value):
                invalid = source(); invalid[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_prepared_context(pool(invalid))

    def test_serialization_round_trip_is_byte_deterministic_and_preserves_text(self) -> None:
        loaded = self._load(pool(source()), name="input.json")
        first = serialize_prepared_context(loaded)
        self.assertEqual(first, serialize_prepared_context(loaded))
        self.assertIn("  Exact source text.\\nUnicode: café.\\n", first)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            write_prepared_context(first_path, loaded)
            round_tripped = load_prepared_context(first_path)
            write_prepared_context(second_path, round_tripped)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(round_tripped, loaded)

    def test_serialization_revalidates_typed_values(self) -> None:
        loaded = self._load(pool(source()))
        invalid_source = replace(loaded.sources[0], checksum="sha256:" + "0" * 64)
        invalid_pool = PreparedContextPool(
            schema_version=loaded.schema_version,
            talk_id=loaded.talk_id,
            split=loaded.split,
            metadata=loaded.metadata,
            sources=(invalid_source,),
        )
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            serialize_prepared_context(invalid_pool)

    def _load(self, document: dict[str, object], *, name: str = "pool.json") -> PreparedContextPool:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            return load_prepared_context(path)


if __name__ == "__main__":
    unittest.main()
