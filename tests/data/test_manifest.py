from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.canonical.core import canonical_content_checksum
from timelymt.data.manifest.builder import build_dataset_manifest
from timelymt.data.manifest.core import (
    build_experimental_split,
    dataset_manifest_checksum,
    lookup_split_for_talk,
    validate_split_manifest,
)


def canonical(talk_id: str, *, speaker: str = "Speaker") -> dict[str, object]:
    return {"schema_version": "1.0.0", "talk": {"talk_id": talk_id, "source_language": "en", "target_language": "vi", "speaker": speaker, "domain": "ai_ml", "provider": "synthetic"}, "source": {"language": "en", "segments": [{"segment_id": "en-1", "index": 0, "text": "One.", "start_ms": 0, "end_ms": 1000}]}, "target_reference": {"language": "vi", "segments": [{"segment_id": "vi-1", "index": 0, "text": "Một."}]}, "alignments": [{"alignment_id": "a-1", "source_segment_ids": ["en-1"], "target_segment_ids": ["vi-1"], "method": "manual"}], "stream": {"timing_mode": "simulated", "tokens": [{"token_id": "tok-1", "index": 0, "text": "One", "source_segment_id": "en-1", "segment_index": 0, "emit_ms": 1000}]}, "provenance": {"processing_version": "test", "processed_at": "2026-08-09T00:00:00Z"}}


def write_canonical(root: Path, document: dict[str, object], directory: str | None = None) -> Path:
    talk_id = document["talk"]["talk_id"]  # type: ignore[index]
    path = root / (directory or talk_id) / "streaming-talk.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class DatasetManifestTests(unittest.TestCase):
    def test_manifest_is_deterministically_ordered_and_checksum_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_canonical(root, canonical("talk-b")); write_canonical(root, canonical("talk-a"))
            first = build_dataset_manifest(processed_root=root)
            second = build_dataset_manifest(processed_root=root)
            self.assertEqual([talk["talk_id"] for talk in first["talks"]], ["talk-a", "talk-b"])
            self.assertEqual(dataset_manifest_checksum(first), dataset_manifest_checksum(second))
            self.assertEqual(first["talks"][0]["content_checksum"], canonical_content_checksum(canonical("talk-a")))

    def test_duplicate_or_invalid_canonical_artifacts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_canonical(root, canonical("talk-a"))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_dataset_manifest([path, path], processed_root=root)
            invalid = canonical("talk-b"); invalid["stream"]["tokens"][0]["source_segment_id"] = "missing"  # type: ignore[index]
            invalid_path = write_canonical(root, invalid)
            with self.assertRaisesRegex(ValueError, "reference"):
                build_dataset_manifest([invalid_path], processed_root=root)

    def test_pilot_lookup_and_experimental_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for talk_id in ("talk-a", "talk-b", "talk-c"):
                write_canonical(root, canonical(talk_id))
            manifest = build_dataset_manifest(processed_root=root)
            pilot = {"schema_version": "1.0.0", "split_type": "pilot", "dataset_manifest_checksum": dataset_manifest_checksum(manifest), "talk_ids": ["talk-a", "talk-b", "talk-c"]}
            validate_split_manifest(pilot, manifest)
            self.assertEqual(lookup_split_for_talk(pilot, "talk-b"), "pilot")
            with self.assertRaisesRegex(ValueError, "only 3 talks"):
                build_experimental_split(manifest, seed=42, train_ratio=.8, dev_ratio=.1, test_ratio=.1)
            split = build_experimental_split(manifest, seed=42, train_ratio=.8, dev_ratio=.1, test_ratio=.1, group_by="talk", allow_tiny_dataset=True)
            self.assertEqual(split, build_experimental_split(manifest, seed=42, train_ratio=.8, dev_ratio=.1, test_ratio=.1, group_by="talk", allow_tiny_dataset=True))
            self.assertEqual(set().union(*split["splits"].values()), {"talk-a", "talk-b", "talk-c"})
            overlap = copy.deepcopy(split); overlap["splits"]["dev"].append(overlap["splits"]["train"][0])
            with self.assertRaisesRegex(ValueError, "multiple"):
                validate_split_manifest(overlap, manifest)
            unknown = copy.deepcopy(split); unknown["splits"]["test"] = ["unknown"]
            with self.assertRaisesRegex(ValueError, "unknown"):
                validate_split_manifest(unknown, manifest)
            mismatch = copy.deepcopy(pilot); mismatch["dataset_manifest_checksum"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_split_manifest(mismatch, manifest)

    def test_speaker_groups_remain_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for talk_id, speaker in (("talk-a", "A"), ("talk-b", "A"), ("talk-c", "C"), ("talk-d", "D")):
                write_canonical(root, canonical(talk_id, speaker=speaker))
            manifest = build_dataset_manifest(processed_root=root)
            split = build_experimental_split(manifest, seed=7, train_ratio=.5, dev_ratio=.25, test_ratio=.25, allow_tiny_dataset=True)
            assignments = {talk_id: name for name, ids in split["splits"].items() for talk_id in ids}
            self.assertEqual(assignments["talk-a"], assignments["talk-b"])
            missing = copy.deepcopy(manifest); missing["talks"][0].pop("speaker")
            with self.assertRaisesRegex(ValueError, "Speaker grouping"):
                build_experimental_split(missing, seed=7, train_ratio=.5, dev_ratio=.25, test_ratio=.25, allow_tiny_dataset=True)

    def test_calibration_talks_are_excluded_from_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                write_canonical(root, canonical(f"talk-{index}", speaker=f"Speaker {index}"))
            manifest = build_dataset_manifest(processed_root=root)
            split = build_experimental_split(
                manifest,
                seed=42,
                train_ratio=.7,
                dev_ratio=.15,
                test_ratio=.15,
                test_exclusions=["talk-0", "talk-1", "talk-2"],
            )
            self.assertTrue({"talk-0", "talk-1", "talk-2"}.isdisjoint(split["splits"]["test"]))
            self.assertEqual(split["strategy"]["test_exclusions"], ["talk-0", "talk-1", "talk-2"])


if __name__ == "__main__":
    unittest.main()
