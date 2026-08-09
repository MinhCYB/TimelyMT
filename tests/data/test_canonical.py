from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.canonical.builder import build_canonical_talk
from timelymt.data.canonical.core import canonical_content_checksum, load_canonical_talk, serialize_canonical_talk, validate_canonical_talk, write_canonical_talk


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def upstream(root: Path, *, unaligned: bool = False) -> dict[str, Path]:
    talk_id = "talk-1"
    raw, parsed, aligned, timed = (root / name for name in ("raw", "parsed", "aligned", "timed"))
    source_segments = [{"segment_id": "en-000001", "index": 0, "text": "One two.", "start_ms": None, "end_ms": None, "timing_source": "none"}, {"segment_id": "en-000002", "index": 1, "text": "Three.", "start_ms": None, "end_ms": None, "timing_source": "none"}]
    target_segments = [{"segment_id": "vi-000001", "index": 0, "text": "Một hai.", "start_ms": None, "end_ms": None, "timing_source": "none"}, {"segment_id": "vi-000002", "index": 1, "text": "Ba.", "start_ms": None, "end_ms": None, "timing_source": "none"}]
    provenance = {"raw_input_path": "input.txt", "source_checksum_sha256": "0" * 64, "parser_name": "Synthetic", "parser_version": "1.0.0", "processed_at": "2026-08-09T00:00:00Z"}
    write_json(raw / "synthetic" / talk_id / "metadata.json", {"candidate": {"id": talk_id, "provider": "synthetic", "title": "Synthetic", "slug": "source-1"}, "provider_metadata": {"duration": "PT1M"}})
    write_json(raw / "synthetic" / talk_id / "acquisition.json", {"candidate_id": talk_id, "acquired_at": "2026-08-09T00:00:00Z"})
    for language, segments in (("en", source_segments), ("vi", target_segments)):
        write_json(parsed / talk_id / ("source.en.json" if language == "en" else "target.vi.json"), {"schema_version": "1.0.0", "talk_id": talk_id, "language": language, "provider": "synthetic", "segmentation": {"method": "synthetic"}, "segments": segments, "provenance": provenance})
    units = [{"alignment_id": "a-000001", "source_segment_ids": ["en-000001"], "target_segment_ids": ["vi-000001"], "source_text": "One two.", "target_text": "Một hai."}]
    if not unaligned:
        units.append({"alignment_id": "a-000002", "source_segment_ids": ["en-000002"], "target_segment_ids": ["vi-000002"], "source_text": "Three.", "target_text": "Ba."})
    write_json(aligned / talk_id / "alignment.json", {"schema_version": "1.0.0", "talk_id": talk_id, "source_language": "en", "target_language": "vi", "method": {"name": "monotonic_length_position_dp", "version": "1.0.0", "parameters": {"max_group_size": 3, "skip_penalty": 1.6}}, "alignments": units, "unaligned_source_segment_ids": ["en-000002"] if unaligned else [], "unaligned_target_segment_ids": ["vi-000002"] if unaligned else []})
    timed_segments = [{"segment_id": "en-000001", "index": 0, "text": "One two.", "start_ms": 0, "end_ms": 800, "tokens": [{"token_id": "tok-000001", "global_index": 0, "segment_index": 0, "source_segment_id": "en-000001", "text": "One", "emit_ms": 400}, {"token_id": "tok-000002", "global_index": 1, "segment_index": 1, "source_segment_id": "en-000001", "text": "two", "emit_ms": 800}]}, {"segment_id": "en-000002", "index": 1, "text": "Three.", "start_ms": 800, "end_ms": 1200, "tokens": [{"token_id": "tok-000003", "global_index": 2, "segment_index": 0, "source_segment_id": "en-000002", "text": "Three", "emit_ms": 1200}]}]
    write_json(timed / talk_id / "source.en.json", {"schema_version": "1.0.0", "talk_id": talk_id, "language": "en", "timing": {"mode": "simulated", "parameters": {"words_per_second": 2.5, "allocation": "character_weighted", "original_timing_source": None, "final_segment_fallback": "not_applicable"}}, "segments": timed_segments, "provenance": {"timing_tool": "timelymt.data.timing", "timing_version": "1.0.0"}})
    return {"raw": raw, "parsed": parsed, "aligned": aligned, "timed": timed}


def build(paths: dict[str, Path]) -> dict[str, object]:
    return build_canonical_talk("talk-1", raw_root=paths["raw"], parsed_root=paths["parsed"], aligned_root=paths["aligned"], timed_root=paths["timed"])


class CanonicalBuilderTests(unittest.TestCase):
    def test_successful_assembly_round_trip_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = upstream(Path(directory))
            first, second = build(paths), build(paths)
            self.assertEqual(canonical_content_checksum(first), canonical_content_checksum(second))
            self.assertNotIn("vi-000001", str(first["stream"]))
            output = Path(directory) / "processed/talk-1/streaming-talk.json"
            write_canonical_talk(output, first)
            self.assertEqual(load_canonical_talk(output), first)
            self.assertEqual(serialize_canonical_talk(first), serialize_canonical_talk(first))

    def test_missing_upstream_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = upstream(Path(directory))
            (paths["timed"] / "talk-1/source.en.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing required upstream artifact"):
                build(paths)

    def test_mismatched_ids_source_ids_and_source_text_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = upstream(Path(directory)); path = paths["timed"] / "talk-1/source.en.json"; document = json.loads(path.read_text(encoding="utf-8"))
            document["talk_id"] = "wrong-talk"; write_json(path, document)
            with self.assertRaisesRegex(ValueError, "Talk IDs"): build(paths)
            document["talk_id"] = "talk-1"; document["segments"][0]["segment_id"] = "en-999999"; write_json(path, document)
            with self.assertRaisesRegex(ValueError, "segment IDs"): build(paths)
            document["segments"][0]["segment_id"] = "en-000001"; document["segments"][0]["text"] = "Changed."; write_json(path, document)
            with self.assertRaisesRegex(ValueError, "text differ"): build(paths)

    def test_invalid_alignment_reference_and_target_stream_leakage_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = upstream(Path(directory)); path = paths["aligned"] / "talk-1/alignment.json"; document = json.loads(path.read_text(encoding="utf-8"))
            document["alignments"][0]["source_segment_ids"] = ["en-999999"]; write_json(path, document)
            with self.assertRaisesRegex(ValueError, "Invalid alignment reference"): build(paths)
            talk = build(upstream(Path(directory) / "clean")); leaked = copy.deepcopy(talk); leaked_stream = leaked["stream"]; assert isinstance(leaked_stream, dict); leaked_tokens = leaked_stream["tokens"]; assert isinstance(leaked_tokens, list); leaked_tokens[0]["target_text"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "Canonical stream token"): validate_canonical_talk(leaked)

    def test_stream_resolution_and_timing_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            talk = build(upstream(Path(directory)))
            stream = talk["stream"]; assert isinstance(stream, dict); self.assertEqual(stream["timing_mode"], "simulated")
            recovered = copy.deepcopy(talk); recovered_stream = recovered["stream"]; assert isinstance(recovered_stream, dict); recovered_stream["timing_mode"] = "recovered_from_caption_starts"; parameters = recovered_stream["timing_parameters"]; assert isinstance(parameters, dict); parameters["original_timing_source"] = "wit3_seekvideo"; validate_canonical_talk(recovered)
            invalid = copy.deepcopy(talk); invalid_stream = invalid["stream"]; assert isinstance(invalid_stream, dict); invalid_tokens = invalid_stream["tokens"]; assert isinstance(invalid_tokens, list); invalid_tokens[0]["source_segment_id"] = "en-999999"
            with self.assertRaisesRegex(ValueError, "reference"): validate_canonical_talk(invalid)

    def test_unaligned_segments_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            talk = build(upstream(Path(directory), unaligned=True))
            source = talk["source"]; target = talk["target_reference"]; provenance = talk["provenance"]; alignments = talk["alignments"]; assert isinstance(source, dict) and isinstance(target, dict) and isinstance(provenance, dict) and isinstance(alignments, list); self.assertEqual((len(source["segments"]), len(target["segments"]), len(alignments)), (2, 2, 1))
            metadata = provenance["metadata"]; assert isinstance(metadata, dict); self.assertEqual(metadata["unaligned_source_segment_ids"], ["en-000002"])


if __name__ == "__main__":
    unittest.main()
