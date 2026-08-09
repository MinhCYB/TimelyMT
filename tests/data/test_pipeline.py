from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from timelymt.data.acquisition.core import Candidate
from timelymt.data.alignment.dp import AlignmentParameters
from timelymt.data.pipeline.calibration import (
    import_alignment_review_tsv,
    load_alignment_config,
    write_alignment_config,
    write_alignment_review_tsv,
)
from timelymt.data.pipeline.core import PipelinePaths, prepare_dataset, process_talk
from timelymt.data.pipeline.qa import build_quality_report, build_snapshot, qa_flags, stable_checksum


def candidate(talk_id: str) -> Candidate:
    return Candidate(talk_id, talk_id, "Title", "Speaker", "ai_ml", "P0", "ted", "https://example.com/talk")


def canonical(talk_id: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "talk": {"talk_id": talk_id, "source_language": "en", "target_language": "vi", "speaker": "Speaker", "domain": "ai_ml", "provider": "ted"},
        "source": {"language": "en", "segments": [{"segment_id": "en-1", "index": 0, "text": "One.", "start_ms": 0, "end_ms": 1000}]},
        "target_reference": {"language": "vi", "segments": [{"segment_id": "vi-1", "index": 0, "text": "Một."}]},
        "alignments": [{"alignment_id": "a-1", "source_segment_ids": ["en-1"], "target_segment_ids": ["vi-1"], "method": "manual"}],
        "stream": {"timing_mode": "simulated", "tokens": [{"token_id": "tok-1", "index": 0, "text": "One", "source_segment_id": "en-1", "segment_index": 0, "emit_ms": 1000}]},
        "provenance": {"processing_version": "test", "processed_at": "2026-08-09T00:00:00Z"},
    }


class PipelineTests(unittest.TestCase):
    def test_alignment_review_tsv_round_trip_updates_only_review_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "alignment-calibration.json"
            tsv_path = root / "alignment-calibration-review.tsv"
            parsed_root = root / "parsed"
            document = review_document()
            review_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            write_alignment_review_tsv(tsv_path, document)
            tsv = tsv_path.read_text(encoding="utf-8")
            self.assertIn('"English\ttext\ncontinued"', tsv)
            self.assertIn("Tiếng Việt\tnguyên vẹn", tsv)
            self.assertTrue(tsv.endswith('\t\t[]\t[]\t\n'))
            tsv_path.write_text(
                tsv.replace('\t\t[]\t[]\t\n', '\tcorrect\t["en-000002"]\t["vi-000002"]\tChecked\n'),
                encoding="utf-8",
                newline="",
            )
            write_parsed_transcript(parsed_root / "talk-1/source.en.json", "en", ["en-000001", "en-000002"])
            write_parsed_transcript(parsed_root / "talk-1/target.vi.json", "vi", ["vi-000001", "vi-000002"])

            self.assertEqual(import_alignment_review_tsv(tsv_path, review_path, parsed_root=parsed_root), 1)
            imported = json.loads(review_path.read_text(encoding="utf-8"))
            self.assertEqual(imported["examples"][0]["review"], {
                "verdict": "correct",
                "preferred_source_ids": ["en-000002"],
                "preferred_target_ids": ["vi-000002"],
                "note": "Checked",
            })
            self.assertEqual(imported["examples"][0]["source_text"], "English\ttext\ncontinued")
            self.assertEqual(imported["examples"][0]["selection_reasons"], ["beginning", "1:1"])

    def test_alignment_review_import_rejects_invalid_verdict_and_metadata_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "alignment-calibration.json"
            tsv_path = root / "alignment-calibration-review.tsv"
            document = review_document()
            review_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            write_alignment_review_tsv(tsv_path, document)
            tsv_path.write_text(
                tsv_path.read_text(encoding="utf-8").replace("\t\t[]\t[]\t\n", "\twrong\t[]\t[]\t\n"),
                encoding="utf-8",
                newline="",
            )
            with self.assertRaisesRegex(ValueError, "Invalid verdict"):
                import_alignment_review_tsv(tsv_path, review_path, parsed_root=root / "parsed")

            write_alignment_review_tsv(tsv_path, document)
            tsv_path.write_text(
                tsv_path.read_text(encoding="utf-8").replace("English", "Changed"), encoding="utf-8", newline=""
            )
            with self.assertRaisesRegex(ValueError, "Calibration metadata changed"):
                import_alignment_review_tsv(tsv_path, review_path, parsed_root=root / "parsed")

    def test_resume_starts_at_first_missing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = PipelinePaths(
                raw_root=root / "raw",
                parsed_root=root / "parsed",
                aligned_root=root / "aligned",
                timed_root=root / "timed",
                processed_root=root / "processed",
                acquisition_results=root / "results.jsonl",
            )
            talk = candidate("talk-1")
            for path in (paths.raw_root / "ted/talk-1/acquisition.json", paths.raw_root / "ted/talk-1/source.en.txt", paths.raw_root / "ted/talk-1/target.vi.txt", paths.parsed_root / "talk-1/source.en.json", paths.parsed_root / "talk-1/target.vi.json"):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text("x", encoding="utf-8")
            calls: list[str] = []
            def run(stage, *_args):
                calls.append(stage)
                outputs = {
                    "align": [paths.aligned_root / "talk-1/alignment.json"],
                    "time": [paths.timed_root / "talk-1/source.en.json"],
                    "canonical": [paths.processed_root / "talk-1/streaming-talk.json"],
                }[stage]
                for output in outputs:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(canonical("talk-1")) if stage == "canonical" else "x", encoding="utf-8")
            with patch("timelymt.data.pipeline.core._run_stage", side_effect=run):
                result = process_talk(talk, adapters={}, paths=paths)
            self.assertEqual(calls, ["align", "time", "canonical"])
            self.assertEqual(result["status"], "accepted")

    def test_talk_failure_does_not_stop_batch(self) -> None:
        with patch("timelymt.data.pipeline.core.process_talk", side_effect=[{"talk_id": "a", "status": "failed"}, {"talk_id": "b", "status": "accepted"}]):
            results = prepare_dataset([candidate("a"), candidate("b")], adapters={})
        self.assertEqual([item["status"] for item in results], ["failed", "accepted"])

    def test_alignment_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alignment.json"
            expected = AlignmentParameters(3, 1.4, 0.4)
            write_alignment_config(path, expected, selection_basis="manual_calibration_v1")
            self.assertEqual(load_alignment_config(path), expected)

    def test_alignment_config_permits_confirmed_four_way_group(self) -> None:
        parameters = AlignmentParameters(4, 1.4, 0.4)
        parameters.validate()
        self.assertEqual(parameters.max_group_size, 4)

    def test_qa_flags_and_snapshot_are_stable(self) -> None:
        talk = canonical("talk-1")
        self.assertIn("extremely_short_talk", qa_flags(talk))
        manifest = {"talks": [{"timing_mode": "simulated", "provider": "ted"}]}
        config = {"version": "1.0.0", "max_group_size": 3, "skip_penalty": 1.6, "group_penalty": .65}
        first = build_snapshot(manifest, split=None, alignment_config=config, known_limitations=[])
        second = build_snapshot(manifest, split=None, alignment_config=config, known_limitations=[])
        self.assertEqual(stable_checksum(first), stable_checksum(second))

    def test_quality_report_aggregates_and_preserves_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "talk-1/streaming-talk.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(canonical("talk-1")), encoding="utf-8")
            report = build_quality_report([path], failed_records=[{"talk_id": "talk-2", "status": "failed"}], aligned_root=root / "aligned")
            self.assertEqual(report["summary"]["accepted"], 1)
            self.assertEqual(report["summary"]["failed_or_rejected"], 1)
            self.assertEqual(report["summary"]["stream_tokens"], 1)


if __name__ == "__main__":
    unittest.main()


def review_document() -> dict[str, object]:
    return {
        "allowed_verdicts": ["correct", "questionable", "incorrect"],
        "examples": [{
            "talk_id": "talk-1",
            "alignment_id": "a-1",
            "current_alignment_type": "1:1",
            "current_cost": 0.5,
            "selection_reasons": ["beginning", "1:1"],
            "source_segment_ids": ["en-000001"],
            "source_text": "English\ttext\ncontinued",
            "target_segment_ids": ["vi-000001"],
            "target_text": "Tiếng Việt\tnguyên vẹn",
            "review": {"verdict": None, "preferred_source_ids": [], "preferred_target_ids": [], "note": "Requires human inspection."},
        }],
    }


def write_parsed_transcript(path: Path, language: str, identifiers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "talk_id": "talk-1",
        "language": language,
        "provider": "test",
        "segmentation": {"method": "test"},
        "segments": [
            {
                "segment_id": identifier,
                "index": index,
                "text": identifier,
                "start_ms": None,
                "end_ms": None,
                "timing_source": "none",
            }
            for index, identifier in enumerate(identifiers)
        ],
        "provenance": {},
    }), encoding="utf-8")
