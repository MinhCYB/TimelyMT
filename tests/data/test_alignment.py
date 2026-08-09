from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from timelymt.data.alignment.core import validate_aligned_transcript
from timelymt.data.alignment.dp import AlignmentParameters, align_transcripts
from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript


def transcript(talk_id: str, language: str, texts: list[str]) -> ParsedTranscript:
    return ParsedTranscript(
        talk_id=talk_id,
        language=language,
        provider="synthetic",
        segmentation_method="synthetic",
        segments=tuple(
            ParsedSegment(f"{language}-{index + 1:06d}", index, text, None, None, "none")
            for index, text in enumerate(texts)
        ),
        provenance={
            "raw_input_path": f"{language}.txt",
            "source_checksum_sha256": "0" * 64,
            "parser_name": "SyntheticParser",
            "parser_version": "1.0.0",
            "processed_at": "2026-08-09T00:00:00Z",
        },
    )


class AlignmentTests(unittest.TestCase):
    def align(
        self,
        source_texts: list[str],
        target_texts: list[str],
        *,
        max_group_size: int = 3,
        skip_penalty: float = 1.6,
        group_penalty: float = 0.65,
    ):
        source = transcript("talk-1", "en", source_texts)
        target = transcript("talk-1", "vi", target_texts)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.en.json"
            target_path = Path(directory) / "target.vi.json"
            source_path.write_text("source", encoding="utf-8")
            target_path.write_text("target", encoding="utf-8")
            aligned = align_transcripts(
                source,
                target,
                source_path=source_path,
                target_path=target_path,
                parameters=AlignmentParameters(max_group_size, skip_penalty, group_penalty),
            )
        return source, target, aligned

    def test_one_to_one(self) -> None:
        _, _, aligned = self.align(["A complete sentence."], ["Một câu hoàn chỉnh."])
        self.assertEqual(aligned.statistics["alignment_type_counts"]["1:1"], 1)

    def test_one_to_two(self) -> None:
        _, _, aligned = self.align(["One sentence with two translated clauses."], ["Một câu.", "Có hai mệnh đề."])
        self.assertEqual(aligned.statistics["alignment_type_counts"]["1:2"], 1)

    def test_two_to_one(self) -> None:
        _, _, aligned = self.align(["First clause.", "Second clause."], ["Mệnh đề thứ nhất và mệnh đề thứ hai."])
        self.assertEqual(aligned.statistics["alignment_type_counts"]["2:1"], 1)

    def test_two_to_two(self) -> None:
        _, _, aligned = self.align(["x", "a" * 100], ["b" * 100, "y"])
        self.assertEqual(aligned.statistics["alignment_type_counts"]["2:2"], 1)

    def test_group_penalty_is_explicit_and_persisted(self) -> None:
        _, _, aligned = self.align(["First.", "Second."], ["Thứ nhất và thứ hai."], group_penalty=0.2)
        self.assertEqual(aligned.method["parameters"]["group_penalty"], 0.2)

    def test_source_skip_and_different_counts(self) -> None:
        _, _, aligned = self.align(["a", "b", "c", "d"], ["một"], max_group_size=3)
        self.assertEqual(len(aligned.unaligned_source_segment_ids), 1)
        self.assertEqual(aligned.statistics["unaligned_source_count"], 1)

    def test_target_skip(self) -> None:
        _, _, aligned = self.align(["one"], ["a", "b", "c", "d"], max_group_size=3)
        self.assertEqual(len(aligned.unaligned_target_segment_ids), 1)
        self.assertEqual(aligned.statistics["unaligned_target_count"], 1)

    def test_annotation_only_segments_align_without_removal(self) -> None:
        _, _, aligned = self.align(["(Applause)"], ["(Applause)"])
        self.assertEqual(aligned.alignments[0].source_text, "(Applause)")
        self.assertEqual(aligned.alignments[0].features["annotation_penalty"], 0.0)

    def test_output_is_deterministic_except_provenance_timestamp(self) -> None:
        _, _, first = self.align(["One.", "Two."], ["Một.", "Hai."])
        _, _, second = self.align(["One.", "Two."], ["Một.", "Hai."])
        self.assertEqual(first.alignments, second.alignments)
        self.assertEqual(first.unaligned_source_segment_ids, second.unaligned_source_segment_ids)
        self.assertEqual(first.statistics, second.statistics)

    def test_all_segments_are_accounted_for_monotonically(self) -> None:
        source, target, aligned = self.align(["a", "b", "c", "d"], ["một", "hai"])
        validate_aligned_transcript(aligned, source, target)
        accounted_source = {
            identifier for unit in aligned.alignments for identifier in unit.source_segment_ids
        } | set(aligned.unaligned_source_segment_ids)
        accounted_target = {
            identifier for unit in aligned.alignments for identifier in unit.target_segment_ids
        } | set(aligned.unaligned_target_segment_ids)
        self.assertEqual(accounted_source, {segment.segment_id for segment in source.segments})
        self.assertEqual(accounted_target, {segment.segment_id for segment in target.segments})

    def test_duplicate_reference_is_rejected(self) -> None:
        source, target, aligned = self.align(["One.", "Two."], ["Một.", "Hai."])
        if len(aligned.alignments) != 2:
            self.skipTest("Synthetic path grouped both units")
        duplicated_unit = replace(
            aligned.alignments[1],
            source_segment_ids=aligned.alignments[0].source_segment_ids,
            source_text=aligned.alignments[0].source_text,
        )
        duplicated = replace(aligned, alignments=(aligned.alignments[0], duplicated_unit))
        with self.assertRaisesRegex(ValueError, "Non-monotonic|Duplicate aligned source"):
            validate_aligned_transcript(duplicated, source, target)

    def test_non_monotonic_reference_is_rejected(self) -> None:
        source, target, aligned = self.align(["One.", "Two."], ["Một.", "Hai."])
        if len(aligned.alignments) != 2:
            self.skipTest("Synthetic path grouped both units")
        reversed_units = (
            replace(aligned.alignments[1], alignment_id="a-000001"),
            replace(aligned.alignments[0], alignment_id="a-000002"),
        )
        invalid = replace(aligned, alignments=reversed_units)
        with self.assertRaisesRegex(ValueError, "Non-monotonic"):
            validate_aligned_transcript(invalid, source, target)

    def test_invalid_language_is_rejected(self) -> None:
        source = transcript("talk-1", "vi", ["Wrong side."])
        target = transcript("talk-1", "vi", ["Đích."])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source"
            target_path = Path(directory) / "target"
            source_path.write_text("source", encoding="utf-8")
            target_path.write_text("target", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source language"):
                align_transcripts(source, target, source_path=source_path, target_path=target_path)

    def test_mismatched_talk_id_is_rejected(self) -> None:
        source = transcript("talk-1", "en", ["Source."])
        target = transcript("talk-2", "vi", ["Đích."])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source"
            target_path = Path(directory) / "target"
            source_path.write_text("source", encoding="utf-8")
            target_path.write_text("target", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "talk IDs"):
                align_transcripts(source, target, source_path=source_path, target_path=target_path)

    def test_empty_transcript_is_rejected(self) -> None:
        source = transcript("talk-1", "en", [])
        target = transcript("talk-1", "vi", ["Đích."])
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source"
            target_path = Path(directory) / "target"
            source_path.write_text("source", encoding="utf-8")
            target_path.write_text("target", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no segments"):
                align_transcripts(source, target, source_path=source_path, target_path=target_path)


if __name__ == "__main__":
    unittest.main()
