from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript, validate_parsed_transcript
from timelymt.data.parsing.ted import TedContinuousTranscriptParser
from timelymt.data.parsing.wit3 import Wit3CaptionParser


FIXTURE = Path(__file__).parents[1] / "fixtures/wit3/sample.en.xml"


class TedContinuousTranscriptParserTests(unittest.TestCase):
    def parse_text(self, text: str, language: str = "en") -> ParsedTranscript:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"source.{language}.txt"
            path.write_text(text, encoding="utf-8")
            return TedContinuousTranscriptParser().parse(path, talk_id="ted-test", language=language)

    def test_conservative_sentence_segmentation_preserves_punctuation(self) -> None:
        transcript = self.parse_text("Hello, everyone; welcome. Dr. Dean is here! Is AI ready? Not yet")
        self.assertEqual(
            [segment.text for segment in transcript.segments],
            ["Hello, everyone; welcome.", "Dr. Dean is here!", "Is AI ready?", "Not yet"],
        )
        self.assertTrue(all(segment.start_ms is None for segment in transcript.segments))
        self.assertTrue(all(segment.end_ms is None for segment in transcript.segments))
        self.assertTrue(all(segment.timing_source == "none" for segment in transcript.segments))

    def test_paragraphs_and_whitespace_are_normalized_in_order(self) -> None:
        transcript = self.parse_text(" First   sentence.\r\nStill first paragraph.\r\n\r\n Second paragraph. ")
        self.assertEqual(
            [segment.text for segment in transcript.segments],
            ["First sentence.", "Still first paragraph.", "Second paragraph."],
        )

    def test_vietnamese_utf8_is_preserved(self) -> None:
        transcript = self.parse_text("Xin chào thế giới! Trí tuệ nhân tạo đang phát triển.", "vi")
        self.assertEqual(transcript.segments[0].text, "Xin chào thế giới!")
        self.assertEqual(transcript.segments[1].text, "Trí tuệ nhân tạo đang phát triển.")
        self.assertEqual(transcript.segments[0].segment_id, "vi-000001")

    def test_segment_ids_are_deterministic(self) -> None:
        first = self.parse_text("One. Two.")
        second = self.parse_text("One. Two.")
        self.assertEqual(
            [segment.segment_id for segment in first.segments],
            [segment.segment_id for segment in second.segments],
        )
        self.assertEqual([segment.index for segment in first.segments], [0, 1])

    def test_empty_transcript_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty TED transcript"):
            self.parse_text(" \r\n\t ")


class Wit3CaptionParserTests(unittest.TestCase):
    def test_selects_talk_and_preserves_caption_order_and_annotations(self) -> None:
        transcript = Wit3CaptionParser().parse(FIXTURE, talk_id="1903", language="en")
        self.assertEqual(transcript.talk_id, "1903")
        self.assertEqual(
            [segment.start_ms for segment in transcript.segments],
            [2939, 7555, 11221, 14500],
        )
        self.assertEqual(transcript.segments[0].text, "(Music)")
        self.assertEqual(transcript.segments[-1].text, "(Applause)")
        self.assertEqual(transcript.segments[1].text, "For any of you who have visited & studied this place...")
        self.assertTrue(all(segment.end_ms is None for segment in transcript.segments))
        self.assertTrue(all(segment.timing_source == "wit3_seekvideo" for segment in transcript.segments))
        self.assertEqual(transcript.provenance["source_metadata"]["speaker"], "Example Speaker")

    def test_parses_one_language_independently(self) -> None:
        transcript = Wit3CaptionParser().parse(FIXTURE, talk_id="1903", language="vi")
        self.assertEqual(transcript.language, "vi")
        self.assertEqual(transcript.segments[0].segment_id, "vi-000001")

    def test_malformed_seekvideo_id_is_rejected(self) -> None:
        xml = "<xml><file><talkid>7</talkid><transcription><seekvideo id='soon'>Text.</seekvideo></transcription></file></xml>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.xml"
            path.write_text(xml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed seekvideo id"):
                Wit3CaptionParser().parse(path, talk_id="7", language="en")

    def test_missing_talk_id_is_rejected(self) -> None:
        xml = "<xml><file><transcription><seekvideo id='1'>Text.</seekvideo></transcription></file></xml>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xml"
            path.write_text(xml, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing a talk ID"):
                Wit3CaptionParser().parse(path, talk_id=None, language="en")

    def test_multiple_talks_require_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple talks"):
            Wit3CaptionParser().parse(FIXTURE, talk_id=None, language="en")

    def test_nested_talk_entries_use_their_own_ids(self) -> None:
        xml = """<xml><file><talk><talkid>7</talkid><transcription><seekvideo id='1'>Seven.</seekvideo></transcription></talk><talk><talkid>8</talkid><transcription><seekvideo id='2'>Eight.</seekvideo></transcription></talk></file></xml>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested.xml"
            path.write_text(xml, encoding="utf-8")
            transcript = Wit3CaptionParser().parse(path, talk_id="8", language="en")
            self.assertEqual(transcript.segments[0].text, "Eight.")


class ParsedTranscriptContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = ParsedTranscript(
            talk_id="talk-1",
            language="en",
            provider="test",
            segmentation_method="test_segments",
            segments=(ParsedSegment("en-000001", 0, "Text.", None, None, "none"),),
            provenance={
                "raw_input_path": "input.txt",
                "source_checksum_sha256": "0" * 64,
                "parser_name": "TestParser",
                "parser_version": "1.0.0",
                "processed_at": "2026-08-09T00:00:00Z",
            },
        )

    def test_provider_neutral_shape_matches_schema_fields(self) -> None:
        document = self.transcript.to_dict()
        schema = json.loads(Path("schemas/parsed-transcript.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(set(document), set(schema["required"]))
        self.assertEqual(set(document["segments"][0]), set(schema["$defs"]["segment"]["required"]))
        self.assertEqual(document["schema_version"], "1.0.0")
        validate_parsed_transcript(self.transcript)

    def test_non_contiguous_indices_are_rejected(self) -> None:
        invalid = replace(
            self.transcript,
            segments=(ParsedSegment("en-000001", 1, "Text.", None, None, "none"),),
        )
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            validate_parsed_transcript(invalid)

    def test_invalid_timestamp_order_is_rejected(self) -> None:
        invalid = replace(
            self.transcript,
            segments=(ParsedSegment("en-000001", 0, "Text.", 10, 9, "other"),),
        )
        with self.assertRaisesRegex(ValueError, "precedes"):
            validate_parsed_transcript(invalid)


if __name__ == "__main__":
    unittest.main()
