from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript, write_parsed_transcript
from timelymt.data.parsing.wit3 import Wit3CaptionParser
from timelymt.data.timing.core import (
    build_timed_source,
    serialize_timed_source,
    validate_timed_source,
)
from timelymt.data.timing.simulation import allocate_emit_times, simulated_duration_ms
from timelymt.data.timing.tokenization import lexical_tokens


FIXTURE = Path(__file__).parents[1] / "fixtures/wit3/sample.en.xml"


def transcript(
    texts: list[str],
    *,
    starts_ms: list[int | None] | None = None,
    timing_source: str = "none",
    source_metadata: dict[str, object] | None = None,
) -> ParsedTranscript:
    starts = starts_ms if starts_ms is not None else [None] * len(texts)
    provenance: dict[str, object] = {
        "raw_input_path": "source.en.txt",
        "source_checksum_sha256": "0" * 64,
        "parser_name": "SyntheticParser",
        "parser_version": "1.0.0",
        "processed_at": "2026-08-09T00:00:00Z",
    }
    if source_metadata is not None:
        provenance["source_metadata"] = source_metadata
    return ParsedTranscript(
        talk_id="talk-1",
        language="en",
        provider="synthetic",
        segmentation_method="synthetic",
        segments=tuple(
            ParsedSegment(f"en-{index + 1:06d}", index, text, starts[index], None, timing_source)
            for index, text in enumerate(texts)
        ),
        provenance=provenance,
    )


def build(source: ParsedTranscript, **kwargs):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.en.json"
        write_parsed_transcript(path, source)
        return build_timed_source(source, source_path=path, **kwargs)


class LexicalTokenizationTests(unittest.TestCase):
    def test_punctuation_and_whitespace_are_not_tokens(self) -> None:
        self.assertEqual(
            lexical_tokens("So, what we're building is a new model..."),
            ["So", "what", "we're", "building", "is", "a", "new", "model"],
        )

    def test_apostrophes_hyphens_numbers_acronyms_and_terms_are_conservative(self) -> None:
        self.assertEqual(
            lexical_tokens("Don't re-tokenize GPT-4, NASA, 3.14, C++ or C#."),
            ["Don't", "re-tokenize", "GPT-4", "NASA", "3.14", "C++", "or", "C#"],
        )

    def test_standalone_punctuation_produces_no_runtime_tokens(self) -> None:
        self.assertEqual(lexical_tokens("... -- !!!"), [])


class SimulationTests(unittest.TestCase):
    def test_duration_is_deterministic_and_configurable(self) -> None:
        self.assertEqual(simulated_duration_ms(6, 2.5), 2400)
        self.assertEqual(simulated_duration_ms(6, 2.0), 3000)
        self.assertEqual(simulated_duration_ms(1, 3.0), 333)

    def test_non_finite_speech_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            simulated_duration_ms(1, float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            build(transcript(["One."]), words_per_second=float("inf"))

    def test_uniform_allocation_and_rounding_reach_segment_end(self) -> None:
        self.assertEqual(allocate_emit_times(["a", "b", "c"], 0, 10, "uniform"), [3, 6, 10])

    def test_character_weighted_allocation_favors_longer_tokens(self) -> None:
        self.assertEqual(
            allocate_emit_times(["a", "long"], 100, 600, "character_weighted"),
            [200, 600],
        )

    def test_segments_form_continuous_clock_and_indices_are_global(self) -> None:
        timed = build(transcript(["One two.", "Three four five."]), words_per_second=2.5)
        self.assertEqual([(segment.start_ms, segment.end_ms) for segment in timed.segments], [(0, 800), (800, 2000)])
        tokens = [token for segment in timed.segments for token in segment.tokens]
        self.assertEqual([token.global_index for token in tokens], list(range(5)))
        self.assertEqual(tokens[-1].emit_ms, 2000)
        self.assertEqual(timed.statistics["effective_tokens_per_second"], 2.5)

    def test_empty_token_segment_is_preserved_without_clock_gap(self) -> None:
        timed = build(transcript(["...", "One."]))
        self.assertEqual(timed.segments[0].tokens, ())
        self.assertEqual((timed.segments[0].start_ms, timed.segments[0].end_ms), (0, 0))
        self.assertEqual(timed.segments[1].start_ms, 0)


class RecoveryTests(unittest.TestCase):
    def test_wit3_next_caption_start_is_authoritative_end(self) -> None:
        parsed = Wit3CaptionParser().parse(FIXTURE, talk_id="1903", language="en")
        timed = build(parsed)
        caption = timed.segments[1]
        self.assertEqual((caption.start_ms, caption.end_ms), (7555, 11221))
        self.assertTrue(all(7555 <= token.emit_ms <= 11221 for token in caption.tokens))
        self.assertEqual(caption.tokens[-1].emit_ms, 11221)
        self.assertEqual(timed.timing["mode"], "recovered_from_caption_starts")
        self.assertEqual(timed.timing["parameters"]["original_timing_source"], "wit3_seekvideo")
        self.assertEqual(timed.timing["parameters"]["final_segment_fallback"], "speech_rate_estimate")

    def test_final_caption_prefers_reliable_source_metadata_duration(self) -> None:
        timed = build(
            transcript(
                ["First.", "Final two."],
                starts_ms=[1000, 2000],
                timing_source="original_caption",
                source_metadata={"duration_ms": 5000},
            )
        )
        self.assertEqual(timed.segments[-1].end_ms, 5000)
        self.assertEqual(timed.timing["parameters"]["final_segment_fallback"], "source_metadata_duration")

    def test_duplicate_starts_allow_shared_emit_times(self) -> None:
        timed = build(
            transcript(
                ["Alpha beta.", "Gamma."],
                starts_ms=[10000, 10000],
                timing_source="original_caption",
            )
        )
        self.assertEqual(timed.segments[0].end_ms, 10000)
        self.assertEqual([token.emit_ms for token in timed.segments[0].tokens], [10000, 10000])
        self.assertEqual([token.global_index for segment in timed.segments for token in segment.tokens], [0, 1, 2])

    def test_decreasing_starts_are_rejected_without_sorting(self) -> None:
        source = transcript(
            ["First.", "Second."],
            starts_ms=[2000, 1000],
            timing_source="original_caption",
        )
        with self.assertRaisesRegex(ValueError, "move backward"):
            build(source)


class TimedSourceContractTests(unittest.TestCase):
    def test_schema_shape_and_source_only_content(self) -> None:
        source = transcript(["AI, however, is changing quickly."])
        timed = build(source)
        document = timed.to_dict()
        schema = json.loads(Path("schemas/timed-source.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(set(document), set(schema["required"]))
        self.assertEqual(set(document["segments"][0]), set(schema["$defs"]["segment"]["required"]))
        self.assertEqual([token["text"] for token in document["segments"][0]["tokens"]], ["AI", "however", "is", "changing", "quickly"])
        self.assertEqual(document["segments"][0]["text"], source.segments[0].text)
        self.assertNotIn("target", serialize_timed_source(timed).lower())
        self.assertNotIn("vi", document)

    def test_validation_rejects_non_monotonic_global_emit_time(self) -> None:
        source = transcript(["One two."])
        timed = build(source)
        first, second = timed.segments[0].tokens
        invalid_segment = replace(timed.segments[0], tokens=(replace(first, emit_ms=second.emit_ms), replace(second, emit_ms=0)))
        invalid = replace(timed, segments=(invalid_segment,))
        with self.assertRaisesRegex(ValueError, "monotonic|reach segment"):
            validate_timed_source(invalid, source)

    def test_serialization_is_deterministic(self) -> None:
        timed = build(transcript(["One two."]))
        self.assertEqual(serialize_timed_source(timed), serialize_timed_source(timed))


if __name__ == "__main__":
    unittest.main()
