from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from timelymt.data.acquisition.core import (
    AcquisitionResult,
    AdapterArtifact,
    AdapterResponse,
    Candidate,
    Discovery,
    ManifestError,
    acquire_candidates,
    artifact_directory,
    load_manifest,
)


def candidate(identifier: str = "talk-1") -> Candidate:
    return Candidate(
        id=identifier,
        slug=identifier,
        title="A talk",
        speaker="A speaker",
        domain="ai",
        priority="P0",
        provider="fake",
        source_url=f"https://example.test/talks/{identifier}",
    )


class FakeAdapter:
    provider = "fake"

    def acquire(self, candidate: Candidate) -> AdapterResponse:
        if candidate.id == "broken":
            raise RuntimeError("provider rejected request")
        return AdapterResponse(
            Discovery(True, True, True, False),
            {"provider_id": "42"},
            (AdapterArtifact("source.en.txt", "Hello.\r\n"), AdapterArtifact("target.vi.txt", "Xin chào.\n")),
        )


class AcquisitionTests(unittest.TestCase):
    def test_manifest_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"candidates": [candidate().__dict__]}), encoding="utf-8")
            self.assertEqual(load_manifest(path), [candidate()])

    def test_duplicate_candidate_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            record = candidate().__dict__
            path.write_text(json.dumps({"candidates": [record, record]}), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "Duplicate candidate id"):
                load_manifest(path)

    def test_result_serialization(self) -> None:
        result = AcquisitionResult(
            "talk-1", "fake", "https://example.test", "2026-01-01T00:00:00Z", "partial", Discovery(True)
        )
        serialized = result.to_dict()
        self.assertEqual(serialized["status"], "partial")
        self.assertTrue(serialized["discovered"]["english_available"])
        self.assertNotIn("failure_reason", serialized)

    def test_artifact_path_generation(self) -> None:
        self.assertEqual(
            artifact_directory(Path("data/streaming/raw"), candidate()),
            Path("data/streaming/raw/fake/talk-1"),
        )

    def test_failure_does_not_stop_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = acquire_candidates(
                [candidate("broken"), candidate("working")],
                {"fake": FakeAdapter()},
                root / "raw",
                root / "results.jsonl",
            )
            self.assertEqual([result.status for result in results], ["failed", "available"])
            self.assertIn("provider rejected request", results[0].failure_reason or "")
            failure_record = root / "raw/fake/broken/acquisition.json"
            self.assertEqual(json.loads(failure_record.read_text(encoding="utf-8"))["status"], "failed")
            source = root / "raw/fake/working/source.en.txt"
            self.assertEqual(source.read_text(encoding="utf-8"), "Hello.\n")
            lines = (root / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)

    def test_empty_transcript_is_rejected_and_logged(self) -> None:
        class EmptyAdapter:
            provider = "fake"

            def acquire(self, candidate: Candidate) -> AdapterResponse:
                return AdapterResponse(Discovery(True), artifacts=(AdapterArtifact("source.en.txt", "  "),))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = acquire_candidates(
                [candidate()], {"fake": EmptyAdapter()}, root / "raw", root / "results.jsonl"
            )[0]
            self.assertEqual(result.status, "failed")
            self.assertIn("Empty transcript artifact rejected", result.failure_reason or "")


if __name__ == "__main__":
    unittest.main()
