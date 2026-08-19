from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional
import unittest


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "compare_demo_traces.py"
SPEC = importlib.util.spec_from_file_location("compare_demo_traces", MODULE_PATH)
assert SPEC and SPEC.loader
compare_demo_traces = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_demo_traces)


def event(index, decision="LISTEN", probability: Optional[float] = 0.5, start=0):
    return {"event_index": index, "source_token_end": index, "observation_ms": index * 100,
            "decision": decision, "p_commit": probability, "candidate_source_start": start,
            "candidate_source_end": index, "candidate_translation": None if decision == "WAIT" else "translation",
            "committed_source_text": "source" if decision == "COMMIT" else None,
            "committed_target_text": "target" if decision == "COMMIT" else None,
            "committed_unit_index": index if decision == "COMMIT" else None, "decision_reason": "policy"}


def trace(events):
    return {"talk_id": "talk", "threshold": 0.6, "checkpoint_sha256": "sha", "source_token_count": len(events),
            "source_final_emit_ms": events[-1]["observation_ms"], "events": events}


class CompareDemoTracesTests(unittest.TestCase):
    def test_synchronized_traces_and_wait_nulls_are_accepted(self):
        real = trace([event(0, "WAIT", None), event(1, "LISTEN", 0.7)])
        zero = trace([event(0, "WAIT", None), event(1, "LISTEN", 0.4)])
        report, bookmarks = compare_demo_traces.analyze(real, zero)
        self.assertEqual(report["probability_differences"]["paired_probability_events"], 1)
        self.assertAlmostEqual(report["probability_differences"]["max_absolute_delta"], 0.3)
        self.assertEqual(bookmarks["bookmarks"], [])

    def test_mismatched_source_timeline_is_rejected(self):
        real, zero = trace([event(0)]), trace([event(0)])
        zero["events"][0]["observation_ms"] = 99
        with self.assertRaisesRegex(ValueError, "source timeline mismatch"):
            compare_demo_traces.require_synchronized(real, zero)

    def test_divergence_classification_and_bookmark_indices(self):
        real = trace([event(0, "COMMIT", 0.8), event(1, "LISTEN", 0.2, 1)])
        zero = trace([event(0, "LISTEN", 0.4), event(1, "COMMIT", 0.8, 0)])
        report, bookmarks = compare_demo_traces.analyze(real, zero)
        self.assertEqual(report["divergence_statistics"]["relation_counts"]["REAL_COMMIT_ZERO_LISTEN"], 1)
        self.assertEqual(report["divergence_statistics"]["relation_counts"]["REAL_LISTEN_ZERO_COMMIT"], 1)
        event_indices = {item["event_index"] for item in real["events"]}
        self.assertTrue(all(bookmark["event_index"] in event_indices for bookmark in bookmarks["bookmarks"]))
