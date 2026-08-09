from __future__ import annotations

import json
import unittest

from timelymt.data.acquisition.core import Candidate
from timelymt.data.acquisition.ted import TedAdapter


def page(language: str, transcript: str, include_vi: bool = True) -> str:
    alternates = '<link rel="alternate" hreflang="en" href="?language=en">'
    if include_vi:
        alternates += '<link rel="alternate" hreflang="vi" href="?language=vi">'
    payload = {
        "@graph": [
            {
                "@type": "VideoObject",
                "name": f"Title {language}",
                "duration": "PT1M",
                "transcript": transcript,
                "author": [{"name": "Speaker"}],
            }
        ]
    }
    return f'<html><head>{alternates}<script type="application/ld+json">{json.dumps(payload)}</script></head></html>'


class TedAdapterTests(unittest.TestCase):
    def test_discovers_and_preserves_public_transcripts(self) -> None:
        requested: list[str] = []

        def fetch(url: str) -> str:
            requested.append(url)
            return page("vi", "Xin chào.") if "language=vi" in url else page("en", "Hello.")

        item = Candidate(
            "ted-test", "test", "Title", "Speaker", "ai", "P0", "ted", "https://www.ted.com/talks/test"
        )
        response = TedAdapter(fetch=fetch).acquire(item)
        self.assertTrue(response.discovery.english_available)
        self.assertTrue(response.discovery.vietnamese_available)
        self.assertFalse(response.discovery.subtitle_timing_available)
        self.assertEqual([artifact.filename for artifact in response.artifacts], ["source.en.txt", "target.vi.txt"])
        self.assertEqual(len(requested), 2)

    def test_does_not_request_unadvertised_vietnamese(self) -> None:
        requested: list[str] = []

        def fetch(url: str) -> str:
            requested.append(url)
            return page("en", "Hello.", include_vi=False)

        item = Candidate(
            "ted-test", "test", "Title", "Speaker", "ai", "P0", "ted", "https://www.ted.com/talks/test"
        )
        response = TedAdapter(fetch=fetch).acquire(item)
        self.assertTrue(response.discovery.english_available)
        self.assertFalse(response.discovery.vietnamese_available)
        self.assertEqual(len(requested), 1)


if __name__ == "__main__":
    unittest.main()
