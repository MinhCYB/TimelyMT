"""Adapter for transcript data publicly embedded in TED talk pages."""

from __future__ import annotations

from html.parser import HTMLParser
import json
import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .core import AdapterArtifact, AdapterResponse, Candidate, Discovery


USER_AGENT = "TimelyMT-Research-Acquisition/0.2 (+https://github.com; contact: research-use)"


class _TedPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.languages: set[str] = set()
        self.json_ld: list[str] = []
        self._capture_json_ld = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and values.get("rel") == "alternate" and values.get("hreflang"):
            self.languages.add(values["hreflang"] or "")
        if tag == "script" and values.get("type") == "application/ld+json":
            self._capture_json_ld = True

    def handle_data(self, data: str) -> None:
        if self._capture_json_ld:
            self.json_ld.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._capture_json_ld = False


class TedAdapter:
    provider = "ted"

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        retries: int = 2,
        request_delay: float = 1.0,
        fetch: Callable[[str], str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.retries = retries
        self.request_delay = request_delay
        self._fetch_override = fetch
        self._last_request_at = 0.0

    def acquire(self, candidate: Candidate) -> AdapterResponse:
        english_page = self._fetch(self._language_url(candidate.source_url, "en"))
        english = self._parse_page(english_page)
        languages = english["languages"]
        artifacts: list[AdapterArtifact] = []
        warnings: list[str] = []

        english_text = english.get("transcript")
        if english_text:
            artifacts.append(AdapterArtifact("source.en.txt", english_text))

        vietnamese_advertised = "vi" in languages
        vietnamese_text: str | None = None
        vietnamese_metadata: dict[str, Any] = {}
        if vietnamese_advertised:
            vietnamese_page = self._fetch(self._language_url(candidate.source_url, "vi"))
            vietnamese = self._parse_page(vietnamese_page)
            vietnamese_text = vietnamese.get("transcript")
            vietnamese_metadata = vietnamese.get("metadata", {})
            if vietnamese_text:
                artifacts.append(AdapterArtifact("target.vi.txt", vietnamese_text))
            else:
                warnings.append("Vietnamese is advertised but no public transcript was embedded")

        english_available = bool(english_text)
        vietnamese_available = bool(vietnamese_text)
        discovery = Discovery(
            english_available=english_available,
            vietnamese_available=vietnamese_available,
            transcript_available=english_available or vietnamese_available,
            subtitle_timing_available=False,
        )
        metadata: dict[str, Any] = dict(english.get("metadata", {}))
        metadata["advertised_languages"] = sorted(languages)
        if vietnamese_metadata:
            metadata["vietnamese_page"] = vietnamese_metadata
        warnings.append("TED public talk-page transcripts do not expose subtitle timing")
        return AdapterResponse(discovery, metadata, tuple(artifacts), tuple(warnings))

    def _fetch(self, url: str) -> str:
        if self._fetch_override is not None:
            return self._fetch_override(url)

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
            try:
                request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read()
                self._last_request_at = time.monotonic()
                return body.decode(charset)
            except HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code < 500 or attempt == self.retries:
                    raise
                last_error = error
            except (URLError, TimeoutError, socket.timeout) as error:
                self._last_request_at = time.monotonic()
                last_error = error
                if attempt == self.retries:
                    raise
            if attempt < self.retries:
                time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"TED request failed: {last_error}")

    @staticmethod
    def _language_url(source_url: str, language: str) -> str:
        return f"{source_url}?{urlencode({'language': language})}"

    @staticmethod
    def _parse_page(page: str) -> dict[str, Any]:
        parser = _TedPageParser()
        parser.feed(page)
        video: dict[str, Any] = {}
        for raw_json in parser.json_ld:
            try:
                value = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            graph = value.get("@graph", []) if isinstance(value, dict) else []
            for item in graph:
                if isinstance(item, dict) and item.get("@type") == "VideoObject":
                    video = item
                    break
            if video:
                break
        metadata: dict[str, Any] = {
            key: video[key]
            for key in ("name", "description", "uploadDate", "datePublished", "duration")
            if key in video
        }
        authors = video.get("author")
        if isinstance(authors, list):
            metadata["authors"] = [item.get("name") for item in authors if isinstance(item, dict)]
        transcript = video.get("transcript")
        return {
            "languages": parser.languages,
            "transcript": transcript if isinstance(transcript, str) and transcript.strip() else None,
            "metadata": metadata,
        }
