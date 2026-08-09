"""Disposable content-addressed cache for frozen translator results."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .core import TranslationResult


CACHE_FORMAT_VERSION = 1


class TranslationCache:
    """File cache whose entries can be safely deleted and regenerated."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(*, source_text: str, model_id: str, model_revision: str | None,
            generation_parameters: Mapping[str, Any], config_version: str,
            device: str, dtype: str) -> str:
        """Return a key without normalizing the source text."""

        identity = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "config_version": config_version,
            "device": device,
            "dtype": dtype,
            "generation_parameters": dict(generation_parameters),
            "model_id": model_id,
            "model_revision": model_revision,
            "source_text": source_text,
        }
        payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str, *, source_text: str | None = None) -> TranslationResult | None:
        path = self.directory / f"{key}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if document.get("format_version") != CACHE_FORMAT_VERSION or document.get("key") != key:
                return None
            result = document["result"]
            if not isinstance(result["translated_text"], str) or not isinstance(result["source_text"], str):
                return None
            if source_text is not None and result["source_text"] != source_text:
                return None
            return TranslationResult(
                translated_text=result["translated_text"],
                source_text=result["source_text"],
                source_token_count=result.get("source_token_count"),
                target_token_count=result.get("target_token_count"),
                metadata=result.get("metadata", {}),
            )
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, result: TranslationResult) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{key}.json"
        document = {"format_version": CACHE_FORMAT_VERSION, "key": key, "result": asdict(result)}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.directory, prefix=f".{key}.", suffix=".tmp", delete=False
        ) as handle:
            json.dump(document, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            temporary = Path(handle.name)
        temporary.replace(destination)
