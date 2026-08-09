"""Small EnViT5 translation smoke CLI."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from .cache import TranslationCache
from .envit5 import EnViT5Translator
from .prefix import translate_prefixes


DEFAULT_CONFIG = Path("configs/translator/envit5.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate English source text with frozen EnViT5")
    parser.add_argument("--text", action="append", required=True, help="English source text or incomplete prefix; repeatable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Disposable derived translator cache directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size for repeated --text values")
    parser.add_argument("--json", action="store_true", help="Write one JSON object per input")
    return parser


def main(argv: list[str] | None = None) -> int:
    if isinstance(sys.stdout, io.TextIOWrapper) and isinstance(sys.stderr, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        cache = TranslationCache(args.cache_dir) if args.cache_dir else None
        translator = EnViT5Translator.from_config(args.config, device=args.device, cache=cache)
        results = translate_prefixes(translator, args.text, batch_size=args.batch_size)
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"translation failed: {error}", file=sys.stderr)
        return 1

    for result in results:
        record = {
            "prefix_index": result.prefix_index,
            "source_text": result.source_text,
            "translated_text": result.translation.translated_text,
            "source_token_count": result.translation.source_token_count,
            "target_token_count": result.translation.target_token_count,
            "metadata": dict(result.translation.metadata),
        }
        if args.json:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        else:
            print(f"[{record['prefix_index']}] {record['source_text']}")
            print(f"Vietnamese: {record['translated_text']}")
            print(json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
