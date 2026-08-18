# Frozen English-to-Vietnamese Translator

## Role

TimelyMT studies when an incremental English-to-Vietnamese hypothesis should be
committed. Translation is therefore a fixed upstream operation rather than the
research contribution. Translation model weights remain frozen for all core
experiments; TimelyMT does not fine-tune the translator in its core scope.

M1.1 provides a provider-neutral `Translator` contract and one implementation,
`EnViT5Translator`, backed by `VietAI/envit5-translation`. No model comparison is
part of this milestone.

## Source-Only Contract

The public boundary is intentionally narrow:

```python
Translator.translate(text: str) -> TranslationResult
Translator.translate_batch(texts: Sequence[str]) -> list[TranslationResult]
```

`TranslationResult` contains the original source text, Vietnamese hypothesis,
token counts when available, and lightweight inference metadata. The API has no
parameter for a target reference, alignment, canonical talk, split information,
or future source context. Gold Vietnamese must never cross this boundary.

## EnViT5 Preprocessing

EnViT5 requires English inputs in the form `en: <source text>`. Here, `en:` is
an internal EnViT5 source-language control prefix. The wrapper adds it only
inside model-specific preprocessing. It does not mutate the returned source
text and never writes the prefix into dataset artifacts. The control prefix is
not part of TimelyMT's semantic source text.

Valid source text is otherwise passed through unchanged. In particular, the
wrapper accepts incomplete prefixes, does not trim them, does not add terminal
punctuation, does not restore subtitle punctuation, and does not attempt sentence
completion. Empty and whitespace-only strings are rejected.

## EnViT5 Output Normalization

EnViT5 decoded English-to-Vietnamese output may begin with its `vi:`
target-language control prefix. That prefix is model formatting, not part of
TimelyMT's semantic Vietnamese target text. At the EnViT5-specific boundary,
the wrapper removes an exact leading `vi:` and then removes at most one
immediately following ASCII space. It preserves output without that prefix and
does not trim, use a language-cleanup regex, remove later `vi:` occurrences,
change case, normalize Unicode, punctuation, or otherwise rewrite content.

Raw decoded text is not persisted. Config version `1.1.0` documents and
fingerprints this deterministic postprocessing rule. Public
`TranslationResult.translated_text`, cache entries, and derived artifacts hold
only the normalized semantic hypothesis. Downstream stability, policy, and
evaluation code must consume that field and never raw model formatting.

`target_token_count` is the EnViT5 tokenizer count of normalized
`translated_text`, with special tokens enabled, matching the source-side
token-count convention. It therefore excludes the removed target control tag.

## Reproducible Inference

`configs/translator/envit5.json` freezes the model identity and decoding policy.
The config pins model revision `840bc88104d5a4277af740eaedb024df8c3093e7`.
M1.1 uses deterministic greedy generation:

- `do_sample: false`
- `num_beams: 1`
- `max_new_tokens: 256`

The default `auto` device policy selects CUDA when available and CPU otherwise.
CUDA inference uses float16; CPU inference uses float32. A caller may explicitly
request `cpu` or `cuda`, and an unavailable requested CUDA device is an error.
Importing `timelymt.translator` does not load model weights. The tokenizer and
model are loaded lazily on first inference or runtime inspection.

The direct pinned `tokenizer.json` path intentionally bypasses AutoTokenizer.
Its validator accepts the Transformers-v5 `extra_special_tokens` API and
checks every pinned `<extra_id_N>` token against its exact ID, preserving the
known-good v4 probe input IDs. TimelyMT does not downgrade PyTorch or
Transformers globally for this frozen translator.

The configured maximum input is 512 tokenizer tokens including the model-specific
`en: ` prefix and special tokens. Tokenization explicitly disables truncation.
Overlength input raises `InputTooLongError`; there is no silent truncation or
sliding-window behavior.

## Batch And Prefix Inference

`translate_batch(texts)` validates every exact source string before model
generation, preserves input order, and uses one tokenized model batch for the
distinct cache misses it receives. It does not implement the batch operation by
calling `translate()` repeatedly. Empty or whitespace-only input and overlength
input are rejected with the same policy as single-item inference.

`translate_prefixes(translator, prefixes, batch_size=8)` is a provider-neutral
utility for a caller-supplied sequence of already-created source prefixes. It
returns `PrefixTranslation` records containing the input index, the exact source
text, and the translation result. It neither derives prefixes nor infers source
sentence boundaries. `batch_size` controls engineering micro-batches only and
does not affect the frozen decoding configuration or output order.

## Translator Result Cache

An optional `TranslationCache` stores disposable, derived inference results. A
recommended location is `outputs/translator/cache/`, outside `data/` and the
frozen M0 artifacts. Its contents are not part of TimelyMT Streaming Dataset v1
and may be deleted and rebuilt at any time.

Entries are content addressed by the exact unmodified source text, model ID,
model revision, deterministic generation configuration, model preprocessing
configuration version, and effective device/dtype class. Thus `hello`,
`hello `, `Hello`, and `hello.` always have distinct keys. Targets, alignment,
talk/split identity, timestamps, and future source are never cache inputs.

The `1.1.0` semantic config version invalidates cache keys from the earlier raw
`vi:` behavior. Those entries are stale derived artifacts and are neither
migrated nor returned under the normalized translator identity.

Cache hits return the persisted translation semantics plus `cache_hit: true` in
metadata and do not invoke `model.generate`. For batch and prefix requests,
hits are resolved first, duplicate misses are generated only once, then all
results are restored to caller order. Writes use a temporary file followed by
an atomic replacement to prevent partial entries after interruption.

For repeatability, retain the frozen config and cache identity fields with any
derived result. Outputs can vary across CPU/CUDA dtype classes, so they are
intentionally partitioned rather than silently reused across those runtimes.

## Download And Model Cache

The first real inference downloads tokenizer and model files from the Hugging
Face Hub into its normal external cache. Model weights are not stored in this
repository. Offline unit tests replace the model boundary and require neither a
network connection nor a GPU.

Smoke usage from the repository root:

```shell
PYTHONPATH=src python -m timelymt.translator.cli \
  --text "Artificial intelligence is changing the world." \
  --device auto \
  --config configs/translator/envit5.json
```

On Windows `cmd.exe`, use `set PYTHONPATH=src` before invoking the Python command.

Kaggle compatibility smoke command (loads only the pinned EnViT5 tokenizer and
model, then translates one synthetic source; it does not run a DEV rollout):

```shell
PYTHONPATH=src python -m timelymt.translator.cli --text "A tiny synthetic English string." --device cuda --config configs/translator/envit5.json
```

## Limitations And Later Work

EnViT5 behavior on incomplete prefixes may be unstable as a prefix grows; that
variation is an experimental signal, not something the wrapper rewrites beyond
the exact target control-tag removal. The wrapper
does not assess translation quality, tune decoding, generate dataset-wide
hypotheses, or benchmark throughput. It also does not split overlength inputs.

M1.2 may consume this frozen source-only API to produce derived translator
artifacts. Later streaming and policy experiments must retain the same frozen
model and generation configuration and remain responsible for controlling their
source input buffer. M1.1 does not implement those experiments.
