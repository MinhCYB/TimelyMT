"""Frozen VietAI EnViT5 English-to-Vietnamese translator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .cache import TranslationCache
from .core import InputTooLongError, TranslationResult, Translator, validate_source_text


SUPPORTED_CONFIG_VERSION = "1.1.0"
SUPPORTED_MODEL_ID = "VietAI/envit5-translation"
SUPPORTED_DEVICES = {"auto", "cpu", "cuda"}
SUPPORTED_DTYPES = {"cpu": "float32", "cuda": "float16"}
MODEL_PREFIX = "en: "
TARGET_CONTROL_PREFIX = "vi:"


@dataclass(frozen=True)
class EnViT5Config:
    config_version: str
    model_id: str
    model_revision: str | None
    source_language: str
    target_language: str
    device_policy: str
    dtype_policy: Mapping[str, str]
    generation_parameters: Mapping[str, Any]
    maximum_input_handling: Mapping[str, Any]
    frozen: bool

    @property
    def max_input_tokens(self) -> int:
        return int(self.maximum_input_handling["max_input_tokens"])


@dataclass
class _Runtime:
    tokenizer: Any
    model: Any
    torch: Any
    device: str
    dtype: str
    resolved_revision: str | None
    parameter_count: int | None


RuntimeLoader = Callable[[EnViT5Config, str], _Runtime]


def load_config(path: Path | str) -> EnViT5Config:
    """Load and strictly validate the frozen EnViT5 configuration."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse translator config {config_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("translator config must be a JSON object")

    required = set(EnViT5Config.__dataclass_fields__)
    missing = required - document.keys()
    unknown = document.keys() - required
    if missing:
        raise ValueError(f"translator config missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"translator config has unknown fields: {', '.join(sorted(unknown))}")

    config = EnViT5Config(**document)
    _validate_config(config)
    return config


def _validate_config(config: EnViT5Config) -> None:
    if config.config_version != SUPPORTED_CONFIG_VERSION:
        raise ValueError(f"unsupported translator config version: {config.config_version!r}")
    if config.model_id != SUPPORTED_MODEL_ID:
        raise ValueError(f"unsupported translator model: {config.model_id!r}")
    if config.model_revision is not None and not isinstance(config.model_revision, str):
        raise ValueError("model_revision must be a string or null")
    if (config.source_language, config.target_language) != ("en", "vi"):
        raise ValueError("translator languages must be en -> vi")
    if config.device_policy not in SUPPORTED_DEVICES:
        raise ValueError(f"unsupported device policy: {config.device_policy!r}")
    if dict(config.dtype_policy) != SUPPORTED_DTYPES:
        raise ValueError(f"dtype_policy must be {SUPPORTED_DTYPES!r}")
    if config.frozen is not True:
        raise ValueError("translator must be frozen")

    generation = config.generation_parameters
    if not isinstance(generation, dict):
        raise ValueError("generation_parameters must be an object")
    required_generation = {"do_sample", "num_beams", "max_new_tokens"}
    if set(generation) != required_generation:
        raise ValueError("generation_parameters must contain only do_sample, num_beams, and max_new_tokens")
    if generation["do_sample"] is not False or generation["num_beams"] != 1:
        raise ValueError("generation must use deterministic greedy decoding")
    if isinstance(generation["max_new_tokens"], bool) or not isinstance(generation["max_new_tokens"], int):
        raise ValueError("max_new_tokens must be an integer")
    if generation["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be positive")

    handling = config.maximum_input_handling
    expected_handling_keys = {"strategy", "max_input_tokens", "includes_model_prefix"}
    if not isinstance(handling, dict) or set(handling) != expected_handling_keys:
        raise ValueError(f"maximum_input_handling must contain {sorted(expected_handling_keys)!r}")
    if handling["strategy"] != "error" or handling["includes_model_prefix"] is not True:
        raise ValueError("maximum input handling must error and count the model prefix")
    maximum = handling["max_input_tokens"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("max_input_tokens must be a positive integer")


def resolve_device(policy: str, *, cuda_available: bool | None = None) -> str:
    """Resolve auto/cpu/cuda without assuming CUDA exists."""

    if policy not in SUPPORTED_DEVICES:
        raise ValueError(f"unsupported device policy: {policy!r}")
    if cuda_available is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError("torch is required for EnViT5 inference") from error
        cuda_available = torch.cuda.is_available()
    if policy == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if policy == "cuda" or (policy == "auto" and cuda_available) else "cpu"


class EnViT5Translator(Translator):
    """Lazy, frozen EnViT5 wrapper with deterministic source-only inference."""

    def __init__(
        self,
        config: EnViT5Config,
        *,
        device: str | None = None,
        cache: TranslationCache | None = None,
        _runtime_loader: RuntimeLoader | None = None,
        _cuda_available: bool | None = None,
    ) -> None:
        selected_policy = config.device_policy if device is None else device
        self.config = config
        self.device = resolve_device(selected_policy, cuda_available=_cuda_available)
        self._runtime_loader = _runtime_loader or _load_runtime
        self._runtime: _Runtime | None = None
        self.cache = cache

    @classmethod
    def from_config(
        cls,
        path: Path | str,
        *,
        device: str | None = None,
        cache: TranslationCache | None = None,
        _runtime_loader: RuntimeLoader | None = None,
        _cuda_available: bool | None = None,
    ) -> EnViT5Translator:
        return cls(
            load_config(path),
            device=device,
            cache=cache,
            _runtime_loader=_runtime_loader,
            _cuda_available=_cuda_available,
        )

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    def translate(self, text: str) -> TranslationResult:
        return self.translate_batch([text])[0]

    def translate_batch(self, texts: Sequence[str]) -> list[TranslationResult]:
        source_texts = list(texts)
        for text in source_texts:
            validate_source_text(text)
        if not source_texts:
            return []

        keys = [self._cache_key(text) for text in source_texts]
        results: list[TranslationResult | None] = [None] * len(source_texts)
        misses: dict[str, list[int]] = {}
        for index, key in enumerate(keys):
            cached = self.cache.get(key, source_text=source_texts[index]) if self.cache else None
            if cached is not None:
                results[index] = self._with_cache_hit(cached, True)
            else:
                misses.setdefault(key, []).append(index)
        if misses:
            miss_sources = [source_texts[indexes[0]] for indexes in misses.values()]
            translated_misses = self._translate_uncached_batch(miss_sources)
            for (key, indexes), translated in zip(misses.items(), translated_misses, strict=True):
                if self.cache:
                    self.cache.put(key, translated)
                for index in indexes:
                    results[index] = self._with_cache_hit(translated, False) if self.cache else translated
        return [result for result in results if result is not None]

    def _cache_key(self, text: str) -> str:
        return TranslationCache.key(
            source_text=text,
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            generation_parameters=self.config.generation_parameters,
            config_version=self.config.config_version,
            device=self.device,
            dtype=self.config.dtype_policy[self.device],
        )

    @staticmethod
    def _with_cache_hit(result: TranslationResult, cache_hit: bool) -> TranslationResult:
        return replace(result, metadata={**result.metadata, "cache_hit": cache_hit})

    def _translate_uncached_batch(self, source_texts: Sequence[str]) -> list[TranslationResult]:

        runtime = self._ensure_loaded()
        model_inputs = [MODEL_PREFIX + text for text in source_texts]
        model_token_ids = runtime.tokenizer(
            model_inputs,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        for token_ids in model_token_ids:
            if len(token_ids) > self.config.max_input_tokens:
                raise InputTooLongError(
                    f"source produces {len(token_ids)} model input tokens; "
                    f"maximum is {self.config.max_input_tokens}; input was not truncated"
                )

        source_token_ids = runtime.tokenizer(
            source_texts,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        encoded = runtime.tokenizer(
            model_inputs,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(runtime.device) for name, tensor in encoded.items()}
        with runtime.torch.inference_mode():
            generated = runtime.model.generate(**encoded, **dict(self.config.generation_parameters))
        decoded_hypotheses = runtime.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded_hypotheses) != len(source_texts):
            raise RuntimeError("model returned a different number of hypotheses than source texts")

        hypotheses = [_normalize_hypothesis(hypothesis) for hypothesis in decoded_hypotheses]
        target_token_ids = runtime.tokenizer(
            hypotheses,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        results: list[TranslationResult] = []
        for source, source_ids, hypothesis, hypothesis_ids in zip(
            source_texts,
            source_token_ids,
            hypotheses,
            target_token_ids,
            strict=True,
        ):
            results.append(
                TranslationResult(
                    translated_text=hypothesis,
                    source_text=source,
                    source_token_count=len(source_ids),
                    target_token_count=len(hypothesis_ids),
                    metadata={
                        "model_id": self.config.model_id,
                        "model_revision": runtime.resolved_revision,
                        "device": runtime.device,
                        "dtype": runtime.dtype,
                        "decoding": "greedy",
                    },
                )
            )
        return results

    def runtime_info(self) -> Mapping[str, Any]:
        """Load if necessary and return lightweight engineering metadata."""

        runtime = self._ensure_loaded()
        return {
            "model_id": self.config.model_id,
            "model_revision": runtime.resolved_revision,
            "device": runtime.device,
            "dtype": runtime.dtype,
            "parameter_count": runtime.parameter_count,
            "loaded": True,
        }

    def _ensure_loaded(self) -> _Runtime:
        if self._runtime is None:
            self._runtime = self._runtime_loader(self.config, self.device)
        return self._runtime


def _normalize_hypothesis(decoded_text: str) -> str:
    """Remove only EnViT5's exact leading Vietnamese control prefix."""

    if not decoded_text.startswith(TARGET_CONTROL_PREFIX):
        return decoded_text
    hypothesis = decoded_text[len(TARGET_CONTROL_PREFIX) :]
    return hypothesis[1:] if hypothesis.startswith(" ") else hypothesis


def _load_runtime(config: EnViT5Config, device: str) -> _Runtime:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "EnViT5 inference requires torch, transformers, and sentencepiece; install project dependencies"
        ) from error

    dtype_name = config.dtype_policy[device]
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    revision_args = {"revision": config.model_revision} if config.model_revision else {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_id, **revision_args)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        **revision_args,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    resolved_revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return _Runtime(tokenizer, model, torch, device, dtype_name, resolved_revision, parameter_count)
