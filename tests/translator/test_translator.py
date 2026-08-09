from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from timelymt.translator import (
    InputTooLongError,
    TranslationCache,
    TranslationResult,
    Translator,
    translate_prefixes,
)
from timelymt.translator.envit5 import (
    MODEL_PREFIX,
    EnViT5Translator,
    _Runtime,
    load_config,
    resolve_device,
)


CONFIG_PATH = Path(__file__).parents[2] / "configs/translator/envit5.json"


class EchoTranslator(Translator):
    def translate(self, text: str) -> TranslationResult:
        return TranslationResult(f"vi:{text}", text, len(text), len(text) + 1, {"frozen": True})


class FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.device: str | None = None

    def to(self, device: str) -> FakeTensor:
        self.device = device
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, decoded_outputs: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.decoded_outputs = decoded_outputs

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        values = list(texts)
        self.calls.append((values, kwargs))
        ids = [[sum(map(ord, token)) % 97 + 1 for token in text.split()] + [99] for text in values]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(ids)}
        return {"input_ids": ids}

    def batch_decode(self, generated: list[list[int]], **kwargs: object) -> list[str]:
        if self.decoded_outputs is not None:
            return self.decoded_outputs[: len(generated)]
        return [f"bản dịch {row[0]}" for row in generated]


class FakeModel:
    def __init__(self) -> None:
        self.generation_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generation_calls.append(kwargs)
        inputs = kwargs["input_ids"]
        assert isinstance(inputs, FakeTensor)
        return [[sum(row) % 97 + 1, 7, 0] for row in inputs.values]


class FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()


class RuntimeFactory:
    def __init__(self, decoded_outputs: list[str] | None = None) -> None:
        self.tokenizer = FakeTokenizer(decoded_outputs)
        self.model = FakeModel()
        self.calls = 0

    def __call__(self, config, device: str) -> _Runtime:
        self.calls += 1
        return _Runtime(
            self.tokenizer,
            self.model,
            FakeTorch(),
            device,
            "float32",
            "test-revision",
            123,
        )


def fake_translator(
    *, max_input_tokens: int = 512, decoded_outputs: list[str] | None = None,
) -> tuple[EnViT5Translator, RuntimeFactory]:
    config = load_config(CONFIG_PATH)
    config = replace(
        config,
        maximum_input_handling={
            "strategy": "error",
            "max_input_tokens": max_input_tokens,
            "includes_model_prefix": True,
        },
    )
    factory = RuntimeFactory(decoded_outputs)
    return EnViT5Translator(config, _runtime_loader=factory, _cuda_available=False), factory


class TranslatorContractTests(unittest.TestCase):
    def test_contract_and_batch_order(self) -> None:
        translator = EchoTranslator()
        results = translator.translate_batch(["first", "second"])
        self.assertEqual([result.source_text for result in results], ["first", "second"])
        self.assertEqual(results[0].translated_text, "vi:first")
        self.assertEqual(results[0].metadata, {"frozen": True})

    def test_contract_is_source_only(self) -> None:
        parameters = inspect.signature(Translator.translate).parameters
        self.assertEqual(list(parameters), ["self", "text"])
        with self.assertRaises(TypeError):
            EchoTranslator().translate("source", target_reference="gold")  # type: ignore[call-arg]


class EnViT5Tests(unittest.TestCase):
    def test_loading_is_lazy_and_result_is_deterministic(self) -> None:
        translator, factory = fake_translator()
        self.assertFalse(translator.is_loaded)
        first = translator.translate("Artificial intelligence")
        second = translator.translate("Artificial intelligence")
        self.assertEqual(factory.calls, 1)
        self.assertEqual(first, second)
        self.assertEqual(first.source_text, "Artificial intelligence")
        self.assertEqual(first.source_token_count, 3)
        self.assertEqual(first.target_token_count, 4)
        self.assertEqual(first.metadata["decoding"], "greedy")
        self.assertEqual(
            {key: factory.model.generation_calls[0][key] for key in ("do_sample", "num_beams", "max_new_tokens")},
            {"do_sample": False, "num_beams": 1, "max_new_tokens": 256},
        )

    def test_empty_and_whitespace_input_are_rejected_before_loading(self) -> None:
        translator, factory = fake_translator()
        for text in ("", " \t\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "must not be empty"):
                translator.translate(text)
        self.assertEqual(factory.calls, 0)

    def test_batch_order_is_preserved(self) -> None:
        translator, _factory = fake_translator()
        results = translator.translate_batch(["one", "two words", "three words here"])
        self.assertEqual([result.source_text for result in results], ["one", "two words", "three words here"])
        self.assertEqual(len({result.translated_text for result in results}), 3)

    def test_batch_and_single_item_semantics_match(self) -> None:
        batched, _factory = fake_translator()
        single, _factory = fake_translator()
        texts = ["one", "two words"]
        self.assertEqual(batched.translate_batch(texts), [single.translate(text) for text in texts])

    def test_exact_target_control_prefix_normalization(self) -> None:
        raw_and_expected = [
            ("vi: Xin chào", "Xin chào"),
            ("vi:Xin chào", "Xin chào"),
            ("Xin chào", "Xin chào"),
            ("vi: Một chuỗi có vi: ở giữa", "Một chuỗi có vi: ở giữa"),
        ]
        translator, _factory = fake_translator(
            decoded_outputs=[raw for raw, _expected in raw_and_expected],
        )
        sources = [f"source {index}" for index in range(len(raw_and_expected))]
        results = translator.translate_batch(sources)
        self.assertEqual([result.translated_text for result in results], [expected for _raw, expected in raw_and_expected])
        self.assertEqual([result.source_text for result in results], sources)

        for raw, expected in raw_and_expected:
            with self.subTest(raw=raw):
                single, _factory = fake_translator(decoded_outputs=[raw])
                first = single.translate("unchanged source")
                second = single.translate("unchanged source")
                self.assertEqual(first, second)
                self.assertEqual(first.translated_text, expected)
                self.assertEqual(first.source_text, "unchanged source")

    def test_target_token_count_describes_normalized_hypothesis(self) -> None:
        translator, factory = fake_translator(decoded_outputs=["vi: Xin chào"])
        result = translator.translate("Hello")
        self.assertEqual(result.translated_text, "Xin chào")
        self.assertEqual(result.target_token_count, 3)
        self.assertEqual(factory.tokenizer.calls[3][0], [result.translated_text])

    def test_duplicate_batch_inputs_generate_once(self) -> None:
        translator, factory = fake_translator()
        results = translator.translate_batch(["same", "other", "same"])
        self.assertEqual([result.source_text for result in results], ["same", "other", "same"])
        self.assertEqual(len(factory.model.generation_calls), 1)
        inputs = factory.model.generation_calls[0]["input_ids"]
        self.assertIsInstance(inputs, FakeTensor)
        self.assertEqual(len(inputs.values), 2)

    def test_model_prefix_is_internal_and_source_is_not_mutated(self) -> None:
        translator, factory = fake_translator()
        source = "  Large language models can"
        result = translator.translate(source)
        self.assertEqual(result.source_text, source)
        self.assertEqual(factory.tokenizer.calls[0][0], [MODEL_PREFIX + source])
        self.assertEqual(factory.tokenizer.calls[1][0], [source])

    def test_no_automatic_punctuation_is_inserted(self) -> None:
        translator, factory = fake_translator()
        translator.translate("Artificial intelligence is changing the way")
        model_text = factory.tokenizer.calls[0][0][0]
        self.assertEqual(model_text, "en: Artificial intelligence is changing the way")
        self.assertFalse(model_text.endswith("."))

    def test_overlength_input_is_not_truncated_or_generated(self) -> None:
        translator, factory = fake_translator(max_input_tokens=4)
        with self.assertRaisesRegex(InputTooLongError, "input was not truncated"):
            translator.translate("one two three")
        self.assertFalse(factory.tokenizer.calls[0][1]["truncation"])
        self.assertEqual(factory.model.generation_calls, [])

    def test_cpu_and_auto_device_resolution(self) -> None:
        self.assertEqual(resolve_device("cpu", cuda_available=True), "cpu")
        self.assertEqual(resolve_device("auto", cuda_available=False), "cpu")
        self.assertEqual(resolve_device("auto", cuda_available=True), "cuda")
        with self.assertRaisesRegex(RuntimeError, "not available"):
            resolve_device("cuda", cuda_available=False)

    def test_config_loading(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config.config_version, "1.1.0")
        self.assertEqual(config.model_id, "VietAI/envit5-translation")
        self.assertEqual(config.model_revision, "840bc88104d5a4277af740eaedb024df8c3093e7")
        self.assertTrue(config.frozen)
        self.assertEqual(config.max_input_tokens, 512)

    def test_invalid_config_is_rejected(self) -> None:
        document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        invalid_documents = [
            {**document, "model_id": "other/model"},
            {**document, "frozen": False},
            {**document, "generation_parameters": {"do_sample": True, "num_beams": 1, "max_new_tokens": 256}},
            {**document, "unknown": "field"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for invalid in invalid_documents:
                with self.subTest(invalid=invalid):
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_config(path)


class TranslatorCacheTests(unittest.TestCase):
    def test_cache_miss_then_hit_avoids_generation_and_survives_new_translator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TranslationCache(directory)
            translator, factory = fake_translator()
            translator.cache = cache
            first = translator.translate("cached source")
            self.assertFalse(first.metadata["cache_hit"])
            self.assertEqual(len(factory.model.generation_calls), 1)

            second_translator, second_factory = fake_translator()
            second_translator.cache = TranslationCache(directory)
            second = second_translator.translate("cached source")
            self.assertTrue(second.metadata["cache_hit"])
            self.assertEqual(second_factory.calls, 0)
            self.assertEqual(first.translated_text, second.translated_text)

    def test_exact_source_and_inference_identity_produce_distinct_keys(self) -> None:
        shared = dict(
            model_id="model", model_revision="revision", generation_parameters={"a": 1},
            config_version="1", device="cpu", dtype="float32",
        )
        keys = {TranslationCache.key(source_text=text, **shared) for text in ("hello", "hello ", "Hello", "hello.")}
        self.assertEqual(len(keys), 4)
        self.assertNotEqual(
            TranslationCache.key(source_text="hello", **shared),
            TranslationCache.key(source_text="hello", **{**shared, "model_revision": "other"}),
        )
        self.assertNotEqual(
            TranslationCache.key(source_text="hello", **shared),
            TranslationCache.key(source_text="hello", **{**shared, "generation_parameters": {"a": 2}}),
        )
        self.assertNotEqual(
            TranslationCache.key(source_text="hello", **shared),
            TranslationCache.key(source_text="hello", **{**shared, "config_version": "2"}),
        )

    def test_old_semantic_version_cache_entry_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            translator, factory = fake_translator(decoded_outputs=["vi: Xin chào"])
            translator.cache = TranslationCache(directory)
            old_key = TranslationCache.key(
                source_text="Hello",
                model_id=translator.config.model_id,
                model_revision=translator.config.model_revision,
                generation_parameters=translator.config.generation_parameters,
                config_version="1.0.0",
                device=translator.device,
                dtype=translator.config.dtype_policy[translator.device],
            )
            translator.cache.put(old_key, TranslationResult("vi: Xin chào", "Hello", target_token_count=4))

            result = translator.translate("Hello")
            self.assertEqual(result.translated_text, "Xin chào")
            self.assertFalse(result.metadata["cache_hit"])
            self.assertEqual(len(factory.model.generation_calls), 1)

    def test_malformed_cache_entry_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TranslationCache(directory)
            key = "broken"
            (Path(directory) / f"{key}.json").write_text("not json", encoding="utf-8")
            self.assertIsNone(cache.get(key))

    def test_stale_cache_format_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TranslationCache(directory)
            key = "stale-format"
            document = {
                "format_version": 0,
                "key": key,
                "result": {"translated_text": "vi", "source_text": "source"},
            }
            (Path(directory) / f"{key}.json").write_text(json.dumps(document), encoding="utf-8")
            self.assertIsNone(cache.get(key))

    def test_cache_entry_with_wrong_source_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TranslationCache(directory)
            key = "wrong-source"
            cache.put(key, TranslationResult("vi", "other source"))
            self.assertIsNone(cache.get(key, source_text="expected source"))


class PrefixInferenceTests(unittest.TestCase):
    def test_prefix_order_and_micro_batches_are_preserved(self) -> None:
        translator, factory = fake_translator()
        prefixes = ["Large", "Large language", "Large language models", "Large language models can"]
        results = translate_prefixes(translator, prefixes, batch_size=2)
        self.assertEqual([result.prefix_index for result in results], [0, 1, 2, 3])
        self.assertEqual([result.source_text for result in results], prefixes)
        self.assertEqual(len(factory.model.generation_calls), 2)

    def test_invalid_prefixes_are_rejected_without_source_mutation(self) -> None:
        translator, _factory = fake_translator(max_input_tokens=4)
        prefixes = ["one two three"]
        with self.assertRaises(InputTooLongError):
            translate_prefixes(translator, prefixes)
        self.assertEqual(prefixes, ["one two three"])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            translate_prefixes(translator, [" "])
        with self.assertRaisesRegex(ValueError, "batch_size"):
            translate_prefixes(translator, ["source"], batch_size=0)


if __name__ == "__main__":
    unittest.main()
