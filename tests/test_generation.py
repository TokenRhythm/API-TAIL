"""CPU-only sampling and input-contract checks; model loading is mocked."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from apitail.generate import sample_logits, generate
from apitail.hf import score

TORCH = importlib.util.find_spec("torch") is not None
TRANSFORMERS = importlib.util.find_spec("transformers") is not None
if TORCH:
    import torch
    torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH, "Optional torch extra is not installed")
class SamplingTests(unittest.TestCase):
    def test_penalties_use_prompt_union_and_presence_only_once(self):
        ids = torch.tensor([[0, 1, 1, 2, 2]])
        logits = torch.tensor([[4., -2., 3., -1., 0.]])
        actual = sample_logits(ids, logits, [0, 1], 2, {
            "repetition_penalty": 2., "presence_penalty": .5, "temperature": .5})
        # Prompt tokens 0/1 and generated tokens 1/2 get repetition penalties.
        # Repeated output token 2 receives one presence penalty, not a count penalty.
        torch.testing.assert_close(actual, torch.tensor([[4., -9., 2., -2., 0.]]))
        torch.testing.assert_close(logits, torch.tensor([[4., -2., 3., -1., 0.]]))

    def test_top_p_keeps_the_token_crossing_probability_boundary(self):
        logits = torch.log(torch.tensor([[.6, .25, .1, .05]], dtype=torch.float64))
        actual = sample_logits(torch.tensor([[3]]), logits, [3], 1, {"top_p": .7})
        self.assertEqual(torch.isfinite(actual).tolist(), [[True, True, False, False]])
        torch.testing.assert_close(actual[0, :2], logits.float()[0, :2])

    def test_top_k_ties_and_min_p_cutoffs(self):
        ids = torch.tensor([[0]])
        tied = sample_logits(ids, torch.tensor([[3., 2., 2., 1.]]), [0], 1, {"top_k": 2})
        self.assertEqual(torch.isfinite(tied).tolist(), [[True, True, True, False]])
        probabilities = torch.tensor([[.6, .25, .1, .05]])
        filtered = sample_logits(ids, probabilities.log(), [0], 1, {"min_p": .2})
        self.assertEqual(torch.isfinite(filtered).tolist(), [[True, True, False, False]])

    def test_sampling_matches_archived_sampler(self):
        source = ROOT / "legacy/code/jobs/paper2/gen_hf.py"
        if not source.is_file():
            self.skipTest("Source checkout legacy sampler is not available")
        text = source.read_text(encoding="utf-8")
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "Sampler")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(1, 31, generator=generator)
        prompt = [1, 3, 5]
        ids = torch.tensor([prompt + [2, 5, 5, 9]])
        prompt_mask = torch.zeros_like(logits, dtype=torch.bool)
        prompt_mask[:, prompt] = True
        for temperature in (0., .7, 1.4):
            for top_p in (1., .8):
                with self.subTest(temperature=temperature, top_p=top_p):
                    settings = {"repetition_penalty": 1.1, "presence_penalty": .4,
                                "temperature": temperature, "top_k": 9, "top_p": top_p, "min_p": .05}
                    expected = namespace["Sampler"](settings, prompt_mask, len(prompt))(ids, logits)
                    actual = sample_logits(ids, logits, prompt, len(prompt), settings)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ScoreInputTests(unittest.TestCase):
    def test_invalid_response_indices_fail_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts, responses = root / "p.jsonl", root / "r.jsonl"
            prompts.write_text('{"pid":"p0","text":"Write one word."}\n', encoding="utf-8")
            for value in (-1, .5, True, "1"):
                with self.subTest(k=value):
                    responses.write_text(json.dumps({"pid": "p0", "k": value, "text": "Word."}) + '\n', encoding="utf-8")
                    with patch("apitail.hf.load_model") as load:
                        with self.assertRaises(ValueError):
                            score("mock-model", prompts, responses, root / "scores.npz", device="cpu")
                        load.assert_not_called()


@unittest.skipUnless(TORCH and TRANSFORMERS, "Optional torch/transformers extras are not installed")
class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompts, self.config = self.root / "p.jsonl", self.root / "config.json"
        self.output = self.root / "r.jsonl"
        self.prompts.write_text('{"pid":"p0","text":"Write one word."}\n', encoding="utf-8")
        self.base = {"seed": 7, "responses_per_prompt": 1, "max_length": 64,
                     "sampling": {"temperature": .7, "max_tokens": 8}}
        self.write_config()

    def write_config(self, **updates):
        self.config.write_text(json.dumps(dict(self.base, **updates)), encoding="utf-8")

    def test_greedy_and_stochastic_generation_payloads(self):
        for temperature in (0., .7):
            with self.subTest(temperature=temperature):
                self.write_config(sampling={"temperature": temperature, "max_tokens": 8,
                                            "top_p": .8, "repetition_penalty": 1.1})
                calls = []
                class Model:
                    generation_config = SimpleNamespace(eos_token_id=4)
                    def generate(self, **kwargs):
                        calls.append(kwargs)
                        return torch.tensor([[0, 1, 2, 4]])
                tokenizer = SimpleNamespace(pad_token_id=None,
                                            decode=lambda ids, skip_special_tokens: "Word.")
                with patch("apitail.generate.load_model", return_value=(torch, tokenizer, Model())) as load, patch("apitail.generate.chat_ids", return_value=[0, 1]):
                    result = generate("mock-model", self.prompts, self.config, self.output,
                                      device="cpu", local_files_only=True)
                    load.assert_called_once_with("mock-model", "cpu", "bfloat16", True)
                cfg = calls[0]["generation_config"]
                self.assertEqual(cfg.do_sample, temperature > 0)
                self.assertEqual(cfg.max_new_tokens, 8)
                # Built-in warpers stay neutral; custom processor applies settings once.
                self.assertEqual((cfg.temperature, cfg.top_k, cfg.top_p), (1., 0, 1.))
                self.assertEqual(len(calls[0]["logits_processor"]), 1)
                self.assertEqual(result["responses"], 1)
                rows = [json.loads(x) for x in self.output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(rows[-1], {"pid": "p0", "k": 0, "text": "Word.", "ids": [2],
                                           "n_tokens": 1, "finish_reason": "stop"})
                self.output.unlink()

    def test_malformed_numeric_config_fails_before_model_load(self):
        changes = [{"responses_per_prompt": 1.5}, {"responses_per_prompt": -1},
                   {"responses_per_prompt": True}, {"seed": 1.5}, {"max_length": -1},
                   {"max_length": 4.5},
                   {"sampling": {"max_tokens": 2.5}}, {"sampling": {"max_tokens": 0}},
                   {"sampling": {"top_k": 1.5}}, {"sampling": {"top_k": -1}},
                   {"sampling": {"temperature": float("nan")}},
                   {"sampling": {"temperature": float("inf")}},
                   {"sampling": {"presence_penalty": float("inf")}},
                   {"sampling": {"repetition_penalty": float("nan")}}]
        for update in changes:
            with self.subTest(update=update):
                self.write_config(**update)
                with patch("apitail.generate.load_model") as load:
                    with self.assertRaises(ValueError):
                        generate("mock-model", self.prompts, self.config, self.output, device="cpu")
                    load.assert_not_called()

    def test_overwrite_and_partial_protected_before_model_load(self):
        for suffix in (".jsonl", ".jsonl.partial"):
            with self.subTest(suffix=suffix):
                retained = self.output.with_suffix(suffix)
                retained.write_text("retained-data", encoding="utf-8")
                with patch("apitail.generate.load_model") as load:
                    with self.assertRaises(FileExistsError):
                        generate("mock-model", self.prompts, self.config, self.output, device="cpu")
                    load.assert_not_called()
                self.assertEqual(retained.read_text(encoding="utf-8"), "retained-data")
                retained.unlink()


if __name__ == "__main__":
    unittest.main()
