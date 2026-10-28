"""Context-boundary and next-token alignment checks without loading a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from apitail.generate import generate
from apitail.hf import score, moments_torch
from apitail.io import load_scores

HAS_BACKEND = all(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers'))


@unittest.skipUnless(HAS_BACKEND, 'Optional torch/transformers extras are not installed')
class ContextBoundaryTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(prefix='apitail-boundary-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = self.root / 'prompts.jsonl'
        self.config = self.root / 'config.json'
        self.responses = self.root / 'responses.jsonl'
        self.output = self.root / 'scores.npz'
        self.prompts.write_text('{"pid":"p","text":"Synthetic prompt."}\n', encoding='utf-8')
        self.config.write_text(json.dumps({'max_length': 4, 'sampling': {'max_tokens': 2, 'temperature': 0}}), encoding='utf-8')
        self.calls = []
        calls = self.calls
        class Model:
            generation_config = SimpleNamespace(eos_token_id=None)
            def generate(self, **kwargs):
                return torch.tensor([[0, 1, 2, 3]])
            def __call__(self, ids):
                calls.append(ids.clone())
                return SimpleNamespace(logits=torch.arange(ids.shape[1] * 4, dtype=torch.float32).reshape(1, ids.shape[1], 4))
        self.model = Model()
        self.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *a, **k: 'xy',
                                         encode=lambda text, **k: [2, 3] if text == 'xy' else [2, 3, 2])

    def test_generated_completion_at_exact_limit_scores_all_targets(self):
        with patch('apitail.generate.load_model', return_value=(self.torch, self.tokenizer, self.model)), patch('apitail.generate.chat_ids', return_value=[0, 1]):
            generate('mock-model', self.prompts, self.config, self.responses, device='cpu')
        observed = []
        def capture(logits, targets, *args, **kwargs):
            observed.append((logits.clone(), targets.clone()))
            return moments_torch(logits, targets, *args, **kwargs)
        with patch('apitail.hf.load_model', return_value=(self.torch, self.tokenizer, self.model)), patch('apitail.hf.chat_ids', return_value=[0, 1]), patch('apitail.hf.moments_torch', side_effect=capture):
            result = score('mock-model', self.prompts, self.responses, self.output, device='cpu', dtype='float32', max_length=4)
        self.assertEqual(result['tokens'], 2)
        self.assertEqual(self.calls[0].tolist(), [[0, 1, 2, 3]])
        self.assertEqual(observed[0][1].tolist(), [2, 3])
        self.assertEqual(observed[0][0].tolist(), [[4., 5., 6., 7.], [8., 9., 10., 11.]])
        bank = load_scores(self.output)
        self.assertEqual(bank.n_tokens.tolist(), [2])
        self.assertEqual(bank.metadata['max_length'], 4)

    def test_one_token_over_limit_is_rejected_before_forward(self):
        self.responses.write_text('{"pid":"p","k":0,"text":"xyz"}\n', encoding='utf-8')
        with patch('apitail.hf.load_model', return_value=(self.torch, self.tokenizer, self.model)), patch('apitail.hf.chat_ids', return_value=[0, 1]):
            with self.assertRaisesRegex(ValueError, 'exceeds max_length'):
                score('mock-model', self.prompts, self.responses, self.output, device='cpu', max_length=4)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
