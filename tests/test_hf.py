"""CPU moment tests; torch is optional."""
import importlib.util
import unittest
import numpy as np
from apitail.core import moments_from_logits

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import torch
    from apitail.hf import moments_torch


@unittest.skipUnless(TORCH_AVAILABLE, 'optional torch dependency is not installed')
class TorchMomentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_vocabulary_agrees_with_float64_probability_oracle(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(23, 41)).astype(np.float32)
        targets = rng.integers(0, 41, size=23)
        grid = np.array([0.3, 0.8, 1.0, 1.8])
        expected = moments_from_logits(logits, targets, grid)
        actual = moments_torch(torch.tensor(logits), torch.tensor(targets), grid, token_chunk=7)
        for name in ['S', 'V']:
            np.testing.assert_allclose(actual[name], expected[name], atol=5e-5, rtol=5e-6)

    def test_logit_shifts_do_not_change_scores(self):
        logits = torch.tensor([[0.25, -0.5, 1.0, 0.0], [4.0, 2.0, -2.0, -1.0]])
        target = torch.tensor([0, 2])
        grid = [0.5, 1.0, 1.5]
        a = moments_torch(logits, target, grid, nucleus=True)
        b = moments_torch(logits + torch.tensor([[64.0], [-64.0]]), target, grid, nucleus=True)
        for field in ['S', 'V', 'S_nucleus', 'V_nucleus', 'n_nucleus', 'nucleus_overflow']:
            np.testing.assert_allclose(a[field], b[field], atol=1e-7, rtol=1e-7)

    def test_token_chunks_preserve_statistics(self):
        rng = np.random.default_rng(91)
        logits = torch.tensor(rng.normal(size=(37, 53)).astype(np.float32))
        targets = torch.tensor(rng.integers(0, 53, 37))
        a = moments_torch(logits, targets, [0.3, 1.0, 1.8], nucleus=True, token_chunk=64)
        for chunk in [1, 7, 13]:
            b = moments_torch(logits, targets, [0.3, 1.0, 1.8], nucleus=True, token_chunk=chunk)
            for name in a:
                np.testing.assert_allclose(a[name], b[name], atol=1e-6, rtol=1e-7, err_msg=f'{chunk=} {name=}')

    def test_small_vocabulary_nucleus_matches_direct_distribution(self):
        logits = np.array([[5.0, 0.0, -1.0, -2.0], [0.2, -0.7, 0.5, 1.1]], dtype=np.float32)
        targets = np.array([3, 1])
        grid = [0.5, 1.0, 1.5]
        result = moments_torch(torch.tensor(logits), torch.tensor(targets), grid, nucleus=True, token_chunk=1)
        expected_s, expected_v, expected_n = [], [], []
        for temperature in grid:
            s, v, n = 0.0, 0.0, 0
            for row, target in zip(logits.astype(float), targets):
                p = np.exp((row - row.max()) / temperature)
                p /= p.sum()
                order = np.argsort(-p)
                sorted_p = p[order]
                retained = order[np.cumsum(sorted_p) - sorted_p < 0.95]
                if target not in retained:
                    continue
                q = p[retained] / p[retained].sum()
                surprise = -np.log(q)
                entropy = np.dot(q, surprise)
                variance = np.dot(q, (surprise - entropy) ** 2)
                s += -np.log(p[target] / p[retained].sum()) - entropy
                v += max(variance, 1e-6)
                n += 1
            expected_s.append(s)
            expected_v.append(v)
            expected_n.append(n)
        np.testing.assert_allclose(result['S_nucleus'], expected_s, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(result['V_nucleus'], expected_v, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(result['n_nucleus'], expected_n)
        np.testing.assert_array_equal(result['nucleus_overflow'], [0, 0, 0])

    def test_single_token_vocabulary_has_zero_excess_and_variance_floor(self):
        result = moments_torch(torch.zeros((4, 1)), torch.zeros(4, dtype=torch.int64), [0.5, 1.0], nucleus=True)
        np.testing.assert_allclose(result['S'], [0, 0])
        np.testing.assert_allclose(result['V'], [4e-6, 4e-6])
        np.testing.assert_allclose(result['S_nucleus'], [0, 0])
        np.testing.assert_allclose(result['V_nucleus'], [4e-6, 4e-6])
        np.testing.assert_array_equal(result['n_nucleus'], [4, 4])

    def test_int32_targets_match_int64_without_mutating_inputs(self):
        logits = torch.tensor([[.1, -.3, 1.2], [-2., .3, .6]], dtype=torch.float32)
        targets = torch.tensor([2, 1], dtype=torch.int32)
        grid = np.asarray([.3, .7, 1., 1.8])
        expected = moments_torch(logits, targets.to(torch.int64), grid, nucleus=True, token_chunk=1)
        actual = moments_torch(logits, targets, grid, nucleus=True, token_chunk=1)
        self.assertEqual(targets.dtype, torch.int32)
        self.assertEqual(targets.tolist(), [2, 1])
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(field=name):
                np.testing.assert_array_equal(actual[name], expected[name])

    def test_rejects_invalid_numeric_inputs(self):
        logits, target = torch.zeros((2, 3)), torch.tensor([0, 1])
        for bad_target in [torch.tensor([0.0, 1.0]), torch.tensor([-1, 0]), torch.tensor([0, 3])]:
            with self.subTest(target=bad_target), self.assertRaises(ValueError):
                moments_torch(logits, bad_target, [1.0])
        for grid in [[], [0], [1, 1], [[1.0]], [np.nan]]:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                moments_torch(logits, target, grid)
        for chunk in [0, -1, 1.5, True]:
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                moments_torch(logits, target, [1.0], token_chunk=chunk)
        with self.assertRaises(ValueError):
            moments_torch(torch.tensor([[float('nan'), 0.0]]), torch.tensor([0]), [1.0])


if __name__ == '__main__':
    unittest.main()
