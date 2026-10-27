import importlib.util
from pathlib import Path
import unittest
import numpy as np
from apitail import baselines as b


class BaselineTests(unittest.TestCase):
    def test_midrank_inside_and_outside_window(self):
        lp = np.array([-1., -2., -4.])
        ranks = np.array([1, 3, 7])
        top = np.array([[-1., -1., -3.], [-1., -2., -2.], [-1., -2., -3.]], dtype=np.float16)
        np.testing.assert_array_equal(b.token_midranks(lp, ranks, top), [1.5, 2.5, 7])
        np.testing.assert_array_equal(b.token_midranks(lp, ranks), ranks)

    def test_detector_entropy_is_window_conditional(self):
        lp = np.log([0.1, 0.2])
        top = np.log(np.array([[0.1, 0.1], [0.2, 0.2]]))
        scores = b.response_detectors(lp, [1, 1], top)
        self.assertAlmostEqual(scores[0], -lp.mean())
        self.assertAlmostEqual(scores[1], np.log(1.5))
        self.assertAlmostEqual(scores[2], np.log(2))

    def test_detector_quantile_and_strict_count(self):
        self.assertAlmostEqual(b.detector_threshold([0, 1, 2, 3, np.nan], 0.5), 1.5)
        np.testing.assert_array_equal(b.detector_counts([[1, 2, 3], [np.nan, 2, 4]], 2), [1, 1])
        self.assertEqual(b.detector_counts([2, 3], 2), 1)

    def test_continuous_composite_cutoff_and_strict_rates(self):
        result = b.calibrate_continuous({
            "a": [0, 1, 2, 3, 4],
            "b": [0, 4, 4, 4, 5],
            "c": [-1, 0, 4, 4],
        }, alpha=0.4)
        self.assertEqual(result["threshold"], 4.)
        self.assertEqual(result["quantiles"], {"a": 3., "b": 4., "c": 4.})
        self.assertEqual(result["decisive"], ["b", "c"])
        self.assertEqual(result["trials"], {"a": 5, "b": 5, "c": 4})
        self.assertEqual(result["rates"], {"a": 0., "b": 0.2, "c": 0.})
        self.assertEqual(result["rule"], "statistic > threshold")

    def test_continuous_higher_quantile_and_negative_met(self):
        # Linear interpolation yields 26.5; this cutoff uses the higher quantile.
        result = b.calibrate_continuous({"a": [0, 1, 2, 100]}, alpha=0.25)
        self.assertEqual(result["threshold"], 100.)
        self.assertEqual(result["rates"]["a"], 0.)
        result = b.calibrate_continuous({"met": [-0.4, -0.2, -0.1, -0.1]})
        self.assertEqual(result["threshold"], -0.1)
        self.assertEqual(result["rates"]["met"], 0.)
        self.assertEqual(result["alpha"], 0.05)

    def test_continuous_calibration_rejects_invalid_inputs(self):
        for mapping in ({}, [], {"a": []}, {"a": 1}, {"a": [[1, 2]]},
                        {"a": [1, np.nan]}, {"a": [1, np.inf]}, {"a": [-np.inf]}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                b.calibrate_continuous(mapping)
        for alpha in (0, 1, -0.1, 1.1, np.nan, np.inf, None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                b.calibrate_continuous({"a": [0, 1]}, alpha=alpha)

    def test_rut_ties_and_explicit_uniforms(self):
        r = b.randomized_ranks([0, 1], [[0, 0], [0, 1]], [0.5, 0.25])
        np.testing.assert_array_equal(r, [0.5, 0.5])
        result = b.rut_statistic([0, 1], [[0, 0], [0, 1]], uniforms=[0.5, 0.25], reference_count=2)
        self.assertAlmostEqual(result['w2'], 1 / 6)
        self.assertEqual(result['api_requests'], 2)
        self.assertEqual(result['local_references'], 4)
        self.assertAlmostEqual(result['statistic'], -np.log(result['pvalue']))

    def test_rut_public_api_matches_frozen_formula(self):
        from scipy.stats import _hypotests
        r = np.array([0.01, 0.1, 0.25, 0.4, 0.51, 0.6, 0.68, 0.8, 0.9, 0.99])
        w, p = b.cramer_von_mises_uniform(r)
        n = len(r)
        manual = 1 / (12 * n) + np.sum(((2 * np.arange(1, n + 1) - 1) / (2 * n) - r) ** 2)
        self.assertAlmostEqual(w, manual)
        self.assertAlmostEqual(p, max(0, 1 - _hypotests._cdf_cvm(manual, n)))

    def test_rut_first_100_references_and_seed(self):
        rng = np.random.default_rng(45)
        target = rng.normal(size=20)
        refs = rng.normal(size=(20, 105))
        a = b.rut_statistic(target, refs, seed=19)
        refs[:, 100:] = 10000
        c = b.rut_statistic(target, refs, seed=19)
        np.testing.assert_array_equal(a['ranks'], c['ranks'])
        self.assertEqual(a['pvalue'], c['pvalue'])
        with self.assertRaises(ValueError): b.rut_statistic(target, refs[:, :99])

    def test_met_analytic_and_unbiased_negative(self):
        x = np.array([[[1, 2], [1, 3]]])
        y = np.array([[[4, 5], [4, 6]]])
        self.assertEqual(b.paired_hamming_mmd2(x, y), 1.0)
        z = np.array([[[1], [2]]])
        self.assertEqual(b.paired_hamming_mmd2(z, z), -1.0)

    def test_met_matches_direct_paired_kernel(self):
        rng = np.random.default_rng(7)
        x = rng.integers(-1, 8, size=(19, 2, 30))
        y = rng.integers(-1, 8, size=(19, 2, 30))
        expected = np.mean(x[:, 0] == x[:, 1]) + np.mean(y[:, 0] == y[:, 1])
        expected -= 2 * np.mean([np.mean(x[:, i] == y[:, j], axis=-1) for i in (0, 1) for j in (0, 1)])
        self.assertAlmostEqual(b.paired_hamming_mmd2(x, y), expected)

    def test_padding_and_invalid_samples(self):
        np.testing.assert_array_equal(b.pad_token_ids([[4], [3, 2, 1]], length=2), [[4, -1], [3, 2]])
        with self.assertRaises(ValueError): b.paired_hamming_mmd2(np.zeros((2, 1, 3), int), np.zeros((2, 1, 3), int))
        with self.assertRaises(ValueError): b.token_midranks([], [])
        with self.assertRaises(ValueError): b.randomized_ranks([1], [[1]], [1.0])
        with self.assertRaises(ValueError): b.response_detectors([-1], [1], [[np.nan]])

    def test_archived_midrank_and_native_met(self):
        root = Path(__file__).resolve().parents[1] / 'legacy' / 'code' / 'bbp'
        if not (root / 'rut.py').exists():
            self.skipTest('source archive is not part of an installed wheel')
        import sys
        sys.path.insert(0, str(root))
        try:
            def load(name, filename):
                spec = importlib.util.spec_from_file_location(name, root / filename)
                mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                return mod
            old_rut = load('native_rut', 'rut.py')
            old_met = load('native_met', 'cmp_met_native.py')
            rng = np.random.default_rng(77)
            lp = rng.normal(-4, 1, 12).astype(np.float32)
            top = np.sort(rng.normal(-3, 1, (12, 20)).astype(np.float16), axis=1)[:, ::-1]
            top[:, 0] = lp.astype(np.float16)
            rank = np.arange(1, 13)
            response = {'lp': lp, 'rk': rank, 'tlp': top}
            np.testing.assert_array_equal(b.token_midranks(lp, rank, top), old_rut.midrank(response))
            x = [rng.integers(-1, 7, size=(m, 50)) for m in [2, 4, 10]]
            y = [rng.integers(-1, 7, size=(m, 50)) for m in [3, 5, 10]]
            self.assertEqual(b.hamming_mmd2(x, y), old_met.mmd(x, y))
        finally:
            sys.path.remove(str(root))


if __name__ == '__main__':
    unittest.main()
