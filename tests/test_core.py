"""Independent numerical and decision-contract tests; no models or API calls."""
import unittest
import numpy as np

from apitail.core import (
    temperature_grid, moments_from_logits, batch_statistic, batch_counts,
    calibrate_counts, rejection_rate,
)


class MomentTests(unittest.TestCase):
    def test_default_grid(self):
        a = temperature_grid()
        self.assertEqual(a.shape, (151,))
        self.assertEqual((a[0], a[70], a[-1]), (0.3, 1.0, 1.8))
        a[0] = -1
        self.assertEqual(temperature_grid()[0], 0.3)

    def test_matches_direct_probability_oracle(self):
        logits = np.array([[0.2, -0.7, 1.2, 0.5], [3.0, 1.0, 0.0, -1.0], [-2.0, 0.0, 0.2, 0.4]])
        target = np.array([2, 1, 0])
        grid = np.array([0.5, 1.0, 1.5])
        actual = moments_from_logits(logits, target, grid)
        expected_s, expected_v = [], []
        for temperature in grid:
            probability = np.exp(logits / temperature)
            probability /= probability.sum(axis=1)[:, None]
            surprisal = -np.log(probability)
            mean = (probability * surprisal).sum(axis=1)
            var = (probability * (surprisal - mean[:, None]) ** 2).sum(axis=1)
            expected_s.append((surprisal[np.arange(3), target] - mean).sum())
            expected_v.append(np.maximum(var, 1e-6).sum())
        np.testing.assert_allclose(actual['S'], expected_s, atol=2e-14, rtol=1e-13)
        np.testing.assert_allclose(actual['V'], expected_v, atol=2e-14, rtol=1e-13)

    def test_per_position_logit_offset_invariance(self):
        logits = np.array([[0.2, -0.4, 0.7], [4.0, 2.0, -2.0]])
        original = moments_from_logits(logits, [0, 2], [0.3, 0.7, 1.8])
        shifted = moments_from_logits(logits + np.array([[1000.0], [-1000.0]]), [0, 2], [0.3, 0.7, 1.8])
        for field in ['S', 'V']:
            np.testing.assert_allclose(original[field], shifted[field], atol=1e-10, rtol=1e-10)

    def test_uniform_distribution_and_variance_floor(self):
        moments = moments_from_logits(np.zeros((5, 4)), [0, 1, 2, 3, 0], [0.5, 1.0])
        np.testing.assert_allclose(moments['S'], [0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(moments['V'], [5e-6, 5e-6], atol=1e-15)

    def test_large_logits_remain_finite(self):
        moments = moments_from_logits([[10000.0, 0.0, -10000.0]], [1], [0.3, 1.8])
        self.assertTrue(np.isfinite(moments['S']).all())
        self.assertTrue(np.isfinite(moments['V']).all())

    def test_invalid_logit_inputs(self):
        cases = [([], [], [1]), ([[0, 1]], [1.0], [1]), ([[0, 1]], [-1], [1]),
                 ([[0, 1]], [2], [1]), ([[0, float('nan')]], [0], [1]),
                 ([[0, 1]], [0], [1, 1]), ([[0, 1]], [0], [0]),
                 ([[0, 1]], [0], [2, 1])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                moments_from_logits(*args)
        with self.assertRaises(ValueError):
            moments_from_logits([[0, 1]], [0], variance_floor=-1)


class AuditTests(unittest.TestCase):
    def test_estimators_have_different_objectives(self):
        grid = np.array([0.5, 1.0, 1.5])
        z = np.array([[4.0, 2.0, 10.0], [-1.0, 1.0, -0.6725], [-3.0, -4 / 3, -0.6725]])
        scales = np.array([[1.0], [2.0], [3.0]])
        S, V = z * scales, np.broadcast_to(scales ** 2, z.shape)
        expected = {'align': (0.5, 1), 'mle': (1.0, 0), 'fixed': (1.0, 0), 'huber': (1.5, 1)}
        for method, (temperature, count) in expected.items():
            with self.subTest(method=method):
                out = batch_statistic(S, V, grid, method=method)
                self.assertEqual(out['temperature'], temperature)
                self.assertEqual(out['count'], count)
                self.assertEqual(out['n'], 3)

    def test_strict_exceedance_and_first_temperature_tie(self):
        S = np.array([[-3.0], [3.0], [np.nextafter(3.0, np.inf)], [0.0]])
        out = batch_statistic(S, np.ones_like(S), [1.0], method='fixed')
        self.assertEqual(out['count'], 1)
        tied = batch_statistic(np.zeros((3, 2)), np.ones((3, 2)), [0.5, 1.0])
        self.assertEqual(tied['temperature_index'], 0)
        self.assertEqual(rejection_rate([2, 3, 4], 3), 1 / 3)

    def test_zero_variance_matches_reference_floor(self):
        result = batch_statistic([[0.0], [4e-6]], [[0.0], [0.0]], [1.0])
        np.testing.assert_allclose(result['z'], [0.0, 4.0])
        self.assertEqual(result['count'], 1)

    def test_draws_reorder_and_repeat_responses(self):
        S = np.array([[4.0, 2.0], [-1.0, 1.0], [-3.0, -4 / 3]])
        V = np.ones_like(S)
        draws = np.array([[0, 1, 2], [2, 0, 1], [0, 0, 0]])
        counts, fitted = batch_counts(S, V, draws, [0.5, 1.0])
        np.testing.assert_array_equal(counts[:2], [1, 1])
        np.testing.assert_array_equal(fitted[:2], [0.5, 0.5])
        self.assertEqual(fitted[2], 1.0)
        self.assertEqual(counts[2], 0)

    def test_invalid_audit_inputs(self):
        good_s, good_v = np.zeros((2, 2)), np.ones((2, 2))
        for grid in ([0.0, 1.0], [1.0, 1.0], [2.0, 1.0]):
            with self.assertRaises(ValueError):
                batch_statistic(good_s, good_v, grid)
        with self.assertRaises(ValueError):
            batch_statistic(good_s, -good_v, [0.5, 1.0])
        with self.assertRaises(ValueError):
            batch_statistic(good_s, good_v, [0.5, 1.5], method='fixed')
        with self.assertRaises(ValueError):
            batch_statistic(good_s, good_v, [0.5, 1.0], method='unknown')
        for bad in ([[0, 2]], [[-1, 0]], [[0.0, 1.0]], [[]], [0, 1]):
            with self.subTest(draws=bad), self.assertRaises(ValueError):
                batch_counts(good_s, good_v, bad, [0.5, 1.0])
        with self.assertRaises(ValueError):
            batch_statistic([[np.inf]], [[1]], [1])


class CalibrationTests(unittest.TestCase):
    def test_higher_quantile_and_worst_configuration(self):
        bank = {'easy': np.zeros(20), 'hard': np.arange(20)}
        out = calibrate_counts(bank)
        self.assertEqual(out['threshold'], 19.0)
        self.assertEqual(out['quantiles'], {'easy': 0.0, 'hard': 19.0})
        self.assertEqual(out['decisive'], 'hard')
        self.assertEqual(out['trials'], {'easy': 20, 'hard': 20})
        for values in bank.values():
            self.assertLessEqual(rejection_rate(values, out['threshold']), 0.05)

    def test_composite_randomized_rule_uses_most_restrictive_tie(self):
        bank = {'a': np.array([0] * 90 + [3] * 10),
                'b': np.array([0] * 80 + [3] * 20),
                'c': np.array([0] * 96 + [3] * 4)}
        out = calibrate_counts(bank, alpha=0.05)
        rule = out['exact']
        self.assertEqual(rule['threshold'], 3.0)
        self.assertEqual(rule['decisive'], 'b')
        self.assertAlmostEqual(rule['gamma'], 0.25)
        for values in bank.values():
            probability = np.mean(values > rule['threshold']) + rule['gamma'] * np.mean(values == rule['threshold'])
            self.assertLessEqual(probability, 0.05 + 1e-14)
        self.assertAlmostEqual(rule['gamma'] * np.mean(bank['b'] == 3), 0.05)

    def test_decisive_ties_preserve_configuration_order(self):
        out = calibrate_counts({'first': [0, 1], 'second': [0, 1]})
        self.assertEqual(out['decisive'], 'first')
        self.assertEqual(out['exact']['decisive'], 'first')

    def test_invalid_calibration_inputs(self):
        for bank in ({}, {'a': []}, {'a': [-1, 0]}, {'a': [1.5]}, {'a': [np.nan]}, {'': [0]}):
            with self.subTest(bank=bank), self.assertRaises(ValueError):
                calibrate_counts(bank)
        for alpha in [0, 1, -1, np.nan, [0.05]]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                calibrate_counts({'a': [0]}, alpha=alpha)
        with self.assertRaises(ValueError):
            rejection_rate([], 0)
        with self.assertRaises(ValueError):
            rejection_rate([0], np.nan)


if __name__ == '__main__':
    unittest.main()
