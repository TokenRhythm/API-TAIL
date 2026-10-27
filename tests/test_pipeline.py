import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from apitail.io import ScoreBank, save_scores, load_scores, write_json, compatible
from apitail.pipeline import calibrate, audit, evaluate


def make_bank(prefix='p', values=None, k=None, n_tokens=None, extra=None, metadata=None):
    values = [0.0, 0.0, 0.0, 0.0] if values is None else values
    n = len(values)
    return ScoreBank(np.asarray(values)[:, None], np.ones((n, 1)), np.array([1.0]),
                     np.array([f'{prefix}{i}' for i in range(n)]),
                     np.zeros(n, dtype=np.int64) if k is None else k,
                     np.ones(n, dtype=np.int64) * 10 if n_tokens is None else n_tokens,
                     metadata or {'model_id': 'synthetic-model', 'scoring': 'fixture-v1',
                                  'system': '', 'chat_kwargs': {}, 'max_length': 128,
                                  'model_config_sha256': 'a' * 64, 'dtype': 'float32'},
                     {} if extra is None else extra)


class ScoreBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='apitail-test-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_safe_npz_roundtrip_preserves_all_fields(self):
        bank = make_bank(extra={'S_nucleus': np.ones((4, 1)),
                                'V_nucleus': np.ones((4, 1)) * 2,
                                'nll': np.arange(4, dtype=float)})
        dest = self.path / 'scores.npz'
        save_scores(dest, bank)
        restored = load_scores(dest)
        for field in ['S', 'V', 'grid', 'pid', 'k', 'n_tokens']:
            np.testing.assert_array_equal(getattr(restored, field), getattr(bank, field))
        self.assertEqual(restored.metadata, bank.metadata)
        for name in bank.extra:
            np.testing.assert_array_equal(restored.extra[name], bank.extra[name])
        with np.load(dest, allow_pickle=False) as payload:
            self.assertTrue(all(payload[name].dtype.kind != 'O' for name in payload.files))
        with self.assertRaises(FileExistsError):
            save_scores(dest, bank)

    def test_object_arrays_are_rejected_on_load(self):
        dest = self.path / 'unsafe.npz'
        np.savez(dest, S=np.zeros((1, 1)), V=np.ones((1, 1)), grid=np.array([1.0]),
                 pid=np.array(['p0']), k=np.array([0]), n_tokens=np.array([1]),
                 metadata=np.array('{}'), extra=np.array([{'not': 'numeric'}], dtype=object))
        with self.assertRaises(ValueError):
            load_scores(dest)

    def test_object_extra_is_rejected_before_save(self):
        with self.assertRaises(ValueError):
            make_bank(extra={'unsafe': np.array([{}, {}, {}, {}], dtype=object)})

    def test_fractional_or_negative_response_keys_are_rejected(self):
        for k in ([0.5, 1.5], [-1, 0]):
            with self.subTest(k=k), self.assertRaises(ValueError):
                make_bank(values=[0, 0], k=k)
        with self.assertRaises(ValueError):
            make_bank(values=[0, 0], n_tokens=[2.5, 3.5])

    def test_prompt_sampling_is_without_replacement_with_multiple_k(self):
        base = make_bank(values=[0, 0, 0, 0])
        bank = ScoreBank(np.repeat(base.S, 2, axis=0), np.repeat(base.V, 2, axis=0),
                         base.grid, np.repeat(base.pid, 2), np.tile([0, 1], 4),
                         np.repeat(base.n_tokens, 2), base.metadata, {})
        draws = bank.draws(3, 100, np.random.default_rng(17))
        for row in draws:
            self.assertEqual(len(set(bank.pid[row])), 3)
        self.assertEqual(set(bank.k[draws].reshape(-1).tolist()), {0, 1})
        np.testing.assert_array_equal(draws, bank.draws(3, 100, np.random.default_rng(17)))
        selected = bank.draws(2, 20, np.random.default_rng(17), ['p0', 'p2'])
        self.assertEqual(set(bank.pid[selected].reshape(-1).tolist()), {'p0', 'p2'})
        for budget in [0, 5, 1.5, True]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                bank.draws(budget, 1, np.random.default_rng(1))

    def test_score_semantics_reject_changed_model_or_dtype(self):
        original = make_bank()
        for field, value in [('model_config_sha256', 'b' * 64), ('dtype', 'bfloat16')]:
            changed = make_bank(metadata={**original.metadata, field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                compatible([original, changed])

    def test_json_overwrite_is_explicit(self):
        path = self.path / 'result.json'
        write_json(path, {'value': 1})
        with self.assertRaises(FileExistsError):
            write_json(path, {'value': 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {'value': 1})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='apitail-pipeline-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        save_scores(self.path / 'easy.npz', make_bank(prefix='cal'))
        save_scores(self.path / 'hard.npz', make_bank(prefix='cal', values=[4, 0, 0, 0]))
        self.cfg = {'members': {'easy': 'easy.npz', 'hard': 'hard.npz'},
                    'budget': 2, 'trials': 200, 'method': 'fixed', 'seed': 19}
        self.config_path = self.path / 'config.json'
        self.config_path.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.rule_path = self.path / 'rule.json'

    def test_worst_configuration_and_strict_audit_rule(self):
        rule = calibrate(self.config_path, self.rule_path)
        self.assertEqual(rule['threshold'], 1)
        self.assertEqual(rule['decisive'], 'hard')
        self.assertEqual(rule['quantiles'], {'easy': 0, 'hard': 1})
        self.assertEqual(rule['calibration_prompt_ids'], ['cal0', 'cal1', 'cal2', 'cal3'])
        for name, values, expected in [('at-threshold', [4, 0], False), ('above-threshold', [4, 4], True)]:
            source = self.path / (name + '.npz')
            save_scores(source, make_bank(prefix='test', values=values))
            result = audit(source, self.rule_path, self.path / (name + '.json'))
            self.assertIs(result['reject'], expected)
            self.assertEqual(result['threshold'], 1)
            self.assertEqual(len(result['calibration_sha256']), 64)

    def test_audit_rejects_overlap_wrong_budget_and_duplicate_prompts(self):
        calibrate(self.config_path, self.rule_path)
        for tag, bank in [('overlap', make_bank(prefix='cal', values=[0, 0])),
                          ('budget', make_bank(prefix='test', values=[0]))]:
            source = self.path / (tag + '.npz')
            save_scores(source, bank)
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                audit(source, self.rule_path, self.path / (tag + '.json'))
        original = make_bank(prefix='test', values=[0, 0])
        original.pid[:] = 'same-prompt'
        original.k[:] = [0, 1]
        source = self.path / 'multi-response.npz'
        save_scores(source, original)
        with self.assertRaises(ValueError):
            audit(source, self.rule_path, self.path / 'multi-response.json')

    def test_audit_rejects_scoring_identity_mismatch(self):
        calibrate(self.config_path, self.rule_path)
        bank = make_bank(prefix='test', values=[0, 0])
        bank.metadata['model_config_sha256'] = 'b' * 64
        source = self.path / 'other-model.npz'
        save_scores(source, bank)
        with self.assertRaises(ValueError):
            audit(source, self.rule_path, self.path / 'other-model.json')

    def test_partial_substitution_endpoints_and_honest_evaluation(self):
        calibrate(self.config_path, self.rule_path)
        base, sub = self.path / 'base.npz', self.path / 'sub.npz'
        save_scores(base, make_bank(prefix='test', values=[0, 0, 0, 0]))
        save_scores(sub, make_bank(prefix='test', values=[4, 4, 4, 4]))
        honest = evaluate(base, self.rule_path, self.path / 'honest.json', trials=30)
        zero = evaluate(base, self.rule_path, self.path / 'zero.json', trials=30, substitute=sub, rho=0)
        all_sub = evaluate(base, self.rule_path, self.path / 'all.json', trials=30, substitute=sub, rho=1)
        self.assertEqual(honest['rate'], 0)
        self.assertEqual(zero['rate'], 0)
        self.assertEqual(all_sub['rate'], 1)
        self.assertEqual(set(all_sub['counts']), {2})
        self.assertEqual(all_sub['kind'], 'simulated_partial_substitution_power')

    def test_partial_substitution_requires_matched_prompts(self):
        calibrate(self.config_path, self.rule_path)
        base, sub = self.path / 'base.npz', self.path / 'sub.npz'
        save_scores(base, make_bank(prefix='test'))
        save_scores(sub, make_bank(prefix='other'))
        with self.assertRaises(ValueError):
            evaluate(base, self.rule_path, self.path / 'mismatch.json', trials=20, substitute=sub)

    def test_fractional_calibration_budgets_are_not_silently_rounded(self):
        for field, value in [('budget', 1.5), ('trials', 1.5), ('budget', True), ('trials', True)]:
            cfg = {**self.cfg, field: value}
            self.config_path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                calibrate(self.config_path, self.path / f'{field}-{value}.json')


if __name__ == '__main__':
    unittest.main()
