"""Calibration artifact validation without restricting custom score-bank metadata."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from apitail.io import ScoreBank, save_scores, write_json
from apitail.pipeline import calibrate, audit, evaluate


class CalibrationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='apitail-rule-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {'model_id': 'fixture', 'model_config_sha256': None, 'dtype': 'float64',
                         'scoring': 'fixture', 'system': '', 'chat_kwargs': {}, 'max_length': 10}
        self.rule = self.prepare('normal', self.metadata)

    def bank(self, prefix, metadata):
        return ScoreBank(np.zeros((3, 1)), np.ones((3, 1)), np.asarray([1.]),
                         np.asarray([prefix + str(i) for i in range(3)]),
                         np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64), metadata, {})

    def prepare(self, name, metadata):
        source = self.root / (name + '-cal.npz')
        save_scores(source, self.bank('cal-', metadata))
        test = self.root / (name + '-test.npz')
        save_scores(test, self.bank('test-', metadata))
        config = self.root / (name + '-config.json')
        write_json(config, {'members': {'honest': source.name}, 'budget': 3,
                            'trials': 5, 'method': 'fixed', 'seed': 7})
        return calibrate(config, self.root / (name + '-rule.json'))

    def reject_with_both_commands(self, rule, label):
        path = self.root / (label + '.json')
        write_json(path, rule)
        for command in (audit, evaluate):
            with self.subTest(command=command.__name__, case=label):
                output = self.root / (label + '-' + command.__name__ + '.json')
                kwargs = {'trials': 3} if command is evaluate else {}
                with self.assertRaises(ValueError):
                    command(self.root / 'normal-test.npz', path, output, **kwargs)
                self.assertFalse(output.exists())

    def test_missing_provenance_fields_are_rejected(self):
        for field in ('score_semantics', 'calibration_prompt_ids'):
            rule = copy.deepcopy(self.rule)
            del rule[field]
            self.reject_with_both_commands(rule, 'missing-' + field)

    def test_malformed_containers_are_rejected(self):
        cases = [('rule-list', []), ('rule-null', None)]
        for field, values in [('score_semantics', [None, []]),
                              ('calibration_prompt_ids', [None, {}, 'cal-0', [['cal-0']], [0]])]:
            for i, value in enumerate(values):
                rule = copy.deepcopy(self.rule)
                rule[field] = value
                cases.append((field + '-' + str(i), rule))
        for label, rule in cases:
            self.reject_with_both_commands(rule, label)

    def test_prompt_ids_require_nonempty_unique_nonempty_strings(self):
        cases = [[], [""], ["cal-0", ""], ["cal-0", "cal-0"]]
        for i, prompt_ids in enumerate(cases):
            rule = copy.deepcopy(self.rule)
            rule['calibration_prompt_ids'] = prompt_ids
            self.reject_with_both_commands(rule, 'invalid-prompt-ids-' + str(i))

    def test_missing_semantics_fields_are_rejected(self):
        for field in self.metadata:
            rule = copy.deepcopy(self.rule)
            del rule['score_semantics'][field]
            self.reject_with_both_commands(rule, 'missing-semantic-' + field)

    def test_none_semantics_values_support_custom_numerical_banks(self):
        rule = self.prepare('custom', {})
        self.assertTrue(all(value is None for value in rule['score_semantics'].values()))
        for command in (audit, evaluate):
            with self.subTest(command=command.__name__):
                kwargs = {'trials': 3} if command is evaluate else {}
                result = command(self.root / 'custom-test.npz', self.root / 'custom-rule.json',
                                 self.root / ('custom-' + command.__name__ + '.json'), **kwargs)
                if command is audit:
                    self.assertFalse(result['reject'])
                else:
                    self.assertEqual(result['rate'], 0.)


if __name__ == '__main__':
    unittest.main()
