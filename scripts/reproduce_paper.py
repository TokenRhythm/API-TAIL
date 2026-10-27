#!/usr/bin/env python3
"""Recompute paper statistics and paired intervals from saved numerical arrays."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


# Table 3 targets are rounded only for comparison with computed rates.
TABLE3_WITHOUT_PENALTIES = {
    'q38_p30': 0.269,
    'q38_user': 0.074,
    'g4_p30': 0.003,
    'g4_user': 0.069,
}
PENALTY_SUFFIXES = frozenset({'pp15', 'pp20', 'rp105', 'rp110', 'pp15rp105'})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(data_dir):
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['outputs']:
        path = data_dir / item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError('Missing or modified frozen asset: ' + item['file'])
    report = {'scope': manifest['scope'], 'checks': 0, 'additional_checks': 0, 'precision_differences': [], 'panels': {}}
    failures = []
    table_rows = []
    power_rows = []

    def check(actual, expected, label, tolerance=1e-12, additional=False):
        report['additional_checks' if additional else 'checks'] += 1
        if not np.isfinite(actual) or abs(actual - expected) > tolerance:
            failures.append({'field': label, 'computed': actual, 'expected': expected, 'tolerance': tolerance})
        elif abs(actual - expected) > 1e-12:
            report['precision_differences'].append({'field': label, 'difference': actual - expected, 'tolerance': tolerance})

    def replay_pool_ablation(meta, rule, arrays, full_quantiles):
        panel = meta['panel']
        labels = {c['id']: c['source_configuration_label']
                  for c in meta['configurations']}
        removed = {key for key, label in labels.items()
                   if label.rsplit('_', 1)[-1] in PENALTY_SUFFIXES}
        if (len(removed) != 5 or
                {labels[key].rsplit('_', 1)[-1] for key in removed} != PENALTY_SUFFIXES):
            raise ValueError('Expected five distinct penalty configurations: ' + panel)
        if any(c['array'] is None for c in rule['calibration']):
            raise ValueError('Pool ablation needs calibration trial arrays: ' + panel)
        if not removed <= full_quantiles.keys():
            raise ValueError('Penalty configurations are missing from calibration: ' + panel)
        # Recalibrate the pool while keeping endpoint draws fixed.
        quantiles = {key: q for key, q in full_quantiles.items() if key not in removed}
        if not quantiles:
            raise ValueError('Empty calibration pool after penalty removal: ' + panel)
        threshold = max(quantiles.values())
        endpoints = {}
        for endpoint in rule['honest']:
            values = arrays[endpoint['array']]
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError('Invalid pool-ablation endpoint statistics: ' + panel)
            endpoints[endpoint['id']] = {
                'far': float(np.mean(values.astype(float) > threshold)),
                'plain': endpoint['plain'], 'trials': len(values),
            }
        plain = {key: val['far'] for key, val in endpoints.items() if val['plain']}
        max_far = max(plain.values())
        rounded = float(f'{max_far:.3f}')
        check(rounded, TABLE3_WITHOUT_PENALTIES[panel],
              panel + '/table3_without_penalties/rounded_max_plain_far',
              additional=True)
        return {
            'N': 200, 'method': 'api_tail',
            'removed_configurations': [
                {'id': key, 'source_configuration_label': labels[key]}
                for key in sorted(removed)
            ],
            'remaining_configurations': len(quantiles),
            'configuration_quantiles': quantiles,
            'decisive_configurations': [key for key, q in quantiles.items() if q == threshold],
            'threshold': threshold, 'endpoint_rates': endpoints,
            'plain_max_far': max_far,
            'maximizing_plain_endpoints': [key for key, far in plain.items() if far == max_far],
            'plain_max_far_3dp': rounded,
            'published_plain_max_far_3dp': TABLE3_WITHOUT_PENALTIES[panel],
            'verification': 'Table 3 rounded FAR; computed from retained calibration and endpoint trial statistics',
        }

    for file in sorted(data_dir.glob('*.json')):
        if file.name == 'manifest.json':
            continue
        meta = json.loads(file.read_text(encoding='utf-8'))
        panel = meta['panel']
        result = {'methods': {}, 'bootstrap_comparisons': {}}
        with np.load(file.with_suffix('.npz'), allow_pickle=False) as arrays:
            for rule in meta['rules']:
                N, method = rule['N'], rule['method']
                tag = f'{panel}/{N}/{method}'
                quantiles = {}
                replayed = 0
                for cfg in rule['calibration']:
                    if cfg['array'] is not None:
                        values = arrays[cfg['array']]
                        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                            raise ValueError('Invalid calibration statistics: ' + tag)
                        q = float(np.quantile(values.astype(float), 1.0 - meta['alpha'], method='higher'))
                        tol = max(1e-12, 2 * abs(float(np.spacing(np.float32(cfg['q95_expected']))))) if rule['continuous'] else 1e-12
                        check(q, cfg['q95_expected'], tag + '/q95/' + cfg['id'], tol)
                        replayed += 1
                    else:
                        # Count baselines use per-configuration quantiles instead of trial arrays.
                        q = float(cfg['q95_expected'])
                    quantiles[cfg['id']] = q
                threshold = max(quantiles.values())
                threshold_tol = max(1e-12, 2 * abs(float(np.spacing(np.float32(rule['threshold_expected']))))) if rule['continuous'] else 1e-12
                check(threshold, rule['threshold_expected'], tag + '/threshold', threshold_tol)
                endpoints, powers = {}, {}
                for kind, field, expectation in [('honest', 'far', 'far_expected'), ('substitutions', 'power', 'power_expected')]:
                    for condition in rule[kind]:
                        if condition['array'] is None:
                            raise ValueError('Missing evaluation statistics: ' + tag + '/' + condition['id'])
                        values = arrays[condition['array']]
                        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
                            raise ValueError('Invalid evaluation statistics: ' + tag)
                        rate = float(np.mean(values.astype(float) > threshold))
                        # Float32 statistics can differ by one trial at a quantile boundary.
                        tol = 1.0 / len(values) + 1e-12 if rule['continuous'] else 1e-12
                        check(rate, condition[expectation], tag + '/' + condition['id'] + '/' + field, tol)
                        if kind == 'honest':
                            endpoints[condition['id']] = {'far': rate, 'plain': condition['plain'], 'trials': len(values)}
                        else:
                            powers[condition['id']] = rate
                            if method == 'api_tail':
                                power_rows.append({'panel': panel, 'N': N, 'substitute': meta['substitutes'][condition['substitute']], 'rho': condition['rho'], 'power': rate})
                max_far = max(v['far'] for v in endpoints.values() if v['plain'])
                rate_tol = max((1.0 / v['trials'] for v in endpoints.values() if v['plain']), default=0) + 1e-12 if rule['continuous'] else 1e-12
                check(max_far, rule['plain_max_far_expected'], tag + '/max_plain_far', rate_tol)
                record = {'threshold': threshold, 'configuration_quantiles': quantiles,
                          'calibration_trial_arrays_replayed': replayed, 'calibration_configurations': len(quantiles),
                          'calibration_input': 'trial_statistics' if replayed == len(quantiles) else 'saved_configuration_quantiles',
                          'plain_max_far': max_far, 'endpoint_rates': endpoints, 'powers': powers}
                result['methods'][f'{N}/{method}'] = record
                if N == 200:
                    table_rows.append({'panel': panel, 'method': meta['method_labels'][method], 'threshold': threshold,
                                       'max_plain_far': max_far, 'power_rho_0_10': powers.get(meta['main_substitution'])})
            ablation_rule = next(r for r in meta['rules']
                                 if r['N'] == 200 and r['method'] == 'api_tail')
            result['pool_ablation_without_penalties'] = replay_pool_ablation(
                meta, ablation_rule, arrays,
                result['methods']['200/api_tail']['configuration_quantiles'])
            boot = meta['bootstrap']
            rates = arrays[boot['array']]
            if rates.shape != (boot['replicates'], len(boot['methods']), len(boot['items'])):
                raise ValueError('Unexpected bootstrap shape: ' + panel)
            item = next(i for i, x in enumerate(boot['items']) if x['kind'] == 'substitution' and x['id'] == meta['main_substitution'])
            a = rates[:, boot['methods'].index('api_tail'), item]
            for comparison in boot['comparisons']:
                method = comparison['alternative']
                b = rates[:, boot['methods'].index(method), item]
                valid = np.isfinite(a) & np.isfinite(b)
                differences = (a - b)[valid]
                if not len(differences):
                    raise ValueError('No paired bootstrap replicates: ' + panel + '/' + method)
                interval = np.quantile(differences.astype(float), [0.025, 0.975]).tolist()
                for i in range(2):
                    tol = max(1e-12, 2 * abs(float(np.spacing(np.float32(comparison['ci95_expected'][i])))))
                    check(interval[i], comparison['ci95_expected'][i], panel + '/bootstrap/' + method + '/' + str(i), tol)
                delta = result['methods']['200/api_tail']['powers'][meta['main_substitution']] - result['methods']['200/' + method]['powers'][meta['main_substitution']]
                check(delta, comparison['delta_expected'], panel + '/delta/' + method)
                result['bootstrap_comparisons'][method] = {'delta_power': delta, 'paired_ci95': interval, 'paired_replicates': int(valid.sum())}
        report['panels'][panel] = result
    report['total_checks'] = report['checks'] + report['additional_checks']
    report['check_groups'] = {'original_frozen_replay': report['checks'],
                              'table3_without_penalties': report['additional_checks']}
    report['failures'] = failures
    report['status'] = 'failed' if failures else ('passed_with_float32_differences' if report['precision_differences'] else 'passed')
    return report, table_rows, power_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'paper')
    parser.add_argument('--output', type=Path, help='Optional output directory for JSON and CSV results.')
    args = parser.parse_args()
    report, table, grid = reproduce(args.data.resolve())
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'replay.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        ablation_rows = []
        for panel in TABLE3_WITHOUT_PENALTIES:
            r = report['panels'][panel]['pool_ablation_without_penalties']
            ablation_rows.append({
                'panel': panel, 'N': r['N'], 'method': r['method'],
                'removed_configurations': len(r['removed_configurations']),
                'remaining_configurations': r['remaining_configurations'],
                'threshold': r['threshold'], 'max_plain_far': r['plain_max_far'],
                'max_plain_far_3dp': r['plain_max_far_3dp'],
                'published_max_plain_far_3dp': r['published_plain_max_far_3dp'],
            })
        for filename, rows in [('main_results.csv', table), ('power_grid.csv', grid),
                               ('pool_ablation.csv', ablation_rows)]:
            with (args.output / filename).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {'status': report['status'], 'checks': report['checks'], 'additional_checks': report['additional_checks'],
               'total_checks': report['total_checks'], 'failures': len(report['failures']),
               'float32_differences': len(report['precision_differences']), 'main_rows': len(table), 'power_grid_rows': len(grid),
               'table3_without_penalties': {p: report['panels'][p]['pool_ablation_without_penalties']['plain_max_far']
                                            for p in TABLE3_WITHOUT_PENALTIES},
               'api_tail': {p: {'threshold': r['methods']['200/api_tail']['threshold'], 'max_plain_far': r['methods']['200/api_tail']['plain_max_far'],
                               'delta_vs_mle': r['bootstrap_comparisons']['temperature_mle']['delta_power']} for p, r in report['panels'].items()}}
    print(json.dumps(summary, indent=2))
    if report['failures']:
        print(json.dumps(report['failures'][:10], indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
