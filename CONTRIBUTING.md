# Contributing

The API-TAIL 0.1.1 package is in `src/apitail/`. [DESIGN_MAP.md](docs/DESIGN_MAP.md) maps paper operations to code, and [SCHEMAS.md](docs/SCHEMAS.md) defines the input and output formats.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,dev]"
python -m unittest discover -s tests -v
python scripts/reproduce_paper.py --output work/paper-check
python -m build
```

Use the same interpreter for installation and testing. This lightweight setup runs 64 tests and skips 14 model-backend tests. Install `.[score]` with a suitable PyTorch build to run all 78 tests. Model weights are needed for the separate model smoke test, not for the unit tests.

Before a GPU run, check available storage and device use. Record the model revision, environment, command, seed and output hashes. The API collector defaults to a dry run; actual requests require `--execute` and credentials in environment variables.

## Changing the method

Tests should cover temperature selection, ties, strict tail and rejection inequalities, variance floors, prompt separation, source compatibility and reference budgets. Scorer changes also need token-alignment and moment-arithmetic checks. Update the scoring metadata version when numerical semantics change.

A new temperature estimator needs corresponding core, pipeline, configuration and documentation changes. A new scorer should produce a validated `ScoreBank`. RUT and MET extensions should preserve explicit local-reference and API-request counts.

When correcting a frozen numerical asset, update its provenance and hashes and identify the affected results. Keep new experiment outputs in separate directories.

## Validation

Report the commands run and distinguish numerical replay, original-function comparison, and raw generation/scoring/evaluation. [REPRODUCTION_STATUS.md](docs/REPRODUCTION_STATUS.md) describes available paper results; [VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md) records software and model checks.

The optional `scripts/verify_archive_scores.py` uses original Python functions and pickle caches from a separately available trusted archive. It requires `--trust-archive`. Public numerical replay uses NPZ files loaded without pickle.

## Contributions and licensing

Submit a focused change with its purpose, relevant checks and numerical or protocol changes. Prefer synthetic test fixtures. Keep credentials, private paths, provider logs and unlicensed data out of contributions.

Code and distributed numerical assets use Apache-2.0. Preserve third-party notices and identify upstream adaptations. External models, datasets and provider outputs retain their own terms; see [ASSETS.md](docs/ASSETS.md).
