# Reproduction status

API-TAIL 0.1.1 provides paper-result replay from saved numerical assets and a workflow for new data. The paper's original API collection and full model scoring have not been rerun end to end.

## Paper-result replay

```bash
python scripts/reproduce_paper.py --output work/paper_replay
```

The replay passes **2,020 checks with zero failures**. It produces 36 method/ablation rows, 144 API-TAIL power-grid cells, 16 paired confidence intervals, and four pool-ablation rows. In the JSON output, `checks=2016` covers the main statistics and `additional_checks=4` covers the pool ablation; `total_checks=2020` is the full count.

| Panel | API-TAIL threshold | Maximum plain FAR | Main 10% substitution power | Power difference versus MLE |
|---|---:|---:|---:|---:|
| Qwen P30 | 7 | 0.03525 | 0.325 | +0.223 |
| Qwen P-user | 5 | 0.00150 | 0.694 | +0.634 |
| Gemma P30 | 8 | 0.00300 | 0.893 | +0.093 |
| Gemma P-user | 5 | 0.06900 | 1.000 | 0.000 |

Integer-count results match the frozen values within `1e-12`. The status is `passed_with_float32_differences`: 82 differences concern continuous-method quantiles or bootstrap interpolation, and one concerns MET's FAR on special endpoint Q18 in Qwen P30. The latter is lower by `1/4000 = 0.00025` after loading the saved float32 statistics and does not change the plain-endpoint maximum. All differences and tolerances are recorded in `replay.json`.

## Table 3: removing penalty configurations

The replay removes the presence-penalty 1.5 and 2.0, repetition-penalty 1.05 and 1.10, and combined penalty configurations. It recalculates the maximum calibration quantile over the remaining pool and applies it to the unchanged endpoint count arrays.

| Panel, in paper column order | Reduced-pool threshold | Recomputed maximum plain FAR | Published FAR |
|---|---:|---:|---:|
| Qwen P30 | 5 | 0.26925 | 0.269 |
| Qwen P-user | 3 | 0.07400 | 0.074 |
| Gemma P30 | 8 | 0.00300 | 0.003 |
| Gemma P-user | 5 | 0.06900 | 0.069 |

The four checks compare FARs rounded to the paper's three decimals. Thresholds and rates come from the numerical arrays; published values are comparison targets. `replay.json` records excluded configurations, retained quantiles, decisive configurations and endpoint rates. `pool_ablation.csv` contains these four rows.

## Public replay coverage

| Result | Available calculation |
|---|---|
| Artifact integrity | Every numerical file is checked against its manifest hash. |
| Composite thresholds | Recompute configuration quantiles and their maximum for API-TAIL, MLE, fixed temperature, nucleus, top-20, RUT and MET where available. |
| NLL, log-rank and entropy count controls | Saved calibration quantiles are inputs; their maximum and evaluation rates are recomputed. Original calibration trial arrays are absent. |
| API-TAIL substitution grid | Recompute all 144 reported rate/budget/substitute cells. |
| Paired intervals | Recompute 16 ordinary 95% intervals from 2,000 saved paired bootstrap replicates per comparison. |
| Reduced calibration pool | Recompute Table 3 thresholds and FARs after removing the five penalty configurations. |
| Mean fitted temperatures | Original evaluation-batch temperatures or sufficient sums/counts are not included. |
| Twenty calibration redraws | Redraw-specific statistic banks and evaluation records are not included. |
| Alternate randomized decision rule | Outside this numerical export. |
| Original collection totals and cost | Request ledgers, token usage and billing/price records are not included. |

Saved tail counts do not determine fitted temperatures. Temperatures in the archive-conformance report describe its small test batches, not the paper's complete evaluation batches. Similarly, simulated audit trials are not unique API responses: the public assets cannot independently establish the paper's 143,955-response total or USD 29.92 expenditure.

The interval replay uses existing bootstrap rates rather than resampling original prompts. See [ASSETS.md](ASSETS.md) for the data required to reconstruct those stages.

## Comparison with original score functions

`core.batch_counts` agrees exactly with the original `ta_run_eval._stats` and `paper2_eval.stats` on **1,280 trial/method comparisons**, with zero K or temperature mismatches. The check uses eight saved float64 score caches: one calibration and one test source per panel, 512 selected responses per source, 32 batches of 100 responses, and five rules (alignment, MLE, fixed temperature, Huber and nucleus).

The original function bodies are compiled unchanged from their source AST and run on a minimal CPU panel. This comparison covers score-to-statistic calculations; it does not run the original prompt sampler or model scorer.

[archive_scores.json](../validation/archive_scores.json) records source, function, design, cache and sampling hashes. To rerun with a separately available trusted archive:

```bash
python scripts/verify_archive_scores.py \
  --archive-root "$ARCHIVE_ROOT" --work-dir "$WORK_DIR" \
  --output "$WORK_DIR/archive_scores.json" --trust-archive
```

The archive check requires NumPy, CPU PyTorch, original Python sources and trusted pickle caches. The `--trust-archive` flag is mandatory because archive code and pickle loading can execute code. Public paper replay uses `allow_pickle=False` and needs neither PyTorch nor the original archive.

## Software and model checks

The full test environment passes **78 tests**. The lightweight CPU environment passes **64 tests and skips 14 optional backend tests**. A local GPU smoke test covers three synthetic prompts, 24 generated tokens and 151 temperatures, including nucleus scoring. Commands, environment details and machine-readable results are in [VALIDATION_REPORT.md](VALIDATION_REPORT.md) and [ENVIRONMENT.md](ENVIRONMENT.md).
