# Assets and data availability

The code and the paper-result replay have different data requirements. The repository includes numerical audit statistics, not model weights or the original API responses. No paid API request is made by the replay command.

## Included frozen numerical assets

`data/paper/` contains four panel manifests and four compressed NumPy archives, plus a SHA-256 provenance manifest. The panels cross Qwen3.8-27B and Gemma-4-31B-it with P30 and provider-default sampling. The total size is approximately 4.36 MiB.

For the main budget of 200 requests, the archives retain the paper methods and the top-20 ablation: API-TAIL, temperature MLE, fixed-temperature full-vocabulary scoring, nucleus-restricted scoring, top-20 alignment, three likelihood/rank/entropy count baselines, RUT, and matched-budget MET. RUT and MET are present only where they were evaluated, on P30. For API-TAIL, the archives additionally retain every reported substitute at budgets 100 and 400, providing the complete 144-cell power grid across substitution rates 0.05, 0.10, and 0.25.

The included data are:

- calibration trial statistics, normally 2,000 per legitimate configuration;
- 4,000 honest audit statistics per evaluated endpoint and 1,000 statistics per substitution setting;
- the saved integer-rule rates from 2,000 paired prompt-bootstrap replicates at budget 200;
- frozen reference values used to check the recomputation, rather than to supply its reported powers or false-alarm rates;
- source artifact names and hashes, and hashes for every distributed numerical file.

`manifest.json` records the original source snapshot. The exports preserve the original numeric dtypes. They omit original prompt IDs, sampled row indices, text, API keys, request metadata, private paths, and an explicit table linking honest endpoint IDs to provider names. Honest endpoints retain the paper's Qxx/Gxx labels; substitute labels match the paper.

Qxx/Gxx labels are pseudonyms. Their ordering separates plain and special endpoints, then sorts source names alphabetically. The paper, configuration descriptions and numerical patterns may reveal endpoint identities.

The `source_configuration_label` field is a historical identifier for a locally generated configuration, retained to help trace calibration conditions. It is not an API endpoint identifier or a command to run.

## What replay computes

Run from the repository root:

```bash
python scripts/reproduce_paper.py --output work/paper_replay
```

Only NumPy and the Python standard library are needed. The script verifies file hashes, recalculates each available calibration 95th percentile using the `higher` quantile rule, takes the maximum across the legitimate pool, and measures the fraction of evaluation statistics strictly exceeding this threshold. It writes:

- `replay.json`: complete calculated thresholds, endpoint rates, power values, paired intervals, and verification differences;
- `main_results.csv`: 200-request comparisons and the top-20 ablation;
- `power_grid.csv`: all 144 API-TAIL substitute/rate/budget combinations;
- `pool_ablation.csv`: the four Table 3 results after removing penalty configurations.

The 2,020 checks include four reduced-pool comparisons for Table 3. Confidence intervals are recalculated from saved paired bootstrap rates; original prompt samples and audit batches are not regenerated.

Two input limitations affect replay:

1. The original archive did not retain calibration trial arrays for NLL, log-rank, and entropy count methods. For these three methods, the released inputs include the saved per-configuration quantiles; the script recomputes their maximum and then recomputes false-alarm rates and powers from saved evaluation counts. It does not reconstruct their upstream pooled 99th-percentile token/response cutoff. The JSON output labels this calibration input as `saved_configuration_quantiles`.
2. RUT and MET statistics were archived as float32. Their replay may differ from a source calculation using unrounded floats at a quantile boundary. The output records these differences. Integer-count methods use strict numerical checks. The numerical tolerance for continuous thresholds is two float32 ULPs; their rates allow at most one trial of discrepancy. Bootstrap interval comparisons allow two float32 ULPs because the source rates and published summaries were stored at float32 precision.

## Assets not bundled

| Asset | Why it is separate | Requirement for a new run |
|---|---|---|
| Claimed model weights and tokenizers | Tens of GB each; separate model terms and versions | Obtain the intended official model revision, record its revision/hash and tokenizer, and configure its local path. |
| WildChat-1M, BigCodeBench, and MATH source datasets | Dataset terms and possible personal information require separate review | Obtain each dataset from its original distributor and comply with its terms. |
| Exact frozen prompt pools and earlier-pool exclusion manifests | They include natural user text and selection metadata | Exact prompt-level reconstruction requires these separately supplied inputs. |
| Original commercial API response text | Not distributed; service terms and privacy constraints apply | Collect new responses with your own account and explicit budget, or separately obtain authorized historical assets. |
| Historical API keys, endpoint account configuration, spend logs | Private operational material | Supply credentials through environment variables; do not commit them. |
| Full scoring caches, raw token arrays, and historical experiments | Large and unnecessary for the included replay | Regenerate from authorized responses with the scorer, or obtain a separately reviewed artifact bundle. |

For new runs, use independently obtained prompts and responses, keep calibration and audit prompts disjoint, and record models, revisions, API parameters, collection times and random seeds. Provider routing and serving implementations can differ from those measured in the paper.

The distributed code and numerical audit statistics use Apache-2.0. External model weights, source datasets, original provider response text and other excluded assets retain their own terms.

The optional [archived-score conformance check](REPRODUCTION_STATUS.md#comparison-with-original-score-functions)
uses private, trusted historical S/V pickle caches to compare the portable core
with unchanged function bodies extracted from the original evaluator. Only the
sanitized result report and verification script are included; the original caches
and temporary numeric fixtures are excluded. The required `--trust-archive` flag
acknowledges that pickle inputs and archive Python can execute code. It is not
needed by the safe public NPZ replay.
