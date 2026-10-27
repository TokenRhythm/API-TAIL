# Paper design and code map

API-TAIL 0.1.1 implements full-vocabulary temperature-aligned likelihood-tail counting. The paper's source code calls this method **`T_align-fsv`**. In the archived code, `T_align` is the top-20 variant, while `ProTA` and `ProTA-N95` are penalty-profile variants.

The public package accepts new data through the CLI and Python interfaces. `legacy/` preserves the paper evaluator and its dependencies. Their data formats and random-number streams differ.

## Method and evaluation

| Paper operation | Public implementation | Original implementation in `legacy/code/` | Notes |
| --- | --- | --- | --- |
| Generate legitimate local responses | `src/apitail/generate.py`, `apitail generate` | `jobs/paper2/gen_hf.py`; Qwen: `bbp/gen_multi.py` | Model and data locations are explicit. GPU regeneration is distinct from offline array tests. |
| Collect text-only API responses | `src/apitail/collect.py`, `apitail collect` | Original collector is not bundled | Collection defaults to a dry run; credentials come from environment variables. |
| Teacher-force responses under the claimed model | `src/apitail/hf.py`, `apitail score` | `jobs/paper2/hf_score.py` | The scorer uses batch size one, disabled TF32 and deterministic Torch settings. |
| Compute response S(T), V(T) | `src/apitail/core.py::moments_from_logits` | `jobs/paper2/hf_score.py::stats` | Full vocabulary; grid 0.30 to 1.80 in steps of 0.01. Original token variance floor is 1e-6. |
| Select aligned temperature and count tails | `core.py::batch_statistic`, `batch_counts` | `jobs/talign/ta_run_eval.py::_stats`, dispatched by `jobs/paper2/paper2_eval.py` | Choose the first grid minimum of abs(sum of standardized response scores); count z > 3. |
| Composite calibration | `core.py::calibrate_counts`, `apitail calibrate` | `jobs/paper2/paper2_eval.py` | Per-configuration 0.95 quantile using `higher`, then maximum across configurations; strict K > tau for the paper's integer rule. |
| Single batch audit / simulation | `apitail audit`, `apitail evaluate` | `jobs/paper2/paper2_eval.py`, `jobs/supp1/engine.py` | Public NPZ score inputs avoid the original evaluator's dependency on old top-20 caches. |
| Endpoint validity, repeated calibration, prompt bootstrap | See release validation scope; complete frozen design remains in the archive | `jobs/paper2/paper2_eval.py` | Endpoint selection uses FAR <= 0.10 and at least 16/20 valid recalibrations, with >=0.99 coverage. Confidence intervals are calculated separately. |
| Paired comparison, Holm adjustment and paper aggregation | Frozen-result/export workflow documented separately | `jobs/paper2/paper2_report.py` | Original report needs panel output and bootstrap shards; source alone cannot regenerate the tables. |

`configs/paper_protocols.json` contains the exact pool memberships and generation parameters extracted from original metadata headers, together with provenance hashes. Main pool sizes are 18, 19, 16 and 18 for Qwen P30, Qwen P-user, Gemma P30 and Gemma P-user. Identity-system-prompt additions are recorded separately in the `+ID` pools. It is a parameter inventory, not a directly executable calibration configuration. It includes neither provider mappings nor the prompt/response dataset.

The inventory retains actual generation counts: Qwen pool metadata records four responses per prompt; Gemma pool metadata records two. Reference samples cost local generation and scoring, in addition to the audited API query budget. Checkpoint labels in the inventory are retained asset names, not verified public download links or redistribution permissions.

## Ablations and baselines

| Comparison | Public callable / supported route | Original identifier and source | Scope and budget |
| --- | --- | --- | --- |
| Full-vocabulary API-TAIL | `core.batch_statistic(..., method="align")` | `T_align-fsv` | N API responses, one per selected prompt. |
| MLE temperature | `core.batch_statistic(..., method="mle")` | `T_MLE-full`, `ta_run_eval.py` | Grid minimum of abs(sum S), then the same standardized tail count and own composite cutoff. |
| Huber temperature alignment | `core.batch_statistic(..., method="huber")` | `T Huber-fsv`, `paper2_eval.py` | Grid minimum of abs(sum clip(z, -1.345, 1.345)); raw z is used for the final tail count. |
| Fixed temperature | `core.batch_statistic(..., method="fixed")` | `FDG-fsv`, `paper2_eval.py` | T=1.0; Fast-DetectGPT-style typicality criterion adapted to this count/calibration protocol. |
| Nucleus-only alignment / MLE | Scorer nucleus arrays plus the corresponding core method; inspect scorer metadata | `N95-full id`, `N95-full MLE`, `hf_score.py` and `ta_run_eval.py` | Historical nucleus uses at most 4096 tokens. Overflow and out-of-nucleus behavior are described below. |
| Top-20 and penalty-profile variants | Archived evaluator | `T_align`, `T_MLE`, `FDG T=1`, `T Huber`, `T+pen (no Huber)`, `ProTA`; N95 T-only/penalty variants | Requires the archived grid caches; the public package does not regenerate the ProTA grids. |
| NLL / log-rank / entropy count | `baselines.response_detectors`, `detector_threshold`, `detector_counts` | `NLL count`, `log-rank count`, `entropy count`, `paper2_eval.py` and `bbp/rut.py` | Per-response 0.99 pooled calibration percentile, followed by a separate composite count cutoff. Entropy is normalized within the supplied top-20 window in this baseline, not full-vocabulary entropy. |
| RUT | `baselines.mean_log_rank`, `randomized_ranks`, `rut_statistic` | `bbp/rut.py`, `jobs/supp1/engine.py` | One API response plus 100 local reference responses per prompt. Mean log midrank, randomized reference rank, Cramer-von Mises uniformity test, then composite calibration of -log(p). Evaluated in the P30 panels. |
| MET at matched API budget | `baselines.pad_token_ids`, `paired_hamming_mmd2` | `MET same-budget`, `paper2_eval.py` | N/2 prompts, 2 API responses and 2 local reference responses per prompt, token length 30. Evaluated in the P30 panels. |
| MET in native protocol | `baselines.hamming_mmd2`; archived `metnat_eval2.py` | `bbp/cmp_met_native.py`, `jobs/paper2/metnat_eval2.py` | N/10 prompts, 10 API and 10 local reference responses per prompt, token length 50, temperature 1 protocol. Native null draws two disjoint groups from 20 references. |

The array baseline functions do not collect responses or choose the legitimate pool. RUT and MET return continuous statistics; use `baselines.calibrate_continuous` to calibrate them directly from null-audit arrays. It takes the maximum of each configuration's `higher` (1-alpha) Monte Carlo quantile and rejects strictly above that cutoff. It also returns all configuration quantiles, the decisive configurations, trial counts and strict calibration exceedance rates. This is not an exact randomized boundary rule, and its calibration rates are not independent false-alarm estimates.

For RUT, each null-audit value is **negative log p**, available as `rut_statistic(...)["statistic"]`; larger values are more suspicious. Do not calibrate upper tails of the raw p-value. For MET, use the raw MMD squared statistic, retaining negative unbiased estimates. The following tiny arrays illustrate the interface; real calibration needs enough independently generated null audits for each legitimate setting.

```python
import numpy as np
from apitail.baselines import calibrate_continuous

# Each element represents one complete null audit of the same method.
null_scores = {"setting_a": np.array([0.1, 0.3, 0.5, 0.8]),
               "setting_b": np.array([0.2, 0.4, 0.9, 1.2])}
cal = calibrate_continuous(null_scores, alpha=0.05)
observed_statistic = 1.4  # RUT: -log(p); MET: MMD^2, calibrated separately.
reject = observed_statistic > cal["threshold"]  # equality never rejects
```

`core.calibrate_counts` is for integer counts; do not round RUT/MET scores or pass them to that function. The likelihood-count baselines can use `calibrate_counts` after their per-response threshold is fixed.

### Details needed for faithful comparisons

- RUT midranks use fp16 equality in the stored top-k window. A token whose reported rank is outside the window retains that rank. The public callable uses the first 100 supplied references, as in the paper evaluator.
- RUT tie randomization is `(L + U * (E + 1)) / (m + 1)`. Callers can provide U explicitly to replay the same draw. The public implementation uses SciPy's public Cramer-von Mises API; its p-value was checked against the frozen evaluator's private SciPy CDF formula.
- MET uses positional Hamming similarity, including equal padding tokens. The MMD estimator omits same-set diagonal pairs and may be negative. Each prompt needs at least two samples on each side. In matched-budget mixture draws, if both API samples come from the same source, they must be two distinct stored responses.
- The likelihood-count 0.99 percentile pools the rows actually drawn during calibration, including repeats. Computing it once over unique saved responses can change the result. NaN response statistics do not count as tail exceedances in the archived implementation.
- Thresholds are recalibrated for each method. Reusing API-TAIL's integer cutoff for MLE, RUT or MET does not reproduce the comparison.

## Original sampling and random streams

`jobs/supp1/engine.py` first restricts to eligible prompts. The ordinary draw selects N distinct positions from that prompt list, with no replacement. Each selected request gets an independent Bernoulli substitution route; one stored completion is then sampled uniformly from the chosen source. Eligible prompts need a base response, a substitute response when applicable, and enough references for RUT.

Qwen calibrates on v3; Gemma calibrates on v5 CAL. Both test on disjoint v5 T1+T2. Main null simulation uses 4000 draws per configuration: even draws calibrate and odd draws estimate held-out false alarms. Alternative simulations use 1000 draws. The integer decision rule is the main reported rule; randomized boundary rejection is a separate diagnostic.

The original NumPy stream is PCG64 with `SeedSequence([20261025, bootstrap_id+1, condition_hash, stream, repeat])`, where the condition hash is the first eight SHA-256 hex digits of the serialized condition. Calibration uses stream 0, held-out/test uses stream 1, and NumPy RUT uniforms use stream+2. The GPU path uses Torch streams derived from these seeds and has a different draw order. Bootstrap resamples the TEST prompt list with replacement using `default_rng(100000+bootstrap_id)`; calibration prompts are not resampled by default. Subsequent sampling is without replacement among positions in that resampled list, so an original prompt ID can recur.

MET has its own reference and routing design and its own stream seeded from 20261026 and a condition hash. It is matched on API request budget, not literally the same one-response-per-prompt draw as API-TAIL. The NumPy interface uses its recorded seed and a different draw order from the archived GPU simulator.

## Numerical scoring and quantization

The reference scorer computes logits in model inference precision, then uses float32 shifted logits/exponentials with float64 moment accumulation. Full-vocabulary scoring is distinct from saving top-20 log-probabilities for baselines. It retokenizes returned text under the claimed tokenizer and preserves `(pid, k)` keys. The original evaluator aligns full-score rows to the earlier grid-cache keys, rejects missing keys and ignores extra full-score rows.

For the historical N95 ablation, the scorer keeps the shortest top-4096 prefix reaching mass 0.95 under the full distribution. If the full top-4096 mass is smaller, it records `overNF` and keeps the available window; there is no automatic vocabulary expansion. A target token outside the retained nucleus contributes zero to nucleus S/V and is excluded from its count. These choices must be accounted for when interpreting the nucleus ablation. They do not constrain the final full-vocabulary score.

Qwen FP8, AWQ INT4 and RedHatAI INT4 pool members use separately retained quantized checkpoints with the original vLLM generation entry. The checkpoint bytes and inference compatibility are not established by a name alone. Gemma `fp8w` members instead use the archived HF generator's e4m3fn rounding: each selected Linear weight row is scaled by its maximum absolute weight / 448, rounded to float8, rescaled, and stored in BF16. Vision/audio modules and the output head are excluded. This is a weight-perturbation ablation, not a native FP8 backend implementation. Provider-reported FP4/FP8 metadata is a separate endpoint classification input, not a locally reproduced quantizer.

## Tests and replay

`tests/test_baselines.py` covers midranks, entropy, detector thresholds, RUT ties and reference selection, CvM agreement, MMD, padding, continuous calibration and invalid inputs. It also compares the public RUT midrank and native MET functions with their archived counterparts.

API-TAIL 0.1.1 passes 78 tests with the optional backend installed; the lightweight CPU environment passes 64 and skips 14 backend tests. Paper replay passes 2,020 checks, and the comparison with original score functions passes 1,280 trial/method checks. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for software and GPU results and [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) for the paper-data coverage.
