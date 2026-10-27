# Data formats and numerical conventions

These formats apply to API-TAIL 0.1.1.

## Text inputs

Prompts are UTF-8 JSONL with one object per line:

```json
{"pid": "cal-001", "text": "Your prompt text"}
```

`pid` is converted to a string and must be unique within the prompt file. Text must be nonempty. Use genuinely disjoint calibration and test prompts; the implementation checks their IDs, not semantic duplication.

Responses are UTF-8 JSONL:

```json
{"pid": "cal-001", "k": 0, "text": "A model completion"}
```

The pair `(pid, k)` identifies a response and must be unique. `k` is a nonnegative integer; if absent, scoring defaults it to zero. Every response must refer to a supplied prompt. Extra fields such as `finish_reason`, `ids`, or `usage` may be retained, but scoring tokenizes `text` under the claimed model tokenizer rather than trusting returned token IDs. Lines with an `_meta` field are ignored by the JSONL reader.

The generation recipe uses `seed`, `responses_per_prompt`, `max_length`, `system`, and a `sampling` object. Supported sampling controls are temperature, top-p, top-k, min-p, presence penalty, repetition penalty, and maximum tokens. `fp8_weight_rounding` is an optional simulated weight-rounding intervention; it is not a native FP8 serving backend.

Endpoint configurations use `url`, `model`, `api_key_env`, `responses_per_prompt`, `timeout`, optional `system`, and a `parameters` object. Credential values and arbitrary headers are rejected in the config; secrets come from the named environment variable. Collection accepts HTTPS endpoints or local HTTP endpoints, and is dry-run unless explicitly executed.

## Public score banks

Score files are compressed NPZ loaded with `allow_pickle=False`. Use `apitail.io.save_scores` and `load_scores` to preserve validation and metadata.

| Field | Shape/type | Meaning |
|---|---|---|
| `S` | (responses, temperatures), float64 | Sum over response tokens of surprisal minus reference entropy |
| `V` | same as S, float64 | Sum of token surprisal variances |
| `grid` | (temperatures,), float64 | Positive, strictly increasing temperatures |
| `pid` | (responses,), string | Prompt identifier for each response |
| `k` | (responses,), integer | Response index within prompt |
| `n_tokens` | (responses,), positive integer | Number of scored response tokens |
| `metadata` | scalar JSON string | Model/scorer identity and provenance |
| `S_nucleus`, `V_nucleus` | same as S, optional | Moments retained by nucleus scoring |
| `n_nucleus` | same as S, optional | Retained token count at each temperature |
| `nucleus_overflow` | same as S, optional | Tokens for which the top-4,096 mass is below 0.95 |
| `nll`, `logrank`, `entropy`, `out20` | (responses,), optional | HF scoring diagnostics |

`S` and `V` must be finite, with nonnegative variance. Extras are non-object arrays whose first dimension matches the response count. The metadata schema is `apitail-scores-v1`. The HF scorer records the model label, local model-config hash when available, dtype, scoring convention, system/chat settings, context limit, library versions, and input-file hashes.

Compatibility checks compare grid, model ID/config hash, dtype, scoring convention, system prompt, chat kwargs, and maximum length. Weight-file hashes are not part of this check. Inspect prompt IDs and metadata before sharing score banks.

The bundled `data/paper/` NPZ files use a separate compact replay schema, documented by their JSON manifests and `scripts/reproduce_paper.py`. They are not response-level score banks for `apitail audit`.

## Scoring and temperature selection

The paper grid contains 151 values from 0.30 to 1.80, inclusive, in steps of 0.01. At each temperature, the full-vocabulary score is `z = S / sqrt(max(V, 1e-12))`. The HF scorer floors each token variance at 1e-6 before summing.

| Method | Temperature choice |
|---|---|
| `align` | First grid minimizer of `abs(sum(z))` |
| `mle` | First grid minimizer of `abs(sum(S))` |
| `huber` | First grid minimizer of `abs(sum(clip(z, -1.345, 1.345)))` |
| `fixed` | T = 1.0 |
| `nucleus` | Alignment using nucleus S/V arrays |

The sum is over responses in the current audit batch. The tail count is `K = count(z > z_cutoff)`, with default cutoff 3. The MLE label refers to a discretized likelihood score-equation rule, not exact optimization of likelihood over the finite grid.

HF scoring has batch size one. Exponentials and products use float32; sums and final moment arithmetic use float64. The NumPy logits reference routine uses float64 throughout and can differ slightly from the HF scorer. The HF backend rejects an empty tokenized response or an input satisfying `prompt_tokens + response_tokens + 1 > max_length`. The archival scorer truncated inputs instead.

With `--nucleus`, the candidate set is limited to the top 4,096 tokens. The distribution is renormalized on the retained set, and response tokens outside that set do not contribute to nucleus S/V. `nucleus_overflow` must be inspected: a nonzero entry means the candidate set did not reach the requested mass. A response with no retained tokens has zero nucleus S/V and therefore zero standardized score under the denominator convention.

## Calibration and results

A pool configuration contains `budget`, `trials`, and a nonempty `members` map from configuration names to score-file paths. Optional fields are `method`, `alpha`, `seed`, and `z_cutoff`. Relative score paths resolve against the pool JSON's directory.

Every simulated batch samples `budget` distinct prompts without replacement and one stored response uniformly per prompt. Thus budget must be no larger than each source's distinct-prompt count. Calibration stores the maximum per-configuration `(1-alpha)` quantile, using NumPy's `higher` convention.

The calibration JSON has schema `apitail-calibration-v1`. It records `threshold`, per-member `quantiles`, the decisive member, trial counts, method, budget, grid, scorer semantics, calibration prompt IDs, and source hashes. An `exact` field records a randomized empirical diagnostic; the CLI audit/evaluate commands use the ordinary strict integer rule, not that randomized diagnostic.

`audit` takes exactly one response per prompt and exactly the calibrated budget. It reports K, fitted temperature, threshold, rejection decision, z values, and input hashes. `evaluate` resamples a score bank and records the rate, per-trial counts and temperatures, seed, budget, and hashes. For substitution, both banks must cover the same held-out prompt IDs; each request is independently assigned to the substitute with probability `rho`. Calibration IDs are rejected in either test bank.

## Baseline Python interfaces

`apitail.baselines` exposes arrays rather than provider calls:

- `rut_statistic(target_scores, reference_scores, reference_count=100)`: one target score per prompt with shape (N,), local reference scores (N, m), and optional explicit uniforms (N,). Scores are response mean log midranks, obtained from `mean_log_rank`. The result includes CvM W2, its SciPy p-value, and `-log(p)` for separate composite calibration.
- `paired_hamming_mmd2(api_tokens, reference_tokens)`: two arrays of shape (prompts, 2, L). The paper's matched-budget setting uses L=30; N=200 API requests therefore means 100 prompts. Local reference generation is additional work.
- `hamming_mmd2`: lists of per-prompt integer arrays, at least two responses on each side. Padding participates in the Hamming kernel; negative unbiased MMD estimates are retained.
- `response_detectors`, `detector_threshold`, and `detector_counts`: response scores, pooled response cutoff, and strict tail counts. The historical entropy control uses renormalized top-20 entropy, whereas the HF score-bank diagnostic `entropy` is full-vocabulary entropy. They are not interchangeable.

Supply reference responses and null-audit arrays separately. RUT p-values require SciPy. Use `baselines.calibrate_continuous` for composite calibration of RUT negative log p-values and MET statistics; see [DESIGN_MAP.md](DESIGN_MAP.md) for the interface and reference budgets.
