# Validation report

Validation for API-TAIL 0.1.2. Machine-readable results and source hashes are in [validation](../validation/).

| Check | Result | Evidence |
|---|---|---|
| Full unit/integration suite | 88 tests passed, zero skips | [software_checks.json](../validation/software_checks.json) |
| Installed wheel in an isolated CPU environment | 69 tests passed, 19 optional tests skipped; synthetic pipeline and pip check passed | [software_checks.json](../validation/software_checks.json) |
| Frozen paper-statistic replay | 2,020 checks, zero failures: 2,016 original checks plus four Table 3 pool-ablation checks | [paper_replay.json](../validation/paper_replay.json) |
| Original S/V functions on saved score arrays | 1,280 trial/method comparisons; zero count or temperature mismatches | [archive_scores.json](../validation/archive_scores.json) |
| Real model GPU generation and scoring | Three synthetic prompts, 24 tokens total, 151 temperatures plus nucleus | [gpu_smoke.json](../validation/gpu_smoke.json) |
| Independent scoring reloads | All 14 output arrays bitwise equal | [gpu_smoke.json](../validation/gpu_smoke.json) |
| Git CRLF checkout regression | Without attributes, frozen JSON hashes fail; with attributes, all paper assets retain their bytes and replay passes | [portability.json](../validation/portability.json) |
| Chinese JSON and prompt IDs | UTF-8 reads/writes pass with a simulated GBK default encoding | [test_text.py](../tests/test_text.py) |
| Legacy checks | 13 baseline tests, six help commands and original/adapted source hashes verified | [legacy/VALIDATION.json](../legacy/VALIDATION.json) |
| File, link and credential heuristics | See final scan | [release_check.json](../validation/release_check.json) |

The checkout regression used real Git with `core.autocrlf=true` on Linux. Native Windows and GitHub-hosted CI were not executed in this validation. The included CI matrix covers Linux and Windows when the repository workflow runs.

## GPU and numerical agreement

The GPU test used the existing Qwen3.8-27B checkpoint on one NVIDIA L20X. Generation and the two scoring passes loaded the model in separate processes. Peak allocated memory was about 50.25 GiB; workers exited and the device was checked for remaining compute processes.

On the same real logits, maximum absolute differences from the original scorer were 3.55e-15 for S and 2.84e-14 for V; nucleus and count fields matched exactly. This small test checks implementation and model loading. Paper-scale detection rates come from the separate frozen-statistic replay.

## Reproduction scope

The replay retains 83 recorded precision differences: 82 from float32 storage or interval interpolation, and one special-endpoint MET false-alarm rate differing by one trial out of 4,000. Its status is `passed_with_float32_differences`. Main plain-endpoint results are unaffected.

Paired intervals are recomputed from saved bootstrap replicates. Three count baselines have saved calibration quantiles rather than full calibration trial arrays. Original text collection and full model scoring were not rerun. Missing inputs for average fitted temperatures, repeated recalibration and collection cost are listed in [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md).

Paid API collection was tested with mocks. The CPU environment was isolated; the GPU environment was reused and has five pre-existing package metadata conflicts, documented in [ENVIRONMENT.md](ENVIRONMENT.md). Model weights and original prompt/response text are separate assets.

## Repeat the checks

```bash
python -m unittest discover -s tests -v
apitail demo --output work/demo
python scripts/reproduce_paper.py --output work/paper
python scripts/check_release.py --output work/check
python scripts/gpu_smoke.py --help
python scripts/verify_archive_scores.py --help
```

The GPU and archive checks require the local model or explicitly trusted archive inputs described by their help commands.
