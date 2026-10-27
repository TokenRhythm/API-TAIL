# Reference implementations

These 23 source files provide the experiment scorer, evaluator and supporting statistics. Use the `apitail` package for the public command-line interface. These reference programs require additional assets and compatible input keys.

`MANIFEST.json` records original and adapted source hashes separately. `SANITIZATION.json` records endpoint-label changes and comment cleanup without publishing an identity mapping.

## Components

| Component | Purpose | Requirements |
| --- | --- | --- |
| `code/jobs/paper2/hf_score.py` | Deterministic full-vocabulary S/V scoring and optional top-20 export | Claimed model/tokenizer, prompts, response JSONL, CUDA PyTorch and Transformers |
| `code/jobs/paper2/gen_hf.py` | Local generation and weight-only FP8 rounding | Model, prompts, task JSON, CUDA and Transformers |
| `code/bbp/gen_multi.py`, `gen_local.py` | vLLM generation with supplied model checkpoints | vLLM, model/tokenizer, tasks and CUDA; GLM branches require additional unbundled code |
| `code/jobs/paper2/paper2_eval.py` | Simulation, calibration and bootstrap | Design JSON, top-20/grid and full-vocabulary caches, reference/endpoint responses, tokenizer |
| `code/jobs/paper2/paper2_report.py` | Aggregate panel results and bootstrap comparisons | Report configuration, panel JSON/NPZ, bootstrap shards and optional diagnostics |
| `code/jobs/paper2/metnat_eval2.py` | Native-protocol MET evaluation | Generation manifest, probes and tokenizable response data |
| `code/jobs/paper2/plain_rule2.py` | Parameter-compliance classification | Endpoint metadata and probe responses |
| `code/jobs/paper2/e6_tools.py` | Score and repeatability comparisons | Score caches and evaluation outputs |
| `code/bbp/rut.py`, `cmp_met_native.py` | RUT and Hamming-MMD reference functions | Score or token arrays |

Prompt banks, response texts, model weights, caches, credentials and endpoint account metadata are not included. The complete experiment therefore requires separately obtained assets. Pickle caches must come from a trusted source.

## Commands and inputs

From the repository root:

```bash
python legacy/run.py score -- --help
python legacy/run.py generate -- --help
python legacy/run.py generate-vllm -- --help
python legacy/run.py evaluate -- --help
python legacy/run.py met-native -- --help
python legacy/run.py met-reference -- --help
```

Model and data paths are explicit arguments. Imports resolve relative to `legacy/code`; `APITAIL_ARCHIVE_ROOT` supplies the asset root. Without it, the default is the absent `legacy/assets-not-included` directory.

Qwen/Gemma endpoint keys use the paper's Qxx/Gxx labels. Substitute-provider labels use provider A/B; other provider keys use distinct placeholders. These are pseudonyms, not irreversible anonymization: the paper's ordering and numerical patterns can permit cross-identification.

**Provider-named input archives need consistent key conversion.** Endpoint keys, condition keys and ID-derived cache filenames must match the labels used here. Keep the identity mapping outside this repository. Changing a design file alone does not convert its caches. The public `apitail` interface and numerical NPZ replay do not depend on these provider labels.

The path-rebasing helper only changes path prefixes:

```bash
python legacy/rebase.py --input /path/to/input.json \
  --output /path/to/converted.json --old-root /original/project/root \
  --archive-root /path/to/assets
```

It does not convert endpoint IDs, rename caches or overwrite output files. The report program accepts one positional report-configuration JSON.

## Numerical conventions

- `T_align-fsv` denotes full-vocabulary API-TAIL; `T_align` denotes top-20 alignment. `ProTA` adds penalty profiling.
- The evaluator aligns S/V rows to `(pid, k)` keys from `*.all.v3cond2.pkl`. Missing keys fail; extra S/V rows are ignored. It also loads Z/ZN grid caches for full-vocabulary methods.
- The 0.95 nucleus is limited to 4096 tokens. `overNF` records insufficient mass; out-of-nucleus targets contribute zero to nucleus S/V. Full-vocabulary scoring has no such window restriction.
- Torch and NumPy draws use different random streams; matching seeds does not imply bitwise equality.
- `gen_hf.py --fp8w 1` rounds selected Linear weights to e4m3fn with per-output-channel scales, then restores BF16 tensors. Vision/audio modules and the output head are excluded. It is a weight-rounding experiment, not native FP8 serving.

`VALIDATION.json` records the executed CPU comparisons, AST checks and command-help checks. Those checks do not establish GPU regeneration, new API collection or full panel replay.
