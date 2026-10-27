# API-TAIL

[中文](README.zh-CN.md) · [Audit walkthrough](docs/QUICKSTART.md) · [Data formats](docs/SCHEMAS.md) · [Validation](docs/VALIDATION_REPORT.md)

**Version 0.1.1.** API-TAIL detects possible model substitution in LLM APIs by scoring responses with the claimed reference model and calibrating over allowed decoding configurations.

## Choose your starting point

| What you want to do | What you need | Start here |
|---|---|---|
| Try the software | Python 3.10+ with pip and venv | CPU quickstart below |
| Recompute the paper's numerical results | Same setup; numerical assets are included | CPU quickstart below |
| Audit your own API | Reference model, calibration/test prompts, suitable hardware and an API account | [Audit walkthrough](docs/QUICKSTART.md) |

The CPU quickstart runs offline after installation. It needs no GPU, model weights or API key. Paper replay uses the supplied saved statistics.

## 1. Get the code

Clone the repository:

```bash
git clone https://github.com/TokenRhythm/API-TAIL.git
cd API-TAIL
```

Alternatively, download and extract the repository ZIP, then open a terminal in the extracted folder containing `pyproject.toml`. Run the following commands from that folder.

## 2. Install and run the CPU quickstart

Choose the block for your operating system.

**Linux / macOS / WSL — Bash**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m apitail --version
python -m apitail doctor
python -m apitail demo --output work/demo1
python scripts/reproduce_paper.py --output work/paper1
```

**Windows — PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m apitail --version
.\.venv\Scripts\python.exe -m apitail doctor
.\.venv\Scripts\python.exe -m apitail demo --output work/demo1
.\.venv\Scripts\python.exe scripts/reproduce_paper.py --output work/paper1
```

If `py` is unavailable, use `python -m venv .venv` for the first Windows command. The remaining Windows commands call the virtual environment's Python directly, so no activation script is needed. The base installation installs NumPy; `torch: null` and `transformers: null` in `doctor` are expected.

## 3. Check the results

- The version command prints `apitail 0.1.1`.
- The demo prints `"synthetic": true` and writes its calibration, audit and evaluation results.
- Paper replay reports `"total_checks": 2020` and `"failures": 0`. Its expected status is `passed_with_float32_differences`; the [reproduction notes](docs/REPRODUCTION_STATUS.md) explain those recorded precision differences.

| Output | Contents |
|---|---|
| `work/demo1/summary.json` | Synthetic example summary |
| `work/demo1/audit.json` | Example batch decision |
| `work/paper1/replay.json` | Paper-result checks |
| `work/paper1/main_results.csv` | Recomputed comparison rows |
| `work/paper1/power_grid.csv` | Recomputed power grid |
| `work/paper1/pool_ablation.csv` | Calibration-pool ablation |

For another run, repeat only the demo and paper-replay commands with unused output names such as `work/demo2` and `work/paper2`. The demo refuses to overwrite an existing directory. Its synthetic rates only demonstrate execution.

## 4. Audit your own API

Continue with the [audit walkthrough](docs/QUICKSTART.md). It uses Bash and starts after installation, without repeating the demo. Windows users can follow it in WSL using a separate Linux environment.

Prepare:

- A locally available checkpoint for the **claimed model**, with a compatible tokenizer and chat template.
- Separate calibration and test prompt sets. The walkthrough uses at least 400 distinct prompts per set and a batch budget of 200.
- Hardware that can load the reference model. The tested 27B checkpoint used about 50 GiB of GPU memory; requirements depend on the model.
- Your API endpoint, model identifier and credential for real API collection.

The workflow is:

```text
Generate legitimate calibration responses → score → calibrate a threshold
Collect API responses → score with the same reference model → audit
```

The guide includes configuration-file creation, credential setup, request preview, execution and result interpretation. Collection previews requests by default; `--execute` sends them. A substitute model is needed only for the guide's optional power experiment.

## Reference and verification

- [Data formats and numerical conventions](docs/SCHEMAS.md)
- [Methods, variants and baseline interfaces](docs/DESIGN_MAP.md)
- [Assets and reproduction coverage](docs/ASSETS.md)
- [Installation environments](docs/ENVIRONMENT.md)
- [Tests and extension instructions](CONTRIBUTING.md)

The [validation report](docs/VALIDATION_REPORT.md) records 78 passing tests with the model backend installed, 64 passing tests with 14 optional skips in the lightweight CPU environment, 2,020 paper-result checks, 1,280 original-function comparisons, and a local GPU smoke test.

## License and citation

Code and the included numerical statistics use [Apache-2.0](LICENSE), copyright TokenRhythm Technologies. Model weights and original prompt/response text are separate assets; their terms are described in [ASSETS.md](docs/ASSETS.md).

The paper is an unpublished manuscript without an arXiv identifier. [CITATION.cff](CITATION.cff) contains its citation metadata.

```bibtex
@unpublished{tokenrhythm2026apitail,
  author = {{TokenRhythm Technologies}},
  title = {{API-TAIL}: Detecting Model Substitution in {LLM APIs} through Temperature-Aligned Likelihood Tails},
  year = {2026},
  note = {Unpublished manuscript}
}
```
