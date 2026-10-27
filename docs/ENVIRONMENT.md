# Tested environments

## CPU

API-TAIL 0.1.1 is tested in a Python 3.13.9 virtual environment without system site-packages. It uses NumPy 2.2.6 and SciPy 1.16.3. The wheel is installed and tested outside the source checkout, and `pip check` passes. Tests that require the optional Torch backend are skipped in this environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu-tested.txt
python -m pip install -e ".[test]"
python -m pip check
python -m unittest discover -s tests -v
```

The pinned versions were tested on Python 3.13.9. For other supported Python versions, use the dependency ranges in pyproject.toml.

## GPU

| Component | Tested version |
|---|---|
| Python | 3.13.9 |
| Torch | 2.6.0+cu124 |
| Transformers | 5.5.0 |
| NumPy | 2.2.6 |
| SciPy | 1.16.3 |
| Accelerate | 1.13.0 |
| Safetensors | 0.7.0 |
| GPU | NVIDIA L20X |
| Model | Qwen3.8-27B |

The GPU checks use an existing environment. Its package metadata has five dependency conflicts:

```text
streamlit 1.51.0 requires pyarrow<22,>=7.0; installed pyarrow 23.0.1
vllm 0.18.0 requires torch==2.10.0; installed torch 2.6.0
vllm 0.18.0 requires torchvision==0.25.0; installed torchvision 0.21.0
vllm 0.18.0 requires transformers<5,>=4.56.0; installed transformers 5.5.0
torchaudio 2.10.0 requires torch==2.10.0; installed torch 2.6.0
```

The tested HF generation/scoring path does not use those conflicting packages. `requirements/gpu-tested.txt` lists the tested top-level versions; it is not a complete transitive dependency lock.

For a new GPU environment, install a driver-compatible PyTorch build and the scoring dependencies:

```bash
python -m pip install -e ".[score]"
apitail doctor --gpu
python scripts/gpu_smoke.py --help
```

The smoke test uses local weights, records software versions and source hashes, and checks repeatability on the tested hardware.

## Platform and service coverage

The CI configuration includes Linux and Windows CPU jobs. Native Windows and GitHub-hosted CI were not executed in this validation. Git checkout with CRLF conversion and a GBK default-encoding regression are covered separately; see [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

API collection is tested with mocks. Real collection requires an endpoint and credential supplied by the user.
