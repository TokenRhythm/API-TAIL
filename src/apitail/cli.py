import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from . import __version__


def _positive(text):
    n = int(text)
    if n <= 0: raise argparse.ArgumentTypeError("must be positive")
    return n


def _emit(result):
    omit = {"z", "counts", "fitted_temperatures", "calibration_prompt_ids"}
    print(json.dumps({k: v for k, v in result.items() if k not in omit}, indent=2, allow_nan=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="apitail", description="Temperature-aligned likelihood-tail API audits")
    p.add_argument("--version", action="version", version=f"apitail {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor", help="Inspect installed packages; --gpu additionally checks CUDA")
    d.add_argument("--gpu", action="store_true")
    demo = sub.add_parser("demo", help="Synthetic offline end-to-end demonstration")
    demo.add_argument("--output", required=True)
    for name in ("score", "generate"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True)
        q.add_argument("--prompts", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
        q.add_argument("--chat-kwargs", default='{"enable_thinking": false}')
        q.add_argument("--local-files-only", action="store_true")
        q.add_argument("--limit", type=_positive)
        if name == "score":
            q.add_argument("--responses", required=True)
            q.add_argument("--max-length", type=_positive, default=4096)
            q.add_argument("--system", default="")
            q.add_argument("--nucleus", action="store_true")
            q.add_argument("--token-chunk", type=_positive, default=64)
        else:
            q.add_argument("--config", required=True)
    c = sub.add_parser("collect", help="Plan or execute OpenAI-compatible requests (default: dry-run)")
    c.add_argument("--config", required=True)
    c.add_argument("--prompts", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--limit", type=_positive)
    c.add_argument("--execute", action="store_true")
    c = sub.add_parser("calibrate")
    c.add_argument("--config", required=True)
    c.add_argument("--output", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--scores", required=True)
    a.add_argument("--calibration", required=True)
    a.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--scores", required=True, help="Honest endpoint scores, or base scores for a mixture")
    e.add_argument("--substitute")
    e.add_argument("--rho", type=float, default=.1)
    e.add_argument("--calibration", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--trials", type=_positive, default=1000)
    e.add_argument("--seed", type=int, default=1234)
    a = p.parse_args(argv)
    try:
        if a.command == "doctor":
            result = {"python": platform.python_version(), "platform": platform.system(), "packages": {}}
            for name in ("api-tail", "numpy", "scipy", "torch", "transformers", "accelerate"):
                try: result["packages"][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: result["packages"][name] = None
            if a.gpu:
                import torch
                result["cuda_available"] = torch.cuda.is_available()
                result["cuda_version"] = torch.version.cuda
                result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        elif a.command == "demo":
            from .demo import demo
            result = demo(a.output)
        elif a.command == "score":
            from .hf import score
            result = score(a.model, a.prompts, a.responses, a.output, a.device, a.dtype,
                           a.max_length, a.system, json.loads(a.chat_kwargs), a.nucleus,
                           a.token_chunk, a.limit, a.local_files_only)
        elif a.command == "generate":
            from .generate import generate
            result = generate(a.model, a.prompts, a.config, a.output, a.device, a.dtype,
                              json.loads(a.chat_kwargs), a.local_files_only, a.limit)
        elif a.command == "collect":
            from .collect import collect
            result = collect(a.config, a.prompts, a.output, a.execute, a.limit)
        elif a.command == "calibrate":
            from .pipeline import calibrate
            result = calibrate(a.config, a.output)
        elif a.command == "audit":
            from .pipeline import audit
            result = audit(a.scores, a.calibration, a.output)
        else:
            from .pipeline import evaluate
            result = evaluate(a.scores, a.calibration, a.output, a.trials, a.seed, a.substitute, a.rho)
        _emit(result)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as e:
        p.exit(2, f"apitail: {e}\n")
