#!/usr/bin/env python3
"""Run local generation and two independent scoring passes on one GPU.

Uses three synthetic prompts with at most eight new tokens each. The work
directory must be outside the repository. An optional reference scorer must
be a trusted Python source file exposing stats(logits, targets)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(gpu, work):
    result = {"free_disk_bytes": shutil.disk_usage(work).free,
              "kubectl_available": shutil.which("kubectl") is not None}
    proc = subprocess.run(["nvidia-smi", "-i", str(gpu),
                           "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if proc.returncode:
        raise RuntimeError("nvidia-smi could not query the selected GPU")
    name, used, total, utilization = [x.strip() for x in proc.stdout.strip().split(",")]
    result.update(gpu_name=name, memory_used_mib=int(used), memory_total_mib=int(total),
                  utilization_percent=int(utilization))
    procs = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    if procs.returncode:
        raise RuntimeError("nvidia-smi could not check compute processes")
    result["compute_process_count"] = len([x for x in procs.stdout.splitlines() if x.strip().isdigit()])
    if result["kubectl_available"]:
        try:
            pods = subprocess.run(["kubectl", "get", "pods", "--request-timeout=5s"],
                                  capture_output=True, text=True, timeout=10)
            result["pod_query_succeeded"] = pods.returncode == 0
        except subprocess.TimeoutExpired:
            result["pod_query_succeeded"] = False
    else:
        result["pod_query_succeeded"] = None
    return result


def load_reference(path):
    spec = importlib.util.spec_from_file_location("apitail_original_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import reference scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "stats", None)):
        raise ValueError("Reference scorer has no callable stats function")
    return module


def worker(args):
    import torch
    import transformers
    from apitail import hf
    from apitail.generate import generate
    from apitail.io import file_sha256, read_jsonl

    work = Path(args.work_dir).resolve()
    prompt_path = ROOT / "examples" / "prompts.jsonl"
    if len(read_jsonl(prompt_path)) != 3:
        raise ValueError("This smoke protocol expects exactly three synthetic prompts")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke check requires a CUDA device")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    comparison = {"performed": False, "responses_checked": 0, "fields": {}}
    original = hf.moments_torch
    reference = load_reference(args.reference_scorer) if args.reference_scorer and args.stage == "score1" else None
    if reference:
        comparison.update(performed=True, source_sha256=file_sha256(args.reference_scorer),
                          atol=1e-6, rtol=1e-6)
        names = ["S", "V", "S_nucleus", "V_nucleus", "n_nucleus", "nucleus_overflow", "out20"]
        def compare(logits, targets, grid, nucleus=False, token_chunk=64):
            actual = original(logits, targets, grid, nucleus=nucleus, token_chunk=token_chunk)
            expected = reference.stats(logits, targets)
            if not np.array_equal(np.asarray(reference.TG), np.asarray(grid)):
                raise ValueError("Reference scorer temperature grid differs")
            for name, tensor in zip(names, expected):
                value = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
                got = np.asarray(actual[name])
                difference = np.abs(got - value)
                absolute = float(np.max(difference))
                relative = float(np.max(difference / np.maximum(np.abs(value), 1e-12)))
                record = comparison["fields"].setdefault(name, {"max_abs_error": 0.0, "max_rel_error": 0.0, "passed": True})
                record["max_abs_error"] = max(record["max_abs_error"], absolute)
                record["max_rel_error"] = max(record["max_rel_error"], relative)
                record["passed"] = record["passed"] and bool(np.allclose(got, value, atol=1e-6, rtol=1e-6))
                if not record["passed"]:
                    raise AssertionError(f"Reference mismatch in {name}")
            comparison["responses_checked"] += 1
            return actual
        hf.moments_torch = compare
    try:
        if args.stage == "generate":
            cfg = {"seed": 71, "responses_per_prompt": 1, "max_length": 512,
                   "sampling": {"temperature": 0.5, "top_p": 1.0, "top_k": 0,
                                "max_tokens": 8, "repetition_penalty": 1.0, "presence_penalty": 0.0}}
            save_json(work / "generation.json", cfg)
            outcome = generate(args.model, prompt_path, work / "generation.json", work / "responses.jsonl",
                               device="cuda:0", dtype="bfloat16", chat_kwargs={"enable_thinking": False},
                               local_files_only=True)
            rows = read_jsonl(work / "responses.jsonl")
            generated_tokens = [int(row["n_tokens"]) for row in rows]
            if len(rows) != 3 or not all(0 < n <= 8 for n in generated_tokens):
                raise AssertionError("Unexpected generated response count or token length")
            outcome.update(generated_token_count=sum(generated_tokens), generated_tokens_per_response=generated_tokens)
        else:
            outcome = hf.score(args.model, prompt_path, work / "responses.jsonl", work / (args.stage + ".npz"),
                               device="cuda:0", dtype="bfloat16", max_length=512,
                               chat_kwargs={"enable_thinking": False}, nucleus=True, token_chunk=64,
                               local_files_only=True)
        torch.cuda.synchronize()
        outcome.pop("output", None)
        outcome.update(stage=args.stage, elapsed_seconds=time.monotonic() - started,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       gpu_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                       transformers_version=transformers.__version__, numpy_version=np.__version__,
                       reference_comparison=comparison)
        save_json(work / (args.stage + ".json"), outcome)
    finally:
        hf.moments_torch = original
        gc.collect()
        torch.cuda.empty_cache()


def run(args):
    from apitail.core import batch_statistic
    from apitail.io import load_scores, file_sha256

    model, work = Path(args.model).resolve(), Path(args.work_dir).resolve()
    if not model.is_dir():
        raise ValueError("--model must be an existing local model directory")
    if work == ROOT or ROOT in work.parents:
        raise ValueError("--work-dir must be outside the release repository")
    work.mkdir(parents=True, exist_ok=True)
    if any((work / name).exists() for name in ["responses.jsonl", "responses.jsonl.partial", "score1.npz", "score2.npz"]):
        raise FileExistsError("Use a fresh work directory; previous smoke outputs exist")
    report_path = Path(args.report).resolve() if args.report else work / "gpu_smoke.json"
    report = {"schema": "apitail-gpu-smoke-v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "model_name": model.name,
              "local_files_only": True, "paid_api_calls": 0, "model_downloads": 0,
              "synthetic_prompts": 3, "max_new_tokens": 8, "temperature_grid_size": 151,
              "independent_scoring_processes": 2,
              "scope": "Small local generation/scoring acceptance check; not a reproduction of paper-scale power/FAR or a cross-hardware determinism claim.",
              "source_sha256": {name: file_sha256(ROOT / "src" / "apitail" / name)
                                for name in ["core.py", "hf.py", "generate.py", "io.py"]},
              "prompts_sha256": file_sha256(ROOT / "examples" / "prompts.jsonl")}
    started = time.monotonic()
    report["before"] = snapshot(args.gpu, work)
    if report["before"]["compute_process_count"]:
        raise RuntimeError("The selected GPU already has compute processes; no job was launched")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
    report["stages"] = []
    current_stage = None
    try:
        for stage in ["generate", "score1", "score2"]:
            current_stage = stage
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--stage", stage,
                       "--model", str(model), "--work-dir", str(work)]
            if args.reference_scorer:
                command.extend(["--reference-scorer", str(Path(args.reference_scorer).resolve())])
            print(f"Starting {stage}", flush=True)
            with open(work / (stage + ".log"), "w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            report["stages"].append(json.loads((work / (stage + ".json")).read_text(encoding="utf-8")))
            print(f"Completed {stage}", flush=True)
        first, second = load_scores(work / "score1.npz"), load_scores(work / "score2.npz")
        arrays1 = {name: getattr(first, name) for name in ["S", "V", "grid", "pid", "k", "n_tokens"]}
        arrays2 = {name: getattr(second, name) for name in arrays1}
        arrays1.update(first.extra)
        arrays2.update(second.extra)
        if arrays1.keys() != arrays2.keys():
            raise AssertionError("Scorer outputs have different fields")
        bitwise = {name: bool(arrays1[name].dtype == arrays2[name].dtype and
                             arrays1[name].shape == arrays2[name].shape and
                             arrays1[name].tobytes() == arrays2[name].tobytes()) for name in arrays1}
        report["bitwise_repeat_fields"] = bitwise
        report["bitwise_repeat_passed"] = all(bitwise.values())
        if not report["bitwise_repeat_passed"]:
            raise AssertionError("Independent score runs were not bitwise identical")
        report["audit_statistic"] = batch_statistic(first.S, first.V, first.grid)
        report["actual_scored_tokens"] = int(first.n_tokens.sum())
        report["actual_scored_responses"] = len(first.pid)
        report["peak_allocated_bytes"] = max(x["peak_allocated_bytes"] for x in report["stages"])
        report["peak_reserved_bytes"] = max(x["peak_reserved_bytes"] for x in report["stages"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", failed_stage=current_stage, exception_type=type(exc).__name__)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        report["after"] = snapshot(args.gpu, work)
        report["gpu_compute_processes_released"] = report["after"]["compute_process_count"] == 0
        save_json(report_path, report)
    print(json.dumps({"status": report["status"], "bitwise_repeat_passed": report["bitwise_repeat_passed"],
                      "scored_tokens": report["actual_scored_tokens"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local model directory")
    parser.add_argument("--work-dir", required=True, help="Fresh scratch directory outside the repository")
    parser.add_argument("--reference-scorer", help="Trusted original scorer exposing stats(logits, targets)")
    parser.add_argument("--report", help="Public summary JSON; defaults to work-dir/gpu_smoke.json")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index; default 0")
    parser.add_argument("--stage", choices=["generate", "score1", "score2"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.stage:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
