#!/usr/bin/env python3
"""Compare the NumPy core with functions extracted unchanged from an archive.

This checks numerical statistics, not the full evaluator. The --trust-archive
flag is required because the original score caches use pickle."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np

PANELS = ("q38_p30", "q38_user", "g4_p30", "g4_user")
METHODS = (
    ("align", "T_align-fsv", "SF", "VF"),
    ("mle", "T_MLE-full", "SF", "VF"),
    ("fixed", "FDG-fsv", "SF", "VF"),
    ("huber", "T Huber-fsv", "SF", "VF"),
    ("nucleus", "N95-full id", "SNF", "VNF"),
)
SOURCE_FILES = {
    "engine": "project/code/jobs/supp1/engine.py",
    "temperature": "project/code/jobs/talign/ta_run_eval.py",
    "paper": "project/code/jobs/paper2/paper2_eval.py",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execute_nodes(nodes, namespace, filename):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)


def assigned_names(node):
    if not isinstance(node, ast.Assign):
        return set()
    names = set()
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(x.id for x in target.elts if isinstance(x, ast.Name))
    return names


def method_registration(node):
    if not isinstance(node, ast.Assign):
        return False
    for target in node.targets:
        if (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "EN"
                and target.value.attr == "METHODS"):
            return True
    return False


def extract_oracle(root, torch):
    texts = {name: (root / rel).read_text(encoding="utf-8") for name, rel in SOURCE_FILES.items()}
    trees = {name: ast.parse(text) for name, text in texts.items()}
    constants = {"TG", "NT", "NA", "NG", "NAG", "I10", "HUBER"}
    eng = {"np": np}
    execute_nodes([n for n in trees["engine"].body
                   if assigned_names(n) & constants], eng, SOURCE_FILES["engine"])
    EN = SimpleNamespace(**{name: eng[name] for name in constants}, METHODS={})
    provenance = []

    def selected(name, functions, assignments):
        picked = [
            n for n in trees[name].body
            if (isinstance(n, ast.FunctionDef) and n.name in functions)
            or bool(assigned_names(n) & assignments)
            or method_registration(n)
        ]
        for n in picked:
            if isinstance(n, ast.FunctionDef):
                segment = ast.get_source_segment(texts[name], n)
                provenance.append({
                    "source": SOURCE_FILES[name], "function": n.name,
                    "start_line": n.lineno, "end_line": n.end_lineno,
                    "function_sha256": hashlib.sha256(segment.encode()).hexdigest(),
                })
        return picked

    def unavailable(*args, **kwargs):
        raise RuntimeError("Unexpected legacy method outside extracted oracle scope")

    ns = {"np": np, "torch": torch, "EN": EN, "_stats0": unavailable}
    execute_nodes(selected("temperature", {"_is_sv", "_stats"}, {"SVFAM", "ZGRID"}),
                  ns, SOURCE_FILES["temperature"])
    EN.Panel = SimpleNamespace(stats=ns["_stats"])
    execute_nodes(selected("paper", {"stats"}, {"OWN_EST", "MET"}),
                  ns, SOURCE_FILES["paper"])
    return ns["stats"], EN.TG, provenance


def resolve_cache(design, role):
    pool = design["pools"]["S_pen"]
    if role == "cal":
        source = pool[0]
        dirs = design["svfull_dir"]
        source_index = 0
    else:
        source_index, source = next(
            (i, name) for i, name in enumerate(design["honest"]) if name not in pool
        )
        dirs = design.get("test_dirs", {}).get("svfull_dir", design["svfull_dir"])
    for directory in dirs:
        path = Path(directory) / (source + ".svfull.pkl")
        if path.is_file():
            return path, source_index
    raise RuntimeError("A selected archive cache is unavailable; check design cache directories")


def check_one(path, panel, role, source_index, max_rows, trials, budget, seed,
              oracle, grid, torch, batch_counts, work):
    with path.open("rb") as stream:
        data = pickle.load(stream)
    fields = data["fields"]
    total_rows = len(fields["SF0"])
    count = min(total_rows, max_rows)
    if count < budget:
        raise RuntimeError("Selected score bank is smaller than the audit budget")
    rng = np.random.default_rng(seed)
    subset = np.sort(rng.choice(total_rows, size=count, replace=False))
    arrays = {}
    for original, public in (("SF0", "SF"), ("VF0", "VF"),
                             ("SNF0", "SNF"), ("VNF0", "VNF")):
        value = np.asarray(fields[original])[subset]
        if value.shape != (count, len(grid)) or not np.isfinite(value).all():
            raise RuntimeError("Invalid selected moment array")
        arrays[public] = np.ascontiguousarray(value)
    # Preserve archive precision for the oracle; the public core uses float64.
    del fields, data
    draws = np.stack([rng.choice(count, size=budget, replace=False)
                      for _ in range(trials)]).astype(np.int64)
    sample = work / (panel + "_" + role + ".npz")
    np.savez_compressed(sample, **arrays, draws=draws, grid=grid)
    with np.load(sample, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in arrays}
        draws = fixture["draws"]
    pan = SimpleNamespace(
        device="cpu", grids={name: torch.from_numpy(value) for name, value in arrays.items()}
    )
    legacy = oracle(pan, draws, np.zeros(trials), [m[1] for m in METHODS])
    comparisons = []
    for method, legacy_name, skey, vkey in METHODS:
        core_method = "align" if method == "nucleus" else method
        K, T = batch_counts(arrays[skey], arrays[vkey], draws, grid, method=core_method)
        legacy_K = np.asarray(legacy[legacy_name])
        legacy_T = np.asarray(legacy[("T", legacy_name)])
        comparisons.append({
            "method": method, "trials": trials,
            "count_mismatches": int(np.count_nonzero(K != legacy_K)),
            "temperature_mismatches": int(np.count_nonzero(T != legacy_T)),
            "max_count_abs_diff": int(np.max(np.abs(K - legacy_K))),
            "max_temperature_abs_diff": float(np.max(np.abs(T - legacy_T))),
            "counts": K.tolist(), "temperatures": T.tolist(),
            "oracle_counts": legacy_K.tolist(), "oracle_temperatures": legacy_T.tolist(),
        })
    return {
        "panel": panel, "role": role,
        "design_source_selector": (
            "pools.S_pen[0]" if role == "cal" else "honest[" + str(source_index) + "]"
        ),
        "cache_sha256": sha256(path), "cache_bytes": path.stat().st_size,
        "total_rows": total_rows, "selected_rows": count,
        "subset_sha256": hashlib.sha256(subset.astype("<i8").tobytes()).hexdigest(),
        "draws_sha256": hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest(),
        "fixture": sample.name, "fixture_sha256": sha256(sample),
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "methods": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trust-archive", action="store_true",
                        help="acknowledge that archive Python and referenced pickle caches are trusted")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if not args.trust_archive:
        parser.error("--trust-archive is required: local archive pickle inputs can execute code")
    if min(args.rows, args.trials, args.budget) <= 0 or args.rows < args.budget:
        parser.error("rows and trials must be positive, with rows >= budget")
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from apitail.core import batch_counts, temperature_grid
    import apitail.core as core
    root = args.archive_root.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    oracle, grid, functions = extract_oracle(root, torch)
    if not np.array_equal(grid, temperature_grid()):
        raise RuntimeError("Original and portable temperature grids differ")
    cases = []
    designs = []
    for panel_index, panel in enumerate(PANELS):
        rel = "project/runs/paper2/designs/design_" + panel + ".json"
        dpath = root / rel
        design = json.loads(dpath.read_text(encoding="utf-8"))
        designs.append({"source": rel, "sha256": sha256(dpath)})
        for role_index, role in enumerate(("cal", "test")):
            cache, source_index = resolve_cache(design, role)
            cases.append(check_one(
                cache, panel, role, source_index, args.rows, args.trials, args.budget,
                args.seed + panel_index * 2 + role_index,
                oracle, grid, torch, batch_counts, args.work_dir
            ))
    methods = [method for case in cases for method in case["methods"]]
    count_bad = sum(m["count_mismatches"] for m in methods)
    temp_bad = sum(m["temperature_mismatches"] for m in methods)
    report = {
        "schema": "api-tail-archive-score-check-v1",
        "status": "passed" if not (count_bad or temp_bad) else "failed",
        "oracle": "unchanged AST-extracted original functions, CPU SimpleNamespace panel",
        "scope": "Saved HF S/V moments to K/T only; no prompt sampler, model scoring, API calls, or calibration rerun",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "torch": torch.__version__, "device": "cpu", "torch_threads": 1},
        "seed": args.seed, "budget": args.budget, "requested_rows": args.rows,
        "trials_per_case": args.trials,
        "cache_cases": len(cases), "method_cases": len(methods),
        "trial_method_comparisons": sum(m["trials"] for m in methods),
        "count_mismatches": count_bad, "temperature_mismatches": temp_bad,
        "sources": [{"source": rel, "sha256": sha256(root / rel)}
                    for rel in SOURCE_FILES.values()],
        "core": {"source": "src/apitail/core.py", "sha256": sha256(Path(core.__file__))},
        "script": {"source": "scripts/verify_archive_scores.py", "sha256": sha256(Path(__file__))},
        "extracted_functions": functions, "designs": designs, "cases": cases,
    }
    output = args.output or (args.work_dir / "archive_scores.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "cache_cases", "method_cases", "trial_method_comparisons",
        "count_mismatches", "temperature_mismatches")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

