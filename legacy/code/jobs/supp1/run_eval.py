"""Evaluate requested methods on shared draws and calibrate each named pool.

Saves per-condition counts and RUT p-values in NPZ files so calibration pools
and endpoint lists can be reevaluated without recomputing GPU statistics."""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(_APITAIL_CODE / 'jobs/supp1'))
import engine as EN  # noqa: E402
from pool_swap import NEW_PLAIN, OLD, OLD_PLAIN, ADDED, PEN  # noqa: E402

CELLS = {"p30": ["self|qwen3.6-27b__provider_B|random_0.25", "self|qwen3.8-flash__provider_B|random_0.25",
                 "self|deepseek-v4.1-flash__provider_A_fp8|random_0.25", "self|qwen3.6-27b__provider_A_fp8|random_0.25",
                 "self|qwen3.6-27b__provider_B|random_0.1", "self|qwen3.8-flash__provider_B|random_0.1",
                 "self|deepseek-v4.1-flash__provider_A_fp8|random_0.1"],
         "user": ["self_u|qwen3.6-27b__provider_B|random_0.25", "self_u|qwen3.8-flash__provider_B|random_0.25",
                  "self_u|deepseek-v4.1-flash__provider_A_fp8|random_0.05", "self_u|qwen3.6-27b__provider_A_fp8|random_0.25",
                  "self_u|qwen3.6-27b__provider_B|random_0.5", "self_u|qwen3.6-27b__provider_B|random_1.0",
                  "self_u|deepseek-v4.1-flash__provider_A_fp8|random_0.1"],
         "glm": ["self|glm-5.2__z-ai_fp8|random_0.1", "self|deepseek-v4.1-flash__provider_A_fp8|random_0.1",
                 "self|glm-5.2__z-ai_fp8|random_0.25", "self|deepseek-v4.1-flash__provider_A_fp8|random_0.25"]}
DEFAULT_METHODS = ["ProTA-N95", "ProTA", "ProTA-W", "T_align", "T+pen (no Huber)", "FDG T=1", "N95 T-only id",
                   "N95 T-only Huber", "N95 pen id", "NLL count", "log-rank count", "entropy count", "RUT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--gridcache", required=True)
    ap.add_argument("--panel", required=True, choices=["p30", "user", "glm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="200")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--cells", default="")
    ap.add_argument("--trials_null", type=int, default=4000)
    ap.add_argument("--trials_alt", type=int, default=1000)
    ap.add_argument("--rep_cal", type=int, default=0)
    ap.add_argument("--rep_test", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reverse", type=int, default=0, help="1: compute methods in reverse order (invariance check)")
    ap.add_argument("--honest", default="all", help="all | none | comma list")
    ap.add_argument("--plist", default="", help="file with a prompt list (e.g. one half) to restrict all draws to")
    ap.add_argument("--no_npz", type=int, default=0)
    ap.add_argument("--fullvocab", default="", help="dir of score_full.py outputs (needed by the *-full / w20r / w100 methods)")
    a = ap.parse_args()
    t0 = time.time()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    methods = a.methods.split(",")
    if a.reverse:
        methods = methods[::-1]
    cells = a.cells.split(";") if a.cells else [c for c in CELLS[a.panel]
                                                if all(x in cfg["sources"] for x in c.split("|")[:2])]
    calib_all = list(dict.fromkeys(OLD[a.panel] + ADDED[a.panel] + PEN[a.panel]))
    honest = [h for h in cfg["honest"] if h not in calib_all] if a.honest == "all" else \
        ([] if a.honest == "none" else a.honest.split(","))
    names = list(dict.fromkeys([v for v in calib_all if v in cfg["sources"]] + honest +
                               [x for c in cells for x in c.split("|")[:2]]))
    grids = sorted({EN.METHODS[m][1] for m in methods if EN.METHODS[m][0] in ("prof", "fixed", "cross")} |
                   {EN.METHODS[m][4] for m in methods if EN.METHODS[m][0] == "cross"} |
                   {g for m in methods if EN.METHODS[m][0] in ("prof", "cross") and EN.METHODS[m][2]
                    for g in EN.PEN_OF[EN.METHODS[m][1]]})
    pan = EN.Panel(cfg, a.gridcache, names, grids=tuple(grids), device=a.device, rut=True,   # Reference requirements also determine prompt eligibility.
                   fullvocab=a.fullvocab or None)
    
    plist = [l.strip() for l in open(a.plist, encoding="utf-8")] if a.plist else None
    conds = []
    for n in [int(x) for x in a.ns.split(",")]:
        conds += [("null", v, None, None, n) for v in calib_all if v in cfg["sources"]]
        conds += [("null", v, None, None, n) for v in honest]
        conds += [("alt",) + tuple(c.split("|")) + (n,) for c in cells]
    store = {}
    skipped = []
    timing = {}
    for c in conds:
        tr = a.trials_null if c[0] == "null" else a.trials_alt
        pool_variant = c[0] == "null" and c[1] in calib_all
        d = pan.draw(c, tr, rep_cal=a.rep_cal, rep_test=a.rep_test, split_even=pool_variant, plist=plist)
        if d is None:
            skipped.append(EN.cond_name(c))
            continue
        rows, uni = d
        t1 = time.time()
        st = pan.stats(rows, uni, methods)
        timing[EN.cond_name(c)] = time.time() - t1
        store[EN.cond_name(c)] = {"rows": rows.astype(np.int32), **st}
    print(f"stats for {len(store)} conditions in {time.time() - t0:.0f}s (skipped {len(skipped)}); "
          f"mean per condition {np.mean(list(timing.values())):.2f}s", flush=True)
    
    npz = {}
    for cn, st in store.items():
        npz[f"{cn}##rows"] = st["rows"]
        for m in methods:
            v = st[m]
            if EN.METHODS[m][0] == "rut":
                npz[f"{cn}##{m}##p"] = v["p"]
            elif EN.METHODS[m][0] != "det":
                npz[f"{cn}##{m}"] = v.astype(np.int16)
                if ("theta", m) in st:
                    npz[f"{cn}##{m}##theta"] = st[("theta", m)].astype(np.int16)
    npz["__det__"] = pan.det.cpu().numpy()
    if not a.no_npz:
        np.savez_compressed(a.out + ".npz", **npz)
    
    pools = {"new": OLD[a.panel] + ADDED[a.panel], "old": OLD[a.panel]}
    if PEN[a.panel]:
        pools["new+pen"] = OLD[a.panel] + ADDED[a.panel] + PEN[a.panel]
    pools = {k: [v for v in p if v in cfg["sources"]] for k, p in pools.items()}
    res = {"cfg": a.cfg, "methods": methods, "cells": cells, "trials": [a.trials_null, a.trials_alt],
           "rep": [a.rep_cal, a.rep_test], "skipped": skipped, "pools": pools, "eval": {}}
    for n in sorted({c[-1] for c in conds}):
        for m in methods:
            raw = {}
            for cn, st in store.items():
                if not cn.endswith(f"|{n}"):
                    continue
                raw[cn] = st[m]
            for pn, pool in pools.items():
                hon = [h for h in calib_all + honest if h not in pool]
                tab = EN.evaluate(m, raw, pool, hon, cells, NEW_PLAIN[a.panel], n)
                for lname, plain in (("old", OLD_PLAIN[a.panel]),):
                    miss = [e for e in plain if e not in tab["endpoint"]]
                    mp = max([tab["endpoint"][e] for e in plain if e in tab["endpoint"]], default=0.0)
                    tab[f"valid_{lname}_list"] = None if miss else bool(
                        tab["gate"]["max_heldout"] <= 0.07 and tab["gate"]["max_lovo"] <= 0.10 and mp <= 0.10)
                res["eval"].setdefault(str(n), {}).setdefault(pn, {})[m] = tab
    json.dump(res, open(a.out + ".json", "w", encoding="utf-8"))
    for n, byp in res["eval"].items():
        for m, t in byp["new"].items():
            g = t["gate"]
            print(f"N{n} {m:18s} valid {int(t['valid'])} ho {g['max_heldout']:.3f} lovo {g['max_lovo']:.3f} plain "
                  f"{(g['max_plain_endpoint'] or 0):.3f} ({g['argmax_plain']}) thr {t['thr']:.1f} [{t['decisive']}] | "
                  + " ".join(f"{k.split('|')[1][:14]}@{k.split('|')[2][-4:]}={v:.3f}" for k, v in t["power"].items()),
                  flush=True)
    print(f"saved {a.out}.json/.npz {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
