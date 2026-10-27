"""Compare score arrays, repeatability and downstream audit decisions.

Commands:
  hfvllm <first.svfull.pkl> <second.svfull.pkl> <n> <out.json>
  selfrepeat <manifest.json> <out.json>
  scoring <manifest.json> <out.json>

Manifests specify the required scorers, cache files and evaluation outputs.
Missing inputs remain unavailable. Comparisons preserve response keys, trial
identities and the fitted temperature of each audit."""
import json
import os
import pickle
import sys

import numpy as np

TG = np.round(np.arange(0.30, 1.81, 0.01), 2)
FIELDS = ("SF0", "VF0", "SNF0", "VNF0")
FAMILY = {"T_align-fsv": "F", "T_MLE-full": "F", "T Huber-fsv": "F", "FDG-fsv": "F", "N95-full id": "NF",
          "N95-full MLE": "NF"}
DET = ("NLL count", "log-rank count", "entropy count")


def zf(d, fam="F"):
    S, V = (d["fields"]["SF0"], d["fields"]["VF0"]) if fam == "F" else (d["fields"]["SNF0"], d["fields"]["VNF0"])
    return S / np.sqrt(np.clip(V, 1e-12, None))


def hfvllm(a, b, n, out):
    A, B = pickle.load(open(a, "rb")), pickle.load(open(b, "rb"))
    pa = {tuple(k): i for i, k in enumerate(A["keys"])}
    common = [i for i, k in enumerate(B["keys"]) if tuple(k) in pa]
    rng = np.random.default_rng(20261025)
    sel = sorted(rng.choice(len(common), min(n, len(common)), replace=False))
    ib = np.array([common[j] for j in sel])
    ia = np.array([pa[tuple(B["keys"][j])] for j in ib])
    za, zb = zf(A)[ia], zf(B)[ib]
    win = (TG >= 0.40 - 1e-9) & (TG <= 1.20 + 1e-9)
    d = np.abs(za - zb)[:, win]
    tk = int(np.argmin(np.abs(zf(A).sum(0))))
    dt = np.abs(za[:, tk] - zb[:, tk])
    res = {"hf": a, "vllm": b, "responses": int(len(set(map(tuple, (A["keys"][i] for i in ia))))),
           "registered_responses": n, "grid": "0.40-1.20", "median": float(np.median(d)),
           "q90": float(np.quantile(d, 0.9)), "max": float(d.max()), "T_hat": float(TG[tk]),
           "at_T_hat": {"median": float(np.median(dt)), "q90": float(np.quantile(dt, 0.9)), "max": float(dt.max())}}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res))


def selfrepeat(manifest, out):
    M = json.load(open(manifest, encoding="utf-8"))
    req = int(M["required_per_panel"])
    panels_req = M.get("panels", ["q38_p30", "q38_user", "g4_p30", "g4_user"])
    rows, per_panel, keys_ok, uniq = [], {}, True, {}
    for e in M["entries"]:
        want = [tuple(k) for k in e["keys"]]
        A, B = pickle.load(open(e["first"], "rb")), pickle.load(open(e["repeat"], "rb"))
        pa = {tuple(k): i for i, k in enumerate(A["keys"])}
        pb = {tuple(k): i for i, k in enumerate(B["keys"])}
        ok = sorted(set(k for k in want if k in pa and k in pb))
        miss = len(set(want)) - len(ok)
        ia, ib = [pa[k] for k in ok], [pb[k] for k in ok]
        eq = {f: (float(np.mean(np.all(np.ascontiguousarray(A["fields"][f][ia]).view(np.uint8).reshape(len(ia), -1)
                                         == np.ascontiguousarray(B["fields"][f][ib]).view(np.uint8).reshape(len(ib), -1),
                                         axis=1)))
                  if ok and A["fields"][f].dtype == B["fields"][f].dtype else (0.0 if ok else None)) for f in FIELDS}
        keys_ok = keys_ok and miss == 0
        uniq.setdefault(e["panel"], set()).update((e["name"],) + tuple(k) for k in ok)
        per_panel[e["panel"]] = len(uniq[e["panel"]])
        rows.append({"panel": e["panel"], "name": e["name"], "selected": len(set(want)), "compared": len(ok),
                     "missing": miss, "bitwise_equal_share": eq})
    vals = [v for r in rows for v in r["bitwise_equal_share"].values() if v is not None]
    size_ok = all(per_panel.get(p_, 0) >= req for p_ in panels_req)
    res = {"pairs": rows, "per_panel_compared": per_panel, "required_per_panel": req, "panels_required": panels_req,
           "responses": sum(per_panel.values()), "count_unit": "unique (source, pid, k) per panel",
           "all_selected_present": keys_ok, "registered_size_reached": size_ok, "complete": bool(keys_ok and size_ok),
           "bitwise_equal_share_min": min(vals) if vals else None}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}))


def load_sv(dirs, name):
    for d in dirs:
        p = os.path.join(d, f"{name}.svfull.pkl")
        if os.path.exists(p):
            return pickle.load(open(p, "rb"))
    return None


def tri(flags, required):
    """True on an observed disagreement; False only if every required scorer's flag is present and all agree"""
    present = {k: v for k, v in flags.items() if v is not None}
    if len(set(present.values())) > 1:
        return True
    return False if all(r in present for r in required) else None


def scoring(manifest, out):
    M = json.load(open(manifest, encoding="utf-8"))
    ev = {k: json.load(open(p + ".json", encoding="utf-8")) for k, p in M["evals"].items()}
    nz = {k: np.load(p + ".npz") for k, p in M["evals"].items()}
    ref = "hf"
    req = M.get("required", list(ev))
    res = {"scorers": list(ev), "required": req, "matrix": {}, "K_agreement": {}, "z_at_fitted_T": {}}
    # Threshold and decision comparisons.
    for n, byarm in ev[ref]["res"].items():
        for arm in [k for k in byarm if not k.startswith("_")]:
            for m in M["methods"]:
                cell = {}
                for sc, e in ev.items():
                    x = ((e["res"].get(n) or {}).get(arm) or {}).get(m)
                    cell[sc] = None if x is None else {"thr": x["thr"], "exact": [x["exact"]["k"], x["exact"]["gamma"]],
                                                       "flags": {k: v["valid"] for k, v in x["verdict"].items()}}
                keys = sorted({k for c in cell.values() if c for k in c["flags"]})
                differs = {k: tri({sc: (c["flags"].get(k) if c else None) for sc, c in cell.items()}, req) for k in keys}
                res["matrix"][f"N{n}|{arm}|{m}"] = {"by_scorer": cell, "flag_differs": differs}
    # Compare trial counts at budget 200; detector thresholds depend on the pool.
    for sc in ev:
        if sc == ref:
            continue
        by = {}
        for k in nz[ref].files:
            parts = k.split("##")
            if parts[0] != "N200" or len(parts) < 3 or parts[-1] in ("rows", "tk") or k not in nz[sc].files:
                continue
            m = parts[2]
            if m not in M["methods"]:
                continue
            pool = parts[3] if (m in DET and len(parts) > 3) else "pool-independent"
            by.setdefault(f"{m}|{pool}", []).append((float(np.mean(nz[ref][k] == nz[sc][k])), parts[1]))
        res["K_agreement"][sc] = {mp: {"median": float(np.median([a for a, _ in v])), "min": float(min(v)[0]),
                                       "worst_condition": min(v)[1], "conditions": len(v)} for mp, v in by.items()}
    # Response scores at each trial's fitted temperature.
    rk = json.load(open(M["evals"][ref] + ".rowkey.json", encoding="utf-8"))
    names = sorted({r[0] for r in rk})
    if M.get("svfull_files"):
        sv = {sc: {nm_: (pickle.load(open(fm[nm_], "rb")) if fm.get(nm_) and os.path.exists(fm[nm_]) else None)
                   for nm_ in names} for sc, fm in M["svfull_files"].items()}
    else:
        sv = {sc: {nm_: load_sv(M["svfull"][sc], nm_) for nm_ in names} for sc in M.get("svfull", {})}
    res["svfull_missing"] = {sc: [nm_ for nm_, v in d.items() if v is None] for sc, d in sv.items()}
    idx = {}
    for sc, d in sv.items():
        pos = {nm_: ({tuple(k): i for i, k in enumerate(v["keys"])} if v is not None else {}) for nm_, v in d.items()}
        idx[sc] = np.array([pos[r[0]].get((r[1], r[2]), -1) for r in rk], np.int64)
    zt = {}
    for sc in sv:
        for fam in ("F", "NF"):
            Z = np.full((len(rk), len(TG)), np.nan)
            for nm_ in names:
                d = sv[sc][nm_]
                if d is None:
                    continue
                rows = [i for i, r in enumerate(rk) if r[0] == nm_ and idx[sc][i] >= 0]
                Z[rows] = zf(d, fam)[idx[sc][rows]]
            zt[(sc, fam)] = Z
    for k in nz[ref].files:
        parts = k.split("##")
        if parts[0] != "N200" or parts[-1] != "tk":
            continue
        cond, m = parts[1], parts[2]
        rows_k = f"N200##{cond}##rows"
        if rows_k not in nz[ref].files or m not in FAMILY or m not in M["methods"] or ("hf", FAMILY[m]) not in zt:
            continue
        rows = nz[ref][rows_k]
        tk = nz[ref][k].astype(np.int64)
        fam = FAMILY[m]
        zh = zt[("hf", fam)][rows, tk[:, None]]
        fin = np.isfinite(zh)
        rec = {"near3_share_hf": float(np.mean(np.abs(zh[fin] - 3) < 0.1)) if fin.any() else None,
               "hf_scored": int(fin.sum()), "expected": int(zh.size), "trials": int(len(tk)),
               "T_fitted_mean": float(TG[tk].mean())}
        for sc in sv:
            if sc == "hf":
                continue
            d = np.abs(zh - zt[(sc, fam)][rows, tk[:, None]])
            d = d[~np.isnan(d)]
            rec[sc] = None if not len(d) else {"median": float(np.median(d)), "q90": float(np.quantile(d, 0.9)),
                                               "max": float(d.max()), "responses": int(len(d))}
        res["z_at_fitted_T"][f"{cond}|{m}"] = rec
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    nflip = sum(1 for v in res["matrix"].values() if any(f is True for f in v["flag_differs"].values()))
    print(json.dumps({"matrix_cells": len(res["matrix"]), "cells_with_flag_difference": nflip,
                      "z_records": len(res["z_at_fitted_T"])}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "hfvllm":
        hfvllm(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif cmd == "selfrepeat":
        selfrepeat(sys.argv[2], sys.argv[3])
    elif cmd == "scoring":
        scoring(sys.argv[2], sys.argv[3])
