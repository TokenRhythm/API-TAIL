"""Evaluate native-protocol MET with composite calibration.

Each prompt supplies ten API responses and ten local reference responses,
truncated or padded to 50 tokens. Even null trials calibrate and odd trials
estimate held-out false alarms. The cutoff is the maximum per-configuration
95th percentile with strict upper-tail rejection. Reference-only calibration
is reported separately."""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, str(_APITAIL_CODE / 'bbp'))
from cmp_met_native import load_ids, mmd  # noqa: E402

G = {}


def trial_set(args):
    """one condition: (kind, base, sub, rho, nreq, trials, seed, stream) -> MMD^2 per trial"""
    kind, base, sub, rho, nreq, trials, seed, stream = args
    rng = np.random.default_rng([seed, stream])
    npr = nreq // 10
    pids = G["pids"]
    out = np.empty(trials)
    for t in range(trials):
        X, Y = [], []
        for ci in rng.choice(len(pids), npr, replace=False):
            pid = pids[ci]
            ref = G["ref"][pid]
            sel = rng.choice(len(ref), 20 if kind == "metnull" else 10, replace=False)
            Y.append(ref[sel[:10]])
            if kind == "metnull":
                X.append(ref[sel[10:]])
                continue
            b = G["banks"][base][pid]
            s = G["banks"][sub][pid] if sub else None
            ob = rng.permutation(len(b))
            os_ = rng.permutation(len(s)) if s is not None else None
            ib = is_ = 0
            rows = []
            for _ in range(10):
                if s is not None and rng.random() < rho:
                    rows.append(s[os_[is_]]); is_ += 1
                else:
                    rows.append(b[ob[ib]]); ib += 1
            X.append(np.array(rows))
        out[t] = mmd(X, Y)
    return args, out


def qh(x):
    return float(np.quantile(x, 0.95, method="higher"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    M = json.load(open(a.manifest, encoding="utf-8"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M["tokenizer"])
    G["ref"] = load_ids(M["ref"], tok)
    cov = M.get("coverage", {})
    api = set(M["honest"]) | set(M["subs"])
    # API sources require a recorded coverage of at least 0.99.
    covered = {n for n in api if cov.get(n) is not None and cov[n] >= 0.99 and os.path.exists(M["sources"][n])}
    G["banks"] = {n: load_ids(p, tok) for n, p in M["sources"].items() if n not in api or n in covered}
    uncovered = sorted(api - covered)
    G["pids"] = sorted(p for p in G["ref"] if len(G["ref"][p]) >= 20 and
                       all(len(b.get(p, [])) >= 10 for b in G["banks"].values()))
    print("prompts with full banks", len(G["pids"]), "uncovered / missing API sources", uncovered, flush=True)
    members = list(dict.fromkeys(v for pool in M["pools"].values() for v in pool))
    honest = [h for h in M["honest"] if h in covered]
    plain = [h for h in M["plain"] if h in covered]
    jobs = []
    for nr in M["nreq"]:
        jobs += [("null", v, None, 0.0, nr, 2000, 11, r) for v in members for r in range(a.reps + 1)]
        jobs += [("null", h, None, 0.0, nr, 1000, 12, 0) for h in honest]
        jobs += [("alt", M["base"], s, rho, nr, 1000, 13, 0) for s in M["subs"] if s in G["banks"] for rho in M["rhos"]]
        jobs += [("metnull", None, None, 0.0, nr, 1000, 17, 0)]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = {args: v for args, v in pool.imap_unordered(trial_set, jobs, chunksize=1)}
    out = {"manifest": M, "n_prompts": len(G["pids"]), "uncovered_sources": uncovered, "plain_eff": plain, "res": {}}
    probes = {}
    for nr in M["nreq"]:
        R = out["res"].setdefault(str(nr), {})
        nat = qh(res[("metnull", None, None, 0.0, nr, 1000, 17, 0)])
        for arm, mem in M["pools"].items():
            cal = {v: res[("null", v, None, 0.0, nr, 2000, 11, 0)] for v in mem}
            thr = max(qh(cal[v][0::2]) for v in mem)
            dec = max(mem, key=lambda v: qh(cal[v][0::2]))
            held = {v: float(np.mean(cal[v][1::2] > thr)) for v in mem}
            ep = {h: float(np.mean(res[("null", h, None, 0.0, nr, 1000, 12, 0)] > thr)) for h in honest}
            pw = {f"{M['base']}|{s}|{rho}": float(np.mean(res[("alt", M["base"], s, rho, nr, 1000, 13, 0)] > thr))
                  for s in M["subs"] if s in G["banks"] for rho in M["rhos"]}
            reps = []
            for r in range(1, a.reps + 1):
                tr = max(qh(res[("null", v, None, 0.0, nr, 2000, 11, r)][0::2]) for v in mem)
                reps.append(max([float(np.mean(res[("null", h, None, 0.0, nr, 1000, 12, 0)] > tr)) for h in plain],
                                default=None))
            mx = max(ep[h] for h in plain) if plain else None
            keep = sum(v is not None and v <= 0.10 for v in reps)
            valid = None if mx is None else bool(mx <= 0.10 and keep >= 16)          # no plain source: unmeasured
            R[arm] = {"thr": thr, "decisive": dec, "heldout": held, "endpoint": ep, "power": pw, "plain_max": mx,
                      "redraws_plain_max": reps, "redraws_kept": keep, "valid": valid,
                      "native_thr": nat, "native_endpoint": {h: float(np.mean(res[("null", h, None, 0.0, nr, 1000, 12, 0)]
                                                                            > nat)) for h in honest}}
            if nr == 200 and arm == "S_main":
                prim = M["primary"]
                probes = {"MET native": {"q38_p30": {"valid": valid,
                                                     "power": pw.get(f"{prim[0]}|{prim[1]}|{prim[2]}")}}}
            print(f"N_req {nr} {arm}: thr {thr:.4f} ({dec}) plain max {mx} kept {keep}/{a.reps} valid {valid} | "
                  + " ".join(f"{k.split('|')[1][:14]}@{k.split('|')[2]}={v:.3f}" for k, v in pw.items()), flush=True)
    json.dump(out, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    old = json.load(open(a.probes, encoding="utf-8")) if os.path.exists(a.probes) else {}
    old.update(probes)
    json.dump(old, open(a.probes, "w", encoding="utf-8"), indent=1)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
