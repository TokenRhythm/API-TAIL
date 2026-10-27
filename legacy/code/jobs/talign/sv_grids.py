"""Compute temperature-only S/V arrays from stored top-20 token scores.

Window scores normalize over all retained tokens. Nucleus scores renormalize
the 0.95 support and include only positions whose selected token lies inside.
Variances are floored at 1e-6. The script checks S/sqrt(V) against cached grids
and stops if the maximum difference exceeds 2e-3.

Each source cache stores response keys, S/V grids, retained-token counts and
response lengths."""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import json
import sys

import numpy as np

sys.path.insert(0, str(_APITAIL_CODE / 'bbp'))
TG = np.round(np.arange(0.30, 1.81, 0.01), 2)
PN = 0.95


def window_col(r, ids):
    """Return the selected token ID column in the top-20 window, or -1 if absent."""
    tid, rk = r["tid"], r["rk"]
    ins = (rk >= 1) & (rk <= tid.shape[1])
    ycol = np.full(len(r["lp"]), -1, np.int64)
    for t in range(len(ycol)):
        if ins[t]:
            w = np.nonzero(tid[t] == ids[t])[0]
            if len(w):
                ycol[t] = int(w[0])
    return ycol


def one(r, ids):
    tl = r["tlp"].astype(np.float64)
    ok = ~np.isnan(tl)
    lp = r["lp"].astype(np.float64)
    ycol = window_col(r, ids)
    L = len(lp)
    nT = len(TG)
    if L == 0:
        z = np.zeros(nT)
        return z, z, z, z, np.zeros(nT, np.int16), 0
    base = np.where(ok, tl, 0.0)
    order = np.argsort(-np.where(ok, base, -np.inf), axis=-1, kind="stable")
    inv = np.argsort(order, axis=-1, kind="stable")
    yin = ycol >= 0
    yc = np.maximum(ycol, 0)
    T = TG[:, None, None]
    u = np.where(ok[None], base[None] / T, -np.inf)                       # nT, L, K
    m = u.max(-1, keepdims=True)
    e = np.exp(u - m)
    s_ = e.sum(-1, keepdims=True)
    q = e / s_
    lse = (m + np.log(s_))[..., 0]
    logq = np.where(q > 0, u - lse[..., None], 0.0)
    H = -(q * logq).sum(-1)
    var = np.maximum((q * logq ** 2).sum(-1) - H ** 2, 1e-6)
    sy = -(lp[None] / TG[:, None] - lse)
    SZ = (sy - H).sum(-1)
    VZ = var.sum(-1)
    sw = -np.take_along_axis(logq, np.broadcast_to(yc[None, :, None], logq.shape[:-1] + (1,)), axis=-1)[..., 0]
    qs = np.take_along_axis(q, np.broadcast_to(order[None], q.shape), axis=-1)
    keep_s = (np.cumsum(qs, -1) - qs) < PN
    keep_s[..., 0] = True
    keep = np.take_along_axis(keep_s, np.broadcast_to(inv[None], q.shape), axis=-1) & ok[None]
    mass = np.maximum((q * keep).sum(-1), 1e-300)
    qn = np.where(keep, q / mass[..., None], 0.0)
    lqn = np.where(qn > 0, np.log(np.maximum(qn, 1e-300)), 0.0)
    Hn = -(qn * lqn).sum(-1)
    varn = np.maximum((qn * lqn ** 2).sum(-1) - Hn ** 2, 1e-6)
    yn = yin[None] & np.take_along_axis(keep, np.broadcast_to(yc[None, :, None], q.shape[:-1] + (1,)), axis=-1)[..., 0]
    syn = sw + np.log(mass)
    SN = np.where(yn, syn - Hn, 0.0).sum(-1)
    VN = np.where(yn, varn, 0.0).sum(-1)
    return SZ, VZ, SN, VN, yn.sum(-1).astype(np.int16), L


def run(args):
    name, cfgp, gridcache, out_dir, split_tag = args
    from transformers import AutoTokenizer
    from eval_tvmix2 import Src
    from cmp_eval_v3 import attach_tid
    t0 = time.time()
    cfg = json.load(open(cfgp, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    sp, tp = cfg["sources"][name]
    src = Src(name, sp, tp, tok)
    attach_tid(src, sp)
    c = pickle.load(open(os.path.join(gridcache, f"{name}.{split_tag}.v3cond2.pkl"), "rb"))
    keys = c["keys"]
    R = len(keys)
    out = {k: np.zeros((R, len(TG))) for k in ("SZ", "VZ", "SN", "VN")}
    out["nN"] = np.zeros((R, len(TG)), np.int16)
    out["L"] = np.zeros(R, np.int32)
    mdz = mdn = 0.0
    for j, k in enumerate(keys):
        g = c["grids"][k]
        r = src.sc[k]
        SZ, VZ, SN, VN, nN, L = one(r, src.ids[k][: len(r["lp"])])
        out["SZ"][j], out["VZ"][j], out["SN"][j], out["VN"][j], out["nN"][j], out["L"][j] = SZ, VZ, SN, VN, nN, L
        if L:
            zc = np.asarray(g[0], np.float64)[:, 0, 0]
            znc = np.asarray(g[-1]["zN"], np.float64)[:, 0, 0]
            mdz = max(mdz, float(np.max(np.abs(SZ / np.sqrt(VZ) - zc))))
            mdn = max(mdn, float(np.max(np.abs(SN / np.sqrt(np.maximum(VN, 1e-12)) - znc))))
    del c
    if max(mdz, mdn) > 2e-3:                       # Check numerical agreement before writing the cache.
        raise RuntimeError(f"{name}: recomputed z differs from the cached grids by {max(mdz, mdn):.2e} > 2e-3")
    out.update(keys=keys, maxdiff_Z=mdz, maxdiff_N=mdn)
    tmp = os.path.join(out_dir, f"{name}.sv.pkl.tmp")
    pickle.dump(out, open(tmp, "wb"), protocol=4)
    os.replace(tmp, os.path.join(out_dir, f"{name}.sv.pkl"))
    return name, R, mdz, mdn, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--gridcache", required=True)
    ap.add_argument("--names", default="", help="default: every source with a cache file in --gridcache")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--split_tag", default="all")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = a.names.split(",") if a.names else sorted(f[: -len(f".{a.split_tag}.v3cond2.pkl")] for f in os.listdir(a.gridcache)
                                                       if f.endswith(f".{a.split_tag}.v3cond2.pkl"))
    names = [n for n in names if n]
    todo = [n for n in names if not os.path.exists(os.path.join(a.out_dir, f"{n}.sv.pkl"))]
    summ = {}
    for n in names:                                 # existing files: their stored check values count as well
        if n not in todo:
            d = pickle.load(open(os.path.join(a.out_dir, f"{n}.sv.pkl"), "rb"))
            summ[n] = {"rows": len(d["keys"]), "maxdiff_Z": d["maxdiff_Z"], "maxdiff_N": d["maxdiff_N"]}
    with Pool(a.procs) as p:
        for name, R, mdz, mdn, dt in p.imap_unordered(run, [(n, a.cfg, a.gridcache, a.out_dir, a.split_tag) for n in todo]):
            summ[name] = {"rows": R, "maxdiff_Z": mdz, "maxdiff_N": mdn}
            print(f"{name}: {R} rows, max |z - cached| Z {mdz:.2e} N {mdn:.2e}, {dt:.0f}s", flush=True)
    worst = max((max(v["maxdiff_Z"], v["maxdiff_N"]) for v in summ.values()), default=0.0)
    json.dump({"sources": summ, "worst": worst}, open(os.path.join(a.out_dir, "sv_summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"done; {len(summ)} sources, worst difference {worst:.2e}", flush=True)
    if worst > 2e-3:
        raise SystemExit(f"recomputed z differs from the cached grids by {worst:.2e} > 2e-3")


if __name__ == "__main__":
    main()
