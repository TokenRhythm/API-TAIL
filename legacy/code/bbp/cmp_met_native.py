"""Evaluate Hamming-MMD under MET's native sampling protocol.

Method: Gao, Liang and Guestrin, arXiv:2410.20247.

Each trial uses N/10 prompts, ten API responses and ten local references per
prompt, with responses padded to 50 tokens. Requests are routed independently
and use distinct stored samples. The reference-only null uses two disjoint
sets of ten samples per prompt."""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenarios import make_policies  # noqa: E402

G = {}
L = 50
PAD = -1


def load_ids(path, tok):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if "_meta" in r or not isinstance(r.get("text"), str):
            continue
        ids = tok.encode(r["text"], add_special_tokens=False)[:L]
        a = np.full(L, PAD, np.int64)
        a[: len(ids)] = ids
        out.setdefault(r["pid"], []).append(a)
    return {p: np.array(v) for p, v in out.items()}


def mmd(X, Y):
    """X, Y: lists over prompts of (m, L) arrays; same-prompt kernel = share of agreeing positions."""
    sxx = nxx = sxy = nxy = syy = nyy = 0.0
    for x, y in zip(X, Y):
        kxx = (x[:, None, :] == x[None, :, :]).mean(-1)
        kyy = (y[:, None, :] == y[None, :, :]).mean(-1)
        kxy = (x[:, None, :] == y[None, :, :]).mean(-1)
        sxx += kxx.sum() - np.trace(kxx); nxx += kxx.size - len(kxx)
        syy += kyy.sum() - np.trace(kyy); nyy += kyy.size - len(kyy)
        sxy += kxy.sum(); nxy += kxy.size
    return float(sxx / nxx - 2 * sxy / nxy + syy / nyy)


def run_condition(args):
    kind, bname, sname, pol, nreq, trials, seed = args
    rng = np.random.default_rng(seed)
    npr = nreq // 10
    pids = G["pids"]
    if npr > len(pids):
        return (kind, bname, sname, pol, nreq), None
    route = G["policies"][pol] if pol else None
    out = {"met": []}
    for _ in range(trials):
        X, Y = [], []
        for ci in rng.choice(len(pids), npr, replace=False):
            pid = pids[ci]
            ref = G["ref"][pid]
            sel = rng.choice(len(ref), 20 if kind == "metnull" else 10, replace=False)
            Y.append(ref[sel[:10]])
            if kind == "metnull":
                X.append(ref[sel[10:]])
                continue
            base = G["banks"][bname][pid]
            sub = G["banks"][sname][pid] if sname else None
            ob, os_ = rng.permutation(len(base)), (rng.permutation(len(sub)) if sub is not None else None)
            ib = is_ = 0
            rows = []
            for _r in range(10):
                if sub is not None and route(pid, rng):
                    rows.append(sub[os_[is_]]); is_ += 1
                else:
                    rows.append(base[ob[ib]]); ib += 1
            X.append(np.array(rows))
        out["met"].append(mmd(X, Y))
    return (kind, bname, sname, pol, nreq), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    G["ref"] = load_ids(cfg["ref"], tok)
    G["banks"] = {n: load_ids(p, tok) for n, p in cfg["sources"].items()}
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(cfg["prompts_all"], encoding="utf-8")}
    G["policies"] = make_policies(P)
    G["pids"] = sorted(p for p in G["ref"] if len(G["ref"][p]) >= 20 and
                       all(len(b.get(p, [])) >= 10 for b in G["banks"].values()))
    print("prompts with full banks", len(G["pids"]), flush=True)
    T = cfg["trials"]
    jobs = [("null", v, None, None, nr, T, 11) for v in cfg["calib"] + cfg["honest"] for nr in cfg["nreq"]]
    jobs += [("metnull", None, None, None, nr, 1000, 17) for nr in cfg["nreq"]]
    jobs += [("alt", b, s, pol, nr, T, 13) for b, s, pol, nr in cfg["alt"]]
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = {k: v for k, v in pool.imap_unordered(run_condition, jobs, chunksize=1) if v is not None}
    json.dump({"cfg": cfg, "results": {"|".join(map(str, k)): v for k, v in res.items()}}, open(a.out, "w", encoding="utf-8"))
    print("saved", a.out, len(res), "of", len(jobs))


if __name__ == "__main__":
    main()
