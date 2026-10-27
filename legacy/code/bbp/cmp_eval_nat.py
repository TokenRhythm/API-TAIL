"""Evaluate response and paired-sample detectors on shared routed requests.

Paired methods use N/2 prompts with two API requests each; RUT uses N prompts
with one request each. MET compares the paired API samples with two local
reference samples per prompt. The output stores per-trial statistics for
subsequent composite-null calibration."""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as FE  # noqa: E402
from eval_tvmix2 import TG, pll, pair_ll_grid, z_grid, tempered_grid  # noqa: E402
from final_eval import fit_ti, lr_of  # noqa: E402
from rut import logrank_mid, rand_rank, cvm_p  # noqa: E402

G = FE.G
PAD = -1



def zt(src, key, L):
    """z grid of the response truncated to its first L tokens (L=0: whole response)."""
    if not L:
        return src.z(key)
    c = src.__dict__.setdefault("ztc", {})
    if (key, L) not in c:
        r = src.sc[key]
        c[(key, L)] = z_grid({"lp": r["lp"][:L], "tlp": r["tlp"][:L]})
    return c[(key, L)]


def pllt(sa, ka, sb, kb, L):
    if not L:
        return pll(sa, ka, sb, kb)
    c = sa.__dict__.setdefault("plc", {})
    ck = (ka, sb.name, kb, L)
    if ck not in c:
        r = sa.sc[ka]
        c[ck] = pair_ll_grid({"lp": r["lp"][:L], "tlp": r["tlp"][:L]}, sa.ids[ka][:L], sb.ids[kb][:L])
    return c[ck]


def block_grid(src, key, B=32):
    """Per-block excess-surprisal sums S_b(T) and variances V_b(T) (blocks of B tokens, last partial block kept
    if it has >= B/2 tokens): arrays (n_blocks, |TG|)."""
    c = src.__dict__.setdefault("blc", {})
    if key not in c:
        r = src.sc[key]
        p, lse = tempered_grid(r["tlp"])
        logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), 0.0)
        ent = -(p * logp).sum(-1)
        var = np.maximum((p * logp ** 2).sum(-1) - ent ** 2, 1e-6)
        s = -(r["lp"][None].astype(np.float64) / TG[:, None] - lse) - ent      # |TG| x L
        L = s.shape[1]
        cuts = [(a, min(a + B, L)) for a in range(0, L, B) if min(a + B, L) - a >= B // 2]
        S = np.array([s[:, a:b].sum(1) for a, b in cuts]) if cuts else np.zeros((0, len(TG)))
        V = np.array([var[:, a:b].sum(1) for a, b in cuts]) if cuts else np.zeros((0, len(TG)))
        c[key] = (S.astype(np.float32), V.astype(np.float32))
    return c[key]


def mismatch_frac(src, key):
    """Greedy-DiFR quantity for this response: share of tokens that are not the claimed model's argmax (ties at
    fp16 resolution count as matches)."""
    r = src.sc[key]
    top = r["tlp"][:, 0].astype(np.float16)
    mis = (r["rk"] > 1) & ~(r["lp"].astype(np.float16) >= top)
    return float(np.mean(mis)) if len(mis) else np.nan



def met_ids(ids, L):
    a = np.full(L, PAD, np.int64)
    a[: min(L, len(ids))] = ids[:L]
    return a


def mmd_pairs(X, Y):
    """X, Y: (n_prompts, 2, L) token arrays (two API / two reference samples per prompt). Kernel = position-wise
    agreement count / L (diagonal-normalized Hamming kernel, zero across prompts); unbiased same-prompt averages."""
    L = X.shape[-1]
    kxx = np.mean(X[:, 0] == X[:, 1], -1)
    kyy = np.mean(Y[:, 0] == Y[:, 1], -1)
    kxy = np.mean([np.mean(X[:, a] == Y[:, b], -1) for a in (0, 1) for b in (0, 1)], 0)
    del L
    return float(np.mean(kxx) - 2 * np.mean(kxy) + np.mean(kyy))


def ref_pair(pid, rng, m=2):
    pool = G["ref_ids"][pid]
    sel = rng.choice(len(pool), m, replace=False)
    return [pool[j] for j in sel]



def draw_pairs(base, sub, route, n, rng, pids, adv1=False):
    """n prompts x 2 requests; request 2 always from the base when adv1."""
    chosen = rng.choice(len(pids), n, replace=False)
    out = []
    for ci in chosen:
        pid = pids[ci]
        used, reqs = set(), []
        for j in range(2):
            routed = sub is not None and route(pid, rng)
            src = sub if (routed and not (adv1 and j == 1)) else base
            ks = [k for k in src.by_pid[pid] if (src.name, k) not in used]
            if not ks:
                raise ValueError(f"{src.name} has fewer than 2 responses for {pid}")
            k = ks[rng.integers(len(ks))]
            used.add((src.name, k))
            reqs.append((src, (pid, k)))
        out.append(reqs)
    return out, [pids[c] for c in chosen]


def draw_singles(base, sub, route, n, rng, pids):
    chosen = rng.choice(len(pids), n, replace=False)
    out = []
    for ci in chosen:
        pid = pids[ci]
        src = sub if (sub is not None and route(pid, rng)) else base
        ks = src.by_pid[pid]
        out.append((src, (pid, ks[rng.integers(len(ks))])))
    return out



def a_family(obs, rng, L=0, full=True, own=None):
    lls = [pllt(a[0], a[1], b[0], b[1], L) for a, b in obs]
    zz = [(zt(a[0], a[1], L), zt(b[0], b[1], L)) for a, b in obs]
    idx = rng.permutation(len(obs))
    h = [idx[: len(idx) // 2], idx[len(idx) // 2:]]

    def crossfit(z_trim, iters):
        resp, Ts = [], []
        for f in (0, 1):
            ti = fit_ti(lls, zz, h[1 - f], z_trim=z_trim, iters=iters)
            Ts.append(ti)
            resp += [(i, j, ti) for i in h[f] for j in (0, 1)]
        return resp, Ts

    cf, Ts = crossfit(3.0, 2)
    o = {"zA": [float(zz[i][j][0, ti]) for i, j, ti in cf],
         "dA": [float(zz[i][j][1, ti]) for i, j, ti in cf],
         "seA": [float(zz[i][j][2, ti]) for i, j, ti in cf],
         "T": float(np.mean(TG[Ts])),
         "Tagree": float(TG[int(np.argmax(np.sum(lls, 0)))])}
    allz = np.array([g[0] for pair in zz for g in pair])           # responses x |TG|
    tal = int(np.argmin(np.abs(allz.mean(0))))
    o["Talign"] = float(TG[tal])
    if not full:
        return o, cf
    tif = fit_ti(lls, zz, list(range(len(obs))))
    o["zA_full"] = [float(g[0, tif]) for pair in zz for g in pair]
    cf0, _ = crossfit(3.0, 0)
    o["zA_notrim"] = [float(zz[i][j][0, ti]) for i, j, ti in cf0]
    cf2, _ = crossfit(2.0, 2)
    o["zA_trim2"] = [float(zz[i][j][0, ti]) for i, j, ti in cf2]
    o["zA_align"] = allz[:, tal].astype(float).tolist()
    for nm, ti in G["fixedT"].items():                               # Fixed temperature supplied by the configuration.
        o[f"zA_{nm}"] = allz[:, ti].astype(float).tolist()
    if own is not None:                                              # oracle: the base's own T_agree
        o["zA_own"] = allz[:, own].astype(float).tolist()
    return o, cf


def run_condition(args):
    kind, bname, sname, pol, nreq, trials, seed = args
    base, sub = G["srcs"][bname], (G["srcs"][sname] if sname else None)
    route = G["policies"][pol] if pol else None
    n = nreq // 2
    pids = [p for p in G["common"] if len(base.by_pid.get(p, [])) >= 2
            and (sub is None or len(sub.by_pid.get(p, [])) >= 2)]
    if len(pids) < n:
        return (kind, bname, sname, pol, nreq), None
    rut_ok = kind in ("null", "alt") and G["ref"] is not None and len(pids) >= nreq   # N_req distinct prompts
    rng = np.random.default_rng(seed)
    out = {}

    def put(k, v):
        out.setdefault(k, []).append(v)

    for _ in range(trials):
        if kind == "metnull":       # MET native null: two local reference samples play the API
            chosen = rng.choice(len(pids), n, replace=False)
            X, Y = [], []
            for ci in chosen:
                four = ref_pair(pids[ci], rng, 4)
                X.append(four[:2]); Y.append(four[2:])
            for L in G["met_L"]:
                put(f"met{L}", mmd_pairs(np.array([[met_ids(x, L) for x in p] for p in X]),
                                         np.array([[met_ids(y, L) for y in p] for p in Y])))
            continue
        if kind.startswith("amix"):  # amix<f>[_adv1]: A-mixed
            f = float(kind[4:].split("_")[0])
            npair = max(2, int(round(f * nreq / 2)))
            nsing = nreq - 2 * npair
            o2, used = draw_pairs(base, sub, route, npair, rng, pids, adv1=kind.endswith("adv1"))
            usedset = set(used)
            rest = [p for p in pids if p not in usedset]
            o1 = draw_singles(base, sub, route, nsing, rng, rest) if nsing > 0 else []
            o, cf = a_family(o2, rng, full=False)
            Tall = int(np.argmin(np.abs(TG - o["T"])))   # singles scored at the mean cross-fitted T
            zs = o["zA"] + [float(s.z(k)[0, Tall]) for s, k in o1]
            put("zA", zs)
            continue
        o2, used = draw_pairs(base, sub, route, n, rng, pids, adv1=(kind == "adv1"))
        o, cf = a_family(o2, rng, full=(kind != "adv1"), own=G["ownT"].get(bname))
        for k, v in o.items():
            put(k, v)
        if G["prefix_L"]:
            for L in G["prefix_L"]:
                put(f"zA_L{L}", a_family(o2, rng, L=L, full=False)[0]["zA"])
        if G["blocks"]:
            zb = []
            for i, j, ti in cf:
                S, V = block_grid(*o2[i][j])
                zb += (S[:, ti] / np.sqrt(V[:, ti])).astype(float).tolist()
            put("zBlk", zb)
        resp = [rq for reqs in o2 for rq in reqs]
        for c in G["cands"]:
            put(f"lB:{c}", [lr_of(s, k, c) for s, k in resp])
        for c, sc in G["cls"].items():
            put(f"cls:{c}", [sc.get((s.name,) + k, np.nan) for s, k in resp])
        X = [[met_ids(s.ids[k], max(G["met_L"])) for s, k in reqs] for reqs in o2]
        Y = [[met_ids(y, max(G["met_L"])) for y in ref_pair(p, rng)] for p in used]
        X, Y = np.array(X), np.array(Y)
        for L in G["met_L"]:
            put(f"met{L}", mmd_pairs(X[..., :L], Y[..., :L]))
        if rut_ok:
            if True:
                o1 = draw_singles(base, sub, route, nreq, rng, pids)
                ranks = []
                for s, k in o1:
                    sv = G["mid"].setdefault((s.name, k), logrank_mid(s.sc[k]))
                    ranks.append(rand_rank(sv, G["ref"][k[0]][:100], rng))
                put("rut", cvm_p(np.array(ranks)))
    return (kind, bname, sname, pol, nreq), out


def init(cfg):
    FE.init(cfg)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    split = {json.loads(l)["pid"]: json.loads(l)["split"] for l in open(cfg["prompts"], encoding="utf-8")}
    G["common"] = [p for p in G["common"] if split[p] == "test"]
    G["met_L"] = cfg.get("met_L", [30])
    G["prefix_L"] = cfg.get("prefix_L", [])
    G["blocks"] = cfg.get("blocks", 0)
    G["fixedT"] = {k: int(np.argmin(np.abs(TG - v))) for k, v in cfg.get("fixedT", {}).items()}
    G["ref_ids"] = {}
    for path in cfg["ref_text"]:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if "_meta" in r or r["pid"] not in set(G["common"]) or not isinstance(r.get("text"), str):
                continue
            G["ref_ids"].setdefault(r["pid"], []).append(
                tok.encode(r["text"], add_special_tokens=False)[: max(G["met_L"])])
    short = [p for p in G["common"] if len(G["ref_ids"].get(p, [])) < 4]
    if short:
        raise SystemExit(f"{len(short)} test prompts have fewer than 4 reference texts for MET")
    G["ownT"] = {}
    for name in set(cfg["calib"] + cfg["honest"] + cfg["bases"]):
        src = G["srcs"][name]
        lls = [pll(src, (p, 0), src, (p, 1)) for p in G["common"] if 0 in src.by_pid.get(p, []) and 1 in src.by_pid.get(p, [])]
        G["ownT"][name] = int(np.argmax(np.sum(lls, 0)))
    G["cls"] = {}
    for c, path in cfg.get("cls", {}).items():
        d = json.load(open(path, encoding="utf-8"))
        G["cls"][c] = {(s, p, int(k)): v for s, rows in d.items() for p, k, v in rows}
    print("init done: test prompts", len(G["common"]), "sources", len(G["srcs"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    T = cfg["trials"]
    jobs = [("null", v, None, None, nr, T, 11) for v in cfg["calib"] + cfg["honest"] for nr in cfg["nreq"]]
    jobs += [("metnull", cfg["met_null_base"], None, None, nr, cfg.get("met_null_trials", 1000), 17)
             for nr in cfg["nreq"]]
    for b, s, pol, nr in cfg["alt"]:
        jobs.append(("alt", b, s, pol, nr, T, 13))
    for kind, b, s, pol, nr in cfg.get("extra", []):
        jobs.append((kind, b, s, pol, nr, T, 19))
    init(cfg)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = {k: v for k, v in pool.imap_unordered(run_condition, jobs, chunksize=1) if v is not None}
    json.dump({"cfg": cfg, "results": {"|".join(map(str, k)): v for k, v in res.items()}}, open(a.out, "w", encoding="utf-8"))
    print("saved", a.out, len(res), "of", len(jobs))


if __name__ == "__main__":
    main()
