"""Compute RUT statistics from teacher-forced response scores.

Method: Zhu et al., arXiv:2506.06975.

The response score is mean log token rank. For L smaller reference scores,
E ties and uniform U, the exchangeability-adjusted rank is
(L + U*(E+1))/(m+1). The alternative rank convention is (L + U*E)/m.
Cramer-von Mises tests these ranks against the uniform distribution."""
import json

import numpy as np
from scipy import stats


def load_scored(prefix):
    """Return {pid: [dict(lp=..., rk=..., k=..., [tlp=top-K logprobs])]} from <prefix>.npz/.index.jsonl."""
    z = np.load(prefix + ".npz")
    off, lp, rk = z["offsets"], z["lp"], z["rk"]
    tlp = z["topk_lp"] if "topk_lp" in z.files else None
    out = {}
    for j, l in enumerate(open(prefix + ".index.jsonl", encoding="utf-8")):
        r = json.loads(l)
        if r.get("n", 0) <= 0:
            continue
        a, b = off[j], off[j + 1]
        d = {"k": r["k"], "lp": lp[a:b], "rk": rk[a:b]}
        if tlp is not None:
            d["tlp"] = tlp[a:b]
        out.setdefault(r["pid"], []).append(d)
    return out


def midrank(resp):
    """Recompute tied ranks inside the stored top-K at fp16 resolution.
    Keep the reported rank for tokens outside the stored window."""
    rk = resp["rk"].astype(np.float64)
    if "tlp" not in resp:
        return rk
    t16 = resp["lp"].astype(np.float16)[:, None]
    alt = resp["tlp"]
    valid = ~np.isnan(alt.astype(np.float32))
    g = np.sum(valid & (alt > t16), axis=1)
    e = np.sum(valid & (alt == t16), axis=1)
    inside = resp["rk"] <= alt.shape[1]
    return np.where(inside & (e >= 1), g + (e + 1) / 2.0, rk)


def logrank_mid(resp):
    return float(np.mean(np.log(midrank(resp))))


def load_scored_multi(prefixes):
    out = {}
    for pre in prefixes:
        for pid, lst in load_scored(pre).items():
            out.setdefault(pid, []).extend(lst)
    return out


def logrank(resp):
    return float(np.mean(np.log(resp["rk"].astype(np.float64))))


def score_pool(scored, fn=logrank):
    return {pid: np.array([fn(r) for r in lst]) for pid, lst in scored.items()}


def rand_rank(t, refs, rng, mode="exact"):
    L = np.sum(refs < t)
    E = np.sum(refs == t)
    u = rng.random()
    m = len(refs)
    if mode == "paper":
        return (L + u * E) / m
    return (L + u * (E + 1)) / (m + 1)


def cvm_p(r):
    return float(stats.cramervonmises(np.clip(r, 0, 1), "uniform").pvalue)


def run_trials(ref, tgt, sub=None, q=0.0, n=100, m=100, trials=500, alpha=0.05, seed=0, mode="exact",
               pids=None, return_p=False, route=None):
    """Power of RUT. ref/tgt/sub: {pid: np.array of scores}. With prob q (or route(pid) if given)
    the target response comes from `sub` instead of `tgt`."""
    rng = np.random.default_rng(seed)
    if pids is None:
        pids = sorted(set(ref) & set(tgt) & (set(sub) if sub is not None else set(tgt)))
    pids = [p for p in pids if len(ref[p]) >= m]
    if len(pids) < n:
        raise ValueError(f"only {len(pids)} prompts with >= {m} reference scores")
    ps = np.empty(trials)
    for t in range(trials):
        chosen = rng.choice(len(pids), size=n, replace=False)
        ranks = np.empty(n)
        for i, ci in enumerate(chosen):
            pid = pids[ci]
            use_sub = (route(pid, rng) if route is not None else (sub is not None and rng.random() < q))
            pool = sub[pid] if use_sub else tgt[pid]
            ts = pool[rng.integers(len(pool))]
            refs = ref[pid] if len(ref[pid]) == m else ref[pid][rng.choice(len(ref[pid]), m, replace=False)]
            ranks[i] = rand_rank(ts, refs, rng, mode)
        ps[t] = cvm_p(ranks)
    power = float(np.mean(ps < alpha))
    return (power, ps) if return_p else power


def power_curve(ref, tgt, sub, qs=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0), **kw):
    seed0 = kw.pop("seed", 0)
    pw = [run_trials(ref, tgt, sub, q=q, seed=seed0 + int(q * 1000), **kw) for q in qs]
    auc = float(np.trapezoid(pw, qs))
    return dict(qs=list(qs), power=pw, auc=auc)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
