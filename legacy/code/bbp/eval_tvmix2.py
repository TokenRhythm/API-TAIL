"""Compare paired temperature statistics and RUT at equal API request budgets.

Paired statistics use two requests per prompt; RUT uses one request per prompt
and 100 local references. Response z grids and pair-agreement likelihoods are
precomputed over temperature."""


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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rut import load_scored, load_scored_multi, logrank_mid, rand_rank, cvm_p  # noqa: E402
from scenarios import make_policies  # noqa: E402

TG = np.round(np.arange(0.30, 1.81, 0.01), 2)
R = str(_APITAIL_ARCHIVE / 'runs')
G = None  # globals filled in worker init


def tempered_grid(tl):
    z = np.where(np.isnan(tl), -np.inf, tl.astype(np.float64))[None] / TG[:, None, None]
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    s = e.sum(-1, keepdims=True)
    return e / s, (m + np.log(s))[..., 0]


def z_grid(r):
    """Per-response grids over TG: z = S/sqrt(V), D = S/L (per-token excess surprisal), SE = sqrt(V)/L."""
    p, lse = tempered_grid(r["tlp"])
    logp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), 0.0)
    ent = -(p * logp).sum(-1)
    var = np.maximum((p * logp ** 2).sum(-1) - ent ** 2, 1e-6)
    lpy = r["lp"][None].astype(np.float64) / TG[:, None] - lse
    S = np.sum(-lpy - ent, 1)
    V = var.sum(1)
    L = len(r["lp"])
    return np.stack([S / np.sqrt(V), S / L, np.sqrt(V) / L]).astype(np.float32)


def pair_ll_grid(ra, ida, idb):
    n = min(len(ida), len(idb), len(ra["lp"]))
    l = 0
    while l < n and ida[l] == idb[l]:
        l += 1
    dv = l < n
    tl = ra["tlp"][: l + (1 if dv else 0)]
    if len(tl) == 0:
        return np.zeros(len(TG), np.float32)
    p, _ = tempered_grid(tl)
    c = np.clip((p ** 2).sum(-1), 1e-12, 1 - 1e-12)
    ll = np.log(c[:, :l]).sum(1) + (np.log(1 - c[:, l]) if dv else 0.0)
    return ll.astype(np.float32)


class Src:
    def __init__(self, name, scored, text, tok):
        self.name = name
        self.sc = {}
        for pid, lst in load_scored(scored).items():
            for r in lst:
                self.sc[(pid, r["k"])] = r
        self.ids = {}
        for l in open(text, encoding="utf-8"):
            r = json.loads(l)
            if "_meta" in r or not r.get("text") or (r["pid"], r["k"]) not in self.sc:
                continue
            self.ids[(r["pid"], r["k"])] = tok.encode(r["text"], add_special_tokens=False)
        self.by_pid = {}
        for (pid, k) in self.sc:
            if (pid, k) in self.ids:
                self.by_pid.setdefault(pid, []).append(k)
        self.zc, self.pc = {}, {}

    def z(self, key):
        if key not in self.zc:
            self.zc[key] = z_grid(self.sc[key])
        return self.zc[key]


def pll(sa, ka, sb, kb):
    ck = (sb.name, kb)
    if ck not in sa.pc.setdefault(ka, {}):
        sa.pc[ka][ck] = pair_ll_grid(sa.sc[ka], sa.ids[ka], sb.ids[kb])
    return sa.pc[ka][ck]


def tvmix(obs, tau, z_trim=3.0, iters=2, eps=(0.0, 0.02, 0.05), k=3.0):
    lls = [pll(a[0], a[1], b[0], b[1]) for a, b in obs]
    gg = [(a[0].z(a[1]), b[0].z(b[1])) for a, b in obs]
    ti = int(np.argmax(np.sum(lls, 0)))
    for _ in range(iters):
        keep = [i for i in range(len(obs)) if gg[i][0][0, ti] <= z_trim and gg[i][1][0, ti] <= z_trim]
        if len(keep) >= max(10, len(obs) // 2):
            ti = int(np.argmax(np.sum([lls[i] for i in keep], 0)))
    M = np.array([g[:, ti] for pair in gg for g in pair])  # responses x (z, D, SE)
    out = {"cz": int(np.sum(M[:, 0] > tau))}
    for e in eps:
        out[f"cD{e}"] = int(np.sum(M[:, 1] > e + k * M[:, 2]))
    return out, float(TG[ti])


def draw(base, sub, route, n, r, rng, pids):
    chosen = rng.choice(len(pids), n, replace=False)
    out = []
    for ci in chosen:
        pid = pids[ci]
        used, reqs = set(), []
        for _ in range(r):
            src = sub if (sub is not None and route(pid, rng)) else base
            ks = [k for k in src.by_pid[pid] if (src.name, k) not in used]
            if not ks:  # never reuse a stored response as a second, "independent" request
                raise ValueError(f"{src.name} has fewer than {r} responses for {pid}")
            k = ks[rng.integers(len(ks))]
            used.add((src.name, k))
            reqs.append((src, k if isinstance(k, tuple) else (pid, k)))
        out.append(reqs)
    return out


def rut_stat(obs):
    rng = G["rng_rut"]
    ranks = []
    for reqs in obs:
        src, key = reqs[0]
        s = G["mid"].setdefault((src.name, key), logrank_mid(src.sc[key]))
        ranks.append(rand_rank(s, G["ref"][key[0]][:100], rng))
    return -np.log(max(cvm_p(np.array(ranks)), 1e-300))  # larger = more anomalous


def run_condition(args):
    kind, bname, sname, pol, n, trials, seed = args
    base, sub = G["srcs"][bname], (G["srcs"][sname] if sname else None)
    route = G["policies"][pol] if pol else None
    pids = [p for p in G["common"] if len(base.by_pid.get(p, [])) >= 2
            and (sub is None or len(sub.by_pid.get(p, [])) >= 2)]
    rng = np.random.default_rng(seed)
    G["rng_rut"] = np.random.default_rng(seed + 7)
    tv, ru, Ts = {}, [], []
    for _ in range(trials):
        o2 = draw(base, sub, route, n, 2, rng, pids)
        s, T = tvmix(o2, G["tau"])
        for kk, vv in s.items():
            tv.setdefault(kk, []).append(vv)
        Ts.append(T)
        if G["ref"] is not None:
            o1 = draw(base, sub, route, 2 * n, 1, rng, pids)
            ru.append(rut_stat(o1))
    return (kind, bname, sname, pol, n), {"tv": tv, "rut": ru, "T": Ts}


def init(cfg):
    global G
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{R}/../models/Qwen3.8-27B")
    srcs = {name: Src(name, s, t, tok) for name, (s, t) in cfg["sources"].items()}
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(f"{R}/../data/prompts_v1.jsonl", encoding="utf-8")}
    ref = None
    if cfg.get("ref"):
        ref = {}
        for pid, lst in load_scored_multi(cfg["ref"]).items():
            ref[pid] = np.array([logrank_mid(x) for x in lst])
        common = sorted(p for p in ref if len(ref[p]) >= 100)
    else:
        common = sorted(P)
    G = {"srcs": srcs, "policies": make_policies(P), "ref": ref, "common": common, "mid": {}, "tau": cfg["tau"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    jobs = []
    for nm in cfg["calib"] + cfg["honest"]:
        for n in cfg["ns"]:
            jobs.append(("null", nm, None, None, n, cfg["trials"], 11))
    for b in cfg["bases"]:
        for s in cfg["subs"]:
            for pol in cfg["policies"]:
                for n in cfg["ns"]:
                    jobs.append(("alt", b, s, pol, n, cfg["trials"], 13))
    init(cfg)  # load once in the parent; forked workers inherit G copy-on-write
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = dict(pool.map(run_condition, jobs, chunksize=1))
    out = {"cfg": cfg, "results": {"|".join(map(str, k)): v for k, v in res.items()}}
    json.dump(out, open(a.out, "w", encoding="utf-8"))
    print("saved", a.out, len(res))


if __name__ == "__main__":
    main()
