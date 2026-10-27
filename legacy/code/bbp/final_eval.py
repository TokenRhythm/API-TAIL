"""Evaluate cross-fitted typicality, candidate likelihood ratios and RUT.

Paired methods share prompts and independently routed requests. Temperature
for one half is fitted on the other half. RUT uses the same request budget
with one response per prompt. Raw per-trial values are saved for calibration."""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tvmix2 import TG, Src, pll, draw  # noqa: E402
from rut import load_scored, load_scored_multi, logrank_mid, rand_rank, cvm_p  # noqa: E402
from scenarios import make_policies  # noqa: E402

G = {}


def cand_scores(prefix):
    """Per-(pid,k) (summed logprob, scored token count) under a candidate model."""
    out = {}
    for pid, lst in load_scored(prefix).items():
        for r in lst:
            out[(pid, r["k"])] = (float(np.sum(r["lp"])), len(r["lp"]))
    return out


def fit_ti(lls, zz, idx, z_trim=3.0, iters=2):
    ti = int(np.argmax(np.sum([lls[i] for i in idx], 0)))
    for _ in range(iters):
        keep = [i for i in idx if zz[i][0][0, ti] <= z_trim and zz[i][1][0, ti] <= z_trim]
        if len(keep) >= max(8, len(idx) // 2):
            ti = int(np.argmax(np.sum([lls[i] for i in keep], 0)))
    return ti


def stat_A(obs, rng):
    """Returns per-response z at the cross-fitted T, the mean cross-fitted T, and the trial's two views:
    T_agree (pair agreement MLE over all pairs, untrimmed) and T_align (where the mean z over all responses is 0)."""
    lls = [pll(a[0], a[1], b[0], b[1]) for a, b in obs]
    zz = [(a[0].z(a[1]), b[0].z(b[1])) for a, b in obs]
    idx = rng.permutation(len(obs))
    h = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
    zs, Ts = [], []
    for f in (0, 1):
        ti = fit_ti(lls, zz, h[1 - f])
        Ts.append(float(TG[ti]))
        zs += [zz[i][j][0, ti] for i in h[f] for j in (0, 1)]
    t_agree = float(TG[int(np.argmax(np.sum(lls, 0)))])
    mz = np.mean([g[0] for pair in zz for g in pair], 0)
    t_align = float(TG[int(np.argmin(np.abs(mz)))])
    return np.array(zs), float(np.mean(Ts)), t_agree, t_align


def lr_of(src, key, cname):
    """Per-token candidate LR; NaN unless both models scored the complete response (no truncation)."""
    if not G["full"].get((src.name, key), {}).get(cname, False):
        return np.nan
    lc = G["cand"][cname][src.name][key][0]
    return (lc - float(np.sum(src.sc[key]["lp"]))) / max(1, len(src.sc[key]["lp"]))


def binom_p(k, n, pi):
    return float(stats.binom.sf(k - 1, n, pi)) if n > 0 else 1.0


def rut_p(obs1, rng):
    ranks = []
    for reqs in obs1:
        src, key = reqs[0]
        s = G["mid"].setdefault((src.name, key), logrank_mid(src.sc[key]))
        ranks.append(rand_rank(s, G["ref"][key[0]][:100], rng))
    return cvm_p(np.array(ranks))


def run_condition(args):
    kind, bname, sname, pol, n, trials, seed = args
    base, sub = G["srcs"][bname], (G["srcs"][sname] if sname else None)
    route = G["policies"][pol] if pol else None
    # both requests of a prompt may hit the same source, so every source needs two distinct responses there
    pids = [p for p in G["common"] if len(base.by_pid.get(p, [])) >= 2
            and (sub is None or len(sub.by_pid.get(p, [])) >= 2)]
    if len(pids) < (2 * n if G.get("ref") is not None else n):  # RUT needs 2n distinct prompts
        return (kind, bname, sname, pol, n), None
    rng = np.random.default_rng(seed)
    out = {"zA": [], "T": [], "Tagree": [], "Talign": [], "lB": {c: [] for c in G["cands"]}, "rut": []}
    for _ in range(trials):
        o2 = draw(base, sub, route, n, 2, rng, pids)
        zs, T, ta, tl = stat_A(o2, rng)
        out["zA"].append(zs.tolist()); out["T"].append(T); out["Tagree"].append(ta); out["Talign"].append(tl)
        for c in G["cands"]:
            out["lB"][c].append([lr_of(s, k, c) for reqs in o2 for (s, k) in reqs])
        if G.get("ref") is not None:
            o1 = draw(base, sub, route, 2 * n, 1, rng, pids)
            out["rut"].append(rut_p(o1, rng))
    return (kind, bname, sname, pol, n), out


def init(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    G["srcs"] = {name: Src(name, s, t, tok) for name, (s, t) in cfg["sources"].items()}
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(cfg["prompts"], encoding="utf-8")}
    G["policies"] = make_policies(P)
    G["cands"] = list(cfg.get("cands", {}))
    bases = [os.path.basename(s) for s, _ in cfg["sources"].values()]
    assert len(set(bases)) == len(bases), "candidate files are looked up by basename: source basenames must be unique"
    G["cand"] = {c: {name: cand_scores(f"{d}/{os.path.basename(cfg['sources'][name][0])}")
                     for name in cfg["sources"] if os.path.exists(f"{d}/{os.path.basename(cfg['sources'][name][0])}.npz")}
                 for c, d in cfg.get("cands", {}).items()}
    # B uses a response only if M and the candidate both scored all of it (score_tf truncates at max_model_len,
    # and the candidates' tokenizer differs from M's)
    missing_tok = [c for c in G["cands"] if c not in cfg.get("cand_tokenizer", {})]
    if missing_tok:
        raise SystemExit(f"cand_tokenizer missing for {missing_tok}: needed to reject truncated candidate scores")
    ctok = {}
    for c, path in cfg.get("cand_tokenizer", {}).items():
        ctok[c] = ctok.get(path) or AutoTokenizer.from_pretrained(path)
        ctok[path] = ctok[c]
    G["full"] = {}
    used = set(cfg["calib"] + cfg["honest"] + cfg["bases"] + cfg["subs"])
    for name, (_, tpath) in cfg["sources"].items():
        src = G["srcs"][name]
        for l in open(tpath, encoding="utf-8"):
            r = json.loads(l)
            key = (r.get("pid"), r.get("k"))
            if "_meta" in r or key not in src.ids:
                continue
            m_full = len(src.sc[key]["lp"]) == len(src.ids[key])
            ok = {}
            for c in G["cands"]:
                rec = G["cand"][c].get(name, {}).get(key)
                ok[c] = m_full and rec is not None and \
                    rec[1] == len(ctok[c].encode(r["text"], add_special_tokens=False))
            G["full"][(name, key)] = ok
        for c in G["cands"]:
            cov = np.mean([G["full"].get((name, k), {}).get(c, False) for k in src.ids]) if src.ids else 0.0
            if name in used:
                print(f"candidate coverage {c:10s} {name:40s} {cov:.4f}", flush=True)
                if cov < 0.9:
                    raise SystemExit(f"candidate {c} covers only {cov:.1%} of {name}: score it before evaluating")
    G["mid"] = {}
    G["ref"] = None
    if cfg.get("ref"):
        G["ref"] = {pid: np.array([logrank_mid(x) for x in lst]) for pid, lst in load_scored_multi(cfg["ref"]).items()}
        G["common"] = sorted(p for p in G["ref"] if len(G["ref"][p]) >= 100)
    else:
        G["common"] = sorted(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--skip_null", type=int, default=0, help="only substitution conditions (merge with a full run)")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    assert min(cfg["ns"]) >= 2, "two-fold cross-fitting needs n >= 2"
    jobs = [] if a.skip_null else \
        [("null", nm, None, None, n, cfg["trials"], 11) for nm in cfg["calib"] + cfg["honest"] for n in cfg["ns"]]
    jobs += [("alt", b, s, pol, n, cfg["trials"], 13) for b in cfg["bases"] for s in cfg["subs"]
             for pol in cfg["policies"] for n in cfg["ns"]]
    init(cfg)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = {k: v for k, v in pool.map(run_condition, jobs, chunksize=1) if v is not None}
    json.dump({"cfg": cfg, "results": {"|".join(map(str, k)): v for k, v in res.items()}}, open(a.out, "w", encoding="utf-8"))
    print("saved", a.out, len(res))


if __name__ == "__main__":
    main()
