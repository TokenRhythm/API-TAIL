"""Evaluate and bootstrap audits on disjoint calibration and test prompts.

The design JSON supplies source files, score-cache directories, prompt
selectors, legitimate pools, honest endpoints, substitutes and methods.
When test_dirs is present, non-pool sources resolve only in those directories.
Full-vocabulary S/V rows are aligned by response key to the grid-cache rows.

Even null trials calibrate; odd null trials estimate held-out false alarms.
Each method is calibrated separately for each pool. The integer rule uses
the maximum per-configuration 95th percentile and strict rejection above it.
The randomized count rule also assigns rejection probability at the boundary.

RUT uses negative log p. Matched-budget MET uses its own paired draw with
two API and two reference samples per prompt. Bootstrap replicates resample
test prompts with replacement and redraw calibration trials before refitting.

Usage: paper2_eval.py --design design.json --out prefix [--device cuda:0]"""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(_APITAIL_CODE / 'jobs/supp1'))
sys.path.insert(0, str(_APITAIL_CODE / 'jobs/talign'))
import engine as EN  # noqa: E402
import ta_run_eval as TA  # noqa: E402  (registers T_MLE, N95 T-only MLE, T_align-fsv, T_MLE-full, N95-full id / MLE)


EN.METHODS["T Huber-fsv"] = ("prof", "Z", False, "svhub", "F")    # sum_r clamp(z_r, +-1.345) = 0
EN.METHODS["FDG-fsv"] = ("prof", "Z", False, "svfix", "F")        # fixed T = 1.0
OWN_EST = ("svhub", "svfix")
HF_FAMILY = ["T_align-fsv", "T_MLE-full", "T Huber-fsv", "FDG-fsv", "N95-full id", "N95-full MLE"]
E6_N = 200
MET = "MET same-budget"
ALPHA, GATE, REDRAW_KEEP = 0.05, 0.10, 16
CONTINUOUS = ("rut", "rutiu", "met")



def _init(self, cfg, gridcache, names, sv_dir=None, svfull_dir=None, **kw):
    """Load response S/V arrays and align them by key to the grid-cache rows."""
    TA._init0(self, cfg, gridcache, names, **kw)
    if sv_dir:
        parts = {f: [] for f in ("SZ", "VZ", "SN", "VN", "nN", "L")}
        for name in self.names:
            d = pickle.load(open(os.path.join(sv_dir, f"{name}.sv.pkl"), "rb"))
            pos = {tuple(k): j for j, k in enumerate(d["keys"])}
            idx = np.array([pos[tuple(k)] for k in self.keys[name]], np.int64)
            for f in parts:
                parts[f].append(np.asarray(d[f])[idx])
        for f in ("SZ", "VZ", "SN", "VN"):
            self.grids[f] = torch.from_numpy(np.concatenate(parts[f])).to(self.device)
        self.nN = torch.from_numpy(np.concatenate(parts["nN"]).astype(np.float64)).to(self.device)
        self.Ltok = torch.from_numpy(np.concatenate(parts["L"]).astype(np.float64)).to(self.device)
    if svfull_dir:
        parts = {f: [] for f in ("SF0", "VF0", "SNF0", "VNF0", "nNF0", "overNF")}
        ntok = []
        self.scorer_ids = {}
        for name in self.names:
            d = pickle.load(open(os.path.join(svfull_dir, f"{name}.svfull.pkl"), "rb"))
            pos = {tuple(k): j for j, k in enumerate(d["keys"])}
            miss = [k for k in self.keys[name] if tuple(k) not in pos]
            assert not miss, f"svfull {name}: {len(miss)} grid rows without a full-vocabulary score"
            idx = np.array([pos[tuple(k)] for k in self.keys[name]], np.int64)
            for f in parts:
                parts[f].append(np.asarray(d["fields"][f])[idx])
            ntok.append(np.asarray(d["ntok"])[idx])
            self.scorer_ids[name] = d.get("scorer", {"impl": "vllm"}).get("impl")
        for f, g in (("SF0", "SF"), ("VF0", "VF"), ("SNF0", "SNF"), ("VNF0", "VNF")):
            self.grids[g] = torch.from_numpy(np.concatenate(parts[f])).to(self.device)
        self.nNF = torch.from_numpy(np.concatenate(parts["nNF0"]).astype(np.float64)).to(self.device)
        self.ovNF = torch.from_numpy(np.concatenate(parts["overNF"]).astype(np.float64)).to(self.device)
        self.LtokF = torch.from_numpy(np.concatenate(ntok).astype(np.float64)).to(self.device)


def linkdir(tag, dirs, names, suffix):
    """one directory with a symlink per source to its file in the first listed dir that has it"""
    d = f"/tmp/paper2_{tag}_{os.getpid()}"
    os.makedirs(d, exist_ok=True)
    for n in names:
        hits = [os.path.join(x, n + suffix) for x in dirs if os.path.exists(os.path.join(x, n + suffix))]
        assert hits, f"no {suffix} file for {n} in {dirs}"
        dst = os.path.join(d, n + suffix)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(hits[0]), dst)
    return d



def stats(pan, rows, uni, methods, tau=3.0):
    """engine statistics (patched by ta_run_eval for the S / V families) plus the svhub / svfix full-vocabulary rules;
    adds ('T', m) = profiled temperature per trial, ('ret', m) = N95 / NF retention and ('ovr', m) = NF overflow share."""
    methods = [m for m in methods if m != MET]
    own = [m for m in methods if EN.METHODS[m][0] == "prof" and EN.METHODS[m][3] in OWN_EST]
    rest = [m for m in methods if m not in own]
    out = EN.Panel.stats(pan, rows, uni, rest) if rest else {}
    rt = rows if torch.is_tensor(rows) else torch.from_numpy(rows).to(pan.device)
    T_, N = rt.shape
    bs = max(1, int(3e8 // (N * EN.NT * 8)))
    for m in own:
        d = EN.METHODS[m]
        S, V = pan.grids["S" + d[4]], pan.grids["V" + d[4]]
        K = torch.empty(T_, dtype=torch.int32, device=pan.device)
        TH = torch.empty(T_, dtype=torch.int32, device=pan.device)
        for b0 in range(0, T_, bs):
            r = rt[b0: b0 + bs]
            zb = S[r] / torch.sqrt(torch.clamp(V[r], min=1e-12))
            if d[3] == "svhub":
                tk = torch.argmin(zb.clamp(-EN.HUBER, EN.HUBER).sum(1).abs(), dim=1)
            else:
                tk = torch.full((len(r),), EN.I10, dtype=torch.int64, device=pan.device)
            z = zb.gather(2, tk[:, None, None].expand(-1, N, 1))[..., 0]
            K[b0: b0 + len(r)] = (z > tau).sum(1).int()
            TH[b0: b0 + len(r)] = (tk * EN.NAG).int()
        out[m] = K.cpu().numpy()
        out[("theta", m)] = TH.cpu().numpy()
    for m in methods:
        if ("theta", m) not in out:
            continue
        th = np.asarray(out[("theta", m)], np.int64)
        out[("T", m)] = EN.TG[th // EN.NAG]
        d = EN.METHODS[m]
        fam = d[4] if len(d) > 4 else d[1]
        tk = torch.from_numpy(th // EN.NAG).to(pan.device)[:, None].expand(-1, N)
        if fam == "ZN" and hasattr(pan, "nN") and not d[2]:         # penalty-profiling N95: no retention cache
            out[("ret", m)] = (pan.nN[rt, tk].sum(1) / pan.Ltok[rt].sum(1).clamp(min=1)).cpu().numpy()
        elif fam == "NF" and hasattr(pan, "nNF"):
            den = pan.LtokF[rt].sum(1).clamp(min=1)
            out[("ret", m)] = (pan.nNF[rt, tk].sum(1) / den).cpu().numpy()
            out[("ovr", m)] = (pan.ovNF[rt, tk].sum(1) / den).cpu().numpy()
    return out


def take(st, idx):
    """the trials idx of a stats dict"""
    out = {}
    for k, v in st.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            out[k] = {kk: np.asarray(vv)[idx] for kk, vv in v.items()}
        else:
            out[k] = np.asarray(v)[idx]
    return out


def kind(m):
    return "met" if m == MET else EN.METHODS[m][0]


def transform(m, raw, tau=None):
    k = kind(m)
    if k in ("rut", "rutiu"):
        return -np.log(np.maximum(np.asarray(raw["p"], float), 1e-300))
    if k == "det":
        x = np.asarray(raw, float)
        return np.sum(np.nan_to_num(x, nan=-np.inf) > tau, axis=1).astype(float)
    return np.asarray(raw, float)


def det_tau(raws):
    allv = np.concatenate([np.asarray(r, float).reshape(-1) for r in raws])
    allv = allv[~np.isnan(allv)]
    return float(np.quantile(allv, 0.99))


def q_higher(x, q=0.95):
    return float(np.quantile(np.asarray(x, float), q, method="higher"))


def crit(s, alpha=ALPHA):
    """randomized critical pair of one member's count statistic"""
    vals, cnt = np.unique(np.asarray(s, float), return_counts=True)
    p = cnt / cnt.sum()
    surv = 1.0 - np.cumsum(p)                            # P(s > vals[i])
    i = int(np.argmax(surv <= alpha + 1e-12))
    g = (alpha - surv[i]) / p[i] if p[i] > 0 else 0.0
    return float(vals[i]), float(min(max(g, 0.0), 1.0))


def rate_int(s, thr):
    return float(np.mean(np.asarray(s, float) > thr))


def rate_ex(s, k, g):
    s = np.asarray(s, float)
    return float(np.mean(s > k) + g * np.mean(s == k))


def calibrate(m, cal, members):
    """cal: {member: raw entry of method m on the calibration trials}. Returns the arm's rule and the members' s."""
    tau = det_tau([cal[v] for v in members]) if kind(m) == "det" else None
    s = {v: transform(m, cal[v], tau) for v in members}
    qs = {v: q_higher(s[v]) for v in members}
    dec = max(qs, key=qs.get)
    if kind(m) in CONTINUOUS:
        ex = {"k": qs[dec], "gamma": 0.0, "decisive": dec}
    else:
        best = None
        for v in members:
            k, g = crit(s[v])
            if best is None or (k, -g) > (best[0], -best[1]):
                best = (k, g, v)
        ex = {"k": best[0], "gamma": best[1], "decisive": best[2]}
    return {"thr": qs[dec], "decisive": dec, "q95": qs, "tau": tau, "exact": ex}, s



class MetBank:
    """first-L token ids per (source, pid) and per reference prompt, tokenized with the claimed model's tokenizer"""

    def __init__(self, cfg, names, L):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        self.L = L
        self.bank = {}
        for n in names:
            path, mk = cfg["sources"][n][1], int(cfg.get("max_k", {}).get(n, 99))
            by = {}
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                if "_meta" in r or not isinstance(r.get("text"), str) or not r["text"] or int(r.get("k", 0)) >= mk:
                    continue
                by.setdefault(r["pid"], {})[int(r.get("k", 0))] = self.ids(tok, r["text"])
            self.bank[n] = {p: np.stack([v[k] for k in sorted(v)]) for p, v in by.items()}
        self.ref = {}
        for path in cfg.get("ref_text", []):
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                if "_meta" in r or not isinstance(r.get("text"), str):
                    continue
                self.ref.setdefault(r["pid"], []).append(self.ids(tok, r["text"]))
        self.ref = {p: np.stack(v) for p, v in self.ref.items()}

    def ids(self, tok, text):
        a = np.full(self.L, -1, np.int64)
        x = tok.encode(text, add_special_tokens=False)[: self.L]
        a[: len(x)] = x
        return a


def mmd_pairs(X, Y):
    """X, Y: (n_prompts, 2, L); position-wise agreement kernel, unbiased same-prompt averages (cmp_eval_nat.mmd_pairs)"""
    kxx = np.mean(X[:, 0] == X[:, 1], -1)
    kyy = np.mean(Y[:, 0] == Y[:, 1], -1)
    kxy = np.mean([np.mean(X[:, a] == Y[:, b], -1) for a in (0, 1) for b in (0, 1)], 0)
    return float(np.mean(kxx) - 2 * np.mean(kxy) + np.mean(kyy))


def _stack(bank, el, L):
    """(len(el), max count, L) token array of one bank over the eligible prompt list, and the per-prompt counts"""
    cnt = np.array([len(bank[p]) for p in el], np.int64)
    arr = np.full((len(el), int(cnt.max()), L), -1, np.int32)
    for i, p in enumerate(el):
        arr[i, : cnt[i]] = bank[p]
    return arr, cnt


def _pick2(u0, u1, cnt0, cnt1, same):
    """two indices: independent picks, or two distinct indices when both requests use the same bank"""
    i0 = np.minimum((u0 * cnt0).astype(np.int64), cnt0 - 1)
    i1s = np.minimum((u1 * (cnt0 - 1)).astype(np.int64), np.maximum(cnt0 - 2, 0))
    i1s = i1s + (i1s >= i0)
    i1d = np.minimum((u1 * cnt1).astype(np.int64), cnt1 - 1)
    return i0, np.where(same, i1s, i1d)


def met_draw(mb, base, sub, rho, N, trials, plist, key, stream, rep):
    """MET's own paired draw (vectorized): every trial takes N/2 distinct eligible prompts; each of a prompt's 2
    requests is routed to the substitute with probability rho and takes a stored response of its source (two distinct
    responses when both requests use the same source); the reference side is 2 distinct reference samples of the
    prompt. Returns (trials,) MMD^2, or None when fewer than N/2 prompts are eligible."""
    n2 = N // 2
    srcs = [base] + ([sub] if sub else [])
    el = [p for p in plist if all(len(mb.bank[s].get(p, ())) >= 2 for s in srcs) and len(mb.ref.get(p, ())) >= 2]
    if len(el) < n2:
        return None
    h = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
    rng = np.random.default_rng([20261026, h, stream, rep])
    Ab, cb = _stack(mb.bank[base], el, mb.L)
    As, cs = _stack(mb.bank[sub], el, mb.L) if sub else (Ab, cb)
    Ar, cr = _stack(mb.ref, el, mb.L)
    out = np.empty(trials)
    for t0 in range(0, trials, 250):                        # chunks keep the arrays small
        T = min(250, trials - t0)
        pi = np.argpartition(rng.random((T, len(el))), n2 - 1, axis=1)[:, :n2]     # (T, n2) eligible-prompt rows
        routed = (rng.random((T, n2, 2)) < rho) if sub else np.zeros((T, n2, 2), bool)
        u = rng.random((T, n2, 4))
        c0 = np.where(routed[..., 0], cs[pi], cb[pi])
        c1 = np.where(routed[..., 1], cs[pi], cb[pi])
        i0, i1 = _pick2(u[..., 0], u[..., 1], c0, c1, (routed[..., 0] == routed[..., 1]) | (base == sub))
        x0 = np.where(routed[..., 0, None], As[pi, np.minimum(i0, As.shape[1] - 1)], Ab[pi, np.minimum(i0, Ab.shape[1] - 1)])
        x1 = np.where(routed[..., 1, None], As[pi, np.minimum(i1, As.shape[1] - 1)], Ab[pi, np.minimum(i1, Ab.shape[1] - 1)])
        r0, r1 = _pick2(u[..., 2], u[..., 3], cr[pi], cr[pi], np.ones((T, n2), bool))
        y0, y1 = Ar[pi, r0], Ar[pi, r1]
        kxx = np.mean(x0 == x1, -1).mean(1)
        kyy = np.mean(y0 == y1, -1).mean(1)
        kxy = (np.mean(x0 == y0, -1) + np.mean(x0 == y1, -1) + np.mean(x1 == y0, -1) + np.mean(x1 == y1, -1)).mean(1) / 4
        out[t0: t0 + T] = kxx - 2 * kxy + kyy
    return out


def met_null(mb, N, trials, plist):
    """MET's native null: two reference samples play the API (prompts with >= 4 references)"""
    n2 = N // 2
    el = [p for p in plist if len(mb.ref.get(p, ())) >= 4]
    if len(el) < n2:
        return None
    rng = np.random.default_rng([20261026, 17, N])
    out = np.empty(trials)
    for t in range(trials):
        ch = rng.choice(len(el), n2, replace=False)
        X = np.empty((n2, 2, mb.L), np.int64)
        Y = np.empty((n2, 2, mb.L), np.int64)
        for i, c in enumerate(ch):
            four = mb.ref[el[c]][rng.choice(len(mb.ref[el[c]]), 4, replace=False)]
            X[i], Y[i] = four[:2], four[2:]
        out[t] = mmd_pairs(X, Y)
    return out



def select(P, common, sels):
    out = []
    for p in common:
        for s in sels:
            pre, _, sp = s.partition(":")
            if p.startswith(pre) and (not sp or P[p]["split"] == sp):
                out.append(p)
                break
    return out


def setup(a, D):
    cov = D.get("coverage", {})
    # Do not load substitutes without a coverage record of at least 0.99.
    D["subst_all"] = [list(x) for x in D["subst"]]
    D["subst"] = [x for x in D["subst"] if cov.get(x[1]) is not None and cov[x[1]] >= 0.99]
    names = list(dict.fromkeys([v for pool in D["pools"].values() for v in pool] + D["honest"] +
                               [x for s in D["subst"] for x in s]))
    cfg = json.load(open(D["cfg"], encoding="utf-8"))
    methods = [m for m in D["methods"] if m != MET]
    # top-20 grids: Z and ZN always (the top-20 MLE controls read z from them), the method's own grid, and the
    # penalty-score grids of the penalty-profiling methods; the S / V families need no grid
    grids = sorted({"Z", "ZN"} | {EN.METHODS[m][1] for m in methods if kind(m) in ("prof", "fixed")
                                  and EN.METHODS[m][1] in EN.GRID_FIELDS} |
                   {g for m in methods if kind(m) == "prof" and EN.METHODS[m][2] for g in EN.PEN_OF[EN.METHODS[m][1]]})
    tag = D["panel"]
    pool_names = {v for pool in D["pools"].values() for v in pool}
    td = D.get("test_dirs") or {}

    def link(key, suffix):
        """pool members from D[key]; the other sources from D["test_dirs"][key] when given (same link directory)"""
        if not D.get(key):
            return None
        if key not in td:
            return linkdir(tag + "_" + key, D[key], names, suffix)
        d = linkdir(tag + "_" + key, D[key], [n for n in names if n in pool_names], suffix)
        linkdir(tag + "_" + key, td[key], [n for n in names if n not in pool_names], suffix)
        return d
    gc = link("gridcache", ".all.v3cond2.pkl")
    sv = link("sv_dir", ".sv.pkl")
    sf = link("svfull_dir", ".svfull.pkl")
    EN.Panel.__init__ = lambda self, *x, **k: _init(self, *x, sv_dir=sv, svfull_dir=sf, **k)
    pan = EN.Panel(cfg, gc, names, grids=tuple(grids), device=a.device, rut=bool(D.get("rut")))
    P = pan.P
    CAL = select(P, pan.common, D["cal"])
    TEST = select(P, pan.common, D["test"])
    HALF = {h: select(P, pan.common, s) for h, s in D.get("halves", {}).items()}
    assert CAL and TEST and not set(CAL) & set(TEST), "calibration and test prompt domains must be disjoint"
    for h, pl in HALF.items():
        assert set(pl) <= set(TEST), h
    cov = D.get("coverage", {})
    plain = [e for e in D["plain"] if cov.get(e) is not None and cov[e] >= 0.99]   # no record -> not plain_eff
    mb = MetBank(cfg, names, D["met"]["L"]) if (MET in D["methods"] and D.get("met")) else None
    return pan, names, CAL, TEST, HALF, plain, mb



def run_main(a, D, pan, CAL, TEST, HALF, plain, mb):
    methods = D["methods"]
    arms = D["pools"]
    members = list(dict.fromkeys(v for pool in arms.values() for v in pool))
    sets = {"TEST": TEST, **HALF}
    out = {"design": D, "methods": methods, "n_cal_prompts": len(CAL), "n_test_prompts": len(TEST),
           "n_half": {h: len(p) for h, p in HALF.items()}, "plain_eff": plain,
           "plain_dropped_coverage": [e for e in D["plain"] if e not in plain],
           "plain_no_coverage_record": [e for e in D["plain"] if D.get("coverage", {}).get(e) is None],
           "scorer": getattr(pan, "scorer_ids", {}), "args": vars(a), "res": {}, "seconds": {}}
    npz = {}
    t0 = time.time()
    cov = D.get("coverage", {})
    covered_subs = {s for _, s in D["subst"] if cov.get(s) is not None and cov[s] >= 0.99}
    out["subst_uncovered"] = sorted({s for _, s in D.get("subst_all", D["subst"])} - covered_subs)
    ev, od = np.arange(0, a.trials_null, 2), np.arange(1, a.trials_null, 2)
    for n in [int(x) for x in a.ns.split(",")]:
        tn = time.time()
        R = out["res"].setdefault(str(n), {})
        cal_e, cal_o = {}, {}
        for v in members:
            d = pan.draw(("null", v, None, None, n), a.trials_null, plist=CAL, split_even=True)
            assert d is not None, f"calibration member {v}: fewer than {n} eligible prompts"
            st = stats(pan, d[0], d[1], methods)
            if mb:
                st[MET] = met_draw(mb, v, None, 0.0, n, a.trials_null, CAL, f"null|{v}|{n}", 1, 0)
            cal_e[v], cal_o[v] = take(st, ev), take(st, od)
        conds = {f"honest|{h}": ("null", h, None, None, n) for h in D["honest"]}
        for base, sub in D["subst"]:
            if sub not in covered_subs:                 # Uncovered substitutes remain unavailable.
                continue
            for r in D["rhos"]:
                conds[f"sub|{base}|{sub}|{r}"] = ("alt", base, sub, f"random_{r}", n)
        test = {}
        for sn, pl in sets.items():
            test[sn] = {}
            for cn, c in conds.items():
                tr = a.trials_null if cn.startswith("honest") else a.trials_alt
                d = pan.draw(c, tr, plist=pl, split_even=False)
                if d is None:
                    continue
                st = stats(pan, d[0], d[1], methods)
                if mb:
                    st[MET] = met_draw(mb, c[1], c[2], float(c[3].split("_")[1]) if c[3] else 0.0, n, tr, pl,
                                       f"{sn}|{cn}|{n}", 1, 0)
                test[sn][cn] = st
        native = {}
        if mb:
            mn = met_null(mb, n, 1000, TEST)
            native["met_thr"] = q_higher(mn) if mn is not None else None
        missing = sorted({cn for cn in conds if cn not in test["TEST"]})
        print(f"N{n}: calibration {len(members)} members, test {len(test['TEST'])} conditions "
              f"(missing {missing}) {time.time() - t0:.0f}s", flush=True)
        for arm, mem in arms.items():
            A = R.setdefault(arm, {})
            for m in methods:
                if m == MET and any(cal_e[v].get(MET) is None for v in mem):
                    A[m] = None                                   # not computable for this arm
                    continue
                rule, _ = calibrate(m, {v: cal_e[v][m] for v in mem}, mem)
                ex = rule["exact"]
                x = dict(rule, members=mem)
                # diagnostics per member: held-out = odd trials under the arm's rule; LOVO = odd trials under the rule
                # calibrated without the member; integer and randomized (exact) versions
                held, held_x, lovo, lovo_x = {}, {}, {}, {}
                for v in mem:
                    so = transform(m, cal_o[v][m], rule["tau"])
                    held[v], held_x[v] = rate_int(so, rule["thr"]), rate_ex(so, ex["k"], ex["gamma"])
                    rest = [u for u in mem if u != v]
                    rr, _ = calibrate(m, {u: cal_e[u][m] for u in rest}, rest)
                    sl = transform(m, cal_o[v][m], rr["tau"])
                    lovo[v] = rate_int(sl, rr["thr"])
                    lovo_x[v] = rate_ex(sl, rr["exact"]["k"], rr["exact"]["gamma"])
                x["old_gate"] = {"heldout": held, "lovo": lovo, "heldout_exact": held_x, "lovo_exact": lovo_x,
                                 "max_heldout": max(held.values()), "max_lovo": max(lovo.values()),
                                 "max_heldout_exact": max(held_x.values()), "max_lovo_exact": max(lovo_x.values()),
                                 "argmax_lovo": max(lovo, key=lovo.get),
                                 "gate_old": bool(max(held.values()) <= 0.07 and max(lovo.values()) <= 0.10)}
                for sn in sets:
                    ep, epx, pw, pwx, nat = {}, {}, {}, {}, {}
                    for cn, st in test[sn].items():
                        if st.get(m) is None:
                            continue
                        s = transform(m, st[m], rule["tau"])
                        key = cn.split("|", 1)[1]
                        (ep if cn.startswith("honest") else pw)[key] = rate_int(s, rule["thr"])
                        (epx if cn.startswith("honest") else pwx)[key] = rate_ex(s, ex["k"], ex["gamma"])
                        if kind(m) in ("rut", "rutiu"):
                            nat[cn] = float(np.mean(np.asarray(st[m]["p"]) < 0.05))
                        elif m == MET and native.get("met_thr") is not None:
                            nat[cn] = float(np.mean(np.asarray(st[m]) > native["met_thr"]))
                    x[sn] = {"endpoint": ep, "endpoint_exact": epx, "power": pw, "power_exact": pwx, "native": nat}
                    pl = [e for e in plain if e in ep]
                    x[sn]["missing_plain"] = [e for e in plain if e not in ep]
                    for rk, key in (("int", "endpoint"), ("exact", "endpoint_exact")):
                        vals = sorted(((x[sn][key][e], e) for e in pl), reverse=True)
                        x[sn][f"plain_max_{rk}"] = vals[0][0] if vals else None
                        x[sn][f"plain_argmax_{rk}"] = vals[0][1] if vals else None
                        x[sn][f"plain_second_{rk}"] = vals[1][0] if len(vals) > 1 else None
                x["redraws"] = {"thr": [], "exact": [], "tau": []}
                for key in ("T", "ret", "ovr"):
                    if (key, m) in cal_e[mem[0]]:
                        x[f"{key}_mean"] = {**{f"cal|{v}": float(np.mean(cal_e[v][(key, m)])) for v in mem},
                                            **{cn: float(np.mean(st[(key, m)])) for cn, st in test["TEST"].items()
                                               if (key, m) in st}}
                A[m] = x
        # per-trial statistics of the main pass (for independent recomputation)
        for v in members:
            for m in methods:
                if kind(m) == "det" or cal_e[v].get(m) is None:
                    continue
                npz[f"N{n}##cal|{v}##{m}"] = (transform(m, cal_e[v][m]).astype(np.float32) if kind(m) in CONTINUOUS
                                             else np.asarray(cal_e[v][m], np.int16))
        for cn, st in test["TEST"].items():
            for m in methods:
                if st.get(m) is None:
                    continue
                if kind(m) == "det":
                    for arm in arms:
                        if R[arm].get(m):
                            npz[f"N{n}##{cn}##{m}##{arm}"] = transform(m, st[m], R[arm][m]["tau"]).astype(np.int16)
                else:
                    npz[f"N{n}##{cn}##{m}"] = (transform(m, st[m]).astype(np.float32) if kind(m) in CONTINUOUS
                                              else np.asarray(st[m], np.int16))
        if n == E6_N:                       # Save trial rows and fitted temperatures for score comparisons.
            for v in members:
                d = pan.draw(("null", v, None, None, n), a.trials_null, plist=CAL, split_even=True)
                npz[f"N{n}##cal|{v}##rows"] = d[0].astype(np.int32)
            for cn, c in conds.items():
                d = pan.draw(c, a.trials_null if cn.startswith("honest") else a.trials_alt, plist=TEST,
                             split_even=False)
                if d is not None:
                    npz[f"N{n}##{cn}##rows"] = d[0].astype(np.int32)
            for v in members:
                for m in HF_FAMILY:
                    if m in methods and ("theta", m) in cal_e[v]:
                        th = np.empty(a.trials_null, np.int16)
                        th[ev], th[od] = cal_e[v][("theta", m)] // EN.NAG, cal_o[v][("theta", m)] // EN.NAG
                        npz[f"N{n}##cal|{v}##{m}##tk"] = th
            for cn, st in test["TEST"].items():
                for m in HF_FAMILY:
                    if m in methods and ("theta", m) in st:
                        npz[f"N{n}##{cn}##{m}##tk"] = (np.asarray(st[("theta", m)]) // EN.NAG).astype(np.int16)
            json.dump([[nm_, k[0], int(k[1])] for nm_, k in pan.rowkey], open(a.out + ".rowkey.json", "w", encoding="utf-8"))
        del cal_o
        # calibration re-draws: the even (calibration) stream re-drawn with rep_cal = r; test statistics fixed
        for r in range(1, a.cal_reps + 1):
            calr = {}
            for v in members:
                d = pan.draw(("null", v, None, None, n), a.trials_null, plist=CAL, split_even=True, rep_cal=r)
                rows_e = d[0][ev]
                st = stats(pan, rows_e, d[1][ev], methods)
                if mb:
                    st[MET] = met_draw(mb, v, None, 0.0, n, len(ev), CAL, f"null|{v}|{n}", 0, r)
                calr[v] = st
            for arm, mem in arms.items():
                for m in methods:
                    x = R[arm][m]
                    if x is None:
                        continue
                    rule, _ = calibrate(m, {v: calr[v][m] for v in mem}, mem)
                    x["redraws"]["thr"].append(rule["thr"])
                    x["redraws"]["exact"].append([rule["exact"]["k"], rule["exact"]["gamma"]])
                    x["redraws"]["tau"].append(rule["tau"])
                    for sn in sets:
                        pl = [e for e in plain if f"honest|{e}" in test[sn] and test[sn][f"honest|{e}"].get(m) is not None]
                        si = {e: transform(m, test[sn][f"honest|{e}"][m], rule["tau"]) for e in pl}
                        ri = {e: rate_int(si[e], rule["thr"]) for e in pl}
                        rx = {e: rate_ex(si[e], rule["exact"]["k"], rule["exact"]["gamma"]) for e in pl}
                        for rk, rr in (("int", ri), ("exact", rx)):
                            worst = x[sn][f"plain_argmax_{rk}"]          # the main pass's worst plain source
                            x["redraws"].setdefault(f"plain_max_{rk}_{sn}", []).append(max(rr.values(), default=None))
                            x["redraws"].setdefault(f"plain_max_wo_worst_{rk}_{sn}", []).append(
                                max([v for e, v in rr.items() if e != worst], default=None))
            if r % 5 == 0:
                print(f"N{n}: re-draw {r}/{a.cal_reps} {time.time() - t0:.0f}s", flush=True)
        
        for arm in arms:
            for m in methods:
                x = R[arm][m]
                if x is None:
                    continue
                x["verdict"] = {}
                for sn in sets:
                    for rk in ("int", "exact"):
                        mx = x[sn][f"plain_max_{rk}"]
                        reps = x["redraws"].get(f"plain_max_{rk}_{sn}", [])
                        keep = int(sum(v is not None and v <= GATE for v in reps))
                        unmeasured = bool(x[sn]["missing_plain"]) or mx is None
                        ok_main = (not unmeasured) and mx <= GATE
                        sens = {}
                        for g in (0.08, 0.10, 0.12):                # the whole rule at another plain threshold
                            kg = int(sum(v is not None and v <= g for v in reps))
                            sens[str(g)] = None if unmeasured else bool(mx <= g and kg >= REDRAW_KEEP)
                        second = x[sn][f"plain_second_{rk}"]
                        keep_wo = int(sum(v is not None and v <= GATE
                                          for v in x["redraws"].get(f"plain_max_wo_worst_{rk}_{sn}", [])))
                        x["verdict"][f"{sn}_{rk}"] = {
                            "valid": None if unmeasured else bool(ok_main and keep >= REDRAW_KEEP),
                            "main_ok": None if unmeasured else bool(ok_main), "unmeasured": unmeasured,
                            "redraws_kept": keep, "n_redraws": len(reps), "plain_max": mx,
                            "plain_argmax": x[sn][f"plain_argmax_{rk}"], "gate_sensitivity": sens,
                            "valid_without_worst": None if unmeasured else bool(
                                second is not None and second <= GATE and keep_wo >= REDRAW_KEEP)}
                v = x["verdict"]["TEST_int"]
                print(f"N{n} {arm:9s} {m:18s} thr {x['thr']:.2f} ({x['decisive']}) exact k {x['exact']['k']:.2f} "
                      f"g {x['exact']['gamma']:.2f} | plain max {v['plain_max'] if v['plain_max'] is None else round(v['plain_max'], 3)}"
                      f" ({v['plain_argmax']}) kept {v['redraws_kept']}/{v['n_redraws']} valid {v['valid']} | " +
                      " ".join(f"{k.split('|')[1][:16]}@{k.split('|')[-1]}={val:.3f}"
                               for k, val in x["TEST"]["power"].items() if k.endswith("|0.1")), flush=True)
        out["res"][str(n)]["_native"] = native
        out["seconds"][str(n)] = round(time.time() - tn, 1)
        json.dump(out, open(a.out + ".json", "w", encoding="utf-8"))
        np.savez_compressed(a.out + ".npz", **npz)
        print(f"N{n} done {time.time() - t0:.0f}s", flush=True)
    return out



def run_boot(a, D, pan, CAL, TEST, plain, mb):
    methods = [m for m in D["methods"] if m != MET or mb is not None]
    arms = D["pools"]
    members = list(dict.fromkeys(v for pool in arms.values() for v in pool))
    n = a.boot_n
    cov = D.get("coverage", {})
    cells = [tuple(c) for c in D.get("boot_cells", []) if cov.get(c[1]) is not None and cov[c[1]] >= 0.99]
    items = [f"sub|{b}|{s}|{r}" for b, s, r in cells] + [f"honest|{e}" for e in plain]
    lo, hi = [int(x) for x in a.brange.split(":")]
    arr = np.full((hi - lo, len(arms), len(methods), len(items), 2), np.nan, np.float32)
    thr = np.full((hi - lo, len(arms), len(methods), 3), np.nan, np.float64)
    resample_cal = bool(D.get("boot_resample_cal", False))
    t0 = time.time()
    for bi, b in enumerate(range(lo, hi)):
        rng = np.random.default_rng(100000 + b)
        tl = [TEST[i] for i in rng.integers(len(TEST), size=len(TEST))]
        cl = [CAL[i] for i in rng.integers(len(CAL), size=len(CAL))] if resample_cal else CAL
        cal = {}
        for v in members:
            d = pan.draw_gpu(("null", v, None, None, n), a.boot_trials_cal, b=b, plist=cl, split_even=False)
            cal[v] = stats(pan, d[0], d[1], methods)
            if mb is not None:                       # MET's own paired draw with replicate streams
                cal[v][MET] = met_draw(mb, v, None, 0.0, n, a.boot_trials_cal, cl, f"boot|null|{v}|{n}", 100 + b, 0)
        test = {}
        for it in items:
            if it.startswith("honest"):
                c, tr = ("null", it.split("|", 1)[1], None, None, n), a.boot_trials_honest
            else:
                _, base, sub, r = it.split("|")
                c, tr = ("alt", base, sub, f"random_{r}", n), a.trials_alt
            d = pan.draw_gpu(c, tr, b=b, plist=tl, split_even=False)
            if d is not None:
                test[it] = stats(pan, d[0], d[1], methods)
                if mb is not None:
                    test[it][MET] = met_draw(mb, c[1], c[2], float(c[3].split("_")[1]) if c[3] else 0.0, n, tr, tl,
                                             f"boot|{it}|{n}", 100 + b, 0)
        for ai, (arm, mem) in enumerate(arms.items()):
            for mi, m in enumerate(methods):
                if m == MET and any(cal[v].get(MET) is None for v in mem):
                    continue                                 # MET not computable for this arm (stays NaN)
                rule, _ = calibrate(m, {v: cal[v][m] for v in mem}, mem)
                ex = rule["exact"]
                thr[bi, ai, mi] = [rule["thr"], ex["k"], ex["gamma"]]
                for ii, it in enumerate(items):
                    if it in test and test[it].get(m) is not None:
                        s = transform(m, test[it][m], rule["tau"])
                        arr[bi, ai, mi, ii] = [rate_int(s, rule["thr"]), rate_ex(s, ex["k"], ex["gamma"])]
        if bi % 10 == 0:
            print(f"bootstrap {b} ({bi + 1}/{hi - lo}) {time.time() - t0:.0f}s", flush=True)
            np.savez_compressed(a.out + f".boot_{lo}_{hi}.npz", rates=arr, thr=thr, done=bi + 1)
    np.savez_compressed(a.out + f".boot_{lo}_{hi}.npz", rates=arr, thr=thr, done=hi - lo)
    json.dump({"arms": list(arms), "methods": methods, "items": items, "rules": ["int", "exact"], "N": n,
               "brange": [lo, hi], "resample_cal": resample_cal, "seconds": round(time.time() - t0, 1),
               "trials": {"cal": a.boot_trials_cal, "honest": a.boot_trials_honest, "alt": a.trials_alt}},
              open(a.out + f".boot_{lo}_{hi}.json", "w", encoding="utf-8"), indent=1)
    print("bootstrap saved", time.time() - t0, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="100,200,400")
    ap.add_argument("--trials_null", type=int, default=4000)
    ap.add_argument("--trials_alt", type=int, default=1000)
    ap.add_argument("--cal_reps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--boot", action="store_true")
    ap.add_argument("--brange", default="0:2000")
    ap.add_argument("--boot_n", type=int, default=200)
    ap.add_argument("--boot_trials_cal", type=int, default=2000)
    ap.add_argument("--boot_trials_honest", type=int, default=1000)
    a = ap.parse_args()
    D = json.load(open(a.design, encoding="utf-8"))
    pan, names, CAL, TEST, HALF, plain, mb = setup(a, D)
    print(f"panel {D['panel']}: {len(names)} sources, CAL {len(CAL)} TEST {len(TEST)} halves "
          f"{ {h: len(p) for h, p in HALF.items()} } plain {len(plain)} MET {'yes' if mb else 'no'}", flush=True)
    if a.boot:
        run_boot(a, D, pan, CAL, TEST, plain, mb)
    else:
        run_main(a, D, pan, CAL, TEST, HALF, plain, mb)


if __name__ == "__main__":
    main()
