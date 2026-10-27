"""Compute profile-based typicality, response scores, RUT and paired MET.

The temperature/penalty grid supports full top-20 windows and renormalized
nuclei. Profiling first aligns temperature for each penalty pair, then selects
the pair whose penalty-score sums are closest to zero. Conditional variants
use only positions whose selected token is in their retained support.

Response fields include z grids, NLL, mean log midrank and window entropy.
RUT uses stored local references. MET uses a separate paired draw from prompts
with at least two samples on each side. Missing fields remain unavailable.
Results are stored by condition, field and trial."""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

try:      # Disable transparent huge pages to keep forked grid caches shared.
    import ctypes                                     
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(41, 1, 0, 0, 0)   # PR_SET_THP_DISABLE.
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tvmix2 import TG, Src, z_grid  # noqa: E402
from cmp_eval_nat import PAD, draw_pairs, draw_singles, met_ids, mmd_pairs  # noqa: E402
from rut import load_scored_multi, logrank_mid, rand_rank, cvm_p  # noqa: E402
from scenarios import make_policies  # noqa: E402

AG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
GG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
TCHUNK = 151
HUBER = 1.345
I10 = int(np.argmin(np.abs(TG - 1.0)))
SUPPORTS = [(20, 0.8), (20, 0.95), (20, 1.0), (None, 0.8), (None, 0.95), (None, 1.0)]   # strict -> loose
DELTAS = [0.0, 0.3]
PW = [0.8, 0.95, 1.0]                                       # nuclei of the window-conditional z
PN = 0.95                                                   # nucleus of the conditional ProTA-N95
DOMS = {("wildchat", "en"): 0, ("wildchat", "zh"): 1, ("bigcodebench", "en"): 2, ("math", "en"): 3}
G = {}


def attach_tid(src, prefix):
    z = np.load(prefix + ".npz")
    off, tid = z["offsets"], z["topk_id"]
    for j, l in enumerate(open(prefix + ".index.jsonl", encoding="utf-8")):
        r = json.loads(l)
        key = (r["pid"], r["k"])
        if r.get("n", 0) > 0 and key in src.sc:
            src.sc[key]["tid"] = tid[off[j]: off[j + 1]]


def grids(r, ids, pids):
    """Compute z, presence-score and repetition-score grids over (T, a, g)."""
    lp = r["lp"].astype(np.float64)
    L = len(lp)
    tl = r["tlp"].astype(np.float64)
    tid = r["tid"]
    ok = ~np.isnan(tl)
    Ro = np.zeros(tl.shape, bool)       # candidate generated so far (presence penalty)
    Rp = np.zeros(tl.shape, bool)       # candidate in prompt or generated so far (repetition penalty)
    yo = np.zeros(L, bool)
    yp = np.zeros(L, bool)
    seen, pset = set(), set(pids)
    for t in range(L):
        Ro[t] = [int(x) in seen for x in tid[t]]
        Rp[t] = [int(x) in seen or int(x) in pset for x in tid[t]]
        yo[t] = ids[t] in seen
        yp[t] = ids[t] in seen or ids[t] in pset
        seen.add(ids[t])
    Ro &= ok
    Rp &= ok
    rk = r["rk"]
    ins = (rk >= 1) & (rk <= tid.shape[1])
    ycol = np.full(L, -1, np.int16)
    for t in range(L):
        if ins[t]:
            w = np.nonzero(tid[t] == ids[t])[0]
            if len(w):
                ycol[t] = int(w[0])
    yin = ycol >= 0                                                                   # chosen token in the window
    yc = np.maximum(ycol, 0).astype(np.int64)
    nT, nA, nG = len(TG), len(AG), len(GG)
    zW, saW, sgW = (np.empty((nT, nA, nG)) for _ in range(3))
    zN, saN, sgN = (np.empty((nT, nA, nG)) for _ in range(3))
    z = np.empty((nT, nA, nG))
    sa = np.empty((nT, nA, nG))
    sg = np.empty((nT, nA, nG))
    A = AG[None, :, None, None, None]
    Gm = GG[None, None, :, None, None]
    base = np.where(ok, tl, 0.0)[None, None, None] - A * Ro[None, None, None] - Gm * Rp[None, None, None]
    pen_y = A[..., 0] * yo[None, None, None] + Gm[..., 0] * yp[None, None, None]           # 1,nA,nG,L
    order = np.argsort(-np.where(ok[None, None, None], base, -np.inf), axis=-1, kind="stable")   # T-free ranking
    inv = np.argsort(order, axis=-1, kind="stable")
    ycol_i = yc[None, None, None, :, None]
    for c0 in range(0, nT, TCHUNK):
        T = TG[c0: c0 + TCHUNK, None, None, None, None]
        u = np.where(ok[None, None, None], base / T, -np.inf)                              # t,nA,nG,L,K
        m = u.max(-1, keepdims=True)
        e = np.exp(u - m)
        s_ = e.sum(-1, keepdims=True)
        q = e / s_
        lse = (m + np.log(s_))[..., 0]
        logq = np.where(q > 0, u - lse[..., None], 0.0)
        H = -(q * logq).sum(-1)
        var = np.maximum((q * logq ** 2).sum(-1) - H ** 2, 1e-6)
        sy = -((lp[None, None, None] - pen_y) / T[..., 0] - lse)
        z[c0: c0 + TCHUNK] = (sy - H).sum(-1) / np.sqrt(var.sum(-1))
        # Use the stored fp16 window column for both target surprisal and moments.
        
        
        sw = -np.take_along_axis(logq, np.broadcast_to(ycol_i, logq.shape[:-1] + (1,)), axis=-1)[..., 0]
        sa[c0: c0 + TCHUNK] = (yo[None, None, None] - (q * Ro[None, None, None]).sum(-1)).sum(-1)
        sg[c0: c0 + TCHUNK] = (yp[None, None, None] - (q * Rp[None, None, None]).sum(-1)).sum(-1)
        # W: only positions whose chosen token is in the window; surprisal sw = -log q(y) from the window column
        mw = yin[None, None, None]
        Vw = np.where(mw, var, 0.0).sum(-1)
        zW[c0: c0 + TCHUNK] = np.where(mw, sw - H, 0.0).sum(-1) / np.sqrt(np.maximum(Vw, 1e-12))
        saW[c0: c0 + TCHUNK] = np.where(mw, yo[None, None, None] - (q * Ro[None, None, None]).sum(-1), 0.0).sum(-1)
        sgW[c0: c0 + TCHUNK] = np.where(mw, yp[None, None, None] - (q * Rp[None, None, None]).sum(-1), 0.0).sum(-1)
        # N95: nucleus of q (sorted by the T-free ranking), q renormalized to it; positions with y in the nucleus
        qs = np.take_along_axis(q, np.broadcast_to(order, q.shape), axis=-1)
        keep_s = (np.cumsum(qs, -1) - qs) < PN
        keep_s[..., 0] = True
        keep = np.take_along_axis(keep_s, np.broadcast_to(inv, q.shape), axis=-1) & ok[None, None, None]
        mass = np.maximum((q * keep).sum(-1), 1e-300)
        qn = np.where(keep, q / mass[..., None], 0.0)
        lqn = np.where(qn > 0, np.log(np.maximum(qn, 1e-300)), 0.0)
        Hn = -(qn * lqn).sum(-1)
        varn = np.maximum((qn * lqn ** 2).sum(-1) - Hn ** 2, 1e-6)
        yn = mw & np.take_along_axis(keep, np.broadcast_to(ycol_i, q.shape[:-1] + (1,)), axis=-1)[..., 0]
        syn = sw + np.log(mass)
        Vn = np.where(yn, varn, 0.0).sum(-1)
        zN[c0: c0 + TCHUNK] = np.where(yn, syn - Hn, 0.0).sum(-1) / np.sqrt(np.maximum(Vn, 1e-12))
        saN[c0: c0 + TCHUNK] = np.where(yn, yo[None, None, None] - (qn * Ro[None, None, None]).sum(-1), 0.0).sum(-1)
        sgN[c0: c0 + TCHUNK] = np.where(yn, yp[None, None, None] - (qn * Rp[None, None, None]).sum(-1), 0.0).sum(-1)
    hit_bad = int(np.sum(ins & (ycol < 0)))
    # detector family (per response): mean NLL, mean mid-rank log-rank (RUT's score), mean top-20 entropy at T = 1
    p1 = np.where(ok, np.exp(tl - np.log(np.where(ok, np.exp(tl), 0.0).sum(1, keepdims=True))), 0.0)
    ent = float(np.mean(-(p1 * np.where(p1 > 0, np.log(np.maximum(p1, 1e-300)), 0.0)).sum(1))) if L else 0.0
    det = np.array([float(-np.mean(lp)) if L else 0.0, logrank_mid(r) if L else 0.0, ent], np.float64)
    compact = dict(tl=tl.astype(np.float32), Ro=Ro, Rp=Rp, ycol=ycol, lp=lp.astype(np.float32), yo=yo, yp=yp,
                   m1=np.where(ok, np.exp(tl), 0.0).sum(1).astype(np.float32))
    cond = {k: v.astype(np.float32) for k, v in (("zW", zW), ("saW", saW), ("sgW", sgW), ("zN", zN), ("saN", saN),
                                                    ("sgN", sgN))}
    return (z.astype(np.float32), sa.astype(np.float32), sg.astype(np.float32), det, compact, hit_bad, int(np.sum(ins)),
            cond)


def support_stats(c, T, a, g):
    """Window-conditional z (nuclei PW) and violation bits for one response at decoding theta = (T, a, g)."""
    tl, ycol = c["tl"].astype(np.float64), c["ycol"]
    L = len(ycol)
    if L == 0:
        return np.zeros(len(PW), np.float32), 0
    b = np.where(np.isnan(tl), -np.inf, tl - a * c["Ro"] - g * c["Rp"])                 # L, K penalized logits
    by = c["lp"].astype(np.float64) - a * c["yo"] - g * c["yp"]                          # L
    order = np.argsort(-b, axis=1, kind="stable")
    bs = np.take_along_axis(b, order, 1)
    u = bs / T
    e = np.exp(u - u[:, :1])
    q = e / e.sum(1, keepdims=True)
    cb = np.cumsum(q, 1) - q
    inw = ycol >= 0
    pos = np.argmax(order == np.where(inw, ycol, -9)[:, None], axis=1)                  # sorted position (in-window)
    nvalid = np.sum(np.isfinite(bs), 1)
    zout = np.zeros(len(PW), np.float32)
    keepn = {}
    for ip, p in enumerate(PW):
        keep = (cb < p) if p < 1.0 else np.isfinite(bs)
        keep[:, 0] = True
        keepn[p] = keep.sum(1)
        qk = np.where(keep, q, 0.0)
        qp = qk / qk.sum(1, keepdims=True)
        lq = np.where(qp > 0, np.log(np.maximum(qp, 1e-300)), 0.0)
        H = -(qp * lq).sum(1)
        var = np.maximum((qp * lq ** 2).sum(1) - H ** 2, 1e-6)
        yk = inw & keep[np.arange(L), pos]
        ly = lq[np.arange(L), pos]
        D = np.where(yk, -ly - H, 0.0).sum()
        V = np.where(yk, var, 0.0).sum()
        zout[ip] = D / np.sqrt(V) if V > 0 else 0.0
    bits = 0
    b20 = bs[np.arange(L), np.maximum(nvalid - 1, 0)]          # lowest valid window logit (20th when the window is full)
    for si, (k, p) in enumerate(SUPPORTS):
        if k is None and p >= 1.0:
            continue
        if p < 1.0:
            blast = bs[np.arange(L), keepn[p] - 1]
            in_nuc = inw & (cb[np.arange(L), pos] < p) | (inw & (pos == 0))
        for di, dl in enumerate(DELTAS):
            v = np.zeros(L, bool)
            if p < 1.0:                                               # in-window, outside the nucleus by > Delta
                v |= inw & ~in_nuc & ((blast - by) > dl)
            if k == 20:                                               # outside the window: all 20 window tokens beat it
                v |= ~inw & (nvalid >= 20) & ((b20 - by) > dl)
            elif p < 1.0 and T <= 1.0 and a == 0.0 and g == 0.0:     # top-p only: nucleus inside the window for sure
                v |= ~inw & (c["m1"] > p) & (nvalid >= 20) & ((b20 - by) > dl)
            if v.any():
                bits |= 1 << (2 * si + di)
    return zout, bits


_SUPC = {}


def _sup_cached(name, key, c, iT, ia, ig, T_, a_, g_):
    """support_stats memoized per (response, profiled grid point): theta-hat takes few values within a condition."""
    ck = (name, key, iT, ia, ig)
    v = _SUPC.get(ck)
    if v is None:
        if len(_SUPC) > 2_000_000:
            _SUPC.clear()
        v = _SUPC[ck] = support_stats(c, T_, a_, g_)
    return v


def pre_worker(job):
    name, keys = job
    src = G["srcs"][name]
    out = []
    for k in keys:
        out.append((name, k, grids(src.sc[k], src.ids[k][: len(src.sc[k]["lp"])], G["prompt_ids"][k[0]])))
    return out


def huber(x):
    return np.clip(x, -HUBER, HUBER)


def profile_idx(Z, SA, SG, psi):
    """T from the (psi-)alignment equation for every penalty pair, then the penalty pair whose two score equations are
    closest to zero. Z, SA, SG: (N, nT, nA, nG). Returns the grid indices (iT, ia, ig)."""
    Tk = np.argmin(np.abs(psi(Z).mean(0)), 0)                                         # nA, nG
    ia, ig = np.meshgrid(np.arange(Z.shape[2]), np.arange(Z.shape[3]), indexing="ij")
    obj = SA[:, Tk, ia, ig].sum(0) ** 2 + SG[:, Tk, ia, ig].sum(0) ** 2
    a, g = np.unravel_index(int(np.argmin(obj)), obj.shape)
    return int(Tk[a, g]), int(a), int(g)


def profile(Z, SA, SG, psi):
    t, a, g = profile_idx(Z, SA, SG, psi)
    return Z[:, t, a, g], float(TG[t]), float(AG[a]), float(GG[g])


def trial_stats(resp, rng):
    """resp: list of (src, key) in draw order."""
    C = [G["cache"][(s.name, k)] for s, k in resp]
    Z = np.stack([c[0] for c in C])           # N, nT, nA, nG
    SA = np.stack([c[1] for c in C])
    SG = np.stack([c[2] for c in C])
    Z0 = Z[:, :, 0, 0]
    o = {}
    ti = int(np.argmin(np.abs(Z0.mean(0))))
    th = int(np.argmin(np.abs(huber(Z0).mean(0))))
    o["z_al"], o["z_hub"], o["z_fdg1"] = Z0[:, ti], Z0[:, th], Z0[:, I10]
    o["T_al"] = float(TG[ti])
    o["z_pen"], o["T_pen"], o["a_pen"], _ = profile(Z[..., :1], SA[..., :1], SG[..., :1] * 0, lambda x: x)
    o["z_dec"], o["T_dec"], o["a_dec"], o["g_dec"] = profile(Z, SA, SG, lambda x: x)
    o["z_penhub"], o["T_penhub"], o["a_penhub"], _ = profile(Z[..., :1], SA[..., :1], SG[..., :1] * 0, huber)
    for tag, cc in (("10", 1.0), ("20", 2.0)):          # Alternative Huber constants.
        o[f"z_dechub{tag}"] = profile(Z, SA, SG, lambda x, cc=cc: np.clip(x, -cc, cc))[0]
    iT, ia, ig = profile_idx(Z, SA, SG, huber)
    o["z_dechub"] = Z[:, iT, ia, ig]
    o["T_dechub"], o["a_dechub"], o["g_dechub"] = float(TG[iT]), float(AG[ia]), float(GG[ig])
    det = np.stack([c[3] for c in C])
    o["d_nll"], o["d_lrank"], o["d_ent"] = det[:, 0], det[:, 1], det[:, 2]
    for tag, kz, ka, kg in (("w", "zW", "saW", "sgW"), ("n95", "zN", "saN", "sgN")):     # conditional ProTA
        Zc = np.stack([c[5][kz] for c in C])
        jT, ja, jg = profile_idx(Zc, np.stack([c[5][ka] for c in C]), np.stack([c[5][kg] for c in C]), huber)
        o[f"z_{tag}"] = Zc[:, jT, ja, jg]
        o[f"T_{tag}"], o[f"a_{tag}"], o[f"g_{tag}"] = float(TG[jT]), float(AG[ja]), float(GG[jg])
    if False:                                    # Support statistics are disabled.
        T_, a_, g_ = float(TG[iT]), float(AG[ia]), float(GG[ig])
        zw, vb = zip(*(_sup_cached(s.name, k, c[4], iT, ia, ig, T_, a_, g_) for (s, k), c in zip(resp, C)))
        zw = np.stack(zw)
        o["zw80"], o["zw95"], o["zw100"] = zw[:, 0], zw[:, 1], zw[:, 2]
        o["vb"] = np.array(vb, np.int32)
    dom = np.array([G["dom"][k[0]] for _, k in resp], np.int8)
    o["dom"] = dom
    zx = np.full(len(resp), np.nan, np.float32)
    for d in range(4):                         # ProTA-X: profile on the other domains, score this one
        m = dom == d
        if m.any() and (~m).any():             # no other-domain response: z_x stays NaN (never counted as exceedance)
            t2, a2, g2 = profile_idx(Z[~m], SA[~m], SG[~m], huber)
            zx[m] = Z[m, t2, a2, g2]
    o["z_x"] = zx
    if G["ref"] is not None:
        ranks = [rand_rank(c[3][1], G["ref"][k[0]][:100], rng) for c, (_, k) in zip(C, resp)]
        o["rut"] = float(cvm_p(np.array(ranks)))
    return o


def ref_pair(pid, rng, m=2):
    pool = G["ref_ids"][pid]
    sel = rng.choice(len(pool), m, replace=False)
    return [pool[j] for j in sel]


def met_stats(o2, used, rng):
    X = np.array([[met_ids(s.ids[k], max(G["met_L"])) for s, k in reqs] for reqs in o2])
    Y = np.array([[met_ids(y, max(G["met_L"])) for y in ref_pair(p, rng)] for p in used])
    return {f"met{L}": mmd_pairs(X[..., :L], Y[..., :L]) for L in G["met_L"]}


def run_condition(args):
    kind, bname, sname, pol, nreq, trials, seed = args
    if kind == "metnull":                    # MET native null: two local reference samples play the API
        pids = [p for p in G["common"] if len(G["ref_ids"].get(p, [])) >= 4]
        n2 = nreq // 2
        if len(pids) < n2:
            return (kind, bname, sname, pol, nreq), None
        rng = np.random.default_rng(seed)
        acc = {}
        for _ in range(trials):
            X, Y = [], []
            for ci in rng.choice(len(pids), n2, replace=False):
                four = ref_pair(pids[ci], rng, 4)
                X.append(four[:2]), Y.append(four[2:])
            X = np.array([[met_ids(x, max(G["met_L"])) for x in pr] for pr in X])
            Y = np.array([[met_ids(y, max(G["met_L"])) for y in pr] for pr in Y])
            for L in G["met_L"]:
                acc.setdefault(f"met{L}", []).append(mmd_pairs(X[..., :L], Y[..., :L]))
        return (kind, bname, sname, pol, nreq), {k: np.asarray(v, np.float32) for k, v in acc.items()}
    base, sub = G["srcs"][bname], (G["srcs"][sname] if sname else None)
    route = G["policies"][pol] if pol else None
    pairs = G["draw"] == "pairs"
    need = 2 if pairs else 1
    n = nreq // 2 if pairs else nreq
    pids = [p for p in G["common"] if len(base.by_pid.get(p, [])) >= need
            and (sub is None or len(sub.by_pid.get(p, [])) >= need)]
    if G["ref"] is not None:
        pids = [p for p in pids if len(G["ref"].get(p, [])) >= G["ref_min"]]
    if len(pids) < n:
        return (kind, bname, sname, pol, nreq), None
    pids2 = [p for p in pids if len(base.by_pid.get(p, [])) >= 2 and (sub is None or len(sub.by_pid.get(p, [])) >= 2)
             and len(G["ref_ids"].get(p, [])) >= 2] if G["met"] else []
    do_met = G["met"] and len(pids2) >= nreq // 2
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(trials):
        if pairs:
            o2, _ = draw_pairs(base, sub, route, n, rng, pids)
            resp = [rq for reqs in o2 for rq in reqs]
        else:
            resp = draw_singles(base, sub, route, n, rng, pids)
        st = trial_stats(resp, rng)
        if do_met:                             # MET's own N/2 x 2 draw, always from the MET-eligible prompts
            o2m, used = draw_pairs(base, sub, route, nreq // 2, rng, pids2)
            st.update(met_stats(o2m, used, rng))
        for k, v in st.items():           # Copy fields so saved results do not retain each trial's full grid.
            acc.setdefault(k, []).append(np.array(v, copy=True) if isinstance(v, np.ndarray) else v)  
            
    out = {}
    for k, v in acc.items():
        out[k] = np.asarray(v, np.float32) if np.ndim(v[0]) == 0 else np.asarray(v, v[0].dtype)
    return (kind, bname, sname, pol, nreq), out


def init(cfg):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(cfg["prompts"], encoding="utf-8")}
    G["policies"] = make_policies(P)
    sp = cfg.get("_split", "test")
    G["common"] = sorted(p for p, r in P.items() if sp == "all" or r["split"] in sp.split(","))
    if not G["common"]:
        raise SystemExit(f"split {sp!r} selects no prompt of {cfg['prompts']} (v3 splits are H1 / H2; use --split all)")
    G["dom"] = {p: DOMS[(P[p]["domain"], P[p]["lang"])] for p in G["common"]}
    ck = cfg.get("chat_kwargs", {"enable_thinking": False})
    G["prompt_ids"] = {}
    for p in G["common"]:                     # the claimed model's chat prompt, exactly as score_tf.py builds it
        ids = tok.apply_chat_template([{"role": "user", "content": P[p]["text"]}], add_generation_prompt=True,
                                      tokenize=True, **ck)
        G["prompt_ids"][p] = [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]   
    need = set(cfg["calib"] + cfg["honest"] + cfg["bases"] + cfg["subs"])
    G["srcs"] = {}
    maxk = cfg.get("max_k", {})
    for name in sorted(need):
        s, t = cfg["sources"][name]
        src = Src(name, s, t, tok)
        if name in maxk:                       # drop extra samples per prompt (pool homogeneity)
            for key in [kk for kk in src.sc if kk[1] >= maxk[name]]:
                src.sc.pop(key, None)
                src.ids.pop(key, None)
            src.by_pid = {}
            for (pid, k) in src.sc:
                if (pid, k) in src.ids:
                    src.by_pid.setdefault(pid, []).append(k)
        attach_tid(src, s)
        G["srcs"][name] = src
    G["ref_ids"] = {}
    if G["met"]:
        G["met_L"] = cfg.get("met_L", [30])
        cs = set(G["common"])
        for path in cfg["ref_text"]:
            for l in open(path, encoding="utf-8"):
                r = json.loads(l)
                if "_meta" in r or r["pid"] not in cs or not isinstance(r.get("text"), str):
                    continue
                G["ref_ids"].setdefault(r["pid"], []).append(tok.encode(r["text"], add_special_tokens=False)[: max(G["met_L"])])
        print("MET reference prompts with >= 4 samples:", sum(len(v) >= 4 for v in G["ref_ids"].values()), flush=True)
    G["ref"] = None
    G["ref_min"] = int(cfg.get("ref_min", 100))         # Required reference responses per prompt.
    if cfg.get("ref") and G["rut"]:
        refs = load_scored_multi(cfg["ref"])
        G["ref"] = {p: np.array([logrank_mid(r) for r in refs.get(p, [])]) for p in G["common"]}
    ex = G["prompt_ids"][G["common"][0]]
    assert len(ex) > 10 and all(isinstance(x, int) for x in ex), ex[:5]
    print("sources loaded", len(G["srcs"]), "| prompts", len(G["common"]), "| prompt ids of", G["common"][0], len(ex),
          f"| rut ref (>= {G['ref_min']} per prompt)", None if G["ref"] is None else
          sum(len(v) >= G["ref_min"] for v in G["ref"].values()), flush=True)


def consolidate_cache():
    """Replace response-grid arrays with views into contiguous memory mappings.
    This avoids retaining many heap allocations in forked workers."""
    keys = list(G["cache"])
    if not keys:
        return
    f0 = G["cache"][keys[0]]
    n = len(keys)
    big = {i: np.empty((n,) + f0[i].shape, f0[i].dtype) for i in (0, 1, 2, 3)}
    cnames = sorted(f0[5]) if f0[5] is not None else []
    cbig = {c: np.empty((n,) + f0[5][c].shape, f0[5][c].dtype) for c in cnames}
    for j, k in enumerate(keys):
        g = G["cache"][k]
        for i in (0, 1, 2, 3):
            big[i][j] = g[i]
        for c in cnames:
            cbig[c][j] = g[5][c]
    for j, k in enumerate(keys):
        G["cache"][k] = (big[0][j], big[1][j], big[2][j], big[3][j], None, {c: cbig[c][j] for c in cnames} or None)
    import gc
    gc.collect()
    G["_cache_big"] = (big, cbig)


def _guarded(it, stall_s=2700):
    """Abort if a multiprocessing iterator yields no result within stall_s seconds."""
    while True:
        try:
            yield it.next(timeout=stall_s)
        except StopIteration:
            return
        except mp.TimeoutError:
            raise SystemExit(f"pool stalled: no result for {stall_s}s (a worker was probably killed); rerun")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=96)
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--nreq", default="")
    ap.add_argument("--only", default="", help="comma list of source names (smoke test)")
    ap.add_argument("--split", default="test", help="split value(s), comma separated, or 'all'")
    ap.add_argument("--draw", default="singles", choices=["singles", "pairs"])
    ap.add_argument("--support", type=int, default=0, help="1: compute the ProTA-tau support quantities")
    ap.add_argument("--rut", type=int, default=1)
    ap.add_argument("--met", type=int, default=0, help="1: MET same-budget (+ metnull conditions)")
    ap.add_argument("--ag", default="")
    ap.add_argument("--gg", default="")
    ap.add_argument("--tchunk", type=int, default=151)
    ap.add_argument("--shard", default="", help="i/k: run only every k-th condition starting at i (merge_shards.py)")
    ap.add_argument("--grid_cache", default="", help="dir of per-source grid files: load when present and current, "
                    "else compute and save (a pure cache of pre_worker's output)")
    ap.add_argument("--grid_only", type=int, default=0, help="1: stop after the grids are computed and cached")
    ap.add_argument("--kinds", default="all", choices=["all", "null", "alt"],
                    help="run only the null conditions (pool variants, honest sources, MET null) or only the alternatives "
                         "(cmp3: nulls with more trials than alternatives; merge with merge_v3.py)")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    global AG, GG, TCHUNK
    if a.ag:
        AG = np.array([float(x) for x in a.ag.split(",")])
    if a.gg:
        GG = np.array([float(x) for x in a.gg.split(",")])
    if len(AG) == 0 or AG[0] != 0.0 or len(GG) == 0 or GG[0] != 0.0:
        ap.error("--ag / --gg must be nonempty and start with 0 (index 0 is read as the zero-penalty surface)")
    if a.tchunk <= 0:
        ap.error("--tchunk must be a positive integer")
    TCHUNK = a.tchunk
    cfg["_split"] = a.split
    G["draw"], G["support"], G["rut"], G["met"] = a.draw, bool(a.support), bool(a.rut), bool(a.met)
    print("split", a.split, "draw", a.draw, "support", a.support, "AG", AG.tolist(), "GG", GG.tolist(), flush=True)
    if a.only:
        keep = set(a.only.split(","))
        for f in ("calib", "honest", "bases", "subs"):
            cfg[f] = [v for v in cfg[f] if v in keep]
        cfg["alt"] = [x for x in cfg["alt"] if x[0] in keep and x[1] in keep]
    T = a.trials or cfg["trials"]
    nreq = [int(x) for x in a.nreq.split(",")] if a.nreq else cfg["nreq"]
    t0 = time.time()
    init(cfg)
    common = set(G["common"])
    jobs = []
    for name, src in G["srcs"].items():
        keys = [k for k in src.sc if k[0] in common and k in src.ids]
        for i in range(0, len(keys), 50):
            jobs.append((name, keys[i: i + 50]))
    G["cache"] = {}
    bad = tot = 0
    stamp = lambda name: [[p, os.path.getmtime(p)] for p in (cfg["sources"][name][0] + ".npz", cfg["sources"][name][1])]
    cfile = lambda name: os.path.join(a.grid_cache, f"{name}.{a.split.replace(',', '+')}.v3cond2.pkl")  
    cached = set()
    if a.grid_cache:
        os.makedirs(a.grid_cache, exist_ok=True)
        for name in G["srcs"]:
            if os.path.exists(cfile(name)):
                c = pickle.load(open(cfile(name), "rb"))
                if c["stamp"] == stamp(name) and c["keys"] == sorted(k for _, ks in jobs if _ == name for k in ks):
                    G["cache"].update({(name, k): g for k, g in c["grids"].items()})
                    bad += c["bad"]
                    tot += c["tot"]
                    cached.add(name)
        print(f"grid cache: {len(cached)} of {len(G['srcs'])} sources loaded", flush=True)
    per = {}
    with mp.get_context("fork").Pool(a.procs) as pool:
        for out in _guarded(pool.imap_unordered(pre_worker, [j for j in jobs if j[0] not in cached], chunksize=1)):
            for name, k, g in out:
                G["cache"][(name, k)] = g[:4] + (None, g[7])
                bad += g[5]
                tot += g[6]
                pb = per.setdefault(name, [0, 0])
                pb[0] += g[5]
                pb[1] += g[6]
    if a.grid_cache:
        for name, (b_, t_) in per.items():
            keys = sorted(k for _, ks in jobs if _ == name for k in ks)
            pickle.dump({"stamp": stamp(name), "keys": keys, "bad": b_, "tot": t_,
                         "grids": {k: G["cache"][(name, k)] for k in keys}}, open(cfile(name), "wb"), protocol=4)
    print(f"grids {len(G['cache'])} responses in {time.time() - t0:.0f}s; top-20 id mismatch {bad}/{tot}", flush=True)
    if a.grid_only:
        assert bad == 0, "re-tokenized response ids do not match the scorer's top-20 ids"
        print("grid_only: cached", sorted(per), flush=True)
        return
    assert bad == 0, "re-tokenized response ids do not match the scorer's top-20 ids"
    consolidate_cache()
    checked = 0
    for name in list(G["srcs"])[:3]:                                  # z(a=0, g=0) must equal the reference z grid
        src = G["srcs"][name]
        ks = [k for k in src.sc if (name, k) in G["cache"]][:30]
        if ks:
            d = max(float(np.max(np.abs(G["cache"][(name, k)][0][:, 0, 0] - z_grid(src.sc[k])[0]))) for k in ks)
            print(f"check {name}: max |z(a=0) - z_grid| = {d:.2e}", flush=True)
            assert d < 1e-3, "zero-penalty surface does not reproduce z_grid"
            checked += 1
    assert checked, "no source available for the z_grid check"
    jobs = [("null", v, None, None, nr, T, 11) for v in cfg["calib"] + cfg["honest"] for nr in nreq]
    jobs += [("alt", b, s, pol, nr, T, 13) for b, s, pol, nr in cfg["alt"] if nr in nreq]
    if G["met"]:
        jobs += [("metnull", cfg["met_null_base"], None, None, nr, T, 17) for nr in nreq]
    if a.kinds != "all":
        jobs = [j for j in jobs if (j[0] == "alt") == (a.kinds == "alt")]
    if a.shard:                                   # conditions are independent (fixed per-condition seeds)
        i, k = map(int, a.shard.split("/"))
        jobs = jobs[i::k]
    with mp.get_context("fork").Pool(a.procs) as pool:
        outs = list(_guarded(pool.imap_unordered(run_condition, jobs, chunksize=1)))
    res = {k: v for k, v in outs if v is not None}
    skipped = ["|".join(map(str, k)) for k, v in outs if v is None]
    print(f"skipped conditions (too few eligible prompts): {len(skipped)} {skipped[:20]}", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump({"cfg": {k: cfg[k] for k in ("calib", "honest", "bases", "subs", "alt")} |
                            {"pool_classes": cfg.get("pool_classes", {}), "draw": a.draw, "split": a.split,
                             "supports": SUPPORTS, "deltas": DELTAS, "pw": PW, "plain": cfg.get("plain", []),
                             "pool_penalty": cfg.get("pool_penalty", []), "met_null_base": cfg.get("met_null_base"),
                             "unseen": cfg.get("unseen", []), "mix_members": cfg.get("mix_members", []),
                             "met_L": cfg.get("met_L", []) if G["met"] else [],
                             "ref_min": G.get("ref_min", 100)},
                     "skipped": skipped, "requested": ["|".join(map(str, j[:5])) for j in jobs],
                     "results": {"|".join(map(str, k)): v for k, v in res.items()}}, f, protocol=4)
    print(f"saved {a.out}: {len(res)} of {len(jobs)} conditions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
