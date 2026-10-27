"""Compute audit statistics from response-grid caches.

Each trial selects distinct entries from an eligible prompt list, routes
requests and samples stored responses uniformly. Calibration and test draws
use separate seed streams; RUT tie-breaking uses separate uniforms.
Bootstrap lists may contain duplicate original prompt IDs.

The engine profiles temperature and penalties, counts response tails, and
computes RUT against the first 100 local references. Reductions use float64.
GPU and NumPy draws implement the same sampling rule with different streams."""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(_APITAIL_CODE / 'bbp'))

TG = np.round(np.arange(0.30, 1.81, 0.01), 2)
NT, NA, NG = 151, 7, 7
NAG = NA * NG
I10 = int(np.argmin(np.abs(TG - 1.0)))
HUBER = 1.345
MASTER = 20261025
GRID_FIELDS = {"Z": (0, None), "SA": (1, None), "SG": (2, None), "ZW": (5, "zW"), "SAW": (5, "saW"), "SGW": (5, "sgW"),
               "ZN": (5, "zN"), "SAN": (5, "saN"), "SGN": (5, "sgN")}
PEN_OF = {"Z": ("SA", "SG"), "ZW": ("SAW", "SGW"), "ZN": ("SAN", "SGN"), "ZF": ("SAF", "SGF"), "Z20": ("SA20", "SG20"),
          "Z100": ("SA100", "SG100")}
FV_FIELDS = ("ZF", "SAF", "SGF", "Z20", "SA20", "SG20", "Z100", "SA100", "SG100", "ZE", "SAE", "SRE")

# method name -> definition. prof: grid, penalties profiled?, psi; fixed: grid column; det: column of det; rut.
METHODS = {
    "ProTA-N95": ("prof", "ZN", True, "hub"),
    "ProTA": ("prof", "Z", True, "hub"),
    "ProTA-W": ("prof", "ZW", True, "hub"),
    "T_align": ("prof", "Z", False, "id"),
    "T Huber": ("prof", "Z", False, "hub"),
    "T+pen (no Huber)": ("prof", "Z", True, "id"),
    "FDG T=1": ("fixed", "Z", I10 * NAG),
    "N95 T-only id": ("prof", "ZN", False, "id"),       # Temperature-only alignment on the nucleus grid.
    "N95 T-only Huber": ("prof", "ZN", False, "hub"),
    "N95 pen id": ("prof", "ZN", True, "id"),
    "W T-only id": ("prof", "ZW", False, "id"),
    "NLL count": ("det", 0),
    "log-rank count": ("det", 1),
    "entropy count": ("det", 2),
    "RUT": ("rut",),
    # Full-vocabulary and window grids recomputed from the same logits.
    "T_align-full": ("prof", "ZF", False, "id"),
    "T Huber-full": ("prof", "ZF", False, "hub"),
    "T+pen-full": ("prof", "ZF", True, "id"),
    "ProTA-full": ("prof", "ZF", True, "hub"),
    "FDG-full": ("fixed", "ZF", I10 * NAG),
    "T_align-w20r": ("prof", "Z20", False, "id"),
    "ProTA-w20r": ("prof", "Z20", True, "hub"),
    "T_align-w100": ("prof", "Z100", False, "id"),
    "ProTA-w100": ("prof", "Z100", True, "hub"),
    # Profile theta on the top-20 grid and score it with full-vocabulary moments.
    "full@ProTA": ("cross", "Z", True, "hub", "ZF"),
    "full@T_align": ("cross", "Z", False, "id", "ZF"),
    "w20r@ProTA": ("cross", "Z", True, "hub", "Z20"),
    # Per-response reference scores added with Panel.add_scalar.
    "TrueCfg z count": ("scal", "ztrue", 3.0),
    "LR oracle 0.05": ("lr", "llr", 0.05),
    "LR oracle 0.10": ("lr", "llr", 0.10),
    "LR oracle 0.25": ("lr", "llr", 0.25),
    # Intersection-union RUT uses the maximum CvM p-value across reference sources.
    # Reference counts are provided by Panel.add_rut_refs.
    "RUT-IU": ("rutiu", ("self", "h_topk20", "h_T07", "h_T10", "h_gcdef", "h_T07p08", "h_sysdefault")),
}


def cond_name(c):
    return "|".join(map(str, c))


def cond_hash(c):
    return int(hashlib.sha256(cond_name(c).encode()).hexdigest()[:8], 16)


def make_route_masks(P):
    """Deterministic routing policies of scenarios.make_policies as per-pid booleans (random_rho handled separately)."""
    lens = np.array([len(p["text"]) for p in P.values()])
    med = float(np.median(lens))
    return {"route_chat_only": {p: r["domain"] == "wildchat" for p, r in P.items()},
            "route_zh_only": {p: r["lang"] == "zh" for p, r in P.items()},
            "route_short_prompts": {p: len(r["text"]) < med for p, r in P.items()},
            "route_code_math_only": {p: r["domain"] in ("bigcodebench", "math") for p, r in P.items()}}


class Panel:
    """Hold concatenated response-grid caches for one panel on a GPU."""

    def __init__(self, cfg, gridcache, names, grids=("Z", "SA", "SG", "ZN", "SAN", "SGN"), device="cuda:0",
                 rut=True, split_tag="all", verbose=True, fullvocab=None):
        from rut import load_scored_multi, logrank_mid
        t0 = time.time()
        self.cfg, self.device, self.names = cfg, device, list(names)
        self.P = {json.loads(l)["pid"]: json.loads(l) for l in open(cfg["prompts"], encoding="utf-8")}
        self.common = sorted(self.P)
        self.pidx = {p: i for i, p in enumerate(self.common)}
        self.route_masks = make_route_masks(self.P)
        self.ref_min = int(cfg.get("ref_min", 100))
        self.refs = None
        if rut and cfg.get("ref"):
            refs = load_scored_multi(cfg["ref"])
            self.refs = {p: np.array([logrank_mid(r) for r in refs.get(p, [])]) for p in self.common}
        nP = len(self.common)
        self.start, self.count, self.keys = {}, {}, {}
        rows_total = 0
        caches = {}
        for name in self.names:
            c = pickle.load(open(os.path.join(gridcache, f"{name}.{split_tag}.v3cond2.pkl"), "rb"))
            keys = [k for k in c["keys"] if k[0] in self.pidx]
            caches[name] = (keys, c["grids"])
            st = np.full(nP, -1, np.int64)
            ct = np.zeros(nP, np.int64)
            for j, (pid, k) in enumerate(keys):
                i = self.pidx[pid]
                if st[i] < 0:
                    st[i] = rows_total + j
                ct[i] += 1
            self.start[name], self.count[name], self.keys[name] = st, ct, keys
            self.row_of = None
            rows_total += len(keys)
        self.R = rows_total
        self.grids = {}
        fvg = [g for g in grids if g in FV_FIELDS]
        grids = [g for g in grids if g not in FV_FIELDS]
        host = {g: np.empty((rows_total, NT * NAG), np.float32) for g in grids}
        det = np.empty((rows_total, 3), np.float64)
        r0 = 0
        for name in self.names:
            keys, gr = caches.pop(name)
            for j, k in enumerate(keys):
                g = gr[k]
                for f in grids:
                    i, sub = GRID_FIELDS[f]
                    host[f][r0 + j] = (g[i] if sub is None else g[i][sub]).reshape(-1)
                det[r0 + j] = g[3]
            r0 += len(keys)
            del gr
        for f in grids:
            self.grids[f] = torch.from_numpy(host.pop(f)).to(device)
        if fvg:                                                     # score_full.py outputs, same key order
            parts = {g: [] for g in fvg}
            for name in self.names:
                fv = pickle.load(open(os.path.join(fullvocab, f"{name}.full.pkl"), "rb"))
                assert [tuple(k) for k in fv["keys"]] == [tuple(k) for k in self.keys[name]], f"key order {name}"
                for g in fvg:
                    parts[g].append(fv["fields"][g])
                del fv
            for g in fvg:
                self.grids[g] = torch.from_numpy(np.concatenate(parts.pop(g))).to(device)
        self.det = torch.from_numpy(det).to(device)
        self.rowkey = [(name, k) for name in self.names for k in self.keys[name]]
        self.rut_L = self.rut_E = self.rut_m = None
        if self.refs is not None:                                   # per-row RUT counts against the first 100 refs
            L = np.zeros(rows_total, np.float64)
            E = np.zeros(rows_total, np.float64)
            m = np.zeros(rows_total, np.float64)
            for r, (name, (pid, k)) in enumerate(self.rowkey):
                ref = self.refs[pid][:100]
                t = det[r, 1]
                L[r], E[r], m[r] = np.sum(ref < t), np.sum(ref == t), len(ref)
            self.rut_L, self.rut_E, self.rut_m = (torch.from_numpy(x).to(device) for x in (L, E, m))
        if verbose:
            print(f"panel: {len(self.names)} sources, {rows_total} rows, grids {list(grids)} on {device} "
                  f"in {time.time() - t0:.0f}s", flush=True)

    def add_rut_refs(self, name, prefixes, m_max=100):
        """Per-row RUT counts (L, E, m) of every panel row against another reference set (scored prefixes, e.g. the
        samples of one permitted deployment); prompts without references get m = 0 (rank = the uniform draw)."""
        from rut import load_scored_multi, logrank_mid
        refs = load_scored_multi(prefixes)
        R = {p: np.array([logrank_mid(r) for r in refs.get(p, [])])[:m_max] for p in self.common}
        det1 = self.det[:, 1].cpu().numpy()
        L = np.zeros(self.R)
        E = np.zeros(self.R)
        m = np.zeros(self.R)
        for r, (_, (pid, _k)) in enumerate(self.rowkey):
            ref = R[pid]
            L[r], E[r], m[r] = np.sum(ref < det1[r]), np.sum(ref == det1[r]), len(ref)
        if not hasattr(self, "rut_extra"):
            self.rut_extra = {}
        self.rut_extra[name] = tuple(torch.from_numpy(x).to(self.device) for x in (L, E, m))
        return {"prompts_with_refs": int(sum(len(v) > 0 for v in R.values())), "min_refs": int(min(len(v) for v in R.values() if len(v)))}

    def add_scalar(self, name, values):
        """values: dict (source, (pid, k)) -> float; rows without a value get NaN."""
        arr = np.full(self.R, np.nan, np.float64)
        for r, nk in enumerate(self.rowkey):
            v = values.get((nk[0], tuple(nk[1])))
            if v is not None:
                arr[r] = v
        if not hasattr(self, "scal"):
            self.scal = {}
        self.scal[name] = torch.from_numpy(arr).to(self.device)
        return int(np.isnan(arr).sum())

    def row_index(self):
        if self.row_of is None:
            self.row_of = {nk: r for r, nk in enumerate(self.rowkey)}
        return self.row_of

    def eligible(self, base, sub, plist=None):
        plist = self.common if plist is None else plist
        cb = self.count[base]
        cs = self.count[sub] if sub else None
        out = []
        for p in plist:
            i = self.pidx[p]
            if cb[i] < 1 or (cs is not None and cs[i] < 1):
                continue
            if self.refs is not None and len(self.refs[p]) < self.ref_min:
                continue
            out.append(i)
        return np.array(out, np.int64)

    
    def draw(self, c, trials, b=-1, plist=None, rep_cal=0, rep_test=0, split_even=True):
        """c = (kind, base, sub, policy, N). Returns (rows (trials, N) int64, rut uniforms (trials, N)) or None."""
        kind, base, sub, pol, N = c
        el = self.eligible(base, sub, plist)
        if len(el) < N:
            return None
        hc = cond_hash(c)
        rows = np.empty((trials, N), np.int64)
        uni = np.empty((trials, N), np.float64)
        tidx = np.arange(trials)
        groups = [(tidx[tidx % 2 == 0], 0, rep_cal), (tidx[tidx % 2 == 1], 1, rep_test)] if split_even else \
            [(tidx, 1, rep_test)]
        for ti, s, rep in groups:
            if not len(ti):
                continue
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, s, rep])))
            keys = rng.random((len(ti), len(el)))
            sel = np.argpartition(keys, N - 1, axis=1)[:, :N]
            pi = el[sel]                                                     # (n, N) prompt indices
            if sub is None:
                routed = np.zeros(pi.shape, bool)
            elif pol.startswith("random_"):
                routed = rng.random(pi.shape) < float(pol.split("_")[1])
            else:
                msk = self.route_masks[pol]
                routed = np.array([msk[self.common[i]] for i in range(len(self.common))])[pi]
            u = rng.random(pi.shape)
            st = np.where(routed, self.start[sub][pi] if sub else 0, self.start[base][pi])
            ct = np.where(routed, self.count[sub][pi] if sub else 1, self.count[base][pi])
            rows[ti] = st + np.minimum((u * ct).astype(np.int64), ct - 1)
            rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, s + 2, rep])))
            uni[ti] = rng2.random(pi.shape)
        return rows, uni

    def draw_mix(self, name, comps, N, trials, plist=None, stream=1, rep=0, b=-1):
        """Draw N distinct prompt-list entries and route each request to a component.
        Eligible prompts have responses from every component and enough references.
        The condition name selects the random stream."""
        srcs = [s for s, _ in comps]
        w = np.array([p for _, p in comps], float)
        w = w / w.sum()
        plist = self.common if plist is None else plist
        el = []
        for p in plist:
            i = self.pidx[p]
            if all(self.count[s][i] >= 1 for s in srcs) and (self.refs is None or len(self.refs[p]) >= self.ref_min):
                el.append(i)
        el = np.array(el, np.int64)
        if len(el) < N:
            return None
        hc = cond_hash(("mix", name, N))
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, stream, rep])))
        keys = rng.random((trials, len(el)))
        pi = el[np.argpartition(keys, N - 1, axis=1)[:, :N]]
        comp = np.searchsorted(np.cumsum(w), rng.random(pi.shape), side="right").clip(0, len(srcs) - 1)
        u = rng.random(pi.shape)
        st = np.zeros(pi.shape, np.int64)
        ct = np.ones(pi.shape, np.int64)
        for j, s in enumerate(srcs):
            m = comp == j
            st[m], ct[m] = self.start[s][pi[m]], self.count[s][pi[m]]
        rows = st + np.minimum((u * ct).astype(np.int64), ct - 1)
        rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, stream + 2, rep])))
        return rows, rng2.random(pi.shape), comp

    def draw_stream(self, c, n, stream, rep, b=-1, plist=None):
        """n trials of one stream (0 = calibration / even trials, 1 = the rest) exactly as draw() makes them."""
        kind, base, sub, pol, N = c
        el = self.eligible(base, sub, plist)
        if len(el) < N:
            return None
        hc = cond_hash(c)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, stream, rep])))
        keys = rng.random((n, len(el)))
        pi = el[np.argpartition(keys, N - 1, axis=1)[:, :N]]
        if sub is None:
            routed = np.zeros(pi.shape, bool)
        elif pol.startswith("random_"):
            routed = rng.random(pi.shape) < float(pol.split("_")[1])
        else:
            msk = self.route_masks[pol]
            routed = np.array([msk[self.common[i]] for i in range(len(self.common))])[pi]
        u = rng.random(pi.shape)
        st = np.where(routed, self.start[sub][pi] if sub else 0, self.start[base][pi])
        ct = np.where(routed, self.count[sub][pi] if sub else 1, self.count[base][pi])
        rows = st + np.minimum((u * ct).astype(np.int64), ct - 1)
        rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([MASTER, b + 1, hc, stream + 2, rep])))
        return rows, rng2.random(pi.shape)

    def draw_gpu(self, c, trials, b=-1, plist=None, rep_cal=0, rep_test=0, split_even=True, el=None):
        """Same draw rule as draw(), generated on the GPU with torch generators (used for the large bootstrap; its random
        stream differs from draw(), the distribution is the same). Returns (rows, uni) as numpy arrays or None."""
        kind, base, sub, pol, N = c
        if el is None:
            el = self.eligible(base, sub, plist)
        if len(el) < N:
            return None
        dev = self.device
        hc = cond_hash(c)
        elt = torch.from_numpy(el).to(dev)
        rows = torch.empty((trials, N), dtype=torch.int64, device=dev)
        uni = torch.empty((trials, N), dtype=torch.float64, device=dev)
        tidx = torch.arange(trials, device=dev)
        groups = [(tidx[tidx % 2 == 0], 0, rep_cal), (tidx[tidx % 2 == 1], 1, rep_test)] if split_even else \
            [(tidx, 1, rep_test)]
        if not hasattr(self, "_st_t"):
            self._st_t = {n: torch.from_numpy(self.start[n]).to(dev) for n in self.names}
            self._ct_t = {n: torch.from_numpy(self.count[n]).to(dev) for n in self.names}
            self._mask_t = {k: torch.tensor([v[p] for p in self.common], device=dev) for k, v in self.route_masks.items()}
        for ti, s, rep in groups:
            n = len(ti)
            if not n:
                continue
            seed = int(np.random.SeedSequence([MASTER, b + 1, hc, s, rep]).generate_state(1, np.uint64)[0] >> 1)
            g = torch.Generator(device=dev)
            g.manual_seed(seed)
            keys = torch.rand((n, len(el)), generator=g, device=dev)
            sel = torch.topk(keys, N, dim=1, largest=False, sorted=False).indices
            pi = elt[sel]
            if sub is None:
                routed = torch.zeros(pi.shape, dtype=torch.bool, device=dev)
            elif pol.startswith("random_"):
                routed = torch.rand(pi.shape, generator=g, device=dev) < float(pol.split("_")[1])
            else:
                routed = self._mask_t[pol][pi]
            u = torch.rand(pi.shape, generator=g, device=dev, dtype=torch.float64)
            stb, ctb = self._st_t[base][pi], self._ct_t[base][pi]
            if sub is not None:
                st = torch.where(routed, self._st_t[sub][pi], stb)
                ct = torch.where(routed, self._ct_t[sub][pi], ctb)
            else:
                st, ct = stb, ctb
            rows[ti] = st + torch.minimum((u * ct).long(), ct - 1)
            uni[ti] = torch.rand(pi.shape, generator=g, device=dev, dtype=torch.float64)
        return rows, uni

    
    @torch.no_grad()
    def stats(self, rows, uni, methods, tau=3.0, batch=None, rut_p=True):
        """rows (T, N) int64 numpy, uni (T, N). Returns {method: per-trial K (int) or det values (T, N) or RUT dict},
        plus theta indices for profile methods under key ('theta', method)."""
        T, N = rows.shape
        dev = self.device
        rt = rows if torch.is_tensor(rows) else torch.from_numpy(rows).to(dev)
        if torch.is_tensor(uni):
            uni_t = uni
        else:
            uni_t = torch.from_numpy(uni).to(dev) if uni is not None else None
        out = {}
        if batch is None:
            batch = max(1, int(6e8 // (N * NT * NAG * 4)))
        for m in methods:
            d = METHODS[m]
            if d[0] in ("prof", "cross"):
                g, pen, psi = d[1], d[2], d[3]
                gz = d[4] if d[0] == "cross" else g
                K = torch.empty(T, dtype=torch.int32, device=dev)
                TH = torch.empty(T, dtype=torch.int32, device=dev)
                Zg = self.grids[g]
                if not pen:
                    cols = torch.arange(NT, device=dev) * NAG
                    Z0 = Zg[:, cols]
                for b0 in range(0, T, batch if pen else batch * 40):
                    r = rt[b0: b0 + (batch if pen else batch * 40)]
                    if not pen:
                        X = Z0[r]                                            # B, N, NT
                        Xp = X.clamp(-HUBER, HUBER) if psi == "hub" else X
                        s = Xp.double().sum(1)
                        tk = torch.argmin(s.abs(), dim=1)
                        z = X.gather(2, tk[:, None, None].expand(-1, N, 1))[..., 0]
                        th = tk * NAG
                        if gz != g:
                            z = self.grids[gz][r, th[:, None].expand(-1, N)]
                    else:
                        X = Zg[r]                                            # B, N, NT*NAG
                        Xp = X.clamp(-HUBER, HUBER) if psi == "hub" else X
                        s = Xp.double().sum(1).view(-1, NT, NAG)
                        tk = torch.argmin(s.abs(), dim=1)                    # B, NAG (first minimum)
                        col = tk * NAG + torch.arange(NAG, device=dev)[None]
                        ga, gg = PEN_OF[g]
                        sa = self.grids[ga][r[..., None], col[:, None, :]].double().sum(1)
                        sg = self.grids[gg][r[..., None], col[:, None, :]].double().sum(1)
                        ag = torch.argmin(sa * sa + sg * sg, dim=1)
                        th = col.gather(1, ag[:, None])[:, 0]
                        if gz == g:
                            z = X.gather(2, th[:, None, None].expand(-1, N, 1))[..., 0]
                        else:                                        # cross: z of another grid at this theta
                            z = self.grids[gz][r, th[:, None].expand(-1, N)]
                    K[b0: b0 + len(r)] = (z > tau).sum(1).int()
                    TH[b0: b0 + len(r)] = th.int()
                out[m] = K.cpu().numpy()
                out[("theta", m)] = TH.cpu().numpy()
            elif d[0] == "fixed":
                _, g, col = d
                z = self.grids[g][rt, col]
                out[m] = (z > tau).sum(1).int().cpu().numpy()
            elif d[0] == "det":
                out[m] = self.det[rt, d[1]].cpu().numpy()                    # T, N values; counted after the fit
            elif d[0] == "scal":                                             # count of a per-response scalar > tau
                out[m] = (self.scal[d[1]][rt] > d[2]).sum(1).int().cpu().numpy()
            elif d[0] == "lr":                                               # sum log(1 - rho + rho * q_S / q_M)
                x = self.scal[d[1]][rt]
                out[m] = torch.logaddexp(torch.full_like(x, float(np.log1p(-d[2]))), float(np.log(d[2])) + x
                                         ).sum(1).cpu().numpy()
            elif d[0] == "rut":
                u = uni_t
                rk = ((self.rut_L[rt] + u * (self.rut_E[rt] + 1.0)) / (self.rut_m[rt] + 1.0)).clamp(0, 1)
                rk, _ = torch.sort(rk, dim=1)
                n = N
                uu = (2.0 * torch.arange(1, n + 1, device=dev, dtype=torch.float64) - 1.0) / (2.0 * n)
                w = 1.0 / (12.0 * n) + ((uu[None] - rk) ** 2).sum(1)
                wn = w.cpu().numpy()
                if rut_p:                       # p only when needed (-log p is increasing in W^2 for fixed N)
                    from scipy.stats import _hypotests as H
                    out[m] = {"w2": wn, "p": np.maximum(0.0, 1.0 - H._cdf_cvm(wn, n))}
                else:
                    out[m] = {"w2": wn}
            elif d[0] == "rutiu":                                            # max over deployments of the CvM p
                from scipy.stats import _hypotests as H
                uu = (2.0 * torch.arange(1, N + 1, device=dev, dtype=torch.float64) - 1.0) / (2.0 * N)
                ps = []
                for nm in d[1]:
                    L_, E_, m_ = (self.rut_L, self.rut_E, self.rut_m) if nm == "self" else self.rut_extra[nm]
                    rk = ((L_[rt] + uni_t * (E_[rt] + 1.0)) / (m_[rt] + 1.0)).clamp(0, 1)
                    rk, _ = torch.sort(rk, dim=1)
                    wn = (1.0 / (12.0 * N) + ((uu[None] - rk) ** 2).sum(1)).cpu().numpy()
                    ps.append(np.maximum(0.0, 1.0 - H._cdf_cvm(wn, N)))
                ps = np.stack(ps, 1)
                out[m] = {"p": ps.max(1), "p_dep": ps}
        return out



def q_higher(x, q=0.95):
    return float(np.quantile(np.asarray(x, float), q, method="higher"))


def stat_values(method, raw, pool, ev):
    """raw: {cond: stats-output entry}; returns {cond: per-trial s (larger = more suspicious)} and fit info.
    det methods: tau = pooled 99% quantile (numpy linear) of the pool's even-trial responses; s = K = #{x > tau}."""
    d = METHODS[method]
    if d[0] in ("prof", "fixed", "cross", "scal", "lr"):
        return {k: np.asarray(v, float) for k, v in raw.items()}, {}
    if d[0] in ("rut", "rutiu"):
        return {k: -np.log(np.maximum(v["p"], 1e-300)) for k, v in raw.items()}, {}
    allv = np.concatenate([raw[v][ev[v]].reshape(-1) for v in pool])
    allv = allv[~np.isnan(allv)]
    tau = float(np.quantile(allv, 0.99))
    return {k: np.sum(v > tau, axis=1).astype(float) for k, v in raw.items()}, {"tau": tau}


def pi0_of(method, raw, pool, ev, N, tau_info):
    """Informational: cmp_report's null rate = max over pool variants of the per-response exceedance rate (even trials)."""
    d = METHODS[method]
    if d[0] in ("prof", "fixed", "cross", "scal"):
        return max(max(float(np.mean(raw[v][ev[v]])) / N for v in pool), 1e-4)
    if d[0] == "det":
        return max(max(float(np.mean(raw[v][ev[v]] > tau_info["tau"])) for v in pool), 1e-4)
    return None


def evaluate(method, raw, pool, honest, alts, plain, N, gate=(0.07, 0.10, 0.10)):
    """raw: {cond_name: stats entry} for the pool variants ('null|v|None|None|N'), honest sources and alternatives of
    one budget N. Same rule as cmp_report.evaluate; returns thresholds, held-out, LOVO, endpoint, power, validity."""
    key = lambda v: f"null|{v}|None|None|{N}"
    ntr = len(raw[key(pool[0])]["p"] if METHODS[method][0] in ("rut", "rutiu") else raw[key(pool[0])])
    even = np.arange(ntr) % 2 == 0
    evm = {key(v): even for v in pool}
    poolk = [key(v) for v in pool]
    s, info = stat_values(method, raw, poolk, evm)
    qs = {v: q_higher(s[key(v)][even]) for v in pool}
    thr = max(qs.values())
    tab = {"thr": thr, "decisive": max(qs, key=qs.get), "per_variant_q95": qs, "heldout": {}, "lovo": {},
           "endpoint": {}, "power": {}, **info}
    pi0 = pi0_of(method, raw, poolk, evm, N, info)
    if pi0 is not None:
        tab["pi0"] = pi0
    for v in pool:
        tab["heldout"][v] = float(np.mean(s[key(v)][~even] > thr))
        rest = [u for u in pool if u != v]
        s2, _ = stat_values(method, raw, [key(u) for u in rest], {key(u): even for u in rest})
        thr2 = max(q_higher(s2[key(u)][even]) for u in rest)
        tab["lovo"][v] = float(np.mean(s2[key(v)] > thr2))
    for e in honest:
        if key(e) in s:
            tab["endpoint"][e] = float(np.mean(s[key(e)] > thr))
    for a in alts:
        k = f"alt|{a}|{N}"
        if k in s:
            tab["power"][a] = float(np.mean(s[k] > thr))
    pl = [e for e in plain if e in tab["endpoint"]]
    tab["missing_plain"] = [e for e in plain if e not in tab["endpoint"]]
    tab["gate"] = {"max_heldout": max(tab["heldout"].values()), "max_lovo": max(tab["lovo"].values()),
                   "max_plain_endpoint": max([tab["endpoint"][e] for e in pl], default=None),
                   "argmax_plain": max(pl, key=lambda e: tab["endpoint"][e]) if pl else None,
                   "argmax_lovo": max(tab["lovo"], key=tab["lovo"].get)}
    tab["valid"] = bool(tab["gate"]["max_heldout"] <= gate[0] and tab["gate"]["max_lovo"] <= gate[1] and
                        all(tab["endpoint"][e] <= gate[2] for e in pl) and not tab["missing_plain"])
    return tab
