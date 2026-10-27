"""Calibrate profiled typicality and support-aware statistics.

Even trials fit the composite threshold; odd trials estimate false alarms.
Support-aware statistics combine tail counts with support violations or
aggregate domain-specific scores. Each method requires complete calibration
fields and records unavailable endpoint or substitution results explicitly.

Usage: prota_report.py results.pkl out.json [--methods core|tau|all]"""
import argparse
import json
import multiprocessing as mp
import pickle
import sys

import numpy as np

import cmp_report as CR
from cmp_report import evaluate, nlbt, threshold

SPECIAL_QWEN = ('Q02', 'Q13', 'Q18', 'Q19', 'Q21')


def bit(vb, s, d):
    return ((np.asarray(vb) >> (2 * s + d)) & 1).astype(bool)


class ProTaTau:
    def __init__(self, spec):
        self.eta_rule, self.delta = spec["eta"], spec["delta"]
        self.comb, self.zsup = spec["combine"], spec["z_support"]
        self.classes, self.supports = spec["pool_classes"], spec["supports"]
        self.di = spec["deltas"].index(self.delta)

    def _sat(self, v, s):
        k_s, p_s = self.supports[s]
        k_v, p_v = self.classes[v]
        return (k_s is None or k_v == 20) and p_v <= p_s + 1e-9

    def _choose(self, vb):
        """vb: (N,) bits of one trial -> index of the chosen support (len(supports)-1 = untruncated)."""
        for s in range(len(self.supports) - 1):
            if self.eta[s] is not None and np.mean(bit(vb, s, self.di)) <= self.eta[s]:
                return s
        return len(self.supports) - 1

    def _z(self, r, t, s):
        if self.zsup == "top20" or s == len(self.supports) - 1:
            return np.asarray(r["z_dechub"][t], float)
        p = self.supports[s][1]
        return np.asarray(r[{0.8: "zw80", 0.95: "zw95", 1.0: "zw100"}[p]][t], float)

    def _parts(self, r, t):
        vb = np.asarray(r["vb"][t])
        s = self._choose(vb)
        z = self._z(r, t, s)
        v = bit(vb, s, self.di) if s < len(self.supports) - 1 else np.zeros(len(vb), bool)
        return z, v, s

    def fit(self, cal, ev):
        S = len(self.supports) - 1
        share = {v: [float(np.mean([np.mean(bit(r["vb"][t], s, self.di)) for t in np.where(ev[v])[0]]))
                     for s in range(S)] for v, r in cal.items()}
        self.eta, self.mm = [], []
        for s in range(S):
            sin = [share[v][s] for v in cal if self._sat(v, s)]
            sout = [share[v][s] for v in cal if not self._sat(v, s)]
            m_in, m_out = (max(sin) if sin else 0.0), (min(sout) if sout else None)
            if m_out is None or m_in >= m_out:
                self.eta.append(None)
            else:
                self.eta.append(float(np.sqrt(max(m_in, 1e-3) * m_out)) if self.eta_rule == "geo_mid" else 0.6 * m_out)
            self.mm.append((m_in, m_out))
        ez, ev_, eu = {}, {}, {}
        for v, r in cal.items():
            zz, vv, uu = [], [], []
            for t in np.where(ev[v])[0]:
                z, viol, _ = self._parts(r, t)
                zz.append(np.mean(z > 3)), vv.append(np.mean(viol)), uu.append(np.mean((z > 3) | viol))
            ez[v], ev_[v], eu[v] = np.mean(zz), np.mean(vv), np.mean(uu)
        self.pi_u = max(max(eu.values()), 1e-4)
        self.pi_z = max(max(ez.values()), 1e-4)
        self.pi_v = max(max(ev_.values()), 1e-4)
        if self.comb == "maxcomb":
            self.tz = max(threshold(_Comp(self, "z"), cal, ev), 1e-6)
            self.tv = max(threshold(_Comp(self, "v"), cal, ev), 1e-6)

    def component(self, r, which):
        out = []
        for t in range(len(r["vb"])):
            z, viol, s = self._parts(r, t)
            if which == "z":
                out.append(nlbt(int(np.sum(z > 3)), len(z), self.pi_z))
            else:
                out.append(nlbt(int(np.sum(viol)), len(viol), self.pi_v) if s < len(self.supports) - 1 else 0.0)
        return np.array(out)

    def __call__(self, r):
        if self.comb == "maxcomb":
            return np.maximum(self.component(r, "z") / self.tz, self.component(r, "v") / self.tv)
        out = []
        for t in range(len(r["vb"])):
            z, viol, _ = self._parts(r, t)
            out.append(nlbt(int(np.sum((z > 3) | viol)), len(z), self.pi_u))
        return np.array(out)


class _Comp:
    def __init__(self, st, which):
        self.st, self.which = st, which

    def __call__(self, r):
        return self.st.component(r, self.which)


class ProTaX:
    def __init__(self, spec):
        self.field = spec.get("field", "z_x")

    def _counts(self, r, t, d):
        m = np.asarray(r["dom"][t]) == d
        z = np.asarray(r[self.field][t], float)[m]
        return int(np.sum(z > 3)), int(m.sum())

    def fit(self, cal, ev):
        self.pi = []
        for d in range(4):
            rates = []
            for v, r in cal.items():
                k = n = 0
                for t in np.where(ev[v])[0]:
                    a, b = self._counts(r, t, d)
                    k, n = k + a, n + b
                rates.append(k / n if n else 0.0)
            self.pi.append(max(max(rates), 1e-4))
        self.t = [max(threshold(_Dom(self, d), cal, ev), 1e-6) for d in range(4)]

    def dom_stat(self, r, d):
        out = []
        for t in range(len(r["dom"])):
            k, n = self._counts(r, t, d)
            out.append(nlbt(k, n, self.pi[d]) if n else 0.0)
        return np.array(out)

    def __call__(self, r):
        return np.maximum.reduce([self.dom_stat(r, d) / self.t[d] for d in range(4)])


class _Dom:
    def __init__(self, st, d):
        self.st, self.d = st, d

    def __call__(self, r):
        return self.st.dom_stat(r, self.d)


_orig_make_stat = CR.make_stat


def make_stat(spec):
    if spec["type"] == "protatau":
        return ProTaTau(spec)
    if spec["type"] == "protax":
        return ProTaX(spec)
    return _orig_make_stat(spec)


CR.make_stat = make_stat


def tau_configs(tail="top20"):
    out = []
    for eta in ("geo_mid", "0.6m_out"):
        for delta in (0.0, 0.3):
            for comb in ("union", "maxcomb"):
                for zs in ("top20", "S_hat"):
                    out.append(dict(eta=eta, delta=delta, combine=comb, z_support=zs, tail=tail))
    return out


def tau_name(c):
    return f"ProTA-tau[{c['eta']},d{c['delta']},{c['combine']},{c['z_support']},{c['tail']}]"


def methods(cfg, which):
    core = [("ProTA", {"type": "zcount", "field": "z_dechub"}),
            ("FDG T=1", {"type": "zcount", "field": "z_fdg1"}),
            ("T_align", {"type": "zcount", "field": "z_al"}),
            ("T+pen (no Huber)", {"type": "zcount", "field": "z_dec"}),
            ("T Huber", {"type": "zcount", "field": "z_hub"}),
            ("T+presence", {"type": "zcount", "field": "z_pen"}),
            ("NLL count", {"type": "q99count", "field": "d_nll"}),
            ("log-rank count", {"type": "q99count", "field": "d_lrank"}),
            ("entropy count", {"type": "q99count", "field": "d_ent"}),
            ("RUT", {"type": "scalar", "field": "rut", "transform": "neglog"}),
            ("ProTA mean z", {"type": "zmean", "field": "z_dechub"}),
            ("ProTA HC", {"type": "hc", "field": "z_dechub"}),
            ("ProTA top-5 z", {"type": "topk", "field": "z_dechub", "k": 5}),
            ("ProTA tau 2.5", {"type": "zcount", "field": "z_dechub", "tau": 2.5}),
            ("ProTA tau 3.5", {"type": "zcount", "field": "z_dechub", "tau": 3.5}),
            ("ProTA Huber c=1.0", {"type": "zcount", "field": "z_dechub10"}),
            ("ProTA Huber c=2.0", {"type": "zcount", "field": "z_dechub20"}),
            ("T+presence Huber", {"type": "zcount", "field": "z_penhub"}),
            ("MET same-budget", {"type": "scalar", "field": "met30"}),
            ("MET same-budget L50", {"type": "scalar", "field": "met50"}),
            ("MET same-budget L250", {"type": "scalar", "field": "met250"}),
            ("ProTA-X", {"type": "protax", "field": "z_x"})]
    tau = []
    if cfg.get("pool_classes"):
        for c in tau_configs():
            tau.append((tau_name(c), dict(type="protatau", field="vb", pool_classes=cfg["pool_classes"],
                                          supports=[list(x) for x in cfg["supports"]], deltas=list(cfg["deltas"]),
                                          **{k: c[k] for k in ("eta", "delta", "combine", "z_support")})))
    return {"core": core, "tau": tau, "all": core + tau}[which]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl")
    ap.add_argument("out")
    ap.add_argument("--methods", default="core", choices=["core", "tau", "all"])
    ap.add_argument("--pool", default="base", choices=["base", "pen", "self"],
                    help="base = cfg calib; pen = + penalty variants; self = the self variants only (ablation A6)")
    ap.add_argument("--pool_add", default="", help="comma list of variants added to the pool (E13)")
    ap.add_argument("--plain", default="", help="comma list overriding the plain-endpoint list")
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    d = pickle.load(open(a.pkl, "rb"))
    cfg, res = d["cfg"], d["results"]
    pen = cfg.get("pool_penalty") or [h for h in cfg["honest"] if h.startswith(("h_", "hu_"))]
    eps = [h for h in cfg["honest"] if h not in pen]
    if a.plain:
        plain = a.plain.split(",")
    elif cfg.get("plain"):
        plain = cfg["plain"]
    else:
        plain = [e for e in eps if not any(x in e for x in SPECIAL_QWEN)]
    alt = [list(x) for x in sorted({tuple(x[:3]) for x in cfg["alt"]})]
    req = d.get("requested")
    nreq = sorted({int(k.split("|")[-1]) for k in (req if req else res)})
    pool = cfg["calib"] + (pen if a.pool == "pen" else [])
    if a.pool == "self":
        pool = [v for v in cfg["calib"] if v in ("self", "self2", "self_u", "self_u2")]
    pool += [v for v in a.pool_add.split(",") if v and v not in pool]
    if not res:
        json.dump({"pool": a.pool, "error": "no results in the file", "requested": d.get("requested"),
                   "skipped": d.get("skipped")}, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ERROR: no results in", a.pkl, flush=True)
        sys.exit(1)
    have = set().union(*(set(v.keys()) for v in res.values()))
    specs = []
    mets = sorted(int(f[3:]) for f in have if f.startswith("met") and f[3:].isdigit())
    extra_methods = [(f"MET same-budget L{L}", {"type": "scalar", "field": f"met{L}"}) for L in mets] \
        if a.methods in ("core", "all") else []
    for name, stat in [m for m in methods(cfg, a.methods) if not m[0].startswith("MET same-budget")] + extra_methods:
        f = stat.get("field")
        if any(x not in have for x in needed_fields(stat)):
            continue
        extra = {}
        if f and f.startswith("met"):                       # MET's native rule: its own simulated null, 95%
            extra = {"native": "metnull", "metnull_base": cfg.get("met_null_base")}
        if f == "rut":
            extra = {"native": "p_lt_0.05"}
        specs.append(dict(name=name, stat=stat, pool=pool, honest=eps + pen, plain=plain, alt=alt, nreq=nreq,
                          len_field="z_dechub", **extra))
    _G["res"], _G["requested"] = res, d.get("requested")
    if not specs:
        json.dump({"pool": a.pool, "error": f"no selected method ({a.methods}) has its fields in the results"},
                  open(a.out, "w", encoding="utf-8"), indent=1)
        print("ERROR: no selected method available", flush=True)
        sys.exit(1)
    with mp.get_context("fork").Pool(min(a.procs, len(specs))) as P:
        outs = P.map(_eval_one, specs, chunksize=1)
    summ = {}
    for spec, ev in zip(specs, outs):
        summ[spec["name"]] = ev
        for n, t in ev.items():
            if "error" not in t:
                g = t["gate"]
                print(f"{spec['name']:58s} N={n:>4s} valid={int(t['valid'])} ho {g['max_heldout']:.3f} lovo "
                      f"{(g['max_lovo'] or 0):.3f} plain {(g['max_plain_endpoint'] or 0):.3f}", flush=True)
    json.dump({"pool": a.pool, "pool_variants": pool, "plain": plain, "methods": summ}, open(a.out, "w", encoding="utf-8"), indent=1)


_G = {}


def needed_fields(stat):
    """Per-trial fields a statistic reads (a condition without them cannot be evaluated by that method)."""
    k = stat["type"]
    if k == "protatau":
        return ["vb", "z_dechub"] + (["zw80", "zw95", "zw100"] if stat.get("z_support") == "S_hat" else [])
    if k == "protax":
        return [stat.get("field", "z_x"), "dom"]
    if k == "max":
        return sorted({f for c in stat["comps"] for f in needed_fields(c)})
    if k == "tolband":
        return ["dA", "seA"]
    return [stat["field"]]


def evaluate_checked(spec, res, requested=None):
    """cmp_report.evaluate on a view that keeps only conditions carrying every field the method needs; a budget whose
    pool variants lack a result or a needed field is an explicit error; then complete() lists missing endpoints /
    alternatives and invalidates budgets with incomplete plain endpoints."""
    need = needed_fields(spec["stat"])
    ok = lambda key: key in res and all(f in res[key] for f in need)
    view = {k: v for k, v in res.items() if k.startswith("metnull|") or all(f in v for f in need)}
    ev = {}
    for n in spec["nreq"]:
        bad = [v for v in spec["pool"] if not ok(f"null|{v}|None|None|{n}")]
        if bad:
            ev[str(n)] = {"error": f"pool variants lacking a result or a field of {need}: {bad}"}
            continue
        ev.update(evaluate(dict(spec, nreq=[n]), view))
    return complete(spec, ev, res, requested)


def complete(spec, ev, res, requested=None):
    """Field-level completeness per budget (cmp_report's evaluate() skips absent conditions silently): a plain endpoint
    whose result is absent or lacks a field the method needs makes the N invalid; requested alternatives that are
    absent or lack a field are listed (missing_alt). Budgets requested but never computed are reported as errors."""
    need = needed_fields(spec["stat"])
    have = lambda key: key in res and all(f in res[key] for f in need)
    for n in spec["nreq"]:
        t = ev.setdefault(str(n), {"error": "no results for this budget"})
        if "error" in t:
            continue
        miss_p = [e for e in spec["plain"] if not have(f"null|{e}|None|None|{n}")]
        miss_a = [k for k in (requested or []) if k.startswith("alt|") and k.endswith(f"|{n}") and not have(k)]
        t["missing_plain"], t["missing_alt"] = miss_p, miss_a
        if miss_p:
            t["valid"] = False
            t["incomplete"] = True
    return ev


def _eval_one(spec):
    return evaluate_checked(spec, _G["res"], _G.get("requested"))


if __name__ == "__main__":
    main()
