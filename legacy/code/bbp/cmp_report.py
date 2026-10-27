"""Calibrate detector statistics against a composite null.

For each budget, the threshold is the largest configuration-specific 95th
percentile on even-indexed trials. Odd trials estimate held-out false alarms.
Leave-one-variant-out checks refit the threshold without that configuration.
Larger statistics indicate stronger evidence against the null.

Usage: cmp_report.py spec.json out.json"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import ndtr

ALPHA = 0.05


def nlbt(k, n, pi):
    return float(-stats.binom.logsf(k - 1, n, pi)) if n else 0.0


def hc_stat(z):
    """Higher Criticism of the one-sided p-values of z (Donoho-Jin), over the smallest half."""
    p = np.sort(np.clip(1 - ndtr(np.asarray(z, float)), 1e-12, 1 - 1e-12))
    n = len(p)
    i = np.arange(1, n + 1)
    hc = np.sqrt(n) * (i / n - p) / np.sqrt(p * (1 - p))
    return float(np.max(hc[: max(1, n // 2)]))



class CountStat:
    """Per-response values -> indicator above a per-response threshold -> -log binomial tail with a fitted null rate.
    thr: fixed number (e.g. z > 3) or 'q99' (99% quantile of the pooled permitted responses, fitted)."""

    def __init__(self, field, thr, ind=None):
        self.field, self.thr, self.ind = field, thr, ind

    def values(self, r, t):
        if self.ind is not None:
            return self.ind(r, t)
        return np.asarray(r[self.field][t], float)

    def fit(self, cal, ev):
        if self.thr == "q99":
            allv = np.concatenate([self.values(r, t) for v, r in cal.items() for t in np.where(ev[v])[0]])
            allv = allv[~np.isnan(allv)]
            self.tau = float(np.quantile(allv, 0.99))
        else:
            self.tau = self.thr
        rates = []
        for v, r in cal.items():
            x = np.concatenate([self.values(r, t) for t in np.where(ev[v])[0]])
            x = x[~np.isnan(x)]
            rates.append(float(np.mean(x > self.tau)) if len(x) else 0.0)
        self.pi = max(max(rates), 1e-4)

    def __call__(self, r):
        out = []
        for t in range(len(r[self.field])):
            x = self.values(r, t)
            x = x[~np.isnan(x)]
            out.append(nlbt(int(np.sum(x > self.tau)), len(x), self.pi))
        return np.array(out)


class ScalarStat:
    def __init__(self, fn):
        self.fn = fn

    def fit(self, cal, ev):
        pass

    def __call__(self, r):
        return np.asarray(self.fn(r), float)


class MaxComb:
    """max_k s_k / t_k over components, each scaled by its own composite threshold (then recalibrated)."""

    def __init__(self, comps):
        self.comps = comps

    def fit(self, cal, ev):
        for c in self.comps:
            c.fit(cal, ev)
        self.scale = [max(threshold(c, cal, ev), 1e-6) for c in self.comps]

    def __call__(self, r):
        return np.maximum.reduce([c(r) / s for c, s in zip(self.comps, self.scale)])


def threshold(stat, cal, ev):
    return max(float(np.quantile(stat(r)[ev[v]], 1 - ALPHA, method="higher")) for v, r in cal.items())


def make_stat(spec):
    k = spec["type"]
    if k == "zcount":                          # A family: K = #{z > tau}
        return CountStat(spec["field"], spec.get("tau", 3.0))
    if k == "tolband":                         # A': per-response D > eps + k SE
        eps, kk = spec["eps"], spec.get("k", 3.0)
        return CountStat("dA", 0.5, ind=lambda r, t: (np.asarray(r["dA"][t]) > eps + kk * np.asarray(r["seA"][t])).astype(float))
    if k == "q99count":                        # B, CLS, Greedy: per-response score, pooled 99% quantile
        return CountStat(spec["field"], "q99")
    if k == "zmean":
        return ScalarStat(lambda r: [np.mean(z) for z in r[spec["field"]]])
    if k == "hc":
        return ScalarStat(lambda r: [hc_stat(z) for z in r[spec["field"]]])
    if k == "topk":
        return ScalarStat(lambda r: [np.sum(np.sort(z)[-spec["k"]:]) for z in r[spec["field"]]])
    if k == "scalar":                          # RUT (-log p), MET (MMD^2), KBF (-log p), IRIS (K), G
        f, tr = spec["field"], spec.get("transform")
        if tr == "neglog":
            return ScalarStat(lambda r: -np.log(np.maximum(np.asarray(r[f], float), 1e-300)))
        if tr == "gap":
            return ScalarStat(lambda r: np.asarray(r["Talign"], float) - np.asarray(r["Tagree"], float))
        return ScalarStat(lambda r: r[f])
    if k == "max":
        return MaxComb([make_stat(s) for s in spec["comps"]])
    raise ValueError(k)


def native_decisions(spec, r, res, n):
    """Published rule per method (None if not applicable)."""
    k = spec.get("native")
    if k is None:
        return None
    spec = dict(spec, field=spec.get("field") or spec["stat"].get("field"))   # the field the published rule reads
    if k == "p_lt_0.05":
        return (np.asarray(r[spec["field"]]) < 0.05).astype(float)
    if k == "stored":
        return np.asarray(r[spec["native_field"]], float)
    if k == "metnull":                         # MET: simulated null of the reference against itself, 95%
        nul = res.get(f"metnull|{spec['metnull_base']}|None|None|{n}")
        if nul is None:
            return None
        t = float(np.quantile(nul[spec["field"]], 0.95))
        return (np.asarray(r[spec["field"]]) > t).astype(float)
    if k == "difr_band":                       # DiFR: batch mean above the honest band's 1% FPR threshold
        band = [res.get(f"null|{v}|None|None|{n}") for v in spec["band"]]
        if any(b is None for b in band):
            return None
        bm = np.concatenate([[np.nanmean(x) for x in b[spec["field"]]] for b in band])
        t = float(np.quantile(bm, 0.99))
        return np.array([float(np.nanmean(x) > t) for x in r[spec["field"]]])
    raise ValueError(k)


def evaluate(spec, res):
    """One method on one results file -> {N: {...}}."""
    out = {}
    for n in spec["nreq"]:
        R = lambda kind, b, s=None, p=None: res.get(f"{kind}|{b}|{s}|{p}|{n}")
        pool = spec["pool"]
        nk = spec.get("null_kind", "null")         # A-mixed has its own null runs
        cal = {v: R(nk, v) for v in pool}
        if any(x is None for x in cal.values()):
            out[str(n)] = {"error": f"missing pool results: {[v for v, x in cal.items() if x is None]}"}
            continue
        need = spec.get("field") or spec["stat"].get("field")
        if need and any(need not in x for x in cal.values()):
            out[str(n)] = {"error": f"field {need} absent at this budget"}
            continue
        ntr = len(next(iter(cal.values()))[spec.get("len_field", spec.get("field") or "zA")])
        ev = {v: np.arange(ntr) % 2 == 0 for v in pool}
        st = make_stat(spec["stat"])
        st.fit(cal, ev)
        t = threshold(st, cal, ev)
        tab = {"thr": t, "heldout": {}, "lovo": {}, "endpoint": {}, "power": {}, "native_fpr": {}, "native_power": {}}
        for v in pool:
            s = st(cal[v])
            tab["heldout"][v] = float(np.mean(s[~ev[v]] > t))
            if len(pool) > 1:
                rest = {u: cal[u] for u in pool if u != v}
                st2 = make_stat(spec["stat"])
                st2.fit(rest, ev)
                tab["lovo"][v] = float(np.mean(st2(cal[v]) > threshold(st2, rest, ev)))
            nd = native_decisions(spec, cal[v], res, n)
            if nd is not None:
                tab["native_fpr"][v] = float(np.mean(nd))
        for e in spec["honest"]:
            r = R("null", e)
            if r is None or (need and need not in r):
                continue
            tab["endpoint"][e] = float(np.mean(st(r) > t))
            nd = native_decisions(spec, r, res, n)
            if nd is not None:
                tab["native_fpr"][e] = float(np.mean(nd))
        for key in spec.get("alt", []):
            kind, (b, s, p) = (key[0], key[1:]) if len(key) == 4 else ("alt", key)
            r = R(kind, b, s, p)
            if r is None or (need and need not in r):
                continue
            name = f"{b}|{s}|{p}" if kind == "alt" else f"{kind}:{b}|{s}|{p}"
            try:
                tab["power"][name] = float(np.mean(st(r) > t))
            except KeyError:          # a component (B / CLS) is not computed for this condition kind
                continue
            nd = native_decisions(spec, r, res, n)
            if nd is not None:
                tab["native_power"][name] = float(np.mean(nd))
        plain = [e for e in spec.get("plain", []) if e in tab["endpoint"]]
        tab["valid"] = bool(max(tab["heldout"].values()) <= 0.07 and
                            (not tab["lovo"] or max(tab["lovo"].values()) <= 0.10) and
                            all(tab["endpoint"][e] <= 0.10 for e in plain))
        tab["gate"] = {"max_heldout": max(tab["heldout"].values()),
                       "max_lovo": max(tab["lovo"].values()) if tab["lovo"] else None,
                       "max_plain_endpoint": max([tab["endpoint"][e] for e in plain], default=None)}
        out[str(n)] = tab
    return out


def main(spec_path, out_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    cache, summary = {}, {}
    for m in spec["methods"]:
        f = m["file"]
        if not os.path.exists(f):
            print(f"{m['name']:28s} (results file not there yet: {f})")
            continue
        if f not in cache:
            d = json.load(open(f, encoding="utf-8"))
            cache[f] = d["results"] if "libraries" not in d else d["results"]
        res = cache[f]
        if "library" in m:
            res = res[m["library"]]
        try:
            summary[m["name"]] = evaluate(m, res)
        except Exception as ex:  # one broken method must not hide the others
            print(f"{m['name']:28s} FAILED: {type(ex).__name__}: {ex}")
            continue
        for n, tab in summary[m["name"]].items():
            if "error" in tab:
                print(f"{m['name']:28s} N={n:>4s} ERROR {tab['error']}")
                continue
            pw = " ".join(f"{k.split('|')[1][:18]}:{k.split('|')[2][-4:]}={v:.3f}" for k, v in list(tab["power"].items())[:6])
            print(f"{m['name']:28s} N={n:>4s} valid={int(tab['valid'])} gate={json.dumps(tab['gate'])} {pw}")
    json.dump(summary, open(out_path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
