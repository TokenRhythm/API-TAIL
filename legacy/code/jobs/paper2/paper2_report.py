"""Aggregate calibrated audits, paired bootstrap comparisons and diagnostics.

The configuration supplies panel outputs, bootstrap shards, optional scorer
comparisons and collection records. Missing measurements remain None rather
than becoming failures or zero power. Paired differences preserve replicate
identity. Results are written as JSON and Markdown.

Usage: paper2_report.py report_config.json"""
import json
import math
import os
import sys

import numpy as np

N0 = "200"
ALPHA = 0.05
COMP = [("T_align-fsv", "T_align"), ("T_align-fsv", "ProTA-N95"), ("T_align", "N95 T-only id"),
        ("T_align-fsv", "N95-full id"), ("T_align", "FDG T=1"), ("T_align-fsv", "T_MLE-full"),
        ("T_align", "T Huber"), ("T_align", "T+pen (no Huber)")]
HF_FAMILY = ("T_align-fsv", "T_MLE-full", "T Huber-fsv", "FDG-fsv", "N95-full id", "N95-full MLE")
AXIS5_ALT = {"hf": ("vllm1", "vllm2"), "top20": ("tf_hf",)}
CORE20 = ["T_align", "T Huber", "T+pen (no Huber)", "ProTA", "N95 T-only id", "N95 T-only Huber", "N95 pen id",
          "ProTA-N95", "T_MLE", "N95 T-only MLE", "T_align-fsv", "T_MLE-full", "T Huber-fsv", "FDG-fsv", "N95-full id",
          "N95-full MLE", "FDG T=1", "NLL count", "log-rank count", "entropy count"]


def registered_methods(n, P):
    """Return the core methods plus protocol-specific methods that were evaluated."""
    reg = list(CORE20)
    if n.endswith("_p30"):
        reg += ["RUT", "MET same-budget"]
    optional = (["RUT"] if not n.endswith("_p30") else []) + (["T+pen-full", "ProTA-full"] if n.startswith("q38") else [])
    if P is not None and P.ev is not None:
        reg += [m for m in optional if m in P.ev["methods"]]       # optional methods count once evaluated
    return reg


def axis5_registered(n, P):
    """Return whether the panel has the required alternative-scorer comparison."""
    if n == "q38_p30":
        return True
    return n == "q38_user" and bool(P.spec.get("alt_scorers"))
M10 = ["T_align", "T_MLE", "T Huber", "ProTA-N95", "N95 T-only id", "FDG T=1", "T_align-fsv", "T_MLE-full",
       "N95-full id", "RUT"]
S1_METHODS = ["RUT", "MET same-budget", "MET native", "KBF-N", "IRIS-E", "IRIS-U"]
S1_MUST = ["RUT|q38_p30", "MET same-budget|q38_p30", "MET native|q38_p30", "RUT|g4_p30"]


def wilson(p, n, z=1.96):
    if p is None or not n:
        return None
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cellkey(c):
    return f"{c[0]}|{c[1]}|{c[2]}"


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, TypeError):
        return None


class Panel:
    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        self.ev = load(spec["eval"])
        self.R = self.ev["res"] if self.ev else {}
        self.trials = self.ev["args"]["trials_null"] if self.ev else None
        self.trials_alt = self.ev["args"]["trials_alt"] if self.ev else None
        self.boot = None
        parts, missing = [], []
        for pre in spec.get("boot", []):
            meta = load(pre + ".json")
            if meta is None or not os.path.exists(pre + ".npz"):
                missing.append(pre)
                continue
            z = np.load(pre + ".npz")
            parts.append((meta, z["rates"][: int(z["done"])]))
        if parts:
            self.boot = {"meta": parts[0][0], "rates": np.concatenate([p[1] for p in parts])}
        # Report incomplete bootstrap shards instead of treating them as full runs.
        self.boot_info = {"shards_registered": len(spec.get("boot", [])), "shards_loaded": len(parts),
                          "shards_missing": missing, "replicates": int(sum(len(p[1]) for p in parts)),
                          "complete": bool(parts) and not missing and int(sum(len(p[1]) for p in parts)) == 2000}

    @property
    def measured(self):
        return self.ev is not None

    def x(self, arm, m, n=N0):
        return (self.R.get(n, {}).get(arm) or {}).get(m)

    def valid(self, arm, m, key="TEST_int", n=N0):
        x = self.x(arm, m, n)
        return None if x is None else x["verdict"][key]["valid"]

    def power(self, arm, m, cell, rule="power", n=N0, dom="TEST"):
        x = self.x(arm, m, n)
        return None if x is None or cell is None else x[dom][rule].get(cellkey(cell))

    def fpr(self, arm, m, ep, rule="endpoint", n=N0, dom="TEST"):
        x = self.x(arm, m, n)
        return None if x is None else x[dom][rule].get(ep)

    def series(self, arm, m, item, rule=0):
        """bootstrap rates by replicate (NaN where the replicate lacks the value); None if not recorded"""
        if self.boot is None:
            return None
        meta = self.boot["meta"]
        if arm not in meta["arms"] or m not in meta["methods"] or item not in meta["items"]:
            return None
        return self.boot["rates"][:, meta["arms"].index(arm), meta["methods"].index(m), meta["items"].index(item), rule]


def paired(sa, sb):
    """differences over the replicates where both values exist (replicate identity kept)"""
    if sa is None or sb is None:
        return None
    ok = ~np.isnan(sa) & ~np.isnan(sb)
    return (sa - sb)[ok] if ok.any() else None


def holm(fam):
    order = sorted(fam, key=lambda k: fam[k]["p"])
    m_, going, adj = len(order), True, 0.0
    for r, k in enumerate(order, 1):
        step = ALPHA / (m_ + 1 - r)
        adj = max(adj, min(1.0, (m_ + 1 - r) * fam[k]["p"]))
        rej = going and fam[k]["p"] <= step
        going = rej
        fam[k].update(holm_rank=r, holm_step=step, p_holm=adj, rejected=bool(rej))
    return fam


def comparisons(P, arm, cname):
    """Compute eight paired comparisons with a fixed Holm family size.
    Missing or invalid comparisons receive p=1 and an explicit status."""
    cell = P.spec["cells"].get(cname)
    item = "sub|" + cellkey(cell) if cell else None
    fam = {}
    for a_, b_ in COMP:
        pa, pb = P.power(arm, a_, cell), P.power(arm, b_, cell)
        va, vb = P.valid(arm, a_), P.valid(arm, b_)
        d = paired(P.series(arm, a_, item), P.series(arm, b_, item)) if item else None
        rec = {"estimate": None if pa is None or pb is None else pa - pb, "valid": [va, vb], "p": 1.0, "ci95": None,
               "B": 0}
        if pa is None or pb is None or va is None or vb is None or d is None or not len(d):
            rec["status"] = "unmeasured"
        elif not (va and vb):
            rec["status"] = "invalid"
            rec["ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            rec["B"] = int(len(d))
        else:
            rec["status"] = "ok"
            rec["ci95"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
            rec["p"] = float((1 + np.sum(d <= 0)) / (len(d) + 1))
            rec["B"] = int(len(d))
        fam[f"{a_} - {b_}"] = rec
    return holm(fam)


def id_diffs(P):
    out = {}
    for cname, cell in P.spec["cells"].items():
        item = "sub|" + cellkey(cell)
        for m in (P.ev or {}).get("methods", []):
            a, b = P.power("S_main+ID", m, cell), P.power("S_main", m, cell)
            xa, xb = P.x("S_main+ID", m), P.x("S_main", m)
            d = paired(P.series("S_main+ID", m, item), P.series("S_main", m, item))
            out.setdefault(cname, {})[m] = {
                "diff": None if a is None or b is None else a - b,
                "ci95": None if d is None else [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                "Kstar": None if xa is None or xb is None else [xb["thr"], xa["thr"]]}
    return out


def endpoint_ci(P):
    """every N, pool, rule and domain: rate, Wilson 95%; bootstrap 95% where the bootstrap recorded it (N 200, TEST)"""
    out = {}
    for n, byarm in P.R.items():
        for arm, bym in byarm.items():
            if arm.startswith("_"):
                continue
            for m, x in bym.items():
                if x is None:
                    continue
                for dom in [d for d in ("TEST", "T1", "T2") if d in x]:
                    for ri, (rk, key) in enumerate((("int", "endpoint"), ("exact", "endpoint_exact"))):
                        for ep, r in x[dom][key].items():
                            rec = {"rate": r, "wilson": wilson(r, P.trials), "boot95": None}
                            if n == N0 and dom == "TEST":
                                s = P.series(arm, m, f"honest|{ep}", ri)
                                s = None if s is None else s[~np.isnan(s)]
                                if s is not None and len(s):
                                    rec["boot95"] = [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]
                            out.setdefault(n, {}).setdefault(arm, {}).setdefault(m, {}).setdefault(dom, {}) \
                               .setdefault(rk, {})[ep] = rec
    return out


def flips(P, alt):
    """Compare method validity across settings; unavailable settings remain None."""
    out = {}
    for m in M10:
        x = P.x("S_main", m)
        if x is None:
            out[m] = None
            continue
        v = x["verdict"]

        def diff(a, b):
            return None if a is None or b is None else a != b
        t = v["TEST_int"]
        r = {"1": diff(t["valid"], v["TEST_exact"]["valid"]),
             "2": None if t["main_ok"] is None else t["main_ok"] != (t["redraws_kept"] >= 16),
             "3": diff(v.get("T1_int", {}).get("valid"), v.get("T2_int", {}).get("valid"))}
        g = t["gate_sensitivity"]
        r["4"] = tri([diff(g.get("0.08"), g.get("0.12")), diff(t["valid_without_worst"], t["valid"])])
        # Compare each method with the scorers appropriate to its score family.
        
        need_alt = AXIS5_ALT["hf" if m in HF_FAMILY else "top20"]
        flags = {"main": t["valid"]}
        for k in need_alt:
            ev = (alt or {}).get(k)
            y = None if ev is None else ((ev["res"].get(N0) or {}).get("S_main") or {}).get(m)
            flags[k] = None if y is None else y["verdict"]["TEST_int"]["valid"]
        present = [f for f in flags.values() if f is not None]
        r["5"] = True if len(set(present)) > 1 else (False if len(present) == len(flags) else None)
        out[m] = r
    return out


def tri(vals):
    """three-state OR: True if any True, False if all present and False, None otherwise"""
    if any(v is True for v in vals):
        return True
    return False if all(v is False for v in vals) else None


tor = tri


def tand(vals):
    """three-state AND: False if any False, True if all present and True, None otherwise"""
    if any(v is False for v in vals):
        return False
    return True if vals and all(v is True for v in vals) else None


def canon(path):
    """canonical absolute output path (same project root for job lists and ledger rows)"""
    return os.path.normcase(os.path.realpath(os.path.abspath(path.replace(chr(92), "/"))))


def count_rule(sat, counted):
    if counted < 2:
        return "不可判"
    if counted >= 3 and sat == counted:
        return "支持"
    if (1 <= sat < counted) or (counted == 2 and sat == counted):
        return "部分支持"
    return "不支持"


def collection_tables(C):
    """Aggregate coverage and cumulative spend once per output file.
    Impute eligible unbilled requests from successful token usage and snapshot prices."""
    cc = C.get("collection")
    if not cc:
        return None
    led = {}
    if os.path.exists(cc["ledger"]):
        for l in open(cc["ledger"], encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            run = r.get("run", "").replace(chr(92), "/")
            if canon(run).startswith(canon("runs/paper2") + os.sep) or canon(run).startswith(canon("runs/paper2") + "/"):
                led[canon(run)] = float(r.get("spent_usd") or 0)    # last row per output file
    snap = (load(cc.get("snapshot", "")) or {}).get("endpoints", {})
    price = {(m, e["tag"]): (float(e["pricing"]["prompt"]), float(e["pricing"]["completion"]))
             for m, eps in snap.items() for e in eps}
    rows, seen = [], set()
    for jl in cc["jobs"]:
        for j in load(jl) or []:
            out = canon(j["out"])
            if out in seen:                                  # one row (and one imputation) per output file
                continue
            seen.add(out)
            n_prompts = sum(1 for _ in open(j["prompts"], encoding="utf-8")) if os.path.exists(j["prompts"]) else None
            if j.get("limit") and n_prompts is not None:
                n_prompts = min(n_prompts, int(j["limit"]))
            ok = {}
            if os.path.exists(out):
                for l in open(out, encoding="utf-8"):
                    try:
                        r = json.loads(l)
                    except Exception:
                        continue
                    if r.get("status") == "ok":
                        ok[(r["pid"], int(r["k"]))] = r.get("usage") or {}
            imputed = None
            if j["tag"].startswith("provider_B") and (j["model"], j["tag"]) in price:
                pi, po = price[(j["model"], j["tag"])]
                imputed = sum((u.get("prompt_tokens") or 0) * pi + (u.get("completion_tokens") or 0) * po
                              for u in ok.values())
            exp = None if n_prompts is None else n_prompts * j["n"]
            rows.append({"job_list": os.path.basename(jl), "out": out, "requests_expected": exp, "ok": len(ok),
                         "coverage": None if not exp else len(ok) / exp, "ledger_usd": led.get(out),
                         "provider_imputed_usd": imputed})
    in_lists = {r["out"] for r in rows}
    return {"rows": rows, "ledger_usd_campaign": sum(led.values()),
            "ledger_usd_outside_job_lists": sum(v for k, v in led.items() if k not in in_lists),
            "provider_imputed_usd": sum(r["provider_imputed_usd"] or 0 for r in rows)}


def main():
    C = json.load(open(sys.argv[1], encoding="utf-8"))
    PN = {n: Panel(n, s) for n, s in C["panels"].items()}
    probes = load(C.get("probes_file", "")) or C.get("probes", {}) or {}
    rep = {"comparisons": {}, "id": {}, "endpoint_ci": {}, "flips": {}, "S": {}, "pred": {}, "meta": {}}
    for n, P in PN.items():
        if not P.measured:
            continue
        rep["comparisons"][n] = {arm: comparisons(P, arm, P.spec["primary"]) for arm in ("S_main", "S_pen")
                                 if P.x(arm, "T_align") is not None}
        rep["id"][n] = id_diffs(P)
        rep["endpoint_ci"][n] = endpoint_ci(P)
        alt = {k: load(v) for k, v in P.spec.get("alt_scorers", {}).items()}
        rep["flips"][n] = flips(P, alt)
        rep["meta"][n] = {"scorer": P.ev.get("scorer"), "eval_seconds": P.ev.get("seconds"), "plain_eff": P.ev["plain_eff"],
                          "plain_dropped_coverage": P.ev.get("plain_dropped_coverage"),
                          "plain_no_coverage_record": P.ev.get("plain_no_coverage_record"),
                          "n_cal_prompts": P.ev["n_cal_prompts"], "n_test_prompts": P.ev["n_test_prompts"],
                          "bootstrap": P.boot_info}
    rep["collection"] = collection_tables(C)
    rep["source_times"] = load(C.get("source_times", ""))    # Generation and scoring time per source.
    e6 = C.get("e6", {})
    rep["scoring"] = {k: load(v) for k, v in e6.items()}
    S = rep["S"]
    # Classify only cells with measured validity and power.
    cells = {}
    for meth in S1_METHODS:
        for n, P in PN.items():
            if meth in ("RUT", "MET same-budget"):
                v = P.valid("S_main", meth) if P.measured else None
                pw = P.power("S_main", meth, P.spec["cells"].get(P.spec["primary"])) if P.measured else None
            else:
                pr = (probes.get(meth) or {}).get(n)
                v, pw = (None, None) if pr is None else (pr.get("valid"), pr.get("power"))
            if v is None or (v and pw is None):
                cls = "未测"
            else:
                cls = "无效" if not v else ("有效有功效" if pw > 0.10 else "有效无功效")
            cells[f"{meth}|{n}"] = cls
    must_missing = [c for c in S1_MUST if cells.get(c, "未测") == "未测"]
    measured = [k for k, v in cells.items() if v != "未测"]
    wp = [k for k in measured if cells[k] == "有效有功效"]
    S["S1"] = {"cells": cells, "measured": len(measured), "valid_with_power": wp, "must_missing": must_missing,
               "verdict": "不可判" if (len(measured) < 6 or must_missing) else
               ("支持" if not wp else "部分支持" if len(wp) == 1 else "不支持")}

    # Count comparisons only when both methods are valid and paired estimates exist.
    def line(comp_key, ma, mb, dmin, extra=None):
        counted = sat = 0
        detail = {}
        for n, P in PN.items():
            c = rep["comparisons"].get(n, {}).get("S_main", {}).get(comp_key)
            if not P.measured or c is None or c["status"] != "ok" or not (P.valid("S_main", ma) and P.valid("S_main", mb)):
                detail[n] = "not counted"
                continue
            ex = extra(P) if extra else True
            if ex is None:
                detail[n] = "not counted (component unmeasured)"
                continue
            counted += 1
            ok = c["rejected"] and c["estimate"] >= dmin and ex
            sat += bool(ok)
            detail[n] = {"rejected": c["rejected"], "estimate": c["estimate"], "satisfied": bool(ok)}
        return {"counted": counted, "satisfied": sat, "verdict": count_rule(sat, counted), "detail": detail}

    def fdg_low(P):
        p = P.power("S_main", "FDG T=1", P.spec["cells"].get(P.spec["primary"]))
        return None if p is None else p <= 0.10
    S["S2"] = line("T_align - FDG T=1", "T_align", "FDG T=1", 0.30, fdg_low)
    l3 = line("T_align - N95 T-only id", "T_align", "N95 T-only id", 0.20)
    l4 = line("T_align-fsv - N95-full id", "T_align-fsv", "N95-full id", 0.20)
    vs = [l3["verdict"], l4["verdict"]]
    S["S3"] = {"top20": l3, "full": l4, "verdict":
               "不可判" if all(v == "不可判" for v in vs) else "不支持" if "不支持" in vs else
               "支持" if all(v == "支持" for v in vs) else "部分支持"}
    # Require both valid methods and a measured interval.
    s4 = {}
    for n, P in PN.items():
        for key, comp in (("T_align - T Huber", "T Huber"), ("T_align - T+pen (no Huber)", "T+pen (no Huber)")):
            c = rep["comparisons"].get(n, {}).get("S_main", {}).get(key)
            if c is None or c["ci95"] is None or not (P.valid("S_main", "T_align") and P.valid("S_main", comp)):
                continue
            lo, hi = c["ci95"]
            s4[f"{n}|{key}"] = "成分有益" if hi < 0 else ("无一致增益" if lo > -0.10 else "不确定")
    ncell = len({k.split("|")[0] for k in s4})
    S["S4"] = {"pairs": s4, "counted_cells": ncell, "verdict":
               "不可判" if ncell < 2 else "不支持" if "成分有益" in s4.values() else
               ("支持" if ncell >= 3 and all(v == "无一致增益" for v in s4.values()) else "部分支持")}
    # A measured change establishes a flip; missing settings remain unknown.
    
    
    
    
    meas_p = [n for n, f in rep["flips"].items()
              if any(r is not None and any(v is not None for v in r.values()) for r in f.values())]
    status = {}
    for a in "12345":
        vals = []
        for n in meas_p:
            P = PN[n]
            if a == "5" and not axis5_registered(n, P):
                continue
            reg = registered_methods(n, P)
            for m in M10:
                if m not in reg:
                    continue
                r = rep["flips"][n].get(m) if m in P.ev["methods"] else None
                vals.append(None if r is None else r.get(a))
        status[a] = tri(vals) if vals else None
    axes = {a: status[a] is True for a in "12345"}
    k = sum(axes.values())
    S["S5"] = {"axes": axes, "axis_status": status, "measured_panels": len(meas_p),
               "verdict": "不可判" if len(meas_p) < 2 else "支持" if k >= 3 else "部分支持" if k == 2 else "不支持"}
    # Families without identity-prompt pools remain unmeasured.
    fam_eff = {}
    for fam in ("qwen", "gemma"):
        ps = [P for P in PN.values() if P.measured and P.spec["family"] == fam
              and any(P.x("S_main+ID", m) is not None for m in ("T_align-fsv", "T_align"))]
        if not ps:
            fam_eff[fam] = None
            continue
        # An observed witness establishes the effect.
        # Absence requires every candidate witness to be measured and fail.
        
        cost_w, cover_w = [], []
        for P in ps:
            for m in ("T_align-fsv", "T_align"):
                d = rep["id"][P.name].get(P.spec["primary"], {}).get(m)
                if not d or d["diff"] is None:
                    cost_w.append(None)
                elif d["diff"] > -0.10:
                    cost_w.append(False)                       # the point difference alone excludes a cost witness
                else:
                    cost_w.append(None if d["ci95"] is None else bool(d["ci95"][1] < 0))
                for ep in P.ev["plain_eff"]:
                    a, b = P.fpr("S_main", m, ep), P.fpr("S_main+ID", m, ep)
                    cover_w.append(tand([None if a is None else a > 0.10, None if b is None else b <= 0.10]))
        cost, cover = tri(cost_w), tri(cover_w)
        effect = tri([cost, cover])
        fam_eff[fam] = {"cost": cost, "coverage": cover, "effect": effect,
                        "witnesses": {"cost": len(cost_w), "coverage": len(cover_w)}}
    eff = {f: (None if v is None else v["effect"]) for f, v in fam_eff.items()}
    q, g = eff.get("qwen"), eff.get("gemma")
    if q is None and g is None:
        v6 = "不可判"
    elif g is None:
        v6 = "部分支持" if q else "不可判（单家族）"
    elif q is None:
        v6 = "部分支持" if g else "不可判（单家族）"
    else:
        ne = int(q) + int(g)
        v6 = "支持" if ne == 2 else "部分支持" if ne == 1 else "不支持"
    S["S6"] = {"families": fam_eff, "verdict": v6}
    # Rank valid methods at the primary condition; powers within 0.01 are tied.
    tops = {}
    for n, P in PN.items():
        if not P.measured:
            continue
        cell = P.spec["cells"].get(P.spec["primary"])
        val = [(P.power("S_main", m, cell), m) for m in P.ev["methods"] if P.valid("S_main", m)
               and P.power("S_main", m, cell) is not None]
        if len(val) <= 1:
            continue
        best = max(v for v, _ in val)
        tops[n] = sorted(m for v, m in val if best - v <= 0.01)
    names = list(tops)
    allsame = bool(names) and bool(set.intersection(*[set(tops[a]) for a in names]))
    anypair = any(set(tops[a]) & set(tops[b]) for i, a in enumerate(names) for b in names[i + 1:])
    S["S7"] = {"tops": tops, "verdict": "不可判" if len(names) < 2 else "支持" if allsame else
               "部分支持" if anypair else "不支持"}
    rep["pred"] = predictions(PN, rep, probes)
    json.dump(rep, open(C["out"] + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    write_md(rep, C["out"] + ".md")
    print(json.dumps({k: v["verdict"] for k, v in S.items()}, ensure_ascii=False))


def predictions(PN, rep, probes):
    """Evaluate compound conditions with three-state logic.
    Missing measurements remain unknown; observed failures can falsify a conjunction."""
    q30, qu, gu = (PN.get(k) for k in ("q38_p30", "q38_user", "g4_user"))
    q30 = q30 if q30 is not None and q30.measured else None
    qu = qu if qu is not None and qu.measured else None
    gu = gu if gu is not None and gu.measured else None
    endpoint_one, endpoint_two = "Q02", "Q13"
    out = {}

    def c(v, f):
        """component: f(v) when v is measured, else None"""
        return None if v is None else bool(f(v))

    def inr(lo, hi):
        return lambda v: lo <= v <= hi

    def put(k, value, fc, fz):
        out[k] = {"value": value, "result": {True: "命中", False: "未命中", None: "不可判"}[fc], "falsified": fz}

    def power_row(k, P, m, cell, lo, hi, fz_rule):
        p = P.power("S_main", m, P.spec["cells"].get(cell))
        put(k, p, c(p, inr(lo, hi)), c(p, fz_rule))

    if q30:
        cc = q30.spec["cells"]
        a, b, v = (q30.fpr("S_main", "T_align", endpoint_one), q30.fpr("S_main", "T_align", endpoint_one, "endpoint_exact"),
                   q30.valid("S_main", "T_align"))
        put("P1", [a, b, v], tand([c(a, inr(0, 0.133)), c(b, inr(0.026, 0.176)), c(v, lambda x: x is True)]),
            c(v, lambda x: x is False))
        a, b, v = (q30.fpr("S_main", "T_align-fsv", endpoint_one), q30.fpr("S_main", "T_align-fsv", endpoint_one, "endpoint_exact"),
                   q30.valid("S_main", "T_align-fsv"))
        put("P2", [a, b, v], tand([c(a, inr(0.05, 0.35)), c(b, inr(0.19, 0.35)), c(v, lambda x: x is False)]),
            c(a, lambda x: x < 0.05))
        a, v = q30.fpr("S_main", "T_MLE", endpoint_one), q30.valid("S_main", "T_MLE")
        put("P3", [a, v], tand([c(a, inr(0.089, 0.239)), c(v, lambda x: x is False)]), c(v, lambda x: x is True))
        x = q30.x("S_main", "ProTA-N95")
        a, v = (None if x is None else x["TEST"]["plain_max_int"]), q30.valid("S_main", "ProTA-N95")
        put("P4", [a, v], tand([c(a, inr(0.021, 0.171)), c(v, lambda x_: x_ is True)]), c(v, lambda x_: x_ is False))
        power_row("P5", q30, "T_align", "Q2", 0.27, 0.52, lambda p: p < 0.12)
        power_row("P6", q30, "ProTA-N95", "Q2", 0, 0.20, lambda p: p > 0.30)
        power_row("P7", q30, "FDG T=1", "Q2", 0, 0.10, lambda p: p > 0.30)
        power_row("P8", q30, "T_align-fsv", "Q1", 0.30, 0.95, lambda p: p < 0.15)
        cq = [cc["Q1"][0], "qwen3.6-27b__provider_B", 0.25]
        a, b = q30.power("S_main", "T_align", cq), q30.power("S_main", "N95 T-only id", cq)
        dd = paired(q30.series("S_main", "T_align", "sub|" + cellkey(cq)),
                    q30.series("S_main", "N95 T-only id", "sub|" + cellkey(cq)))
        lo = None if dd is None else float(np.quantile(dd, 0.025))
        d = None if a is None or b is None else a - b
        put("P9", [d, lo], tand([c(d, inr(0.35, 0.80)), c(lo, lambda x_: x_ > 0)]), c(d, lambda x_: x_ < 0.20))
        x = q30.x("S_main", "RUT")
        if x is None:
            put("P10", None, None, None)
        else:
            nat = x["TEST"]["native"]
            vv = [q30.valid("S_main", "RUT"), q30.power("S_main", "RUT", cc.get("Q1")), x["old_gate"]["max_lovo"],
                  nat.get(f"honest|{endpoint_one}"), nat.get(f"honest|{endpoint_two}")]
            fc = tand([c(vv[0], lambda x_: x_ is True), c(vv[1], inr(0, 0.10)), c(vv[2], inr(0.30, 0.80)),
                       c(vv[3], lambda x_: x_ >= 0.80), c(vv[4], lambda x_: x_ >= 0.80)])
            fz = tor([c(vv[0], lambda x_: x_ is False), c(vv[1], lambda x_: x_ > 0.30)])
            put("P10", vv, fc, fz)
        v, pw = q30.valid("S_main", "MET same-budget"), q30.power("S_main", "MET same-budget", cc.get("Q1"))
        # forecast: invalid, or valid with power in [0, 0.10]; falsifier: valid and power > 0.30
        fc = tor([c(v, lambda x_: x_ is False), tand([c(v, lambda x_: x_ is True), c(pw, inr(0, 0.10))])])
        fz = tand([c(v, lambda x_: x_ is True), c(pw, lambda x_: x_ > 0.30)])
        put("P11", [v, pw], fc, fz)
        fl, agree = [], []
        for m in ("T Huber", "ProTA"):
            y = q30.x("S_main", m)
            vi = None if y is None else y["verdict"]["TEST_int"]["valid"]
            vx = None if y is None else y["verdict"]["TEST_exact"]["valid"]
            fl.append(tand([c(vi, lambda x_: x_ is True), c(vx, lambda x_: x_ is False)]))
            agree.append(None if vi is None or vx is None else vi == vx)
        put("P19", fl, tor(fl), tand(agree))
        d = rep["id"].get("q38_p30", {}).get("Q1", {}).get("T_align-fsv")
        dk = None if not d or d["Kstar"] is None else d["Kstar"][1] - d["Kstar"][0]
        dv = None if not d else d["diff"]
        put("P23", d, tand([c(dk, lambda x_: x_ >= 1), c(dv, inr(-0.40, -0.05))]),
            tand([c(dk, lambda x_: x_ == 0), c(dv, lambda x_: abs(x_) < 0.02)]))
    if qu:
        cc = qu.spec["cells"]
        x = qu.x("S_main", "T_align-fsv")
        a, v = (None if x is None else x["TEST"]["plain_max_int"]), qu.valid("S_main", "T_align-fsv")
        put("P12", [a, v], tand([c(a, inr(0, 0.121)), c(v, lambda x_: x_ is True)]), c(v, lambda x_: x_ is False))
        power_row("P13", qu, "T_align", "U2", 0, 0.10, lambda p: p > 0.30)
        power_row("P14", qu, "T_align-fsv", "U2", 0.13, 0.38, lambda p: p < 0.05 or p > 0.60)
        power_row("P15", qu, "T_align-fsv", "U1", 0.20, 1.0, lambda p: p < 0.05)
        power_row("P16", qu, "ProTA-N95", "U1", 0, 0.135, lambda p: p > 0.30)
        a, b = qu.power("S_main", "T_align", cc.get("U5")), qu.power("S_main", "T_align", cc.get("U5"), "power_exact")
        put("P17", [a, b], tand([c(a, inr(0.22, 0.47)), c(b, inr(0.29, 0.54))]), c(a, lambda x_: x_ < 0.07))
        a, b = qu.power("S_main", "ProTA-N95", cc.get("U5")), qu.power("S_main", "ProTA-N95", cc.get("U5"), "power_exact")
        put("P18", [a, b], tand([c(a, inr(0.44, 0.69)), c(b, inr(0.58, 0.83))]), c(a, lambda x_: x_ < 0.29))
        ids = rep["id"].get("q38_user", {})
        ds = [ids.get(cn, {}).get(m, {}).get("diff") for cn in ("U1", "U2", "U3", "U4", "U5")
              for m in registered_methods("q38_user", qu)]
        put("P24", [min([x_ for x_ in ds if x_ is not None], default=None),
                    max([x_ for x_ in ds if x_ is not None], default=None), sum(x_ is None for x_ in ds)],
            tand([c(x_, lambda y_: abs(y_) <= 0.05) for x_ in ds]), tor([c(x_, lambda y_: y_ <= -0.15) for x_ in ds]))
    if gu:
        x, xi = gu.x("S_main", "T_align-fsv"), gu.x("S_main+ID", "T_align-fsv")
        pe = gu.ev["plain_eff"]

        def over(y):
            return [None] if y is None or not pe else [c(y["TEST"]["endpoint"].get(e), lambda v_: v_ > 0.10) for e in pe]
        o, oi = over(x), over(xi)
        k = None if None in o else sum(o)
        ki = None if None in oi else sum(oi)
        fc = tor(o)                                                    # >= 1 endpoint above 0.10
        zero = lambda lst: None if tor(lst) is None else not tor(lst)  # the side's count is 0 (three-state)
        fz = tand([zero(o), zero(oi)])                                 # falsifier: 0 under S_main and under +ID
        put("P25", [sum(1 for z_ in o if z_), k, ki], fc, fz)
        cc = gu.spec["cells"]
        v = gu.valid("S_main", "T_align-fsv")
        pw = gu.power("S_main", "T_align-fsv", cc.get("G3"))
        put("P26", [v, pw], c(pw, lambda x_: x_ >= 0.60) if v is True else None,
            c(pw, lambda x_: x_ < 0.30) if v is True else None)
        a, b = gu.power("S_main", "ProTA-N95", cc.get("G3")), gu.power("S_main", "T_align-fsv", cc.get("G3"))
        d = None if a is None or b is None else a - b
        put("P27", d, c(d, lambda x_: x_ <= 0.10), c(d, lambda x_: x_ > 0.20))
    sc = rep["scoring"]
    sr = sc.get("selfrepeat") or {}
    sh = sr.get("bitwise_equal_share_min")
    ppc = sr.get("per_panel_compared") or {}
    okr = (sr.get("all_selected_present") is True and sh is not None
           and all((ppc.get(p_) or 0) >= 2000 for p_ in ("q38_p30", "q38_user", "g4_p30", "g4_user")))
    put("P20", [sh, sr.get("responses"), sr.get("complete")], (sh == 1.0) if okr else None, (sh < 1.0) if okr else None)
    hv = sc.get("hf_vs_vllm") or {}
    ok = (hv.get("responses") or 0) >= 200
    md, q9 = hv.get("median"), hv.get("q90")
    put("P21", [md, q9, hv.get("responses")],
        tand([c(md, lambda x_: x_ <= 0.01), c(q9, lambda x_: x_ <= 0.10)]) if ok else None,
        tor([c(md, lambda x_: x_ > 0.03), c(q9, lambda x_: x_ > 0.30)]) if ok else None)
    if q30 is None or not q30.spec.get("alt_scorers"):
        put("P22", None, None, None)
    else:
        alt = {k: load(v) for k, v in q30.spec["alt_scorers"].items()}
        ax5 = []
        for m in HF_FAMILY:                           # Compare each full-vocabulary method across its available scorers.
            x = q30.x("S_main", m)
            flags = {"hf": None if x is None else x["verdict"]["TEST_int"]["valid"]}
            for k in AXIS5_ALT["hf"]:
                y = None if alt.get(k) is None else ((alt[k]["res"].get(N0) or {}).get("S_main") or {}).get(m)
                flags[k] = None if y is None else y["verdict"]["TEST_int"]["valid"]
            pres = [f for f in flags.values() if f is not None]
            ax5.append(True if len(set(pres)) > 1 else (False if len(pres) == len(flags) else None))
        n_true = sum(1 for a_ in ax5 if a_ is True)
        put("P22", ax5, tor(ax5), None if tor(ax5) is None else n_true == 0)
    for k, meth in (("P28", ("KBF-N", "IRIS-E", "IRIS-U")), ("P29", ("MET native",))):
        vals = [(probes.get(m) or {}).get("q38_p30") for m in meth]
        if all(v_ is None for v_ in vals):
            continue                                  # An unmeasured probe remains unavailable.
        # A valid method above 0.10 disproves the low-power condition.
        # Establishing it requires known validity and power for all relevant methods.
        viol, unknown, n_valid = False, False, 0
        fzw = []
        for v_ in vals:
            vv_ = None if v_ is None else v_.get("valid")
            pp_ = None if v_ is None else v_.get("power")
            fzw.append(tand([c(vv_, lambda x_: x_ is True), c(pp_, lambda p_: p_ > 0.30)]))
            if vv_ is True:
                n_valid += 1
                if pp_ is None:
                    unknown = True
                elif pp_ > 0.10:
                    viol = True
            elif vv_ is None and (pp_ is None or pp_ > 0.10):
                unknown = True                          # a measured power <= 0.10 satisfies it whatever the validity
        fc = False if viol else (None if unknown or n_valid == 0 else True)
        put(k, vals, fc, tor(fzw))
    S5 = rep["S"]["S5"]
    ax = S5["axis_status"]
    kx = sum(1 for v_ in ax.values() if v_ is True)
    if S5["measured_panels"] < 2:
        put("P30", kx, None, None)
    else:
        nfalse = sum(1 for v_ in ax.values() if v_ is False)
        fc = True if kx >= 3 else (False if kx + (5 - kx - nfalse) < 3 else None)
        fz = False if kx >= 2 else (True if kx + (5 - kx - nfalse) <= 1 else None)
        put("P30", kx, fc, fz)
    for i in range(1, 31):
        out.setdefault(f"P{i}", {"value": None, "result": "未做", "falsified": None})
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def write_md(rep, path):
    L = ["# Paper round 2: registered quantities (auto-generated by paper2_report.py)", "",
         "## Verdicts S1-S7", "", "| S | verdict |", "|---|---|"]
    L += [f"| {k} | {v['verdict']} |" for k, v in rep["S"].items()]
    L += ["", "## Bootstrap replicates per panel (registered B = 2000 = 8 shards x 250)", "",
          "| panel | replicates | shards loaded / registered | complete |", "|---|---|---|---|"]
    for n, m in rep["meta"].items():
        bi = m.get("bootstrap") or {}
        L.append(f"| {n} | {bi.get('replicates')} | {bi.get('shards_loaded')} / {bi.get('shards_registered')} | "
                 f"{bi.get('complete')} |")
    L += ["", "## Registered comparisons (N=200, integer rule; one-sided bootstrap p, Holm over 8)", ""]
    for n, arms in rep["comparisons"].items():
        for arm, fam in arms.items():
            L += [f"### {n} {arm}", "", "| comparison | estimate | 95% CI | p | Holm rejected | status |",
                  "|---|---|---|---|---|---|"]
            for k, v in fam.items():
                ci = "—" if not v.get("ci95") else f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                est = "—" if v["estimate"] is None else f"{v['estimate']:+.3f}"
                L.append(f"| {k} | {est} | {ci} | {v['p']:.4f} | {v['rejected']} | {v['status']} |")
            L.append("")
    L += ["## Predictions", "", "| # | value | forecast | falsifier crossed |", "|---|---|---|---|"]
    L += [f"| {k} | {json.dumps(v['value'], ensure_ascii=False, default=float)} | {v['result']} | {v['falsified']} |"
          for k, v in rep["pred"].items()]
    col = rep.get("collection")
    if col:
        L += ["", f"## Collection: campaign ledger {col['ledger_usd_campaign']:.2f} USD (of which outside the job lists "
                  f"{col['ledger_usd_outside_job_lists']:.2f}), provider_B imputed {col['provider_imputed_usd']:.2f} USD",
              "", "| output | ok / expected | coverage | ledger USD | provider_B imputed USD |", "|---|---|---|---|---|"]
        f3 = lambda v: "—" if v is None else f"{v:.3f}"
        L += [f"| {r['out']} | {r['ok']} / {r['requests_expected']} | {f3(r['coverage'])} | {f3(r['ledger_usd'])} | "
              f"{f3(r['provider_imputed_usd'])} |" for r in col["rows"]]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
