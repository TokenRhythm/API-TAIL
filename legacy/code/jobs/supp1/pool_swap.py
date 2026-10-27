"""Recalibrate saved trial statistics against alternative legitimate pools.

Endpoint lists affect validity checks, not calibration thresholds.
--null_trials limits the number of retained null trials before calibration.

Usage: pool_swap.py --pkl results.pkl --panel p30|user|glm --out result.json"""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time

sys.path.insert(0, str(_APITAIL_CODE / 'bbp'))
import prota_report as PR  # noqa: E402

try:                                          # Disable transparent huge pages for shared caches in forked workers.
    import ctypes                             
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(41, 1, 0, 0, 0)
except Exception:
    pass

GL = "glm-5.3__"
OLD = {"p30": ["self", "self2", "h_topk20", "h_T07", "h_T10", "h_gcdef", "h_T07p08", "h_sysdefault", "q_fp8", "q_awq_int4",
               "q_int4_redhat"],
       "user": ["self_u", "self_u2", "hu_T07p08k20", "hu_k20p1", "hu_nok", "hu_T07p095", "hu_T1p1", "hu_T1p1b", "hu_T07p1",
                "hu_sysdefault", "qu_fp8", "qu_awq"],
       "glm": ["self", "gcdef", "g_T07", "g_T06p095", "g_T08p095", "gcdef2", "self2", "g_T05p095", "g_T10p1"]}
ADDED = {"p30": ["h_T10b", "q_fp8_T10"], "user": ["hu_T1p1c", "qu_fp8_T1p1"],
         "glm": ["g_fp8_T05", "g_fp8_T07", "g_sys_T05", "g_sys_T07"]}
PEN = {"p30": ["h_pp15", "h_pp20", "h_rp105", "h_rp110", "h_pp15rp105"],
       "user": ["hu_pp15", "hu_pp20", "hu_rp105", "hu_rp110", "hu_pp15rp105"], "glm": []}
OLD_PLAIN = {
    "p30": ['Q04', 'Q12', 'Q09', 'Q16', 'Q11', 'Q08', 'Q17', 'Q07', 'Q10', 'Q15', 'Q01', 'Q14', 'Q06', 'Q05', 'Q03'],
    "user": ['Q02', 'Q07', 'Q03', 'Q04', 'Q12', 'Q14', 'Q21', 'Q13', 'Q11', 'Q05', 'Q16', 'Q17', 'Q01'],
    "glm": [GL + e for e in ["z-ai_fp8", "together", "legacy_provider_12_fp8", "baseten_fp4", "morph_fp8"]]}
NEW_PLAIN = {
    "p30": ['Q04', 'Q12', 'Q09', 'Q16', 'Q11', 'Q08', 'Q17', 'Q07', 'Q10', 'Q15', 'Q01', 'Q14', 'Q06', 'Q13', 'Q02', 'Q05', 'Q03'],
    "glm": [GL + e for e in ["provider_B", "together", "legacy_provider_12_fp8", "legacy_provider_16_fp8", "morph_fp8", "legacy_provider_09",
                             "gmicloud_fp8", "fireworks"]]}
NEW_PLAIN["user"] = NEW_PLAIN["p30"]
BASE_METHODS = [("ProTA-N95", {"type": "zcount", "field": "z_n95"}), ("ProTA", {"type": "zcount", "field": "z_dechub"}),
                ("ProTA-W", {"type": "zcount", "field": "z_w"}), ("FDG T=1", {"type": "zcount", "field": "z_fdg1"}),
                ("T_align", {"type": "zcount", "field": "z_al"}), ("T+pen (no Huber)", {"type": "zcount", "field": "z_dec"}),
                ("NLL count", {"type": "q99count", "field": "d_nll"}),
                ("log-rank count", {"type": "q99count", "field": "d_lrank"}),
                ("entropy count", {"type": "q99count", "field": "d_ent"}),
                ("RUT", {"type": "scalar", "field": "rut", "transform": "neglog"})]
G = {}


def std_pools(panel, have):
    old, add, pen = OLD[panel], ADDED[panel], PEN[panel]
    pools = {"old": old, "new": old + add}
    for v in add:                                              # one added variant at a time
        pools[f"old+{v}"] = old + [v]
    if panel == "glm":
        pools["new-sys"] = old + ["g_fp8_T05", "g_fp8_T07"]
        pools["new-fp8"] = old + ["g_sys_T05", "g_sys_T07"]
    if pen:
        pools["new+pen"] = old + add + pen
        pools["old+pen"] = old + pen
    return {k: v for k, v in pools.items() if all(x in have for x in v)}


def run_one(job):
    pname, pool, mname, stat, n = job
    res = G["res"]
    extra = {"native": "p_lt_0.05"} if stat.get("field") == "rut" else {}
    honest = sorted({k.split("|")[1] for k in res if k.startswith("null|") and k.split("|")[1] not in pool})
    spec = dict(name=mname, stat=stat, pool=pool, honest=honest, plain=[], alt=G["cell_alt"], nreq=[n], len_field="z_dechub"
                if "z_dechub" in next(iter(res.values())) else stat["field"], **extra)
    try:
        ev = PR.evaluate_checked(spec, res, None)[str(n)]
    except Exception as e:                                     # a field absent in this pickle
        return pname, mname, n, {"error": f"{type(e).__name__}: {e}"}
    if "error" in ev:
        return pname, mname, n, {"error": ev["error"]}
    out = {"thr": ev["thr"], "heldout": ev["heldout"], "lovo": ev["lovo"], "endpoint": ev["endpoint"],
           "power": {k: v for k, v in ev["power"].items() if k in G["cells"]}}
    for lname, plain in (("new", NEW_PLAIN[G["panel"]]), ("old", OLD_PLAIN[G["panel"]])):
        miss = [e for e in plain if e not in ev["endpoint"]]
        mp_ = max([ev["endpoint"][e] for e in plain if e in ev["endpoint"]], default=0.0)
        out[f"valid_{lname}"] = (None if miss else bool(max(ev["heldout"].values()) <= 0.07 and
                                                            max(ev["lovo"].values()) <= 0.10 and mp_ <= 0.10))
        out[f"missing_{lname}"] = miss
        out[f"max_plain_{lname}"] = mp_
        out[f"over_{lname}"] = {e.split("__")[1]: ev["endpoint"][e] for e in plain if ev["endpoint"].get(e, 0) > 0.10}
    out["max_heldout"], out["max_lovo"] = max(ev["heldout"].values()), max(ev["lovo"].values())
    out["argmax_lovo"] = max(ev["lovo"], key=ev["lovo"].get)
    return pname, mname, n, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--panel", required=True, choices=["p30", "user", "glm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="200")
    ap.add_argument("--null_trials", type=int, default=0)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--pools", default="")
    ap.add_argument("--cells", default="", help="comma list of base|sub|policy keys to report powers for")
    a = ap.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.pkl, "rb"))
    res = d["results"]
    if a.null_trials:
        for k, v in res.items():
            if k.startswith("null|") or k.startswith("metnull|"):
                res[k] = {f: x[: a.null_trials] for f, x in v.items()}
    have = {k.split("|")[1] for k in res if k.startswith("null|")}
    pools = std_pools(a.panel, have)
    if a.pools:
        pools = {kv.split("=")[0]: kv.split("=")[1].split(",") for kv in a.pools.split(";")}
    fields = set().union(*(set(v) for v in res.values()))
    meths = [m for m in BASE_METHODS if m[1]["field"] in fields]
    meths += [(f"MET same-budget L{f[3:]}", {"type": "scalar", "field": f}) for f in sorted(fields)
              if f.startswith("met") and f[3:].isdigit()]
    G["res"], G["panel"] = res, a.panel
    G["alt"] = [list(x) for x in sorted({tuple(k.split("|")[1:4]) for k in res if k.startswith("alt|")})]
    G["cells"] = a.cells.split(",") if a.cells else [
        f"{b}|{s}|{p}" for b, s, p in G["alt"] if p in ("random_0.05", "random_0.1", "random_0.25")]
    G["cell_alt"] = [c.split("|") for c in G["cells"]]
    ns = [int(x) for x in a.ns.split(",")]
    jobs = [(pn, pv, mn, st, n) for pn, pv in pools.items() for mn, st in meths for n in ns]
    print(f"loaded {a.pkl} in {time.time() - t0:.0f}s; pools {list(pools)}; methods {[m for m, _ in meths]}; "
          f"{len(jobs)} jobs", flush=True)
    with mp.get_context("fork").Pool(min(a.procs, len(jobs))) as P:
        outs = P.map(run_one, jobs, chunksize=1)
    S = {"pkl": a.pkl, "panel": a.panel, "null_trials": a.null_trials, "pools": pools, "results": {}}
    for pn, mn, n, o in outs:
        S["results"].setdefault(pn, {}).setdefault(mn, {})[str(n)] = o
    json.dump(S, open(a.out, "w", encoding="utf-8"))
    for pn in pools:
        for mn, _ in meths:
            for n in ns:
                o = S["results"][pn][mn][str(n)]
                if "error" in o:
                    print(f"{pn:14s} {mn:22s} N{n} ERROR {o['error'][:80]}")
                    continue
                print(f"{pn:14s} {mn:22s} N{n} valid new/old {o['valid_new']}/{o['valid_old']} ho {o['max_heldout']:.3f} "
                      f"lovo {o['max_lovo']:.3f} ({o['argmax_lovo']}) plain new {o['max_plain_new']:.3f} old "
                      f"{o['max_plain_old']:.3f} over-new {o['over_new']}", flush=True)
    print(f"saved {a.out} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
