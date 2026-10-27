"""Classify endpoint parameter compliance from probes and serving metadata.

Classification uses quantization labels, documented prompt modifications,
completion lengths, a one-sided Fisher comparison of token agreement at
T=0 versus T=1, and probe coverage. Detector scores are not used.
Class priority is special, uncovered, then plain; all reasons are retained.

Usage: plain_rule2.py probe_dir metadata.json qwen_tokenizer gemma_tokenizer out.json"""
import json
import os
import sys

from scipy.stats import fisher_exact
from transformers import AutoTokenizer

DOC_INJECT = {"legacy_provider_05": "docs.legacy_provider_05.ai: legacy_provider_05 adds its own system prompts unless "
                        "legacy_provider_05_parameters.include_legacy_provider_05_system_prompt=false"}
FP4 = ("fp4", "nvfp4", "mxfp4")


def rows(path):
    if not os.path.exists(path):
        return []
    out = {}
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("status") == "ok":
            out[(r["pid"], int(r["k"]))] = r
    return list(out.values())


def first16(tok, s):
    return tuple(tok(s, add_special_tokens=False)["input_ids"][:16])


def det_share(rs, tok, same):
    by = {}
    for r in rs:
        by.setdefault(r["pid"], {})[int(r["k"])] = first16(tok, r.get("content") or "")
    pairs = [v for v in by.values() if 0 in v and 1 in v]
    if not pairs:
        return None, 0
    eq = sum(v[0] == v[1] for v in pairs)
    return (eq if same else len(pairs) - eq) / len(pairs), len(pairs)


def temp_test(s0, n0, s1, n1):
    """one-sided Fisher: agreement at T=0 (s0 * n0 pairs) > agreement at T=1 ((1 - s1) * n1 pairs)."""
    a0, a1 = round(s0 * n0), round((1 - s1) * n1)
    return float(fisher_exact([[a0, n0 - a0], [a1, n1 - a1]], alternative="greater")[1])


def mean_tokens(rs):
    xs = [(r.get("usage") or {}).get("completion_tokens") for r in rs]
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs) if xs else None), len(xs)


def main(D, snap, qtok, gtok, out):
    S = json.load(open(snap, encoding="utf-8"))["endpoints"]
    res = {}
    for fam, model, tokp in (("q38", "qwen/qwen3.8-27b", qtok), ("g4", "google/gemma-4-31b-it", gtok)):
        tok = AutoTokenizer.from_pretrained(tokp)
        for e in S[model]:
            tag = e["tag"]
            q = e.get("quantization") or "unknown"
            name = f"{model.split('/')[1]}__{tag.replace('/', '_')}"
            st = {"tag": tag, "quantization": q}
            s0, n0 = det_share(rows(f"{D}/{fam}/det_T0/{name}.jsonl"), tok, True)
            s1, n1 = det_share(rows(f"{D}/{fam}/det_T1/{name}.jsonl"), tok, False)
            st.update(T0_identical=s0, T0_pairs=n0, T1_different=s1, T1_pairs=n1)
            st["temp_p"] = temp_test(s0, n0, s1, n1) if n0 and n1 else None
            st["mt_p30"], st["mt_p30_n"] = mean_tokens(rows(f"{D}/{fam}/echo_p30/{name}.jsonl"))
            st["mt_user"], st["mt_user_n"] = mean_tokens(rows(f"{D}/{fam}/echo_user/{name}.jsonl"))
            for pan in ("p30", "user"):
                unc, spec = [], []
                if n0 < 15 or n1 < 15:
                    unc.append(f"R4 precheck pairs T0 {n0} / T1 {n1} < 15")
                ek = {"p30": "mt_p30", "user": "mt_user"}[pan]
                if st[ek + "_n"] < 8:
                    unc.append(f"R4 echo ok {st[ek + '_n']} < 8")
                if q.lower() in FP4 or "fp4" in tag.lower():
                    spec.append(f"R1 {q if q != 'unknown' else 'fp4 tag'}")
                if tag.split("/")[0] in DOC_INJECT:
                    spec.append("R2 documented prompt injection")
                lim = {"p30": 37.5, "user": 500.0}[pan]
                if st[ek] is not None and st[ek] > lim:
                    spec.append(f"R3a mean output {st[ek]:.1f} > {lim}")
                if st["temp_p"] is not None and st["temp_p"] >= 0.05:
                    spec.append(f"R3b temperature not honoured (agree T0 {s0:.2f} vs T1 {1 - s1:.2f}, p {st['temp_p']:.3f})")
                cls = "special" if spec else ("uncovered" if unc else "plain")
                res.setdefault(fam, {}).setdefault(pan, {})[name] = dict(st, cls=cls, reasons=unc + spec)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for fam, pans in res.items():
        for pan, eps in pans.items():
            print(f"== {fam} {pan}: plain {sum(v['cls'] == 'plain' for v in eps.values())}, special "
                  f"{sum(v['cls'] == 'special' for v in eps.values())}, uncovered "
                  f"{sum(v['cls'] == 'uncovered' for v in eps.values())}")
            for name, v in sorted(eps.items(), key=lambda kv: (kv[1]["cls"], kv[0])):
                mt = {k: round(v[k], 1) for k in ("mt_p30", "mt_user") if v.get(k) is not None}
                print(f"  {v['cls']:9s} {name:40s} T0id {v['T0_identical']} T1diff {v['T1_different']} p "
                      f"{None if v['temp_p'] is None else round(v['temp_p'], 4)} mt {mt} | {'; '.join(v['reasons'])}")


if __name__ == "__main__":
    main(*sys.argv[1:])
