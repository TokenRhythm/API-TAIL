"""Add temperature-alignment and likelihood-estimation rules to the evaluator.

MLE selects the grid point minimizing abs(sum S); standardized alignment
minimizes abs(sum S/sqrt(V)). Both count response scores above 3 at the selected
temperature. S/V arrays must align with the response keys in the grid caches.
Top-20 controls use the same cached z surface after temperature selection."""


from pathlib import Path as _APITAIL_PATH
import os as _APITAIL_OS
_APITAIL_CODE = next(p for p in _APITAIL_PATH(__file__).resolve().parents if p.name == "code")
_APITAIL_ARCHIVE = _APITAIL_PATH(_APITAIL_OS.environ.get("APITAIL_ARCHIVE_ROOT", str(_APITAIL_CODE.parent / "assets-not-included")))

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, str(_APITAIL_CODE / 'jobs/supp1'))
import engine as EN  # noqa: E402
import run_eval  # noqa: E402

# Temperature-only estimators read S/V arrays through Panel.stats.

EN.METHODS["T_MLE"] = ("prof", "Z", False, "mle")
EN.METHODS["N95 T-only MLE"] = ("prof", "ZN", False, "mle")
# F uses the full vocabulary; NF uses its renormalized 0.95 nucleus.
# Standardized alignment sums S/sqrt(V); MLE sums S.
EN.METHODS["T_align-fsv"] = ("prof", "Z", False, "svid", "F")
EN.METHODS["T_MLE-full"] = ("prof", "Z", False, "mle", "F")
EN.METHODS["N95-full id"] = ("prof", "Z", False, "svid", "NF")
EN.METHODS["N95-full MLE"] = ("prof", "Z", False, "mle", "NF")
SVFAM = {"Z": "Z", "ZN": "N"}
# Top-20 controls share the cached z surface after selecting temperature.

ZGRID = {"Z": "Z", "N": "ZN"}
SV_DIR = None
SVFULL_DIR = None

_init0 = EN.Panel.__init__
_stats0 = EN.Panel.stats
_sv0 = EN.stat_values
_pi00 = EN.pi0_of


def _init(self, *args, **kw):
    _init0(self, *args, **kw)
    parts = {f: [] for f in ("SZ", "VZ", "SN", "VN")}
    for name in self.names:
        d = pickle.load(open(os.path.join(SV_DIR, f"{name}.sv.pkl"), "rb"))
        sel = [j for j, k in enumerate(d["keys"]) if k[0] in self.pidx]
        assert [tuple(d["keys"][j]) for j in sel] == [tuple(k) for k in self.keys[name]], f"sv key order {name}"
        for f in parts:
            parts[f].append(np.asarray(d[f])[sel])
    for f in parts:
        self.grids[f] = torch.from_numpy(np.concatenate(parts[f])).to(self.device)
    if SVFULL_DIR:
        fparts = {f: [] for f in ("SF0", "VF0", "SNF0", "VNF0")}
        for name in self.names:
            d = pickle.load(open(os.path.join(SVFULL_DIR, f"{name}.svfull.pkl"), "rb"))
            sel = [j for j, k in enumerate(d["keys"]) if k[0] in self.pidx]
            assert [tuple(d["keys"][j]) for j in sel] == [tuple(k) for k in self.keys[name]], f"svfull key order {name}"
            for f in fparts:
                fparts[f].append(np.asarray(d["fields"][f])[sel])
        for f, g in (("SF0", "SF"), ("VF0", "VF"), ("SNF0", "SNF"), ("VNF0", "VNF")):
            self.grids[g] = torch.from_numpy(np.concatenate(fparts[f])).to(self.device)


def _is_sv(d):
    return d[0] == "prof" and d[3] in ("mle", "svid")


def _stats(self, rows, uni, methods, tau=3.0, batch=None, rut_p=True):
    mle = [m for m in methods if _is_sv(EN.METHODS[m])]
    rest = [m for m in methods if m not in mle]
    out = _stats0(self, rows, uni, rest, tau=tau, batch=batch, rut_p=rut_p) if rest else {}
    if not mle:
        return out
    rt = rows if torch.is_tensor(rows) else torch.from_numpy(rows).to(self.device)
    T_, N = rt.shape
    bs = max(1, int(3e8 // (N * EN.NT * 8)))
    for m in mle:
        d = EN.METHODS[m]
        fam = d[4] if len(d) > 4 else SVFAM[d[1]]
        S, V = self.grids["S" + fam], self.grids["V" + fam]
        est = d[3]
        K = torch.empty(T_, dtype=torch.int32, device=self.device)
        TH = torch.empty(T_, dtype=torch.int32, device=self.device)
        for b0 in range(0, T_, bs):
            r = rt[b0: b0 + bs]
            Sb = S[r]                                                  # B, N, NT
            Vb = V[r]
            zb = Sb / torch.sqrt(torch.clamp(Vb, min=1e-12))           # responses with V = 0 (nothing kept) give z = 0
            tk = torch.argmin((zb if est == "svid" else Sb).sum(1).abs(), dim=1)   # first minimum, as the engine
            idx = tk[:, None, None].expand(-1, N, 1)
            if fam in ZGRID:                                           # top-20 families: the engine's cached z grid
                zg = self.grids[ZGRID[fam]]
                z = zg[r, tk[:, None].expand(-1, N) * EN.NAG].double()
            else:                                                      # full-vocabulary families: S / sqrt(V)
                z = zb.gather(2, idx)[..., 0]
            K[b0: b0 + len(r)] = (z > tau).sum(1).int()
            TH[b0: b0 + len(r)] = (tk * EN.NAG).int()
        out[m] = K.cpu().numpy()
        out[("theta", m)] = TH.cpu().numpy()
    return out


def _stat_values(method, raw, pool, ev):
    if EN.METHODS[method][0] == "mle":                                 # Handle methods whose kind is not prof.
        return {k: np.asarray(v, float) for k, v in raw.items()}, {}
    return _sv0(method, raw, pool, ev)


def _pi0_of(method, raw, pool, ev, N, tau_info):
    if EN.METHODS[method][0] == "mle":
        return max(max(float(np.mean(raw[v][ev[v]])) / N for v in pool), 1e-4)
    return _pi00(method, raw, pool, ev, N, tau_info)


EN.Panel.__init__ = _init
EN.Panel.stats = _stats
EN.stat_values = _stat_values
EN.pi0_of = _pi0_of

if __name__ == "__main__":
    i = sys.argv.index("--sv_dir")
    SV_DIR = sys.argv[i + 1]
    del sys.argv[i: i + 2]
    if "--svfull_dir" in sys.argv:
        i = sys.argv.index("--svfull_dir")
        SVFULL_DIR = sys.argv[i + 1]
        del sys.argv[i: i + 2]
    run_eval.main()
