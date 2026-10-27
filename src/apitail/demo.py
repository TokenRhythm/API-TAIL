"""Synthetic score banks, calibration, and audit."""
from pathlib import Path
import numpy as np
from .core import temperature_grid, moments_from_logits
from .io import ScoreBank, save_scores, write_json
from .pipeline import calibrate, audit, evaluate


def demo(output):
    root = Path(output)
    if root.exists():
        raise FileExistsError("Demo output already exists; choose a new directory")
    root.mkdir(parents=True)
    grid = temperature_grid()
    rng = np.random.default_rng(17)
    logits = rng.normal(size=(12, 24))
    meta = {"model_id": "synthetic-categorical", "scoring": "numpy-float64-reference",
            "system": "", "chat_kwargs": {}, "max_length": 12, "synthetic": True}
    def make(n, temp, prefix, alternate=False):
        S, V = [], []
        dist = np.exp(((-logits if alternate else logits) / temp) - ((-logits if alternate else logits) / temp).max(1, keepdims=True))
        dist /= dist.sum(1, keepdims=True)
        for i in range(n):
            y = np.asarray([rng.choice(24, p=p) for p in dist])
            out = moments_from_logits(logits, y, grid)
            S.append(out["S"]); V.append(out["V"])
        return ScoreBank(np.stack(S), np.stack(V), grid, np.asarray([prefix+str(i) for i in range(n)]),
                         np.zeros(n, int), np.full(n, 12), meta.copy(), {})
    for tag, temp in [("t07", .7), ("t10", 1.0)]:
        save_scores(root/(tag+".npz"), make(40, temp, "cal-"))
    honest, substitute = make(40, 1.0, "test-"), make(40, 1.0, "test-", True)
    save_scores(root/"honest.npz", honest)
    save_scores(root/"substitute.npz", substitute)
    batch = ScoreBank(honest.S[:20], honest.V[:20], grid, honest.pid[:20], honest.k[:20],
                      honest.n_tokens[:20], meta.copy(), {})
    save_scores(root/"batch.npz", batch)
    write_json(root/"pool.json", {"budget": 20, "trials": 200, "seed": 9, "method": "align",
                                 "members": {"t07": "t07.npz", "t10": "t10.npz"}})
    calibrate(root/"pool.json", root/"calibration.json")
    a = audit(root/"batch.npz", root/"calibration.json", root/"audit.json")
    h = evaluate(root/"honest.npz", root/"calibration.json", root/"honest-eval.json", trials=200)
    s = evaluate(root/"honest.npz", root/"calibration.json", root/"sub-eval.json",
                 trials=200, substitute=root/"substitute.npz", rho=.25)
    report = dict(synthetic=True, audit_reject=a["reject"], honest_rate=h["rate"], substitution_rate=s["rate"],
                  warning="This toy example tests plumbing; these rates are not research findings.")
    write_json(root/"summary.json", report)
    return report
