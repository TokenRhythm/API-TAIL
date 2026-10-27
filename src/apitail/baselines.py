"""Array-based RUT, Hamming-MMD, and detector-count controls.

RUT takes audited response scores and local reference scores per prompt.
Matched-budget MET uses two audited responses per prompt, so N audited
requests cover N/2 prompts. Local references do not consume the audit budget.
Scores and token IDs must come from the claimed reference model."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DETECTOR_NAMES = ("negative_log_likelihood", "log_rank", "window_entropy")


def token_midranks(log_probs, ranks, topk_log_probs=None):
    """Compute midranks, resolving ties inside the stored top-K window.
    
    ranks are 1 + the number of strictly greater full-vocabulary logits.
    Ties outside the stored window retain the supplied full-vocabulary rank.
    Missing top-K entries may be NaN. Empty responses are rejected."""
    lp = np.asarray(log_probs, dtype=np.float64)
    rk = np.asarray(ranks, dtype=np.float64)
    if lp.ndim != 1 or lp.size == 0 or rk.shape != lp.shape:
        raise ValueError("log_probs and ranks must be nonempty equal-length vectors")
    if not np.isfinite(lp).all() or not np.isfinite(rk).all() or (rk < 1).any():
        raise ValueError("finite log probabilities and ranks >= 1 are required")
    if topk_log_probs is None:
        return rk.copy()
    top = np.asarray(topk_log_probs, dtype=np.float16)
    if top.ndim != 2 or top.shape[0] != lp.size or top.shape[1] == 0:
        raise ValueError("topk_log_probs must have shape (tokens, K), K >= 1")
    target = lp.astype(np.float16)[:, None]
    valid = ~np.isnan(top)
    greater = np.sum(valid & (top > target), axis=1)
    equal = np.sum(valid & (top == target), axis=1)
    return np.where((rk <= top.shape[1]) & (equal >= 1),
                    greater + (equal + 1) / 2.0, rk)


def mean_log_rank(log_probs, ranks, topk_log_probs=None):
    """Response score used by RUT and by the log-rank count control."""
    return float(np.log(token_midranks(log_probs, ranks, topk_log_probs)).mean())


def response_detectors(log_probs, ranks, topk_log_probs):
    """Return mean NLL, mean log midrank, and renormalized-window entropy.
    
    Entropy is conditional on the stored top-K window at T=1. The top-20
    control uses float16 stored log-probabilities."""
    lp = np.asarray(log_probs, dtype=np.float64)
    rank_score = mean_log_rank(lp, ranks, topk_log_probs)
    top = np.asarray(topk_log_probs, dtype=np.float64)
    valid = np.isfinite(top)
    if not valid.any(axis=1).all():
        raise ValueError("each token must have at least one finite top-K log probability")
    safe = np.where(valid, top, -np.inf)
    exp = np.exp(safe - np.max(safe, axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    entropy = -(prob * np.log(np.maximum(prob, 1e-300))).sum(axis=1).mean()
    return np.array([-lp.mean(), rank_score, entropy], dtype=np.float64)


def detector_threshold(calibration_values, quantile=0.99):
    """Pooled response-score quantile using NumPy's linear interpolation.
    
    Pool response scores from calibration draws across legitimate configurations,
    including repeated drawn rows. The resulting response threshold requires
    separate composite calibration of the audit-level counts."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    vals = np.asarray(calibration_values, dtype=np.float64).ravel()
    vals = vals[~np.isnan(vals)]
    if not vals.size or not np.isfinite(vals).all():
        raise ValueError("at least one finite calibration value is required")
    return float(np.quantile(vals, quantile, method="linear"))


def detector_counts(batch_values, threshold):
    """Strict upper-tail counts on the last axis; NaNs do not exceed cutoff."""
    values = np.asarray(batch_values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0 or not np.isfinite(threshold):
        raise ValueError("a nonempty response axis and finite threshold are required")
    count = np.sum(values > threshold, axis=-1)
    return int(count) if count.ndim == 0 else count


def calibrate_continuous(mapping, alpha=0.05):
    """Calibrate a strict upper-tail Monte Carlo rule across null settings.

    ``mapping`` maps each legitimate configuration to a nonempty finite 1-D
    array of whole-audit statistics. For RUT use ``-log(p)`` (the ``statistic``
    returned by :func:`rut_statistic`), not the p-value. For MET use MMD^2,
    including negative estimates. Each method needs its own calibration.

    The cutoff is the largest per-configuration ``higher`` (1-alpha) quantile.
    Reject only when a new statistic is strictly greater than ``threshold``.
    ``rates`` are strict exceedance rates on the supplied calibration arrays;
    they are not independent false-alarm estimates or population guarantees.
    This is a finite Monte Carlo cutoff, not an exact randomized boundary rule.
    """
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("mapping must contain at least one null configuration")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a finite number strictly between zero and one") from exc
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    arrays, quantiles = {}, {}
    for name, values in mapping.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"configuration {name!r} needs a nonempty finite 1-D array")
        arrays[name] = arr
        quantiles[name] = float(np.quantile(arr, 1 - alpha, method="higher"))
    threshold = max(quantiles.values())
    return {
        "threshold": threshold,
        "quantiles": quantiles,
        "decisive": [name for name, value in quantiles.items() if value == threshold],
        "trials": {name: int(arr.size) for name, arr in arrays.items()},
        "rates": {name: float(np.mean(arr > threshold)) for name, arr in arrays.items()},
        "alpha": alpha,
        "quantile_method": "higher",
        "rule": "statistic > threshold",
        "calibration": "monte_carlo_continuous",
    }


def randomized_ranks(target_scores, reference_scores, uniforms):
    """Randomized ranks: (L + U*(E+1))/(m+1), including ties.
    
    References are local samples, not additional audited API requests.
    Pass uniforms explicitly to share random draws across comparisons."""
    target = np.asarray(target_scores, dtype=np.float64)
    refs = np.asarray(reference_scores, dtype=np.float64)
    u = np.asarray(uniforms, dtype=np.float64)
    if target.ndim != 1 or not target.size or refs.ndim != 2:
        raise ValueError("target_scores is (N,), reference_scores is (N,m)")
    if refs.shape[0] != target.size or refs.shape[1] == 0 or u.shape != target.shape:
        raise ValueError("one target, reference row, and uniform are required per prompt")
    if not all(np.isfinite(a).all() for a in (target, refs, u)) or ((u < 0) | (u >= 1)).any():
        raise ValueError("scores must be finite and uniforms must lie in [0,1)")
    less = (refs < target[:, None]).sum(axis=1)
    equal = (refs == target[:, None]).sum(axis=1)
    return (less + u * (equal + 1)) / (refs.shape[1] + 1)


def cramer_von_mises_uniform(ranks):
    """Return (W^2, p-value) using the public SciPy finite-sample API."""
    from scipy.stats import cramervonmises
    r = np.asarray(ranks, dtype=np.float64)
    if r.ndim != 1 or r.size < 2 or not np.isfinite(r).all() or ((r < 0) | (r > 1)).any():
        raise ValueError("at least two finite ranks in [0,1] are required")
    result = cramervonmises(r, "uniform")
    return float(result.statistic), float(result.pvalue)


def rut_statistic(target_scores, reference_scores, *, uniforms=None, seed=0,
                  reference_count=100):
    """RUT statistic using the first reference_count samples per prompt.
    
    The default reference_count is 100. Composite calibration uses
    statistic=-log(max(p, 1e-300))."""
    target = np.asarray(target_scores, dtype=np.float64)
    refs = np.asarray(reference_scores, dtype=np.float64)
    if not isinstance(reference_count, (int, np.integer)) or reference_count < 1:
        raise ValueError("reference_count must be a positive integer")
    if refs.ndim != 2 or refs.shape[1] < reference_count:
        raise ValueError("insufficient local references per prompt")
    if uniforms is None:
        uniforms = np.random.default_rng(seed).random(target.shape)
    ranks = randomized_ranks(target, refs[:, :reference_count], uniforms)
    w2, p = cramer_von_mises_uniform(ranks)
    return {"w2": w2, "pvalue": p, "statistic": float(-np.log(max(p, 1e-300))),
            "ranks": ranks, "api_requests": int(target.size),
            "local_references": int(target.size * reference_count)}


def pad_token_ids(responses, length=30, pad=-1):
    """Truncate/pad token ID sequences. Padding contributes to the Hamming kernel."""
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError("length must be a positive integer")
    rows = list(responses)
    out = np.full((len(rows), length), pad, dtype=np.int64)
    for i, row in enumerate(rows):
        x = np.asarray(row)
        if x.ndim != 1 or (x.size and not np.issubdtype(x.dtype, np.integer)):
            raise ValueError("responses must be vectors of integer token IDs")
        n = min(len(x), length)
        out[i, :n] = x[:n]
    return out


def hamming_mmd2(api_tokens, reference_tokens):
    """Unbiased same-prompt Hamming MMD^2; negative estimates are retained.
    
    Inputs contain one (samples, L) integer array per prompt. At least two
    samples are needed on each side. Sample counts may differ across prompts;
    same-side and cross-side sums use their respective pair-count denominators."""
    xs, ys = list(api_tokens), list(reference_tokens)
    if not xs or len(xs) != len(ys):
        raise ValueError("equal nonzero numbers of API and reference prompts are required")
    sxx = syy = sxy = 0.0
    nxx = nyy = nxy = 0
    length = None
    for x, y in zip(xs, ys):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or y.ndim != 2 or min(len(x), len(y)) < 2:
            raise ValueError("each prompt needs at least two samples on each side")
        if x.shape[1] == 0 or x.shape[1] != y.shape[1] or (length is not None and x.shape[1] != length):
            raise ValueError("all samples must have a common positive padded length")
        if not np.issubdtype(x.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Hamming inputs must contain integer token IDs")
        length = x.shape[1]
        xx = (x[:, None, :] == x[None, :, :]).mean(axis=-1)
        yy = (y[:, None, :] == y[None, :, :]).mean(axis=-1)
        xy = (x[:, None, :] == y[None, :, :]).mean(axis=-1)
        sxx += float(xx.sum() - np.trace(xx)); nxx += xx.size - len(x)
        syy += float(yy.sum() - np.trace(yy)); nyy += yy.size - len(y)
        sxy += float(xy.sum()); nxy += xy.size
    return float(sxx / nxx - 2 * sxy / nxy + syy / nyy)


def paired_hamming_mmd2(api_tokens, reference_tokens):
    """Matched-budget Hamming MMD for arrays (N/2 prompts, 2 samples, L).
    
    The default response length L is 30. Two distinct stored completions must
    be selected when a pair uses the same source. Reference sampling consumes
    local work, not audited API requests."""
    x, y = np.asarray(api_tokens), np.asarray(reference_tokens)
    if x.ndim != 3 or y.ndim != 3 or x.shape != y.shape or x.shape[1] != 2:
        raise ValueError("paired inputs must have equal shape (prompts,2,tokens)")
    return hamming_mmd2(x, y)
