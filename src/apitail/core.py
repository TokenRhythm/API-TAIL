"""API-TAIL numerical statistics and composite calibration.

The mle option minimizes the absolute likelihood score equation on the
temperature grid, rather than maximizing discrete-grid likelihood.
Calibration thresholds describe the supplied empirical null samples.
NumPy moments use float64; the HF backend uses float32 exponentials and
products with float64 moment sums."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

_METHODS = frozenset({"align", "mle", "fixed", "huber"})
_HUBER = 1.345
_VARIANCE_DENOMINATOR_FLOOR = 1e-12


def temperature_grid() -> np.ndarray:
    """Return the 151 temperatures from 0.30 to 1.80 in increments of 0.01."""
    return np.round(np.arange(30, 181, dtype=np.float64) / 100.0, 2)


def _float_array(value, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if arr.ndim != ndim or any(size == 0 for size in arr.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _grid(value) -> np.ndarray:
    grid = _float_array(value, "grid", 1)
    if np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must contain strictly increasing positive temperatures")
    return grid


def _finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def moments_from_logits(logits, targets, grid=None, variance_floor=1e-6) -> dict:
    """Compute full-vocabulary response moments S(T) and V(T).
    
    logits has shape (response positions, vocabulary); targets gives the observed
    token at each position. Each logit row must predict the corresponding target
    under teacher forcing. Apply variance_floor to each token variance before
    summing. Temperatures are processed sequentially to bound working memory."""
    lg = _float_array(logits, "logits", 2)
    target = np.asarray(targets)
    if target.ndim != 1 or len(target) != len(lg):
        raise ValueError("targets must have one token ID per logit row")
    if not np.issubdtype(target.dtype, np.integer) or np.issubdtype(target.dtype, np.bool_):
        raise ValueError("targets must be integer token IDs")
    if np.any(target < 0) or np.any(target >= lg.shape[1]):
        raise ValueError("target token IDs are outside the vocabulary")
    tg = temperature_grid() if grid is None else _grid(grid)
    floor = _finite_scalar(variance_floor, "variance_floor")
    if floor < 0:
        raise ValueError("variance_floor must be nonnegative")
    S, V = np.empty(len(tg)), np.empty(len(tg))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            centered = lg - lg.max(axis=1, keepdims=True)
            for i, temperature in enumerate(tg):
                u = centered / temperature
                logq = u - np.log(np.exp(u).sum(axis=1, keepdims=True))
                q = np.exp(logq)
                # Avoid 0 * log(0), and square only values having nonzero mass.
                safe_logq = np.where(q > 0, logq, 0.0)
                H = -(q * safe_logq).sum(axis=1)
                deviation = np.where(q > 0, safe_logq + H[:, None], 0.0)
                var = (q * deviation * deviation).sum(axis=1)
                S[i] = (-logq[np.arange(len(lg)), target] - H).sum()
                V[i] = np.maximum(var, floor).sum()
    except FloatingPointError as exc:
        raise ValueError("logits or temperatures exceed the supported numerical range") from exc
    if not np.all(np.isfinite(S)) or not np.all(np.isfinite(V)):
        raise ValueError("non-finite score moments")
    return {"S": S, "V": V}


def _validate_batch(S, V, grid, method: str, z_cutoff):
    s = _float_array(S, "S", 2)
    v = _float_array(V, "V", 2)
    tg = _grid(grid)
    if s.shape != v.shape or s.shape[1] != len(tg):
        raise ValueError("S and V must have matching (responses, temperatures) shapes")
    if np.any(v < 0):
        raise ValueError("V must be nonnegative")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}")
    cutoff = _finite_scalar(z_cutoff, "z_cutoff")
    if method == "fixed" and not np.any(np.isclose(tg, 1.0, rtol=0, atol=1e-12)):
        raise ValueError("fixed-temperature testing requires T=1 in the grid")
    return s, v, tg, cutoff


def _select(s, v, grid, method):
    # s, v: (trials, responses, temperatures). argmin chooses the first tie.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = s / np.sqrt(np.maximum(v, _VARIANCE_DENOMINATOR_FLOOR))
            if method == "fixed":
                index = int(np.flatnonzero(np.isclose(grid, 1.0, rtol=0, atol=1e-12))[0])
                tk = np.full(len(s), index, dtype=np.int64)
            else:
                objective = s if method == "mle" else np.clip(z, -_HUBER, _HUBER) if method == "huber" else z
                tk = np.argmin(np.abs(objective.sum(axis=1)), axis=1)
    except FloatingPointError as exc:
        raise ValueError("moments exceed the supported numerical range") from exc
    selected = np.take_along_axis(z, tk[:, None, None], axis=2)[..., 0]
    return tk, selected


def batch_statistic(S, V, grid, method="align", z_cutoff=3.0) -> dict:
    """Fit one batch temperature and count strictly upper-tail responses.
    
    align minimizes abs(sum(S / sqrt(V))); mle minimizes abs(sum(S)); huber
    minimizes abs(sum(clip(S / sqrt(V), -1.345, 1.345))); fixed uses T=1.
    Variance denominators are floored at 1e-12, including zero-token nucleus
    responses."""
    s, v, tg, cutoff = _validate_batch(S, V, grid, method, z_cutoff)
    tk, z = _select(s[None], v[None], tg, method)
    index = int(tk[0])
    return {"temperature": float(tg[index]), "temperature_index": index,
            "count": int(np.count_nonzero(z[0] > cutoff)), "n": len(s),
            "z": z[0].tolist(), "method": method}


def batch_counts(S, V, draws, grid, method="align", z_cutoff=3.0):
    """Evaluate audit draws, returning (integer counts, fitted temperatures).

    ``draws`` has shape (trials, requests) and indexes rows of S/V. Repeated row
    indices are allowed for bootstrap use. Working arrays are bounded by chunking
    trials; the input score bank remains in memory. Prompt-level sampling and
    calibration/test separation must be enforced by the caller.
    """
    s, v, tg, cutoff = _validate_batch(S, V, grid, method, z_cutoff)
    idx = np.asarray(draws)
    if idx.ndim != 2 or any(size == 0 for size in idx.shape):
        raise ValueError("draws must be a nonempty (trials, requests) array")
    if not np.issubdtype(idx.dtype, np.integer) or np.issubdtype(idx.dtype, np.bool_):
        raise ValueError("draws must contain integer row indices")
    if np.any(idx < 0) or np.any(idx >= len(s)):
        raise ValueError("draw row index is outside the score bank")
    n_trials, n_requests = idx.shape
    # At most about 32 MiB for the three main batch arrays, except a single
    # unusually large audit which necessarily needs more than this bound.
    chunk = max(1, min(256, (32 * 1024 * 1024) // (n_requests * len(tg) * 8 * 3)))
    counts = np.empty(n_trials, dtype=np.int64)
    temperatures = np.empty(n_trials, dtype=np.float64)
    for lo in range(0, n_trials, chunk):
        hi = min(n_trials, lo + chunk)
        tk, z = _select(s[idx[lo:hi]], v[idx[lo:hi]], tg, method)
        counts[lo:hi] = (z > cutoff).sum(axis=1)
        temperatures[lo:hi] = tg[tk]
    return counts, temperatures


def _counts(value, name):
    a = _float_array(value, name, 1)
    if np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return a


def _critical_pair(counts, alpha):
    values, frequencies = np.unique(counts, return_counts=True)
    probabilities = frequencies / frequencies.sum()
    survival = 1.0 - np.cumsum(probabilities)
    index = int(np.flatnonzero(survival <= alpha + 1e-12)[0])
    gamma = (alpha - survival[index]) / probabilities[index]
    return float(values[index]), float(np.clip(gamma, 0.0, 1.0))


def calibrate_counts(mapping, alpha=0.05) -> dict:
    """Calibrate the worst-member integer threshold and randomized alternative.

    Each mapping entry is an array of honest simulated audit counts for one
    legitimate configuration. The ordinary rule is K > threshold, with the
    empirical (1-alpha) quantile computed by NumPy's ``higher`` convention.
    The randomized diagnostic rejects above ``exact.threshold`` and with
    probability ``exact.gamma`` at equality. The latter matches the empirical
    reference distribution, not a finite-sample guarantee on unseen traffic.
    In tied decisive configurations, the mapping's insertion order is retained.
    """
    a = _finite_scalar(alpha, "alpha")
    if not 0 < a < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("calibration must be a nonempty mapping of configurations to counts")
    bank = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError("configuration names must be nonempty strings")
        bank[name] = _counts(value, f"counts[{name}]")
    quantiles = {name: float(np.quantile(values, 1 - a, method="higher"))
                 for name, values in bank.items()}
    decisive = max(quantiles, key=quantiles.get)
    best = None
    for name, values in bank.items():
        k, gamma = _critical_pair(values, a)
        if best is None or (k, -gamma) > (best[0], -best[1]):
            best = (k, gamma, name)
    return {"threshold": quantiles[decisive], "alpha": a,
            "quantiles": quantiles, "decisive": decisive,
            "trials": {name: len(values) for name, values in bank.items()},
            "exact": {"threshold": best[0], "gamma": best[1], "decisive": best[2]}}


def rejection_rate(counts, threshold) -> float:
    """Fraction of supplied audit counts strictly greater than threshold."""
    values = _counts(counts, "counts")
    cutoff = _finite_scalar(threshold, "threshold")
    return float(np.mean(values > cutoff))
