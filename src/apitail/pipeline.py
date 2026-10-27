from pathlib import Path
import json
import numpy as np
from .core import batch_statistic, batch_counts, calibrate_counts, rejection_rate
from .io import load_scores, compatible, file_sha256, write_json
from .validation import integer, number


def _method_arrays(bank, method):
    if method == "nucleus":
        if not {"S_nucleus", "V_nucleus"} <= set(bank.extra):
            raise ValueError("Nucleus scores missing: score with --nucleus")
        return bank.extra["S_nucleus"], bank.extra["V_nucleus"], "align"
    return bank.S, bank.V, method


def calibrate(config_path, output):
    path = Path(config_path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("members", "budget", "trials"):
        if key not in cfg:
            raise ValueError(f"Calibration config missing {key}")
    if not cfg["members"]:
        raise ValueError("Calibration pool is empty")
    if any(type(cfg[k]) is not int or cfg[k] <= 0 for k in ("budget", "trials")):
        raise ValueError("budget and trials must be positive integers")
    n, trials = cfg["budget"], cfg["trials"]
    method, cutoff = cfg.get("method", "align"), float(cfg.get("z_cutoff", 3.0))
    seed = integer(cfg.get("seed", 20261025), "seed", 0)
    banks, hashes, counts = [], {}, {}
    ids = set()
    for i, (name, rel) in enumerate(sorted(cfg["members"].items())):
        src = (path.parent / rel).resolve()
        bank = load_scores(src)
        banks.append(bank)
        compatible(banks)
        rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
        draw = bank.draws(n, trials, rng)
        S, V, estimator = _method_arrays(bank, method)
        counts[name], _ = batch_counts(S, V, draw, bank.grid, method=estimator, z_cutoff=cutoff)
        ids.update(bank.prompt_ids)
        hashes[name] = file_sha256(src)
    rule = calibrate_counts(counts, alpha=float(cfg.get("alpha", 0.05)))
    rule.update(schema="apitail-calibration-v1", budget=n, method=method, z_cutoff=cutoff,
                seed=seed, grid=banks[0].grid.tolist(), score_semantics={k: banks[0].metadata.get(k)
                for k in ("model_id", "model_config_sha256", "dtype", "scoring", "system", "chat_kwargs", "max_length")},
                calibration_prompt_ids=sorted(ids), source_sha256=hashes,
                sampling="distinct prompts without replacement; one uniform response per prompt")
    write_json(output, rule)
    return rule


def _check_rule(bank, rule, allow_overlap=False):
    if rule.get("schema") != "apitail-calibration-v1":
        raise ValueError("Unsupported calibration artifact")
    if not np.array_equal(bank.grid, rule["grid"]):
        raise ValueError("Calibration and audit grids differ")
    for k, value in rule.get("score_semantics", {}).items():
        if bank.metadata.get(k) != value:
            raise ValueError(f"Calibration and audit scoring differ on {k}")
    overlap = set(bank.prompt_ids) & set(rule.get("calibration_prompt_ids", []))
    if overlap and not allow_overlap:
        raise ValueError(f"{len(overlap)} audit prompt ids overlap calibration; use disjoint prompts")


def audit(scores, calibration, output):
    bank = load_scores(scores)
    rule = json.loads(Path(calibration).read_text(encoding="utf-8"))
    _check_rule(bank, rule)
    if len(bank.pid) != rule["budget"] or len(bank.prompt_ids) != len(bank.pid):
        raise ValueError("An audit needs exactly budget responses, one per distinct prompt")
    S, V, estimator = _method_arrays(bank, rule["method"])
    result = batch_statistic(S, V, bank.grid, method=estimator, z_cutoff=rule["z_cutoff"])
    result.update(threshold=rule["threshold"], reject=bool(result["count"] > rule["threshold"]),
                  rule="strict integer K > threshold", calibration_sha256=file_sha256(calibration),
                  score_sha256=file_sha256(scores), score_method=rule["method"])
    write_json(output, result)
    return result


def evaluate(scores, calibration, output, trials=1000, seed=1234, substitute=None, rho=0.1):
    integer(trials, "trials", 1)
    integer(seed, "seed", 0)
    number(rho, "rho", 0, 1)
    bank = load_scores(scores)
    rule = json.loads(Path(calibration).read_text(encoding="utf-8"))
    _check_rule(bank, rule)
    method, cutoff, n = rule["method"], rule["z_cutoff"], rule["budget"]
    S, V, estimator = _method_arrays(bank, method)
    rng = np.random.default_rng(seed)
    if substitute is None:
        draw = bank.draws(n, trials, rng)
        kind = "honest_endpoint_false_alarm"
    else:
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        sub = load_scores(substitute)
        compatible([bank, sub])
        _check_rule(sub, rule)
        if bank.prompt_ids != sub.prompt_ids:
            raise ValueError("Base and substitute must cover the same prompt ids")
        groups_a, groups_b = {}, {}
        for i, pid in enumerate(bank.pid): groups_a.setdefault(str(pid), []).append(i)
        for i, pid in enumerate(sub.pid): groups_b.setdefault(str(pid), []).append(i + len(bank.pid))
        ids = bank.prompt_ids
        if n > len(ids): raise ValueError("Insufficient distinct prompts for budget")
        draw = np.empty((trials, n), np.int64)
        for i in range(trials):
            chosen = rng.choice(len(ids), n, replace=False)
            replace = rng.random(n) < rho
            draw[i] = [rng.choice((groups_b if replace[j] else groups_a)[ids[k]])
                       for j, k in enumerate(chosen)]
        s2, v2, _ = _method_arrays(sub, method)
        S, V = np.concatenate([S, s2]), np.concatenate([V, v2])
        kind = "simulated_partial_substitution_power"
    counts, temps = batch_counts(S, V, draw, bank.grid, method=estimator, z_cutoff=cutoff)
    result = dict(kind=kind, rate=rejection_rate(counts, rule["threshold"]),
                  trials=trials, budget=n, seed=seed, threshold=rule["threshold"],
                  mean_temperature=float(np.mean(temps)), counts=counts.tolist(),
                  fitted_temperatures=temps.tolist(), calibration_sha256=file_sha256(calibration),
                  source_sha256=file_sha256(scores))
    if substitute is not None:
        result.update(rho=rho, substitute_sha256=file_sha256(substitute))
    write_json(output, result)
    return result
