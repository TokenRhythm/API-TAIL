"""Transformers scoring with float32 exponentials and products.

Moment sums and final moment arithmetic use float64."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import time
from pathlib import Path
import numpy as np
from .io import ScoreBank, prompt_map, read_jsonl, save_scores, file_sha256
from .validation import integer


def deterministic_torch():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch


def load_model(model_path, device="cuda:0", dtype="bfloat16", local_files_only=False):
    torch = deterministic_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu --dtype float32 for a small model")
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt, device_map=device,
                                                local_files_only=local_files_only)
    model.eval()
    return torch, tok, model


def chat_ids(tok, prompt, system="", chat_kwargs=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **(chat_kwargs or {}))
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    if not ids:
        raise ValueError("The chat template produced no prompt tokens")
    return list(ids)


def moments_torch(logits, targets, grid, nucleus=False, token_chunk=64):
    import torch
    if type(token_chunk) is not int or token_chunk <= 0 or logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("Expected nonempty (tokens, vocabulary) logits and positive chunk size")
    if targets.dtype not in (torch.int32, torch.int64) or targets.shape != (len(logits),) or not torch.isfinite(logits).all():
        raise ValueError("Invalid targets or nonfinite model logits")
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("Target token outside vocabulary")
    targets = targets.to(dtype=torch.int64)
    tg = np.asarray(grid, dtype=np.float64)
    if tg.ndim != 1 or len(tg) == 0 or not np.isfinite(tg).all() or (tg <= 0).any() or (np.diff(tg) <= 0).any():
        raise ValueError("Invalid temperature grid")
    names = ["S", "V"] + (["S_nucleus", "V_nucleus", "n_nucleus", "nucleus_overflow"] if nucleus else [])
    sums = {n: torch.zeros(len(tg), dtype=torch.float64, device=logits.device) for n in names}
    nll_sum = entropy_sum = logrank_sum = 0.0
    out20 = 0
    for start in range(0, len(logits), token_chunk):
        lg = logits[start:start + token_chunk].float()
        y = targets[start:start + token_chunk]
        lp = lg - lg.max(1, keepdim=True).values
        ly = lp.gather(1, y[:, None])[:, 0]
        ranks = (lp > ly[:, None]).sum(1) + 1
        out20 += int((ranks > 20).sum().item())
        logrank_sum += float(torch.log(ranks.double()).sum().item())
        lp1 = torch.log_softmax(lg, dim=1)
        nll_sum += float((-lp1.gather(1, y[:, None])[:, 0]).double().sum().item())
        entropy_sum += float((-(lp1.exp() * lp1)).sum(dtype=torch.float64).item())
        if nucleus:
            tv, ti = torch.topk(lp, min(4096, lp.shape[1]), dim=1)
            hit = ti == y[:, None]
            ink, rk = hit.any(1), hit.int().argmax(1)
        for i, temp in enumerate(tg):
            e = torch.exp(lp / float(temp))
            el = e * lp
            f0 = e.sum(1, dtype=torch.float64)
            f1 = el.sum(1, dtype=torch.float64)
            f2 = (el * lp).sum(1, dtype=torch.float64)
            mean, second = f1 / f0, f2 / f0
            lz = torch.log(f0)
            h = lz - mean / temp
            var = ((second - mean * mean) / (temp * temp)).clamp(min=1e-6)
            sums["S"][i] += (lz - ly.double() / temp - h).sum()
            sums["V"][i] += var.sum()
            if nucleus:
                q = torch.exp(tv / float(temp)) / f0[:, None].float()
                cs = q.cumsum(1)
                keep = cs - q < .95
                keep[:, 0] = True
                mass = (q * keep).sum(1).clamp(min=1e-30)
                qn = torch.where(keep, q / mass[:, None], torch.zeros_like(q))
                lqn = torch.where(qn > 0, qn.clamp(min=1e-30).log(), torch.zeros_like(qn))
                hn = -(qn * lqn).sum(1, dtype=torch.float64)
                vn = ((qn * lqn * lqn).sum(1, dtype=torch.float64) - hn * hn).clamp(min=1e-6)
                valid = ink & keep.gather(1, rk[:, None])[:, 0]
                syn = lz - ly.double() / temp + mass.double().log()
                sums["S_nucleus"][i] += torch.where(valid, syn - hn, torch.zeros_like(hn)).sum()
                sums["V_nucleus"][i] += torch.where(valid, vn, torch.zeros_like(vn)).sum()
                sums["n_nucleus"][i] += valid.double().sum()
                sums["nucleus_overflow"][i] += (cs[:, -1] < .95).double().sum()
    result = {k: v.cpu().numpy() for k, v in sums.items()}
    result.update(nll=nll_sum / len(logits), logrank=logrank_sum / len(logits),
                  entropy=entropy_sum / len(logits), out20=out20)
    return result


def score(model_path, prompts, responses, output, device="cuda:0", dtype="bfloat16",
          max_length=4096, system="", chat_kwargs=None, nucleus=False,
          token_chunk=64, limit=None, local_files_only=False):
    from .core import temperature_grid
    if Path(output).exists():
        raise FileExistsError(output)
    integer(max_length, "max_length", 1)
    integer(token_chunk, "token_chunk", 1)
    if chat_kwargs is not None and not isinstance(chat_kwargs, dict):
        raise ValueError("chat_kwargs must be an object")
    P = prompt_map(prompts)
    rows = read_jsonl(responses)
    keys, accepted = set(), []
    for row in rows:
        pid, k = str(row["pid"]), integer(row.get("k", 0), "response k", 0)
        if (pid, k) in keys:
            raise ValueError(f"Duplicate response key {(pid, k)}")
        keys.add((pid, k))
        if pid not in P or not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError(f"Response {(pid, k)} is missing its prompt or text")
        accepted.append((pid, k, row["text"]))
    if limit is not None:
        if limit <= 0: raise ValueError("limit must be positive")
        accepted = accepted[:limit]
    if not accepted:
        raise ValueError("No responses")
    torch, tok, model = load_model(model_path, device, dtype, local_files_only)
    import transformers
    start = time.monotonic()
    grid = temperature_grid()
    stats, lengths = [], []
    for pid, k, text in accepted:
        pi = chat_ids(tok, P[pid]["text"], system, chat_kwargs)
        rid = tok.encode(text, add_special_tokens=False)
        # Reject overlength responses so each score covers the complete text.
        if not rid or len(pi) + len(rid) > max_length:
            raise ValueError(f"Response {(pid, k)} is empty after tokenization or exceeds max_length")
        ids = torch.tensor([pi + rid], device=device)
        with torch.inference_mode():
            lg = model(ids).logits[0, len(pi)-1:len(pi)+len(rid)-1].float()
            y = torch.tensor(rid, device=device)
            stats.append(moments_torch(lg, y, grid, nucleus, token_chunk))
        lengths.append(len(rid))
    names = set(stats[0]) - {"S", "V"}
    extra = {name: np.asarray([r[name] for r in stats]) for name in names}
    config_path = Path(model_path) / "config.json"
    meta = dict(schema="apitail-scores-v1", model_id=Path(model_path).name if Path(model_path).exists() else model_path,
                model_config_sha256=file_sha256(config_path) if config_path.is_file() else None,
                scoring="full-vocabulary-fp32-exp-fp64-sums-v1", dtype=dtype, device_type=device.split(":")[0],
                system=system, chat_kwargs=chat_kwargs or {}, max_length=max_length,
                transformers=transformers.__version__, torch=torch.__version__,
                responses_sha256=file_sha256(responses), prompts_sha256=file_sha256(prompts),
                seconds=time.monotonic()-start, nucleus_top_limit=4096 if nucleus else None,
                deterministic_algorithms=True, batch_size=1, token_chunk=token_chunk)
    bank = ScoreBank(np.stack([r["S"] for r in stats]), np.stack([r["V"] for r in stats]), grid,
                     np.asarray([r[0] for r in accepted]), np.asarray([r[1] for r in accepted]),
                     np.asarray(lengths), meta, extra)
    save_scores(output, bank)
    return dict(responses=len(accepted), tokens=sum(lengths), seconds=meta["seconds"], output=str(output))
