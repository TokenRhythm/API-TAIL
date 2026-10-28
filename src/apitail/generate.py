"""Local generation with one prompt per batch.

Sampling applies repetition penalty, presence penalty, temperature, top-k,
top-p, and min-p in that order. Each response receives a separate seed."""
import json
from pathlib import Path
import numpy as np
from .io import prompt_map, file_sha256
from .hf import load_model, chat_ids
from .validation import integer, number, reject_credentials


def sample_logits(input_ids, scores, prompt_ids, prompt_length, settings):
    import torch
    s = scores.float()
    vocab = s.shape[1]
    output = input_ids[:, prompt_length:]
    seen_out = torch.zeros_like(s, dtype=torch.bool)
    if output.numel():
        seen_out.scatter_(1, output, True)
    seen_prompt = torch.zeros_like(s, dtype=torch.bool)
    seen_prompt[:, prompt_ids] = True
    rep = float(settings.get("repetition_penalty", 1.0))
    if rep != 1:
        s = torch.where(seen_out | seen_prompt, torch.where(s > 0, s / rep, s * rep), s)
    s = s - float(settings.get("presence_penalty", 0.0)) * seen_out.float()
    temp = float(settings.get("temperature", 1.0))
    if temp > 0: s = s / temp
    k = int(settings.get("top_k", 0))
    if k > 0:
        boundary = torch.topk(s, min(k, vocab), dim=1).values[:, -1:]
        s = s.masked_fill(s < boundary, float("-inf"))
    top_p = float(settings.get("top_p", 1.0))
    if top_p < 1:
        values, order = s.sort(descending=True, dim=1)
        probs = values.softmax(1)
        drop = probs.cumsum(1) - probs >= top_p
        drop[:, 0] = False
        s = s.scatter(1, order, values.masked_fill(drop, float("-inf")))
    min_p = float(settings.get("min_p", 0.0))
    if min_p > 0:
        probs = s.softmax(1)
        s = s.masked_fill(probs < min_p * probs.max(1, keepdim=True).values, float("-inf"))
    return s


def generate(model_path, prompts, config_path, output, device="cuda:0", dtype="bfloat16",
             chat_kwargs=None, local_files_only=False, limit=None):
    from transformers import GenerationConfig, LogitsProcessorList
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Generation config must be an object")
    reject_credentials(cfg)
    fp8_weight_rounding = cfg.get("fp8_weight_rounding", False)
    if not isinstance(fp8_weight_rounding, bool):
        raise ValueError("fp8_weight_rounding must be a boolean")
    settings = cfg.get("sampling", {})
    if not isinstance(settings, dict):
        raise ValueError("sampling must be an object")
    temperature = number(settings.get("temperature", 1.0), "temperature", 0)
    number(settings.get("top_p", 1.0), "top_p", 1e-300, 1)
    number(settings.get("repetition_penalty", 1.0), "repetition_penalty", 1e-300)
    number(settings.get("presence_penalty", 0.0), "presence_penalty")
    number(settings.get("min_p", 0.0), "min_p", 0, 1)
    integer(settings.get("top_k", 0), "top_k", 0)
    integer(settings.get("max_tokens", 30), "max_tokens", 1)
    integer(cfg.get("seed", 1), "seed", 0)
    integer(cfg.get("max_length", 4096), "max_length", 1)
    n = integer(cfg.get("responses_per_prompt", 1), "responses_per_prompt", 1)
    P = list(prompt_map(prompts).items())
    if limit is not None:
        P = P[:integer(limit, "limit", 1)]
    dst = Path(output)
    partial = dst.with_suffix(dst.suffix + ".partial")
    if dst.exists() or partial.exists(): raise FileExistsError(output)
    torch, tok, model = load_model(model_path, device, dtype, local_files_only)
    if fp8_weight_rounding:
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and "vision" not in name and "audio" not in name and module is not model.get_output_embeddings():
                    w = module.weight.float()
                    scale = (w.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
                    module.weight.copy_(((w / scale).to(torch.float8_e4m3fn).float() * scale).to(module.weight.dtype))
    eos = model.generation_config.eos_token_id
    eos_list = eos if isinstance(eos, list) else ([] if eos is None else [eos])
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos_list[0] if eos_list else 0)
    generation = GenerationConfig(do_sample=temperature > 0, temperature=1.0, top_k=0, top_p=1.0,
        max_new_tokens=int(settings.get("max_tokens", 30)), eos_token_id=eos, pad_token_id=pad)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(partial, "x", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"config": cfg, "batch_size": 1, "prompts_sha256": file_sha256(prompts)}}) + "\n")
        for i, (pid, prompt) in enumerate(P):
            pi = chat_ids(tok, prompt["text"], cfg.get("system", ""), chat_kwargs)
            if len(pi) + generation.max_new_tokens > int(cfg.get("max_length", 4096)):
                raise ValueError(f"Prompt {pid} exceeds configured context length")
            inp = torch.tensor([pi], device=device)
            class Processor:
                def __call__(self, input_ids, scores):
                    return sample_logits(input_ids, scores, pi, len(pi), settings)
            for k in range(n):
                torch.manual_seed(int(cfg.get("seed", 1))*100003 + i*n+k)
                with torch.inference_mode():
                    out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                        generation_config=generation, logits_processor=LogitsProcessorList([Processor()]))
                tokens = out[0, len(pi):].tolist()
                cut = next((j for j, x in enumerate(tokens) if x in eos_list), None)
                if cut is not None: tokens = tokens[:cut]
                text = tok.decode(tokens, skip_special_tokens=True)
                if not text:
                    raise ValueError(f"Empty completion for {(pid, k)}; partial file retained")
                f.write(json.dumps(dict(pid=pid, k=k, text=text, ids=tokens, n_tokens=len(tokens),
                                       finish_reason="stop" if cut is not None else "length"), ensure_ascii=False) + "\n")
                f.flush()
                written += 1
    partial.replace(dst)
    return {"responses": written, "output": str(dst)}
