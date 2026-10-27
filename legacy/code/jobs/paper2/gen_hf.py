"""Generate local responses with Transformers and an explicit sampling order.

The logits processor applies repetition penalty, presence penalty,
temperature, top-k, top-p and min-p, followed by multinomial sampling.
Built-in Transformers warpers are disabled. Prompts are left-padded and
batched by length. Output rows retain both decoded text and generated IDs.

Usage: gen_hf.py --model MODEL --tasks tasks.json [--batch 64]"""
import argparse
import json
import os
import time

import torch

DEF = dict(n=1, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
           max_tokens=30, system="", seed=1, split="", limit=0)


class Sampler:
    """vLLM-order sampling processor for one batch; prompt token sets are fixed per row (pads excluded)."""

    def __init__(self, t, prompt_mask, L0):
        self.t, self.L0 = t, L0
        self.pmask = prompt_mask                       # (B, V) bool: token occurs in the row's prompt

    def __call__(self, input_ids, scores):
        t = self.t
        s = scores.float()
        B, V = s.shape
        out = input_ids[:, self.L0:]
        omask = torch.zeros(B, V, dtype=torch.bool, device=s.device)
        if out.shape[1]:
            omask.scatter_(1, out.clamp(max=V - 1), True)
        if t["repetition_penalty"] != 1.0:
            m = omask | self.pmask
            r = t["repetition_penalty"]
            s = torch.where(m, torch.where(s > 0, s / r, s * r), s)
        if t["presence_penalty"] != 0.0:
            s = s - t["presence_penalty"] * omask.float()
        if t["temperature"] > 0:
            s = s / t["temperature"]
        if t["top_k"] and t["top_k"] > 0:
            kth = torch.topk(s, min(t["top_k"], V), dim=1).values[:, -1:]
            s = s.masked_fill(s < kth, float("-inf"))
        if t["top_p"] < 1.0:
            ss, si = torch.sort(s, descending=True, dim=1)
            p = torch.softmax(ss, dim=1)
            cum = torch.cumsum(p, dim=1)
            drop = (cum - p) >= t["top_p"]           # vLLM: a token whose preceding mass reaches top_p is cut
            drop[:, 0] = False
            s = s.scatter(1, si, ss.masked_fill(drop, float("-inf")))
        if t["min_p"] > 0:
            p = torch.softmax(s, dim=1)
            s = s.masked_fill(p < t["min_p"] * p.max(1, keepdim=True).values, float("-inf"))
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tok_budget", type=int, default=120000, help="max (prompt + new) tokens per batch")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--fp8w", type=int, default=0, help="1: weight-only FP8 (e4m3, per-output-channel scale) rounding of "
                                                         "every linear layer of the language model (FP8-member emulation)")
    ap.add_argument("--chat_kwargs", default='{"enable_thinking": false}')
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    tasks = [dict(DEF, **t) for t in json.load(open(a.tasks, encoding="utf-8"))]
    todo = [t for t in tasks if not os.path.exists(t["out"])]
    print(json.dumps({"tasks": len(tasks), "todo": len(todo)}), flush=True)
    if not todo:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    if a.fp8w:
        n_q = 0
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear) and "vision" not in name and "audio" not in name \
                        and mod is not model.get_output_embeddings():
                    w = mod.weight.data.float()
                    sc = (w.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
                    mod.weight.data = ((w / sc).to(torch.float8_e4m3fn).float() * sc).to(torch.bfloat16)
                    n_q += 1
        print(json.dumps({"fp8w_linear_layers": n_q}), flush=True)
    ck = json.loads(a.chat_kwargs)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    V = model.get_output_embeddings().weight.shape[0]
    for t in todo:
        t0 = time.time()
        P = [json.loads(l) for l in open(t["prompts"], encoding="utf-8")]
        if t["split"]:
            P = [p for p in P if p.get("split") == t["split"]]
        if t["limit"]:
            P = P[: t["limit"]]
        msgs = lambda p: ([{"role": "system", "content": t["system"]}] if t["system"] else []) + \
            [{"role": "user", "content": p["text"]}]
        enc, too_long = [], []
        for i, p in enumerate(P):
            ids = tok.apply_chat_template(msgs(p), add_generation_prompt=True, tokenize=True, **ck)
            ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
            if len(ids) + t["max_tokens"] > a.max_model_len:      # as a vLLM engine with max_model_len would refuse
                too_long.append(p["pid"])
                continue
            enc.append((i, p, ids))
        jobs = [(i, p, ids, k) for i, p, ids in enc for k in range(t["n"])]
        jobs.sort(key=lambda x: len(x[2]))
        gc = GenerationConfig(do_sample=t["temperature"] > 0, temperature=1.0, top_k=0, top_p=1.0,
                              max_new_tokens=t["max_tokens"], eos_token_id=sorted(eos),
                              pad_token_id=tok.pad_token_id)              # T = 0: greedy, as the vLLM generator
        rows = []
        batches, b0 = [], 0
        while b0 < len(jobs):                                         # length-sorted, token-budgeted batches
            bs = 1
            while bs < a.batch and b0 + bs < len(jobs) and \
                    (bs + 1) * (len(jobs[b0 + bs][2]) + t["max_tokens"]) <= a.tok_budget:
                bs += 1
            batches.append((b0, jobs[b0: b0 + bs]))
            b0 += bs
        for b0, bj in batches:
            L0 = max(len(x[2]) for x in bj)
            inp = torch.full((len(bj), L0), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(bj), L0), dtype=torch.long)
            pm = torch.zeros((len(bj), V), dtype=torch.bool)
            for r, (_, _, ids, _) in enumerate(bj):
                inp[r, L0 - len(ids):] = torch.tensor(ids)
                att[r, L0 - len(ids):] = 1
                pm[r, torch.tensor(ids)] = True
            torch.manual_seed(t["seed"] * 100003 + b0)
            with torch.no_grad():
                g = model.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), generation_config=gc,
                                   logits_processor=LogitsProcessorList([Sampler(t, pm.cuda(), L0)]))
            for r, (i, p, ids, k) in enumerate(bj):
                new = g[r, L0:].tolist()
                cut = next((j for j, x in enumerate(new) if x in eos), None)
                fin = "stop" if cut is not None else "length"
                new = new[:cut] if cut is not None else [x for x in new if x != tok.pad_token_id]
                rows.append({"pid": p["pid"], "k": k, "text": tok.decode(new, skip_special_tokens=True), "ids": new,
                             "n_tokens": len(new), "finish_reason": fin})
            print(json.dumps({"out": os.path.basename(t["out"]), "done": len(rows), "of": len(jobs),
                              "sec": round(time.time() - t0, 1)}), flush=True)
        rows.sort(key=lambda r: (r["pid"], r["k"]))
        os.makedirs(os.path.dirname(t["out"]), exist_ok=True)
        with open(t["out"] + ".tmp", "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": {k: t[k] for k in DEF}, "engine": "hf-generate", "fp8w": a.fp8w,
                                "skipped_too_long": too_long}) + "\n")
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(t["out"] + ".tmp", t["out"])
        print(json.dumps({"task": t["out"], "rows": len(rows), "sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
