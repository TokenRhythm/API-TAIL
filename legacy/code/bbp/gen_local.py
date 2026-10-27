"""Generate local reference, legitimate-configuration or substitute responses.

Uses the vLLM offline engine and disjoint seed ranges for each prompt.
Output JSONL rows contain pid, k, text, n_tokens, finish_reason and cfg."""
import argparse
import json
import os
import time


def build_messages(text, system):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": text})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=0.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--system", default="")
    ap.add_argument("--chat_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--domains", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/N: keep prompts[i::N]")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.88)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--kv_cache_dtype", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra_llm", default="{}")
    ap.add_argument("--glm_dense", type=int, default=0)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = [json.loads(l) for l in open(a.prompts, encoding="utf-8")]
    gidx = {p["pid"]: i for i, p in enumerate(prompts)}  # global index -> disjoint per-prompt seed ranges
    if a.domains:
        prompts = [p for p in prompts if p["domain"] in a.domains.split(",")]
    if a.split:
        prompts = [p for p in prompts if p["split"] == a.split]
    if a.limit:
        prompts = prompts[: a.limit]
    if a.shard:
        i, n_sh = map(int, a.shard.split("/"))
        prompts = prompts[i::n_sh]

    llm_kw = dict(model=a.model, tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
                  gpu_memory_utilization=a.gpu_util, seed=a.seed, trust_remote_code=True,
                  enable_prefix_caching=True, kv_cache_dtype=a.kv_cache_dtype, dtype=a.dtype)
    if a.quantization:
        llm_kw["quantization"] = a.quantization
    llm_kw.update(json.loads(a.extra_llm))
    if a.glm_dense:
        import glm_dense
        assert a.max_model_len <= 2048, "dense equivalence of DSA needs prompt+response < index_topk"
        llm_kw.update(glm_dense.llm_kwargs())
    t0 = time.time()
    llm = LLM(**llm_kw)
    tok = llm.get_tokenizer()
    ck = json.loads(a.chat_kwargs)
    inputs = []
    for p in prompts:
        ids = tok.apply_chat_template(build_messages(p["text"], a.system), add_generation_prompt=True,
                                      tokenize=True, **ck)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        inputs.append(TokensPrompt(prompt_token_ids=list(ids)))
    # vLLM seeds child j of an n>1 request with seed+j, so a shared per-request seed would give sample j of
    # every prompt the same random stream (cross-prompt coupling). Give each prompt a disjoint seed range.
    assert a.n < 1000 and len(gidx) < 10_000
    sps = [SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, min_p=a.min_p,
                          presence_penalty=a.presence_penalty, repetition_penalty=a.repetition_penalty,
                          max_tokens=a.max_tokens, seed=a.seed * 10_000_000 + gidx[p["pid"]] * 1000)
           for p in prompts]
    t1 = time.time()
    outs = llm.generate(inputs, sps, use_tqdm=False)
    t2 = time.time()
    cfg = dict(model=os.path.basename(a.model.rstrip("/")), temperature=a.temperature, top_p=a.top_p,
               top_k=a.top_k, min_p=a.min_p, presence_penalty=a.presence_penalty,
               repetition_penalty=a.repetition_penalty, max_tokens=a.max_tokens, system=a.system,
               chat_kwargs=ck, quantization=a.quantization, kv_cache_dtype=a.kv_cache_dtype, tag=a.tag)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ntok = 0
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": cfg, "n_prompts": len(prompts), "load_s": round(t1 - t0, 1),
                            "gen_s": round(t2 - t1, 1)}) + "\n")
        for p, o in zip(prompts, outs):
            for k, c in enumerate(o.outputs):
                ntok += len(c.token_ids)
                f.write(json.dumps({"pid": p["pid"], "k": k, "text": c.text, "n_tokens": len(c.token_ids),
                                    "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n")
    print(json.dumps({"out": a.out, "prompts": len(prompts), "samples": len(prompts) * a.n, "tokens": ntok,
                      "load_s": round(t1 - t0, 1), "gen_s": round(t2 - t1, 1)}))


if __name__ == "__main__":
    main()
