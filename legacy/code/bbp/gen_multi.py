"""Run multiple sampling tasks with one loaded vLLM model.

Tasks specify output and prompt paths, sampling parameters, response counts
and seeds. Each prompt receives a disjoint seed range. A prompt-level
max_tokens value overrides the task value; large response counts are split
into seed blocks of 500."""
import argparse
import json
import os
import time

from gen_local import build_messages

DEFAULTS = dict(n=1, temperature=0.5, top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
                max_tokens=30, system="", seed=1, split="", limit=0, tag="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True, help="JSON file with a list of tasks")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_util", type=float, default=0.88)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--chat_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--extra_llm", default="{}")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tasks = [dict(DEFAULTS, **t) for t in json.load(open(a.tasks, encoding="utf-8"))]
    todo = [t for t in tasks if not os.path.exists(t["out"])]
    print(json.dumps({"tasks": len(tasks), "todo": len(todo)}), flush=True)
    if not todo:
        return
    llm_kw = dict(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util, seed=1,
                  trust_remote_code=True, enable_prefix_caching=True)
    if a.quantization:
        llm_kw["quantization"] = a.quantization
    llm_kw.update(json.loads(a.extra_llm))
    t0 = time.time()
    llm = LLM(**llm_kw)
    tok = llm.get_tokenizer()
    ck = json.loads(a.chat_kwargs)
    for t in todo:
        allp = [json.loads(l) for l in open(t["prompts"], encoding="utf-8")]
        gidx = {p["pid"]: i for i, p in enumerate(allp)}
        prompts = [p for p in allp if not t["split"] or p.get("split") == t["split"]]
        if t["limit"]:
            prompts = prompts[: t["limit"]]
        if t["temperature"] == 0:  # vLLM allows only n=1 for greedy: send n separate single requests
            blocks = [(b, 1) for b in range(t["n"])]
        elif t["n"] >= 1000:
            blocks = [(b, min(500, t["n"] - b)) for b in range(0, t["n"], 500)]
        else:
            blocks = [(0, t["n"])]
        inputs, sps, owners = [], [], []
        for p in prompts:
            ids = tok.apply_chat_template(build_messages(p["text"], t["system"]), add_generation_prompt=True,
                                          tokenize=True, **ck)
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            for bi, (k0, nb) in enumerate(blocks):
                inputs.append(TokensPrompt(prompt_token_ids=list(ids)))
                sps.append(SamplingParams(n=nb, temperature=t["temperature"], top_p=t["top_p"], top_k=t["top_k"],
                                          min_p=t["min_p"], presence_penalty=t["presence_penalty"],
                                          repetition_penalty=t["repetition_penalty"],
                                          max_tokens=int(p.get("max_tokens", t["max_tokens"])),
                                          seed=(t["seed"] * 100 + bi) * 10_000_000 + gidx[p["pid"]] * 1000))
                owners.append((p["pid"], k0))
        t1 = time.time()
        outs = llm.generate(inputs, sps, use_tqdm=False)
        cfg = {k: t[k] for k in DEFAULTS} | dict(model=os.path.basename(a.model.rstrip("/")),
                                                 quantization=a.quantization, chat_kwargs=ck, prompts=t["prompts"])
        os.makedirs(os.path.dirname(os.path.abspath(t["out"])), exist_ok=True)
        tmp = t["out"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": cfg, "n_prompts": len(prompts), "gen_s": round(time.time() - t1, 1)}) + "\n")
            for (pid, k0), o in zip(owners, outs):
                for j, c in enumerate(o.outputs):
                    f.write(json.dumps({"pid": pid, "k": k0 + j, "text": c.text, "n_tokens": len(c.token_ids),
                                        "finish_reason": c.finish_reason}, ensure_ascii=False) + "\n")
        os.replace(tmp, t["out"])
        print(json.dumps({"out": t["out"], "prompts": len(prompts), "n": t["n"], "s": round(time.time() - t1, 1)}),
              flush=True)
    print(json.dumps({"done": len(todo), "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
