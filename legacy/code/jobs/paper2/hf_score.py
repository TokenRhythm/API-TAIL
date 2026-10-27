"""Score responses with deterministic, batch-size-one teacher forcing.

For each response and temperature, accumulate full-vocabulary excess
surprisal S and variance V. Also retain nucleus-restricted moments, kept-token
counts and top-4096 nucleus overflow counts. Responses are retokenized under
the claimed tokenizer and truncated to the model-length limit.

Output caches contain response keys, the temperature grid, scorer metadata
and moment arrays. The optional --tf_dir output stores token log probabilities,
ranks and top-k values for response-score baselines."""
import argparse
import json
import os
import pickle
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402
import torch  # noqa: E402

TG = np.round(np.arange(0.30, 1.81, 0.01), 2)
NT = len(TG)
NK = 4096
PN = 0.95


def stats(lg, y):
    """lg (n, V) float32 logits at the response positions, y (n,) targets -> per-T sums over the positions."""
    lp = lg - lg.max(1, keepdim=True).values
    lyp = lp.gather(1, y[:, None])[:, 0]
    out20 = int(((lp > lyp[:, None]).sum(1) >= 20).sum())
    n = lp.shape[0]
    f0 = torch.empty(n, NT, dtype=torch.float64, device=lp.device)
    f1, f2 = torch.empty_like(f0), torch.empty_like(f0)
    for i, t in enumerate(TG):
        e = torch.exp(lp / float(t))
        el = e * lp
        f0[:, i] = e.sum(1, dtype=torch.float64)
        f1[:, i] = el.sum(1, dtype=torch.float64)
        f2[:, i] = (el * lp).sum(1, dtype=torch.float64)
    T = torch.tensor(TG, dtype=torch.float64, device=lp.device)[None]
    Eb, Eb2 = f1 / f0, f2 / f0
    lZ = torch.log(f0)
    H = lZ - Eb / T
    var = torch.clamp((Eb2 - Eb * Eb) / (T * T), min=1e-6)
    SF = (lZ - lyp.double()[:, None] / T - H).sum(0)
    VF = var.sum(0)
    tv, ti = torch.topk(lp, NK, dim=1)
    hit = ti == y[:, None]
    inK = hit.any(1)
    rk = torch.argmax(hit.int(), 1)
    SN = torch.zeros(NT, dtype=torch.float64, device=lp.device)
    VN, CN, OV = torch.zeros_like(SN), torch.zeros_like(SN), torch.zeros_like(SN)
    for i, t in enumerate(TG):
        q = torch.exp(tv / float(t)) / f0[:, i: i + 1].float()
        cs = torch.cumsum(q, 1)
        keep = (cs - q) < PN
        keep[:, 0] = True
        mass = (q * keep).sum(1).clamp(min=1e-30)
        qn = torch.where(keep, q / mass[:, None], torch.zeros_like(q))
        lqn = torch.where(qn > 0, torch.log(qn.clamp(min=1e-30)), torch.zeros_like(qn))
        Hn = -(qn * lqn).sum(1, dtype=torch.float64)
        Vn = torch.clamp((qn * lqn * lqn).sum(1, dtype=torch.float64) - Hn * Hn, min=1e-6)
        yk = inK & keep.gather(1, rk[:, None])[:, 0]
        syn = torch.log(f0[:, i]) - lyp.double() / float(t) + torch.log(mass.double())
        SN[i] = torch.where(yk, syn - Hn, torch.zeros_like(Hn)).sum()
        VN[i] = torch.where(yk, Vn, torch.zeros_like(Vn)).sum()
        CN[i] = yk.double().sum()
        OV[i] = (cs[:, -1] < PN).double().sum()
    return SF, VF, SN, VN, CN, OV, out20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keys", default="", help="json {name: [[pid, k], ...]}: score exactly these keys in this order")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chat_kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--system", default="")
    ap.add_argument("--tf_dir", default="", help="also write score_tf.py-format top-K files here")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tf_only", type=int, default=0, help="1: only the top-K files (e.g. RUT references), no svfull")
    a = ap.parse_args()
    assert not a.tf_only or a.tf_dir, "--tf_only needs --tf_dir"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    scorer = {"impl": "hf", "transformers": transformers.__version__, "torch": torch.__version__,
              "deterministic": True, "batch": 1, "model": a.model, "system": a.system, "chat_kwargs": a.chat_kwargs}
    ck = json.loads(a.chat_kwargs)
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(a.prompts, encoding="utf-8")}
    want = json.load(open(a.keys, encoding="utf-8")) if a.keys else {}
    os.makedirs(a.out_dir, exist_ok=True)
    for path, name in zip(a.inputs, a.names):
        outp = os.path.join(a.out_dir, f"{name}.svfull.pkl")
        tfb = os.path.join(a.tf_dir, os.path.basename(path).replace(".jsonl", "")) if a.tf_dir else ""
        if (a.tf_only or os.path.exists(outp)) and (not tfb or os.path.exists(tfb + ".index.jsonl")):
            print("exists", outp, flush=True)
            continue
        t0 = time.time()
        rows = {}
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if "_meta" in r or not isinstance(r.get("text"), str) or not r["text"]:
                continue
            rows[(r["pid"], r.get("k", 0))] = r["text"]
        keys = [tuple(k) for k in want[name]] if name in want else sorted(rows)
        if a.limit:
            keys = keys[: a.limit]
        R = len(keys)
        fl = {f: np.zeros((R, NT)) for f in ("SF0", "VF0", "SNF0", "VNF0")}
        fl.update({f: np.zeros((R, NT), np.int32) for f in ("nNF0", "overNF")})
        ntok, out20 = np.zeros(R, np.int64), np.zeros(R, np.int64)
        tf_rows, tf_lp, tf_rk, tf_tl, tf_ti = [], [], [], [], []
        for j, (pid, k) in enumerate(keys):
            msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": P[pid]["text"]}]
            pi = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **ck)
            pi = list(pi["input_ids"] if hasattr(pi, "keys") else pi)
            rid = tok.encode(rows[(pid, k)], add_special_tokens=False)
            if len(pi) + len(rid) + 1 > a.max_model_len:
                rid = rid[: max(0, a.max_model_len - len(pi) - 1)]
            if tfb:
                tf_rows.append({"pid": pid, "k": k, "n": len(rid)})
            if not rid:
                continue
            ids = torch.tensor([pi + rid], device="cuda:0")
            with torch.no_grad():
                lg = model(ids).logits[0, len(pi) - 1: len(pi) + len(rid) - 1].float()
            y = torch.tensor(rid, device="cuda:0")
            if tfb:
                ls = torch.log_softmax(lg, dim=1)
                ly = ls.gather(1, y[:, None])[:, 0]
                tf_lp.append(ly.cpu().numpy().astype(np.float32))
                tf_rk.append(((ls > ly[:, None]).sum(1) + 1).cpu().numpy().astype(np.int32))
                tv_, ti_ = torch.topk(ls, a.topk, dim=1)
                tf_tl.append(tv_.cpu().numpy().astype(np.float16))
                tf_ti.append(ti_.cpu().numpy().astype(np.int32))
            if a.tf_only:
                continue
            SF, VF, SN, VN, CN, OV, o20 = stats(lg, y)
            fl["SF0"][j], fl["VF0"][j], fl["SNF0"][j], fl["VNF0"][j] = (SF.cpu().numpy(), VF.cpu().numpy(),
                                                                          SN.cpu().numpy(), VN.cpu().numpy())
            fl["nNF0"][j], fl["overNF"][j] = CN.cpu().numpy().astype(np.int32), OV.cpu().numpy().astype(np.int32)
            ntok[j], out20[j] = len(rid), o20
            if (j + 1) % 250 == 0:
                print(json.dumps({"file": name, "done": j + 1, "of": R, "sec": round(time.time() - t0, 1)}), flush=True)
        out = {"keys": keys, "NK": NK, "ntok": ntok, "out20": out20, "grid": {"T": TG.tolist()}, "scorer": scorer,
               "fields": fl}
        if not a.tf_only:
            pickle.dump(out, open(outp + ".tmp", "wb"), protocol=4)
            os.replace(outp + ".tmp", outp)
        if tfb:
            os.makedirs(a.tf_dir, exist_ok=True)
            offsets = np.zeros(len(tf_rows) + 1, np.int64)
            for j, r in enumerate(tf_rows):
                offsets[j + 1] = offsets[j] + r["n"]
            np.savez_compressed(tfb + ".npz", offsets=offsets, lp=np.concatenate(tf_lp), rk=np.concatenate(tf_rk),
                                topk_lp=np.concatenate(tf_tl), topk_id=np.concatenate(tf_ti))
            with open(tfb + ".index.jsonl.tmp", "w", encoding="utf-8") as f:
                for r in tf_rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(tfb + ".index.jsonl.tmp", tfb + ".index.jsonl")
        print(json.dumps({"file": outp, "responses": R, "sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
