# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MoE copy of ~/logprob_eval.py (27-09-2026): the same documents tokenised with the SERVED model's tokenizer
# (HORIZON_LP_TOKENIZER, default nvidia/Qwen3.6-35B-A3B-NVFP4), same output dir and --compare. Tags: moe-*.
# Long-prompt quality probe: per-token log-probabilities of REAL documents (prose, code, wiki) at 1k-32k tokens.
# Prefill of these prompts is exactly the path K3 (fp4x2) takes, so comparing two configurations token by token
# measures its quality effect directly (like perplexity), with thousands of tokens per document.
#   logprob_eval.py <tag>                 -> ~/logprob_eval/<tag>.json
#   logprob_eval.py --compare <tagA> <tagB>
# Requests are sent one at a time (deterministic prefill).
import glob, json, math, os, sys, urllib.request

OUTDIR = os.path.expanduser("~/logprob_eval")
URL = "http://localhost:8000/v1/completions"
LENGTHS = (1024, 4096, 16384, 32768)


def corpus():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("HORIZON_LP_TOKENIZER",
                                                       "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4"))
    texts = {}
    p = os.path.expanduser("$HK_WORK/corpus/prose.md")
    if os.path.exists(p):
        texts["prose"] = open(p, encoding="utf-8", errors="ignore").read()
    V = "$HK_VENV/lib/python3.12/site-packages/vllm"
    files = sorted(glob.glob(f"{V}/v1/spec_decode/*.py") + glob.glob(f"{V}/v1/worker/gpu/*.py")
                   + glob.glob(f"{V}/model_executor/layers/*.py"))
    texts["code"] = "\n\n".join(open(f).read() for f in files)
    try:
        import sys as _s; _s.path.insert(0, "$HK_DATA/llmc-pkgs")
        from datasets import load_dataset
        wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts["wiki"] = "".join(wt["text"])
    except Exception as ex:  # noqa: BLE001
        print("wikitext unavailable:", ex)
    docs = {}
    for name, t in texts.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        for L in LENGTHS:
            if len(ids) >= L:
                docs[f"{name}_{L}"] = ids[:L]
    return docs


def prompt_logprobs(ids):
    body = {"model": "qwen3.8", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    pl = r["choices"][0]["prompt_logprobs"]
    lp, top1 = [], []
    for pos, d in enumerate(pl):
        if d is None:
            continue
        actual = str(ids[pos])
        lp.append(d[actual]["logprob"])
        top1.append(min(d.items(), key=lambda kv: kv[1].get("rank", 99))[0])   # model's own top-1 at this position
    return lp, top1


def run(tag):
    os.makedirs(OUTDIR, exist_ok=True)
    docs = corpus()
    out = {}
    for name, ids in docs.items():
        lp, top1 = prompt_logprobs(ids)
        out[name] = {"lp": lp, "top1": top1}
        print(f"{tag} {name}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f}", flush=True)
    json.dump(out, open(f"{OUTDIR}/{tag}.json", "w"))


def compare(ta, tb):
    A = json.load(open(f"{OUTDIR}/{ta}.json")); B = json.load(open(f"{OUTDIR}/{tb}.json"))
    tot = {"tokens": 0, "dnll": 0.0, "absd": 0.0, "top1_same": 0}
    for name in sorted(set(A) & set(B)):
        a, b = A[name], B[name]
        n = min(len(a["lp"]), len(b["lp"]))
        nll_a = -sum(a["lp"][:n]) / n; nll_b = -sum(b["lp"][:n]) / n
        absd = sum(abs(x - y) for x, y in zip(a["lp"][:n], b["lp"][:n])) / n
        same = sum(x == y for x, y in zip(a["top1"][:n], b["top1"][:n])) / n
        print(f"{name:14s} n={n:6d}  NLL {nll_a:.4f} -> {nll_b:.4f}  (PPL x{math.exp(nll_b - nll_a):.4f})  "
              f"mean|dlogp| {absd:.4f}  top1 same {100 * same:.2f}%")
        tot["tokens"] += n; tot["dnll"] += (nll_b - nll_a) * n; tot["absd"] += absd * n; tot["top1_same"] += same * n
    T = tot["tokens"]
    print(f"ALL            n={T:6d}  dNLL {tot['dnll'] / T:+.5f} nats/token  (PPL x{math.exp(tot['dnll'] / T):.4f})  "
          f"mean|dlogp| {tot['absd'] / T:.4f}  top1 same {100 * tot['top1_same'] / T:.2f}%")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
