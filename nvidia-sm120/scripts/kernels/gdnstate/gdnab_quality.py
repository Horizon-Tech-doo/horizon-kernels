# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Decode-path quality probes for the GDN-state A/B (29-09-2026). The risk of a 16-bit recurrent state is in DECODE (the
state is rounded after every step of 1-4 tokens); prefill rounds it once per 8k chunk, so prompt log-probs alone cannot
see it (docs/gdn-state-design.md section 5). Two probes against the running endpoint (:8000):

  gdnab_quality.py TAG decode   long-decode fingerprint: 6 long prompts (prose / code, 8k and 32k tokens + a short
                                "continue" instruction), greedy, 1024 generated tokens, thinking off, logprobs of the
                                chosen tokens -> $HK_WORK/gdnstate/ab/decode-TAG.json
  gdnab_quality.py TAG math     long GENERATION: AIME 2025 (30) + MATH-500 first 100, thinking on (reasoning_effort
                                high, T 0.6, top_p 0.95, seed 0, 32k tokens; eval_math_chat.py's prompt and grading),
                                32 in flight -> $HK_WORK/gdnstate/ab/math-TAG.jsonl + one summary line
  gdnab_quality.py --compare-decode TAG_A TAG_B    per prompt: identical-prefix length, token agreement, mean chosen
                                logprob (report only: different numerics diverge by design)
"""
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.expanduser("$HK_WORK/gdnstate/ab")
URL = "http://localhost:8000/v1"
MODEL_DIR = "$HK_DATA/models/Qwen3.8-27B-NVFP4"


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def prompts():
    import glob
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prose = open(os.path.expanduser("$HK_WORK/corpus/prose.md"), encoding="utf-8", errors="ignore").read()
    V = "$HK_VENV/lib/python3.12/site-packages/vllm"
    code = "\n\n".join(open(f).read() for f in sorted(glob.glob(f"{V}/v1/worker/gpu/*.py")))
    out = {}
    for name, text in (("prose", prose), ("code", code)):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for n in (8192, 32768):
            for part in (0, 1) if n == 8192 else (0,):
                body = tok.decode(ids[part * n: part * n + n])
                instr = ("\n\nContinue the text above in the same style for as long as you can."
                         if name == "prose" else "\n\n# Continue this code base with the next function:\n")
                out[f"{name}_{n}_{part}"] = body + instr
    return out


def decode(tag):
    res = {}
    for name, p in prompts().items():
        r = post("/completions", {"model": "qwen3.8", "prompt": p, "max_tokens": 1024, "temperature": 0,
                                  "logprobs": 1, "seed": 0})
        c = r["choices"][0]
        lp = c.get("logprobs") or {}
        res[name] = {"tokens": lp.get("tokens", []), "token_logprobs": lp.get("token_logprobs", []),
                     "finish": c.get("finish_reason"), "prompt_tokens": r["usage"]["prompt_tokens"]}
        n = len(res[name]["tokens"])
        mlp = sum(x for x in res[name]["token_logprobs"] if x is not None) / max(1, n)
        print(json.dumps({"tag": tag, "probe": "decode", "prompt": name, "gen_tokens": n,
                          "mean_chosen_logprob": round(mlp, 4)}), flush=True)
    json.dump(res, open(f"{OUT}/decode-{tag}.json", "w"))


def math(tag):
    sys.path.insert(0, "$HK_DATA/eval-pkgs"); sys.path.insert(1, "$HK_DATA/llmc-pkgs")
    from datasets import load_dataset
    from math_verify import parse, verify
    items = [("aime25", r["problem"], str(r["answer"])) for r in load_dataset("math-ai/aime25", split="test")]
    items += [("math500", r["problem"], r["answer"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")][:100]

    def ask(it):
        body = {"model": "qwen3.8", "messages": [{"role": "user", "content": it[1] + "\n\nPlease reason step by step, "
                "and put your final answer within \\boxed{}."}], "temperature": 0.6, "top_p": 0.95,
                "max_tokens": 32768, "reasoning_effort": "high", "seed": 0}
        m = post("/chat/completions", body)
        return (m["choices"][0]["message"].get("content") or ""), m["choices"][0].get("finish_reason"), \
            m["usage"]["completion_tokens"]

    def boxed(t):
        i = t.rfind("\\boxed{")
        if i < 0:
            return None
        j, d = i + 7, 1
        while j < len(t) and d:
            d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
        return t[i + 7:j - 1]

    with ThreadPoolExecutor(32) as pool:
        outs = list(pool.map(ask, items))
    summ = {"tag": tag, "probe": "math"}
    with open(f"{OUT}/math-{tag}.jsonl", "w") as f:
        for (s, q, gold), (ans, fin, ntok) in zip(items, outs):
            b = boxed(ans)
            ok = bool(b is not None and verify(parse("$" + gold + "$"), parse("$" + b + "$")))
            f.write(json.dumps({"set": s, "gold": gold, "pred": b, "ok": ok, "finish": fin, "gen_tokens": ntok}) + "\n")
            summ.setdefault(s, {"ok": 0, "n": 0, "cut": 0, "tokens": 0})
            summ[s]["ok"] += ok; summ[s]["n"] += 1; summ[s]["cut"] += fin == "length"; summ[s]["tokens"] += ntok
    print(json.dumps(summ), flush=True)


def compare_decode(ta, tb):
    A = json.load(open(f"{OUT}/decode-{ta}.json")); B = json.load(open(f"{OUT}/decode-{tb}.json"))
    for name in sorted(set(A) & set(B)):
        a, b = A[name]["tokens"], B[name]["tokens"]
        n = min(len(a), len(b)); pre = next((i for i in range(n) if a[i] != b[i]), n)
        la = [x for x in A[name]["token_logprobs"] if x is not None]
        lb = [x for x in B[name]["token_logprobs"] if x is not None]
        print(json.dumps({"compare": [ta, tb], "prompt": name, "identical_prefix": pre, "of": n,
                          "mean_lp_a": round(sum(la) / max(1, len(la)), 4),
                          "mean_lp_b": round(sum(lb) / max(1, len(lb)), 4)}))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "--compare-decode":
        compare_decode(sys.argv[2], sys.argv[3])
    elif sys.argv[2] == "decode":
        decode(sys.argv[1])
    elif sys.argv[2] == "math":
        math(sys.argv[1])
    else:
        sys.exit(__doc__)
