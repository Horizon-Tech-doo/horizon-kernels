# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Decode-path fingerprint (27-09-2026, ops track): greedy generations with the chosen token's log-prob, to compare a
# kernel that only runs in DECODE steps (e.g. the GDN spec-verify core, HORIZON_GDN) between arms. logprob_eval.py
# measures prefill only and cannot see such a kernel.
#   decode_check.py TAG                 -> $HK_WORK/ops/dc/TAG.json
#   decode_check.py --compare TAGA TAGB
# Parts: seq (10 short chat prompts, one at a time, 400 tokens), long (continuations after ~4k / ~16k / ~32k-token
# real prose, one at a time, 400 tokens), conc (8 prompts at once, 300 tokens; batch composition can vary run to run).
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

U = "http://localhost:8001"
D = os.path.expanduser("$HK_WORK/ops/dc")
P = ["Explain how a suspension bridge carries load.", "Write a Python function that merges two sorted lists.",
     "What is 17 * 23 + 144 / 12? Show steps.", "Summarise the causes of the First World War in 5 bullet points.",
     "Translate to German: The weather is nice today and we are going to the lake.",
     "Write a SQL query for the top 3 customers by revenue per country.", "Give me a haiku about GPUs.",
     "What are the differences between TCP and UDP?",
     "Solve: a train leaves at 14:05 and travels 180 km at 72 km/h. When does it arrive?",
     "Write a short product description for a reusable water bottle."]


def post(path, body, t=1800):
    r = urllib.request.urlopen(urllib.request.Request(U + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=t)
    return json.load(r)


def chat(p, n):
    r = post("/v1/chat/completions", {"model": "qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                                      "temperature": 0, "logprobs": True, "top_logprobs": 0,
                                      "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]["logprobs"]["content"]
    return {"tokens": [x["token"] for x in c], "lp": [x["logprob"] for x in c]}


def cont(text, n):
    r = post("/v1/completions", {"model": "qwen3.8", "prompt": text, "max_tokens": n, "temperature": 0, "logprobs": 0})
    lp = r["choices"][0]["logprobs"]
    return {"tokens": lp["tokens"], "lp": lp["token_logprobs"], "prompt_tokens": r["usage"]["prompt_tokens"]}


def run(tag):
    os.makedirs(D, exist_ok=True)
    out = {"tag": tag, "time": time.strftime("%d-%m-%Y %H:%M")}
    chat("Hello", 8)
    out["seq"] = [chat(p, 400) for p in P]
    prose = open(os.path.expanduser("$HK_WORK/corpus/prose.md"), encoding="utf-8", errors="ignore").read()
    out["long"] = [cont(prose[:chars], 400) for chars in (16000, 64000, 128000)]
    with ThreadPoolExecutor(8) as ex:
        out["conc"] = list(ex.map(lambda p: chat(p, 300), P[:8]))
    json.dump(out, open(f"{D}/{tag}.json", "w"))
    print(json.dumps({"decode_check": tag, "seq_tokens": sum(len(x["tokens"]) for x in out["seq"]),
                      "long_prompt_tokens": [x["prompt_tokens"] for x in out["long"]]}), flush=True)


def compare_records(A, B):
    """Compare the full expected fixture; never silently zip truncated results."""
    import math
    res = {"compare": [A.get("tag"), B.get("tag")]}
    for part, expected in (("seq", 10), ("long", 3), ("conc", 8)):
        if len(A.get(part, [])) != expected or len(B.get(part, [])) != expected:
            raise ValueError(f"{part}: expected {expected} responses in both arms")
        same = 0; div = []; details = []; maxd = 0.0; n = 0; lpeq = 0; missing = 0
        for idx, (x, y) in enumerate(zip(A[part], B[part])):
            for record in (x, y):
                if not record['tokens'] or len(record['tokens']) != len(record['lp']):
                    raise ValueError(f"{part}[{idx}]: empty or incomplete token/logprob record")
                if any(v is not None and not math.isfinite(v) for v in record['lp']):
                    raise ValueError(f"{part}[{idx}]: non-finite log probability")
            if part == 'long' and x.get('prompt_tokens') != y.get('prompt_tokens'):
                raise ValueError(f"{part}[{idx}]: mismatched prompt length")
            if x['tokens'] == y['tokens']:
                same += 1
            else:
                k = next((i for i, (u, v) in enumerate(zip(x['tokens'], y['tokens'])) if u != v),
                         min(len(x['tokens']), len(y['tokens'])))
                div.append(k)
                details.append(dict(prompt_index=idx, token_index=k,
                                    tokens_a=len(x['tokens']), tokens_b=len(y['tokens'])))
            for u, v, lu, lv in zip(x['tokens'], y['tokens'], x['lp'], y['lp']):
                if u != v:
                    break
                if lu is None or lv is None:
                    missing += 1
                else:
                    n += 1
                    maxd = max(maxd, abs(lu-lv)); lpeq += lu == lv
        res[part] = dict(identical=f"{same}/{expected}", first_divergence=div,
                         divergence_details=details, max_abs_dlogprob_before_divergence=maxd,
                         logprob_bit_equal=f"{lpeq}/{n}", missing_logprob_pairs=missing)
    return res


def compare(a, b):
    with open(f"{D}/{a}.json") as f:
        A = json.load(f)
    with open(f"{D}/{b}.json") as f:
        B = json.load(f)
    print(json.dumps(compare_records(A, B), indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
