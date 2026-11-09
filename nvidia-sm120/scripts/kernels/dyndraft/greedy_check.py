# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Temperature-0 output check for draft-count changes (27-09-2026).
#   greedy_check.py TAG            -> $HK_WORK/dyndraft/greedy-TAG.json
#   greedy_check.py --compare A B
# 10 prompts, 300 tokens each, sent (a) one at a time (batch 1: a dynamic schedule uses its 1-user draft count)
# and (b) 3 copies of all 10 at once (30 in flight: the high-load draft count). Saves token ids, so a comparison
# shows the first differing position. Drafts never decide a token (the main model verifies each one), so arms
# differ only where the main model's own arithmetic differs between batch shapes; the comparison shows how often.
import json, os, sys, threading, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
OUT = os.path.expanduser("$HK_WORK/dyndraft")
P = ["Explain how a suspension bridge carries load.", "Write a Python function that merges two sorted lists.",
     "What is 17 * 23 + 144 / 12? Show steps.", "Summarise the causes of the First World War in 5 bullet points.",
     "Translate to German: The weather is nice today and we are going to the lake.",
     "Write a SQL query for the top 3 customers by revenue per country.", "Give me a haiku about GPUs.",
     "What are the differences between TCP and UDP?",
     "Solve: a train leaves at 14:05 and travels 180 km at 72 km/h. When does it arrive?",
     "Write a short product description for a reusable water bottle."]


def gen(p):
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": 300, "temperature": 0,
         "logprobs": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(),
                                                                {"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]
    return {"text": c["message"]["content"], "toks": [t["token"] for t in c["logprobs"]["content"]],
            "lps": [t["logprob"] for t in c["logprobs"]["content"]]}


def run(tag):
    seq = [gen(p) for p in P]
    conc = [None] * (3 * len(P))
    def w(i):
        conc[i] = gen(P[i % len(P)])
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(conc))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    os.makedirs(OUT, exist_ok=True)
    json.dump({"seq": seq, "conc": conc}, open(f"{OUT}/greedy-{tag}.json", "w"))
    print("saved", tag)


def cmp_lists(a, b):
    same = 0; firsts = []; maxd = 0.0
    for x, y in zip(a, b):
        n = min(len(x["toks"]), len(y["toks"]))
        k = next((i for i in range(n) if x["toks"][i] != y["toks"][i]), None)
        if k is None and len(x["toks"]) == len(y["toks"]):
            same += 1
            maxd = max([maxd] + [abs(p - q) for p, q in zip(x["lps"], y["lps"])])
        else:
            firsts.append(k if k is not None else n)
    return {"identical": f"{same}/{len(a)}", "first_diff_positions": firsts, "max_abs_logprob_diff_identical": round(maxd, 5)}


def compare(a, b):
    A = json.load(open(f"{OUT}/greedy-{a}.json")); B = json.load(open(f"{OUT}/greedy-{b}.json"))
    res = {"pair": [a, b], "sequential": cmp_lists(A["seq"], B["seq"]), "concurrent": cmp_lists(A["conc"], B["conc"]),
           f"{a}_seq_vs_conc": cmp_lists(A["seq"] * 3, A["conc"]), f"{b}_seq_vs_conc": cmp_lists(B["seq"] * 3, B["conc"])}
    print(json.dumps(res))


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
