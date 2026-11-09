# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# invest_eval.py TAG DOCSET NAMES [--ruler] (fp8fix, 27-09-2026): one vLLM arm of the long-context investigation.
#   DOCSET: q38 | moe ($HK_WORK/fp8fix/invest_docs_<set>.json from invest_docs.py); NAMES: comma list or "all16"
#   --ruler: RULER-style synthetic recall at 16k and 32k (multi-key needle, variable tracking, 2-hop QA; 4 each)
# Per-token prompt log-probs (same request as logprob_eval.py) -> ~/logprob_eval/inv-TAG.json
import json, os, random, re, sys, time, urllib.request
F = os.path.expanduser("$HK_WORK/fp8fix")
OUT = os.path.expanduser("~/logprob_eval")
BASE = "http://localhost:8000"
tag, dset, names = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(f"{F}/invest_docs_{dset}.json"))
if names == "all16":
    names = ",".join(sorted(k for k in docs if k.endswith("_16384")))


def post(path, body, timeout=1800):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=timeout))


def prompt_lp(ids):
    r = post("/v1/completions", {"model": "qwen3.8", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
    lp = []
    for pos, d in enumerate(r["choices"][0]["prompt_logprobs"]):
        if d is not None:
            lp.append(d[str(ids[pos])]["logprob"])
    return lp


out = {"_meta": {"tag": tag, "docset": dset, "time": time.strftime("%d-%m-%Y %H:%M")}}
for n in names.split(","):
    if not n:
        continue
    t0 = time.time()
    lp = prompt_lp(docs[n]["ids"])
    out[n] = {"lp": lp}
    print(f"inv-{tag} {n}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open(f"{OUT}/inv-{tag}.json", "w"))

# ---------------------------------------------------------------- RULER-style synthetic recall
if "--ruler" in sys.argv:
    hay = [p for p in re.split(r"\n\s*\n", open(os.path.expanduser("$HK_WORK/fp8fix/md/haystack.md"),
                                                     encoding="utf-8", errors="ignore").read()) if len(p) > 200]
    CPT = 3.0   # chars per token of this haystack (Qwen3.8 tokenizer, measured 391,686 / 130,522)

    def haystack(rng, chars):
        s, parts = 0, []
        while s < chars:
            p = hay[rng.randrange(len(hay))]; parts.append(p); s += len(p) + 2
        return parts

    def place(rng, parts, items):
        for it in items:
            parts.insert(rng.randrange(len(parts) + 1), it)
        return "\n\n".join(parts)

    def ask(ctx, q):
        r = post("/v1/chat/completions", {"model": "qwen3.8", "messages": [{"role": "user", "content": ctx + "\n\n" + q}],
                                          "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        return r["choices"][0]["message"].get("content") or "", r["usage"]["prompt_tokens"]

    WORDS = "amber falcon harbor quartz violet meadow ember glacier saddle orchard lantern thunder compass willow".split()
    CITIES = "Sarajevo Mostar Tuzla Zenica Bihac Banja-Luka Trebinje Livno Konjic Travnik Jajce Goražde".split()
    res = {}
    for L in (16384, 32768):
        chars = int((L - 400) * CPT)
        for task in ("mk_niah", "vt", "qa2"):
            scores = []
            for i in range(4):
                rng = random.Random(f"{task}-{L}-{i}")
                parts = haystack(rng, chars)
                if task == "mk_niah":
                    keys = rng.sample(WORDS, 4); vals = [str(rng.randint(1000000, 9999999)) for _ in keys]
                    items = [f"One of the special magic numbers for {k}-{i} is: {v}." for k, v in zip(keys, vals)]
                    j = rng.randrange(4)
                    a, pt = ask(place(rng, parts, items), f"What is the special magic number for {keys[j]}-{i} mentioned in the text above? Answer with the number only.")
                    scores.append(1.0 if vals[j] in a else 0.0)
                elif task == "vt":
                    v0 = str(rng.randint(10000, 99999)); d0 = str(rng.randint(10000, 99999))
                    names_ = [f"X{rng.randint(100, 999)}{c}" for c in "ABCDE"]
                    dist = [f"Y{rng.randint(100, 999)}{c}" for c in "ABCDE"]
                    items = [f"VAR {names_[0]} = {v0}"] + [f"VAR {names_[k]} = VAR {names_[k - 1]}" for k in range(1, 5)]
                    items += [f"VAR {dist[0]} = {d0}"] + [f"VAR {dist[k]} = VAR {dist[k - 1]}" for k in range(1, 5)]
                    ctx = "\n\n".join(parts)
                    pos = sorted(rng.randrange(len(ctx)) for _ in range(len(items)))  # chains keep their order
                    order = items[:5]
                    for k, it in enumerate(items[5:]):
                        order.insert(rng.randrange(len(order) + 1), it)
                    chunks, last = [], 0
                    for p_, it in zip(pos, order):
                        chunks += [ctx[last:p_], "\n" + it + ".\n"]; last = p_
                    chunks.append(ctx[last:])
                    a, pt = ask("".join(chunks), f"Find all variables that are assigned the value {v0} in the text above, directly or through other variables. Answer with the variable names only, separated by commas.")
                    scores.append(sum(n_ in a for n_ in names_) / 5 - 0.2 * sum(d in a for d in dist))
                else:
                    people = [f"{rng.choice(['Amir', 'Lejla', 'Tarik', 'Selma', 'Kenan', 'Ajla', 'Mirza', 'Emina'])} {rng.choice(WORDS).title()}-{k}" for k in range(8)]
                    depts = [f"{w.title()} unit" for w in rng.sample(WORDS, 8)]
                    cities = rng.sample(CITIES, 8)
                    items = [f"Record: {p_} works in the {d}." for p_, d in zip(people, depts)]
                    items += [f"Record: the {d} is located in {c}." for d, c in zip(depts, cities)]
                    j = rng.randrange(8)
                    a, pt = ask(place(rng, parts, items), f"Using the records in the text above: in which city does {people[j]} work? Answer with the city only.")
                    scores.append(1.0 if cities[j].lower() in a.lower() else 0.0)
            res[f"{task}_{L // 1000}k"] = {"score": round(sum(scores) / len(scores), 3), "all": scores, "prompt_tokens": pt}
            print(f"inv-{tag} RULER {task} {L}: {res[f'{task}_{L // 1000}k']}", flush=True)
    out["_ruler"] = res
    json.dump(out, open(f"{OUT}/inv-{tag}.json", "w"))
