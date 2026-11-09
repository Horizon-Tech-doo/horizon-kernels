# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MTP plan, stage 1 (v2): self-distillation data. The served model answers a mixed prompt set:
#   English chat (ultrachat), maths (GSM8K), code (Magicoder)  ~75%
#   multilingual chat (Aya dataset: German, French, Italian, Bosnian/Croatian/Serbian, Slovenian)  ~25%
# 25% of prompts are answered with thinking on. Output: one JSON line per answer with the language, so draft-vocabulary
# coverage can be checked per language. Resumable: prompts already in the output file are skipped.
#   mtp_gen_v2.py [N_TOTAL=4000] [CONCURRENCY=48]
import hashlib, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "$HK_DATA/llmc-pkgs")
from datasets import load_dataset

OUT = "$HK_DATA/models/mtp-data/gen_v2.jsonl"
URL = "http://localhost:8000/v1/chat/completions"
N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 48
AYA_LANGS = {"German", "French", "Italian", "Bosnian", "Croatian", "Serbian", "Slovenian", "Standard German"}
random.seed(0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def log(*a):
    print(time.strftime("%d-%m-%Y %H:%M:%S"), *a, flush=True)


n_multi = N_TOTAL // 4
n_chat, n_math = int(N_TOTAL * 0.40), int(N_TOTAL * 0.15)
n_code = N_TOTAL - n_multi - n_chat - n_math
P = []
uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[2000:8000]")   # 0-2000 were NVFP4 calibration
P += [("en", "chat", m["messages"][0]["content"]) for m in uc.select(range(n_chat))]
P += [("en", "math", x["question"]) for x in load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1).select(range(n_math))]
P += [("en", "code", x["problem"]) for x in load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train").shuffle(seed=1).select(range(n_code))]
try:
    aya = load_dataset("CohereForAI/aya_dataset", split="train")
    aya = aya.filter(lambda r: r["language"] in AYA_LANGS and len(r["inputs"]) > 20).shuffle(seed=2)
    P += [(r["language"], "multi", r["inputs"]) for r in aya.select(range(min(n_multi, len(aya))))]
    log("aya prompts:", min(n_multi, len(aya)), "from", sorted({r for r in aya["language"]}))
except Exception as ex:  # noqa: BLE001
    log("Aya dataset unavailable, English only:", type(ex).__name__, ex)
random.shuffle(P)

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["pid"])
        except Exception:  # noqa: BLE001
            pass
todo = [(i, p) for i, p in enumerate(P) if hashlib.sha1(p[2].encode()).hexdigest()[:16] not in done]
log(f"{len(P)} prompts, {len(done)} already done, {len(todo)} to generate, concurrency {CONC}")


def gen(item):
    i, (lang, kind, prompt) = item
    think = (i % 4 == 0)
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1536,
            "temperature": 0.7, "top_p": 0.9, "chat_template_kwargs": {"enable_thinking": think}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                                    {"Content-Type": "application/json"}), timeout=900))
        m = r["choices"][0]["message"]
        return {"pid": hashlib.sha1(prompt.encode()).hexdigest()[:16], "lang": lang, "kind": kind, "prompt": prompt,
                "think": think, "reasoning": m.get("reasoning") or m.get("reasoning_content") or "",
                "content": m.get("content") or "", "completion_tokens": r["usage"]["completion_tokens"]}
    except Exception:  # noqa: BLE001
        return None


t = time.time(); n = 0; toks = 0
with open(OUT, "a") as f, ThreadPoolExecutor(CONC) as ex:
    for res in ex.map(gen, todo):
        if res and res["content"]:
            f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush(); n += 1; toks += res["completion_tokens"]
            if n % 200 == 0:
                log(f"{n}/{len(todo)} done, {toks} tokens, {toks / (time.time() - t):.0f} tok/s")
log("DONE", n, "answers,", toks, "tokens in", round((time.time() - t) / 60, 1), "min")
