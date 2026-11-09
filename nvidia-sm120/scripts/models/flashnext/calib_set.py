#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Calibration set for REAP expert pruning of Qwen3.8-Flash-Next (03-10-2026). Public Hugging Face data only.

  calib_set.py --out calib.jsonl [--total 3000] [--seed 7] [--strict] [--no-decontam]

Mix (default shares of --total; every source is optional and reported if it fails to load):
  code       CommitPackFT (python/js/ts/go/rust/java/c++ commits: instruction + old file -> new file), MBPP *train*
             (task ids 601-974, never the 500 test tasks we evaluate on), OpenCodeReasoning prompts + reasoning
  tools      Glaive function-calling v2 (system with function schemas, multi-turn), SmolTalk apigen-80k
  agentic    nebius/SWE-agent-trajectories (multi-turn agent runs with tool output)
  languages  German, Bosnian, Croatian, Serbian web text (FineWeb-2 test split) + Aya instructions in those languages
  chat       UltraChat-200k (train_sft), SmolTalk (all)
  maths      GSM8K *train*
  long       CNN/DailyMail train, several articles concatenated (~12k tokens) with a summarisation request
  safety     Anthropic hh-rlhf harmless-base, the preferred ("chosen") answers: keeps the experts that carry safety behaviour in the calibration signal (the NVIDIA Open Model License forbids reducing guardrail efficacy)
Each record: {"id", "source", "messages": [...]} (or "text" for raw documents). Conversations keep the assistant turns:
calibration prefills the whole conversation, so the experts see answer-style tokens too (REAP's own calibration mix is
also agentic/coding conversations: github.com/CerebrasResearch/reap).

No customer data: this script reads only the public datasets named here; --extra-jsonl refuses paths that look like
recorded traffic, telemetry or capture data. Decontamination: 13-gram (lower-cased words) overlap against the eval
sets we report (GSM8K test, MBPP test, HumanEval, MATH-500, AIME 2025, IFEval, MMLU-Pro test); overlapping records are
dropped and counted in the manifest.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time

SHARES = {  # fraction of --total
    "commitpackft": 0.08, "mbpp_train": 0.05, "opencodereasoning": 0.12,
    "glaive_tools": 0.08, "apigen_tools": 0.05, "swe_agent": 0.10,
    "lang_de": 0.07, "lang_bs": 0.03, "lang_hr": 0.03, "lang_sr": 0.03, "aya_bcs_de": 0.04,
    "ultrachat": 0.11, "smoltalk": 0.08, "gsm8k_train": 0.07, "cnn_long": 0.04, "hh_harmless": 0.02,
}
COMMIT_LANGS = ["python", "javascript", "typescript", "go", "rust", "java", "c++"]
FORBIDDEN_LOCAL = ("traffic", "telemetry", "session", "capture")
MAX_CHARS = 32000
LONG_CHARS = 48000


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def ds():
    import datasets  # noqa: PLC0415
    return datasets


def take(it, n):
    out = []
    for x in it:
        out.append(x)
        if len(out) >= n:
            break
    return out


def clip(s, n=MAX_CHARS):
    return s if len(s) <= n else s[:n]


# ---- sources -------------------------------------------------------------------------------------------------------

def src_commitpackft(n, rng):
    per = max(1, n // len(COMMIT_LANGS))
    out = []
    for lang in COMMIT_LANGS:
        d = ds().load_dataset("json", data_files=f"hf://datasets/bigcode/commitpackft/data/{lang}/data.jsonl",
                              split="train", streaming=True)
        for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=2000), per):
            if not r.get("new_contents"):
                continue
            user = (f"{r.get('subject') or r.get('message', '')}\n\nFile `{r.get('new_file', '')}`:\n```\n"
                    f"{clip(r.get('old_contents') or '', 12000)}\n```\nApply the change and return the full file.")
            out.append({"source": f"commitpackft/{lang}", "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": "```\n" + clip(r["new_contents"], 12000) + "\n```"}]})
    return out


def src_mbpp_train(n, rng):
    d = ds().load_dataset("google-research-datasets/mbpp", "full", split="train")   # task ids 601-974
    rows = list(d)
    rng.shuffle(rows)
    return [{"source": "mbpp/train", "messages": [
        {"role": "user", "content": r["text"] + "\nYour code should pass: " + r["test_list"][0]},
        {"role": "assistant", "content": "```python\n" + r["code"].replace("\r\n", "\n") + "\n```"}]}
        for r in rows[:n]]


def src_opencodereasoning(n, rng):
    d = ds().load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True)
    out = []
    for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000), n):
        out.append({"source": "opencodereasoning", "messages": [
            {"role": "user", "content": clip(r["input"], 8000)},
            {"role": "assistant", "content": clip(r["output"], 20000)}]})
    return out


GLAIVE_TURN = re.compile(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*")


def parse_glaive(system, chat):
    msgs = [{"role": "system", "content": system.replace("SYSTEM: ", "", 1).strip()}]
    parts = GLAIVE_TURN.split(chat)
    for i in range(1, len(parts) - 1, 2):
        role, text = parts[i], parts[i + 1].replace("<|endoftext|>", "").strip()
        if not text:
            continue
        msgs.append({"role": {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}[role],
                     "content": text})
    return msgs


def src_glaive_tools(n, rng):
    d = ds().load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    out = []
    for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000), n * 2):
        if "functions" not in r["system"]:
            continue
        m = parse_glaive(r["system"], r["chat"])
        if len(m) >= 3:
            out.append({"source": "glaive_tools", "messages": m})
        if len(out) >= n:
            break
    return out


def src_apigen_tools(n, rng):
    d = ds().load_dataset("HuggingFaceTB/smoltalk", "apigen-80k", split="train", streaming=True)
    return [{"source": "smoltalk/apigen", "messages": r["messages"]}
            for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000), n)]


def src_swe_agent(n, rng):
    d = ds().load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
    out = []
    for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=500), n * 2):
        traj = r["trajectory"]
        if isinstance(traj, str):
            try:
                traj = ast.literal_eval(traj)
            except (ValueError, SyntaxError):
                continue
        msgs = []
        for t in traj:
            role = {"system": "system", "user": "user", "ai": "assistant", "assistant": "assistant"}.get(t.get("role"))
            text = t.get("text") or t.get("content") or t.get("system_prompt") or ""
            if role and text:
                msgs.append({"role": role, "content": clip(text, 6000)})
        if len(msgs) >= 4:
            out.append({"source": "swe_agent", "messages": msgs[:40]})
        if len(out) >= n:
            break
    return out


def _fineweb(lang, n, rng, name):
    d = ds().load_dataset("parquet", data_files=f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang}/test/*.parquet",
                          split="train", streaming=True)
    return [{"source": name, "text": clip(r["text"], 12000)}
            for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=2000), n) if len(r["text"]) > 400]


def src_lang_de(n, rng):
    return _fineweb("deu_Latn", n, rng, "fineweb2/de")


def src_lang_bs(n, rng):
    return _fineweb("bos_Latn", n, rng, "fineweb2/bs")


def src_lang_hr(n, rng):
    return _fineweb("hrv_Latn", n, rng, "fineweb2/hr")


def src_lang_sr(n, rng):
    return _fineweb("srp_Cyrl", n, rng, "fineweb2/sr")


def src_aya_bcs_de(n, rng):
    want = {"German", "Bosnian", "Croatian", "Serbian"}
    d = ds().load_dataset("CohereLabs/aya_dataset", split="train", streaming=True)
    out = []
    for r in d:
        if r["language"] in want:
            out.append({"source": f"aya/{r['language']}", "messages": [
                {"role": "user", "content": r["inputs"]}, {"role": "assistant", "content": r["targets"]}]})
            if len(out) >= n:
                break
    return out


def src_ultrachat(n, rng):
    d = ds().load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    return [{"source": "ultrachat", "messages": r["messages"]}
            for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000), n)]


def src_smoltalk(n, rng):
    d = ds().load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    return [{"source": f"smoltalk/{r.get('source', '?')}", "messages": r["messages"]}
            for r in take(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=10000), n)]


def src_gsm8k_train(n, rng):
    rows = list(ds().load_dataset("openai/gsm8k", "main", split="train"))
    rng.shuffle(rows)
    return [{"source": "gsm8k/train", "messages": [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"].replace("####", "The answer is")}]} for r in rows[:n]]


def src_cnn_long(n, rng):
    d = ds().load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    it = iter(d.shuffle(seed=rng.randint(0, 10**6), buffer_size=5000))
    out = []
    for _ in range(n):
        arts, size = [], 0
        while size < LONG_CHARS:
            a = next(it)["article"]
            arts.append(a)
            size += len(a)
        doc = "\n\n---\n\n".join(f"Article {i + 1}:\n{a}" for i, a in enumerate(arts))
        out.append({"source": "cnn_dailymail/long", "messages": [
            {"role": "user", "content": doc + "\n\nSummarise each article in two sentences, then list the people "
                                              "mentioned in more than one article."}]})
    return out


HH_TURN = re.compile(r"\n\n(Human|Assistant): ")


def src_hh_harmless(n, rng):
    d = ds().load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    idx = list(range(len(d)))
    rng.shuffle(idx)
    out = []
    for i in idx[: n * 2]:
        parts = HH_TURN.split("\n\n" + d[i]["chosen"].lstrip())
        msgs = [{"role": "user" if parts[k] == "Human" else "assistant", "content": parts[k + 1].strip()}
                for k in range(1, len(parts) - 1, 2) if parts[k + 1].strip()]
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
            out.append({"source": "hh_rlhf/harmless", "messages": msgs})
        if len(out) >= n:
            break
    return out


SOURCES = {k: globals()["src_" + k] for k in SHARES}


# ---- decontamination -----------------------------------------------------------------------------------------------

WORD = re.compile(r"\w+", re.UNICODE)


def ngrams(text, n=13):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_ngrams():
    """13-grams of every eval prompt we report on (and its reference answer where it is text)."""
    D = ds()
    texts = []
    loaders = [
        ("gsm8k", lambda: [r["question"] + " " + r["answer"] for r in D.load_dataset("openai/gsm8k", "main", split="test")]),
        ("mbpp", lambda: [r["text"] + " " + r["code"] for r in D.load_dataset("google-research-datasets/mbpp", "full", split="test")]),
        ("humaneval", lambda: [r["prompt"] + r["canonical_solution"] for r in D.load_dataset("openai/openai_humaneval", split="test")]),
        ("math500", lambda: [r["problem"] + " " + r["solution"] for r in D.load_dataset("HuggingFaceH4/MATH-500", split="test")]),
        ("aime25", lambda: [r["problem"] for r in D.load_dataset("math-ai/aime25", split="test")]),
        ("ifeval", lambda: [r["prompt"] for r in D.load_dataset("google/IFEval", split="train")]),
        ("mmlu_pro", lambda: [r["question"] for r in D.load_dataset("TIGER-Lab/MMLU-Pro", split="test")]),
    ]
    grams, stats = set(), {}
    for name, fn in loaders:
        try:
            t = fn()
        except Exception as e:  # noqa: BLE001
            stats[name] = f"FAILED: {type(e).__name__}: {e}"[:200]
            continue
        for x in t:
            grams |= ngrams(x)
        stats[name] = len(t)
        texts += t
    return grams, stats


def record_text(r):
    if "text" in r:
        return r["text"]
    return "\n".join(str(m.get("content") or "") for m in r["messages"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="comma list of sources (default: all)")
    ap.add_argument("--extra-jsonl", action="append", default=[], help="extra PUBLIC records (same format)")
    ap.add_argument("--no-decontam", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail if any source fails")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    names = [s for s in SHARES if not a.only or s in a.only.split(",")]
    recs, manifest = [], {"seed": a.seed, "total_requested": a.total, "sources": {}, "date": time.strftime("%d-%m-%Y")}
    for s in names:
        n = max(1, round(SHARES[s] * a.total))
        t0 = time.time()
        try:
            got = SOURCES[s](n, random.Random(rng.randint(0, 10**9)))
            manifest["sources"][s] = {"requested": n, "got": len(got), "seconds": round(time.time() - t0)}
            recs += got
            log(f"{s}: {len(got)}/{n}")
        except Exception as e:  # noqa: BLE001
            manifest["sources"][s] = {"requested": n, "got": 0, "error": f"{type(e).__name__}: {e}"[:300]}
            log(f"{s}: FAILED {type(e).__name__}: {e}")
            if a.strict:
                raise
    for p in a.extra_jsonl:
        if any(k in p.lower() for k in FORBIDDEN_LOCAL):
            raise SystemExit(f"{p}: refusing a path that looks like recorded traffic / telemetry / capture data")
        extra = [json.loads(l) for l in open(p) if l.strip()]
        recs += extra
        manifest["sources"][f"extra:{p}"] = {"got": len(extra)}
    if not a.no_decontam:
        grams, st = eval_ngrams()
        manifest["decontam_eval_sets"] = st
        kept, dropped = [], {}
        for r in recs:
            if ngrams(record_text(r)) & grams:
                dropped[r["source"]] = dropped.get(r["source"], 0) + 1
            else:
                kept.append(r)
        manifest["decontam_dropped"] = dropped
        recs = kept
    rng.shuffle(recs)
    with open(a.out, "w") as f:
        for i, r in enumerate(recs):
            r["id"] = f"calib-{i:05d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["records"] = len(recs)
    manifest["chars"] = sum(len(record_text(r)) for r in recs)
    manifest["sha256"] = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    json.dump(manifest, open(a.out + ".manifest.json", "w"), indent=1)
    log(json.dumps({k: manifest[k] for k in ("records", "chars", "sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
