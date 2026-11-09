#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""summarize_arms.py RESULTS_DIR ARM [ARM ...]: one Markdown summary of the rented-box arms (03-10-2026).

Reads, per arm directory: steps.jsonl (exit codes), the JSON summary lines our harnesses print into eval.log
(HumanEval, maths, tool smoke, long-context gates, needles, LongBench, extra checks, GSM8K-200 / MMLU-Pro-200),
lm-eval's results_*.json, extra-<arm>.json (131k-document NLL), load.jsonl (capacity_load.py), and the pruning
reports (prune-p35.json / prune-p50.json). The FIRST arm is the reference for the deltas (A = full model).
Interpretation and the gate decision are a separate, human step (flashnext-plan (not in this release), gates)."""
import glob
import json
import os
import sys

LM_KEYS = {"gsm8k_cot_llama": ("exact_match,flexible-extract", "exact_match,strict-match"),
           "mbpp_instruct": ("pass_at_1,extract_code", "pass_at_1,none", "pass@1,none"),
           "ifeval": ("prompt_level_strict_acc,none", "inst_level_strict_acc,none")}


def jlines(path):
    out = []
    if os.path.exists(path):
        for line in open(path, errors="ignore"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def arm_metrics(d, arm):
    m = {}
    for r in jlines(os.path.join(d, "steps.jsonl")):
        m.setdefault("steps_failed", [])
        if r.get("rc"):
            m["steps_failed"].append(f"{r['step']}:{r['rc']}")
    for r in jlines(os.path.join(d, "eval.log")):
        if "humaneval_pass@1" in r:
            m["HumanEval"] = r["humaneval_pass@1"]
        if "set" in r and "acc" in r:
            m[f"{r['set']}:{r['mode']}"] = r["acc"]
        if "safety" in r and isinstance(r["safety"], dict):
            for k, v in r["safety"].items():
                m[f"safety {k}"] = v
        if "tool_smoke_passed" in r:
            m["tool smoke"] = f"{r['tool_smoke_passed']}/{r['of']}"
        if "long_context" in r:
            m["long-context gate (A,B,C)"] = r["long_context"]
            m["long-context gates"] = r.get("gates")
        if "accuracy" in r and "n" in r and "tag" in r:
            m["LongBench v2 subset"] = r["accuracy"]
        if any(k.startswith("needle_") for k in r):
            m["needles"] = {k: v for k, v in r.items() if k.startswith("needle_")}
        for k in ("gsm8k", "mmlu_pro", "GSM8K", "MMLU-Pro"):
            if k in r and isinstance(r[k], (int, float)):
                m[f"{k}-200"] = r[k]
    for f in glob.glob(os.path.join(d, "lm", "nothink", "**", "results_*.json"), recursive=True):
        res = json.load(open(f)).get("results", {})
        for task, keys in LM_KEYS.items():
            for k in keys:
                if task in res and k in res[task]:
                    m[f"{task} ({k.split(',')[0]})"] = round(res[task][k], 4)
                    break
    ex = os.path.join(d, f"extra-{arm}.json")
    if os.path.exists(ex):
        e = json.load(open(ex))
        lp = e.get("long_lp") or {}
        if "nll" in lp:
            m["131k doc NLL"] = round(lp["nll"], 4)
    for r in jlines(os.path.join(d, "load.jsonl")):
        if r.get("test") == "level":
            m[f"load {r['context'] // 1024}k x{r['level']}"] = (f"{r['total_tok_s']} tok/s, user {r['user_tok_s_p50']}, "
                                                                f"TTFT p50/p95 {r['ttft_p50']}/{r['ttft_p95']} s, KV "
                                                                f"{r['peak_kv_usage']}, preempt {r['preemptions']}, "
                                                                f"{'clean' if r['clean'] else 'NOT clean'}")
        if r.get("test") == "probe_result":
            m[f"max clean concurrency @{r['context'] // 1024}k"] = r["max_clean_concurrency"]
        if r.get("test") == "vllm_estimates":
            for k, v in r.items():
                if k not in ("tag", "ts", "test"):
                    m[f"vLLM estimate {k}"] = v
    return m


def main():
    root, arms = sys.argv[1], sys.argv[2:]
    data = {a: arm_metrics(os.path.join(root, a), a) for a in arms}
    keys = []
    for a in arms:
        for k in data[a]:
            if k not in keys:
                keys.append(k)
    print(f"# Flash-Next rented-box summary ({', '.join(arms)})\n")
    for t in ("p35", "p50"):
        p = os.path.join(root, f"prune-{t}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            low = r.get("low_retention_layers", [])
            print(f"- prune {t}: {r['num_experts_src']} -> {r['num_experts_out']} experts per block, "
                  f"{r.get('gib_saved', '?')} GiB saved, MTP {r.get('mtp_policy')}, "
                  f"{len(low)} block(s) under the retention warning")
    print("\n| metric | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for k in keys:
        row = []
        for a in arms:
            v = data[a].get(k, "")
            row.append(json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        print(f"| {k} | " + " | ".join(row) + " |")
    ref = arms[0]
    print(f"\nDeltas vs {ref} (numeric metrics):\n")
    for a in arms[1:]:
        ds = []
        for k in keys:
            x, y = data[ref].get(k), data[a].get(k)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                ds.append(f"{k} {y - x:+.4f}")
        print(f"- {a}: " + "; ".join(ds))


if __name__ == "__main__":
    main()
