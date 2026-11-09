# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""paired_speed.py TAG [--levels 1,8,32,64] [--rounds 2] [--tries 4]   (30-09-2026)
Fixed-token paired load ($HK_WORK/ops/candidates/paired-load/bench_load_paired.py, imported unchanged: same prompts,
seed and forced 600 output tokens in every arm) at each level, counting only CLEAN rounds (traffic_guard_passed: no
request other than ours was running). On a shared server, a round (or warm-up) that other requests trip is
retried, up to --tries times per needed round, with a pause. -> $HK_WORK/kvfp4/paired-TAG.json
{"levels": {"32": {"clean_tok_s": [...], "ttft_median_s": [...], "dirty": n}}, "complete": bool}"""
import argparse
import json
import os
import sys
import time

PL = os.path.expanduser("$HK_WORK/ops/candidates/paired-load")
sys.path.insert(0, PL)
os.chdir(PL)
import bench_load_paired as B  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--levels", default="1,8,32,64")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--tries", type=int, default=4)
ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
ap.add_argument("--metrics-url", default="http://localhost:8000/metrics")
a = ap.parse_args()
out = {"tag": a.tag, "levels": {}, "rounds": a.rounds, "tokens": 600, "seed": 29}
for n in [int(x) for x in a.levels.split(",")]:
    rec = {"clean_tok_s": [], "ttft_median_s": [], "dirty": 0, "errors": []}
    budget = a.rounds + a.tries
    warm = False
    while budget > 0 and len(rec["clean_tok_s"]) < a.rounds:
        budget -= 1
        try:
            if not warm:
                w = B.run_level(a.url, a.metrics_url, "qwen3.8", n, 32, 29)
                if not w["traffic_guard_passed"]:
                    rec["dirty"] += 1
                    time.sleep(15)
                    continue
                warm = True
            r = B.run_level(a.url, a.metrics_url, "qwen3.8", n, 600, 29)
        except Exception as e:  # noqa: BLE001
            rec["errors"].append(f"{type(e).__name__}: {str(e)[:120]}")
            time.sleep(15)
            continue
        if r["traffic_guard_passed"]:
            rec["clean_tok_s"].append(r["total_tok_s"])
            rec["ttft_median_s"].append(r["ttft_median_s"])
        else:
            rec["dirty"] += 1
            time.sleep(15)
    out["levels"][str(n)] = rec
    print(json.dumps({"tag": a.tag, "level": n, **{k: v for k, v in rec.items() if k != "errors"}}), flush=True)
out["complete"] = all(len(v["clean_tok_s"]) >= a.rounds for v in out["levels"].values())
json.dump(out, open(os.path.expanduser(f"$HK_WORK/kvfp4/paired-{a.tag}.json"), "w"))
sys.exit(0 if out["complete"] else 1)
