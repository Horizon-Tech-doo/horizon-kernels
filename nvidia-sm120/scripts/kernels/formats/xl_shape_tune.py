# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S2 (formats track): config sweep of decode_regfrag_xl for one NVFP4 shape (default the lm_head 248320x5120).
#   xl_shape_tune.py NxK "M1,M2,..." "cfg1;cfg2;..."   cfg = warps_m,WM,warps_n,nb,block_K,split_k,stages[,min_blocks[,late]]
#   or "rf" (the current default path). Appends one JSON line per (M, cfg) to $HK_WORK/formats/xl_shape_tune.jsonl.
# Method (k2_race.py): CUDA-graph replay, >= 512 MB of rotated weight copies, separator kernel, 3 rounds, fastest;
# correctness of every cfg vs an exact FP32 reference (exact class <= 0.0025) and determinism.
import json
import os
import sys
import time

import torch

from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4.decode_contract import dequantize_reference, synthetic_layer

N, K = (int(v) for v in sys.argv[1].split("x"))
MS = [int(v) for v in sys.argv[2].split(",")]
CFGS = []
for c in sys.argv[3].split(";"):
    p = c.split(",")
    CFGS.append(None if p[0] == "auto" else (("rf",) if p[0] == "rf" else tuple(int(v) for v in p)))
OUT = os.environ.get("OUT", "$HK_WORK/formats/xl_shape_tune.jsonl")
SEP = torch.zeros(1, device="cuda")


def graph_us(fns, iters=3, rounds=3):
    fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for f in fns:
        f()
    torch.cuda.synchronize()
    for attempt in range(3):
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                for _ in range(iters):
                    for f in fns:
                        f()
            break
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt == 2:
                raise
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (iters * len(fns)))
    return min(ts)


copies = max(2, int(512e6 // (N * K * 0.5625)) + 1)
lays = [synthetic_layer(N, K, seed=s) for s in range(copies)]
preps = [xl.prepare(*l) for l in lays]
wref = dequantize_reference(*lays[0])
sep = graph_us([lambda: None] * 8)
out = open(OUT, "a")
print(json.dumps({"N": N, "K": K, "copies": copies, "sep_us": round(sep, 2)}), flush=True)
for m in MS:
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ wref.T
    best = None
    for cfg in CFGS:
        row = {"N": N, "K": K, "M": m, "cfg": list(cfg) if cfg else ["auto", list(xl.config(N, K, m))],
               "time": time.strftime("%H:%M")}
        try:
            y = xl.linear(x, preps[0], cfg)
            y2 = xl.linear(x, preps[0], cfg)
            row["err"] = round(((y.float() - ref).norm() / ref.norm()).item(), 6)
            row["det"] = bool(torch.equal(y, y2))
            t = graph_us([lambda p=p: xl.linear(x, p, cfg) for p in preps]) - sep
            row["us"] = round(t, 2)
            row["tbs"] = round(N * K * 0.5625 / (t * 1e-6) / 1e12, 3)
            row["tflops"] = round(2 * m * N * K / (t * 1e-6) / 1e12, 1)
            if row["err"] <= 0.0025 and row["det"] and (best is None or t < best[0]):
                best = (t, cfg)
        except Exception as ex:  # noqa: BLE001
            row["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n"); out.flush()
    print(json.dumps({"M": m, "best_us": round(best[0], 2) if best else None,
                      "best_cfg": (list(best[1]) if best[1] else "auto") if best else None}),
          flush=True)
