# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S4 tuning (formats track): config sweep of decode_w8a16 for FP8 shapes vs vLLM's CUTLASS W8A8 (GEMM only).
#   w8_tune.py "N1xK1,N2xK2" "M1,M2" "cfg1;cfg2;..."    cfg = warps_m,WM,warps_n,nb,block_K,split_k,stages[,min_blocks]
#   split_k 0 = auto (smallest split giving >= 188 CTAs). Appends to $HK_WORK/formats/w8_tune.jsonl.
# Method: CUDA-graph replay, >= 512 MB of rotated weight copies, separator kernel, 3 rounds, fastest; every cfg
# checked against an exact FP32 reference (<= 0.0025) and for determinism.
import json
import os
import sys
import time

import torch

from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8

SHAPES = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
MS = [int(v) for v in sys.argv[2].split(",")]
RAW = [tuple(int(v) for v in c.split(",")) for c in sys.argv[3].split(";")]
OUT = os.environ.get("OUT", "$HK_WORK/formats/w8_tune.jsonl")
SEP = torch.zeros(1, device="cuda")


def graph_us(fns, iters=4, rounds=3):
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


def fill(cfg, N, K, M):
    cfg = list(cfg)
    if cfg[5] == 0:
        bm, bn = cfg[0] * cfg[1], cfg[2] * 32 * cfg[3]
        cfg[5] = w8._choose_split(-(-M // bm) * (N // bn), K // cfg[4])
    return tuple(cfg)


def main():
    from vllm import _custom_ops as ops
    out = open(OUT, "a")
    sep = graph_us([lambda: None] * 8)
    for N, K in SHAPES:
        copies = max(2, int(512e6 // (N * K)) + 1)
        ws = torch.tensor(1e-3, device="cuda")
        wts = [(torch.randn(N, K, device="cuda") * 64).to(torch.float8_e4m3fn) for _ in range(copies)]
        preps = [w8.prepare(w, ws) for w in wts]
        wref = wts[0].float() * 1e-3
        for m in MS:
            x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ wref.T
            xq, xs = ops.scaled_fp8_quant(x, torch.tensor([4.0 / 448], device="cuda"))
            t_c = graph_us([lambda w=w: ops.cutlass_scaled_mm(xq, w.t(), out_dtype=torch.bfloat16, scale_a=xs,
                                                                scale_b=ws.reshape(1)) for w in wts]) - sep
            best = None
            for raw in RAW + [tuple(w8.config(N, K, m))]:
                cfg = fill(raw, N, K, m)
                row = {"N": N, "K": K, "M": m, "cfg": list(cfg), "us_cutlass": round(t_c, 2), "time": time.strftime("%H:%M")}
                try:
                    if (K // cfg[4]) % cfg[5] or N % (cfg[2] * 32 * cfg[3]) or w8.smem_bytes(cfg) > 99 * 1024:
                        continue
                    y = w8.linear(x, preps[0], cfg)
                    y2 = w8.linear(x, preps[0], cfg)
                    row["err"] = round(((y.float() - ref).norm() / ref.norm()).item(), 6)
                    row["det"] = bool(torch.equal(y, y2))
                    t = graph_us([lambda p=p: w8.linear(x, p, cfg) for p in preps]) - sep
                    row.update({"us": round(t, 2), "pct_1.8": round(100 * N * K / (t * 1e-6) / 1.8e12, 1),
                                "vs_cutlass": round(t_c / t, 3)})
                    if row["err"] <= 0.0025 and row["det"] and (best is None or t < best[0]):
                        best = (t, cfg)
                except Exception as ex:  # noqa: BLE001
                    row["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
                print(json.dumps(row), flush=True)
                out.write(json.dumps(row) + "\n"); out.flush()
            print(json.dumps({"N": N, "K": K, "M": m, "cutlass_us": round(t_c, 2),
                              "best_us": round(best[0], 2) if best else None, "best_cfg": list(best[1]) if best else None,
                              "best_pct_1.8": round(100 * N * K / (best[0] * 1e-6) / 1.8e12, 1) if best else None}),
                  flush=True)
        del wts, preps
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
