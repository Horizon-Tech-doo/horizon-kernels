#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""pi_probe.py (Horizon Tech, 30-09-2026): op-level probe for "P1 chunk-invariant prefill" (kernel plan §5.2 step 9).

Question per op: is token row r's output bit-identical whatever the step's row count M (1..8192), and what does the
invariant path cost? Runs in the vLLM Python environment; needs the GPU. Small tensors only (peak ~1 GB).

  N  NVFP4 linear, the 5 Qwen3.8 shapes, random weights (row invariance is a kernel property, not a weight property):
     fp4x2 = prefill_rs.linear_native(passes=2) with horizon_ops.row_scale's quantizer (production's HORIZON_PREFILL_RS=1
     path), regfrag = decode_regfrag_xl (1024-row pieces above 1024, as the plugin). Every M's rows [0, M) are compared
     with the largest M's rows [0, M).
  S  K5 skinny BF16 GEMM (in_proj_a/b, N 96, K 5120) at every M up to 8192 with the dispatcher's config and with larger
     tiles (checks the per-element summation order does not depend on the tile config), vs cuBLAS; timings.
  A  FlashInfer FA2 paged prefill (BF16 q, FP8 KV, HND, page 16, 24/4 heads, dim 256): per-call time with the planner's
     KV split vs disable_split_kv, and the invariance of a short chunk against the same rows inside a long chunk.
Timings: CUDA events, min over 5 rounds of the median of 10 calls (other load on the same GPU only inflates them).
Writes one JSON file."""
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

OUT = sys.argv[1]
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"N", "S", "A"}
dev = "cuda"
R = {"time": time.strftime("%d-%m-%Y %H:%M"), "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
     "N": {}, "S": {}, "A": {}}


def save():
    tmp = OUT + ".tmp"
    json.dump(R, open(tmp, "w"), indent=1)
    os.replace(tmp, OUT)


def tms(fn, reps=10, rounds=5):
    best = None
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        m = statistics.median(ts)
        best = m if best is None else min(best, m)
    return round(best, 2)


def rows_vs_ref(y, ref):
    d = (y.float() - ref[:y.shape[0]].float()).abs()
    return {"bit_identical": bool(torch.equal(y, ref[:y.shape[0]])), "max_abs": float(d.max()),
            "rows_identical_frac": round(float((d.amax(1) == 0).float().mean()), 6)}


# ------------------------------------------------------------------ N: NVFP4
if "N" in ONLY:
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
    from tilelang.contrib.sm120_nvfp4 import prefill_rs
    try:
        from tilelang.contrib.horizon_ops import row_scale
        prefill_rs.quantize_rs = row_scale.quantize_rs
        R["N"]["quantizer"] = "horizon_ops.row_scale"
    except Exception as e:  # noqa: BLE001
        R["N"]["quantizer"] = f"prefill_rs.quantize_rs (row_scale unavailable: {e!r})"
    MS = [1, 2, 3, 4, 5, 8, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 256, 511, 512, 513, 1000, 1024,
          1025, 1600, 2048, 3200, 4096]
    for (N, K) in [(16384, 5120), (34816, 5120), (5120, 17408), (5120, 6144), (14336, 5120)]:
        g = torch.Generator(device=dev).manual_seed(N + K)
        packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8, generator=g)
        scale = (torch.rand(N, K // 16, device=dev, generator=g) * 1.5 + 0.25).to(torch.float8_e4m3fn)
        gs = torch.ones(1, device=dev, dtype=torch.float32)
        wn = prefill_rs.prepare_weight_native(packed, scale, gs, shard_sizes=[N])
        wr = xl.prepare(packed.view(torch.int8), scale, wn.alpha)
        mmax = 2048 if N > 20000 else 4096
        ms = [m for m in MS if m <= mmax]
        x = torch.randn(mmax, K, device=dev, generator=g).to(torch.bfloat16)

        def fp4x2(a):
            return prefill_rs.linear_native(a, wn, passes=2)

        def regfrag(a):
            if a.shape[0] <= xl.SUPPORTS_M:
                return xl.linear(a, wr)
            return torch.cat([xl.linear(a[i:i + xl.SUPPORTS_M], wr) for i in range(0, a.shape[0], xl.SUPPORTS_M)])

        res = {}
        for name, fn in (("fp4x2", fp4x2), ("regfrag", regfrag)):
            ref = fn(x)
            per = {}
            for m in ms:
                per[m] = rows_vs_ref(fn(x[:m].contiguous()), ref)
            bad = [m for m, v in per.items() if not v["bit_identical"]]
            res[name] = {"row_invariant": not bad, "differs_at_M": bad,
                         "max_abs": max(v["max_abs"] for v in per.values())}
            del ref
            torch.cuda.empty_cache()
        a1, a2 = fp4x2(x[:1024]), regfrag(x[:1024])
        res["fp4x2_vs_regfrag_rel_err"] = float(((a1.float() - a2.float()).norm() / a2.float().norm()))
        res["timing_us"] = {m: {"fp4x2": tms(lambda: fp4x2(x[:m])), "regfrag": tms(lambda: regfrag(x[:m]))}
                            for m in (1, 4, 16, 64, 128, 256, 512)}
        R["N"][f"{N}x{K}"] = res
        print("N", N, K, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                              if kk != "timing_us"})
                                     for k, v in res.items() if k != "timing_us"}), flush=True)
        del packed, scale, wn, wr, x
        torch.cuda.empty_cache()
        save()

# ------------------------------------------------------------------ S: K5 skinny BF16 GEMM
if "S" in ONLY:
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    N, K = 96, 5120
    g = torch.Generator(device=dev).manual_seed(3)
    x = torch.randn(8192, K, device=dev, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    ref = sk.linear(x, w, cfg=sk.config(N, K, 8192))
    ms = sorted(set(list(range(1, 65)) + [96, 127, 128, 129, 256, 512, 533, 960, 1024, 1025, 1600, 1920, 1984, 2048,
                                          2304, 2368, 3200, 4096, 4800, 6400, 8000, 8192]))
    bad = [m for m in ms if not torch.equal(sk.linear(x[:m].contiguous(), w), ref[:m])]
    cfgs = {}
    for cfg in [(8, 4, 128), (8, 8, 128), (16, 4, 128), (16, 2, 128), (4, 8, 128)]:
        try:
            y = sk.linear(x, w, cfg=cfg)
            cfgs[str(cfg)] = {"bit_identical_to_dispatch_config": bool(torch.equal(y, ref)),
                              "us": {m: tms(lambda cfg=cfg, m=m: sk.linear(x[:m], w, cfg=cfg))
                                     for m in (48, 128, 512, 1600, 4096, 8192)}}
        except Exception as e:  # noqa: BLE001
            cfgs[str(cfg)] = {"error": repr(e)[:300]}
    cub = F.linear(x, w)
    R["S"] = {"row_invariant_dispatch_configs": not bad, "differs_at_M": bad,
              "vs_cublas_max_abs": float((ref.float() - cub.float()).abs().max()),
              "vs_cublas_rel_err": float((ref.float() - cub.float()).norm() / cub.float().norm()),
              "cublas_us": {m: tms(lambda m=m: F.linear(x[:m], w)) for m in (48, 128, 512, 1600, 4096, 8192)},
              "configs": cfgs}
    print("S", json.dumps({k: v for k, v in R["S"].items() if k != "configs"}), flush=True)
    for c, v in cfgs.items():
        print("S cfg", c, json.dumps(v), flush=True)
    del x, w, ref, cub
    torch.cuda.empty_cache()
    save()

# ------------------------------------------------------------------ A: FA2 prefill split vs no split
if "A" in ONLY:
    import flashinfer
    QH, KVH, D, PAGE = 24, 4, 256, 16
    LMAX = 65536
    g = torch.Generator(device=dev).manual_seed(5)
    kv = (torch.randn(LMAX // PAGE, 2, KVH, PAGE, D, device=dev, generator=g) * 1.5).to(torch.float8_e4m3fn)
    q_all = torch.randn(2048, QH, D, device=dev, generator=g).to(torch.bfloat16)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    def plan_run(a, b, dsk):
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        npg = (b + PAGE - 1) // PAGE
        i32 = dict(dtype=torch.int32)
        wr.plan(torch.tensor([0, b - a], **i32), torch.tensor([0, npg], **i32), torch.arange(npg, device=dev, **i32),
                torch.tensor([b - PAGE * (npg - 1)], **i32), QH, KVH, D, PAGE, causal=True, sm_scale=D ** -0.5,
                q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn, o_data_type=torch.bfloat16,
                disable_split_kv=dsk)
        q = q_all[:b - a]
        return wr, q

    out = {}
    for L in (4096, 16384, 65536):
        for qn in (16, 64, 256, 512, 1024, 1600):
            row = {}
            for dsk in (False, True):
                wr, q = plan_run(L - qn, L, dsk)
                row["nosplit_us" if dsk else "split_us"] = tms(lambda: wr.run(q, kv, k_scale=1.0, v_scale=1.0))
                row["nosplit_out" if dsk else "split_out"] = wr.run(q, kv, k_scale=1.0, v_scale=1.0)
            row["split_equals_nosplit"] = bool(torch.equal(row.pop("split_out"), row.pop("nosplit_out")))
            row["cost_ratio"] = round(row["nosplit_us"] / row["split_us"], 3)
            out[f"kv{L}_q{qn}"] = row
            print("A", L, qn, json.dumps(row), flush=True)
    # invariance: last 64 rows as their own no-split chunk == the same rows inside a 1600-row no-split chunk
    w1, q1 = plan_run(LMAX - 1600, LMAX, True)
    big = w1.run(q_all[:1600], kv, k_scale=1.0, v_scale=1.0)
    w2, _ = plan_run(LMAX - 64, LMAX, True)
    small = w2.run(q_all[1536:1600].contiguous(), kv, k_scale=1.0, v_scale=1.0)
    out["nosplit_tail64_vs_chunk1600_at_64k"] = bool(torch.equal(small, big[1536:]))
    R["A"] = out
    save()
print("DONE", flush=True)
save()
