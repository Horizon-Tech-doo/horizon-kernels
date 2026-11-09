# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""bench_serving_gpu.py OUT.jsonl --pkg-dir DIR [quick]   (29-09-2026 tuning, vLLM stopped, under the GPU lock)
Speed of the serving kernels in the shapes vLLM sends, CUDA-graph replay, >512 MiB of rotated caches per format, 3
interleaved rounds, median of 10:
  decode: kv_fp4_serving.decode_ragged (+ combine) for QL 1 / 2 / 4, MINC 128 / 256 x token groups TW 1 / 2 / 4
          (NST 3, 3, 2), 4 CTAs/SM split plan, vs
          FlashInfer XQA on an FP8 cache (what production runs), B 1 / 8 / 32, context 1k / 4k / 32k
  write : kv_fp4_serving.write (search on / off) vs the TileLang kv_write_fp4 vs vLLM's FP8 reshape_and_cache_flash,
          at 1 / 4 / 32 / 128 / 8192 tokens
Speed only; correctness is test_kvfp4_serving_gpu.py.
Env (optional): BENCH_VARIANTS="minc:tw:nst:cps,..." (decode variants; cps = CTAs per SM for the split plan),
BENCH_B="1,8,32", BENCH_CTX="1024,4096,32768", BENCH_QL="4,2,1", BENCH_WRITE=0 skips the write sweep.
BENCH_HAD=1: every fp4 decode variant runs with the fused Hadamard q rotation (signs), plus "sep" = the old separate
rotate_q launch + the plain kernel (first variant's config); the write sweep adds the fused / TileLang Hadamard writes."""
import importlib
import json
import math
import os
import sys

import torch

OUT = sys.argv[1]
QUICK = "quick" in sys.argv
sys.path.insert(0, sys.argv[sys.argv.index("--pkg-dir") + 1])
SV = importlib.import_module("horizon_kvfp4.kernels.kv_fp4_serving")
AD = importlib.import_module("horizon_kvfp4.kernels.attn_decode_fp4")
RF = importlib.import_module("horizon_kvfp4.kernels.kv_fp4_ref")
HAD = os.environ.get("BENCH_HAD", "0") == "1"
K8 = os.environ.get("BENCH_K8", "0") == "1"          # K8V4 slot: FP8 keys + FP4 values
KW = importlib.import_module("horizon_kvfp4.kernels.kv_write_fp4")
dev = torch.device("cuda")
D, PS, KVH, G = 256, 64, 4, 6
HQ = KVH * G
SCALE = D ** -0.5
WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)


def log(r):
    print(json.dumps(r), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    for a in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn()
            return g
        except Exception as e:  # noqa: BLE001
            print("capture retry", a, e, flush=True)
            g = torch.cuda.CUDAGraph()
    raise RuntimeError("capture failed")


def timeit(gs):
    best = {}
    names = list(gs)
    for rnd in range(3):
        for n in names[rnd % len(names):] + names[:rnd % len(names)]:
            g, div = gs[n]
            g.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(10):
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / div)
            ts.sort()
            best[n] = min(best.get(n, 1e30), ts[5])
    return {k: round(v, 2) for k, v in best.items()}


def fp4_cache(NP, gen):
    if K8:
        kv = torch.randint(0, 256, (NP, KVH, PS, 400), device=dev, generator=gen, dtype=torch.uint8)
        kv[..., :256] &= 0xB7                                    # finite e4m3 keys of moderate size
        kv[..., 384:] = torch.randint(0x20, 0x41, kv[..., 384:].shape, device=dev, generator=gen, dtype=torch.uint8)
        t = kv.permute(0, 2, 1, 3)
        return SV.flat_views(t[..., :256], t[..., 384:], t[..., 256:384], t[..., 384:])
    kv = torch.randint(0, 256, (NP, 2 * KVH, PS, 144), device=dev, generator=gen, dtype=torch.uint8)
    kv[..., 128:] = torch.randint(0x20, 0x41, kv[..., 128:].shape, device=dev, generator=gen, dtype=torch.uint8)
    t = kv.permute(0, 2, 1, 3)
    views = (t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:])
    return SV.flat_views(*views)


def decode_bench():
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask
    gen = torch.Generator(device=dev)
    gen.manual_seed(9)
    ws32 = torch.empty(8 * 1024 * 1024, dtype=torch.float32, device=dev)
    ctxs = (1024, 32768) if QUICK else (1024, 4096, 32768)
    ctxs = [int(x) for x in os.environ.get("BENCH_CTX", ",".join(map(str, ctxs))).split(",")]
    variants = [tuple(int(y) for y in v.split(":")) for v in os.environ.get(
        "BENCH_VARIANTS", "128:1:3:4,128:2:3:4,128:4:2:4,256:1:3:4,256:2:3:4,256:4:2:4").split(",")]
    for B in [int(x) for x in os.environ.get("BENCH_B", "1,8,32").split(",")]:
        for L in ctxs:
            if B * L > 1_100_000:
                continue
            for QL in [int(x) for x in os.environ.get("BENCH_QL", "4,2,1").split(",")]:
                NP = B * math.ceil(L / PS) + 3
                perm = torch.randperm(NP - 1, device=dev, generator=gen) + 1
                bt = perm[: B * math.ceil(L / PS)].view(B, -1).int().contiguous()
                sl = torch.full((B,), L, dtype=torch.int32, device=dev)
                qsl = torch.arange(0, (B + 1) * QL, QL, dtype=torch.int32, device=dev)
                q = torch.randn(B * QL, HQ, D, device=dev, generator=gen).to(torch.bfloat16)
                o = torch.empty_like(q)
                n8 = max(2, math.ceil(513 * 1024 ** 2 / (NP * KVH * PS * 2 * D)))
                n4 = max(2, math.ceil(513 * 1024 ** 2 / (NP * KVH * PS * (400 if K8 else 288))))
                c8 = []
                for _ in range(n8):
                    kv = (torch.randn(NP, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
                    c8.append((kv[..., :D], kv[..., D:]))
                c4 = [fp4_cache(NP, gen) for _ in range(n4)]
                gk = torch.ones(KVH, device=dev)
                mask = (_make_xqa_draft_block_mask(QL, True, dev).unsqueeze(0).expand(B, -1, -1).contiguous()
                        if QL > 1 else None)

                def gx():
                    for kc, vc in c8:
                        xqa_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace_buffer=WS, block_tables=bt,
                                                       seq_lens=sl, max_seq_len=L, bmm1_scale=SCALE, bmm2_scale=1.0,
                                                       window_left=-1, out=o, sinks=None, kv_layout="HND",
                                                       q_len_per_req=QL, mask=mask)

                sg = RF.hadamard_signs(D).to(dev) if HAD else None
                qr = torch.empty_like(q)

                def mk(minc, tw, nst, cps, sep=False):
                    def f():
                        for flat, st in c4:
                            if sep:              # old path: separate rotation launch, plain kernel
                                AD.rotate_q(q, sg, out=qr)
                            SV.decode_ragged(qr if sep else q, flat, st, PS, bt, sl, qsl, SCALE, 1.0, gk, gk, QL, o,
                                             ws32, nst=nst, min_chunk=minc, splits=SV.plan_splits(B, KVH, cps), tw=tw,
                                             signs=None if sep else sg, k8=K8)
                    return f
                gs = {"xqa_fp8": (graph_of(gx), n8)}
                for minc, tw, nst, cps in variants:
                    if SV.smem_bytes(G, QL, SV.default_ncw(G, QL), nst, tw, int(HAD), int(K8)) > 99 * 1024:
                        continue
                    gs[f"fp4_mc{minc}_tw{tw}_n{nst}_c{cps}"] = (graph_of(mk(minc, tw, nst, cps)), n4)
                if HAD:
                    minc, tw, nst, cps = variants[0]
                    gs["sep_rot"] = (graph_of(mk(minc, tw, nst, cps, sep=True)), n4)
                r = timeit(gs)
                best = min(v for k, v in r.items() if k.startswith("fp4"))
                log({"test": "decode", "had": HAD, "k8": K8, "B": B, "ctx": L, "QL": QL, **r,
                     "fp4_best_vs_xqa": round(r["xqa_fp8"] / best, 3)})
                del c8, c4, gs
                torch.cuda.empty_cache()


def write_bench():
    from vllm import _custom_ops  # noqa: F401
    gen = torch.Generator(device=dev)
    gen.manual_seed(10)
    for NT in (1, 4, 32, 128, 8192):
        NP = math.ceil(NT / PS) + 3
        qkv = torch.randn(NT, 3 * KVH * D, device=dev, generator=gen).to(torch.bfloat16)
        key, val = qkv[:, KVH * D:2 * KVH * D].view(NT, KVH, D), qkv[:, 2 * KVH * D:].view(NT, KVH, D)
        slots = torch.arange(NT, device=dev, dtype=torch.int64)
        kv4 = torch.zeros(NP, 2 * KVH, PS, 144, device=dev, dtype=torch.uint8)
        t = kv4.permute(0, 2, 1, 3)
        views = (t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:])
        flat, st = SV.flat_views(*views)
        kv8 = torch.zeros(NP, KVH, PS, 2 * D, device=dev, dtype=torch.float8_e4m3fn)
        k8, v8 = kv8.permute(0, 2, 1, 3).split(D, dim=-1)
        one = torch.ones(1, device=dev)
        g4 = torch.ones(KVH, device=dev)
        gs = {"fp4_cuda_search": (graph_of(lambda: SV.write(key, val, flat, st, PS, slots, g4, g4, True)), 1),
              "fp4_cuda_nosearch": (graph_of(lambda: SV.write(key, val, flat, st, PS, slots, g4, g4, False)), 1),
              "fp4_tilelang_search": (graph_of(lambda: KW.write(key, val, *views, slots, g4, g4, None, True)), 1),
              "fp8_reshape_and_cache": (graph_of(lambda: torch.ops._C_cache_ops.reshape_and_cache_flash(
                  key, val, k8, v8, slots, "fp8", one, one)), 1)}
        if K8:
            kv8 = torch.zeros(NP, KVH, PS, 400, device=dev, dtype=torch.uint8)
            t8 = kv8.permute(0, 2, 1, 3)
            f8, s8 = SV.flat_views(t8[..., :256], t8[..., 384:], t8[..., 256:384], t8[..., 384:])
            gs["k8v4_cuda_search"] = (graph_of(lambda: SV.write(key, val, f8, s8, PS, slots, g4, g4, True, None,
                                                                True)), 1)
        if HAD:
            sg = RF.hadamard_signs(D).to(dev)
            gs["fp4_cuda_had"] = (graph_of(lambda: SV.write(key, val, flat, st, PS, slots, g4, g4, True, sg)), 1)
            gs["fp4_tilelang_had"] = (graph_of(lambda: KW.write(key, val, *views, slots, g4, g4, sg, True)), 1)
        log({"test": "write", "NT": NT, **timeit(gs)})


if __name__ == "__main__":
    if os.environ.get("BENCH_WRITE", "1") == "1":
        write_bench()
    decode_bench()
    log({"test": "complete"})
