# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for the Horizon NVFP4 KV-cache kernels (kv_write_fp4, attn_decode_fp4). NOT RUN YET (29-09-2026).
Run only under the GPU lock with vLLM stopped, e.g.
    $HK_WORK/gpu_lock.sh kvfp4-gate python test_kvfp4_gpu.py OUT.jsonl [quick] [--pkg-dir DIR]
--pkg-dir DIR: import the candidate as package `hzkv` from DIR (staged copies, like kvfp4_lower.py); default imports
tilelang.contrib.horizon_ops.{kv_write_fp4,attn_decode_fp4,kv_fp4_ref}.

Gates (every one must pass; the terminal record says complete=true only if all ran):
  G1 write exactness: cache bytes == kv_fp4_ref.write_reference (computed on CPU) for had 0/1 x search 0/1, random /
     outlier-heavy / zero / tiny / huge tokens, padded slots (-1) skipped, bytes outside written slots untouched;
     deterministic; two CUDA-graph replays of the write give the same bytes
  G2 rotate_q exactness: == kv_fp4_ref.rotate_q (CPU) bit for bit, graph replay
  G3 decode: relative L2 <= 0.0025 vs the fp64 attention over the SAME dequantised cache (kv_fp4_ref
     .attention_reference with the kernel's own bf16 q'), pipe 0 and 1, QL 1 and 4, dense (4 kv heads, GQA 6) and MoE
     (2 kv heads, GQA 8) geometry, non-unit gk / gv; finite; deterministic; two graph replays after NaN poisoning ==
     eager. Also REPORTED (not gated): error vs attention over the unquantised bf16 K/V and vs the FP8 cache.
  G4 speed (report): graph replay, >512 MiB of rotated KV copies PER FORMAT, 3 interleaved rounds, median of 10:
     attn_decode_fp4 (pipe 0/1) vs FlashInfer XQA on the FP8 cache (what vLLM runs today) vs our FP8 attn_decode, at
     B 1 / 8 / 32, context 1k .. 128k, QL 4 and 1; memory floor at 1.63 TB/s for both formats. kv write: NVFP4 vs
     vLLM's FP8 reshape_and_cache_flash at 1 / 32 / 8192 tokens.
"""
import importlib
import json
import math
import os
import sys

import torch

OUT = sys.argv[1]
QUICK = "quick" in sys.argv[2:]
if "--pkg-dir" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--pkg-dir") + 1])
    KW = importlib.import_module("hzkv.kv_write_fp4")
    AD = importlib.import_module("hzkv.attn_decode_fp4")
    R = importlib.import_module("hzkv.kv_fp4_ref")
    A8 = importlib.import_module("hzkv.attn_decode")
    PKG = "hzkv"
else:
    KW = importlib.import_module("tilelang.contrib.horizon_ops.kv_write_fp4")
    AD = importlib.import_module("tilelang.contrib.horizon_ops.attn_decode_fp4")
    R = importlib.import_module("tilelang.contrib.horizon_ops.kv_fp4_ref")
    A8 = importlib.import_module("tilelang.contrib.horizon_ops.attn_decode")
    PKG = "horizon_ops"

dev = torch.device("cuda")
PS = 64
D = 256
SCALE = D ** -0.5
LIMIT = 0.0025
WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
COMPLETE = {"G1": False, "G2": False, "G3": False, "G4": False}


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def graph_of(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    for attempt in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn()
            return g
        except Exception as e:  # noqa: BLE001
            print("capture retry", attempt, e, flush=True)
            g = torch.cuda.CUDAGraph()
    raise RuntimeError("capture failed")


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def block_table(B, lens, gen):
    pages = [math.ceil(L / PS) for L in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=gen) + 1
    bt = torch.zeros(B, max(pages), dtype=torch.int32)
    o = 0
    for b, p in enumerate(pages):
        bt[b, :p] = perm[o:o + p].int()
        o += p
    slots = torch.tensor([int(bt[b, p // PS]) * PS + p % PS for b, L in enumerate(lens) for p in range(L)],
                         dtype=torch.int64)
    return NP, bt, slots


def kv_tokens(NT, KVH, kind, gen):
    k = torch.randn(NT, KVH, D, generator=gen)
    v = torch.randn(NT, KVH, D, generator=gen) * torch.exp(0.5 * torch.randn(NT, 1, 1, generator=gen))
    if kind == "outlier":
        ch = torch.tensor([0, 1, 64, 65, 126, 127, 200, 201])
        k[:, :, ch] = k[:, :, ch] * 6 + 10
        k[0] *= 8
    elif kind == "edge":                       # each edge row only if the case has that many tokens (NT may be 1)
        if NT > 0: k[0] = 0
        if NT > 1: v[1] = 0
        if NT > 2: k[2] *= 1e-30
        if NT > 3: v[3] *= 1e30
        if NT > 4: k[4, :, :16] = 300.0
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------------------ G1 write
def g1():
    gen = torch.Generator().manual_seed(1)
    KVH = 4
    signs = R.hadamard_signs(D)
    for kind in ("random", "outlier", "edge"):
        for NT in (1, 7, 300):
            NP, _, _ = block_table(1, [NT], gen)
            slots = torch.randperm(NP * PS, generator=gen)[:NT].to(torch.int64)
            slots[torch.rand(NT, generator=gen) < 0.1] = -1          # padding tokens
            key, val = kv_tokens(NT, KVH, kind, gen)
            gk = torch.tensor([1.0, 0.5, 2.0, 1.0])
            gv = torch.tensor([1.0, 1.0, 0.25, 4.0])
            for had in (1, 0):
                for search in (1, 0):
                    sg = signs if had else None
                    ref = R.alloc_cache(NP, PS, KVH, D, fill=0xA5)
                    R.write_reference(key, val, R.split_views(ref, KVH), slots, gk, gv, sg, bool(search))
                    got = R.alloc_cache(NP, PS, KVH, D, device=dev, fill=0xA5)
                    views = R.split_views(got, KVH)
                    args = (key.to(dev), val.to(dev), *views, slots.to(dev), gk.to(dev), gv.to(dev),
                            sg.to(dev) if sg is not None else None, bool(search))
                    KW.write(*args)
                    torch.cuda.synchronize()
                    exact = torch.equal(got.cpu(), ref)
                    first = got.clone()
                    got.fill_(0xA5)
                    KW.write(*args)
                    det = torch.equal(got, first)
                    g = graph_of(lambda: KW.write(*args))
                    rep = True
                    for _ in range(2):
                        got.fill_(0xA5)
                        g.replay()
                        torch.cuda.synchronize()
                        rep = rep and torch.equal(got, first)
                    del g
                    nbad = int((got.cpu() != ref).sum())
                    log({"test": "G1_write", "kind": kind, "NT": NT, "had": had, "search": search, "exact": exact,
                         "mismatched_bytes": nbad, "deterministic": det, "graph_replay_ok": rep})
                    assert exact and det and rep, "G1 write gate"
    COMPLETE["G1"] = True


# --------------------------------------------------------------------------------------------------------- G2 rotate_q
def g2():
    gen = torch.Generator().manual_seed(2)
    signs = R.hadamard_signs(D)
    for NT, HQ in ((1, 24), (32, 24), (129, 16)):
        q = (torch.randn(NT, HQ, D, generator=gen) * 3).to(torch.bfloat16)
        ref = R.rotate_q(q, signs)
        out = torch.empty(NT, HQ, D, dtype=torch.bfloat16, device=dev)
        qd, sd = q.to(dev), signs.to(dev)
        AD.rotate_q(qd, sd, out)
        torch.cuda.synchronize()
        exact = torch.equal(out.cpu(), ref)
        first = out.clone()
        g = graph_of(lambda: AD.rotate_q(qd, sd, out))
        rep = True
        for _ in range(2):
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            rep = rep and torch.equal(out, first)
        log({"test": "G2_rotate_q", "NT": NT, "HQ": HQ, "exact": exact, "graph_replay_ok": rep})
        assert exact and rep, "G2 rotate gate"
    COMPLETE["G2"] = True


# ----------------------------------------------------------------------------------------------------------- G3 decode
def build(B, lens, QL, KVH, G, gen, had=True, gk=None, gv=None):
    NP, bt, slots = block_table(B, lens, gen)
    NT = sum(lens)
    key, val = kv_tokens(NT, KVH, "outlier" if had else "random", gen)
    gk = torch.ones(KVH) if gk is None else gk
    gv = torch.ones(KVH) if gv is None else gv
    signs = R.hadamard_signs(D) if had else None
    cache = R.alloc_cache(NP, PS, KVH, D, device=dev, fill=0x77)
    views = R.split_views(cache, KVH)
    KW.write(key.to(dev), val.to(dev), *views, slots.to(dev), gk.to(dev), gv.to(dev),
             signs.to(dev) if signs is not None else None)
    q = torch.randn(B * QL, KVH * G, D, generator=gen).to(torch.bfloat16).to(dev)
    qr = AD.rotate_q(q, signs.to(dev)) if had else q
    return dict(cache=cache, views=views, bt=bt.to(dev), sl=torch.tensor(lens, dtype=torch.int32, device=dev),
                key=key, val=val, q=q, qr=qr, gk=gk.to(dev), gv=gv.to(dev), QL=QL, KVH=KVH, G=G, lens=lens,
                signs=signs)


def run_fp4(c, out, pipe, splits=None):
    AD.decode(c["qr"], *c["views"], c["bt"], c["sl"], 0, SCALE, 1.0, c["QL"], c["gk"], c["gv"], out=out, pipe=pipe,
              splits=splits)


def unquantised_reference(c):
    """fp64 attention of the ORIGINAL q over the unquantised bf16 K / V (what an unquantised cache would give)"""
    KVH, G, QL = c["KVH"], c["G"], c["QL"]
    out = torch.empty(c["q"].shape, dtype=torch.float64, device=dev)
    o = 0
    K = c["key"].double().to(dev)
    V = c["val"].double().to(dev)
    for b, L in enumerate(c["lens"]):
        for t in range(QL):
            lim = L - QL + t + 1
            qq = c["q"][b * QL + t].double().view(KVH, G, D)
            s = torch.einsum("hgd,lhd->hgl", qq, K[o:o + lim]) * SCALE
            out[b * QL + t] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, -1), V[o:o + lim]).reshape(-1, D)
        o += L
    return out


def fp8_reference(c):
    """fp64 attention of the original q over an FP8 (scale 1) copy of K / V: today's cache error, for comparison"""
    k8 = c["key"].float().clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.bfloat16)
    v8 = c["val"].float().clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.bfloat16)
    return unquantised_reference(dict(c, key=k8, val=v8))


def g3():
    gen = torch.Generator().manual_seed(3)
    cases = [(1, [37], 1), (1, [1000], 4), (1, [1601], 4), (3, [5, 3300, 900], 4), (8, [4000] * 8, 4),
             (2, [20000, 7], 1), (1, [65536], 4)]
    if QUICK:
        cases = cases[:4]
    for KVH, G in ((4, 6), (2, 8)):
        for B, lens, QL in cases:
            for had in (True, False):
                gk = torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH])
                gv = torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH])
                c = build(B, lens, QL, KVH, G, gen, had=had, gk=gk, gv=gv)
                ref = R.attention_reference(c["qr"], c["views"], c["bt"], c["sl"], QL, SCALE, c["gk"], c["gv"])
                rec = {"test": "G3_decode", "KVH": KVH, "G": G, "B": B, "lens": lens[:4], "QL": QL, "had": had}
                for pipe in (0, 1):
                    o1 = torch.empty_like(c["q"])
                    o2 = torch.empty_like(c["q"])
                    run_fp4(c, o1, pipe)
                    run_fp4(c, o2, pipe)
                    torch.cuda.synchronize()
                    r = rel(o1, ref)
                    fin = bool(torch.isfinite(o1).all())
                    det = torch.equal(o1, o2)
                    exp = o1.clone()
                    g = graph_of(lambda: run_fp4(c, o1, pipe))
                    rep = True
                    for _ in range(2):
                        o1.fill_(float("nan"))
                        g.replay()
                        torch.cuda.synchronize()
                        rep = rep and torch.equal(o1, exp)
                    del g
                    rec.update({f"pipe{pipe}_rel_l2": r, f"pipe{pipe}_finite": fin, f"pipe{pipe}_deterministic": det,
                                f"pipe{pipe}_graph_replay_ok": rep})
                    assert fin and det and rep and r <= LIMIT, f"G3 decode gate pipe {pipe}: {r}"
                if sum(lens) <= 70000:                      # reported, not gated: the format's own error
                    ure = unquantised_reference(c)
                    rec["fp4_vs_unquantised_rel_l2"] = rel(exp, ure)
                    rec["fp8_vs_unquantised_rel_l2"] = rel(fp8_reference(c), ure)
                log(rec)
    COMPLETE["G3"] = True


# ------------------------------------------------------------------------------------------------------------ G4 speed
def g4():
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask
    gen = torch.Generator(device=dev)
    gen.manual_seed(4)
    KVH, G = 4, 6
    HQ = KVH * G
    ctxs = (1024, 32768) if QUICK else (1024, 4096, 16384, 32768, 131072)
    for B in (1, 8, 32):
        for L in ctxs:
            if B * L > 1_100_000:
                continue
            for QL in (4, 1):
                NP = B * math.ceil(L / PS) + 3
                perm = torch.randperm(NP - 1, device=dev, generator=gen) + 1
                bt = perm[: B * math.ceil(L / PS)].view(B, -1).int().contiguous()
                sl = torch.full((B,), L, dtype=torch.int32, device=dev)
                q = torch.randn(B * QL, HQ, D, device=dev, generator=gen).to(torch.bfloat16)
                fp8_bytes = NP * KVH * PS * 2 * D
                fp4_bytes = R.page_bytes(PS, KVH, D) * NP
                n8 = max(2, math.ceil(513 * 1024 ** 2 / fp8_bytes))
                n4 = max(2, math.ceil(513 * 1024 ** 2 / fp4_bytes))
                caches8 = []
                for _ in range(n8):
                    kv = (torch.randn(NP, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
                    caches8.append((kv[..., :D], kv[..., D:]))
                caches4 = []
                for _ in range(n4):
                    cache = torch.randint(0, 256, (NP, 2 * KVH, PS, D // 2 + D // 16), device=dev, generator=gen,
                                          dtype=torch.uint8)
                    v = R.split_views(cache, KVH)
                    for sf in (v[1], v[3]):                 # valid, finite e4m3 scales (codes 0x20 .. 0x40)
                        sf.copy_(torch.randint(0x20, 0x41, sf.shape, device=dev, generator=gen, dtype=torch.uint8))
                    caches4.append(v)
                gk = torch.ones(KVH, device=dev)
                o = torch.empty_like(q)
                mask = (_make_xqa_draft_block_mask(QL, True, dev).unsqueeze(0).expand(B, -1, -1).contiguous()
                        if QL > 1 else None)

                def gx():
                    for kc, vc in caches8:
                        xqa_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace_buffer=WS,
                                                       block_tables=bt, seq_lens=sl, max_seq_len=L,
                                                       bmm1_scale=SCALE, bmm2_scale=1.0, window_left=-1, out=o,
                                                       sinks=None, kv_layout="HND", q_len_per_req=QL, mask=mask)

                def g8():
                    for kc, vc in caches8:
                        A8.decode(q, kc, vc, bt, sl, L, SCALE, 1.0, QL, out=o, kv_layout="HND", pipe=1, block_N=32)

                def g4p(pipe):
                    def f():
                        for v in caches4:
                            AD.decode(q, *v, bt, sl, L, SCALE, 1.0, QL, gk, gk, out=o, pipe=pipe)
                    return f
                gs = {"xqa_fp8": graph_of(gx), "ours_fp8_pipe": graph_of(g8), "fp4_pipe0": graph_of(g4p(0)),
                      "fp4_pipe1": graph_of(g4p(1))}
                layers = {"xqa_fp8": n8, "ours_fp8_pipe": n8, "fp4_pipe0": n4, "fp4_pipe1": n4}
                best = {}
                for rnd in range(3):
                    names = list(gs)
                    for n in names[rnd % len(names):] + names[:rnd % len(names)]:
                        g = gs[n]
                        g.replay()
                        torch.cuda.synchronize()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        ts = []
                        for _ in range(10):
                            e0.record()
                            g.replay()
                            e1.record()
                            e1.synchronize()
                            ts.append(e0.elapsed_time(e1) * 1000 / layers[n])
                        ts.sort()
                        best[n] = min(best.get(n, 1e30), ts[len(ts) // 2])
                log({"test": "G4_decode_speed", "B": B, "ctx": L, "QL": QL, "page_size": PS,
                     "rotated_bytes_fp8": n8 * fp8_bytes, "rotated_bytes_fp4": n4 * fp4_bytes,
                     **{f"{n}_us": round(t, 2) for n, t in best.items()},
                     "floor_fp8_us": round(B * L * KVH * 2 * D / 1.63e6, 2),
                     "floor_fp4_us": round(B * L * KVH * 2 * (D // 2 + D // 16) / 1.63e6, 2),
                     "fp4_best_vs_xqa": round(best["xqa_fp8"] / min(best["fp4_pipe0"], best["fp4_pipe1"]), 3),
                     "note": "fp8 timings over random fp8 caches; the XQA quality baseline is gated separately in "
                             "test_attn_v3"})
                del caches8, caches4, gs
                torch.cuda.empty_cache()
    # kv write: NVFP4 (had 1, search 1) vs vLLM's FP8 reshape_and_cache_flash
    from vllm import _custom_ops  # noqa: F401  (registers torch.ops._C_cache_ops)
    for NT in (1, 32, 8192):
        NP = math.ceil(NT / PS) + 3
        key = torch.randn(NT, KVH, D, device=dev).to(torch.bfloat16)
        val = torch.randn(NT, KVH, D, device=dev).to(torch.bfloat16)
        slots = torch.arange(NT, device=dev, dtype=torch.int64)
        c4 = R.alloc_cache(NP, PS, KVH, D, device=dev)
        v4 = R.split_views(c4, KVH)
        kv8 = torch.zeros(NP, KVH, PS, 2 * D, device=dev, dtype=torch.float8_e4m3fn)
        k8, v8 = kv8.permute(0, 2, 1, 3).split(D, dim=-1)
        one = torch.ones(1, device=dev)
        sg = R.hadamard_signs(D).to(dev)
        fns = {"fp4_write": lambda: KW.write(key, val, *v4, slots, one.expand(KVH), one.expand(KVH), sg),
               "fp8_reshape_and_cache_flash": lambda: torch.ops._C_cache_ops.reshape_and_cache_flash(
                   key, val, k8, v8, slots, "fp8", one, one)}
        res = {}
        for n, f in fns.items():
            g = graph_of(f)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(20):
                e0.record()
                g.replay()
                e1.record()
                e1.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000)
            res[n] = round(sorted(ts)[10], 2)
        log({"test": "G4_write_speed", "NT": NT, **{f"{n}_us": t for n, t in res.items()}})
    COMPLETE["G4"] = True


if __name__ == "__main__":
    log({"test": "configuration", "package": PKG, "quick": QUICK, "page_size": PS, "limit_rel_l2": LIMIT})
    g1()
    g2()
    g3()
    g4()
    log({"test": "complete", "complete": all(COMPLETE.values()), **COMPLETE, "deployment_eligible": False})
    print("TEST_KVFP4_GPU_DONE", flush=True)
