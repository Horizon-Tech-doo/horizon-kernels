# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for attn_decode_fp4_v2 (register-level FP4 unpack) against v1, our FP8 kernel and FlashInfer XQA.
Run with vLLM stopped:
    python test_kvfp4_gpu_v2.py OUT.jsonl [quick] --pkg-dir DIR
--pkg-dir DIR imports the candidate as package `hzkv` (staged copies); default tilelang.contrib.horizon_ops.

Gates (terminal record complete=true only if all ran; the 0.0025 limit is the same as test_kvfp4_gpu.py G3):
  G3v2 decode correctness of v2: relative L2 <= 0.0025 vs fp64 attention over the SAME dequantised cache
       (kv_fp4_ref.attention_reference with the kernel's own bf16 q'), dense (4 kv heads, GQA 6) and MoE (2, GQA 8)
       geometry, QL 1 and 4, Hadamard on / off, non-unit gk / gv, cache bytes outside written slots = 0x7f (NaN as
       e4m3: partial blocks must stay finite), configs NST 4 / 3 and min_chunk 64 / 32, default and fixed split
       counts; finite; deterministic (two runs bit-equal); two graph replays after NaN poisoning == eager.
       Also reported: max |v2 - v1| (not gated; summation order differs).
  G4v2 speed (report): graph replay, >512 MiB rotated KV copies PER FORMAT, 3 interleaved rounds, median of 10:
       XQA FP8 | our FP8 attn_decode (pipe1) | FP4 v1 (pipe1) | FP4 v2 variants (NST, CTAs per SM, split cap);
       B 1 / 8 / 32, context 1k .. 128k (quick: 1k, 32k), QL 4 and 1; FP8 / FP4 memory floors at 1.63 TB/s;
       rotate_q time for the same token count.
"""
import importlib
import json
import math
import sys

import torch

OUT = sys.argv[1]
QUICK = "quick" in sys.argv[2:]
if "--pkg-dir" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--pkg-dir") + 1])
    PFX = "hzkv"
else:
    PFX = "tilelang.contrib.horizon_ops"
KW = importlib.import_module(f"{PFX}.kv_write_fp4")
AD = importlib.import_module(f"{PFX}.attn_decode_fp4")
V2 = importlib.import_module(f"{PFX}.attn_decode_fp4_v2")
R = importlib.import_module(f"{PFX}.kv_fp4_ref")
A8 = importlib.import_module(f"{PFX}.attn_decode")

dev = torch.device("cuda")
PS, D = 64, 256
SCALE = D ** -0.5
LIMIT = 0.0025
WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
COMPLETE = {"G3v2": False, "G4v2": False}


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


def build(B, lens, QL, KVH, G, gen, had, gk, gv):
    pages = [math.ceil(L / PS) for L in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=gen) + 1
    bt = torch.zeros(B, max(pages), dtype=torch.int32)
    o = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[o:o + p].int()
        o += p
    slots = torch.tensor([int(bt[i, p // PS]) * PS + p % PS for i, L in enumerate(lens) for p in range(L)],
                         dtype=torch.int64)
    NT = sum(lens)
    key = torch.randn(NT, KVH, D, generator=gen)
    key[:, :, [0, 1, 64, 65, 126, 127]] = key[:, :, [0, 1, 64, 65, 126, 127]] * 6 + 10
    val = torch.randn(NT, KVH, D, generator=gen) * torch.exp(0.5 * torch.randn(NT, 1, 1, generator=gen))
    key, val = key.to(torch.bfloat16), val.to(torch.bfloat16)
    signs = R.hadamard_signs(D) if had else None
    cache = R.alloc_cache(NP, PS, KVH, D, device=dev, fill=0x7F)
    views = R.split_views(cache, KVH)
    KW.write(key.to(dev), val.to(dev), *views, slots.to(dev), gk.to(dev), gv.to(dev),
             signs.to(dev) if signs is not None else None)
    q = torch.randn(B * QL, KVH * G, D, generator=gen).to(torch.bfloat16).to(dev)
    qr = AD.rotate_q(q, signs.to(dev)) if had else q
    return dict(views=views, bt=bt.to(dev), sl=torch.tensor(lens, dtype=torch.int32, device=dev), qr=qr,
                gk=gk.to(dev), gv=gv.to(dev), QL=QL, lens=lens)


def g3():
    gen = torch.Generator().manual_seed(31)
    cases = [(1, [37], 1), (1, [1000], 4), (1, [1601], 4), (3, [5, 3300, 900], 4), (2, [16, 17], 1),
             (8, [4000] * 8, 4), (2, [20000, 7], 1), (1, [65536], 4)]
    if QUICK:
        cases = cases[:5]
    configs = [dict(nst=4, min_chunk=64), dict(nst=3, min_chunk=64), dict(nst=4, min_chunk=32)]
    for KVH, G in ((4, 6), (2, 8)):
        for B, lens, QL in cases:
            for had in (True, False):
                c = build(B, lens, QL, KVH, G, gen, had, torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH]),
                          torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH]))
                ref = R.attention_reference(c["qr"], c["views"], c["bt"], c["sl"], QL, SCALE, c["gk"], c["gv"])
                o1 = torch.empty_like(c["qr"])
                AD.decode(c["qr"], *c["views"], c["bt"], c["sl"], 0, SCALE, 1.0, QL, c["gk"], c["gv"], out=o1,
                          pipe=1)
                rec = {"test": "G3v2_decode", "KVH": KVH, "G": G, "B": B, "lens": lens[:4], "QL": QL, "had": had,
                       "v1_rel_l2": rel(o1, ref)}
                for cfg in (configs if (KVH, G) == (4, 6) else configs[:1]):
                    for splits in (None, 7):
                        tag = f"nst{cfg['nst']}_mc{cfg['min_chunk']}_s{splits or 'auto'}"
                        fn = lambda out: V2.decode(c["qr"], *c["views"], c["bt"], c["sl"], max(lens), SCALE, 1.0,  # noqa: E731
                                                   QL, c["gk"], c["gv"], out=out, splits=splits, **cfg)
                        a = torch.empty_like(c["qr"])
                        bb = torch.empty_like(c["qr"])
                        fn(a)
                        fn(bb)
                        torch.cuda.synchronize()
                        r = rel(a, ref)
                        fin = bool(torch.isfinite(a).all())
                        det = torch.equal(a, bb)
                        exp = a.clone()
                        gph = graph_of(lambda: fn(a))
                        rep = True
                        for _ in range(2):
                            a.fill_(float("nan"))
                            gph.replay()
                            torch.cuda.synchronize()
                            rep = rep and torch.equal(a, exp)
                        del gph
                        rec[tag] = {"rel_l2": r, "finite": fin, "deterministic": det, "graph_ok": rep,
                                    "max_abs_vs_v1": (exp.float() - o1.float()).abs().max().item()}
                        assert fin and det and rep and r <= LIMIT, f"G3v2 {tag}: {r}"
                log(rec)
    COMPLETE["G3v2"] = True


def g4():
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask
    gen = torch.Generator(device=dev)
    gen.manual_seed(4)
    KVH, G = 4, 6
    HQ = KVH * G
    ctxs = (1024, 32768) if QUICK else (1024, 4096, 16384, 32768, 131072)
    v2cfg = {"v2_nst4_cps4": dict(nst=4, ctas_per_sm=4), "v2_nst3_cps4": dict(nst=3, ctas_per_sm=4),
             "v2_nst4_cps2": dict(nst=4, ctas_per_sm=2), "v2_nst4_cps4_mc32": dict(nst=4, ctas_per_sm=4, min_chunk=32)}
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
                    for sf in (v[1], v[3]):
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

                def g4v1():
                    for v in caches4:
                        AD.decode(q, *v, bt, sl, L, SCALE, 1.0, QL, gk, gk, out=o, pipe=1)

                def mk(cfg):
                    splits = V2.plan_splits_v2(B, KVH, L, cfg.get("ctas_per_sm", 4), cfg.get("min_chunk", 64))

                    def f():
                        for v in caches4:
                            V2.decode(q, *v, bt, sl, L, SCALE, 1.0, QL, gk, gk, out=o, splits=splits, **cfg)
                    return f, splits
                gs = {"xqa_fp8": (graph_of(gx), n8), "ours_fp8_pipe": (graph_of(g8), n8),
                      "fp4_v1_pipe1": (graph_of(g4v1), n4)}
                splits_used = {}
                for name, cfg in v2cfg.items():
                    f, sp = mk(cfg)
                    gs[name] = (graph_of(f), n4)
                    splits_used[name] = sp
                sg = R.hadamard_signs(D).to(dev)
                qo = torch.empty_like(q)
                gs["rotate_q"] = (graph_of(lambda: AD.rotate_q(q, sg, qo)), 1)
                best = {}
                names = list(gs)
                for rnd in range(3):
                    for n in names[rnd % len(names):] + names[:rnd % len(names)]:
                        g, layers = gs[n]
                        g.replay()
                        torch.cuda.synchronize()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        ts = []
                        for _ in range(10):
                            e0.record()
                            g.replay()
                            e1.record()
                            e1.synchronize()
                            ts.append(e0.elapsed_time(e1) * 1000 / layers)
                        ts.sort()
                        best[n] = min(best.get(n, 1e30), ts[len(ts) // 2])
                f4 = B * L * KVH * 2 * (D // 2 + D // 16) / 1.63e6
                v2best = min(v for k, v in best.items() if k.startswith("v2_"))
                log({"test": "G4v2_decode_speed", "B": B, "ctx": L, "QL": QL, "page_size": PS,
                     "rotated_bytes_fp8": n8 * fp8_bytes, "rotated_bytes_fp4": n4 * fp4_bytes,
                     **{f"{n}_us": round(t, 2) for n, t in best.items()}, "v2_splits": splits_used,
                     "floor_fp8_us": round(B * L * KVH * 2 * D / 1.63e6, 2), "floor_fp4_us": round(f4, 2),
                     "v2_best_vs_floor_fp4": round(v2best / f4, 3),
                     "v2_best_vs_xqa": round(best["xqa_fp8"] / v2best, 3),
                     "v2_best_vs_v1": round(best["fp4_v1_pipe1"] / v2best, 3)})
                del caches8, caches4, gs
                torch.cuda.empty_cache()
    COMPLETE["G4v2"] = True


if __name__ == "__main__":
    log({"test": "configuration", "package": PFX, "quick": QUICK, "page_size": PS, "limit_rel_l2": LIMIT})
    g3()
    g4()
    log({"test": "complete", "complete": all(COMPLETE.values()), **COMPLETE, "deployment_eligible": False})
    print("TEST_KVFP4_GPU_V2_DONE", flush=True)
