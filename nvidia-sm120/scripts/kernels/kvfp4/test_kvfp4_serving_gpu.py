# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for the serving kernels (kv_fp4_serving) and the backend's prefill scheme, standalone (vLLM stopped).
    python test_kvfp4_serving_gpu.py OUT.jsonl --pkg-dir DIR     (DIR/horizon_kvfp4/kernels = the kernel copies)
The cache is built exactly like HorizonFp4Backend sees it: vLLM's logical (blocks, 2*KVH, PS, 144) uint8 tensor, in
both physical layouts (head-major HND and token-major NHD), slot = [128 data | 16 scales], written with kv_write_fp4.
  S1 decode_ragged: mixed query lengths in one batch (4, 2, 1, zero-length padding, a 9-token prefill row that must
     be skipped), several pages, relative L2 <= 0.0025 vs the fp64 reference over the same dequantised cache for every
     decode row; rows of skipped / padding requests untouched; deterministic; two graph replays == eager
  S2 dequant_tokens: bit-exact vs kv_fp4_ref.dequantize_rows for spans crossing pages, both layouts
  S4 CUDA write kernel (kv_fp4_serving.write): cache bytes == kv_fp4_ref.write_reference, key / value as strided
     slices of a qkv tensor, random / outlier / edge tokens, search on / off, padded slots, both layouts, graph replay
  S5 batched prefill: dequant_multi bit-exact + FlashInfer ragged FA2 over several interleaved prefill rows, relative
     L2 <= 0.0025 vs fp64 in fp16 (the backend's prefill dtype); bf16 reported
  S3 chunked prefill (HorizonFp4Impl._prefill_rows, contexts that do not fit the batched scratch): dequantised chunks
     + FlashInfer FA2 single prefill + merge_state in fp16, the backend's 32k / 64k chunks at 0.4k .. 130k context
     gated at relative L2 <= 0.0025 vs fp64; tiny chunks with many merges reported.
     (29-09-2026 t6, bf16: one FA2 call 0.0021..0.0026, 2..3 chunks 0.0026..0.0029: FA2's bf16 P / outputs, hence fp16)
"""
import importlib
import json
import math
import sys

import torch

OUT = sys.argv[1]
sys.path.insert(0, sys.argv[sys.argv.index("--pkg-dir") + 1])
SV = importlib.import_module("horizon_kvfp4.kernels.kv_fp4_serving")
KW = importlib.import_module("horizon_kvfp4.kernels.kv_write_fp4")
R = importlib.import_module("horizon_kvfp4.kernels.kv_fp4_ref")
AD = importlib.import_module("horizon_kvfp4.kernels.attn_decode_fp4")
dev = torch.device("cuda")
D, PS, LIMIT = 256, 64, 0.0025
DONE = {}


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


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


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def make_cache(NP, KVH, layout):
    if layout == "HND":
        kv = torch.full((NP, 2 * KVH, PS, 144), 0x7F, dtype=torch.uint8, device=dev)
    else:
        kv = torch.full((NP, PS, 2 * KVH, 144), 0x7F, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
    t = kv.permute(0, 2, 1, 3)
    views = (t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:])
    flat, strides = SV.flat_views(*views)
    return kv, views, flat, strides


def build(lens, qls, KVH, G, layout, gen, signs=None):
    B = len(lens)
    pages = [max(1, math.ceil(L / PS)) for L in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=gen) + 1
    bt = torch.zeros(B, max(pages) + 2, dtype=torch.int32)
    o = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[o:o + p].int()
        o += p
    kv, views, flat, strides = make_cache(NP, KVH, layout)
    slots = torch.tensor([int(bt[i, p // PS]) * PS + p % PS for i, L in enumerate(lens) for p in range(L)],
                         dtype=torch.int64)
    NT = sum(lens)
    key = torch.randn(NT, KVH, D, generator=gen)
    key[:, :, :4] = key[:, :, :4] * 6 + 10
    val = torch.randn(NT, KVH, D, generator=gen)
    key, val = key.to(torch.bfloat16).to(dev), val.to(torch.bfloat16).to(dev)
    gk = torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH], device=dev)
    gv = torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH], device=dev)
    KW.write(key, val, *views, slots.to(dev), gk, gv, None if signs is None else signs.to(dev), True)
    qsl = [0]
    for ql in qls:
        qsl.append(qsl[-1] + ql)
    q = torch.randn(qsl[-1], KVH * G, D, generator=gen).to(torch.bfloat16).to(dev)
    return dict(views=views, flat=flat, strides=strides, bt=bt.to(dev), sl=torch.tensor(lens, dtype=torch.int32,
                device=dev), qsl=torch.tensor(qsl, dtype=torch.int32, device=dev), qsl_l=qsl, q=q, gk=gk, gv=gv,
                lens=lens, qls=qls, KVH=KVH, G=G, signs=signs)


def reference_rows(c, i, sm):
    """fp64 attention of request i's query rows over its dequantised cache (causal among its own query tokens)"""
    KVH, G = c["KVH"], c["G"]
    L, ql, q0 = c["lens"][i], c["qls"][i], c["qsl_l"][i]
    K, V = c.get("gather", R.gather)(c["views"], c["bt"][i], L)
    out = torch.empty(ql, KVH * G, D, dtype=torch.float64, device=dev)
    for t in range(ql):
        lim = L - ql + t + 1
        qq = c["q"][q0 + t].double().view(KVH, G, D)
        s = torch.einsum("hgd,lhd->hgl", qq, K[:lim]) * c["gk"].double().view(KVH, 1, 1) * sm
        p = torch.softmax(s, -1)
        out[t] = torch.einsum("hgl,lhd->hgd", p, V[:lim] * c["gv"].double().view(1, KVH, 1)).reshape(-1, D)
    return out


def s1():
    gen = torch.Generator().manual_seed(51)
    sm = D ** -0.5
    ws = torch.empty(8 * 1024 * 1024, dtype=torch.float32, device=dev)
    for KVH, G in ((4, 6), (2, 8)):
        for layout in ("HND", "NHD"):
            lens = [300, 17, 1601, 5, 0, 2100, 64]
            qls = [4, 2, 1, 4, 0, 9, 1]                    # request 4: CUDA-graph padding, request 5: prefill row
            c = build(lens, qls, KVH, G, layout, gen)
            # (qlmax, nst, min_chunk, token groups, splits): splits None = the default plan (26 / 53 here); 1 = the
            # single-split path (the decode kernel writes the output, no combine), as the backend runs from ~24 requests
            for qlmax, nst, minc, tw, spl in ((4, 3, 64, 1, None), (4, 3, 128, 1, None), (2, 3, 128, 1, None),
                                              (1, 3, 128, 1, None), (4, 3, 256, 1, None), (2, 3, 256, 1, None),
                                              (1, 3, 256, 1, None), (4, 3, 128, 2, None), (2, 3, 128, 2, None),
                                              (1, 3, 128, 2, None), (4, 2, 64, 4, None), (2, 2, 128, 4, None),
                                              (1, 2, 256, 4, None), (1, 3, 64, 4, None), (4, 2, 128, 4, 1),
                                              (2, 2, 128, 4, 1), (1, 2, 128, 4, 1), (4, 3, 256, 1, 1), (1, 3, 128, 2, 1),
                                              (2, 2, 128, 4, 3)):
                if SV.smem_bytes(G, qlmax, SV.default_ncw(G, qlmax), nst, tw) > 99 * 1024:
                    continue                               # does not fit one SM (MoE geometry, QL4, 4 groups)
                # output longer than the query (vLLM passes its padded output buffer): the padding rows stay untouched
                out = torch.full((c["q"].shape[0] + 3, *c["q"].shape[1:]), float("nan"), dtype=c["q"].dtype, device=dev)
                fn = lambda: SV.decode_ragged(c["q"], c["flat"], c["strides"], PS, c["bt"], c["sl"], c["qsl"], sm,  # noqa: E731
                                              1.0, c["gk"], c["gv"], qlmax, out, ws, nst=nst, min_chunk=minc, tw=tw,
                                              splits=spl)
                fn()
                torch.cuda.synchronize()
                first = out.clone()
                out.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                det = torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                g = graph_of(fn)
                rep = True
                for _ in range(2):
                    out.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    rep = rep and torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                del g
                worst, skipped_ok = 0.0, True
                for i in range(len(lens)):
                    q0, q1 = c["qsl_l"][i], c["qsl_l"][i + 1]
                    if qls[i] == 0:
                        continue
                    if qls[i] > qlmax:
                        skipped_ok = skipped_ok and bool(torch.isnan(first[q0:q1]).all())
                        continue
                    worst = max(worst, rel(first[q0:q1], reference_rows(c, i, sm)))
                skipped_ok = skipped_ok and bool(torch.isnan(first[c["q"].shape[0]:]).all())
                ok = worst <= LIMIT and det and rep and skipped_ok
                log({"test": "S1_decode_ragged", "KVH": KVH, "G": G, "layout": layout, "qlmax": qlmax, "nst": nst,
                     "min_chunk": minc, "tw": tw, "splits": spl,
                     "worst_rel_l2": worst, "deterministic": det, "graph_ok": rep, "prefill_row_untouched": skipped_ok,
                     "ok": ok})
                assert ok, "S1"
    DONE["S1"] = True


def s2():
    gen = torch.Generator().manual_seed(52)
    for layout in ("HND", "NHD"):
        c = build([1000, 300], [1, 1], 4, 6, layout, gen)
        ok_all = True
        for i, (a, n) in enumerate(((0, 1000), (63, 130), (999, 1), (100, 200))):
            row = 0 if a + n <= 1000 else 1
            ok_ = torch.empty(n, 4, D, dtype=torch.bfloat16, device=dev)
            ov = torch.empty_like(ok_)
            SV.dequant_tokens(c["flat"], c["strides"], PS, c["bt"][row], a, n, ok_, ov)
            K, V = R.gather(c["views"], c["bt"][row], a + n)
            ok_all = ok_all and torch.equal(ok_.double(), K[a:a + n]) and torch.equal(ov.double(), V[a:a + n])
        log({"test": "S2_dequant_exact", "layout": layout, "ok": ok_all})
        assert ok_all, "S2"
    DONE["S2"] = True


def s3(pdt=torch.float16):
    """chunked prefill: chunks + FA2 + merge_state in pdt (fp16 = the backend default). Gate (<= 0.0025) on the chunk
    sizes the backend uses (HORIZON_KVFP4_CHUNK 32k / 64k); tiny chunks (many merges) are reported only."""
    from flashinfer import merge_state, single_prefill_with_kv_cache
    gen = torch.Generator().manual_seed(53)
    sm = D ** -0.5
    worst = 0.0
    cases = [(1500, 300, 256, False), (2000, 37, 512, False), (400, 100, 32768, True), (30000, 1024, 32768, True),
             (40000, 2048, 32768, True), (70000, 4096, 32768, True), (70000, 4096, 65536, True),
             (130000, 2048, 65536, True)]
    for layout in ("HND", "NHD"):
        for L, ql, chunk, gated in cases:
            if layout == "NHD" and L > 70000:
                continue
            c = build([L], [ql], 4, 6, layout, gen)
            sk = torch.empty(max(chunk, ql), 4, D, dtype=pdt, device=dev)
            sv = torch.empty_like(sk)
            qi = c["q"].to(pdt)
            tail = max(0, L - max(chunk, ql))
            spans = [(a, min(a + chunk, tail)) for a in range(0, tail, chunk)] + [(tail, L)]
            o = s = None
            for a, b in spans:
                kk, vv = SV.dequant_tokens(c["flat"], c["strides"], PS, c["bt"][0], a, b - a, sk, sv)
                oc, lc = single_prefill_with_kv_cache(qi, kk, vv, causal=(b == L), sm_scale=sm, backend="fa2",
                                                      return_lse=True)
                o, s = (oc, lc) if o is None else merge_state(o, s, oc, lc)
            cu = dict(c, gk=torch.ones(4, device=dev), gv=torch.ones(4, device=dev))
            r = rel(o.to(torch.bfloat16), reference_rows(cu, 0, sm))        # the backend stores bf16
            if gated:
                worst = max(worst, r)
            log({"test": "S3_prefill_chunks", "dtype": str(pdt)[6:], "layout": layout, "L": L, "ql": ql, "chunk": chunk,
                 "chunks": len(spans), "rel_l2": r, "gated": gated, "ok": (r <= LIMIT) or not gated})
            del c, sk, sv
            torch.cuda.empty_cache()
    log({"test": "S3_summary", "dtype": str(pdt)[6:], "worst_gated": worst, "ok": worst <= LIMIT})
    assert worst <= LIMIT, "S3"
    DONE["S3"] = True


def s4(had=False):
    """CUDA write kernel (kv_fp4_serving.write, 4 threads per block): cache bytes == kv_fp4_ref.write_reference.
    had=True (S6): keys rotated inside the kernel (fused Hadamard) == write_reference with the same signs"""
    gen = torch.Generator().manual_seed(56 if had else 54)
    signs = R.hadamard_signs(D) if had else None
    sd = signs.to(dev) if had else None
    for layout in ("HND", "NHD"):
        for kind in ("random", "outlier", "edge"):
            for search in (True, False):
                NT, KVH, NP = 300, 4, 12
                kv, views, flat, strides = make_cache(NP, KVH, layout)
                ref_kv = kv.cpu().clone()
                t = ref_kv.permute(0, 2, 1, 3)
                ref_views = (t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:])
                slots = torch.randperm(NP * PS, generator=gen)[:NT].to(torch.int64)
                slots[torch.rand(NT, generator=gen) < 0.1] = -1
                qkv = torch.randn(NT, 3 * KVH * D, generator=gen)          # key / value as strided qkv slices
                if kind == "outlier":
                    qkv[:, :8] = qkv[:, :8] * 6 + 10
                elif kind == "edge":
                    qkv[0] = 0
                    qkv[1] *= 1e-30
                    qkv[2] *= 1e30
                qkv = qkv.to(torch.bfloat16)
                key = qkv[:, KVH * D:2 * KVH * D].view(NT, KVH, D)
                val = qkv[:, 2 * KVH * D:].view(NT, KVH, D)
                gk = torch.tensor([1.0, 0.5, 2.0, 1.0])
                gv = torch.tensor([1.0, 1.0, 0.25, 4.0])
                R.write_reference(key, val, ref_views, slots, gk, gv, signs, search)
                qd = qkv.to(dev)
                kd_, vd_ = qd[:, KVH * D:2 * KVH * D].view(NT, KVH, D), qd[:, 2 * KVH * D:].view(NT, KVH, D)
                sd_, gkd, gvd = slots.to(dev), gk.to(dev), gv.to(dev)     # host copies stay outside the graph capture
                fn = lambda: SV.write(kd_, vd_, flat, strides, PS, sd_, gkd, gvd, search, sd)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                exact = torch.equal(kv.cpu(), ref_kv)
                g = graph_of(fn)
                kv.fill_(0x7F)
                g.replay()
                torch.cuda.synchronize()
                rep = torch.equal(kv.cpu(), ref_kv)
                del g
                log({"test": "S6_write_had_exact" if had else "S4_write_exact", "layout": layout, "kind": kind,
                     "search": search, "exact": exact,
                     "graph_ok": rep, "mismatched_bytes": int((kv.cpu() != ref_kv).sum())})
                assert exact and rep, "S6" if had else "S4"
    DONE["S6" if had else "S4"] = True


def s7():
    """fused-Hadamard ragged decode: keys written rotated; decode_ragged(q, signs) must equal, BITWISE, the separate
    path rotate_q(q) + decode_ragged(q', no signs) (the kernel rotates q in shared memory with the reference butterfly
    order and one bf16 rounding), be deterministic and replay; and be within 0.0025 of the fp64 reference computed
    with the reference-rotated query over the dequantised (rotated) keys"""
    gen = torch.Generator().manual_seed(57)
    sm = D ** -0.5
    signs = R.hadamard_signs(D)
    sd = signs.to(dev)
    ws = torch.empty(8 * 1024 * 1024, dtype=torch.float32, device=dev)
    for KVH, G in ((4, 6), (2, 8)):
        for layout in ("HND", "NHD"):
            lens = [300, 17, 1601, 5, 0, 2100, 64]
            qls = [4, 2, 1, 4, 0, 9, 1]
            c = build(lens, qls, KVH, G, layout, gen, signs)
            q_rot = AD.rotate_q(c["q"], sd)
            ref_rot = R.rotate_q(c["q"].cpu(), signs).to(dev)
            rot_exact = torch.equal(q_rot, ref_rot)
            cref = dict(c, q=ref_rot)
            for qlmax, nst, minc, tw, spl in ((4, 3, 128, 1, None), (4, 2, 128, 4, 1), (2, 2, 128, 4, None),
                                              (1, 2, 128, 4, 1), (1, 3, 256, 1, None), (4, 3, 256, 1, 1),
                                              (2, 3, 128, 2, 3)):
                if SV.smem_bytes(G, qlmax, SV.default_ncw(G, qlmax), nst, tw, 1) > 99 * 1024:
                    continue
                NT = c["q"].shape[0]
                out = torch.full((NT + 3, *c["q"].shape[1:]), float("nan"), dtype=c["q"].dtype, device=dev)
                sep = torch.full_like(out, float("nan"))
                args = (c["flat"], c["strides"], PS, c["bt"], c["sl"], c["qsl"], sm, 1.0, c["gk"], c["gv"], qlmax)
                fn = lambda: SV.decode_ragged(c["q"], *args, out, ws, nst=nst, min_chunk=minc, tw=tw,  # noqa: E731
                                              splits=spl, signs=sd)
                SV.decode_ragged(q_rot, *args, sep, ws, nst=nst, min_chunk=minc, tw=tw, splits=spl)
                fn()
                torch.cuda.synchronize()
                first = out.clone()
                same_sep = torch.equal(torch.nan_to_num(first, 7.0), torch.nan_to_num(sep, 7.0))
                out.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                det = torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                g = graph_of(fn)
                rep = True
                for _ in range(2):
                    out.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    rep = rep and torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                del g
                worst = 0.0
                for i in range(len(lens)):
                    q0, q1 = c["qsl_l"][i], c["qsl_l"][i + 1]
                    if qls[i] == 0 or qls[i] > qlmax:
                        continue
                    worst = max(worst, rel(first[q0:q1], reference_rows(cref, i, sm)))
                ok = rot_exact and same_sep and det and rep and worst <= LIMIT
                log({"test": "S7_decode_had", "KVH": KVH, "G": G, "layout": layout, "qlmax": qlmax, "nst": nst,
                     "min_chunk": minc, "tw": tw, "splits": spl, "rotate_q_exact": rot_exact,
                     "bitwise_vs_separate_rotation": same_sep, "worst_rel_l2": worst, "deterministic": det,
                     "graph_ok": rep, "ok": ok})
                assert ok, "S7"
    DONE["S7"] = True


def s9():
    """batched-prefill workspace (29-09-2026 A/B crash): 8 new prompts of 64..129 tokens in one step made FlashInfer's
    split-KV plan ask for a 138 MB tmp_v in the old 128 MB workspace. Records whether 128 MB still overflows for this
    shape (reported), then plans + runs with the backend's workspace size (HORIZON_KVFP4_PREFILL_WS_MB, default 512)
    and checks the fp16 result against fp64 (gate 0.0025)"""
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    gen = torch.Generator().manual_seed(59)
    sm = D ** -0.5
    lens = [102, 64, 93, 102, 85, 129, 100, 93]
    c = build(lens, lens, 4, 6, "HND", gen)
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    N = cu[-1]
    qo = torch.tensor(cu, dtype=torch.int32)

    def plan(mb):
        ws = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=dev)
        w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        w.plan(qo, qo, 24, 4, D, causal=True, sm_scale=sm, q_data_type=torch.float16, kv_data_type=torch.float16)
        return w, ws
    try:
        plan(128)
        old_overflows = False
    except Exception as e:  # noqa: BLE001
        old_overflows = "overflow" in str(e).lower()
    import os
    mb = int(os.environ.get("HORIZON_KVFP4_PREFILL_WS_MB", "512"))
    w, ws = plan(mb)
    sk = torch.empty(N, 4, D, dtype=torch.float16, device=dev)
    sv = torch.empty_like(sk)
    rows = torch.arange(len(lens), dtype=torch.int32, device=dev)
    SV.dequant_multi(c["flat"], c["strides"], PS, c["bt"], rows, qo.to(dev), len(lens), N, sk, sv)
    o = w.run(c["q"].to(torch.float16), sk, sv).to(torch.bfloat16)
    cref = dict(c, gk=torch.ones(4, device=dev), gv=torch.ones(4, device=dev))
    rels = [rel(o[cu[i]:cu[i + 1]], reference_rows(cref, i, sm)) for i in range(len(lens))]
    ok = max(rels) <= LIMIT
    log({"test": "S9_prefill_workspace", "workspace_mb": mb, "old_128mb_overflows": old_overflows,
         "worst_rel_l2": max(rels), "ok": ok})
    assert ok, "S9"
    DONE["S9"] = True


# ------------------------------------------------------------------------------ K8V4: FP8 keys + FP4 values (30-09-2026)
def make_cache_k8(NP, KVH, layout):
    """vLLM's logical (pages, KVH, PS, 400) uint8 K8V4 block in both physical layouts"""
    if layout == "HND":
        kv = torch.full((NP, KVH, PS, R.K8_SLOT), 0x7F, dtype=torch.uint8, device=dev)
    else:
        kv = torch.full((NP, PS, KVH, R.K8_SLOT), 0x7F, dtype=torch.uint8, device=dev).permute(0, 2, 1, 3)
    kd, vd, vsf = R.split_views_k8(kv)
    flat, strides = SV.flat_views(kd, vsf, vd, vsf)          # ksf slot unused by the K8 kernels (pass vsf)
    return kv, (kd, vd, vsf), flat, strides


def kv_values(NT, KVH, gen, kind="real"):
    """keys with large-magnitude channels, some beyond e4m3's 448 at k_scale 1 (saturation) and some tiny ones
    (e4m3 subnormals); values ordinary"""
    key = torch.randn(NT, KVH, D, generator=gen)
    key[:, :, :4] = key[:, :, :4] * 6 + 10
    if kind == "edge":
        key[:, :, 8:12] *= 200.0
        key[:, :, 12:16] *= 1e-3
        key[0] = 0
    val = torch.randn(NT, KVH, D, generator=gen)
    return key.to(torch.bfloat16), val.to(torch.bfloat16)


def build_k8(lens, qls, KVH, G, layout, gen, gk=None):
    """paged K8V4 cache written by the CPU reference (independent of the kernel under test)"""
    B = len(lens)
    pages = [max(1, math.ceil(L / PS)) for L in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=gen) + 1
    bt = torch.zeros(B, max(pages) + 2, dtype=torch.int32)
    o = 0
    for i, p in enumerate(pages):
        bt[i, :p] = perm[o:o + p].int()
        o += p
    kv, views, flat, strides = make_cache_k8(NP, KVH, layout)
    slots = torch.tensor([int(bt[i, p // PS]) * PS + p % PS for i, L in enumerate(lens) for p in range(L)],
                         dtype=torch.int64)
    key, val = kv_values(sum(lens), KVH, gen, "edge")
    gk = gk if gk is not None else torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH])
    gv = torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH])
    cpu = kv.cpu().clone()
    R.write_reference_k8(key, val, R.split_views_k8(cpu), slots, gk, gv, True)
    kv.copy_(cpu)
    qsl = [0]
    for ql in qls:
        qsl.append(qsl[-1] + ql)
    q = torch.randn(qsl[-1], KVH * G, D, generator=gen).to(torch.bfloat16).to(dev)
    return dict(views=views, flat=flat, strides=strides, bt=bt.to(dev), sl=torch.tensor(lens, dtype=torch.int32,
                device=dev), qsl=torch.tensor(qsl, dtype=torch.int32, device=dev), qsl_l=qsl, q=q, gk=gk.to(dev),
                gv=gv.to(dev), lens=lens, qls=qls, KVH=KVH, G=G, gather=R.gather_k8, kv=kv)


def s10():
    """K8V4 write (kv_fp4_serving.write, k8=True): cache bytes == write_reference_k8 (both layouts, search on / off,
    padded slots, graph replay), and the K bytes == vLLM's own fp8 reshape_and_cache_flash (production's K cache)"""
    from vllm import _custom_ops  # noqa: F401
    gen = torch.Generator().manual_seed(60)
    for layout in ("HND", "NHD"):
        for kind in ("real", "edge"):
            for search in (True, False):
                NT, KVH, NP = 300, 4, 12
                kv, views, flat, strides = make_cache_k8(NP, KVH, layout)
                ref = kv.cpu().clone()
                slots = torch.randperm(NP * PS, generator=gen)[:NT].to(torch.int64)
                slots[torch.rand(NT, generator=gen) < 0.1] = -1
                key, val = kv_values(NT, KVH, gen, kind)
                qkv = torch.cat([torch.randn(NT, KVH * D, generator=gen).to(torch.bfloat16),
                                 key.reshape(NT, -1), val.reshape(NT, -1)], dim=1)
                gk = torch.tensor([1.0, 0.5, 2.0, 1.0])
                gv = torch.tensor([1.0, 1.0, 0.25, 4.0])
                R.write_reference_k8(key, val, R.split_views_k8(ref), slots, gk, gv, search)
                qd = qkv.to(dev)
                kd_, vd_ = qd[:, KVH * D:2 * KVH * D].view(NT, KVH, D), qd[:, 2 * KVH * D:].view(NT, KVH, D)
                sd_, gkd, gvd = slots.to(dev), gk.to(dev), gv.to(dev)
                fn = lambda: SV.write(kd_, vd_, flat, strides, PS, sd_, gkd, gvd, search, None, True)  # noqa: E731
                fn()
                torch.cuda.synchronize()
                exact = torch.equal(kv.cpu(), ref)
                g = graph_of(fn)
                kv.fill_(0x7F)
                g.replay()
                torch.cuda.synchronize()
                rep = torch.equal(kv.cpu(), ref)
                del g
                # production's fp8 K bytes: vLLM reshape_and_cache_flash into an FP8 cache, per head with its k_scale
                k8v = torch.zeros(NP, PS, KVH, 2 * D, dtype=torch.float8_e4m3fn, device=dev)
                vllm_same = True
                for h in range(KVH):
                    kc, vc = k8v[:, :, h:h + 1, :D], k8v[:, :, h:h + 1, D:]
                    one = torch.ones(1, device=dev)
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        kd_[:, h:h + 1].contiguous(), vd_[:, h:h + 1].contiguous(), kc, vc, sd_, "fp8",
                        gk[h:h + 1].to(dev), one)
                torch.cuda.synchronize()
                kk = k8v[..., :D].view(torch.uint8).cpu()
                ours = R.split_views_k8(kv.cpu())[0]
                for t, sl in enumerate(slots.tolist()):
                    if sl >= 0:
                        vllm_same = vllm_same and torch.equal(kk[sl // PS, sl % PS], ours[sl // PS, sl % PS])
                ok = exact and rep and vllm_same
                log({"test": "S10_write_k8_exact", "layout": layout, "kind": kind, "search": search, "exact": exact,
                     "graph_ok": rep, "k_bytes_equal_vllm_fp8": vllm_same,
                     "mismatched_bytes": int((kv.cpu() != ref).sum()), "ok": ok})
                assert ok, "S10"
    DONE["S10"] = True


def s11():
    """K8V4 ragged decode (k8=True): every decode row within 0.0025 of the fp64 reference over the same dequantised
    cache, skipped / padding rows untouched, deterministic, two graph replays == eager (as S1)"""
    gen = torch.Generator().manual_seed(61)
    sm = D ** -0.5
    ws = torch.empty(8 * 1024 * 1024, dtype=torch.float32, device=dev)
    for KVH, G in ((4, 6), (2, 8)):
        for layout in ("HND", "NHD"):
            lens = [300, 17, 1601, 5, 0, 2100, 64]
            qls = [4, 2, 1, 4, 0, 9, 1]
            c = build_k8(lens, qls, KVH, G, layout, gen)
            for qlmax, nst, minc, tw, spl in ((4, 3, 64, 1, None), (4, 3, 256, 1, None), (2, 3, 128, 2, None),
                                              (1, 3, 128, 1, None), (4, 2, 128, 4, 1), (2, 2, 128, 4, None),
                                              (1, 2, 128, 4, 1), (4, 3, 256, 1, 1), (2, 2, 128, 4, 3)):
                if SV.smem_bytes(G, qlmax, SV.default_ncw(G, qlmax), nst, tw, 0, 1) > 99 * 1024:
                    continue
                NT = c["q"].shape[0]
                out = torch.full((NT + 3, *c["q"].shape[1:]), float("nan"), dtype=c["q"].dtype, device=dev)
                fn = lambda: SV.decode_ragged(c["q"], c["flat"], c["strides"], PS, c["bt"], c["sl"], c["qsl"],  # noqa: E731
                                              sm, 1.0, c["gk"], c["gv"], qlmax, out, ws, nst=nst, min_chunk=minc,
                                              tw=tw, splits=spl, k8=True)
                fn()
                torch.cuda.synchronize()
                first = out.clone()
                out.fill_(float("nan"))
                fn()
                torch.cuda.synchronize()
                det = torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                g = graph_of(fn)
                rep = True
                for _ in range(2):
                    out.fill_(float("nan"))
                    g.replay()
                    torch.cuda.synchronize()
                    rep = rep and torch.equal(torch.nan_to_num(out, 7.0), torch.nan_to_num(first, 7.0))
                del g
                worst, skipped_ok = 0.0, bool(torch.isnan(first[NT:]).all())
                for i in range(len(lens)):
                    q0, q1 = c["qsl_l"][i], c["qsl_l"][i + 1]
                    if qls[i] == 0:
                        continue
                    if qls[i] > qlmax:
                        skipped_ok = skipped_ok and bool(torch.isnan(first[q0:q1]).all())
                        continue
                    worst = max(worst, rel(first[q0:q1], reference_rows(c, i, sm)))
                ok = worst <= LIMIT and det and rep and skipped_ok
                log({"test": "S11_decode_k8", "KVH": KVH, "G": G, "layout": layout, "qlmax": qlmax, "nst": nst,
                     "min_chunk": minc, "tw": tw, "splits": spl, "worst_rel_l2": worst, "deterministic": det,
                     "graph_ok": rep, "untouched_ok": skipped_ok, "ok": ok})
                assert ok, "S11"
    DONE["S11"] = True


def s12():
    """K8V4 dequant (prefill): dequant_tokens and dequant_multi bit-exact vs gather_k8, bf16 and fp16 outputs"""
    gen = torch.Generator().manual_seed(62)
    ok_all = True
    for layout in ("HND", "NHD"):
        c = build_k8([1000, 300, 77], [1, 1, 1], 4, 6, layout, gen)
        for dt in (torch.bfloat16, torch.float16):
            for a, n in ((0, 1000), (63, 130), (999, 1), (100, 200)):
                ok_ = torch.empty(n, 4, D, dtype=dt, device=dev)
                ov = torch.empty_like(ok_)
                SV.dequant_tokens(c["flat"], c["strides"], PS, c["bt"][0], a, n, ok_, ov, k8=True)
                K, V = R.gather_k8(c["views"], c["bt"][0], a + n)
                ok_all = ok_all and torch.equal(ok_.double(), K[a:a + n]) and torch.equal(ov.double(), V[a:a + n])
            cu = [0, 1000, 1300, 1377]
            sk = torch.empty(1377, 4, D, dtype=dt, device=dev)
            sv = torch.empty_like(sk)
            SV.dequant_multi(c["flat"], c["strides"], PS, c["bt"], torch.tensor([0, 1, 2], dtype=torch.int32, device=dev),
                             torch.tensor(cu, dtype=torch.int32, device=dev), 3, 1377, sk, sv, k8=True)
            for r, L in enumerate((1000, 300, 77)):
                K, V = R.gather_k8(c["views"], c["bt"][r], L)
                ok_all = ok_all and torch.equal(sk[cu[r]:cu[r + 1]].double(), K) and torch.equal(sv[cu[r]:cu[r + 1]].double(), V)
        log({"test": "S12_dequant_k8_exact", "layout": layout, "ok": ok_all})
        assert ok_all, "S12"
    DONE["S12"] = True


def s13():
    """K8V4 batched fp16 prefill (the backend's path): dequant_multi (k8) + FlashInfer ragged FA2 over interleaved
    prefill rows, within 0.0025 of fp64 (k_scale folded into sm_scale as in the backend)"""
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    gen = torch.Generator().manual_seed(63)
    sm = D ** -0.5
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    gk1 = torch.ones(4)
    for layout in ("HND", "NHD"):
        lens = [700, 3, 1500, 4, 300, 5000]
        qls = [700, 2, 200, 4, 37, 4096]
        c = build_k8(lens, qls, 4, 6, layout, gen, gk=gk1)
        pre = [i for i in range(len(lens)) if qls[i] > 4]
        cu, qo = [0], [0]
        for i in pre:
            cu.append(cu[-1] + lens[i])
            qo.append(qo[-1] + qls[i])
        N = cu[-1]
        sk = torch.empty(N, 4, D, dtype=torch.float16, device=dev)
        sv = torch.empty_like(sk)
        SV.dequant_multi(c["flat"], c["strides"], PS, c["bt"], torch.tensor(pre, dtype=torch.int32, device=dev),
                         torch.tensor(cu, dtype=torch.int32, device=dev), len(pre), N, sk, sv, k8=True)
        idx = torch.cat([torch.arange(c["qsl_l"][i], c["qsl_l"][i + 1]) for i in pre]).to(dev)
        w.plan(torch.tensor(qo, dtype=torch.int32), torch.tensor(cu, dtype=torch.int32), 24, 4, D, causal=True,
               sm_scale=sm, q_data_type=torch.float16, kv_data_type=torch.float16)
        o = w.run(c["q"].index_select(0, idx).to(torch.float16), sk, sv).to(torch.bfloat16)
        cref = dict(c, gv=torch.ones(4, device=dev))
        rels = [rel(o[qo[r]:qo[r + 1]], reference_rows(cref, i, sm)) for r, i in enumerate(pre)]
        ok = max(rels) <= LIMIT
        log({"test": "S13_prefill_k8", "layout": layout, "rel_l2": rels, "ok": ok})
        assert ok, "S13"
    DONE["S13"] = True


def s8():
    """fused-Hadamard prefill: rotate_q straight to fp16 == fp32 reference rotation rounded once to fp16 (bitwise), and
    the batched fp16 prefill over rotated keys within 0.0025 of the fp64 reference"""
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    gen = torch.Generator().manual_seed(58)
    sm = D ** -0.5
    signs = R.hadamard_signs(D)
    sd = signs.to(dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    for layout in ("HND", "NHD"):
        lens = [700, 3, 1500, 300, 5000]
        qls = [700, 2, 200, 37, 4096]
        c = build(lens, qls, 4, 6, layout, gen, signs)
        pre = [i for i in range(len(lens)) if qls[i] > 4]
        q16 = AD.rotate_q(c["q"], sd, dtype=torch.float16)
        rot_exact = torch.equal(q16.cpu(), R.rotate(c["q"].cpu(), signs).to(torch.float16))
        cu, qo = [0], [0]
        for i in pre:
            cu.append(cu[-1] + lens[i])
            qo.append(qo[-1] + qls[i])
        N = cu[-1]
        sk = torch.empty(N, 4, D, dtype=torch.float16, device=dev)
        sv = torch.empty_like(sk)
        SV.dequant_multi(c["flat"], c["strides"], PS, c["bt"], torch.tensor(pre, dtype=torch.int32, device=dev),
                         torch.tensor(cu, dtype=torch.int32, device=dev), len(pre), N, sk, sv)
        idx = torch.cat([torch.arange(c["qsl_l"][i], c["qsl_l"][i + 1]) for i in pre]).to(dev)
        w.plan(torch.tensor(qo, dtype=torch.int32), torch.tensor(cu, dtype=torch.int32), 24, 4, D, causal=True,
               sm_scale=sm, q_data_type=torch.float16, kv_data_type=torch.float16)
        o = w.run(q16.index_select(0, idx), sk, sv).to(torch.bfloat16)
        cref = dict(c, q=R.rotate(c["q"].cpu(), signs).to(dev), gk=torch.ones(4, device=dev),
                    gv=torch.ones(4, device=dev))
        rels = [rel(o[qo[r]:qo[r + 1]], reference_rows(cref, i, sm)) for r, i in enumerate(pre)]
        ok = rot_exact and max(rels) <= LIMIT
        log({"test": "S8_prefill_had", "layout": layout, "rotate_q_fp16_exact": rot_exact, "rel_l2": rels, "ok": ok})
        assert ok, "S8"
    DONE["S8"] = True


def s5(pdt=torch.float16, gated=True):
    """batched prefill: dequant_multi (packed, bit-exact) + FlashInfer ragged FA2 for several requests whose query rows
    are interleaved with decode rows, vs the fp64 reference (FA2 precision: gate 0.003, see S3)"""
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
    gen = torch.Generator().manual_seed(55)
    sm = D ** -0.5
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    worst = 0.0
    for layout in ("HND", "NHD"):
        lens = [700, 3, 1500, 4, 300, 5000]
        qls = [700, 2, 200, 4, 37, 4096]                 # rows 0, 2, 4, 5 are prefills (interleaved with decodes)
        c = build(lens, qls, 4, 6, layout, gen)
        pre = [i for i in range(len(lens)) if qls[i] > 4]
        cu = [0]
        for i in pre:
            cu.append(cu[-1] + lens[i])
        N = cu[-1]
        sk = torch.empty(N, 4, D, dtype=pdt, device=dev)
        sv = torch.empty_like(sk)
        SV.dequant_multi(c["flat"], c["strides"], PS, c["bt"], torch.tensor(pre, dtype=torch.int32, device=dev),
                         torch.tensor(cu, dtype=torch.int32, device=dev), len(pre), N, sk, sv)
        exact = True
        for r, i in enumerate(pre):
            K, V = R.gather(c["views"], c["bt"][i], lens[i])
            exact = exact and torch.equal(sk[cu[r]:cu[r + 1]].double(), K) and torch.equal(sv[cu[r]:cu[r + 1]].double(), V)
        qo = [0]
        for i in pre:
            qo.append(qo[-1] + qls[i])
        idx = torch.cat([torch.arange(c["qsl_l"][i], c["qsl_l"][i + 1]) for i in pre]).to(dev)
        w.plan(torch.tensor(qo, dtype=torch.int32), torch.tensor(cu, dtype=torch.int32), 24, 4, D, causal=True,
               sm_scale=sm, q_data_type=pdt, kv_data_type=pdt)
        o = w.run(c["q"].index_select(0, idx).to(pdt), sk, sv).to(torch.bfloat16)     # the backend stores bf16
        cu1 = dict(c, gk=torch.ones(4, device=dev), gv=torch.ones(4, device=dev))
        rels = []
        for r, i in enumerate(pre):
            rels.append(rel(o[qo[r]:qo[r + 1]], reference_rows(cu1, i, sm)))
        worst = max(worst, max(rels))
        ok = exact and (max(rels) <= LIMIT or not gated)
        log({"test": "S5_batched_prefill", "dtype": str(pdt)[6:], "gated": gated, "layout": layout,
             "dequant_multi_exact": exact, "rel_l2": rels, "ok": ok})
        assert ok, "S5"
    if gated:
        DONE["S5"] = True


if __name__ == "__main__":
    import os
    only = os.environ.get("KVFP4_TESTS")         # e.g. "S9,S7" runs only those (the others count as not run)
    if only:
        fns = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5, "S6": lambda: s4(had=True), "S7": s7, "S8": s8,
               "S9": s9, "S10": s10, "S11": s11, "S12": s12, "S13": s13}
        for k in only.split(","):
            fns[k]()
        log({"test": "complete", "subset": only, "complete": all(DONE.get(k) for k in only.split(","))})
        print("TEST_KVFP4_SERVING_DONE", flush=True)
        raise SystemExit
    s2()
    s4()
    s1()
    s5()
    s5(torch.bfloat16, gated=False)   # reported: the old bf16 prefill
    s3()
    s4(had=True)
    s7()
    s8()
    s9()
    s10()
    s11()
    s12()
    s13()
    log({"test": "complete", "complete": all(DONE.get(k) for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
                                                                  "S9", "S10", "S11", "S12", "S13"))})
    print("TEST_KVFP4_SERVING_DONE", flush=True)
