# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# P-ATTN v1 gate: horizon_ops.attn_prefill vs FlashInfer BatchPrefillWithPagedKVCacheWrapper (what vLLM runs for
# prefill) vs fp64 on sampled rows, paged FP8 KV (page 1600), causal chunked prefill; speed at chunk 8192 with
# 8k..64k keys (a 64k prompt's chunks).
import importlib, json, math, sys
import torch
import flashinfer

OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
ap = importlib.import_module(f"tilelang.contrib.{PKG}.attn_prefill")
dev = torch.device("cuda")
KVH, G, D, PS = 4, 6, 256, 1600
HQ = KVH * G
SCALE = D ** -0.5
KS, VS = 0.05, 0.05
WS = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)


def log(r):
    print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")


def make(qlens, kvlens, gen):
    B = len(qlens)
    pages = [math.ceil(l / PS) for l in kvlens]
    NP = sum(pages) + 2
    perm = (torch.randperm(NP - 1, generator=torch.Generator().manual_seed(B)) + 1)
    MP = max(pages)
    bt = torch.zeros(B, MP, dtype=torch.int32); o = 0
    ind = []
    for b, p in enumerate(pages):
        bt[b, :p] = perm[o:o + p]; ind += perm[o:o + p].tolist(); o += p
    kv = (torch.randn(NP, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
    kc, vc = kv[..., :D], kv[..., D:]
    q = torch.randn(sum(qlens), HQ, D, device=dev, generator=gen).to(torch.bfloat16)
    qo = torch.zeros(B + 1, dtype=torch.int32); qo[1:] = torch.cumsum(torch.tensor(qlens), 0)
    kvi = torch.zeros(B + 1, dtype=torch.int32); kvi[1:] = torch.cumsum(torch.tensor(pages), 0)
    last = torch.tensor([l - (p - 1) * PS for l, p in zip(kvlens, pages)], dtype=torch.int32)
    return q, kc, vc, bt.to(dev), qo.to(dev), torch.tensor(kvlens, dtype=torch.int32, device=dev), \
        (qo.to(dev), kvi.to(dev), torch.tensor(ind, dtype=torch.int32, device=dev), last.to(dev))


def fi_run(q, kc, vc, plan):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(WS, "HND")
    qo, kvi, ind, last = plan
    w.plan(qo, kvi, ind, last, HQ, KVH, D, PS, causal=True, sm_scale=SCALE, q_data_type=torch.bfloat16,
           kv_data_type=torch.float8_e4m3fn)
    return w, (lambda: w.run(q, (kc, vc), k_scale=KS, v_scale=VS))


def ref_rows(q, kc, vc, bt, qo, kvl, rows):
    out = {}
    for (b, t) in rows:
        L = int(kvl[b]); ql = int(qo[b + 1] - qo[b])
        pg = bt[b, : math.ceil(L / PS)].long()
        K = kc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * KS
        V = vc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * VS
        lim = L - ql + t + 1
        qq = q[int(qo[b]) + t].double().view(KVH, G, D)
        s = torch.einsum("hgd,lhd->hgl", qq, K[:lim]) * SCALE
        out[(b, t)] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, -1), V[:lim]).reshape(HQ, D)
    return out


gen = torch.Generator(device=dev); gen.manual_seed(0)
for qlens, kvlens in (([100], [100]), ([37, 500], [1000, 500]), ([2048], [9000]), ([8192], [16384])):
    q, kc, vc, bt, qo, kvl, plan = make(qlens, kvlens, gen)
    w, run = fi_run(q, kc, vc, plan)
    o_fi = run()
    o_hz = ap.prefill(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")
    o_hz2 = ap.prefill(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")
    o_s64 = ap.prefill(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, BN=64, share=1, kv_layout="HND")
    o_v2 = ap.prefill_v2(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")
    o_v3 = ap.prefill_v3(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")
    torch.cuda.synchronize()
    rows = [(b, t) for b in range(len(qlens)) for t in sorted({0, 1, qlens[b] // 2, qlens[b] - 1})]
    ref = ref_rows(q, kc, vc, bt, qo, kvl, rows)
    ef = max(float((o_fi[int(qo[b]) + t].double() - r).abs().max()) for (b, t), r in ref.items())
    eh = max(float((o_hz[int(qo[b]) + t].double() - r).abs().max()) for (b, t), r in ref.items())
    log({"test": "prefill_correct", "qlens": qlens, "kvlens": kvlens, "fi_max_abs": ef, "ours_max_abs": eh,
         "ours_vs_fi_max_abs": float((o_fi.float() - o_hz.float()).abs().max()),
         "deterministic": bool(torch.equal(o_hz, o_hz2)),
         "s64_vs_ours_max_abs": float((o_s64.float() - o_hz.float()).abs().max()),
         "v2_max_abs": max(float((o_v2[int(qo[b]) + t].double() - r).abs().max()) for (b, t), r in ref.items()),
         "v3_max_abs": max(float((o_v3[int(qo[b]) + t].double() - r).abs().max()) for (b, t), r in ref.items())})
    del kc, vc
for kvlen in (8192, 16384, 32768, 65536):
    q, kc, vc, bt, qo, kvl, plan = make([8192], [kvlen], gen)
    w, run = fi_run(q, kc, vc, plan)
    ts = {}
    for name, fn in (("flashinfer", run), ("ours", lambda: ap.prefill(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")),
                     ("ours_s64", lambda: ap.prefill(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, BN=64, share=1, kv_layout="HND")),
                     ("ours_v2", lambda: ap.prefill_v2(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND")),
                     ("ours_v3", lambda: ap.prefill_v3(q, kc, vc, qo, bt, kvl, SCALE * KS, VS, kv_layout="HND"))):
        fn(); torch.cuda.synchronize()
        best = 1e30
        for _ in range(3):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize(); best = min(best, e0.elapsed_time(e1))
        ts[name] = round(best, 2)
    flops = 4 * 8192 * (kvlen - 8192 / 2) * HQ * D
    log({"test": "prefill_bench", "q": 8192, "kv": kvlen, **{k + "_ms": v for k, v in ts.items()},
         "flashinfer_tflops": round(flops / ts["flashinfer"] / 1e9, 1), "ours_tflops": round(flops / ts["ours"] / 1e9, 1),
         "ours_s64_tflops": round(flops / ts["ours_s64"] / 1e9, 1),
         "ours_v2_tflops": round(flops / ts["ours_v2"] / 1e9, 1),
         "ours_v3_tflops": round(flops / ts["ours_v3"] / 1e9, 1)})
    del kc, vc
    torch.cuda.empty_cache()
print("TEST_PREFILL_DONE", flush=True)
# GDN chunked prefill gate (horizon_ops.gdn_prefill vs FLA), run in the same slot
try:
    import runpy
    sys.argv = ["test_gdn_prefill.py", "$HK_WORK/ops/gdn_prefill.jsonl", PKG]
    runpy.run_path("$HK_WORK/ops/test_gdn_prefill.py", run_name="__main__")
except Exception as e:  # noqa: BLE001
    import traceback; traceback.print_exc()
    print(json.dumps({"test": "gdnpf_error", "error": str(e)[:500]}), flush=True)
