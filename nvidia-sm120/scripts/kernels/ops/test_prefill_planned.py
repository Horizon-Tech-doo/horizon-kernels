# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# GPU candidate check: OUT.jsonl CANDIDATE.py; run only under GPU lock.
import importlib, json, math, sys
import torch
import flashinfer

OUT = sys.argv[1]
import importlib.util
spec = importlib.util.spec_from_file_location("planned_prefill_candidate", sys.argv[2])
ap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ap)
dev = torch.device("cuda")
KVH, G, D, PS = 4, 6, 256, 64
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



gen = torch.Generator(device=dev); gen.manual_seed(3)
for qlens, kvlens in (([17, 63], [100, 129]), ([1024], [4096]), ([8192], [16384])):
    q, kc, vc, bt, qo, kvl, plan = make(qlens, kvlens, gen)
    w, fi = fi_run(q, kc, vc, plan)
    fi_out = fi().clone()
    rows = [(b, t) for b in range(len(qlens)) for t in sorted({0, 1, qlens[b]//2, qlens[b]-1})]
    refs = ref_rows(q, kc, vc, bt, qo, kvl, rows)
    runners = {"flashinfer": fi}
    for name, fn, tq in (("v1", ap.prefill, 16), ("v2", ap.prefill_v2, 8), ("v3", ap.prefill_v3, 16)):
        tiles = ap.plan_prefill_tiles(qo, tq, dev)
        out = torch.empty_like(q)
        def run(fn=fn, tiles=tiles, out=out):
            return fn(q, kc, vc, qo, bt, kvl, SCALE*KS, VS, out=out, tiles=tiles, kv_layout="HND")
        eager = fn(q, kc, vc, qo, bt, kvl, SCALE*KS, VS, kv_layout="HND").clone()
        run(); torch.cuda.synchronize()
        assert torch.equal(out, eager), "planned/eager mismatch"
        assert torch.isfinite(out).all()
        rel = max(float((out[int(qo[b])+t].double()-r).norm()/r.norm()) for (b,t),r in refs.items())
        assert rel <= .0025
        for attempt in range(3):
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    run()
                break
            except RuntimeError:
                if attempt == 2: raise
        for _ in range(2):
            out.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
            assert torch.equal(out, eager), "graph replay mismatch"
        del graph
        log({"test":"planned_gate", "variant":name, "qlens":qlens, "kvlens":kvlens, "rel_l2":rel, "ok":True})
        runners[name] = run
    # Prepared-run eager timing: same planning exclusion on both sides.
    # This is warm-cache diagnostic timing, not a production promotion gate.
    samples = {name: [] for name in runners}
    names = list(runners)
    for rnd in range(3):
        for name in names[rnd:] + names[:rnd]:
            run = runners[name]
            run(); torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); run(); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end))
    log({"test":"planned_timing", "qlens":qlens, "kvlens":kvlens, "page_size":PS,
         "flashinfer_backend":getattr(w,"_backend",None), "ms":samples,
         "measurement":"prepared eager, warm cache; diagnostic only"})
log({"test":"complete", "ok":True})
