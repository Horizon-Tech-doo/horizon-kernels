# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# GPU candidate check: OUT.jsonl CANDIDATE.py; run only under GPU lock.
import importlib, json, math, sys
import torch
import flashinfer

OUT = sys.argv[1]
import importlib.util
from functools import partial
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



# Focused short-prefill race; all query rows checked, rotated KV and graphs.
def capture(fn):
    fn(); torch.cuda.synchronize()
    for attempt in range(3):
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph,capture_error_mode="thread_local"): fn()
            return graph
        except RuntimeError:
            if attempt==2: raise
            torch.cuda.synchronize()

assert not __import__('pathlib').Path(OUT).exists(), 'Refusing to mix result runs'
gen=torch.Generator(device=dev).manual_seed(3)
separator=torch.zeros(256*1024*1024,device=dev,dtype=torch.uint8)
for qlens,kvlens in (([1],[128]),([17,63],[100,129])):
    q,kc,vc,bt,qo,kvl,plan=make(qlens,kvlens,gen)
    wrapper,_=fi_run(q,kc,vc,plan)
    packed=torch.cat((kc,vc),dim=-1)
    copies=math.ceil(513*1024**2/packed.numel())
    storage=[packed.clone() for _ in range(copies)]
    cache=[(x[...,:D],x[...,D:]) for x in storage]
    rotated_bytes=sum(x.untyped_storage().nbytes() for x in storage)
    assert rotated_bytes>512*1024**2
    refs=ref_rows(q,kc,vc,bt,qo,kvl,[(b,t) for b in range(len(qlens)) for t in range(qlens[b])])
    out=torch.empty_like(q)
    runners={'flashinfer':lambda k,v:wrapper.run(q,(k,v),out=out,k_scale=KS,v_scale=VS)}
    for name,fn,tq in (
        ('v3_baseline',ap.prefill_v3,16),
        ('q8_k32_w8',partial(ap.prefill_v3,TQ=8,BN=32,num_warps=8,policy_qk=3,policy_pv=3,pad_rows=True),8),
        ('q4_k32_w4',partial(ap.prefill_v3,TQ=4,BN=32,num_warps=4,policy_qk=3,policy_pv=3,pad_rows=True),4)):
        tiles=ap.plan_prefill_tiles(qo,tq,dev)
        runners[name]=lambda k,v,fn=fn,tiles=tiles:fn(q,k,v,qo,bt,kvl,SCALE*KS,VS,out=out,tiles=tiles,kv_layout='HND')
    graphs={};eligible={}
    for name,fn in runners.items():
        fn(*cache[0]);torch.cuda.synchronize();expected=out.clone()
        error=max(float((out[int(qo[b])+t].double()-r).norm()/r.norm()) for (b,t),r in refs.items())
        eligible[name]=math.isfinite(error) and error<=.0025 and bool(torch.isfinite(out).all())
        graph=capture(lambda fn=fn:[fn(k,v) for k,v in cache])
        for _ in range(2):
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            assert torch.equal(out,expected),(name,'graph mismatch')
        graphs[name]=graph
        log({'test':'short_prefill_gate','qlens':qlens,'kvlens':kvlens,'variant':name,'reference_rows':sum(qlens),'relative_l2':error,'quality_pass':eligible[name],'graph_replays':2})
    names=list(graphs);timings={name:[] for name in names}
    for rnd in range(3):
        for name in names[rnd:]+names[:rnd]:
            samples=[]
            for _ in range(5):
                separator.add_(1) # outside timed graph, evicts preceding cache footprint
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples.append(start.elapsed_time(end)*1000/copies)
            timings[name].append(min(samples))
    log({'test':'short_prefill_timing','qlens':qlens,'kvlens':kvlens,'backend':str(getattr(wrapper,'_backend','unknown')),'page_size':PS,'copies':copies,'rotated_cache_bytes':rotated_bytes,'microseconds':timings,'quality_pass':eligible,'deployment_eligible':False})
    del graphs,graph,storage,cache,packed,refs,wrapper,runners
    torch.cuda.empty_cache()
log({'test':'complete','cases':2,'deployment_eligible':False})
