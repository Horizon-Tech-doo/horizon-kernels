"""P1 follow-up, one GCD, the server's environment. python p1_ref64.py <out.jsonl>
Part A (error above 6,144 tokens): horizon_lumi_prefill's dense() path (bf16 -> fp16 kernel -> bf16, padded to its bucket) and torch SDPA on bf16, both against a float64 reference computed in chunks of
query rows so that it fits in memory; n up to 16384, Hkv 2, G 6, D 256 (the served 27B under TP 2). Inputs: randn (as p1_check.py) and "peaky" (q and k scaled by 2.5: sharper attention, larger logits).
Per case: relative Frobenius error of the whole output, the worst single row, and ours against SDPA (what the server's verify arm reports against vLLM's kernel).
Part B (chunks with cached context, plugin-only form): the square causal kernel at Lk = context + chunk with the chunk's queries in rows context..Lk and zero queries before (causal masking makes the chunk
rows exact; the context rows are computed and dropped). Timed as it would be wired: gather of the context K and V from a paged cache [blocks, block_size, Hkv, D] through a block table (random order),
bf16 -> fp16 copies into the padded buffers, kernel, copy of the chunk rows out. Parts timed separately: gather+copies, kernel. Chunk 1568 rows, context 1568 x 1..9, block sizes 16 and 784.
One case (context 4704) is checked against the float64 reference of the chunk rows."""
import json, statistics, sys, time
import torch, torch.nn.functional as F
import horizon_lumi_prefill as h
out = open(sys.argv[1], "a")
G, D, Hkv = 6, 256, 2; H = Hkv * G
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it=20):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
def ref64_rows(q, k, v, r0, r1):
    """float64 causal GQA attention for query rows r0..r1 of a length-n sequence: [r1-r0, H, D]."""
    n = k.shape[0]; o = torch.empty(r1 - r0, H, D, dtype=torch.float64, device=q.device)
    for hk in range(Hkv):
        kk, vv = k[:, hk].double(), v[:, hk].double()                                  # [n, D]
        qq = q[r0:r1, hk * G:(hk + 1) * G].double()                                    # [r, G, D]
        s = torch.einsum("rgd,nd->rgn", qq, kk) / D ** 0.5
        mask = torch.arange(n, device=q.device)[None, :] > torch.arange(r0, r1, device=q.device)[:, None]
        s.masked_fill_(mask[:, None, :], float("-inf"))
        o[:, hk * G:(hk + 1) * G] = torch.einsum("rgn,nd->rgd", torch.softmax(s, -1), vv)
        del s, kk, vv, qq
    return o
def errs(get, q, k, v, n, step=384):
    """(rel Frobenius, worst row rel, max abs) of get(r0, r1) [rows, H, D] against the float64 reference, accumulated over row chunks."""
    num = den = 0.0; worst = 0.0; mabs = 0.0
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step); r = ref64_rows(q, k, v, r0, r1); d = get(r0, r1).double() - r
        num += float((d * d).sum()); den += float((r * r).sum())
        worst = max(worst, float((d.flatten(1).norm(dim=1) / r.flatten(1).norm(dim=1).clamp_min(1e-300)).max())); mabs = max(mabs, float(d.abs().max())); del r, d
    return (num / den) ** 0.5, worst, mabs
# ---- Part A
for kind, sc in (("randn", 1.0), ("peaky", 2.5)):
    for n in (2048, 6144, 8192, 12000, 14112, 16384):
        g = torch.Generator(device="cuda").manual_seed(n * 3 + int(sc * 10))
        q = (torch.randn(n, H, D, device="cuda", generator=g) * sc).bfloat16(); k = (torch.randn(n, Hkv, D, device="cuda", generator=g) * sc).bfloat16(); v = torch.randn(n, Hkv, D, device="cuda", generator=g).bfloat16()
        L = h.bucket(n); kern = h.build(L, Hkv, G, D)
        o = torch.empty(n, H, D, device="cuda", dtype=torch.bfloat16); h.dense(kern, L, q, k, v, o)
        sd = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.repeat_interleave(G, 1).transpose(0, 1)[None], v.repeat_interleave(G, 1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1).contiguous()
        eo = errs(lambda a, b: o[a:b], q, k, v, n); es = errs(lambda a, b: sd[a:b], q, k, v, n)
        dd = (o.float() - sd.float())
        P(kind="err", input=kind, n=n, L=L, finite=bool(torch.isfinite(o).all()), rel_ours=eo[0], worst_row_ours=eo[1], max_abs_ours=eo[2], rel_sdpa=es[0], worst_row_sdpa=es[1], max_abs_sdpa=es[2],
          ours_vs_sdpa=float(dd.norm() / sd.float().norm()), max_abs_ours_vs_sdpa=float(dd.abs().max()), ref_absmax=float(sd.float().abs().max()))
        del q, k, v, o, sd, dd; torch.cuda.empty_cache()
# ---- Part B
C = 1568
g = torch.Generator(device="cuda").manual_seed(7)
for ctx in [C * i for i in range(1, 10)]:
    Lk = ctx + C
    q = torch.randn(Lk, H, D, device="cuda", generator=g).bfloat16(); k = torch.randn(Lk, Hkv, D, device="cuda", generator=g).bfloat16(); v = torch.randn(Lk, Hkv, D, device="cuda", generator=g).bfloat16()
    t0 = time.time(); kern = h.build(Lk, Hkv, G, D); tb = time.time() - t0
    Qb = torch.zeros(1, Lk, H, D, device="cuda", dtype=torch.float16); Kb = torch.zeros(1, Lk, Hkv, D, device="cuda", dtype=torch.float16); Vb = torch.zeros_like(Kb)
    outc = torch.empty(C, H, D, device="cuda", dtype=torch.bfloat16)
    row = dict(kind="chunk", ctx=ctx, chunk=C, Lk=Lk, build_s=round(tb, 1))
    for bs in (16, 784):
        nb = ctx // bs; perm = torch.randperm(nb, device="cuda", generator=g)
        Kc = torch.empty(nb, bs, Hkv, D, device="cuda", dtype=torch.bfloat16); Vc = torch.empty_like(Kc)
        Kc[perm] = k[:ctx].view(nb, bs, Hkv, D); Vc[perm] = v[:ctx].view(nb, bs, Hkv, D)        # block perm[i] of the cache holds the sequence's block i
        def gather():
            Kb[0, :ctx].copy_(Kc.index_select(0, perm).view(ctx, Hkv, D)); Vb[0, :ctx].copy_(Vc.index_select(0, perm).view(ctx, Hkv, D))
            Kb[0, ctx:].copy_(k[ctx:]); Vb[0, ctx:].copy_(v[ctx:]); Qb[0, ctx:].copy_(q[ctx:])
        def full():
            gather(); O = kern(Qb, Kb, Vb); outc.copy_(O[0, ctx:])
        full(); full()
        tg, tf = [], []
        for rd in range(7):
            tg.append(tm(gather)); tf.append(tm(full))
        row[f"gather_ms_bs{bs}"] = round(statistics.median(tg), 4); row[f"wired_ms_bs{bs}"] = round(statistics.median(tf), 4)
        del Kc, Vc
    tk = [tm(lambda: kern(Qb, Kb, Vb)) for _ in range(7)]; row["kernel_ms"] = round(statistics.median(tk), 4)
    if ctx == C * 3:
        r = ref64_rows(q, k, v, ctx, Lk); dlt = outc.double() - r
        row["rel_chunk_rows_vs_fp64"] = float(dlt.norm() / r.norm()); row["finite"] = bool(torch.isfinite(outc).all()); del r, dlt
    P(**row)
    del q, k, v, Qb, Kb, Vb, kern; torch.cuda.empty_cache()
P(kind="done")
