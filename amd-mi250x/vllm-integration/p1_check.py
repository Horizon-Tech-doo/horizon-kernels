"""P1 gate before any server run: the plugin's own dense() path (pad to bucket, bf16 -> fp16 -> bf16) on one GCD. python p1_check.py <out.jsonl> [Hkv list] [lengths]
Per (Hkv, n): error of ours and of torch SDPA (bf16) against a float64 reference (n <= 6144), then interleaved timings over ROUNDS rounds:
  ours = dense() with its copies, at bucket(n);  kern = the kernel alone on the padded buffers;  sdpa = F.scaled_dot_product_attention, bf16, exact n, K/V expanded outside the timed call.
SDPA is a sanity reference, NOT vLLM's prefill kernel: speed against stock is the server's time to first token (p1.sbatch), correctness against stock is HLP_VERIFY (arm fv_).
Lengths default to on-bucket, just-above-bucket (worst padding) and mid-gap values. The fail field lists every (Hkv, n) with ours' error above 2x SDPA's or above 2e-3, or a non-finite output."""
import json, statistics, sys, time
import torch, torch.nn.functional as F
import horizon_lumi_prefill as h
out = open(sys.argv[1], "a")
HK = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4]
B = h.buckets()
NS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else sorted({1024, 1025, 1500, 2048, 2049, 3000, 4096, 4097, 6000, 8192, 8193, 12000, 16384} | {b + 1 for b in B[5:40:7]})
G, D, ROUNDS = 6, 256, 7
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def ref64(q, k, v):
    qq, kk, vv = q.double().transpose(0, 1), k.double().repeat_interleave(G, 1).transpose(0, 1), v.double().repeat_interleave(G, 1).transpose(0, 1)
    s = qq @ kk.transpose(1, 2) / D ** 0.5
    n = s.size(-1); s.masked_fill_(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    return (F.softmax(s, -1) @ vv).transpose(0, 1)
def tm(f):
    torch.cuda.synchronize(); t = time.perf_counter(); f(); torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3
fail = []
for Hkv in HK:
    H = Hkv * G
    for n in NS:
        L = h.bucket(n)
        if L is None: continue
        g = torch.Generator(device="cuda").manual_seed(n * 7 + Hkv)
        q = torch.randn(n, H, D, device="cuda", generator=g).bfloat16(); k = torch.randn(n, Hkv, D, device="cuda", generator=g).bfloat16(); v = torch.randn(n, Hkv, D, device="cuda", generator=g).bfloat16()
        t0 = time.time(); kern = h.build(L, Hkv, G, D); tb = time.time() - t0
        o = torch.empty(n, H, D, device="cuda", dtype=torch.bfloat16)
        for _ in range(2): h.dense(kern, L, q, k, v, o)          # second call runs over buffers a call already filled
        qt, kt, vt = q.transpose(0, 1)[None], k.repeat_interleave(G, 1).transpose(0, 1)[None], v.repeat_interleave(G, 1).transpose(0, 1)[None]
        sd = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
        row = dict(Hkv=Hkv, n=n, L=L, pad_cost=round((L / n) ** 2, 4), build_s=round(tb, 2), finite=bool(torch.isfinite(o).all()))
        if n <= 6144:
            r = ref64(q, k, v); nr = r.norm()
            row["rel_ours"] = float((o.double() - r).norm() / nr); row["rel_sdpa"] = float((sd()[0].transpose(0, 1).double() - r).norm() / nr)
            row["max_abs_ours"] = float((o.double() - r).abs().max()); del r
            if row["rel_ours"] > max(2 * row["rel_sdpa"], 2e-3): fail.append([Hkv, n, "error"])
        if not row["finite"]: fail.append([Hkv, n, "nonfinite"])
        Qb, Kb, Vb = h._bufs(L, H, Hkv, D, q.device)
        arms = {"ours": lambda: h.dense(kern, L, q, k, v, o), "kern": lambda: kern(Qb, Kb, Vb), "sdpa": sd}
        for f in arms.values(): f(); f()
        ts = {a: [] for a in arms}
        for rd in range(ROUNDS):
            for a in (list(arms) if rd % 2 == 0 else list(arms)[::-1]): ts[a].append(tm(arms[a]))
        for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 4); row[a + "_all"] = [round(x, 4) for x in ts[a]]
        row["sdpa_over_ours"] = round(row["sdpa_ms"] / row["ours_ms"], 4); row["copy_share"] = round(1 - row["kern_ms"] / row["ours_ms"], 4)
        P(kind="cell", **row)
        del q, k, v, o, qt, kt, vt; torch.cuda.empty_cache()
# an input outside fp16's range must give a non-finite output, which is what the plugin's guard keys on
n, Hkv = 1500, HK[0]; L = h.bucket(n); kern = h.build(L, Hkv, G, D)
q = torch.randn(n, Hkv * G, D, device="cuda").bfloat16(); k = torch.randn(n, Hkv, D, device="cuda").bfloat16(); v = torch.randn(n, Hkv, D, device="cuda").bfloat16()
o = torch.empty_like(q); k[700, 0, 3] = 1e6; h.dense(kern, L, q, k, v, o); g1 = not bool(torch.isfinite(o).all())
k[700, 0, 3] = 0.5; h.dense(kern, L, q, k, v, o); g2 = bool(torch.isfinite(o).all())     # and the inf in the buffer must be gone on the next call
v[1490, 0, 5] = 1e6; h.dense(kern, L, q, k, v, o)                                         # inf lands in V row 1490 ...
o2 = torch.empty(1400, Hkv * G, D, device="cuda", dtype=torch.bfloat16); h.dense(kern, L, q[:1400], k[:1400], v[:1400], o2)   # ... which a shorter call in the same bucket must not see
g3 = bool(torch.isfinite(o2).all())
if not (g1 and g2 and g3): fail.append(["guard", g1, g2, g3])
P(kind="summary", guard_trips=g1, guard_clears=g2, shorter_call_clean=g3, fail=fail, ok=not fail)
