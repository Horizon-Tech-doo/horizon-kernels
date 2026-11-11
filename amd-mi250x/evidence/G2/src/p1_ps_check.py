"""P1 gate for gqa_fa_ps.py, one GCD: the weight-scaled kernel against the landed one on the server's own activations. python p1_ps_check.py <out.jsonl> <dump dir> [more dump dirs]
For every dumped verify step (queries of one chunk, all keys and values, vLLM's kernel's output): both kernels through the plugin's buffers exactly as the server runs them (square kernel at
Lk = context + chunk, the chunk's queries in the last rows, bf16 -> fp16 in, fp16 -> bf16 out), each against the float64 reference of the chunk rows; then interleaved timings of
  landed (gqa_fa), landed_b (the same kernel again: identical-code control), scaled (gqa_fa_ps), 7 rounds, order rotated.
The two modules are loaded side by side: gqa_fa first (its compile hook installed), then gqa_fa_ps.load, which stacks its hook. PS_ONLY=1: compile the scaled kernels only and exit (precompile)."""
import glob, importlib.util, json, os, statistics, sys, time
import torch
import horizon_lumi_prefill as h
out = open(sys.argv[1], "a"); G, D = 6, 256
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it=10):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
fa = h._mod()
fa._install_hook()                                   # the landed module's compile hook first; gqa_fa_ps stacks its own on top (see its docstring)
s = importlib.util.spec_from_file_location("gqa_fa_ps", os.path.join(os.environ["HLP_SCRIPTS"], "gqa_fa_ps.py")); w = importlib.util.module_from_spec(s); s.loader.exec_module(w)
ps = w.load(os.path.join(os.environ["HLP_SCRIPTS"], "gqa_fa.py"))
def ref_rows(q, k, v, ctx):
    n, H, _ = q.shape; Hkv = k.shape[1]; Lk = k.shape[0]; o = torch.empty(n, H, D, dtype=torch.float64, device="cuda")
    for r0 in range(0, n, 256):
        r1 = min(n, r0 + 256)
        for hk in range(Hkv):
            sc = torch.einsum("rgd,nd->rgn", q[r0:r1, hk * G:(hk + 1) * G].double(), k[:, hk].double()) / D ** 0.5
            mask = torch.arange(Lk, device="cuda")[None, :] > (ctx + torch.arange(r0, r1, device="cuda"))[:, None]
            sc.masked_fill_(mask[:, None, :], float("-inf"))
            o[r0:r1, hk * G:(hk + 1) * G] = torch.einsum("rgn,nd->rgd", torch.softmax(sc, -1), v[:, hk].double()); del sc
    return o
fail = []
files = sorted(f for d in sys.argv[2:] for f in glob.glob(d + "/hlp-dump-*.pt"))
for f in files:
    t = torch.load(f); q, k, v = t["q"].cuda(), t["k"].cuda(), t["v"].cuda(); stock = t["stock"].cuda().double(); ctx = int(t["ctx"])
    n, H, _ = q.shape; Hkv = k.shape[1]; Lk = k.shape[0]
    Q = torch.zeros(1, Lk, H, D, device="cuda", dtype=torch.float16); K = k.to(torch.float16)[None].contiguous(); V = v.to(torch.float16)[None].contiguous(); Q[0, ctx:].copy_(q)
    kern = {"landed": fa.gqa_flash_fast(1, Lk, Hkv, G, D, True, **fa.d256_tile(Lk, 1)), "scaled": ps.gqa_flash_fast(1, Lk, Hkv, G, D, True, **ps.d256_tile(Lk, 1))}
    ref = ref_rows(q, k, v, ctx); nr = ref.norm()
    row = dict(kind="cell", file=f.split("/")[-1], n=n, ctx=ctx, Lk=Lk, rel_stock=float((stock - ref).norm() / nr))
    for a, kk in kern.items():
        o = kk(Q, K, V)[0, ctx:].to(torch.bfloat16)
        d = o.double() - ref
        row["rel_" + a] = float(d.norm() / nr); row["worst_row_" + a] = float((d.flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp_min(1e-300)).max()); row["finite_" + a] = bool(torch.isfinite(o).all())
    arms = {"landed": lambda: kern["landed"](Q, K, V), "landed_b": lambda: kern["landed"](Q, K, V), "scaled": lambda: kern["scaled"](Q, K, V)}
    for g in arms.values(): tm(g, 2)
    ts = {a: [] for a in arms}; names = list(arms)
    for rd in range(7):
        for i in range(3): a = names[(i + rd) % 3]; ts[a].append(tm(arms[a]))
    for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 4)
    row["landed_over_scaled"] = round(row["landed_ms"] / row["scaled_ms"], 4); row["noise"] = round(abs(row["landed_b_ms"] / row["landed_ms"] - 1), 4)
    if not row["finite_scaled"] or row["rel_scaled"] > 1.25 * row["rel_stock"]: fail.append([row["file"], row["rel_scaled"], row["rel_stock"]])
    P(**row); del t, q, k, v, Q, K, V, ref, kern; torch.cuda.empty_cache()
# random inputs must not get worse either (weights near 1, large logits): n 4096 and 16384
for n, sc in ((4096, 1.0), (16384, 1.0), (4096, 4.0)):
    g = torch.Generator(device="cuda").manual_seed(n + int(sc))
    q = (torch.randn(n, 12, D, device="cuda", generator=g) * sc).bfloat16(); k = (torch.randn(n, 2, D, device="cuda", generator=g) * sc).bfloat16(); v = torch.randn(n, 2, D, device="cuda", generator=g).bfloat16()
    ref = ref_rows(q, k, v, 0); nr = ref.norm(); row = dict(kind="random", n=n, scale=sc)
    for a, mm in (("landed", fa), ("scaled", ps)):
        o = mm.gqa_flash_fast(1, n, 2, G, D, True, **mm.d256_tile(n, 1))(q.to(torch.float16)[None].contiguous(), k.to(torch.float16)[None].contiguous(), v.to(torch.float16)[None].contiguous())[0].to(torch.bfloat16)
        row["rel_" + a] = float((o.double() - ref).norm() / nr); row["finite_" + a] = bool(torch.isfinite(o).all())
    if not row["finite_scaled"] or row["rel_scaled"] > 1.25 * row["rel_landed"]: fail.append(["random", n, sc])
    P(**row); del q, k, v, ref
P(kind="summary", files=len(files), fail=fail, ok=not fail)
