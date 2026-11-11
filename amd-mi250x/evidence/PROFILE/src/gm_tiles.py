"""PROFILE / GEMM tile sweep for the server's default prefill chunk, one GCD. python gm_tiles.py <out.jsonl> [M list] [NxK list]
For M = 1568 (the chunk vLLM schedules at its default max_num_batched_tokens with this model) and 784, at the server's five large linear shapes: gemm_bf16's copy of gemm_hand_rows over a list of tiles and
schedules, each against vLLM's own path (stock) in the same process; stock_b is the identical-code control. A form that fails to build or is wrong is recorded and skipped.
Interleaved rounds, rotated order (GM_REV=1 reverses). This is a search, not a landing race: a winner here still needs two opposite-order jobs."""
import json, os, statistics, sys, time
import torch
from vllm.model_executor.layers import utils as U
import gemm_bf16 as gb
out = open(sys.argv[1], "a")
MS = [int(x) for x in sys.argv[2].split(":")] if len(sys.argv) > 2 else [1568, 784]
NK = [tuple(map(int, x.split("x"))) for x in sys.argv[3].split(":")] if len(sys.argv) > 3 else [(17408, 5120), (5120, 8704), (8192, 5120), (7168, 5120), (5120, 3072)]
TILES = [(224, 256), (224, 224), (224, 192), (224, 160), (224, 128), (112, 256), (112, 128), (256, 128), (128, 128), (160, 256), (192, 256), (96, 256), (224, 96), (224, 64)]
SCHED = {"xs4": dict(pad=0, plw=1, xs=1, cn=1), "plw": dict(pad=32, plw=1, cn=1), "plw16": dict(pad=16, plw=1, cn=1), "x8_42": dict(pad=0, plw=1, nw=8, wg=(4, 2), xs=1, cn=1), "x8_24": dict(pad=0, plw=1, nw=8, wg=(2, 4), xs=1, cn=1)}
ROUNDS, REV = 5, os.environ.get("GM_REV") == "1"
f16, bf = torch.float16, torch.bfloat16
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
m = gb.module()
for N, K in NK:
    gen = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=gen) * 0.02).bfloat16()
    for M in MS:
        x = torch.randn(M, K, device="cuda", generator=gen).bfloat16()
        ref = x.float() @ w.float().T; nr = ref.norm()
        es = float((U.rocm_unquantized_gemm_impl(x, w, None).float() - ref).norm() / nr)
        arms = {"stock": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "stock_b": lambda: U.rocm_unquantized_gemm_impl(x, w, None)}
        meta = {}
        for (bM, bN) in TILES:
            for sn, kw in SCHED.items():
                nm = f"{bM}x{bN}:{sn}"
                try:
                    kk = dict(kw); pad = kk.pop("pad")
                    k = m.gemm_hand_rows(M, N, K, bM, bN, pad, **kk)
                    f = (lambda k=k: k(x.view(f16), w.view(f16))[:M].view(bf))
                    c = f(); e = float((c.float() - ref).norm() / nr)
                    if tuple(c.shape) != (M, N) or not bool(torch.isfinite(c).all()) or e > 1.25 * es:
                        P(kind="bad", M=M, N=N, K=K, form=nm, err=e, err_stock=es, shape=list(c.shape)); continue
                    arms[nm] = f; meta[nm] = dict(err=e, blocks=-(-M // bM) * -(-N // bN))
                except Exception as ex:
                    P(kind="nobuild", M=M, N=N, K=K, form=nm, err=repr(ex)[-200:])
        it = max(5, min(200, int(1e12 / (2.0 * M * N * K))))
        for f in arms.values(): tm(f, 2)
        ts = {a: [] for a in arms}; names = list(arms)[::-1] if REV else list(arms)
        for rd in range(ROUNDS):
            for i in range(len(names)):
                a = names[(i + rd) % len(names)]; ts[a].append(tm(arms[a], it))
        md = {a: statistics.median(v) for a, v in ts.items()}
        res = sorted(((round(md["stock"] / md[a], 4), a) for a in arms if a not in ("stock", "stock_b")), reverse=True)
        P(kind="cell", M=M, N=N, K=K, err_stock=es, stock_ms=round(md["stock"], 5), stock_TF=round(2.0 * M * N * K / md["stock"] * 1e-9, 1), noise=round(abs(md["stock_b"] / md["stock"] - 1), 4), iters=it, forms=len(res),
          best=res[:6], all={a: [r, meta[a]["blocks"], round(meta[a]["err"], 6)] for r, a in res}, raw={a: [round(v, 5) for v in ts[a]] for a in ("stock", "stock_b", *[a for _, a in res[:3]])})
        del x, ref, arms; torch.cuda.empty_cache()
    del w; torch.cuda.empty_cache()
P(kind="done")
