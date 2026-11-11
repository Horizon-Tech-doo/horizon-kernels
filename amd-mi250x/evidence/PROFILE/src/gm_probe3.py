"""PROFILE / GEMM gate 3 (gm_probe2.py plus the ours_r224 arm), one GCD, the server's environment: tilelang/gemm_bf16.py against the server's own dense-linear path at the shapes the served 27B issues under TP 2.
python gm_probe3.py <out.jsonl> [M list] [N:K list]      env GM_REV=1 reverses the arm order (the second job of a pair)
Arms per (M, N, K), all on the same bf16 activations and bf16 weight (C[M,N] = A[M,K] @ W[N,K]^T):
  stock     vllm rocm_unquantized_gemm_impl (what the server runs: wvSplitK for M <= 5, else torch F.linear = hipBLASLt)
  stock_b   the same call again: the identical-code control, |stock_b / stock - 1| is this cell's noise
  ours      gemm_bf16 at the bucket Mb >= M (multiples of 128 from 1024; 16/32/64 below), as wired: copy A into a zero-padded [Mb, K] buffer when Mb != M, run, slice [:M]
  ours_bare the kernel alone on an [Mb, K] input (no copy)
  ours_r224 gemm_bf16's copy of gemm_hand_rows on a 224x256 tile (XOR-swizzled LDS, partial waits: the HAND_FILL schedule of 16384x4096x4096), M as given: rows past M read zeros through the buffer
            range check, so there is no padded copy. 1568 = 7 x 224 is the server's prefill chunk at its default settings. Only where N is a multiple of 256.
  hb_fp16 / ours_fp16   F.linear and gemm_prefetch on fp16 copies: the landed fp16 ratio in the same process, for reference
Rounds interleaved, order rotated per round; ms per call from ITERS calls between two synchronisations. Error against an fp32 product of the same bf16 inputs. fail lists every cell where ours is
non-finite or its error exceeds 1.25x stock's."""
import json, os, statistics, sys, time
import torch, torch.nn.functional as F
from vllm.model_executor.layers import utils as U
import gemm_prefetch as g
import gemm_bf16 as gb
out = open(sys.argv[1], "a")
MS = [int(x) for x in sys.argv[2].split(":")] if len(sys.argv) > 2 else [8, 32, 784, 1568, 3920, 7056, 15680, 1024, 2048, 4096, 8192, 16384]
NK = [tuple(map(int, x.split("x"))) for x in sys.argv[3].split(":")] if len(sys.argv) > 3 else [(17408, 5120), (5120, 8704), (8192, 5120), (7168, 5120), (5120, 3072)]
ROUNDS, REV = 7, os.environ.get("GM_REV") == "1"
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
def pick(mod, M, N, K):
    """(kernel, how, Mb) from a module with gemm_auto / gemm_small_m, at the first bucket that has a hand form."""
    if M < 1024:
        for Mc in [M] + [b for b in (16, 32, 64) if b > M]:
            k = mod.gemm_small_m(Mc, N, K)
            if k is not None: return k, "small_m", Mc
        return None, None, M
    for Mc in sorted({M, -(-M // 128) * 128, -(-M // 256) * 256}):
        k = mod.gemm_auto(Mc, N, K)
        if k is not None: return k, "auto:" + str(g.select_mode(Mc, N, K)), Mc
    return None, None, M
P(kind="env", torch=torch.__version__, hip=torch.version.hip, mfma=gb.MFMA, rev=REV, rounds=ROUNDS)
fail = []
for N, K in NK:
    gen = torch.Generator(device="cuda").manual_seed(N + K)
    w = (torch.randn(N, K, device="cuda", generator=gen) * 0.02).bfloat16(); w16 = w.to(torch.float16)
    for M in MS:
        x = torch.randn(M, K, device="cuda", generator=gen).bfloat16()
        ref = x.float() @ w.float().T; nr = ref.norm()
        err = lambda c: float((c.float() - ref).norm() / nr)
        row = dict(kind="cell", M=M, N=N, K=K)
        try:
            t0 = time.time(); kb, how, Mb = pick(gb, M, N, K); row.update(ours=how, M_bucket=Mb if kb is not None else None, build_s=round(time.time() - t0, 1))
            arms = {"stock": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "stock_b": lambda: U.rocm_unquantized_gemm_impl(x, w, None)}
            if kb is not None:
                wb = gb.small_m_b(Mb, w) if how == "small_m" else w
                xb = torch.zeros(Mb, K, device="cuda", dtype=torch.bfloat16); xb[:M].copy_(x)
                if Mb != M:
                    xp = torch.zeros(Mb, K, device="cuda", dtype=torch.bfloat16)
                    def ours():
                        xp[:M].copy_(x); return kb(xp, wb)[:M]
                else:
                    def ours(): return kb(x, wb)
                arms["ours"] = ours; arms["ours_bare"] = lambda: kb(xb, wb)
                kf, howf, Mf = pick(g, Mb, N, K)
                if kf is not None and Mf == Mb:
                    x16 = xb.to(torch.float16); wf = g.small_m_b(Mb, w16) if howf == "small_m" else w16
                    arms["hb_fp16"] = lambda: F.linear(x16, w16); arms["ours_fp16"] = lambda: kf(x16, wf)
            if M >= 512 and N % 256 == 0:
                kr = gb.module().gemm_hand_rows(M, N, K, 224, 256, 0, plw=1, xs=1, cn=1)
                f16 = torch.float16
                arms["ours_r224"] = (lambda: kr(x.view(f16), w.view(f16)).view(torch.bfloat16)) if M % 224 == 0 else (lambda: kr(x.view(f16), w.view(f16))[:M].view(torch.bfloat16))
            for a in ("stock", "ours", "ours_r224"):
                if a in arms:
                    c = arms[a](); row["err_" + a] = err(c[:M]); row["finite_" + a] = bool(torch.isfinite(c).all()); del c
            for a in ("ours", "ours_r224"):
                if a in arms and (not row["finite_" + a] or row["err_" + a] > 1.25 * row["err_stock"]): fail.append([a, M, N, K, row["err_" + a], row["err_stock"]])
            it = max(3, min(200, int(2e12 / (2.0 * M * N * K)) or 3)) if M >= 512 else 200
            for f in arms.values(): tm(f, 3)
            ts = {a: [] for a in arms}; names = list(arms)[::-1] if REV else list(arms)
            for rd in range(ROUNDS):
                for i in range(len(names)):
                    a = names[(i + rd) % len(names)]; ts[a].append(tm(arms[a], it))
            for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 5); row[a + "_all"] = [round(v, 5) for v in ts[a]]
            row["iters"] = it; row["stock_TF"] = round(2.0 * M * N * K / row["stock_ms"] * 1e-9, 2); row["noise"] = round(abs(row["stock_b_ms"] / row["stock_ms"] - 1), 4)
            if "ours" in arms:
                row["stock_over_ours"] = round(row["stock_ms"] / row["ours_ms"], 4); row["stock_over_ours_bare"] = round(row["stock_ms"] / row["ours_bare_ms"], 4)
            if "ours_r224" in arms: row["stock_over_r224"] = round(row["stock_ms"] / row["ours_r224_ms"], 4)
            if "ours_fp16" in arms: row["hbfp16_over_oursfp16"] = round(row["hb_fp16_ms"] / row["ours_fp16_ms"], 4)
        except Exception as e:
            row["error"] = repr(e)[-400:]
        P(**row)
        del x, ref; arms = None; torch.cuda.empty_cache()
    del w, w16; torch.cuda.empty_cache()
P(kind="summary", fail=fail, ok=not fail)
