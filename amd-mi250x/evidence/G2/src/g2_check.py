"""G2 gate (g1_check.py plus padded row counts and the landed table) before any server run: horizon_lumi_gemm's own run() path on one GCD, against vLLM's dense-linear implementation. python g1_check.py <out.jsonl>
Per (M in HLG_MS, shape in HLG_SHAPES): error of ours and of stock against an fp32 product of the same bf16 inputs, bitwise repeatability of ours, then interleaved timings (ROUNDS rounds, rotated):
  ours = run() as the hook calls it;  stock = rocm_unquantized_gemm_impl;  stock_b = the same again (identical-code control).
Then the cases the hook must refuse or pad: a row count not in HLG_MS (None), a shape not in HLG_SHAPES (None), a padded row count within HLG_MAXPAD (equal to stock within the error bound, shorter call after a longer one),
a non-contiguous x, and inputs holding inf / NaN (ours non-finite exactly where stock is). fail lists every miss."""
import json, os, statistics, sys, time
import torch
import horizon_lumi_gemm as h
from vllm.model_executor.layers import utils as U
stock = U.rocm_unquantized_gemm_impl
out = open(sys.argv[1], "a"); ROUNDS = 7
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
fail = []
W = {}
for N, K in sorted(h.shapes()):
    g = torch.Generator(device="cuda").manual_seed(N * 3 + K)
    W[(N, K)] = (torch.randn(N, K, device="cuda", generator=g) * 0.02).bfloat16()
# cases: every compiled row count as it is, and for every padded bucket a row count 29 below it (a 1,568-token chunk plus 3 decode rows at the 1600 bucket), which goes through the padded copy
CASES = [(M, M) for M in h.ms()] + [(Mb - 29, Mb) for Mb in h.pad_ms() if Mb in h.ms()]
for M, Mk in CASES:
    for (N, K), w in W.items():
        g = torch.Generator(device="cuda").manual_seed(M + N + K)
        x = torch.randn(M, K, device="cuda", generator=g).bfloat16()
        if h.bucket(M) != Mk: fail.append([M, N, K, "bucket", h.bucket(M)]); continue
        ref = x.float() @ w.float().T; nr = ref.norm()
        o = h.run(x, w, stock); s = stock(x, w, None)
        if o is None:
            if h.build(Mk, N, K) is not None: fail.append([M, N, K, "not taken"])
            else: P(kind="left_to_stock", M=M, N=N, K=K)
            del x, ref, s; continue
        row = dict(kind="cell", M=M, M_kernel=Mk, N=N, K=K, err_ours=float((o.float() - ref).norm() / nr), err_stock=float((s.float() - ref).norm() / nr), ours_vs_stock=float((o.float() - s.float()).norm() / s.float().norm()),
                   finite=bool(torch.isfinite(o).all()), shape_ok=tuple(o.shape) == (M, N), repeat_bitwise=bool(torch.equal(o, h.run(x, w, stock))), contiguous=o.is_contiguous())
        if not (row["finite"] and row["shape_ok"] and row["repeat_bitwise"]) or row["err_ours"] > 1.25 * row["err_stock"]: fail.append([M, N, K, "error", row["err_ours"], row["err_stock"]])
        del ref, o, s
        it = max(3, min(200, int(2e12 / (2.0 * M * N * K)) or 3))
        arms = {"ours": lambda: h.run(x, w, stock), "stock": lambda: stock(x, w, None), "stock_b": lambda: stock(x, w, None)}
        for f in arms.values(): tm(f, 3)
        ts = {a: [] for a in arms}; names = list(arms)[::-1] if os.environ.get("G1_REV") == "1" else list(arms)     # G1_REV=1: the reversed-order job of a pair
        for rd in range(ROUNDS):
            for i in range(3): a = names[(i + rd) % 3]; ts[a].append(tm(arms[a], it))
        for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 5); row[a + "_all"] = [round(v, 5) for v in ts[a]]
        row["rev"] = os.environ.get("G1_REV") == "1"; row["stock_over_ours"] = round(row["stock_ms"] / row["ours_ms"], 4); row["noise"] = round(abs(row["stock_b_ms"] / row["stock_ms"] - 1), 4)
        P(**row); del x; torch.cuda.empty_cache()
# refusals and edge cases
M0, (N, K) = next((M, nk) for M in h.ms() for nk in sorted(W) if M not in h.pad_ms() and h.build(M, *nk) is not None); w = W[(N, K)]
x = torch.randn(M0 + 13, K, device="cuda").bfloat16()
c1 = h.run(x, w, stock) is None                                             # row count not compiled
c2 = h.run(torch.randn(M0, 4096, device="cuda").bfloat16(), torch.randn(1024, 4096, device="cuda").bfloat16(), stock) is None   # shape not listed
os.environ["HLG_MAXPAD"] = "0.02"
xa = torch.randn(M0, K, device="cuda").bfloat16(); xa[-1, 7] = float("inf"); h.run(xa, w, stock)      # fills nothing persistent, but run a longer call first
xs = torch.randn(M0 - 9, K, device="cuda").bfloat16()
o = h.run(xs, w, stock); s = stock(xs, w, None)
c3 = o is not None and tuple(o.shape) == tuple(s.shape) and bool(torch.isfinite(o).all()) and float((o.float() - s.float()).norm() / s.float().norm()) < 5e-3    # padded call, after a call that left inf in no shared buffer
os.environ["HLG_MAXPAD"] = "0"
xn = torch.randn(M0, 2 * K, device="cuda").bfloat16()[:, ::2]                 # non-contiguous view
o = h.run(xn, w, stock); s = stock(xn.contiguous(), w, None)
c4 = o is not None and float((o.float() - s.float()).norm() / s.float().norm()) < 5e-3
xi = torch.randn(M0, K, device="cuda").bfloat16(); xi[5, 3] = float("inf"); xi[9, 11] = float("nan")
o = h.run(xi, w, stock); s = stock(xi, w, None)
c5 = bool(torch.equal(torch.isfinite(o).all(1), torch.isfinite(s).all(1)))       # the same rows are non-finite
xb = torch.full((M0, K), 1e18, device="cuda").bfloat16(); wb = torch.full_like(w, 1e18)   # sums overflow bf16's range the same way in both (fp32 accumulate, then round)
o = h.run(xb, wb, stock); s = stock(xb, wb, None)
c6 = bool(torch.equal(torch.isfinite(o), torch.isfinite(s)))
for nm, c in (("off_list_rows_refused", c1), ("unlisted_shape_refused", c2), ("padded_call", c3), ("noncontiguous", c4), ("nonfinite_rows_match", c5), ("overflow_matches", c6)):
    if not c: fail.append([nm])
P(kind="summary", off_list_rows_refused=c1, unlisted_shape_refused=c2, padded_call=c3, noncontiguous=c4, nonfinite_rows_match=c5, overflow_matches=c6, counters=h.counters(), fail=fail, ok=not fail)
