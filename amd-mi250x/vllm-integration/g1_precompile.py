"""Compile horizon_lumi_gemm's kernels (every M of HLG_MS x every shape of HLG_SHAPES) into TILELANG_CACHE_DIR, one slice per process. python g1_precompile.py <index> <count>. No GPU work."""
import sys, time
import horizon_lumi_gemm as h
I, N = int(sys.argv[1]), int(sys.argv[2])
keys = [(M, n, k) for M in h.ms() for n, k in sorted(h.shapes())]
for i, key in enumerate(keys):
    if i % N != I: continue
    t = time.time(); k = h.build(*key); print("compiled" if k is not None else "stock", *key, round(time.time() - t, 1), flush=True)
