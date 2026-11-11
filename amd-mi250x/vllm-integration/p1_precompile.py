"""Compile the prefill kernels of every HLP bucket into TILELANG_CACHE_DIR, one slice of the list per process. python p1_precompile.py <index> <count> <Hkv> [G] [D]. No GPU work; the server then loads the TileLang cache."""
import sys, time
import horizon_lumi_prefill as h
I, N, Hkv = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
G = int(sys.argv[4]) if len(sys.argv) > 4 else 6
D = int(sys.argv[5]) if len(sys.argv) > 5 else 256
for i, L in enumerate(h.buckets()):
    if i % N != I: continue
    t = time.time(); h.build(L, Hkv, G, D); print("compiled", L, Hkv, G, D, round(time.time() - t, 1), flush=True)
