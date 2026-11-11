"""Compile horizon_lumi_prefill's query-window kernels (qw_keys(): every chunk of HLP_QW_CHUNKS at every total k x chunk) into TILELANG_CACHE_DIR, one slice per process. HLP_QW=1 must be set.
python qw_precompile.py <index> <count> <Hkv> [G] [D]. No GPU work."""
import sys, time
import horizon_lumi_prefill as h
I, N, Hkv = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
G = int(sys.argv[4]) if len(sys.argv) > 4 else 6
D = int(sys.argv[5]) if len(sys.argv) > 5 else 256
for i, (L, n) in enumerate(h.qw_keys()):
    if i % N != I: continue
    t = time.time(); h.build_qw(L, n, Hkv, G, D); print("compiled qw", L, n, Hkv, G, D, round(time.time() - t, 1), flush=True)
