"""Compile the kernels named by HLD_PRELOAD in TILELANG_CACHE_DIR, one slice of the list per process. python precompile.py <index> <count> <Hkv>. No GPU work; the TileLang cache is what the server then loads."""
import os, sys, time
import horizon_lumi_decode as h
I, N, Hkv = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
items = [(lay, *map(int, it.split(":")), S) for lay in os.environ["HLD_LAYOUTS"].split(",") for it in os.environ["HLD_PRELOAD"].split(",") for S in os.environ.get("HLD_SKIP_LIST", "0").split(",") if S == "0" or it.endswith(":65536")]
for i, (lay, B, L, S) in enumerate(items):
    if i % N != I: continue
    os.environ["HLD_SKIP_S"] = S
    t = time.time(); h.build(lay, B, L, Hkv, 12 // Hkv, 256, 784); print("compiled", lay, B, L, "skipS", S, round(time.time() - t, 1), flush=True)
