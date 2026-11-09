# N4 round 3: standalone hand kernels from the landed hand_source. h11_c8_<shape> is 256x128 c8 pad 32, h11_o128_<shape> 128x256 c8 pad 32, h11_<arm>_<6k|16k> the untuned 128x256 arms.
import json, sys
sys.path.insert(0, "../../scripts")
import gemm_prefetch as g
SH = {"4k": (4096, 4096, 4096), "8k": (8192, 8192, 8192), "16k": (16384, 16384, 16384), "6k": (6144, 6144, 6144), "12k": (12288, 12288, 12288),
      "17k": (4096, 17408, 5120), "32k": (32768, 1024, 2048), "5632": (5632, 3200, 4096), "4kk": (16384, 4096, 4096)}
ARMS = {"p16": dict(pad=16), "a48b16": dict(pad=48, pad_b=16), "a96b16": dict(pad=96, pad_b=16), "fr15": dict(fr=0.15), "fr40": dict(fr=0.4),
        "c4": dict(sw="c4"), "c16": dict(sw="c16"), "r4": dict(sw="r4"), "r8": dict(sw="r8")}
dyn = {}
def w(n, M, N, K, **kw):
    src, d = g.hand_source(M, N, K, entry=n, **kw)
    open(f"h11/{n}.cu", "w").write(src)
    dyn[n] = dict(dyn=d, bM=kw.get("bM", 256), bN=kw.get("bN", 128))
for sn, (M, N, K) in SH.items():
    w(f"h11_c8_{sn}", M, N, K)
    if N % 256 == 0:
        w(f"h11_o128_{sn}", M, N, K, bM=128, bN=256)
for sn in ("6k", "16k"):
    M, N, K = SH[sn]
    for a, kw in ARMS.items():
        w(f"h11_{a}_{sn}", M, N, K, bM=128, bN=256, **kw)
json.dump(dyn, open("h11/dyn.json", "w"), indent=0)
print(len(dyn))
