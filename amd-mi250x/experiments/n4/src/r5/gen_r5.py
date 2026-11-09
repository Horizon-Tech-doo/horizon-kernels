# N4 round 5: standalone hand kernels (hand-scheduled asm, not compiler output) from gp_r5.py, a test copy of the LUMI generator with only the sorted((bM,bN)) gate removed.
import json
import gp_r5 as g
TILE = {"3k": (3072, 3072, 3072), "6k": (6144, 6144, 6144), "12k": (12288, 12288, 12288)}
LOWK = {"lk192": (4096, 4096, 192), "lk512": (4096, 4096, 512), "lk1k": (4096, 4096, 1024), "lk2560": (2560, 2560, 256)}
ARMS = {"c8": dict(bM=256, bN=128), "o128": dict(bM=128, bN=256), "a16": dict(bM=256, bN=192, pad=16), "a32b0": dict(bM=256, bN=192, pad=32, pad_b=0),
        "a0": dict(bM=256, bN=192, pad=0), "b16": dict(bM=192, bN=256, pad=16)}
dyn = {}
def w(n, M, N, K, **kw):
    src, d = g.hand_source(M, N, K, entry=n, **kw)
    open(f"r5/{n}.cu", "w").write(src)
    dyn[n] = dict(dyn=d, bM=kw["bM"], bN=kw["bN"])
for sn, (M, N, K) in TILE.items():
    for a, kw in ARMS.items():
        w(f"r5_{a}_{sn}", M, N, K, **kw)
for sn, (M, N, K) in LOWK.items():
    for a in ("c8", "o128"):
        w(f"r5_{a}_{sn}", M, N, K, **ARMS[a])
json.dump(dyn, open("r5/dyn.json", "w"), indent=0)
print(len(dyn))
