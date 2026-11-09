# N4 round 5 job C: hand-scheduled asm from gp_r5.py. 256x256 at pad 0 (the only pad that fits 64 KiB), 256x192 on non-cubic multiples and at low K, and tuning of the 256x192 tile.
import json
import gp_r5 as g
S = {"3k": (3072, 3072, 3072), "4k": (4096, 4096, 4096), "6k": (6144, 6144, 6144), "8k": (8192, 8192, 8192), "12k": (12288, 12288, 12288), "16k": (16384, 16384, 16384),
     "8x6": (8192, 6144, 4096), "6x8": (6144, 8192, 4096), "6k1k": (6144, 6144, 1024), "3k256": (3072, 3072, 256), "6k256": (6144, 6144, 256)}
c8, o128, a16, b16 = dict(bM=256, bN=128), dict(bM=128, bN=256), dict(bM=256, bN=192, pad=16), dict(bM=192, bN=256, pad=16)
PLAN = {}
for s in ("3k", "4k", "6k", "8k", "12k", "16k"):
    PLAN[s] = {"c8": c8, "o128": o128, "s0": dict(bM=256, bN=256, pad=0)}
PLAN["8x6"] = {"c8": c8, "o128": o128, "a16": a16}
PLAN["6x8"] = {"c8": c8, "o128": o128, "b16": b16}
for s in ("6k1k", "3k256", "6k256"):
    PLAN[s] = {"c8": c8, "o128": o128, "a16": a16, "b16": b16}
TUNE = {"a0b32": dict(bM=256, bN=192, pad=0, pad_b=32), "a16c4": dict(a16, sw="c4"), "a16c16": dict(a16, sw="c16"), "a16r8": dict(a16, sw="r8"), "a16f15": dict(a16, fr=0.15), "a16f40": dict(a16, fr=0.4)}
dyn = {}
for s, arms in PLAN.items():
    M, N, K = S[s]
    for a, kw in arms.items():
        n = f"r5c_{a}_{s}"; src, d = g.hand_source(M, N, K, entry=n, **kw); open(f"r5/{n}.cu", "w").write(src); dyn[n] = dict(dyn=d, bM=kw["bM"], bN=kw["bN"])
for s in ("3k", "6k"):
    M, N, K = S[s]
    for a, kw in {**TUNE, "a16": a16}.items():
        n = f"r5t_{a}_{s}"; src, d = g.hand_source(M, N, K, entry=n, **kw); open(f"r5/{n}.cu", "w").write(src); dyn[n] = dict(dyn=d, bM=kw["bM"], bN=kw["bN"])
json.dump(dyn, open("r5/dync.json", "w"), indent=0)
print(len(dyn))
