# N4 round 5 job D: hand-scheduled asm from gp_r5.py. Blocks are grid cells; fill = B / (110 * ceil(B / 110)) is the hypothesis under test.
import json
import gp_r5 as g
c8, o = dict(bM=256, bN=128), dict(bM=128, bN=256)
a, b = dict(bM=256, bN=192, pad=16), dict(bM=192, bN=256, pad=16)
t = lambda m, n: dict(bM=m, bN=n)
q, p = t(128, 128), t(64, 128)
P = {"b240": (1024, 7680, 2048, dict(o=o, q=q, p=p)), "b300": (1536, 6400, 2048, dict(o=o, q=q, p=p)), "2x6": (2048, 6144, 2048, dict(o=o, a=a, q=q)), "6x2": (6144, 2048, 2048, dict(o=o, q=q)),
 "2304": (2304, 2304, 2304, dict(o=o, q=q, p=p)), "1536": (1536, 1536, 1536, dict(o=o, q=q, p=p)), "3k64": (3072, 3072, 3072, dict(a=a, q=q, p=p)), "5120": (5120, 5120, 5120, dict(o=o, q=q)), "2816": (2816, 2816, 2816, dict(o=o, q=q, p=p))}
L = []
for s, (M, N, K, arms) in P.items():
    it = min(6000, max(4, int(0.4 / (2 * M * N * K / 8e13)))); w = it // 20 + 1; o_ = []
    for x, kw in arms.items():
        n = f"r5e_{x}_{s}"; src, d = g.hand_source(M, N, K, entry=n, **kw); open(f"r5/{n}.cu", "w").write(src)
        o_.append(f"{n},r5/{n}.co,{kw['bM']},{kw['bN']},256,{d}")
    L.append(f"{s} {M} {N} {K} {it} {w} " + " ".join(o_))
open("r5/liste.txt", "w").write("\n".join(L) + "\n"); print(len(L))
