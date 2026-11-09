# N4 round 5 job D: hand-scheduled asm from gp_r5.py. Blocks are grid cells; fill = B / (110 * ceil(B / 110)) is the hypothesis under test.
import json
import gp_r5 as g
c8, o = dict(bM=256, bN=128), dict(bM=128, bN=256)
a, b = dict(bM=256, bN=192, pad=16), dict(bM=192, bN=256, pad=16)
t = lambda m, n: dict(bM=m, bN=n)
P = {  # tag: (M, N, K, arms)
 "6x12": (6144, 12288, 4096, dict(c8=c8, o=o, a=a)), "12x6": (12288, 6144, 4096, dict(c8=c8, o=o, a=a)), "4608": (4608, 4608, 4608, dict(c8=c8, o=o, a=a)), "9216": (9216, 9216, 9216, dict(c8=c8, o=o, a=a)),
 "8x3": (8192, 3072, 4096, dict(c8=c8, o=o, a=a)), "3x8": (3072, 8192, 4096, dict(c8=c8, o=o, b=b)),
 "b240": (1024, 7680, 2048, dict(c8=c8, o=o)), "b280": (1280, 7168, 2048, dict(c8=c8, o=o)), "b300": (1536, 6400, 2048, dict(c8=c8, o=o)), "b330": (1280, 8448, 2048, dict(c8=c8, o=o)),
 "1x8": (1024, 8192, 2048, dict(c8=c8, o=o, q=t(128, 128))), "8x1": (8192, 1024, 2048, dict(c8=c8, o=o, q=t(128, 128))), "2x4": (2048, 4096, 2048, dict(c8=c8, o=o, q=t(128, 128))), "4x2": (4096, 2048, 2048, dict(c8=c8, o=o, q=t(128, 128))),
 "2k": (2048, 2048, 2048, dict(o=o, q=t(128, 128), p64=t(64, 128))), "1k": (1024, 1024, 1024, dict(o=o, q=t(128, 128), p64=t(64, 128), p32=t(32, 128))),
 "128m": (128, 5120, 5120, dict(q=t(128, 128), p64=t(64, 128), p32=t(32, 128))),
 "64m": (64, 5120, 5120, dict(x=t(32, 128), y=t(64, 128), z=t(32, 64))), "32m": (32, 5120, 5120, dict(x=t(32, 128), z=t(32, 64))), "64w": (64, 17408, 5120, dict(x=t(32, 128), y=t(64, 128), v=t(64, 256))),
 "64K": (64, 5120, 17408, dict(x=t(32, 128), y=t(64, 128), z=t(32, 64))), "32w": (32, 17408, 5120, dict(x=t(32, 128), w=t(32, 256))),
}
L = []
for s, (M, N, K, arms) in P.items():
    it = min(6000, max(4, int(0.4 / (2 * M * N * K / 8e13)))); w = it // 20 + 1; o_ = []
    for x, kw in arms.items():
        n = f"r5d_{x}_{s}"; src, d = g.hand_source(M, N, K, entry=n, **kw); open(f"r5/{n}.cu", "w").write(src)
        o_.append(f"{n},r5/{n}.co,{kw['bM']},{kw['bN']},256,{d}")
    L.append(f"{s} {M} {N} {K} {it} {w} " + " ".join(o_))
open("r5/listd.txt", "w").write("\n".join(L) + "\n"); print(len(L))
