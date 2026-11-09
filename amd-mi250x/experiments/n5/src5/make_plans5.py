"""Writes plans5/*.json. grid: d128 tile grid at 2816, 3072, 3584. Tiles needing more than one load step per thread (8 * bN > threads) cannot be built by gqa_flash_fast and are left out."""
import json, os
os.makedirs("plans5", exist_ok=True)
for L in (2816, 3072, 3584):
    g = []
    for r in (96, 112, 128, 144, 160):
        for b in (32, 48, 64):
            if 8 * b <= r * 4:
                g.append(dict(case=f"d128_{L}", P=0, bN=b, th=r * 4, rows=r))
    for P in (24, 32, 40):
        for b in (64, 96):
            if 8 * b <= P * 24:
                g.append(dict(case=f"d128_{L}", P=P, bN=b, th=P * 24))
    json.dump(g, open(f"plans5/grid{L}.json", "w"))
M = dict(rev=False, swap=False, bN=32)
p128 = dict(P=128, th=512, **M); p160 = dict(P=160, th=640, **M)
v22 = []
for c in ("mha_512", "mha_1024", "mha_1536", "mhaB2_1024", "mhaB2_2048", "mhaB4_512"):
    v22 += [dict(case=c, **p128), dict(case=c, **p160)]
json.dump(v22, open("plans5/v22.json", "w"))
ba = []
for c in ("mhaB4_1024", "mhaB16_1024", "mhaB4_4096", "mhaB2_8192"):
    ba += [dict(case=c, **p128), dict(case=c, **p160)]
json.dump(ba, open("plans5/batch.json", "w"))
ilp = dict(sched="gcn-max-ilp")
v18 = []
for c in ("mha_4608", "mha_5120", "mha_6144", "mhaB4_1024", "mhaB16_1024", "mhaB2_8192"):
    v18 += [dict(case=c, lb=4, **p128), dict(case=c, lb=4, **ilp, **p128), dict(case=c, lb=4, **p160), dict(case=c, lb=4, **ilp, **p160)]
json.dump(v18, open("plans5/v18.json", "w"))
r128 = dict(case="d128_16384", P=0, bN=32, th=512, rows=128)
json.dump([r128, dict(r128, lb=4), dict(case="d128_16384", P=32, bN=96, th=768)], open("plans5/attr.json", "w"))
json.dump([dict(r128, lb=4)], open("plans5/attr_ilp.json", "w"))
s = []
for f in ("sweep1.json", "sweep2.json"):
    for t in json.load(open("/scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/" + f)):
        if t[0] == "d128_16384" and t not in s:
            s.append(t)
json.dump(s, open("plans5/sweep16.json", "w"))
print(len(s), [len(json.load(open(f"plans5/grid{L}.json"))) for L in (2816, 3072, 3584)], len(v22), len(ba), len(v18))
l18 = []
for L in (1024, 1536, 2048, 2560, 2816, 3072, 3584, 4096, 4160, 5000, 5120, 6144, 8192, 10240, 12288, 16384):
    l18 += [dict(case=f"d128_{L}", P=0, bN=32, th=512, rows=128), dict(case=f"d128_{L}", P=32, bN=96, th=768)]
json.dump(l18, open("plans5/l18.json", "w"))
# Second validation: lengths where the corrected rule changes or has no timing.
Ls = ("mha_2048", "mha_2560", "mha_3072", "mha_3456", "mha_3584", "mha_4096")
json.dump([dict(case=c, **t) for c in Ls for t in (p128, p160)], open("plans5/v22b.json", "w"))
v18b = []
for c in Ls + ("mha_512", "mha_1024", "mha_1536", "mhaB2_1024", "mhaB2_2048", "mhaB4_512"):
    v18b += [dict(case=c, lb=4, **ilp, **p128), dict(case=c, lb=4, **p160), dict(case=c, lb=4, **ilp, **p160)]
json.dump(v18b, open("plans5/v18b.json", "w"))
# Clang 18 rule inputs: lb at the two lengths where rows 128 is picked, and the sweep's rows 256 bN32 tile in the hand-edited form, same process as P32 bN96.
la = []
for L in (1024, 2560):
    r = dict(case=f"d128_{L}", P=0, bN=32, th=512, rows=128)
    la += [r, dict(r, lb=4), dict(case=f"d128_{L}", P=32, bN=96, th=768)]
la += [dict(case="d128_16384", P=32, bN=96, th=768), dict(case="d128_16384", P=0, bN=32, th=1024, rows=256), dict(case="d128_16384", P=0, bN=64, th=512, rows=128)]
json.dump(la, open("plans5/la18.json", "w"))
