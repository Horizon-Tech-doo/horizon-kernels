"""Writes plans4/*.json for round 4."""
import json, os
os.makedirs("plans4", exist_ok=True)
M = dict(rev=False, swap=False)
d256 = dict(P=32, bN=32, th=768); r128 = dict(P=0, bN=32, th=512, rows=128); p32 = dict(P=32, bN=96, th=768)
p128 = dict(P=128, bN=32, th=512, **M); p160 = dict(P=160, bN=32, th=640, **M)
og = []
for L in (3072, 4096, 4160, 5000, 5120, 6144, 8192, 10240, 12288):      # 4096, 8192: on-grid controls in the same process pattern
    og += [dict(case=f"d256_{L}", **d256), dict(case=f"d128_{L}", **r128), dict(case=f"d128_{L}", **p32)]
json.dump(og, open("plans4/offgrid.json", "w"))
mh = []
for L in (4096, 4160, 4320, 4608, 5120, 6144):
    mh += [dict(case=f"mha_{L}", **p128), dict(case=f"mha_{L}", **p160)]
mh += [dict(case="mha_7168", **p128), dict(case="mha_8192", **p128)]
json.dump(mh, open("plans4/mhab.json", "w"))
ba = []
for B, L in ((4, 1024), (16, 1024), (4, 4096), (2, 8192)):
    ba += [dict(case=f"mhaB{B}_{L}", **p128), dict(case=f"mhaB{B}_{L}", **p160)]
json.dump(ba, open("plans4/batch.json", "w"))
cl = [dict(case="d128_16384", **r128, lb=4), dict(case="d128_16384", **p32), dict(case="mha_8192", **p128, lb=4), dict(case="mha_16384", **p128, lb=4)]
json.dump(cl, open("plans4/cl18.json", "w"))
co = [dict(case="mha_4096", **p160), dict(case="mha_4096", **p160, lb=4)]
json.dump(co, open("plans4/comb.json", "w"))
sh = []
for L in (1024, 1536, 2048, 2560, 3072):
    sh += [dict(case=f"d128_{L}", **r128), dict(case=f"d128_{L}", **p32)]
json.dump(sh, open("plans4/short.json", "w"))
print(len(og), len(mh), len(ba), len(cl), len(co))
