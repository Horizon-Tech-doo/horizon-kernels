"""Writes plans/*.json for round 3."""
import json, os
os.makedirs("plans3", exist_ok=True)
st = []
for L in (4096, 8192, 16384):
    for s in (0, 144, 160, 176, 192, 224, 272):
        st.append(dict(case=f"d128_{L}", P=0, bN=32, th=512, rows=128, stride=s))
    for s in (0, 144, 160, 168):
        if s == 168: continue
        st.append(dict(case=f"d128_{L}", P=32, bN=96, th=768, stride=s))
for s in (0, 144, 160, 176, 192, 224):
    st.append(dict(case="d256_8192", P=32, bN=32, th=768, stride=s))
for s in (0, 144, 160, 176, 192, 224):
    st.append(dict(case="mha_4096", P=160, bN=32, th=640, rev=False, swap=False, stride=s))
for s in (0, 144, 160):
    st.append(dict(case="mha_4096", P=192, bN=96, th=768, rev=False, swap=False, stride=s))
json.dump(st, open("plans3/stride.json", "w"))
print(len(st))

M = dict(rev=False, swap=False)
gf = []
def g(L, P, bN, st=0):
    gf.append(dict(case=f"mha_{L}", P=P, bN=bN, th=P * 4, stride=st, **M))
g(4096, 160, 32); g(4096, 192, 96); g(4160, 160, 32); g(4320, 160, 32); g(4608, 176, 32); g(4608, 176, 64); g(5120, 192, 64); g(5120, 192, 96); g(5120, 160, 32); g(4608, 192, 96)
json.dump(gf, open("plans3/grid.json", "w"))
print(len(gf))

lb = []
for L in (4096, 16384):
    for n in (0, 2, 3, 4, 5, 6):
        lb.append(dict(case=f"d128_{L}", P=0, bN=32, th=512, rows=128, lb=n))
    for n in (0, 2, 3, 4):
        lb.append(dict(case=f"d128_{L}", P=32, bN=96, th=768, lb=n))
for n in (0, 2, 3, 4):
    lb.append(dict(case="d256_8192", P=32, bN=32, th=768, lb=n))
for n in (0, 2, 3, 4, 5):
    lb.append(dict(case="mha_4096", P=160, bN=32, th=640, lb=n, **M))
json.dump(lb, open("plans3/lb.json", "w"))
print(len(lb))

nx = []
for s_ in (0, 160, 192):
    nx.append(dict(case="d128_8192", P=0, bN=32, th=512, rows=128, stride=s_))
    nx.append(dict(case="d128_8192", P=32, bN=96, th=768, stride=s_))
    nx.append(dict(case="d256_8192", P=32, bN=32, th=768, stride=s_))
json.dump(nx, open("plans3/nox.json", "w"))
print(len(nx))

T = dict(d256=dict(P=32, bN=32, th=768), r128=dict(P=0, bN=32, th=512, rows=128), p32=dict(P=32, bN=96, th=768), m160=dict(P=160, bN=32, th=640, **M), m192=dict(P=192, bN=96, th=768, **M))
ba = []
for B, L in ((4, 1024), (16, 1024), (4, 4096), (2, 8192)):
    ba.append(dict(case=f"d256B{B}_{L}", **T["d256"]))
    ba.append(dict(case=f"d128B{B}_{L}", **T["r128"]))
    ba.append(dict(case=f"d128B{B}_{L}", **T["p32"]))
    ba.append(dict(case=f"mhaB{B}_{L}", **T["m160"]))
    ba.append(dict(case=f"mhaB{B}_{L}", **T["m192"]))
json.dump(ba, open("plans3/batch.json", "w"))
re_ = []
for L in (4096, 8192, 16384):
    re_ += [dict(case=f"d256_{L}", **T["d256"]), dict(case=f"d128_{L}", **T["r128"]), dict(case=f"d128_{L}", **T["p32"])]
re_ += [dict(case="mha_4096", **T["m160"]), dict(case="mha_4096", **T["m192"])]
json.dump(re_, open("plans3/real.json", "w"))
six = [dict(case="d256_8192", **T["d256"]), dict(case="d128_4096", **T["r128"]), dict(case="d128_4096", **T["p32"]), dict(case="mha_4096", **T["m160"]), dict(case="mha_4096", **T["m192"])]
json.dump(six, open("plans3/six.json", "w"))
print(len(ba), len(re_), len(six))

rg = []
for L in (4096, 8192, 16384):
    for ring in (False, True):
        rg.append(dict(case=f"d128_{L}", P=0, bN=32, th=512, rows=128, ring=ring))
        rg.append(dict(case=f"d128_{L}", P=32, bN=48, th=768, ring=ring))
    rg.append(dict(case=f"d128_{L}", P=32, bN=96, th=768))
for ring in (False, True):
    rg.append(dict(case="d256_8192", P=32, bN=32, th=768, ring=ring))
    rg.append(dict(case="mha_4096", P=160, bN=32, th=640, ring=ring, **M))
    rg.append(dict(case="mha_4096", P=192, bN=32, th=768, ring=ring, **M))
    rg.append(dict(case="mha_4096", P=192, bN=48, th=768, ring=ring, **M))
json.dump(rg, open("plans3/ring.json", "w"))
print(len(rg))

pw = [dict(case="d128_8192", **T["r128"]), dict(case="d128_8192", **T["p32"]), dict(case="d128_16384", **T["r128"]), dict(case="d128_16384", **T["p32"]), dict(case="d256_8192", **T["d256"]), dict(case="d256_16384", **T["d256"]), dict(case="mha_4096", **T["m160"]), dict(case="mha_8192", **T["m160"]), dict(case="mha_8192", **T["m192"])]
json.dump(pw, open("plans3/power.json", "w"))
lu = []
for s_ in (dict(), dict(lb=2), dict(lb=3), dict(lb=4), dict(stride=160), dict(ring=True)):
    lu.append(dict(case="d128_16384", **T["r128"], **s_))
    if not s_.get("ring"):
        lu.append(dict(case="d128_16384", **T["p32"], **s_))
lu += [dict(case="d128_16384", P=48, bN=32, th=768), dict(case="d128_16384", P=48, bN=96, th=768)]
json.dump(lu, open("plans3/lumi.json", "w"))
ms = []
for L, tl in ((2048, ("m160", "m192")), (8192, ("m160", "m192")), (16384, ("m160", "m192"))):
    for t in tl:
        ms.append(dict(case=f"mha_{L}", **T[t]))
json.dump(ms, open("plans3/mhal.json", "w"))
print(len(pw), len(lu), len(ms))

lg = []
for L in (8192, 16384):
    for rows in (128, 144, 160, 176, 192, 208, 224):
        for bN in (32, 48, 64, 96):
            if bN // 2 * 16 <= rows * 4:
                lg.append(dict(case=f"d128_{L}", P=0, bN=bN, th=rows * 4, rows=rows))
    for P in (128, 144, 160, 176, 192, 208):
        for bN in (32, 48, 64, 96):
            if bN // 2 * 16 <= P * 4:
                lg.append(dict(case=f"mha_{L}", P=P, bN=bN, th=P * 4, **M))
for rows in (128, 160, 176, 192, 208):
    for bN in (32, 48):
        if rows != 192 or bN != 32:
            lg.append(dict(case="d256_16384", P=0, bN=bN, th=rows * 4, rows=rows))
lg.append(dict(case="d256_16384", **T["d256"]))
json.dump(lg, open("plans3/lgrid.json", "w"))
print(len(lg))

json.dump([c for c in re_ if "16384" not in c["case"]], open("plans3/realerr.json", "w"))

fl = [dict(case="d128_8192", **T["r128"]), dict(case="d128_16384", **T["r128"]), dict(case="d128_8192", **T["p32"]), dict(case="d128_16384", **T["p32"]), dict(case="d128_4096", **T["r128"]), dict(case="mha_4096", **T["m160"]), dict(case="d256_8192", **T["d256"])]
json.dump(fl, open("plans3/flags.json", "w"))
m2 = []
for L in (8192, 16384):
    m2 += [dict(case=f"mha_{L}", P=128, bN=32, th=512, **M), dict(case=f"mha_{L}", P=160, bN=32, th=640, **M), dict(case=f"mha_{L}", P=176, bN=48, th=704, **M)]
json.dump(m2, open("plans3/mhal2.json", "w"))
lu2 = []
for s_ in (dict(), dict(lb=4)):
    for L in (4096, 8192):
        lu2.append(dict(case=f"d128_{L}", **T["r128"], **s_))
    lu2.append(dict(case="d128_4096", **T["p32"]))
    lu2.append(dict(case="d128_8192", **T["p32"]))
for s_ in (dict(), dict(lb=2), dict(lb=3), dict(lb=4), dict(lb=5)):
    lu2.append(dict(case="mha_4096", **T["m160"], **s_))
    lu2.append(dict(case="d256_8192", **T["d256"], **s_))
json.dump(lu2, open("plans3/lumi2.json", "w"))

lu3 = []
for L in (4096, 8192, 16384):
    for s_ in (dict(), dict(lb=4)):
        lu3.append(dict(case=f"mha_{L}", P=160, bN=32, th=640, **s_, **M))
        lu3.append(dict(case=f"mha_{L}", P=128, bN=32, th=512, **s_, **M))
        lu3.append(dict(case=f"mha_{L}", P=192, bN=96, th=768, **s_, **M))
json.dump(lu3, open("plans3/lumi3.json", "w"))
