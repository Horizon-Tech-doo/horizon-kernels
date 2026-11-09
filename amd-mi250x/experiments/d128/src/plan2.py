"""plan2.json. Follows the LDS finding: wheel wrapper asks 4096 B more LDS per block than the fork wrapper. Hand rows 128 at bN 48 fits two blocks per CU on the wheel (28672 B); bN 64 does not (36864 B). Same arm fields as plan.py."""
import json
A = []
def arm(L, bN, th, rows, kind, opts=""):
    A.append(dict(case=f"d128_{L}", L=L, bN=bN, th=th, rows=rows, kind=kind, opts=opts, name=f"d128_{L}_r{rows}_b{bN}_t{th}_{opts if kind == 'form' else 'hand'}"))
for L in (3072, 8192, 16384):
    arm(L, 64, 512, 128, "form", "JDRF"); arm(L, 64, 512, 128, "hand"); arm(L, 48, 512, 128, "hand")
    if L == 3072:
        arm(L, 32, 512, 128, "hand")
    else:
        arm(L, 32, 1024, 256, "hand")
json.dump(A, open("plan2.json", "w"), indent=0)
print(len(A))
