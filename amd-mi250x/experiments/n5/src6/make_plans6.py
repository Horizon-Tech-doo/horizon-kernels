"""Round 6 plans. forms.json: C2P source forms at 2816, 3072, 3584, 4096 (C3_ARMS). tile22.json: hand-edited rows 256 vs rows 128 vs P32, image. l18.json: clang 18 rows 128 lb4 vs P32."""
import json, os
os.makedirs("plans", exist_ok=True)
Ls = (2816, 3072, 3584, 4096)
A = []
for L in Ls:
    c = f"d128_{L}"
    A += [[c, 0, 64, 512, 128, "c", "JDRF"], [c, 0, 32, 1024, 256, "c", "SDRF"], [c, 0, 32, 1024, 256, "c", "JDRF"], [c, 0, 32, 512, 128, "c", "SDRF"]]
json.dump(A, open("plans/forms.json", "w"))
T = []
for L in (3072, 4096, 8192, 16384):
    c = f"d128_{L}"
    T += [dict(case=c, P=0, bN=32, th=1024, rows=256), dict(case=c, P=0, bN=32, th=512, rows=128), dict(case=c, P=32, bN=96, th=768)]
json.dump(T, open("plans/tile22.json", "w"))
X = []
for L in (2048, 2816, 3072, 4096, 4160):
    c = f"d128_{L}"
    X += [dict(case=c, P=0, bN=32, th=512, rows=128, lb=4), dict(case=c, P=32, bN=96, th=768)]
json.dump(X, open("plans/l18.json", "w"))
print(len(A), len(T), len(X))
# After the first two jobs: rows 256 against the table's pick at the neighbouring table lengths, so the table does not leave untimed holes around the changed entries.
B = []
for L, pick in ((5120, dict(P=0, bN=32, th=512, rows=128)), (6144, dict(P=0, bN=32, th=512, rows=128)), (10240, dict(P=32, bN=96, th=768)), (12288, dict(P=32, bN=96, th=768))):
    c = f"d128_{L}"
    B += [dict(case=c, P=0, bN=32, th=1024, rows=256), dict(case=c, **pick)]
json.dump(B, open("plans/tile22b.json", "w"))
