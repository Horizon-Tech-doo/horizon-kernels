"""Writes plan.json. form: C2P TileLang source (tlfa.py) on fork-main, with opts. hand: exact bytes of landed gqa_flash_fast (gqa_fa.py a6e9732c) compiled on the wheel, launched in the same process through a fork-main wrapper."""
import json
A = []
def arm(L, bN, th, rows, kind, opts=""):
    A.append(dict(case=f"d128_{L}", L=L, bN=bN, th=th, rows=rows, kind=kind, opts=opts, name=f"d128_{L}_r{rows}_b{bN}_t{th}_{opts if kind == 'form' else 'hand'}"))
for L in (2816, 3072, 3584):
    arm(L, 64, 512, 128, "form", "JDRF"); arm(L, 32, 512, 128, "form", "SDRF"); arm(L, 32, 512, 128, "hand")   # hand: d128_tile's pick at these lengths
arm(4096, 64, 512, 128, "form", "JDRF"); arm(4096, 32, 512, 128, "hand")                                        # control
for L in (8192, 16384):
    arm(L, 64, 512, 128, "form", "JDRF"); arm(L, 32, 1024, 256, "form", "SDRF"); arm(L, 32, 1024, 256, "form", "JDRF"); arm(L, 32, 1024, 256, "hand")   # hand: d128_tile's pick
json.dump(A, open("plan.json", "w"), indent=0)
print(len(A))
