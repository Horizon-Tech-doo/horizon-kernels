"""Prints the tile each rule picks on this stack. CPU only."""
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gqa_fa
print("clang", gqa_fa.clang_major())
for L in (1024, 1536, 2048, 2304, 2560, 2816, 3072, 3584, 4096, 4128, 4160, 5000, 5120, 6144, 8192, 10240, 12288, 16384):
    t = gqa_fa.d128_tile(L); print("d128", L, t["P"], t.get("rows", 0), t["bN"], t["threads"], t.get("lb", 0))
for L, B in ((512, 1), (1024, 1), (1536, 1), (2048, 1), (2560, 1), (3072, 1), (3456, 1), (3584, 1), (4096, 1), (4320, 1), (4608, 1), (8192, 1), (16384, 1), (1024, 2), (2048, 2), (512, 4), (1024, 4), (1024, 16), (4096, 4), (8192, 2)):
    t = gqa_fa.mha_tile(L, B); print("mha", L, B, t["P"], t["threads"], t.get("lb", 0), t.get("sched", ""))
