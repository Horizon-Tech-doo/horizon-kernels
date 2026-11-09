"""Compile one form on the login node and print source + resources. Usage: c1_src.py case P bN threads opts [rows] [flags]"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import tilelang, tlfa
print(tilelang.__file__, tilelang.__version__)
case, P, bN, th, opts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
rows = int(sys.argv[6]) if len(sys.argv) > 6 else 0
kind, L = case.split("_"); L = int(L)
Hkv, G, D, causal = (8, 1, 128, False) if kind == "mha" else (4, 6, int(kind[1:]), True)
k = tlfa.fwd(1, L, Hkv, G, D, causal, P, bN, th, opts, 2, True, True, True, True, rows)
print(k.get_kernel_source())
try:
    print(k.resource_usage)
except Exception as e:
    print("no resource_usage", e)
