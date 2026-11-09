"""Landed module after round 3. Compiles or loads and checks: default fast kernels unchanged, lb on clang 18. Prints registers and occupancy from the cache's resource_usage.json."""
import glob, json, os, sys, time
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
if len(sys.argv) > 1 and sys.argv[1] == "--compile":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import gqa_fa
from n5lib import CASES, inputs, ref
A = [("d128_4096", 0, 32, 512, 128, True, True, 0), ("d128_4096", 32, 96, 768, 0, True, True, 0), ("d256_8192", 32, 32, 768, 0, True, True, 0), ("mha_4096", 160, 32, 640, 0, False, False, 0), ("mha_4096", 160, 32, 640, 0, False, False, 4), ("mha_8192", 128, 32, 512, 0, False, False, 0), ("mha_8192", 128, 32, 512, 0, False, False, 4)]
if os.environ.get("N5_STACK") == "image":
    A = [a for a in A if not a[7]]
cdir = os.environ["TILELANG_CACHE_DIR"] + "/0.1.15/linux-x86_64/kernels"
for case, P, bN, th, rows, rev, swap, lb in A:
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    t0 = time.time()
    k = gqa_fa.gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, th, 2, True, rev, True, swap, rows, lb)
    r = dict(case=case, P=P, bN=bN, th=th, rows=rows, lb=lb, stack=os.environ.get("N5_STACK"))
    fs = sorted(glob.glob(cdir + "/*/resource_usage.json"), key=os.path.getmtime)
    if fs and os.path.getmtime(fs[-1]) >= t0 - 1:
        ru = json.load(open(fs[-1]))["main_kernel"]
        r.update(vgprs=ru["extra"].get("VGPRs"), occ=ru["extra"].get("Occupancy [waves/SIMD]"), scratch=ru["scratch_bytes"])
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        Q, K_, V = inputs("randn0", s)
        O = k(Q, K_, V)
        R = ref(Q, K_, V, causal, G)
        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
        r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K_, V), O)) for _ in range(20))}/20"
    print(json.dumps(r), flush=True)
