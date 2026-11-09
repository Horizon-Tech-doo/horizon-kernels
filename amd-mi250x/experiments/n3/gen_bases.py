# compile TileLang paged decode kernels into the shared cache so gen_n3.py can patch them (no timing here)
import sys, json, torch
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py")
gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE = 4, 6, 256, 16
for B, L, S in [(32,8192,S) for S in (20,39,52,80,104,128)]+[(8,32768,S) for S in (110,220,512)]+[(16,16384,S) for S in (54,108,216)]+[(64,4096,S) for S in (10,20,40)]:
    MAXP = L // PAGE; NBLK = B * MAXP
    for th, bM in [(64,16),(128,32)]:
        gq.decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, S, th, PAGE, bM)
        print(B, L, S, th, bM, "ok", flush=True)
    gq.decode_combine(B, Hkv*G, S, D)
