"""Compile only, no GPU: gate/up th arms, T-row sums, FP8 and FP4 kernels. Reads VGPR and occupancy from resource usage."""
import glob, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_fused as M
_co = subprocess.check_output
subprocess.check_output = lambda c, *a, **k: b"Name: gfx90a\n" if isinstance(c, list) and str(c[0]).endswith("rocminfo") else _co(c, *a, **k)
import n1grid, n1fp
H, I, E, topk = 2048, 512, 256, 8
Tn = int(sys.argv[1])
N = Tn * topk
ILP = M._ILP
def dirs():
    return {os.path.dirname(p) for p in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/device_kernel.cu", recursive=True)}
def one(name, make, oi, fl=None):
    t0 = time.time(); b = dirs()
    try:
        M._build(make(), oi, fl)
    except BaseException as ex:
        print(json.dumps({"kind": "build_failed", "arm": name, "err": (type(ex).__name__ + ": " + str(ex))[-900:]}), flush=True)
        return
    res = [json.load(open(d + "/resource_usage.json"))["main_kernel"]["extra"] for d in dirs() - b]
    print(json.dumps({"kind": "built", "arm": name, "s": round(time.time() - t0, 1), "res": res}), flush=True)
c = dict(M._LARGE); P = M.pmax_rows(N, E, c["bM"])
if "fponly" not in sys.argv:
    for th in (256, 512, 128):
        one(f"gu_square_th{th}_ilp", lambda: M.gate_up(Tn, topk, H, I, E, P, 64, 64, 64, th, 2, "last", "square"), [4], ILP)
    one("sum_dyn", lambda: n1grid.sum_dyn(Tn, topk, H, 512), [1])
    one("sum_exit", lambda: n1grid.sum_exit(Tn, topk, H, 512), [1])
for fmt in ("fp8", "fp4"):
    one(f"gu_{fmt}", lambda: n1fp.gate_up_q(fmt, Tn, topk, H, I, E, P, 64, 64, 64, 256, 2, "square"), [8], ILP)
    one(f"dn_{fmt}", lambda: n1fp.down_q(fmt, Tn, topk, H, I, E, P, 64, 64, 128, 256, 2), [9])
