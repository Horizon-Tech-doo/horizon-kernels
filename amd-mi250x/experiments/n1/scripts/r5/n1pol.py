"""Compile only, no GPU. gate/up or down for fixed T under a warp policy, with gfx90a disassembly and resource usage.
Usage: n1pol.py OUT T stack  (combos below). Needs a TILELANG_CACHE_DIR used by this process only."""
import glob, json, os, re, struct, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
out, Tn, stack = sys.argv[1], int(sys.argv[2]), sys.argv[3]
import moe_fused as M
_co = subprocess.check_output
subprocess.check_output = lambda c, *a, **k: b"Name: gfx90a\n" if isinstance(c, list) and str(c[0]).endswith("rocminfo") else _co(c, *a, **k)      # no GPU here
H, I, E, topk = 2048, 512, 256, 8
clang = M.clang_major()
ILP = ("-mllvm", "-misched=gcn-max-ilp"); U = ("-mllvm", "-unroll-threshold=20000")
FL = {"none": None, "ilp": ILP} if clang >= 20 else {"none": None, "u20k": U}
GU = {"B6": dict(bM=128, bN=128, bK=64, th=512), "L": dict(bM=64, bN=64, bK=64, th=256)}
DN = {"d64": dict(bM=64, dN=64, dK=128, th=256), "d128": dict(bM=128, dN=64, dK=128, th=256)}
os.makedirs(out, exist_ok=True)
objdump = next(p for p in ("/opt/rocm/llvm/bin/llvm-objdump",) if os.path.exists(p))
N = Tn * topk
def dirs():
    return {os.path.dirname(p) for p in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/device_kernel.cu", recursive=True)}
def dump(new, name):
    for dd in new:
        data = open(dd + "/executable.so", "rb").read()
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__")
        while i >= 0:
            c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
            for _ in range(c):
                off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
                tr = data[p:p + tl].decode(errors="replace"); p += tl
                if "gfx90a" in tr and sz > 0:
                    tmp = f"{out}/{name}.co"; open(tmp, "wb").write(data[i + off:i + off + sz])
                    open(f"{out}/{name}.dis", "w").write(subprocess.run([objdump, "-d", tmp], capture_output=True, text=True).stdout)
                    res = json.load(open(dd + "/resource_usage.json"))["main_kernel"]["extra"]
                    print(json.dumps({"kind": "built", "stack": stack, "clang": clang, "arm": name, "res": res}), flush=True)
                    os.remove(tmp)
            i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)
def one(name, make, fl):
    t0 = time.time(); before = dirs()
    try:
        M._build(make(), [4] if name.startswith("gu") else [5], fl)
    except BaseException as ex:
        print(json.dumps({"kind": "build_failed", "stack": stack, "clang": clang, "arm": name, "err": (type(ex).__name__ + ": " + str(ex))[-600:]}), flush=True)
        return
    new = dirs() - before
    dump(new, name)
    print(json.dumps({"kind": "t", "arm": name, "s": round(time.time() - t0, 1), "new_dirs": len(new)}), flush=True)
for tn, c in GU.items():
    P = M.pmax_rows(N, E, c["bM"])
    for pol in ("row", "square"):
        for fn, fl in FL.items():
            one(f"gu_{tn}_{pol}_{fn}", lambda: M.gate_up(Tn, topk, H, I, E, P, c["bM"], c["bN"], c["bK"], c["th"], 2, "last", pol), fl)
for tn, c in DN.items():
    P = M.pmax_rows(N, E, c["bM"])
    for pol in ("row", "square"):
        one(f"dn_{tn}_{pol}", lambda: M.down(Tn, topk, H, I, E, P, c["bM"], c["dN"], c["dK"], c["th"], 2, pol), None)
