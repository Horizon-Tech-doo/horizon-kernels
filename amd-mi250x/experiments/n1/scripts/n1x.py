"""Run raw gate/up code objects through hipModule* inside either stack, so compiler and runtime can be separated.
Same .co bytes, same process harness, same Sid/Bexp/X/W. Prints k1 median ms and rel error of real Inter rows against the
process's own TileLang kernel of the same path. Arg order is the generated one: Bexp, Inter, Sid, W13, X."""
import ctypes as C, glob, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
data, stack, codir = sys.argv[1:4]
D = torch.load(data)
W13, W2 = D["W13"].cuda(), D["W2"].cuda()
d = D[(4096, "uniform")]
X, ids = d["X"].cuda(), d["ids"].cuda()
Xz = torch.cat([X, torch.zeros(1, 2048, device=X.device, dtype=X.dtype)])
import moe_fused as M
m = M.FusedMoE(4096, x="zero", out="fp32")
s, be = m.align(ids.flatten())
N = 32768
ref = m.k1(Xz, W13, s, be)
real = (s < N).nonzero().flatten()
hip = None
for n in ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"):
    try:
        hip = C.CDLL(n); break
    except OSError:
        pass
hip.hipModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
hip.hipModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
hip.hipModuleLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
EV = lambda: torch.cuda.Event(enable_timing=True)
stream = torch.cuda.current_stream().cuda_stream
res = {}
cos = sorted(glob.glob(codir + "/*.co"))
keep = {}
for co in cos:
    name = os.path.basename(co)[:-3]
    info = json.load(open(co[:-3] + ".json"))
    g = info["grid"]
    img = C.create_string_buffer(open(co, "rb").read()); keep[name] = img
    mod = C.c_void_p(); f = C.c_void_p()
    rc1 = hip.hipModuleLoadData(C.byref(mod), img)
    rc2 = hip.hipModuleGetFunction(C.byref(f), mod, b"main_kernel") if rc1 == 0 else -1
    if rc1 or rc2:
        print(json.dumps({"stack": stack, "co": name, "load_rc": [rc1, rc2]}), flush=True); continue
    xx = Xz if "zero" in name else X
    keep[name + "f"] = (mod, f, g, xx)
for r in range(3):
    names = [k[:-1] for k in keep if k.endswith("f")]
    names = names[r % len(names):] + names[:r % len(names)]
    for name in names:
        mod, f, g, xx = keep[name + "f"]
        inter = torch.empty((s.numel(), 512), dtype=torch.float16, device="cuda")
        ptrs = [C.c_void_p(t.data_ptr()) for t in (be, inter, s, W13, xx)]
        arr = (C.c_void_p * 5)(*[C.addressof(p) for p in ptrs])
        def launch():
            rc = hip.hipModuleLaunchKernel(f, g[0], g[1], 1, g[2], 1, 1, g[5], stream, arr, None)
            assert rc == 0, rc
        evs = []
        for _ in range(5): launch()
        for _ in range(50):
            a, b = EV(), EV(); a.record(); launch(); b.record(); evs.append((a, b))
        torch.cuda.synchronize()
        t = torch.tensor([x.elapsed_time(y) for x, y in evs], dtype=torch.float64)
        err = ((inter[real].float() - ref[real].float()).norm() / ref[real].float().norm()).item()
        print(json.dumps({"stack": stack, "co": name, "round": r, "k1_med": round(t.median().item(), 4), "k1_min": round(t.min().item(), 4), "err_vs_native": round(err, 7), "grid": g}), flush=True)
    # same process, TileLang's own kernel as control
    a0 = []
    for _ in range(50):
        a, b = EV(), EV(); a.record(); m.k1(Xz, W13, s, be); b.record(); a0.append((a, b))
    torch.cuda.synchronize()
    t = torch.tensor([x.elapsed_time(y) for x, y in a0], dtype=torch.float64)
    print(json.dumps({"stack": stack, "co": "native_zero", "round": r, "k1_med": round(t.median().item(), 4)}), flush=True)
