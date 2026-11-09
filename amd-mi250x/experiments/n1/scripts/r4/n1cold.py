"""Run a whole MoE call from prebuilt gfx90a code objects through hipModule*, in whatever runtime this process has. A directory
holds ka, kb, k1, k2, ks (.co) and meta.json from n1co.py. Same call and same stage marks as moe_fused.FusedMoE, so a clang 18
object can be timed in the ROCm 7.2.3 image next to vLLM. Argument order and launch geometry come from the generated source."""
import ctypes as C, json, os
import torch
_hip = None
def hip():
    global _hip
    if _hip is None:
        for n in ("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"):
            try:
                h = C.CDLL(n); break
            except OSError:
                h = None
        h.hipModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        h.hipModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        h.hipModuleLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        _hip = h
    return _hip

class Kern:
    def __init__(self, path, m):
        self.img = C.create_string_buffer(open(path, "rb").read())
        self.mod, self.f = C.c_void_p(), C.c_void_p()
        h = hip()
        rc = h.hipModuleLoadData(C.byref(self.mod), self.img)
        if rc: raise RuntimeError(f"hipModuleLoadData rc {rc} {path}")
        rc = h.hipModuleGetFunction(C.byref(self.f), self.mod, b"main_kernel")
        if rc: raise RuntimeError(f"hipModuleGetFunction rc {rc} {path}")
        self.args, self.g = m["args"], m["launch"]

    def __call__(self, **t):
        g = self.g
        ptrs = [C.c_void_p(t[n].data_ptr()) for n in self.args]
        arr = (C.c_void_p * len(ptrs))(*[C.addressof(p) for p in ptrs])
        rc = hip().hipModuleLaunchKernel(self.f, g[0], g[1], 1, g[2], 1, 1, g[3], torch.cuda.current_stream().cuda_stream, arr, None)
        if rc: raise RuntimeError(f"launch rc {rc}")

class COMoE:
    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        for k in ("ka", "kb", "k1", "k2", "ks"):
            setattr(self, k, Kern(os.path.join(d, k + ".co"), self.meta["kernels"][k]))
        self.cfg = self.meta["cfg"]

    def __call__(self, X, W13, W2, ids, wts, rec=None):
        rec = rec or (lambda i: None)
        M = self.meta
        P, NB, E_, N, Tn, H, I = M["P"], M["NB"], M["E"], M["N"], M["T"], M["H"], M["I"]
        dev = X.device
        idf, wf = ids.flatten(), wts.flatten()
        rec(0)
        s = torch.empty((P,), dtype=torch.int32, device=dev)
        b = torch.empty((NB,), dtype=torch.int32, device=dev)
        c = torch.empty((E_,), dtype=torch.int32, device=dev)
        self.ka(Bexp=b, Cur=c, Ids=idf, Sid=s)
        self.kb(Cur=c, Ids=idf, Sid=s)
        rec(1)
        inter = torch.empty((P, I), dtype=torch.float16, device=dev)
        self.k1(Bexp=b, Inter=inter, Sid=s, W13=W13, X=X)
        rec(2)
        y = torch.empty((N + 1, H), dtype=torch.float16, device=dev)
        self.k2(Bexp=b, Inter=inter, Sid=s, W2=W2, Wt=wf, Y=y)
        rec(3)
        o = torch.empty((Tn, H), dtype=torch.float16, device=dev)
        self.ks(O=o, Y=y)
        rec(4)
        return o
