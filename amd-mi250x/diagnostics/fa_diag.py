"""Step-by-step diagnosis of the TileLang AMD flash-attention mismatch on gfx90a (MI250X).
One block of block_M query rows against block_N keys. Each kernel isolates one step; all compared to torch.
  gemm1   : S = Q K^T                     (shared x shared, FullRow, k_pack)
  reduce  : rowmax/rowsum of S, taken directly on the GEMM-output fragment
  pv_reg  : O = fp16(S) V with P in registers (acc_s -> acc_s_cast copy, register-A GEMM)  [what the example does]
  pv_smem : O = fp16(S) V with P through shared memory (shared x shared GEMM)
"""
import json, sys, itertools, torch, tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy

FR = GemmWarpPolicy.FullRow


@tilelang.jit(out_idx=[2])
def k_gemm1(bM, bN, D, threads, kp):
    @T.prim_func
    def main(Q: T.Tensor((bM, D), "float16"), K: T.Tensor((bN, D), "float16"), S: T.Tensor((bM, bN), "float32")):
        with T.Kernel(1, threads=threads) as bx:
            qs = T.alloc_shared((bM, D), "float16"); ks = T.alloc_shared((bN, D), "float16")
            s = T.alloc_fragment((bM, bN), "float32")
            T.copy(Q, qs); T.copy(K, ks); T.clear(s)
            T.gemm(qs, ks, s, transpose_B=True, k_pack=kp, policy=FR)
            T.copy(s, S)
    return main


@tilelang.jit(out_idx=[2, 3])
def k_reduce(bM, bN, D, threads, kp):
    @T.prim_func
    def main(Q: T.Tensor((bM, D), "float16"), K: T.Tensor((bN, D), "float16"), Mx: T.Tensor((bM,), "float32"), Sm: T.Tensor((bM,), "float32")):
        with T.Kernel(1, threads=threads) as bx:
            qs = T.alloc_shared((bM, D), "float16"); ks = T.alloc_shared((bN, D), "float16")
            s = T.alloc_fragment((bM, bN), "float32")
            m = T.alloc_fragment((bM,), "float32"); r = T.alloc_fragment((bM,), "float32")
            T.copy(Q, qs); T.copy(K, ks); T.clear(s)
            T.gemm(qs, ks, s, transpose_B=True, k_pack=kp, policy=FR)
            T.fill(m, -T.infinity("float32"))
            T.reduce_max(s, m, dim=1, clear=False)
            T.reduce_sum(s, r, dim=1)
            T.copy(m, Mx); T.copy(r, Sm)
    return main


@tilelang.jit(out_idx=[3])
def k_pv_reg(bM, bN, D, threads, kp):
    @T.prim_func
    def main(Q: T.Tensor((bM, D), "float16"), K: T.Tensor((bN, D), "float16"), V: T.Tensor((bN, D), "float16"), O: T.Tensor((bM, D), "float32")):
        with T.Kernel(1, threads=threads) as bx:
            qs = T.alloc_shared((bM, D), "float16"); ks = T.alloc_shared((bN, D), "float16"); vs = T.alloc_shared((bN, D), "float16")
            s = T.alloc_fragment((bM, bN), "float32"); p = T.alloc_fragment((bM, bN), "float16")
            o = T.alloc_fragment((bM, D), "float32")
            T.copy(Q, qs); T.copy(K, ks); T.copy(V, vs); T.clear(s); T.clear(o)
            T.gemm(qs, ks, s, transpose_B=True, k_pack=kp, policy=FR)
            T.copy(s, p)
            T.gemm(p, vs, o, policy=FR)
            T.copy(o, O)
    return main


@tilelang.jit(out_idx=[3])
def k_pv_smem(bM, bN, D, threads, kp):
    @T.prim_func
    def main(Q: T.Tensor((bM, D), "float16"), K: T.Tensor((bN, D), "float16"), V: T.Tensor((bN, D), "float16"), O: T.Tensor((bM, D), "float32")):
        with T.Kernel(1, threads=threads) as bx:
            qs = T.alloc_shared((bM, D), "float16"); ks = T.alloc_shared((bN, D), "float16"); vs = T.alloc_shared((bN, D), "float16")
            ps = T.alloc_shared((bM, bN), "float16")
            s = T.alloc_fragment((bM, bN), "float32"); o = T.alloc_fragment((bM, D), "float32")
            T.copy(Q, qs); T.copy(K, ks); T.copy(V, vs); T.clear(s); T.clear(o)
            T.gemm(qs, ks, s, transpose_B=True, k_pack=kp, policy=FR)
            T.copy(s, ps)
            T.gemm(ps, vs, o, policy=FR)
            T.copy(o, O)
    return main


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9)).item()


torch.manual_seed(0)
cfgs = [(64, 64, 128, 256, 2), (64, 32, 128, 256, 2), (64, 64, 128, 256, 1), (128, 32, 128, 256, 2), (64, 64, 128, 128, 1), (32, 32, 64, 64, 1)]
if len(sys.argv) > 1:
    cfgs = [tuple(int(x) for x in sys.argv[1].split(","))]
for bM, bN, D, th, kp in cfgs:
    Q = torch.randn(bM, D, device="cuda", dtype=torch.float16) * 0.5
    K = torch.randn(bN, D, device="cuda", dtype=torch.float16) * 0.5
    V = torch.randn(bN, D, device="cuda", dtype=torch.float16)
    S_ref = Q.float() @ K.float().T
    r = {"bM": bM, "bN": bN, "D": D, "threads": th, "k_pack": kp}
    for name, fn in (("gemm1", lambda: rel(k_gemm1(bM, bN, D, th, kp)(Q, K), S_ref)),
                     ("reduce", lambda: (lambda o: max(rel(o[0], S_ref.max(1).values), rel(o[1], S_ref.sum(1))))(k_reduce(bM, bN, D, th, kp)(Q, K))),
                     ("pv_reg", lambda: rel(k_pv_reg(bM, bN, D, th, kp)(Q, K, V), S_ref.half().float() @ V.float())),
                     ("pv_smem", lambda: rel(k_pv_smem(bM, bN, D, th, kp)(Q, K, V), S_ref.half().float() @ V.float()))):
        try:
            r[name] = round(fn(), 6)
        except Exception as e:  # noqa
            r[name] = "ERR " + (type(e).__name__ + ": " + str(e))[-200:]
    print(json.dumps(r), flush=True)
