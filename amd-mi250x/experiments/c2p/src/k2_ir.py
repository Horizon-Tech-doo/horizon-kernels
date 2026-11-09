import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import tilelang, tilelang.rocm.language as T
from tvm import IRModule
from tvm.target import Target
from tilelang.tileop.base import GemmWarpPolicy
M, N, K, bM, bN, bK = 8192, 8192, 8192, 256, 128, 64
@T.prim_func
def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
    with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=256) as (bx, by):
        T.use_swizzle(8, order="col") if 0 else None
        A_s = T.alloc_shared((bM, bK), T.float16)
        B_s = T.alloc_shared((bN, bK), T.float16)
        C_l = T.alloc_fragment((bM, bN), T.float32)
        T.clear(C_l)
        for k in T.serial(T.ceildiv(K, bK)):
            T.copy(A[by * bM, k * bK], A_s)
            T.copy(B[bx * bN, k * bK], B_s)
            T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=2)
        T.copy(C_l, C[by * bM, bx * bN])

mod = IRModule.from_expr(main)
tgt = Target(os.environ["TILELANG_DEFAULT_TARGET"]) if 0 else None
import tvm
from tvm import tirx
from tvm.target import Target
target = Target(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
mod = tirx.transform.BindTarget(target)(mod)
mod = tilelang.transform.MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=True, default_threads=128, unsupported_annotations=["cluster_dims"])(mod)

f = mod["main"]
def show(s, d=0):
    print(" " * d, type(s).__name__, end=" ")
    if isinstance(s, tirx.SeqStmt):
        print()
        for x in s.seq: show(x, d + 1)
    elif isinstance(s, tirx.SBlock):
        print(s.name_hint, list(s.alloc_buffers), dict(s.annotations or {}), [type(b) for b in s.match_buffers][:1]); show(s.body, d + 1)
    elif isinstance(s, tirx.For):
        print(s.loop_var, s.min, s.extent, s.kind); show(s.body, d + 1)
    elif isinstance(s, tirx.Evaluate):
        c = s.value; print(c.op.name, [type(a).__name__ for a in c.args], dict(c.annotations) if hasattr(c, "annotations") else "")
    elif hasattr(s, "body"):
        print({k: str(getattr(s, k))[:70] for k in ("var", "value", "attr_key", "node") if hasattr(s, k)}); show(s.body, d + 1)
    else:
        print(str(s)[:150])
show(f.body)
print(type(f.body), f.params)
print(tirx.Bind if hasattr(tirx,"Bind") else None)
seen=[]
def v(n):
    seen.append(type(n).__name__)
    if type(n).__name__ in ("Bind",):
        print("BIND", n.var, type(n.value), n.value)
    if type(n).__name__ == "SBlock":
        print("SB", n.name_hint, [str(i) for i in n.iter_vars], type(n.body).__name__)
    return True
tirx.stmt_functor.pre_order_visit(f.body, v)
import collections
print(collections.Counter(seen))
