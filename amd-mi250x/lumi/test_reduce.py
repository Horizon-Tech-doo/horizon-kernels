"""Row max/sum via T.reduce_max / T.reduce_sum (warp-reduce path in tl_templates/hip/reduce.h, which uses
__builtin_amdgcn_wavefrontsize(); on ROCm 6.2 this goes through the gfx90a shim). Correctness vs torch."""
import json, torch, tilelang
import tilelang.rocm.language as T


@tilelang.jit(out_idx=[1, 2])
def rowreduce(M, N, bM, threads):
    @T.prim_func
    def main(X: T.Tensor((M, N), "float32"), Mx: T.Tensor((M,), "float32"), S: T.Tensor((M,), "float32")):
        with T.Kernel(T.ceildiv(M, bM), threads=threads) as bx:
            x = T.alloc_fragment((bM, N), "float32")
            m = T.alloc_fragment((bM,), "float32")
            s = T.alloc_fragment((bM,), "float32")
            T.copy(X[bx * bM, 0], x)
            T.reduce_max(x, m, dim=1)
            T.reduce_sum(x, s, dim=1)
            T.copy(m, Mx[bx * bM])
            T.copy(s, S[bx * bM])
    return main


for (M, N, bM, th) in [(1024, 64, 64, 256), (1024, 128, 128, 256), (1024, 256, 64, 128), (4096, 128, 128, 256)]:
    x = torch.randn(M, N, device="cuda")
    r = {"M": M, "N": N, "bM": bM, "threads": th}
    try:
        mx, s = rowreduce(M, N, bM, th)(x)
        r["max_err"] = (mx - x.max(1).values).abs().max().item()
        r["sum_err"] = (s - x.sum(1)).abs().max().item()
        r["ok"] = r["max_err"] < 1e-5 and r["sum_err"] < 1e-3
    except Exception as e:  # noqa
        r["ok"] = False; r["error"] = str(e)[-400:]
    print(json.dumps(r), flush=True)
