"""K2b race. One process, randn data, arm order rotated per round, ratio = ours / hipBLASLt through torch in the same process and round, throughput, higher is better.
Arms: pass  = plain TileLang kernel with T.Kernel(gemm_schedule="pgr2"), schedule emitted by the compiler pass
      hand  = gemm_prefetch.gemm_hand, the hand kernel launched by prelude (compared bit for bit on the output)
      llvm  = the same plain kernel without the option, T.annotate_min_blocks_per_sm(2), LLVM's schedule
Usage: k3.py --compile | k3.py --tag r1 [--rounds 3] [--shapes ...]"""
import argparse, json, os, sys, time
sys.path.insert(0, "/scratch/project_XXXXXXXXX/tilelang/scripts")
ap = argparse.ArgumentParser()
ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="c"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--shapes", default="")
a = ap.parse_args()
if a.compile:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import tilelang
import tilelang.rocm.language as T
import gemm_prefetch as gp

SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (16384, 16384, 16384), (4096, 17408, 5120), (32768, 1024, 2048), (6144, 6144, 6144), (12288, 12288, 12288), (5632, 3200, 4096), (16384, 4096, 4096)]
if a.shapes:
    SHAPES = [tuple(map(int, s.split("x"))) for s in a.shapes.split(",")]


def plain(M, N, K, bM, bN, sched, occ):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=256, gemm_schedule=sched) as (bx, by):
            if occ:
                T.annotate_min_blocks_per_sm(occ)
            A_s = T.alloc_shared((bM, 64), T.float16)
            B_s = T.alloc_shared((bN, 64), T.float16)
            C_l = T.alloc_fragment((bM, bN), T.float32)
            T.clear(C_l)
            for k in T.serial(T.ceildiv(K, 64)):
                T.copy(A[by * bM, k * 64], A_s)
                T.copy(B[bx * bN, k * 64], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * bM, bx * bN])
    return main


def arms(M, N, K):
    bM, bN = gp.hand_tile(M, N, K)
    return {"pass": lambda: tilelang.compile(plain(M, N, K, bM, bN, "pgr2", 0), out_idx=[-1]),
            "hand": lambda: gp.gemm_hand(M, N, K, bM, bN),
            "llvm": lambda: tilelang.compile(plain(M, N, K, bM, bN, None, 2), out_idx=[-1])}, (bM, bN)


if a.compile:
    for sh in SHAPES:
        ar, t = arms(*sh)
        for n, f in ar.items():
            k = f(); r = dict(shape=sh, tile=t, arm=n)
            if n == "pass":
                r["pass_fired"] = "hand_body" in k.get_kernel_source()
            for v in dict(getattr(k, "resource_usage", {}) or {}).values():
                r.update(v=v.extra.get("VGPRs"), a=v.extra.get("AGPRs"), scratch=v.scratch_bytes, occ=v.extra.get("Occupancy [waves/SIMD]"))
            print(json.dumps(r), flush=True)
    sys.exit(0)

print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), tilelang=tilelang.__file__, tag=a.tag)), flush=True)


def tm(fn, it):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(it):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / it


rot = sum(map(ord, a.tag))
for (M, N, K) in SHAPES:
    g = torch.Generator().manual_seed(M + N + K)
    A = torch.randn(M, K, generator=g, dtype=torch.float32).to(device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, generator=g, dtype=torch.float32).to(device="cuda", dtype=torch.float16)
    flops = 2.0 * M * N * K
    it = max(4, min(30, int(4e12 / flops * 30) or 4))
    ar, tile = arms(M, N, K)
    ks = {n: f() for n, f in ar.items()}
    Bt = B.T
    ref = (A.float() @ B.float().T)
    outs = {}
    for n, k in ks.items():
        O = k(A, B)
        outs[n] = O
        rec = dict(tag=a.tag, shape=[M, N, K], tile=tile, arm=n, rel_err=((O.float() - ref).norm() / ref.norm()).item(), pass_fired=("hand_body" in k.get_kernel_source()))
        if n != "hand":
            rec["bits_equal_hand"] = None
        print(json.dumps(rec), flush=True)
    print(json.dumps(dict(tag=a.tag, shape=[M, N, K], check="pass_vs_hand_bitwise", equal=bool(torch.equal(outs["pass"], outs["hand"])))), flush=True)
    del outs, ref
    torch.cuda.empty_cache()
    names = list(ks)
    for r in range(a.rounds):
        order = ["hb"] + names[(rot + r) % len(names):] + names[:(rot + r) % len(names)]
        order = order[(r % len(order)):] + order[:(r % len(order))]
        res = {}
        for n in order:
            fn = (lambda: A @ Bt) if n == "hb" else (lambda k=ks[n]: k(A, B))
            for _ in range(3):
                fn()
            res[n] = tm(fn, it)
        print(json.dumps(dict(tag=a.tag, shape=[M, N, K], tile=tile, round=r, order=order, it=it, tf={n: round(flops / v * 1e-9, 2) for n, v in res.items()}, ratio={n: round(res["hb"] / res[n], 4) for n in names})), flush=True)
    del A, B, Bt, ks
    torch.cuda.empty_cache()
