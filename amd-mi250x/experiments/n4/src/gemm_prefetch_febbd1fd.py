"""Dense GEMM on gfx90a with hand-written overlap of global loads and MFMA work (Track 4; also the bottleneck of the
fused MoE and attention kernels).

On HIP, T.Pipelined(num_stages > 1) is stripped to a sequential loop; explicit order/stage are not stripped (compile only). The
hardware has a direct global->LDS load, CK's binaries contain it, we ran it (K1), and TileLang gates it, so every TileLang-generated
K step is: load A/B tiles, barrier, MFMA, barrier. Direct loads did not lead for GEMM (K1, N4). Variants:
  base      : the bench_gemm.py kernel (stages 0)
  regpref   : register prefetch: the next K tile is loaded global -> registers before the MFMAs of the current tile, then
              stored to LDS after them, so the global-load latency overlaps the math (the CK / "software pipelining
              without async copy" pattern; CK does this for V only. As generated the staging sits in scratch, so no overlap)
  dbuf      : two LDS buffers alternating, loads of tile k+1 into the other buffer before computing tile k
C[M,N] = A[M,K] @ B[N,K]^T, fp16, fp32 accumulate. Correctness vs torch fp32, speed vs torch hipBLASLt.
"""
import json
import sys

import torch
import tilelang
import tilelang.rocm.language as T


@tilelang.jit
def gemm_splitk(M, N, K, bM, bN, bK, threads, kp, S):
    """Small-M (decode) GEMM: grid (N/bN, M/bM, S); each block does K/S of the reduction and atomically adds its fp32
    partial into C32 (zeroed by the caller); the caller casts to fp16. Few rows means few blocks otherwise."""
    dtype, acc = T.float16, T.float32
    nk = K // bK // S

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C32: T.Tensor((M, N), acc)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), S, threads=threads) as (bx, by, bz):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            for k in T.serial(nk):
                kk = bz * nk + k
                T.copy(A[by * bM, kk * bK], A_s)
                T.copy(B[bx * bN, kk * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            for i, j in T.Parallel(bM, bN):
                T.atomic_add(C32[by * bM + i, bx * bN + j], C_l[i, j])

    return main


@tilelang.jit(out_idx=[-1])
def gemm(M, N, K, bM, bN, bK, threads, kp, variant, swizzle=0, order="row", pad=0):
    dtype, acc = T.float16, T.float32
    nk = K // bK
    VA = max(1, bM * bK // 8 // threads)          # 16-byte vectors per thread per tile (regstage)
    VB = max(1, bN * bK // 8 // threads)

    @T.prim_func
    def base(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK + pad), dtype)
            B_s = T.alloc_shared((bN, bK + pad), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            if swizzle:                          # concurrent blocks share A and B tiles: block order, not arithmetic
                T.use_swizzle(panel_size=swizzle, order=order)
            T.clear(C_l)
            for k in T.serial(nk):
                if pad:                          # row stride bK + pad halves in LDS: N4, a 128 B row stride is the slow one
                    T.copy(A[by * bM:by * bM + bM, k * bK:k * bK + bK], A_s[0:bM, 0:bK])
                    T.copy(B[bx * bN:bx * bN + bN, k * bK:k * bK + bK], B_s[0:bN, 0:bK])
                    T.gemm(A_s[0:bM, 0:bK], B_s[0:bN, 0:bK], C_l, transpose_B=True, k_pack=kp)
                else:
                    T.copy(A[by * bM, k * bK], A_s)
                    T.copy(B[bx * bN, k * bK], B_s)
                    T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def regpref(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            A_r = T.alloc_fragment((bM, bK), dtype)
            B_r = T.alloc_fragment((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            T.copy(A[by * bM, 0], A_r)
            T.copy(B[bx * bN, 0], B_r)
            for k in T.serial(nk):
                T.copy(A_r, A_s)
                T.copy(B_r, B_s)
                if k + 1 < nk:                       # issue the next tile's global loads before this tile's MFMAs
                    T.copy(A[by * bM, (k + 1) * bK], A_r)
                    T.copy(B[bx * bN, (k + 1) * bK], B_r)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def dbuf(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A0 = T.alloc_shared((bM, bK), dtype)
            B0 = T.alloc_shared((bN, bK), dtype)
            A1 = T.alloc_shared((bM, bK), dtype)
            B1 = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            T.copy(A[by * bM, 0], A0)
            T.copy(B[bx * bN, 0], B0)
            for kk in T.serial(nk // 2):
                k = kk * 2
                T.copy(A[by * bM, (k + 1) * bK], A1)
                T.copy(B[bx * bN, (k + 1) * bK], B1)
                T.gemm(A0, B0, C_l, transpose_B=True, k_pack=kp)
                if k + 2 < nk:
                    T.copy(A[by * bM, (k + 2) * bK], A0)
                    T.copy(B[bx * bN, (k + 2) * bK], B0)
                T.gemm(A1, B1, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    @T.prim_func
    def regstage(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        """Register-staged pipelining (CK style) with an explicit per-thread copy map: thread t owns 16-byte vectors
        v*threads + t of each tile, so no fragment-layout inference is involved. Tile k+1 is loaded global -> registers
        before the MFMAs of tile k and stored to LDS after them; the hardware overlaps the loads with the math."""
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            a_r = T.alloc_local((VA * 8,), dtype)
            b_r = T.alloc_local((VB * 8,), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            tx = T.get_thread_binding()
            T.clear(C_l)
            for v in T.serial(VA):
                for j in T.vectorized(8):
                    a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for v in T.serial(VB):
                for j in T.vectorized(8):
                    b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for k in T.serial(nk):
                for v in T.serial(VA):
                    for j in T.vectorized(8):
                        A_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = a_r[v * 8 + j]
                for v in T.serial(VB):
                    for j in T.vectorized(8):
                        B_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = b_r[v * 8 + j]
                if k + 1 < nk:
                    for v in T.serial(VA):
                        for j in T.vectorized(8):
                            a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                    for v in T.serial(VB):
                        for j in T.vectorized(8):
                            b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])

    return {"base": base, "regpref": regpref, "dbuf": dbuf, "regstage": regstage}[variant]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run_splitk():
    """Track 4 small-M sweep: M in {1, 16, 32, 64} x Qwen-like N/K, split-K S in {1, 2, 4, 8, 16}."""
    torch.manual_seed(0)
    for M, N, K in [(64, 5120, 5120), (32, 5120, 5120), (16, 17408, 5120), (64, 17408, 5120), (64, 5120, 17408), (1, 5120, 5120)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)
        ref = a.float() @ b.float().T
        tms = bench(lambda: a @ b.T)
        nbytes = (N * K + M * K) * 2
        print(json.dumps({"impl": "hipblaslt", "M": M, "N": N, "K": K, "ms": round(tms, 4), "gbps": round(nbytes / tms / 1e6, 1)}), flush=True)
        bM = 16 if M <= 16 else (32 if M <= 32 else 64)
        for bN, bK, th, S in [(64, 64, 64 if bM == 16 else 128, 1), (64, 64, 64 if bM == 16 else 128, 4), (64, 64, 64 if bM == 16 else 128, 8),
                              (128, 64, 128 if bM >= 32 else 64, 4), (64, 128, 64 if bM == 16 else 128, 4), (32, 128, 64, 8), (64, 64, 64 if bM == 16 else 128, 16)]:
            r = {"impl": "splitk", "M": M, "N": N, "K": K, "bM": bM, "bN": bN, "bK": bK, "threads": th, "S": S}
            if K % (bK * S):
                continue
            try:
                Mp = max(M, bM)
                ap = a if M == Mp else torch.cat([a, torch.zeros(Mp - M, K, device="cuda", dtype=a.dtype)])
                if S == 1:                      # no split: plain fp16 kernel, no fp32 zero/atomics/cast
                    k = gemm(Mp, N, K, bM, bN, bK, th, 1, "base")

                    def run():
                        return k(ap, b)[:M]
                else:
                    k = gemm_splitk(Mp, N, K, bM, bN, bK, th, 1, S)
                    c32 = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)

                    def run():
                        c32.zero_()
                        k(ap, b, c32)
                        return c32[:M].half()
                c = run()
                r["rel_err"] = round(((c.float() - ref).norm() / ref.norm()).item(), 6)
                ms = bench(run)
                r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), vs_hipblaslt=round(tms / ms, 3))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-160:]
            print(json.dumps(r), flush=True)


def select_tile(M, N, K):
    """Per-shape tile rule for the script as it stands (N4, jobs 23504069 and 23504582, randn, same process as hipBLASLt).
    Only what TileLang can emit today: tile, thread count, block order. Returns (bM, bN, bK, threads, swizzle, order) or None.
    None means this shape has no measured tile here and the caller should use hipBLASLt.
    Above 12288 K the 128-row 64-deep form loses to 128x256x32 at 512 threads; below it 128x128x64 with a column panel leads,
    and the panel stays at 8 when about 2200 blocks or fewer run, since 16 reads lower there."""
    if M % 128 or N % 128 or K % 64:
        return None
    if K >= 12288 and N % 256 == 0:
        return (128, 256, 32, 512, 16, "row")
    blocks = (M // 128) * (N // 128)
    return (128, 128, 64, 256, 8 if blocks <= 2200 else 16, "column")


def hand_source(M, N, K, bM=256, bN=128, sw="c8", rdiv=1, pad=32, fr=0.25, pad_b=None, b_base=0, guard=False, entry=None):
    """Hand-scheduled K loop for one shape: one inline asm block in the layout of Tensile's PGR2, v_mfma_f32_16x16x16f16, tile bM x bN x 64.
    Memory ops are spread between MFMAs, barriers do not stop math, accumulators sit in AGPRs. Moved from evidence/N4/src/gen_h7.py.
    pad, pad_b: LDS row padding of the A and B tiles in bytes (pad_b None: same as pad). b_base: extra bytes between the A and B tiles.
    guard: no staged load in the second-last K tile, no staged write, load or read in the last one (needs two more copies of the last window).
    entry None gives an always-inline helper hand_body(A, B, C, L) for T.Kernel(prelude=...), a name gives a standalone __global__ kernel.
    Returns (source, dynamic LDS bytes). Hazards are handled by the nops written here, not by LLVM. Needs K / 64 >= 3."""
    CH = 64
    rowa, rowb = CH * 2 + pad, CH * 2 + (pad if pad_b is None else pad_b)
    ta = bM * rowa + b_base
    dyn = ta + bN * rowb
    im, jn, n16 = bM // 32, bN // 32, CH // 16
    tpr, nva, nvb, nch = 2 * n16, n16 * bM // 128, n16 * bN // 128, K // CH
    vrow, lrowa, lrowb = (128 // n16) * K, (128 // n16) * rowa, (128 // n16) * rowb
    if nch < 3 or K % CH or M % bM or N % bN or dyn > 65536 or bM % 32 or bN % 32:
        raise ValueError(f"hand kernel does not take {M}x{N}x{K} tile {bM}x{bN} lds {dyn}")
    xa = 8
    xb = xa + im * 4
    ya = xb + jn * 4
    yb = ya + im * 4
    st = yb + jn * 4
    vend = st + (nva + nvb) * 4
    lwb = "%[lw]" if rowa == rowb else "%[lwb]"
    L = []
    e = L.append

    def vt(b, n=4):
        return f"v[{b}:{b + n - 1}]"

    def mf(ki):
        a, b = (ya, yb) if ki & 1 else (xa, xb)
        return [f"v_mfma_f32_16x16x16f16 %[c{i * jn + j}], {vt(b + j * 4 + kp * 2, 2)}, {vt(a + i * 4 + kp * 2, 2)}, %[c{i * jn + j}]"
                for kp in range(2) for i in range(im) for j in range(jn)]

    def rd(ki):
        a, b = (ya, yb) if ki & 1 else (xa, xb)
        return [f"ds_read_b128 {vt(a + i * 4)}, %[lra] offset:{i * 16 * rowa + ki * 64}" for i in range(im)] + \
               [f"ds_read_b128 {vt(b + j * 4)}, %[lrb] offset:{ta + j * 16 * rowb + ki * 64}" for j in range(jn)]

    S = [(f"ds_write_b128 %[lw], {vt(st + v * 4)} offset:{v * lrowa}",
          f"buffer_load_dwordx4 {vt(st + v * 4)}, %[voa], s[60:63], {'0' if v == 0 else f's{68 + v}'} offen") for v in range(nva)] + \
        [(f"ds_write_b128 {lwb}, {vt(st + (nva + v) * 4)} offset:{ta + v * lrowb}",
          f"buffer_load_dwordx4 {vt(st + (nva + v) * 4)}, %[vob], s[64:67], {'0' if v == 0 else f's{68 + v}'} offen") for v in range(nvb)]
    ns = len(S)

    def bump():
        e("s_cmp_gt_u32 s76, 2")
        e("s_cselect_b32 s77, %[stp], 0")
        e("s_add_u32 s60, s60, s77")
        e("s_addc_u32 s61, s61, 0")
        e("s_add_u32 s64, s64, s77")
        e("s_addc_u32 s65, s65, 0")

    def window(load=True, write=True):
        m1, r0 = mf(1), rd(0)
        w1 = len(m1)
        wn = int(w1 * (1 - fr))
        gapw, gapr, si, ri, barr = max(1, wn // ns), max(1, (w1 - wn) // len(r0)), 0, 0, False

        def put(i):
            e(f"s_waitcnt vmcnt({ns - 1 if load else ns - 1 - i})")
            e(S[i][0])
            if load:
                e(S[i][1])

        for idx, m in enumerate(m1):
            e(m)
            if not write:
                continue
            if si < ns and idx % gapw == 0:
                if si == 0 and load:
                    bump()
                put(si)
                si += 1
            if idx >= wn:
                if not barr:
                    while si < ns:
                        put(si)
                        si += 1
                    e("s_waitcnt lgkmcnt(0)")
                    e("s_barrier")
                    barr = True
                if (idx - wn) % gapr == 0 and ri < len(r0):
                    e(r0[ri])
                    ri += 1
        if write:
            while ri < len(r0):
                e(r0[ri])
                ri += 1

    e("s_nop 4")
    for i, x in enumerate(["%[a0]", "%[a1]", "%[a2]", "%[a3]", "%[b0]", "%[b1]", "%[b2]", "%[b3]"]):
        e(f"s_mov_b32 s{60 + i}, {x}")
    for v in range(1, 8):
        e(f"s_mov_b32 s{68 + v}, %[so{v}]")
    e("s_mov_b32 s76, %[cnt]")
    for _, l in S:
        e(l)
    e("s_waitcnt vmcnt(0)")
    for w, _ in S:
        e(w)
    bump()
    for _, l in S:
        e(l)
    e("s_waitcnt lgkmcnt(0)")
    e("s_barrier")
    for r in rd(0):
        e(r)
    e("LOOP%=:")
    e("s_waitcnt lgkmcnt(0)")
    m0, r1, ri = mf(0), rd(1), 0
    gap = max(1, (len(m0) // rdiv) // len(r1))
    for idx, m in enumerate(m0):
        e(m)
        if ri < len(r1) and idx % gap == 0:
            e(r1[ri])
            ri += 1
    while ri < len(r1):
        e(r1[ri])
        ri += 1
    e("s_waitcnt lgkmcnt(0)")
    e("s_barrier")
    if guard:
        e("s_cmp_gt_u32 s76, 2")
        e("s_cbranch_scc0 TAIL%=")
        window()
        e("s_sub_u32 s76, s76, 1")
        e("s_branch LOOP%=")
        e("TAIL%=:")
        e("s_cmp_eq_u32 s76, 1")
        e("s_cbranch_scc1 LAST%=")
        window(load=False)
        e("s_sub_u32 s76, s76, 1")
        e("s_branch LOOP%=")
        e("LAST%=:")
        window(load=False, write=False)
    else:
        window()
        e("s_sub_u32 s76, s76, 1")
        e("s_cmp_eq_u32 s76, 0")
        e("s_cbranch_scc0 LOOP%=")
    e("s_waitcnt vmcnt(0) lgkmcnt(0)")
    e("s_nop 7")
    e("s_nop 7")
    e("s_nop 7")
    asm = "\\n\\t".join(L)
    ncl = (bM * bN) // 1024
    outs = ", ".join(f'[c{i}] "+a"(c{i})' for i in range(ncl))
    ins = [f'[a{i}] "s"(ra[{i}])' for i in range(4)] + [f'[b{i}] "s"(rb[{i}])' for i in range(4)] + \
          [f'[so{v}] "s"((uint)({v * vrow * 2}))' for v in range(1, 8)] + \
          [f'[stp] "s"((uint)({CH * 2}))', f'[cnt] "s"((uint)({nch}))', '[voa] "v"(voa)', '[vob] "v"(vob)', '[lw] "v"(lw)', '[lra] "v"(lra)', '[lrb] "v"(lrb)']
    lwb_decl = ""
    if rowa != rowb:
        ins.append('[lwb] "v"(lwb)')
        lwb_decl = f"const uint lwb=(uint)((t/{tpr})*{rowb}+(t%{tpr})*16);"
    cl = [f'"v{i}"' for i in range(8, vend)] + [f'"s{i}"' for i in list(range(60, 68)) + list(range(69, 78))] + ['"scc"', '"memory"']
    swline = "const int by=(int)blockIdx.y,bx=(int)blockIdx.x;" if not sw else \
        "const dim3 d3_=tl::rasterization2D%s<%s>(); const int by=(int)d3_.y,bx=(int)d3_.x;" % ("Row" if sw[0] == "r" else "Column", sw[1:])
    wm, wn = ("(t&127)>>6", "t>>7") if bN == 128 else ("t>>7", "(t&127)>>6")
    if entry:
        head = f'extern "C" __global__ void __launch_bounds__(256) {entry}(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {{'
    else:
        head = "__device__ inline __attribute__((always_inline)) void hand_body(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, unsigned char* L_) {"
    clear = " ".join(f"float32x4 c{i}={{0.f,0.f,0.f,0.f}};" for i in range(ncl))
    src = f"""#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/threadblock_swizzle.h>
{head}
  const int t=(int)threadIdx.x,l=t&63; const int wm={wm}, wn={wn}; {swline}
  const int32x4_t ra=make_wave_buffer_resource(A), rb=make_wave_buffer_resource(B);
  const uint voa=(uint)(2*((by*{bM}+t/{tpr})*{K}+(t%{tpr})*8)), vob=(uint)(2*((bx*{bN}+t/{tpr})*{K}+(t%{tpr})*8));
  const uint lw=(uint)((t/{tpr})*{rowa}+(t%{tpr})*16), lra=(uint)((wm*{bM // 2}+(t&15))*{rowa}+((l>>4)&3)*16), lrb=(uint)((wn*{bN // 2}+(t&15))*{rowb}+((l>>4)&3)*16); {lwb_decl}
  {clear}
  asm volatile("{asm}" : {outs} : {", ".join(ins)} : {", ".join(cl)});
  float32x4 cc[{ncl}]={{{",".join(f"c{i}" for i in range(ncl))}}};
  #pragma unroll
  for (int i5=0;i5<{ncl};++i5) {{
    float32x4 v_=cc[i5]; half_t h[4]; uint2 u;
    *((half_t*)(&(((half2*)(&(u.x)))->x)))=(half_t)(v_[0]); *((half_t*)(&(((half2*)(&(u.x)))->y)))=(half_t)(v_[1]);
    *((half_t*)(&(((half2*)(&(u.y)))->x)))=(half_t)(v_[2]); *((half_t*)(&(((half2*)(&(u.y)))->y)))=(half_t)(v_[3]);
    *(uint2*)(C+(((by*{bM}+wm*{bM // 2}+(i5/{jn})*16+(t&15))*{N})+(bx*{bN}+wn*{bN // 2}+(i5%{jn})*16+((t&63)>>4)*4)))=u;
  }}
}}
"""
    return src, dyn


@tilelang.jit(out_idx=[-1])
def gemm_hand(M, N, K, bM=256, bN=128, sw="c8", rdiv=1, pad=32, guard=False):
    """The hand K loop launched through a normal T.Kernel. T.CUDASourceCodeKernel does not work on HIP (codegen drops the source, N4 round 2).
    The kernel body is prelude code; alloc_shared gives the dynamic LDS, which the asm addresses from offset 0."""
    dtype = T.float16
    src, dyn = hand_source(M, N, K, bM, bN, sw, rdiv, pad, guard=guard)

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=256, prelude=src):
            lds = T.alloc_shared((dyn,), T.uint8)
            T.call_extern("handle", "hand_body", T.address_of(A[0, 0]), T.address_of(B[0, 0]), T.address_of(C[0, 0]), T.address_of(lds[0]))

    return main


# Shapes where the 128x256 hand form replaces 256x128 when A and B are not placed (see place_b). With all GCDs busy and the package at its 500 W cap, 256x128 c8 loses here
# (ratio over hipBLASLt, median, three 8-GCD jobs, evidence/N4/RESULTS.md round 3): 16384^3 0.987-0.995 against 1.015-1.016, 6144^3 0.984-0.993 against 0.998-1.005,
# 12288^3 1.020-1.024 against 1.042-1.044, 4096x17408x5120 1.026-1.030 against 1.045-1.049. 32768x1024x2048 stays 256x128 (1.081-1.089 against 1.057-1.062).
HAND_128X256 = {(6144, 6144, 6144), (16384, 16384, 16384), (12288, 12288, 12288), (4096, 17408, 5120)}


# Round 5 (evidence/N4/RESULTS.md, ratios over hipBLASLt): the rule scores each tile by how evenly its blocks fill the 110 CUs, fill = B / (110 * ceil(B / 110)), times a per-tile factor.
# TF follows fill with the tile fixed: 128x256 at K 2048 reads 88, 101, 107, 112 TF at fill 0.73, 0.85, 0.91, 1.0. The exponent and factors are fitted to about 30 shapes, not derived. Why fill matters is a hypothesis.
HAND_CUS = 110
HAND_TILE_FACTOR = {(256, 192): 1.02, (192, 256): 1.02, (128, 128): 0.95, (64, 128): 0.88}
HAND_FILL_EXP = 0.7
# Below these hipBLASLt is used. 1536^3 reads 1.04 at 64x128; 1024^3 reads 1.00-1.01 at best.
HAND_MIN_MN = 1536 * 1536
HAND_MIN_SIDE = 1024
# Below this K the 256x128 tile beats 128x256 on a placed B too: 4096x4096x192 1.244-1.252 against 1.220-1.221, 2560x2560x256 1.267-1.276 against 1.235-1.242. At K 512 and 1024 they tie within 1%.
HAND_LOW_K = 512


def hand_fill(blocks):
    return blocks / (HAND_CUS * -(-blocks // HAND_CUS))


def hand_pad(bM, bN):
    """LDS row padding in bytes: 16 is the largest that fits 64 KiB on 256x192 and 192x256, 32 elsewhere."""
    return 16 if 192 in (bM, bN) else 32


def hand_tile(M, N, K, placed=False):
    """(bM, bN) of the hand kernel for this shape, or None if it does not take the shape (then select_tile or hipBLASLt).
    placed: the caller has A - B = 128 mod 256 bytes (place_b). Candidates are 256x192, 192x256, 128x128, 64x128 and one of 256x128 or 128x256; the 128x256 form needs placed or HAND_128X256 and K >= HAND_LOW_K.
    The highest factor * fill ** HAND_FILL_EXP wins, first listed on a tie. The score predicts our tile, not hipBLASLt: 1536x6400x2048 still reads 0.97-0.99.
    voa and vob are 32-bit byte offsets and the C index is an int, so 2*M*K, 2*N*K at 2^32 or M*N at 2^31 would wrap and return wrong results silently."""
    if K % 64 or K < 192 or M * N < HAND_MIN_MN or min(M, N) < HAND_MIN_SIDE or 2 * max(M, N) * K >= 2**32 or M * N >= 2**31:
        return None
    fits = lambda t: M % t[0] == 0 and N % t[1] == 0
    o128 = (placed or (M, N, K) in HAND_128X256) and K >= HAND_LOW_K
    big = [(128, 256), (256, 128)] if o128 else [(256, 128), (128, 256)]
    big = next((t for t in big if fits(t)), None)
    if big and not o128 and big == (128, 256) and not (placed or (M, N, K) in HAND_128X256):
        big = None  # as before: without placement 128x256 only at the listed shapes
    cand = [t for t in [(256, 192), (192, 256), big, (128, 128), (64, 128)] if t and fits(t)]
    if not cand:
        return None
    return max(cand, key=lambda t: HAND_TILE_FACTOR.get(t, 1.0) * hand_fill((M // t[0]) * (N // t[1])) ** HAND_FILL_EXP)


# Decode shapes where a hand tile beat the compiled forms in two jobs, ratio over hipBLASLt through torch: 64x5120x5120 0.95-0.99 against 0.81-0.85, 32x5120x5120 1.02-1.05 against 0.84-0.86, 16x17408x5120 0.85 against 0.79-0.80.
# Not listed, compiled reads higher or ties: 1x5120x5120, 64x5120x17408, 64x17408x5120. M below bM is padded with zero rows by gemm_small_m. Exact shapes only: no rule fits these reads.
SMALL_M_HAND = {(64, 5120, 5120): (32, 128), (32, 5120, 5120): (32, 64), (16, 17408, 5120): (32, 256)}


def gemm_small_m(M, N, K):
    """Callable (a, b) -> C for a shape in SMALL_M_HAND, else None. Hand-scheduled asm."""
    t = SMALL_M_HAND.get((M, N, K))
    if t is None:
        return None
    Mp = max(M, t[0])
    k = gemm_hand(Mp, N, K, *t, pad=hand_pad(*t))
    if Mp == M:
        return k

    def run(a, b):
        return k(torch.cat([a, torch.zeros(Mp - M, K, device=a.device, dtype=a.dtype)]), b)[:M]
    return run


def place_b(a, b):
    """B with its data address at 128 mod 256 bytes from A's, copying B only if it is not there already. hipBLASLt does not move with this offset; the 128x256 hand kernel gains
    (N4 round 3: 16384^3 1.016 to 1.047 at the cap, and its allocation-dependent low mode at 0.97 goes). A long-lived B needs this once. The result is a view into a slightly larger tensor."""
    if (a.data_ptr() - b.data_ptr()) & 255 == 128:
        return b
    size = b.element_size()
    spare = torch.empty(b.numel() + 256 // size, device=b.device, dtype=b.dtype)
    start = ((a.data_ptr() - spare.data_ptr() - 128) & 255) // size
    out = spare[start:start + b.numel()].view(b.shape)
    out.copy_(b)
    assert (a.data_ptr() - out.data_ptr()) & 255 == 128
    return out


def select_mode(M, N, K, placed=False):
    """'hand' where the hand kernel fits, 'tile' where only select_tile's compiled form fits, None for the caller to use hipBLASLt (also below HAND_MIN_MN or HAND_MIN_SIDE)."""
    if M * N < HAND_MIN_MN or min(M, N) < HAND_MIN_SIDE:
        return None
    if hand_tile(M, N, K, placed):
        return "hand"
    return None if select_tile(M, N, K) is None else "tile"


def gemm_auto(M, N, K, mode=None, placed=False):
    """Kernel for this shape or None. mode None follows select_mode; 'hand' or 'tile' forces one where it fits. placed: the caller used place_b.
    Ratios ours over hipBLASLt on randn are in evidence/N4/RESULTS.md."""
    mode = mode or select_mode(M, N, K, placed)
    if mode == "hand":
        t = hand_tile(M, N, K, placed)
        return gemm_hand(M, N, K, *t, pad=hand_pad(*t)) if t else None
    t = select_tile(M, N, K) if mode == "tile" else None
    if t is None:
        return None
    bM, bN, bK, th, sw, order = t
    return gemm(M, N, K, bM, bN, bK, th, 2, "base", sw, order)


SWEEP_TILES = [(128, 128, 64, 256, 0), (256, 128, 64, 256, 16), (128, 256, 64, 256, 16), (256, 128, 32, 512, 8), (256, 128, 32, 512, 16),
               (128, 128, 64, 256, 16), (128, 256, 32, 512, 8)]
SWEEP_SWIZZLE = [(0, "row"), (8, "row"), (8, "column"), (16, "column")]


def run_sweep():
    """N4: every tile, thread count and swizzle the script can express without hand C, one process per shape.
    argv: sweep M N K rounds iters warmup. hipBLASLt runs in the same process, arm order rotates per round."""
    M, N, K, rounds, iters, warm = map(int, sys.argv[2:8])
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float().T
    flops = 2 * M * N * K
    arms = [("hipblaslt", lambda: a @ b.T, None)]
    for bM, bN, bK, th, pad in SWEEP_TILES:
        if M % bM or N % bN or K % bK:
            continue
        for sw, order in SWEEP_SWIZZLE:
            k = gemm(M, N, K, bM, bN, bK, th, 2, "base", sw, order, pad)
            tag = f"{bM}x{bN}x{bK}t{th}p{pad}{order[0] if sw else ''}{sw or ''}"
            c = k(a, b)
            rel = ((c.float() - ref).norm() / ref.norm()).item()
            print(json.dumps({"compiled": tag, "rel_err": round(rel, 7)}), flush=True)
            arms.append((tag, (lambda k_=k: k_(a, b)), None))
    for r in range(rounds):
        for i in range(len(arms)):
            nm, fn, _ = arms[(i + r) % len(arms)]
            ms = bench(fn, iters, warm)
            print(json.dumps({"impl": nm, "round": r, "pos": i, "M": M, "N": N, "K": K, "data": "randn", "ms": round(ms, 4), "TF": round(flops / ms * 1e-9, 2)}), flush=True)


def run_auto():
    """argv: auto M N K rounds iters warmup [mode|- [placed]]. The chosen kernel against hipBLASLt through torch, same process, rotated order.
    rel_err is against torch fp32, for the kernel and for hipBLASLt. placed runs both sides on place_b's B."""
    M, N, K, rounds, iters, warm = map(int, sys.argv[2:8])
    mode = sys.argv[8] if len(sys.argv) > 8 and sys.argv[8] != "-" else None
    placed = len(sys.argv) > 9 and sys.argv[9] == "placed"
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float().T
    if placed:
        b = place_b(a, b)
    k = gemm_auto(M, N, K, mode, placed)
    if k is None:
        print(json.dumps({"M": M, "N": N, "K": K, "rule": None}), flush=True)
        return

    def check(c):
        return round(((c.float() - ref).norm() / ref.norm()).item(), 7)

    print(json.dumps({"M": M, "N": N, "K": K, "rule": mode or select_mode(M, N, K, placed), "hand_tile": hand_tile(M, N, K, placed), "placed": placed, "a_minus_b_mod256": (a.data_ptr() - b.data_ptr()) & 255, "rel_err": check(k(a, b)), "rel_err_hb": check(a @ b.T)}), flush=True)
    arms = [("hipblaslt", lambda: a @ b.T), ("rule", lambda: k(a, b))]
    for r in range(rounds):
        for i in range(2):
            nm, fn = arms[(i + r) % 2]
            ms = bench(fn, iters, warm)
            print(json.dumps({"impl": nm, "round": r, "pos": i, "M": M, "N": N, "K": K, "data": "randn", "ms": round(ms, 4), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)


def run_small_m():
    """argv: smallm M N K rounds iters warmup [placed]. gemm_small_m against hipBLASLt through torch, same process, rotated order. Placed uses place_b against the padded A's address only when no padding is needed."""
    M, N, K, rounds, iters, warm = map(int, sys.argv[2:8])
    placed = len(sys.argv) > 8 and sys.argv[8] == "placed"
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float().T
    if placed:
        b = place_b(a, b)
    k = gemm_small_m(M, N, K)

    def check(c):
        return round(((c.float() - ref).norm() / ref.norm()).item(), 7)

    print(json.dumps({"M": M, "N": N, "K": K, "tile": SMALL_M_HAND.get((M, N, K)), "placed": placed, "rel_err": check(k(a, b)), "rel_err_hb": check(a @ b.T)}), flush=True)
    arms = [("hipblaslt", lambda: a @ b.T), ("rule", lambda: k(a, b))]
    for r in range(rounds):
        for i in range(2):
            nm, fn = arms[(i + r) % 2]
            ms = bench(fn, iters, warm)
            print(json.dumps({"impl": nm, "round": r, "pos": i, "M": M, "N": N, "K": K, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)


def run_handsrc():
    """argv: handsrc entry M N K out.cu [key=value ...]. Writes a standalone kernel for the C++ harness and prints its dynamic LDS bytes."""
    entry, (M, N, K), out = sys.argv[2], map(int, sys.argv[3:6]), sys.argv[6]
    conv = {"bM": int, "bN": int, "sw": str, "rdiv": int, "pad": int, "pad_b": int, "b_base": int, "fr": float, "guard": lambda v: v == "1"}
    kw = {k: conv[k](v) for k, v in (o.split("=") for o in sys.argv[7:])}
    src, dyn = hand_source(M, N, K, entry=entry, **kw)
    open(out, "w").write(src)
    print(dyn)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "auto":
    run_auto()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "smallm":
    run_small_m()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "handsrc":
    run_handsrc()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "sweep":
    run_sweep()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "splitk":
    run_splitk()
    sys.exit(0)

if __name__ == "__main__":
    shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (4096, 17408, 5120), (32768, 1024, 2048), (64, 5120, 5120)]
    configs = [(128, 128, 32, 256, 2), (256, 128, 32, 256, 2), (128, 128, 64, 256, 2)]
    variants = ["base", "regpref", "dbuf"]
    if len(sys.argv) > 1:
        variants = sys.argv[1].split(",")
    torch.manual_seed(0)
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)
        ref = a.float() @ b.float().T
        tms = bench(lambda: a @ b.T)
        flops = 2 * M * N * K
        print(json.dumps({"impl": "hipblaslt", "M": M, "N": N, "K": K, "ms": round(tms, 4), "tflops": round(flops / tms * 1e-9, 1)}), flush=True)
        for bM, bN, bK, th, kp in configs:
            if M < bM:
                bM = 64 if M >= 64 else M
            for v in variants:
                r = {"impl": v, "M": M, "N": N, "K": K, "bM": bM, "bN": bN, "bK": bK, "threads": th, "k_pack": kp}
                try:
                    k = gemm(M, N, K, bM, bN, bK, th, kp, v)
                    c = k(a, b)
                    r["rel_err"] = round(((c.float() - ref).norm() / ref.norm()).item(), 6)
                    ms = bench(lambda: k(a, b))
                    r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 1), vs_hipblaslt=round(tms / ms, 3))
                except Exception as e:  # noqa: BLE001
                    r["err"] = (type(e).__name__ + ": " + str(e))[-160:]
                print(json.dumps(r), flush=True)
