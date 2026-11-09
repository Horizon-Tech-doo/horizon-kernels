# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K4 task bodies (phase 2b, 27-09-2026): our decode ops as T.macro bodies a persistent kernel can run as tasks.

Every body works on one task (a tile / a row / a slice) with the CTA's 128 threads (4 warps), takes its task indices as
arguments, makes no grid assumption and ends with a CTA barrier, so the next task can reuse the shared buffers.

Bit-identity with the kernel-per-op path (phase 1): the per-output arithmetic of our kernels depends only on
  * W4A16 GEMM (decode_regfrag / decode_regfrag_xl): the split-K factor (which k range each partial covers, alpha applied
    per partial, partials summed in order 0..split-1 in fp32, one bf16 rounding). Tile width (warps_n / nb), tile
    height (WM, padding rows), block_K and the pipeline depth do NOT change any output: each output element is
    accumulated by one thread in ascending 32-wide k chunks (two m16n8k16 MMAs each) whatever the tiling.
    decode_regfrag and its regfrag_xl "twin" are the same arithmetic. So a K4 GEMM task uses the phase-1 split_k of
    that (N, K, M) with a K4 tiling (128 threads, 128 columns).
  * add + RMSNorm (horizon_fused.add_rmsnorm, 640 threads per row): the order of the sum of squares. norm_task_body
    reproduces it with 128 threads: "virtual thread" v = 0..639 owns elements 8v..8v+7 exactly as thread v does in the
    640-thread kernel; real warp w handles virtual warps w, w+4, ..., each with the same butterfly warp reduction, and
    the 20 warp sums are added in warp order.
  * silu * mul (horizon_fused.silu_mul mode 1/2): elementwise, identical per element.
  * skinny BF16 GEMM (in_proj_ba): skinny_body itself (128 threads, as the K5 kernel).
"""

import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

THREADS = 128
BLOCK_K = 64
NB = 1                       # 32-column weight blocks per warp
WARPS_N = 4                  # 4 warps x 32 columns = 128 columns per GEMM task
BLOCK_N = WARPS_N * 32 * NB


def cache_src():
    """L2 loads for data published by other CTAs in the same launch."""
    return r"""
#ifndef HZ_K4_CG_LOADS
#define HZ_K4_CG_LOADS
__device__ __forceinline__ float hz_k4_f32_cg(const void* p) {
  float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void hz_k4_f32x4_cg(float* out, const void* p) {
  float a,b,c,d;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" :
               "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(p) : "memory");
  out[0]=a; out[1]=b; out[2]=c; out[3]=d;
}
__device__ __forceinline__ void hz_k4_f32x8_cg(float* out, const void* p) {
  unsigned long long a,b,c,d;
  asm volatile("ld.global.cg.v4.u64 {%0,%1,%2,%3}, [%4];" :
               "=l"(a), "=l"(b), "=l"(c), "=l"(d) : "l"(p) : "memory");
  out[0]=__uint_as_float(static_cast<unsigned>(a));
  out[1]=__uint_as_float(static_cast<unsigned>(a >> 32));
  out[2]=__uint_as_float(static_cast<unsigned>(b));
  out[3]=__uint_as_float(static_cast<unsigned>(b >> 32));
  out[4]=__uint_as_float(static_cast<unsigned>(c));
  out[5]=__uint_as_float(static_cast<unsigned>(c >> 32));
  out[6]=__uint_as_float(static_cast<unsigned>(d));
  out[7]=__uint_as_float(static_cast<unsigned>(d >> 32));
}
__device__ __forceinline__ unsigned hz_k4_u32_cg(const void* p) {
  unsigned bits; asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(bits) : "l"(p) : "memory"); return bits;
}
__device__ __forceinline__ float hz_k4_bf16_pair_cvt(unsigned bits, int high) {
  return __uint_as_float(high ? (bits & 0xffff0000u) : (bits << 16));
}
__device__ __forceinline__ float hz_k4_bf16_cg(const void* p) {
  unsigned short bits;
  asm volatile("ld.global.cg.b16 %0, [%1];" : "=h"(bits) : "l"(p) : "memory");
  return __uint_as_float(static_cast<unsigned>(bits) << 16);
}
#endif
"""


def gemm_src():
    """prelude the GEMM task body needs (regfrag's packed-word dequant)"""
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag as _rf
    return _rf._DQ_SRC + cache_src()


def gemm_alloc(WM: int):
    """(A_s, Wq_s, S_s, ticket_s) shared buffers for gemm_task_body (call inside the kernel; A_s gets the swizzle)."""
    nblk = BLOCK_N // 32
    kchunks = BLOCK_K // 32
    A_s = T.alloc_shared((WM, BLOCK_K), T.bfloat16)
    Wq_s = T.alloc_shared((nblk, kchunks, 128), T.int32)
    S_s = T.alloc_shared((nblk, kchunks, 16), T.int32)
    ticket_s = T.alloc_shared((1,), T.int32)
    T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})
    return A_s, Wq_s, S_s, ticket_s


def gemm_smem_shapes(WM: int, stages: int):
    """shared buffers of one GEMM pipeline stage set (all GEMM ops of a K4 kernel share one WM/stages)"""
    nblk = BLOCK_N // 32
    kchunks = BLOCK_K // 32
    return (WM, BLOCK_K), (nblk, kchunks, 128), (nblk, kchunks, 16)


def gemm_task_body(N: int, K: int, WM: int, split_k: int, num_stages: int):
    """One (bx = 128-column tile, bk = split) task of Y = x @ W^T for a regfrag-prepared NVFP4 weight (Wg, Sg, alpha).

    Arithmetic identical to decode_regfrag / regfrag_xl with the same split_k (module docstring). Rows: M <= WM (one m
    tile; padding rows wrap to row (i % M) and are never stored).
    split_k > 1: fp32 partials P[bk] (alpha applied), a ticket per tile at tickets[toff + bx] (self-resetting); the last task of
    a tile sums P[0..split_k-1] in order into Y. done_s[0] = 1 on the task that completed the tile (the only one that
    may signal the tile's consumers), else 0.
    Shared buffers come from the caller (gemm_alloc), so every GEMM op of a K4 kernel uses the same stage set."""
    nblk = BLOCK_N // 32
    kchunks = BLOCK_K // 32
    k_tiles = K // BLOCK_K
    assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and k_tiles % split_k == 0, (N, K, split_k)
    assert WM % 16 == 0
    kt_per_split = k_tiles // split_k
    mt = WM // 16
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=1, block_col_warps=WARPS_N,
                                      warp_row_tiles=WM, warp_col_tiles=32 * NB, chunk=BLOCK_K)

    def _partial_sum(P, i, col):
        acc = T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[0,i,col],"r"))
        for sk in range(1, split_k):
            acc = acc + T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[sk,i,col],"r"))
        return acc

    @T.macro
    def body(x, Y, Wg, Sg, alpha, P, tickets, toff, M, bx, bk, tid, done_s, A_s, Wq_s, S_s, ticket_s):
        q_l = T.alloc_local((NB * 4,), i32)
        s_l = T.alloc_local((NB * 2,), i32)
        B_l = T.alloc_local((NB * 32,), bf16)
        A_l = T.alloc_local((mt * 8,), bf16)
        C_l = T.alloc_local((mt * NB * 2 * 8,), f32)
        warp = tid // 32
        lane = tid % 32
        wn = warp
        T.clear(C_l)
        for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
            kc0 = (bk * kt_per_split + kt) * kchunks
            for i, kk in T.Parallel(WM, BLOCK_K):
                A_s[i, kk] = x[i % M, kc0 * 32 + kk]
            T.copy(Wg[bx * nblk, kc0, 0], Wq_s[0:nblk, 0:kchunks, 0:128])
            T.copy(Sg[bx * nblk, kc0, 0], S_s[0:nblk, 0:kchunks, 0:16])
            for c in T.unroll(kchunks):
                for b in T.unroll(NB):
                    for v in T.vectorized(4):
                        q_l[b * 4 + v] = Wq_s[wn * NB + b, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[b * 2 + v] = S_s[wn * NB + b, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                  T.access_ptr(s_l[b * 2], "r", extent=2),
                                  T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                    for i in T.unroll(mt):
                        for b in T.unroll(NB):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8 + 4, T.bool(False))
        for i in T.unroll(mt):
            for b in T.unroll(NB):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = bx * BLOCK_N + wn * 32 * NB + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M:
                            if split_k == 1:
                                for v in T.vectorized(2):
                                    Y[grow, gcol + v] = T.cast(
                                        C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                            else:
                                for v in T.vectorized(2):
                                    P[bk, grow, gcol + v] = C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]
        if split_k == 1:
            if tid == 0:
                done_s[0] = 1
        else:
            T.call_extern("handle", "__threadfence")
            T.sync_threads()
            if tid == 0:
                ticket_s[0] = T.atomic_add(tickets[toff + bx], 1, return_prev=True)
            T.sync_threads()
            if tid == 0:
                done_s[0] = 0
            if ticket_s[0] == split_k - 1:
                T.call_extern("handle", "__threadfence")
                for i, j in T.Parallel(WM, BLOCK_N):
                    if i < M:
                        Y[i, bx * BLOCK_N + j] = T.cast(_partial_sum(P, i, bx * BLOCK_N + j), bf16)
                if tid == 0:
                    tickets[toff + bx] = 0
                    done_s[0] = 1
        T.sync_threads()

    return body


def norm_geometry(H: int, vthreads: int | None = None):
    """Virtual thread count and vectors/thread matching standalone RMSNorm.

    Callers allocate at least vthreads // 32 FP32 shared reduction entries.
    All lanes of an active warp participate; inactive physical warps still
    participate in the CTA barriers. Widths are positive multiples of 256.
    """
    if H <= 0 or H % 256:
        raise ValueError("RMSNorm width must be a positive multiple of 256")
    if vthreads is None:
        vthreads = next(n for n in (640, 128, 64, 32) if H % (8 * n) == 0)
    if vthreads not in (32, 64, 128, 256, 384, 512, 640, 768, 896, 1024) or H % (8 * vthreads):
        raise ValueError("virtual threads must be 32, 64, or a multiple of 128 up to 1024, and divide H / 8")
    return vthreads, H // (8 * vthreads)


def norm_task_body(H: int, eps: float, has_res: bool, vthreads: int | None = None,
                   zero_centered_gamma: bool = True):
    """One BF16 row of residual add + RMSNorm on a 128-thread persistent CTA.

    Preserve the standalone virtual-thread accumulation and warp-sum order.
    Gemma weights use (1 + w); standard RMSNorm uses w. Default H=5120
    geometry and arithmetic are unchanged. red needs vthreads // 32 entries.
    """
    VEC = 8
    vthreads, per = norm_geometry(H, vthreads)
    rounds = (vthreads + THREADS - 1) // THREADS
    vwarps = vthreads // 32
    bf16, f32 = T.bfloat16, T.float32

    @T.macro
    def body(x, r, w, y, r_out, row, tid, red, tot):
        s = T.alloc_local((rounds * per * VEC,), f32)
        xb = T.alloc_local((VEC,), bf16)
        rb = T.alloc_local((VEC,), bf16)
        ob = T.alloc_local((VEC,), bf16)
        wb = T.alloc_local((VEC,), bf16)
        ss = T.alloc_var(f32)
        t = T.alloc_var(f32)
        rstd = T.alloc_var(f32)
        for q in T.unroll(rounds):
            if q * THREADS + tid < vthreads:
                ss = 0.0
                for i in T.unroll(per):
                    k0 = (i * vthreads + q * THREADS + tid) * VEC
                    off = (q * per + i) * VEC
                    for v in T.vectorized(VEC):
                        xb[v] = x[row, k0 + v]
                    if has_res:
                        for v in T.vectorized(VEC):
                            rb[v] = r[row, k0 + v]
                        for v in T.unroll(VEC):
                            s[off + v] = T.cast(xb[v], f32) + T.cast(rb[v], f32)
                            ob[v] = T.cast(s[off + v], bf16)
                        for v in T.vectorized(VEC):
                            r_out[row, k0 + v] = ob[v]
                    else:
                        for v in T.unroll(VEC):
                            s[off + v] = T.cast(xb[v], f32)
                    for v in T.unroll(VEC):
                        ss = ss + s[off + v] * s[off + v]
                ss = T.warp_reduce_sum(ss)
                if tid % 32 == 0:
                    red[q * 4 + tid // 32] = ss
        T.sync_threads()
        if tid == 0:
            t = red[0]
            for j in T.unroll(vwarps - 1):
                t = t + red[j + 1]
            tot[0] = T.rsqrt(t / T.float32(H) + T.float32(eps))
        T.sync_threads()
        rstd = tot[0]
        for q in T.unroll(rounds):
            if q * THREADS + tid < vthreads:
                for i in T.unroll(per):
                    k0 = (i * vthreads + q * THREADS + tid) * VEC
                    off = (q * per + i) * VEC
                    for v in T.vectorized(VEC):
                        wb[v] = w[k0 + v]
                    for v in T.unroll(VEC):
                        if zero_centered_gamma:
                            ob[v] = T.cast((s[off + v] * rstd) * (T.cast(wb[v], f32) + T.float32(1.0)), bf16)
                        else:
                            ob[v] = T.cast((s[off + v] * rstd) * T.cast(wb[v], f32), bf16)
                    for v in T.vectorized(VEC):
                        y[row, k0 + v] = ob[v]
        T.sync_threads()

    return body


def silu_task_body(I: int, cols: int, mode: int = 1):
    """a[row, c0:c0+cols] = bf16(silu(g) * u) for g = gu[row, c], u = gu[row, I + c] (horizon_fused.silu_mul mode 1/2
    arithmetic; needs silu_mul.SILU_SRC in the prelude). cols % 8 == 0."""
    VEC = 8
    assert cols % VEC == 0
    iters = (cols // VEC + THREADS - 1) // THREADS
    fn = {1: "hz_silu_mul_ind", 2: "hz_silu_mul_ind2"}[mode]
    bf16, f32 = T.bfloat16, T.float32

    @T.macro
    def body(gu, a, row, c0, tid):
        xb = T.alloc_local((VEC,), bf16)
        ub = T.alloc_local((VEC,), bf16)
        ob = T.alloc_local((VEC,), bf16)
        for it in T.serial(iters):
            e = (it * THREADS + tid) * VEC
            if e < cols:
                for v in T.vectorized(VEC):
                    xb[v] = T.cast(T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,c0+e+v],"r")),bf16)
                for v in T.vectorized(VEC):
                    ub[v] = T.cast(T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,I+c0+e+v],"r")),bf16)
                for v in T.unroll(VEC):
                    ob[v] = T.cast(T.call_extern(f32, fn, T.cast(xb[v], f32), T.cast(ub[v], f32)), bf16)
                for v in T.vectorized(VEC):
                    a[row, c0 + e + v] = ob[v]
        T.sync_threads()

    return body


# ---------------------------------------------------------------------------------------------------------------
# GEMM task with a hand-written cp.async pipeline (phase 2b): the weight half of the first stages is issued BEFORE the
# task's dependency wait (weights never depend on anything), the activation half after it. While a CTA waits for its
# inputs, its first S-1 weight stages are already streaming into shared memory. Same arithmetic as gemm_task_body.
def _hw_loader_src(tag: str, WM: int, K: int, block_k: int = BLOCK_K, block_n: int = BLOCK_N, threads: int = None):
    """C++ stage loaders for one (WM, K): k4_ldw_<tag> (packed weights + scales) and k4_lda_<tag> (activation rows,
    TileLang's 128-byte XOR swizzle, rows >= M zero-filled)."""
    BLOCK_N, THREADS = block_n, (block_n if threads is None else threads)
    assert THREADS in (128,256) and THREADS >= BLOCK_N
    nblk, kch, rowch = BLOCK_N // 32, block_k // 32, block_k // 8
    a_n, w_n, s_n = WM * rowch, nblk * kch * 32, nblk * kch * 4
    assert w_n % THREADS == 0 and s_n <= THREADS
    a_it, a_g = -(-a_n // THREADS), ("" if a_n % THREADS == 0 else f"if (e < {a_n}u) ")
    w_it = w_n // THREADS
    return f"""
__device__ __forceinline__ void k4_ldw_{tag}(unsigned sW, unsigned sS, int st, const void *wp, const void *sp, int bx,
                                            int kc0, int tid) {{
  const char *w = static_cast<const char *>(wp), *sg = static_cast<const char *>(sp);
  const unsigned ust = st, ukc = kc0, ut = tid, ubx = bx;
#pragma unroll
  for (unsigned v = 0; v < {w_it}u; ++v) {{
    const unsigned e = ut + v * {THREADS}u, r = e / {kch * 32}u, c = (e >> 5) % {kch}u, l = e & 31u;
    xl_cp16s(sW + (((ust * {nblk}u + r) * {kch}u + c) * 512u + l * 16u),
             w + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 512u + l * 16u));
  }}
  if (ut < {s_n}u) {{
    const unsigned r = ut / {kch * 4}u, c = (ut >> 2) % {kch}u, l = ut & 3u;
    xl_cp16s(sS + (((ust * {nblk}u + r) * {kch}u + c) * 64u + l * 16u),
             sg + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 64u + l * 16u));
  }}
}}
__device__ __forceinline__ void k4_lda_{tag}(unsigned sA, int st, const void *xp, int M, int kc0, int tid) {{
  const char *x = static_cast<const char *>(xp);
  const unsigned ust = st, ukc = kc0, ut = tid;
#pragma unroll
  for (unsigned v = 0; v < {a_it}u; ++v) {{
    const unsigned e = ut + v * {THREADS}u, row = e / {rowch}u, ch = e % {rowch}u;
    const bool ok = (int)row < M;
    const unsigned grow = ok ? row : 0u;
    {a_g}xl_cp16z(sA + (((ust * {block_k // 64}u + ch / 8u) * {WM}u + row) * 128u + (((ch & 7u) ^ (row & 7u)) << 4)), x + (grow * {K * 2}u + ukc * 64u + ch * 16u),
                 ok ? 16u : 0u);
  }}
}}
"""


def hw_common_src():
    """cp.async helpers shared by every hand-written GEMM task (regfrag's dequant + xl's copy/commit/wait)."""
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as _xl
    return _xl._XL_SRC + cache_src() + r"""
#ifndef HZ_K4_HW_DEFINED
#define HZ_K4_HW_DEFINED
__device__ __forceinline__ unsigned xl_saddr(const void *p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void xl_cp16s(unsigned s, const void *g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void xl_cp16z(unsigned s, const void *g, unsigned n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(g), "r"(n));
}
#endif
"""


def gemm_hw_alloc(WM: int, S: int, block_k: int = BLOCK_K, block_n: int = BLOCK_N):
    """staged shared buffers (A_s, Wq_s, S_s, ticket_s) + their shared addresses sb (local uint32[3]) for the
    hand-written GEMM tasks; call inside the kernel."""
    nblk, kch = block_n // 32, block_k // 32
    A_s = T.alloc_shared((S, WM, block_k), T.bfloat16)
    Wq_s = T.alloc_shared((S, nblk, kch, 128), T.int32)
    S_s = T.alloc_shared((S, nblk, kch, 16), T.int32)
    ticket_s = T.alloc_shared((1,), T.int32)
    sb = T.alloc_local((3,), T.uint32)
    T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})
    return A_s, Wq_s, S_s, ticket_s, sb


@T.macro
def gemm_hw_addr(A_s, Wq_s, S_s, sb):
    sb[0] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(A_s, "w"))
    sb[1] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(Wq_s, "w"))
    sb[2] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(S_s, "w"))


def gemm_task_hw(tag: str, N: int, K: int, WM: int, split_k: int, S: int, block_k: int = BLOCK_K, block_n: int = BLOCK_N, threads: int = None, flat_shared: bool = False, row_warps: int = 1, late_issue: bool = False, staged_init: bool = False, partial_vector4: bool = False, partial_vector8: bool = False):
    """(src, pre, body) for one GEMM op: `src` goes into the kernel prelude (after hw_common_src()); `pre(Wg, Sg, bx,
    bk, sb, tid)` issues the weight half of stages 0..S-2 (call before the dependency wait); `body(...)` issues the
    activation half, runs the pipeline and the (split-K) epilogue. Arithmetic identical to gemm_task_body."""
    if type(partial_vector8) is not bool or partial_vector4 and partial_vector8:
        raise ValueError("packed reduction modes must be boolean and mutually exclusive")
    if type(partial_vector4) is not bool:
        raise ValueError("packed reduction flag must be boolean")
    if staged_init and S not in (3,4):
        raise ValueError("staged initial copies are validated only for stages3/4")
    if block_n not in (128,256):
        raise ValueError("unsupported GEMM N width")
    BLOCK_N, WARPS_N = block_n, block_n // 32
    if row_warps not in (1,2) or WM % (16 * row_warps):
        raise ValueError("unsupported row-warp geometry")
    ROW_WM = WM // row_warps
    COMPUTE_THREADS = BLOCK_N * row_warps
    THREADS = COMPUTE_THREADS if threads is None else threads
    if THREADS not in (128,256) or THREADS < COMPUTE_THREADS:
        raise ValueError("row/column warp geometry exceeds launch threads")
    nblk, kch = BLOCK_N // 32, block_k // 32
    k_tiles = K // block_k
    assert N % BLOCK_N == 0 and K % block_k == 0 and k_tiles % split_k == 0, (N, K, split_k)
    kps = k_tiles // split_k
    assert kps >= S - 1 and S >= 3, (kps, S)
    mt = ROW_WM // 16
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=row_warps, block_col_warps=WARPS_N,
                                      warp_row_tiles=ROW_WM, warp_col_tiles=32 * NB, chunk=block_k)
    src = _hw_loader_src(tag, WM, K, block_k, block_n, THREADS)
    ldw, lda = f"k4_ldw_{tag}", f"k4_lda_{tag}"

    def _partial_sum(P, i, col):
        acc = T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[0,i,col],"r"))
        for sk in range(1, split_k):
            acc = acc + T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[sk,i,col],"r"))
        return acc

    @T.macro
    def pre(Wg, Sg, bx, bk, sb, tid):
        for p in T.unroll(S - 1):
            T.call_extern("handle", ldw, sb[1], sb[2], p, T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), bx,
                          (bk * kps + p) * kch, tid)
            T.call_extern("handle", "xl_commit")

    @T.macro
    def issue(kt, x, Wg, Sg, sb, stg, M, bx, bk, tid):
        if kt + S - 1 < kps:
            T.call_extern("handle", lda, sb[0], stg[1], T.access_ptr(x, "r"), M,
                          (bk * kps + kt + S - 1) * kch, tid)
            T.call_extern("handle", ldw, sb[1], sb[2], stg[1], T.access_ptr(Wg, "r"),
                          T.access_ptr(Sg, "r"), bx, (bk * kps + kt + S - 1) * kch, tid)
        T.call_extern("handle", "xl_commit")

    @T.macro
    def body(x, Y, Wg, Sg, alpha, P, tickets, toff, M, bx, bk, tid, done_s, A_s, Wq_s, S_s, ticket_s, sb):
        q_l = T.alloc_local((NB * 4,), i32)
        s_l = T.alloc_local((NB * 2,), i32)
        B_l = T.alloc_local((NB * 32,), bf16)
        A_l = T.alloc_local((mt * 8,), bf16)
        C_l = T.alloc_local((mt * NB * 2 * 8,), f32)
        stg = T.alloc_local((2,), i32)
        warp = tid // 32
        lane = tid % 32
        wn = warp // row_warps
        warp_m = warp % row_warps
        T.clear(C_l)
        for p in T.unroll(S - 1):
            T.call_extern("handle", lda, sb[0], p, T.access_ptr(x, "r"), M, (bk * kps + p) * kch, tid)
            if staged_init:
                T.call_extern("handle", "xl_commit")
        if not staged_init:
            T.call_extern("handle", "xl_commit")
        stg[0] = 0
        stg[1] = S - 1
        for kt in T.serial(kps):
            if kt == 0 and not staged_init:
                T.call_extern("handle", "xl_wait<0>")
            else:
                T.call_extern("handle", f"xl_wait<{S - 2}>")
            T.sync_threads()
            if not late_issue:
                issue(kt, x, Wg, Sg, sb, stg, M, bx, bk, tid)
            st = stg[0]
            for c in T.unroll(kch):
                if THREADS == COMPUTE_THREADS or tid < COMPUTE_THREADS:
                    for b in T.unroll(NB):
                        if flat_shared:
                            for v in T.vectorized(4):
                                q_l[b * 4 + v] = Wq_s[st * nblk + wn * NB + b, c, lane * 4 + v]
                            for v in T.vectorized(2):
                                s_l[b * 2 + v] = S_s[st * nblk + wn * NB + b, c, (lane // 4) * 2 + v]
                        else:
                            for v in T.vectorized(4):
                                q_l[b * 4 + v] = Wq_s[st, wn * NB + b, c, lane * 4 + v]
                            for v in T.vectorized(2):
                                s_l[b * 2 + v] = S_s[st, wn * NB + b, c, (lane // 4) * 2 + v]
                        T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                      T.access_ptr(s_l[b * 2], "r", extent=2), T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    if THREADS == COMPUTE_THREADS or tid < COMPUTE_THREADS:
                        emitter.ldmatrix_a(A_l, A_s[st, 0:WM, 0:block_k], c * 2 + s)
                        for i in T.unroll(mt):
                            for b in T.unroll(NB):
                                for j in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                              i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                              ((i * NB + b) * 2 + j) * 8, T.bool(False))
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                              i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                              ((i * NB + b) * 2 + j) * 8 + 4, T.bool(False))
                    if late_issue and c == 0 and s == 0:
                        issue(kt, x, Wg, Sg, sb, stg, M, bx, bk, tid)
            stg[0] = T.if_then_else(stg[0] == S - 1, 0, stg[0] + 1)
            stg[1] = T.if_then_else(stg[1] == S - 1, 0, stg[1] + 1)
        T.call_extern("handle", "xl_wait<0>")
        T.sync_threads()
        if THREADS == COMPUTE_THREADS or tid < COMPUTE_THREADS:
            for i in T.unroll(mt):
                for b in T.unroll(NB):
                    for j in T.unroll(2):
                        for r2 in T.unroll(4):
                            grow = warp_m * ROW_WM + i * 16 + 8 * (r2 % 2) + lane // 4
                            gcol = bx * BLOCK_N + wn * 32 * NB + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                            if grow < M:
                                if split_k == 1:
                                    for v in T.vectorized(2):
                                        Y[grow, gcol + v] = T.cast(
                                            C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                else:
                                    for v in T.vectorized(2):
                                        P[bk, grow, gcol + v] = C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]
        if split_k == 1:
            if tid == 0:
                done_s[0] = 1
        else:
            T.call_extern("handle", "__threadfence")
            T.sync_threads()
            if tid == 0:
                ticket_s[0] = T.atomic_add(tickets[toff + bx], 1, return_prev=True)
            T.sync_threads()
            if tid == 0:
                done_s[0] = 0
            if ticket_s[0] == split_k - 1:
                T.call_extern("handle", "__threadfence")
                if partial_vector8:
                    # Every row and tile starts at a multiple of eight FP32 values.
                    # Each thread owns eight adjacent columns; additions remain sk0..skN.
                    acc8 = T.alloc_local((8,), f32)
                    piece8 = T.alloc_local((8,), f32)
                    for batch in T.unroll(WM * (BLOCK_N // 8) // THREADS):
                        octet = tid + batch * THREADS
                        row = octet // (BLOCK_N // 8)
                        col = bx * BLOCK_N + (octet % (BLOCK_N // 8)) * 8
                        if row < M:
                            T.call_extern("handle", "hz_k4_f32x8_cg", T.access_ptr(acc8, "w", extent=8),
                                          T.access_ptr(P[0,row,col], "r", extent=8))
                            for sk in T.unroll(1, split_k):
                                T.call_extern("handle", "hz_k4_f32x8_cg", T.access_ptr(piece8, "w", extent=8),
                                              T.access_ptr(P[sk,row,col], "r", extent=8))
                                for v in T.unroll(8):
                                    acc8[v] = acc8[v] + piece8[v]
                            for v in T.vectorized(8):
                                Y[row,col+v] = T.cast(acc8[v], bf16)
                elif partial_vector4:
                    # Every row and tile starts at a multiple of four FP32 values.
                    # Each thread owns four adjacent columns; additions remain sk0..skN.
                    acc4 = T.alloc_local((4,), f32)
                    piece4 = T.alloc_local((4,), f32)
                    for batch in T.unroll(WM * (BLOCK_N // 4) // THREADS):
                        quad = tid + batch * THREADS
                        row = quad // (BLOCK_N // 4)
                        col = bx * BLOCK_N + (quad % (BLOCK_N // 4)) * 4
                        if row < M:
                            T.call_extern("handle", "hz_k4_f32x4_cg", T.access_ptr(acc4, "w", extent=4),
                                          T.access_ptr(P[0,row,col], "r", extent=4))
                            for sk in T.unroll(1, split_k):
                                T.call_extern("handle", "hz_k4_f32x4_cg", T.access_ptr(piece4, "w", extent=4),
                                              T.access_ptr(P[sk,row,col], "r", extent=4))
                                for v in T.unroll(4):
                                    acc4[v] = acc4[v] + piece4[v]
                            for v in T.vectorized(4):
                                Y[row,col+v] = T.cast(acc4[v], bf16)
                else:
                    for i, j in T.Parallel(WM, BLOCK_N):
                        if i < M:
                            Y[i, bx * BLOCK_N + j] = T.cast(_partial_sum(P, i, bx * BLOCK_N + j), bf16)
                if tid == 0:
                    tickets[toff + bx] = 0
                    done_s[0] = 1
        T.sync_threads()

    return src, pre, body


# ---------------------------------------------------------------------------------------------------------------
# Pieces of the hand-written GEMM task for kernels that drive the k loop themselves (phase 2b "run ahead"): the loop can
# then use the stage slots freed in a task's last S-1 iterations to issue the NEXT GEMM task's weight stages, so a CTA
# streams weights without a gap across task boundaries (and while the next task waits for its inputs).
# Stage slots rotate: a task whose stage 0 sits in slot st0 finds stage p in slot (st0 + p) % S; the next task's
# st0 is (st0 + kps) % S.
class GemmHW:
    def __init__(self, tag: str, N: int, K: int, WM: int, split_k: int, S: int):
        nblk, kch = BLOCK_N // 32, BLOCK_K // 32
        k_tiles = K // BLOCK_K
        assert N % BLOCK_N == 0 and K % BLOCK_K == 0 and k_tiles % split_k == 0, (N, K, split_k)
        self.kps = kps = k_tiles // split_k
        assert kps >= S and S >= 3, (kps, S)
        self.S, self.K, self.split_k = S, K, split_k
        mt = WM // 16
        bf16, f32, i32 = T.bfloat16, T.float32, T.int32
        emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                          b_transposed=True, block_row_warps=1, block_col_warps=WARPS_N,
                                          warp_row_tiles=WM, warp_col_tiles=32 * NB, chunk=BLOCK_K)
        self.src = _hw_loader_src(tag, WM, K)
        ldw, lda = f"k4_ldw_{tag}", f"k4_lda_{tag}"

        def _partial_sum(P, i, col):
            acc = T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[0,i,col],"r"))
            for sk in range(1, split_k):
                acc = acc + T.call_extern(T.float32,"hz_k4_f32_cg",T.access_ptr(P[sk,i,col],"r"))
            return acc

        @T.macro
        def issue_w(Wg, Sg, bx, bk, p, slot, sb, tid):
            """weight half of this op's stage p (k tile bk * kps + p) into `slot` (no commit)"""
            T.call_extern("handle", ldw, sb[1], sb[2], slot, T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), bx,
                          (bk * kps + p) * kch, tid)

        @T.macro
        def pre(Wg, Sg, bx, bk, st0, sb, tid):
            """weight halves of stages 0..S-2, one commit group each"""
            for p in T.unroll(S - 1):
                issue_w(Wg, Sg, bx, bk, p, (st0 + p) % S, sb, tid)
                T.call_extern("handle", "xl_commit")

        @T.macro
        def prologue(x, M, bk, st0, sb, tid, C_l):
            T.clear(C_l)
            for p in T.unroll(S - 1):
                T.call_extern("handle", lda, sb[0], (st0 + p) % S, T.access_ptr(x, "r"), M, (bk * kps + p) * kch, tid)
            T.call_extern("handle", "xl_commit")

        @T.macro
        def issue_own(x, Wg, Sg, M, bx, bk, kt, slot, sb, tid):
            """stage kt + S - 1 of this task (both halves) into `slot`; call only if kt + S - 1 < kps"""
            T.call_extern("handle", lda, sb[0], slot, T.access_ptr(x, "r"), M, (bk * kps + kt + S - 1) * kch, tid)
            T.call_extern("handle", ldw, sb[1], sb[2], slot, T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), bx,
                          (bk * kps + kt + S - 1) * kch, tid)

        @T.macro
        def compute(slot, A_s, Wq_s, S_s, tid, q_l, s_l, B_l, A_l, C_l):
            warp = tid // 32
            lane = tid % 32
            wn = warp
            for c in T.unroll(kch):
                for b in T.unroll(NB):
                    for v in T.vectorized(4):
                        q_l[b * 4 + v] = Wq_s[slot, wn * NB + b, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[b * 2 + v] = S_s[slot, wn * NB + b, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                  T.access_ptr(s_l[b * 2], "r", extent=2), T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    emitter.ldmatrix_a(A_l, A_s[slot, 0:WM, 0:BLOCK_K], c * 2 + s)
                    for i in T.unroll(mt):
                        for b in T.unroll(NB):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                          ((i * NB + b) * 2 + j) * 8 + 4, T.bool(False))

        @T.macro
        def epilogue(Y, alpha, P, tickets, toff, M, bx, bk, tid, done_s, ticket_s, C_l):
            lane = tid % 32
            wn = tid // 32
            for i in T.unroll(mt):
                for b in T.unroll(NB):
                    for j in T.unroll(2):
                        for r2 in T.unroll(4):
                            grow = i * 16 + 8 * (r2 % 2) + lane // 4
                            gcol = bx * BLOCK_N + wn * 32 * NB + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                            if grow < M:
                                if split_k == 1:
                                    for v in T.vectorized(2):
                                        Y[grow, gcol + v] = T.cast(
                                            C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                else:
                                    for v in T.vectorized(2):
                                        P[bk, grow, gcol + v] = (C_l[((i * NB + b) * 2 + j) * 8 + r2 * 2 + v]
                                                                 * alpha[gcol + v])
            if split_k == 1:
                if tid == 0:
                    done_s[0] = 1
            else:
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(tickets[toff + bx], 1, return_prev=True)
                T.sync_threads()
                if tid == 0:
                    done_s[0] = 0
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    for i, j in T.Parallel(WM, BLOCK_N):
                        if i < M:
                            Y[i, bx * BLOCK_N + j] = T.cast(_partial_sum(P, i, bx * BLOCK_N + j), bf16)
                    if tid == 0:
                        tickets[toff + bx] = 0
                        done_s[0] = 1
            T.sync_threads()

        self.issue_w, self.pre, self.prologue, self.issue_own = issue_w, pre, prologue, issue_own
        self.compute, self.epilogue = compute, epilogue


def gemm_regs(WM: int):
    """local register buffers of the GemmHW pieces: (q_l, s_l, B_l, A_l, C_l); call inside the kernel"""
    mt = WM // 16
    return (T.alloc_local((NB * 4,), T.int32), T.alloc_local((NB * 2,), T.int32),
            T.alloc_local((NB * 32,), T.bfloat16), T.alloc_local((mt * 8,), T.bfloat16),
            T.alloc_local((mt * NB * 2 * 8,), T.float32))
