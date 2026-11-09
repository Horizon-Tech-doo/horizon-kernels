# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 "regfrag_xl": the regfrag W4A16 NVFP4 GEMM for every batch size 1 <= M <= 1024 (drop-in for Marlin).

Contract: decode_contract.py (prepare / linear / SUPPORTS_M = 1024). prepare() IS decode_regfrag.prepare (plus two tiny
per-device scratch buffers), so ONE prepared weight copy serves decode_regfrag and this module: no extra weight memory.

config(N, K, M) picks, per layer shape and M bucket (tuned table _TUNED; heuristic for other shapes), one of
  1. ("rf",): decode_regfrag (unchanged file) - for M <= 64 directly, above 64 as 64-row passes (Marlin's own
     structure for M > 64). decode_regfrag maps padding rows (M .. block_M-1) to row M-1, so every CTA re-reads the
     same x row up to 15 times per k tile: an L2 hot spot that cost up to 20% (gate_up M=33 / 49). Above 16 rows a
     padded row count therefore runs decode_regfrag's tile-kernel twin instead (same block_M / warps / split / stages).
  2. the tile kernel (nvfp4_regfrag_xl_kernel): grid (m tiles, n tiles, split_k), TileLang T.Pipelined cp.async
     pipeline, deterministic split-K (fp32 partials + ticket, fixed-order sum); padding rows wrap to distinct rows.
  3. the range kernel (nvfp4_regfrag_xl_sk_kernel): G CTAs with G a run-time value; the (tile, k-tile) iteration space
     is cut into G contiguous ranges, so one binary does plain tiling (G = tiles), split-K (G = tiles * s) or stream-K
     (G = 188 * c: no wave quantisation; split tiles are summed by their last CTA in fixed CTA order: deterministic).
     Its k loop is a hand-written cp.async pipeline (C++ stage loader writing raw shared addresses, padding rows
     zero-filled; one barrier per k tile). warps_k > 1 adds Marlin-style intra-CTA split-K (measured: never better).
  Both XL kernels: the CTA covers block_M = warps_m * WM rows x block_N = warps_n * 32 * nb columns; warp (wm, wn) owns
  WM rows x 32*nb columns. Per 32-wide k chunk each warp dequantises its nb 32x32 weight blocks ONCE into register B
  fragments (decode_regfrag's packed-word dequant, ~2.1 instructions per weight) and reuses them for its WM/16 m16
  tiles. Tile heights 16..128 (16-row steps) let the M tiling fit the batch instead of padding to 64-row multiples.
  A (activations) is streamed with cp.async into a swizzled shared tile and read with ldmatrix. Grids / ranges put the
  m tiles of one weight column next to each other, so a weight tile is read from DRAM once.
* Deterministic (no fp32 atomics on data), CUDA-graph capturable (workspaces via torch.empty in linear(); ticket
  counters reset themselves). Split launches share a per-device ticket buffer: do not run XL launches of one device
  concurrently on two streams (same rule as decode_regfrag's per-layer counters).
"""

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

from tilelang.contrib.sm120_nvfp4 import decode_regfrag as _rf

SUPPORTS_M = 1024
_CNT = 1 << 14          # split-K ticket counters per device (m tiles x n tiles of one split launch)

# hand-written cp.async pipeline primitives (stream-K kernel): 16-byte L2-only copy, commit, wait (memory clobbers
# keep the compiler from moving shared-memory reads across the wait).
_XL_SRC = _rf._DQ_SRC + r"""
__device__ __forceinline__ void xl_cp16(void *smem, const void *gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void xl_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void xl_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory"); }
"""


# ---------------------------------------------------------------------------------------------------------------
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_regfrag_xl_kernel(N: int, K: int, warps_m: int, WM: int, warps_n: int, nb: int, block_K: int,
                            split_k: int, num_stages: int, min_blocks: int = 1):
    assert WM % 16 == 0 and nb in (1, 2) and block_K % 32 == 0
    block_M = warps_m * WM
    WN = 32 * nb
    block_N = warps_n * WN
    nblk = block_N // 32                 # 32-column weight blocks per CTA
    threads = 32 * warps_m * warps_n
    kchunks = block_K // 32
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    mt = WM // 16                        # m16 tiles per warp
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=warps_m, block_col_warps=warps_n,
                                      warp_row_tiles=WM, warp_col_tiles=WN, chunk=block_K)
    P_shape = (split_k, M, N) if split_k > 1 else (1, 1, 1)

    def _partial_sum(P, i, col):
        acc = P[0, i, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, i, col]
        return acc

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Y: T.Tensor((M, N), bf16),
        Wg: T.Tensor((N // 32, K // 32, 128), i32),
        Sg: T.Tensor((N // 32, K // 32, 16), i32),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor(P_shape, f32),
        counters: T.Tensor((_CNT,), i32),
    ):
        with T.Kernel(T.ceildiv(M, block_M), n_tiles, split_k, threads=threads, prelude=_rf._DQ_SRC) as (bm, bx, bk):
            T.assume(M >= 1)  # v0.1.15 (#3294): M >= 1 is the launch contract; keeps the unguarded loads
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            A_s = T.alloc_shared((block_M, block_K), bf16)
            Wq_s = T.alloc_shared((nblk, kchunks, 128), i32)
            S_s = T.alloc_shared((nblk, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            q_l = T.alloc_local((nb * 4,), i32)
            s_l = T.alloc_local((nb * 2,), i32)
            B_l = T.alloc_local((nb * 32,), bf16)              # per block: 2 k16 steps x 2 n16 tiles x 8 values
            A_l = T.alloc_local((mt * 8,), bf16)               # one k16 step of A for the warp's m16 tiles
            C_l = T.alloc_local((mt * nb * 2 * 8,), f32)       # mt m16 x nb blocks x 2 n16 accumulators
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            wm = warp % warps_m                                # matches emitter.extract_thread_binding (m first)
            wn = warp // warps_m
            m0 = bm * block_M
            T.clear(C_l)

            for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
                kc0 = (bk * kt_per_split + kt) * kchunks
                # padding rows >= M wrap to row (m0 + i) % M (always a valid row; never stored). Clamping them all to
                # row M-1 made every CTA hammer the same L2 lines (M = 33 on gate_up: 15 of 48 rows) and cost up to
                # 25%; a predicated zero-fill load is not pipelined as cp.async by T.Pipelined.
                for i, kk in T.Parallel(block_M, block_K):
                    A_s[i, kk] = x[(m0 + i) % M, kc0 * 32 + kk]
                T.copy(Wg[bx * nblk, kc0, 0], Wq_s)
                T.copy(Sg[bx * nblk, kc0, 0], S_s)
                for c in T.unroll(kchunks):
                    for b in T.unroll(nb):
                        for v in T.vectorized(4):
                            q_l[b * 4 + v] = Wq_s[wn * nb + b, c, lane * 4 + v]
                        for v in T.vectorized(2):
                            s_l[b * 2 + v] = S_s[wn * nb + b, c, (lane // 4) * 2 + v]
                        T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                      T.access_ptr(s_l[b * 2], "r", extent=2),
                                      T.access_ptr(B_l[b * 32], "w", extent=32))
                    for s in T.unroll(2):
                        emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                        for i in T.unroll(mt):
                            for b in T.unroll(nb):
                                for j in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                              i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                              ((i * nb + b) * 2 + j) * 8, T.bool(False))
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                              i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                              ((i * nb + b) * 2 + j) * 8 + 4, T.bool(False))

            # C_l[((i*nb+b)*2+j)*8 + r] sits at row 16i + 8*((r%4)//2) + lane//4,
            #                                   col 32b + 16j + 8*(r//4) + 2*(lane%4) + r%2   (warp-relative)
            for i in T.unroll(mt):
                for b in T.unroll(nb):
                    for j in T.unroll(2):
                        for r2 in T.unroll(4):
                            grow = m0 + wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                            gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                            if grow < M:
                                if split_k == 1:
                                    for v in T.vectorized(2):
                                        Y[grow, gcol + v] = T.cast(
                                            C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                                else:
                                    for v in T.vectorized(2):
                                        P[bk, grow, gcol + v] = C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]

            if split_k > 1:
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(counters[bm * n_tiles + bx], 1, return_prev=True)
                T.sync_threads()
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    for i, j in T.Parallel(block_M, block_N):
                        if m0 + i < M:
                            Y[m0 + i, bx * block_N + j] = T.cast(_partial_sum(P, m0 + i, bx * block_N + j), bf16)
                    if tid == 0:
                        counters[bm * n_tiles + bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# Unified "range" kernel (block_K = 64): grid of G CTAs, G a RUN-TIME value. The (tile, k-tile) iteration space
# (tile-major; tile t = (m tile t % m_tiles, n tile t // m_tiles), so the m tiles of one weight column are adjacent) is
# cut into G contiguous ranges [g*TT/G, (g+1)*TT/G). A CTA walks its range tile by tile; a tile it covers completely is
# written straight to Y, a partial segment goes to its own fp32 slot P[2g + last] (last = the segment ends the CTA's
# range; a CTA has at most one first and one last partial segment). The last CTA to arrive at a split tile (atomic
# ticket) sums the slots of CTAs g0..g1 in that fixed order: deterministic and graph-replayable (tickets reset
# themselves). One compiled kernel serves all partitionings:
#     G = tiles           plain tiling (no partials)
#     G = tiles * s       split-K by s
#     G = 188 * c         stream-K (Marlin-style: no wave quantisation of the tile grid)
# The k loop is a hand-written cp.async pipeline: a C++ stage loader (xl_load_stage, generated per config) writes the
# A tile (same 128-byte XOR swizzle TileLang uses for ldmatrix), packed weights and scales to raw shared addresses, so
# TileLang sees only shared READS in the loop and inserts no barriers of its own: one commit group per stage, constant
# wait counts, ONE __syncthreads per k tile (T.Pipelined needs a static extent and emits two).
def _xl_loader_src(block_M, nblk, K, threads, block_K=64):
    rowch = block_K // 8                    # 16-byte chunks per A row
    kch = block_K // 32                     # 32-wide k chunks (one 512-byte weight block + 64 scale bytes each)
    a_n, w_n, s_n = block_M * rowch, nblk * kch * 32, nblk * kch * 4      # 16-byte copies per stage
    assert block_K in (32, 64) and s_n <= threads
    # per-thread copy loops; a trailing partial round (small tiles: fewer copies than threads) is guarded
    a_it, a_g = -(-a_n // threads), ("" if a_n % threads == 0 else f"if (e < {a_n}u) ")
    w_it, w_g = -(-w_n // threads), ("" if w_n % threads == 0 else f"if (e < {w_n}u) ")
    # TileLang's make_mma_swizzle_layout for bf16 rows of 128 B (block_K 64): chunk ^ (row % 8); of 64 B: chunk ^ (row % 8) / 2
    swz = "(ch ^ (row & 7))" if block_K == 64 else "(ch ^ ((row & 7) >> 1))"
    return _XL_SRC + f"""
__device__ __forceinline__ unsigned xl_saddr(const void *p) {{ return static_cast<unsigned>(__cvta_generic_to_shared(p)); }}
// no "memory" clobber (as TileLang's cp_async_gs): the copy is asynchronous; reads of a stage are ordered after its
// xl_wait + __syncthreads, and nvcc stays free to interleave the next stage's issue with this stage's LDS/LDSM/HMMA.
__device__ __forceinline__ void xl_cp16s(unsigned s, const void *g) {{
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(s), "l"(g));
}}
// zero-filling variant (src-size n = 0 or 16): padding rows >= M read nothing (no L2 hot spot on row M-1)
__device__ __forceinline__ void xl_cp16z(unsigned s, const void *g, unsigned n) {{
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" :: "r"(s), "l"(g), "r"(n));
}}
// one pipeline stage (k tile of {block_K}): A rows m0.. (clamped to M-1), 32x32 weight blocks bx*{nblk}.. and scales.
// 32-bit byte offsets (x <= 36 MB, packed weights <= 90 MB): the tile-invariant parts hoist out of the k loop.
__device__ __forceinline__ void xl_load_stage(unsigned sA, unsigned sW, unsigned sS, int st, const void *xp,
                                              const void *wp, const void *sp, int M, int m0, int bx, int kc0, int tid) {{
  const char *x = static_cast<const char *>(xp), *w = static_cast<const char *>(wp), *sg = static_cast<const char *>(sp);
  const unsigned ust = st, ukc = kc0, ut = tid, ubx = bx;
#pragma unroll
  for (unsigned v = 0; v < {a_it}u; ++v) {{
    const unsigned e = ut + v * {threads}u, row = e / {rowch}u, ch = e % {rowch}u;
    const bool ok = m0 + (int)row < M;
    const unsigned grow = ok ? (unsigned)(m0 + (int)row) : (unsigned)(M - 1);
    {a_g}xl_cp16z(sA + ((ust * {block_M}u + row) * {block_K * 2}u + ({swz} << 4)), x + (grow * {K * 2}u + ukc * 64u + ch * 16u),
                 ok ? 16u : 0u);
  }}
#pragma unroll
  for (unsigned v = 0; v < {w_it}u; ++v) {{
    const unsigned e = ut + v * {threads}u, r = e / {kch * 32}u, c = (e >> 5) % {kch}u, l = e & 31u;
    {w_g}xl_cp16s(sW + (((ust * {nblk}u + r) * {kch}u + c) * 512u + l * 16u),
             w + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 512u + l * 16u));
  }}
  if (ut < {s_n}u) {{
    const unsigned r = ut / {kch * 4}u, c = (ut >> 2) % {kch}u, l = ut & 3u;
    xl_cp16s(sS + (((ust * {nblk}u + r) * {kch}u + c) * 64u + l * 16u),
             sg + (((ubx * {nblk}u + r) * {K // 32}u + ukc + c) * 64u + l * 16u));
  }}
}}
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_regfrag_xl_sk_kernel(N: int, K: int, warps_m: int, WM: int, warps_n: int, nb: int, num_stages: int,
                               min_blocks: int = 1, block_K: int = 64, late_issue: int = 0, warps_k: int = 1):
    """warps_k > 1: Marlin-style intra-CTA split-K. warps_k groups of warps_m x warps_n warps each take every
    warps_k-th 32-wide k chunk of every k tile; at the end of a tile the groups' fp32 accumulators are summed through
    shared memory in fixed group order (deterministic) and group 0 runs the epilogue."""
    assert WM % 16 == 0 and nb in (1, 2) and num_stages >= 2 and block_K in (32, 64)
    block_M = warps_m * WM
    WN = 32 * nb
    block_N = warps_n * WN
    nblk = block_N // 32
    threads = 32 * warps_m * warps_n * warps_k
    kchunks = block_K // 32
    assert kchunks % warps_k == 0, (block_K, warps_k)
    warps_mn = warps_m * warps_n
    CL = WM // 16 * nb * 2 * 8                       # accumulator floats per lane
    # reduction buffer (warps_k - 1) x warps_mn x IDXR x 32 lanes x fp32 with IDXR <= 32 / (warps_k - 1): <= 4 KB per
    # warp of a k group (16 KB for 4 warps); accumulators go through it in RND rounds of IDXR slots per lane
    IDXR = CL
    while warps_k > 1 and IDXR * (warps_k - 1) > 32 and IDXR % 2 == 0:
        IDXR //= 2
    assert CL % IDXR == 0
    RND = CL // IDXR
    n_tiles = N // block_N
    KT = K // block_K
    assert N % block_N == 0 and K % block_K == 0, (N, K, block_N, block_K)
    mt = WM // 16
    M = T.dynamic("M")
    G = T.dynamic("G")
    PS = T.dynamic("PS")
    bf16, f32, i32, i64 = T.bfloat16, T.float32, T.int32, T.int64
    src = _xl_loader_src(block_M, nblk, K, threads, block_K)

    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=warps_m, block_col_warps=warps_n,
                                      warp_row_tiles=WM, warp_col_tiles=WN, chunk=block_K)

    _LATE_ISSUE = bool(late_issue)

    @T.macro
    def _issue(kt, x, Wg, Sg, sb, stg, M, m0, bx, ks, ke, tid):
        """cp.async issue of k tile ks + kt + num_stages - 1 into stage stg[1] (if any) + its commit group."""
        if kt + num_stages - 1 < ke - ks:
            T.call_extern("handle", "xl_load_stage", sb[0], sb[1], sb[2], stg[1], T.access_ptr(x, "r"),
                          T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), M, m0, bx,
                          (ks + kt + num_stages - 1) * kchunks, tid)
        T.call_extern("handle", "xl_commit")

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Y: T.Tensor((M, N), bf16),
        Wg: T.Tensor((N // 32, K // 32, 128), i32),
        Sg: T.Tensor((N // 32, K // 32, 16), i32),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor((PS, block_M, block_N), f32),
        counters: T.Tensor((_CNT,), i32),
        gdim: T.Tensor((G,), T.int8),
    ):
        with T.Kernel(G, threads=threads, prelude=src) as g:
            T.assume(M >= 1)  # v0.1.15 (#3294): M >= 1 is the launch contract; keeps the unguarded loads
            T.assume(G >= 1)  # ... and G >= 1 (the grid): plain divisions by G instead of floor-division fix-ups
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            A_s = T.alloc_shared((num_stages, block_M, block_K), bf16)
            Wq_s = T.alloc_shared((num_stages, nblk, kchunks, 128), i32)
            S_s = T.alloc_shared((num_stages, nblk, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            sb = T.alloc_local((3,), T.uint32)
            stg = T.alloc_local((2,), i32)
            ts = T.alloc_local((4,), i32)
            q_l = T.alloc_local((nb * 4,), i32)
            s_l = T.alloc_local((nb * 2,), i32)
            B_l = T.alloc_local((nb * 32,), bf16)
            A_l = T.alloc_local((mt * 8,), bf16)
            C_l = T.alloc_local((mt * nb * 2 * 8,), f32)
            if warps_k > 1:
                Red_s = T.alloc_shared((warps_k - 1, warps_mn, IDXR, 32), f32)
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            wm = warp % warps_m                         # (matches the emitter's warp_m = (tid // 32) % warps_m)
            wn = (warp // warps_m) % warps_n
            wk = warp // warps_mn
            wmn = warp % warps_mn
            sb[0] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(A_s, "w"))
            sb[1] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(Wq_s, "w"))
            sb[2] = T.call_extern(T.uint32, "xl_saddr", T.access_ptr(S_s, "w"))
            # per-CTA / per-tile scalars live in registers (bound Python expressions would be re-inlined at every
            # use, e.g. a run-time division inside the k loop)
            cs = T.alloc_local((4,), i32)
            cs[0] = T.ceildiv(M, block_M)                                   # m tiles
            cs[1] = cs[0] * (n_tiles * KT)                                  # TT: iterations of the whole problem
            cs[2] = T.cast((T.cast(g, i64) * cs[1]) // G, i32)              # it0
            cs[3] = T.cast((T.cast(g + 1, i64) * cs[1]) // G, i32)          # it1
            m_tiles = cs[0]
            TT = cs[1]
            it0 = cs[2]
            it1 = cs[3]

            for t in T.serial(it0 // KT, (it1 - 1) // KT + 1):
                ts[0] = (t % m_tiles) * block_M                             # m0
                ts[1] = t // m_tiles                                        # bx (n tile)
                ts[2] = T.max(it0, t * KT) - t * KT                         # ks
                ts[3] = T.min(it1, (t + 1) * KT) - t * KT                   # ke
                m0 = ts[0]
                bx = ts[1]
                ks = ts[2]
                ke = ts[3]
                T.clear(C_l)
                for p in T.unroll(num_stages - 1):
                    if p < ke - ks:
                        T.call_extern("handle", "xl_load_stage", sb[0], sb[1], sb[2], p, T.access_ptr(x, "r"),
                                      T.access_ptr(Wg, "r"), T.access_ptr(Sg, "r"), M, m0, bx, (ks + p) * kchunks, tid)
                    T.call_extern("handle", "xl_commit")
                stg[0] = 0                                # stage consumed this k tile
                stg[1] = num_stages - 1                   # stage refilled this k tile
                for kt in T.serial(ke - ks):
                    T.call_extern("handle", f"xl_wait<{num_stages - 2}>")
                    T.sync_threads()
                    if not _LATE_ISSUE:
                        _issue(kt, x, Wg, Sg, sb, stg, M, m0, bx, ks, ke, tid)
                    st = stg[0]
                    for cc in T.unroll(kchunks // warps_k):
                        c = cc * warps_k + wk if warps_k > 1 else cc
                        for b in T.unroll(nb):
                            for v in T.vectorized(4):
                                q_l[b * 4 + v] = Wq_s[st, wn * nb + b, c, lane * 4 + v]
                            for v in T.vectorized(2):
                                s_l[b * 2 + v] = S_s[st, wn * nb + b, c, (lane // 4) * 2 + v]
                            T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l[b * 4], "r", extent=4),
                                          T.access_ptr(s_l[b * 2], "r", extent=2),
                                          T.access_ptr(B_l[b * 32], "w", extent=32))
                        for s in T.unroll(2):
                            emitter.ldmatrix_a(A_l, A_s[st, 0:block_M, 0:block_K], c * 2 + s)
                            for i in T.unroll(mt):
                                for b in T.unroll(nb):
                                    for j in T.unroll(2):
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32",
                                                  A_l.data, i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                                  ((i * nb + b) * 2 + j) * 8, T.bool(False))
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32",
                                                  A_l.data, i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                                  ((i * nb + b) * 2 + j) * 8 + 4, T.bool(False))
                            if _LATE_ISSUE:
                                # next stage's cp.async issue after the first k16 step's MMAs: the tensor pipe restarts
                                # right after the barrier instead of behind ~100 address/LDGSTS instructions
                                if cc == 0:
                                    if s == 0:
                                        _issue(kt, x, Wg, Sg, sb, stg, M, m0, bx, ks, ke, tid)
                    stg[0] = T.if_then_else(stg[0] == num_stages - 1, 0, stg[0] + 1)
                    stg[1] = T.if_then_else(stg[1] == num_stages - 1, 0, stg[1] + 1)
                T.call_extern("handle", "xl_wait<0>")
                T.sync_threads()
                if warps_k > 1:
                    for r in T.unroll(RND):
                        if wk > 0:
                            for q in T.unroll(IDXR):
                                Red_s[wk - 1, wmn, q, lane] = C_l[r * IDXR + q]
                        T.sync_threads()
                        if wk == 0:
                            for kk in T.unroll(warps_k - 1):
                                for q in T.unroll(IDXR):
                                    C_l[r * IDXR + q] = C_l[r * IDXR + q] + Red_s[kk, wmn, q, lane]
                        T.sync_threads()

                if ks == 0 and ke == KT:
                    if wk == 0:
                        for i in T.unroll(mt):
                            for b in T.unroll(nb):
                                for j in T.unroll(2):
                                    for r2 in T.unroll(4):
                                        prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                        gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                        if m0 + prow < M:
                                            for v in T.vectorized(2):
                                                Y[m0 + prow, gcol + v] = T.cast(
                                                    C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                else:
                    if wk == 0:
                        slot = 2 * g + T.if_then_else(it1 <= (t + 1) * KT, 1, 0)
                        for i in T.unroll(mt):
                            for b in T.unroll(nb):
                                for j in T.unroll(2):
                                    for r2 in T.unroll(4):
                                        prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                        pcol = wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                        for v in T.vectorized(2):
                                            P[slot, prow, pcol + v] = (C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v]
                                                                       * alpha[bx * block_N + pcol + v])
                    T.call_extern("handle", "__threadfence")
                    T.sync_threads()
                    if tid == 0:
                        ticket_s[0] = T.atomic_add(counters[t], 1, return_prev=True)
                    T.sync_threads()
                    g0 = T.cast(((T.cast(t * KT + 1, i64) * G) - 1) // TT, i32)
                    g1 = T.cast(((T.cast((t + 1) * KT, i64) * G) - 1) // TT, i32)
                    if ticket_s[0] == g1 - g0:
                        T.call_extern("handle", "__threadfence")
                        if wk == 0:
                            T.clear(C_l)
                            for q in T.serial(g1 - g0 + 1):
                                sl = 2 * (g0 + q) + T.if_then_else(
                                    T.cast((T.cast(g0 + q + 1, i64) * TT) // G, i32) <= (t + 1) * KT, 1, 0)
                                for i in T.unroll(mt):
                                    for b in T.unroll(nb):
                                        for j in T.unroll(2):
                                            for r2 in T.unroll(4):
                                                prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                                pcol = wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                                for v in T.vectorized(2):
                                                    C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] = (
                                                        C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] + P[sl, prow, pcol + v])
                            for i in T.unroll(mt):
                                for b in T.unroll(nb):
                                    for j in T.unroll(2):
                                        for r2 in T.unroll(4):
                                            prow = wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                                            gcol = bx * block_N + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                                            if m0 + prow < M:
                                                for v in T.vectorized(2):
                                                    Y[m0 + prow, gcol + v] = T.cast(
                                                        C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v], bf16)
                        if tid == 0:
                            counters[t] = 0
                    T.sync_threads()

    return main


# ---------------------------------------------------------------------------------------------------------------
# Host side.
_NUM_SMS = _rf._NUM_SMS
_LATE_ISSUE_DEFAULT = 0        # range kernel: issue the next stage after the first k16 step's MMAs (see _issue)
_COUNTERS = {}
_GDIM = {}
_DUMMY = {}
_KERNELS = {}


def _gdim(device):
    """int8 buffer whose [:G] view passes the run-time grid size G of the range kernel (contents unused)."""
    import torch
    b = _GDIM.get(device)
    if b is None:
        b = _GDIM[device] = torch.zeros(1 << 16, dtype=torch.int8, device=device)
    return b


def _dummy_p(device, block_M, block_N):
    import torch
    key = (device, block_M, block_N)
    p = _DUMMY.get(key)
    if p is None:
        p = _DUMMY[key] = torch.zeros(1, block_M, block_N, dtype=torch.float32, device=device)
    return p


def _counters(device):
    import torch
    c = _COUNTERS.get(device)
    if c is None:
        c = _COUNTERS[device] = torch.zeros(_CNT, dtype=torch.int32, device=device)
    return c


def prepare(packed_int8, scale_fp8, alpha_fp32):
    """decode_regfrag.prepare (one weight copy for both kernels) + this module's split-K ticket buffer."""
    prep = _rf.prepare(packed_int8, scale_fp8, alpha_fp32)
    _counters(prep.device)
    _gdim(prep.device)
    return prep


# Tuned buckets: (N, K) -> [(M_max, cfg), ...] ascending; the first bucket with M <= M_max wins. Filled from the
# clean xl_tune sweeps (k2cand/xl); shapes not listed use _generic_config.
_TUNED = {
    (16384, 5120): [(1, ('rf',)), (8, (1, 16, 4, 1, 64, 4, 4)), (16, (1, 16, 4, 1, 64, 2, 4)), (32, (1, 32, 8, 1, 64, 2, 4)), (49, ('rf',)), (56, (1, 64, 4, 1, 64, 4, 3)), (64, ('rf',)), (80, (1, 80, 4, 1, 64, -2, 3, 2)), (96, (2, 48, 4, 1, 64, -1, 3)), (112, (1, 112, 4, 1, 64, -2, 2, 2)), (128, (1, 64, 8, 1, 64, -1, 3)), (160, (1, 80, 4, 1, 64, -2, 3, 2)), (192, (1, 96, 4, 1, 64, -2, 2, 2)), (224, (1, 112, 4, 1, 64, -2, 2, 2)), (240, (1, 80, 4, 1, 64, -2, 3, 2)), (256, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (288, (1, 96, 4, 1, 64, -2, 2, 2)), (320, (1, 80, 4, 1, 64, -2, 3, 2)), (384, (1, 96, 4, 1, 64, -2, 2, 2)), (448, (1, 112, 4, 1, 64, -2, 2, 2)), (512, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (576, (1, 64, 4, 1, 64, -2, 3, 2, 1)), (640, (2, 64, 4, 2, 64, 1, 3)), (896, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (1024, (2, 64, 4, 2, 64, 1, 3))],
    (14336, 5120): [(1, ('rf',)), (8, (1, 16, 4, 1, 64, 4, 4)), (16, (1, 16, 4, 1, 64, 2, 4)), (32, (1, 32, 8, 1, 64, 2, 4)), (49, ('rf',)), (56, (1, 64, 4, 1, 64, 5, 3)), (64, ('rf',)), (80, (1, 80, 4, 1, 64, -2, 3, 2)), (96, (2, 48, 4, 1, 64, -1, 3, 1, 1)), (112, ('rf',)), (128, (1, 64, 4, 1, 64, -2, 3, 2, 1)), (160, (1, 80, 4, 1, 64, -2, 3, 2)), (176, (1, 96, 4, 1, 64, -2, 2, 2)), (192, (1, 64, 8, 1, 64, 1, 3)), (224, (1, 112, 4, 1, 64, -2, 2, 2)), (240, (1, 80, 4, 1, 64, -2, 3, 2)), (256, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (288, (1, 96, 4, 1, 64, -2, 2, 2)), (320, (1, 64, 4, 1, 64, 1, 3)), (384, (2, 64, 4, 2, 64, 1, 3)), (448, (1, 112, 4, 1, 64, -2, 2, 2)), (512, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (576, (1, 64, 4, 1, 64, -2, 3, 2, 1)), (640, (1, 64, 4, 1, 64, 1, 3)), (768, (2, 64, 4, 2, 64, 1, 3)), (1024, (2, 64, 4, 2, 64, -1, 3, 1, 1))],
    (5120, 6144): [(8, ('rf',)), (16, (1, 16, 4, 1, 64, 4, 4)), (32, (1, 32, 4, 1, 64, 4, 4)), (49, ('rf',)), (56, (1, 64, 4, 1, 64, 4, 3)), (80, ('rf',)), (96, (2, 48, 4, 1, 64, -1, 3)), (112, (1, 64, 4, 1, 64, 2, 3)), (128, (1, 64, 4, 1, 64, 4, 3)), (160, (1, 80, 4, 1, 64, -2, 3, 2)), (192, (1, 64, 4, 1, 64, 3, 3)), (240, (1, 80, 4, 1, 64, -2, 3, 2)), (256, (1, 64, 8, 1, 64, 2, 3)), (288, (1, 96, 4, 1, 64, -2, 2, 2)), (320, (1, 80, 4, 1, 64, -2, 3, 2)), (384, (2, 64, 4, 2, 64, 3, 3)), (448, (1, 64, 4, 1, 64, 2, 3)), (512, (2, 64, 4, 2, 64, 2, 3)), (576, (1, 64, 4, 1, 64, 1, 3)), (768, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (896, (1, 64, 4, 1, 64, 1, 3)), (1024, (2, 64, 4, 2, 64, 1, 3))],
    (34816, 5120): [(16, (1, 16, 8, 1, 64, 1, 6)), (32, (1, 32, 8, 1, 64, 1, 4)), (48, ('rf',)), (56, (1, 64, 4, 1, 64, 2, 3)), (64, ('rf',)), (80, (1, 80, 4, 1, 64, -2, 3, 2, 1)), (96, (1, 96, 4, 1, 64, -2, 2, 2)), (112, (1, 112, 4, 1, 64, -2, 2, 2, 1)), (128, (1, 64, 4, 1, 64, 1, 3)), (160, (1, 80, 4, 1, 64, -2, 3, 2)), (192, (1, 96, 4, 1, 64, -2, 2, 2)), (224, (1, 112, 4, 1, 64, -2, 2, 2)), (240, (1, 80, 4, 1, 64, -2, 3, 2)), (256, (1, 64, 4, 1, 64, 1, 3)), (288, (1, 96, 4, 1, 64, -2, 2, 2)), (320, (1, 80, 4, 1, 64, -2, 3, 2)), (384, (1, 128, 4, 1, 64, -2, 2, 2)), (448, (1, 112, 4, 1, 64, -2, 2, 2)), (512, (2, 64, 4, 2, 64, 1, 3)), (576, (1, 64, 4, 1, 64, 1, 3)), (640, (2, 64, 4, 2, 64, 1, 3)), (896, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (1024, (2, 64, 4, 2, 64, 1, 3))],
    (5120, 17408): [(8, ('rf',)), (16, (1, 16, 4, 1, 64, 4, 4)), (32, (1, 32, 4, 1, 64, 4, 4)), (64, ('rf',)), (80, (1, 80, 4, 1, 64, -2, 3, 2)), (96, (2, 48, 4, 1, 64, -1, 3, 1, 1)), (112, ('rf',)), (128, (1, 64, 8, 1, 64, -1, 3)), (160, (1, 80, 4, 1, 64, -2, 3, 2)), (176, (1, 96, 4, 1, 64, -2, 2, 2)), (192, (1, 64, 8, 1, 64, -1, 3)), (224, (1, 112, 4, 1, 64, -2, 2, 2)), (240, (1, 80, 4, 1, 64, -2, 3, 2)), (256, (2, 64, 4, 2, 64, 4, 3)), (288, (1, 96, 4, 1, 64, -2, 2, 2)), (320, (1, 80, 4, 1, 64, -2, 3, 2)), (384, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (448, (1, 112, 4, 1, 64, -2, 2, 2)), (512, (2, 64, 4, 2, 64, -1, 3, 1, 1)), (576, (1, 64, 4, 1, 64, 1, 3)), (1024, (2, 64, 4, 2, 64, -1, 3, 1, 1))],
}


def _generic_config(N, K, M):
    """Heuristic for untuned shapes: decode_regfrag (64-row passes above 64) up to 128 rows, then 128x256 tiles."""
    if M <= 128:
        return ("rf",)
    block_M, block_N = 128, 256
    if N % block_N:
        block_N = 128
    tiles = -(-M // block_M) * (N // block_N)
    split_k = _rf._choose_split_balanced(tiles, K // 64, max_split=4)
    return (2, 64, block_N // 64, 2, 64, split_k, 3) if block_N == 256 else (2, 64, 4, 1, 64, split_k, 3)


def config(N, K, M):
    """Config for 1 <= M <= 1024: ("rf"[, decode_regfrag cfg]) or (warps_m, WM, warps_n, nb, block_K, split_k,
    num_stages[, min_blocks[, late_issue]]).

    split_k >= 1: TileLang-pipelined tile kernel with split-K; 1000 + s: range kernel with G = tiles * s;
    -c: range kernel, stream-K over 188 * c CTAs."""
    for m_max, cfg in _TUNED.get((N, K), ()):
        if M <= m_max:
            return cfg
    return _generic_config(N, K, M)


def smem_bytes(cfg):
    warps_m, WM, warps_n, nb, block_K, split_k, num_stages = cfg[:7]
    block_M, nblk = warps_m * WM, warps_n * nb
    return num_stages * (block_M * block_K * 2 + nblk * (block_K // 32) * 576) + 4


def _rf_padded(M):
    """decode_regfrag clamps padding rows (M..block_M-1) to row M-1, which makes every CTA hammer the same L2 lines once
    there are many of them (gate_up M=33: -20%). Above 16 rows, a row count that is not a multiple of 16 therefore runs
    decode_regfrag's tile-kernel twin (same tile / warps / split / stages, padding rows wrap to distinct rows)."""
    return M > 16 and M % 16 != 0


def _rf_twin(rcfg):
    block_M, n_warps, block_K, split_k, num_stages = rcfg
    return (1, block_M, n_warps, 1, block_K, split_k, num_stages)


def _rf_pass(xs, ys, prep):
    """decode_regfrag's kernel on rows xs (<= 64, contiguous) writing into the row view ys (same as _rf.linear)."""
    import torch
    M, K = xs.shape
    N = prep.N
    cfg = _rf.config(N, K, M)
    if _rf_padded(M):
        _run(xs, ys, prep, _rf_twin(cfg))
        return
    key = (N, K) + tuple(cfg)
    kernel = _rf._KERNELS.get(key)
    if kernel is None:
        kernel = _rf._KERNELS[key] = _rf.nvfp4_regfrag_kernel(N, K, *cfg)
    if cfg[3] > 1:
        P = torch.empty(cfg[3], M, N, dtype=torch.float32, device=xs.device)
    else:
        P = _rf._DUMMY.get(xs.device)
        if P is None:
            P = _rf._DUMMY[xs.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=xs.device)
    kernel(xs, ys, prep.Wg, prep.Sg, prep.alpha, P, prep.counters)


def linear(x, prep, cfg=None):
    """y = x @ W^T (bf16 out) for 1 <= M <= 1024 tokens; deterministic, CUDA-graph capturable.

    cfg optionally overrides config(): ("rf"[, block_M, n_warps, block_K, split_k, num_stages]) = decode_regfrag
    (64-row passes above 64 rows), or (warps_m, WM, warps_n, nb, block_K, split_k, num_stages[, min_blocks[, late
    [, warps_k]]]).
    Split-K launches use a per-device ticket buffer: XL launches of one device must not run concurrently on two
    streams (same rule as decode_regfrag's per-layer counters)."""
    import torch
    M, K = x.shape
    N = prep.N
    assert K == prep.K and 1 <= M <= SUPPORTS_M, (x.shape, prep.N, prep.K)
    if cfg is None:
        cfg = config(N, K, M)
    if cfg[0] == "rf" and M <= _rf.SUPPORTS_M:         # decode_regfrag itself (optionally with its cfg overridden)
        rcfg = tuple(cfg[1:]) if len(cfg) > 1 else _rf.config(N, K, M)
        if not _rf_padded(M):
            return _rf.linear(x, prep, cfg=rcfg)
        cfg = _rf_twin(rcfg)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if cfg[0] == "rf":                                 # decode_regfrag passes over 64-row slices (Marlin's structure)
        x = x.contiguous()
        for r0 in range(0, M, _rf.SUPPORTS_M):
            _rf_pass(x[r0:r0 + _rf.SUPPORTS_M], y[r0:r0 + _rf.SUPPORTS_M], prep)
        return y
    _run(x.contiguous(), y, prep, cfg)
    return y


def _run(x, y, prep, cfg):
    """Launch the XL tile or range kernel for cfg on contiguous x (M, K) into y (M, N)."""
    import torch
    M, K = x.shape
    N = prep.N
    warps_m, WM, warps_n, nb, block_K, split_k, num_stages = cfg[:7]
    min_blocks = cfg[7] if len(cfg) > 7 else 1
    late_issue = cfg[8] if len(cfg) > 8 else _LATE_ISSUE_DEFAULT
    warps_k = cfg[9] if len(cfg) > 9 else 1
    key = (N, K) + tuple(cfg[:7]) + (min_blocks, late_issue, warps_k)
    if split_k < 0 or split_k >= 1000:
        # unified range kernel: split_k = -c -> stream-K over G = 188*c CTAs; 1000 + s -> G = tiles * s (s = 1: plain)
        block_M, block_N = warps_m * WM, warps_n * 32 * nb
        tiles = -(-M // block_M) * (N // block_N)
        G = _NUM_SMS * (-split_k) if split_k < 0 else tiles * (split_k - 1000)
        assert tiles <= _CNT and 1 <= G <= tiles * (K // block_K), (tiles, G)
        kernel = _KERNELS.get(key)
        if kernel is None:
            kernel = _KERNELS[key] = nvfp4_regfrag_xl_sk_kernel(N, K, warps_m, WM, warps_n, nb, num_stages, min_blocks,
                                                                block_K, late_issue, warps_k)
        if G == tiles:                                 # every CTA covers exactly one whole tile: P is never touched
            P = _dummy_p(x.device, block_M, block_N)
        else:
            P = torch.empty(2 * G, block_M, block_N, dtype=torch.float32, device=x.device)
        kernel(x, y, prep.Wg, prep.Sg, prep.alpha, P, _counters(x.device), _gdim(x.device)[:G])
        return
    assert warps_k == 1, "warps_k > 1 needs the range kernel (split_k < 0 or >= 1000)"
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = nvfp4_regfrag_xl_kernel(N, K, warps_m, WM, warps_n, nb, block_K, split_k, num_stages,
                                                         min_blocks)
    if split_k > 1:
        assert -(-M // (warps_m * WM)) * (N // (warps_n * 32 * nb)) <= _CNT
        P = torch.empty(split_k, M, N, dtype=torch.float32, device=x.device)
    else:
        P = _DUMMY.get(x.device)
        if P is None:
            P = _DUMMY[x.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=x.device)
    kernel(x, y, prep.Wg, prep.Sg, prep.alpha, P, _counters(x.device))
