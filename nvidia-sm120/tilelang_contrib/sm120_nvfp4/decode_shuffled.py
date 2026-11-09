# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 decode GEMM candidate "shuffled": Marlin-style pre-shuffled NVFP4 weights, global -> registers, mma.sync directly.

y = x @ W^T for 1 <= M <= 64 tokens, x bf16 (M, K), W NVFP4 (packed e2m1 + ue4m3 block scales + fp32 alpha per column).
Exact class: every weight is dequantised to BF16 exactly (fp4 * ue4m3 has <= 6 significant bits), accumulated in fp32
by m16n8k16 BF16 tensor-core MMAs and rounded to bf16 once. Deterministic (split-K partials summed in fixed order).

Design:
  * prepare() permutes the packed weights ONCE into the per-lane order of the mma.m16n8k16 B fragment. At run time each
    lane issues one 16-byte LDG per (n8 tile, k128 chunk) straight into registers: the 32 nibbles it needs for 8 k16
    steps. No shared memory, no ldmatrix, no re-layout for the weights. Scales (8 bytes per lane per tile-chunk) come
    the same way (the 4 lanes of a quad read the same 8 bytes). A CTA streams one contiguous 8 KB weight block + 1 KB
    of scales per chunk, so every CTA reads two sequential streams.
  * Dequantisation happens in registers with a bit-twiddle: the 8 nibbles of a word are stored at bit positions such
    that `(x << {0,3,6}) & 0x81C081C0` (plus one multiply-gather) drops each nibble into the sign / low exponent /
    top mantissa bits of a bf16 whose value is exactly fp4 * 2^-126 (e=0 codes become bf16 subnormals -> 0 and 0.5).
    One `mul.rn.bf16x2` by the block multiplier bf16(scale * 2^119) (two multipliers built with one PRMT + one LEA
    from the raw ue4m3 bytes, the half broadcast folded into the HMUL2 operand) gives fp4 * scale * 2^-7 exactly;
    alpha*128 in the epilogue restores the magnitude.
    Per 8 weights: 9 bit ops + 2 multiplier ops + 4 HMUL2 = 15 ALU ops, i.e. 1.875 per weight (SASS-verified).
  * Register multi-buffering (`stages`) keeps (stages-1) chunks of 16-byte loads in flight per lane; the small A operand
    (<= 64 rows) is double-buffered in padded (bank-conflict-free) shared memory with cp.async one chunk ahead
    (rows >= M are zero-filled, not read).
  * Deterministic split-K for narrow layers: every CTA writes its fp32 partial tile, then takes a ticket from a per
    column-block counter; the last CTA to arrive sums the split_k partials in fixed order s = 0..split_k-1 (identical
    result whichever CTA is last) and writes bf16. One kernel launch, no fp32 atomics, counters self-reset.
    split_k (a divisor of K/128) is picked by a small time model: DRAM stream with enough bytes in flight vs. the
    busiest SM's MMA work (matters from M ~ 32 on, where BF16 tensor math becomes the floor) plus the reduction tail.

Per CTA (4 warps x 32 columns, 128 threads): block_N = 128, one barrier per 128-k chunk. Per warp and chunk: 4 LDG.128
(weights) + 4 LDG.64 (scales), 8 LDSM (A), 32 * block_M/16 HMMA and ~238 dequant ALU instructions per lane for its
128 weights (1.86 per weight; the whole chunk is 301 SASS instructions at block_M = 16). Registers 136 / 168 / 194 / 225 for
block_M = 16 / 32 / 48 / 64 -> 3 / 3 / 2 / 2 CTAs per SM. In flight: (stages-1) chunks x 4 tiles x 576 B = 4.6 KB
per warp, 18.4 KB per CTA (37-55 KB per SM when 2-3 CTAs are resident).
Bytes per (n8 tile, k128 chunk) per warp: 512 B weights + 64 B scales (exactly the NVFP4 payload, 12.5% scale overhead,
zero waste), read once; per chunk a CTA reads 8 KB of weights and 1 KB of scales, each contiguous and sequential
over the chunks. A: M * 256 B per chunk per CTA from L2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import tilelang
import tilelang.language as T
from tvm.ir import Range
from tvm.tirx import BufferRegion
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

SUPPORTS_M = 64
NUM_SMS = 188
BK = 128          # k per chunk (8 k16 steps): one 16-byte weight load per lane per n8 tile
BKP = BK + 8      # padded shared-memory row (bf16): 272 B stride -> conflict-free ldmatrix

# Bit positions (sign, e1, e0, m) of the 8 nibbles of a stored word. Slot j -> register r_{j//2}, half j%2 (lo, hi).
#   MASK = 0x81C081C0: bf16 bits 15 (sign), 8,7 (exponent low bits), 6 (mantissa msb) in each half.
#   r0 = x & MASK, r1 = (x<<3) & MASK, r2 = (x<<6) & MASK            (1 + 2 + 2 ops)
#   r3 = (((x>>8) & 0x006C006C) * 0x421) & MASK                         (4 ops) the multiply by 1 + 2^5 + 2^10 ORs three
#        shifted copies of the 8 remaining bits; the layout below makes the copies disjoint (no carries), so the mask
#        picks exactly the wanted bit for every target (found by exhaustive search; 9 ops per 8 weights in total).
_NIB_POS = [(15, 8, 7, 6), (31, 24, 23, 22), (12, 5, 4, 3), (28, 21, 20, 19), (9, 2, 1, 0), (25, 18, 17, 16),
            (13, 11, 10, 14), (29, 27, 26, 30)]

_CUDA_PRELUDE = r"""
// d = a * {lo(m), lo(m)} / a * {hi(m), hi(m)} (bf16x2): ptxas folds the half broadcast into the HMUL2 operand (.H0_H0)
__device__ __forceinline__ unsigned tl_k2_mul_lo(unsigned a, unsigned m) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {l, l};\n\tmul.rn.bf16x2 %0, %1, b;\n\t}"
      : "=r"(d) : "r"(a), "r"(m));
  return d;
}
__device__ __forceinline__ unsigned tl_k2_mul_hi(unsigned a, unsigned m) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {h, h};\n\tmul.rn.bf16x2 %0, %1, b;\n\t}"
      : "=r"(d) : "r"(a), "r"(m));
  return d;
}
// bf16x2 {m_j, m_j+1}, m = bf16(scale * 2^119) for the ue4m3 codes c in bytes j, j+1 of sw: bits = 0x7000 | c << 4
// (exact). PRMT puts the two codes in bytes 0 and 2 (zero bytes 1 and 3); one LEA shifts them into place and adds the
// exponent MSBs. Two multipliers for 2 ops.
__device__ __forceinline__ unsigned tl_k2_mult2(unsigned sw, int j) {
  unsigned t = __byte_perm(sw, 0u, 0x4040u + (unsigned)j + ((unsigned)(j + 1) << 8));
  return t * 16u + 0x70007000u;
}
// One stored word = 8 twiddled e2m1 nibbles = the B fragments (b0..b3) of two consecutive k16 steps for one n8 tile.
// sw holds 4 scale codes (bytes) for 4 k16 steps; bsel selects the byte of the first of the two steps.
// out0/out1 receive 4 bf16 each: {b0, b1, b2, b3} of step 2w and step 2w+1 (value = fp4 * scale * 2^-7, exact).
// 15 ops per 8 weights: 9 bit ops (r0..r3), PRMT + LEA (both multipliers), 4 HMUL2.
template <typename TOut>
__device__ __forceinline__ void tl_k2_dq_word(unsigned x, unsigned sw, int bsel, TOut *out0, TOut *out1) {
  unsigned m = tl_k2_mult2(sw, bsel);
  unsigned r0 = x & 0x81C081C0u;
  unsigned r1 = (x << 3) & 0x81C081C0u;
  unsigned r2 = (x << 6) & 0x81C081C0u;
  unsigned r3 = (((x >> 8) & 0x006C006Cu) * 0x421u) & 0x81C081C0u;
  unsigned *o0 = reinterpret_cast<unsigned *>(out0);
  unsigned *o1 = reinterpret_cast<unsigned *>(out1);
  o0[0] = tl_k2_mul_lo(r0, m);
  o0[1] = tl_k2_mul_lo(r1, m);
  o1[0] = tl_k2_mul_hi(r2, m);
  o1[1] = tl_k2_mul_hi(r3, m);
}
// Split-K arrival: called by one thread after the whole CTA stored (and fenced) its partial tile. Returns 1 for the
// last of the nsplit CTAs of this column block, which then owns the fixed-order reduction; it also resets the counter
// for the next launch (every CTA of the block has already arrived, nobody else touches it in this launch).
__device__ __forceinline__ int tl_k2_ticket(int *cnt, int nsplit) {
  int old = atomicAdd(cnt, 1);
  if (old == nsplit - 1) {
    atomicExch(cnt, 0);
    __threadfence();
    return 1;
  }
  return 0;
}
// Fixed-order split-K reduction of one (M x BLOCK_N) column block by the last-arriving CTA:
//   Y[r, c] = TY(alpha[c] * (P_0 + P_1 + ... + P_{SPLIT-1})[r, c]), P_s = rows [s*M, (s+1)*M) of P (row stride N).
// Each thread owns 4 consecutive columns (float4, L2-only loads: other CTAs wrote them) and every RSTEP-th row, so a
// warp moves 512 contiguous bytes per load and all (row, split) loads of a thread are independent.
template <int SPLIT, int BLOCK_M, int BLOCK_N, int THREADS, typename TY>
__device__ __forceinline__ void tl_k2_reduce(const float *P, TY *Y, const float *alpha, int M, int N, int n0) {
  constexpr int C4 = BLOCK_N / 4;
  constexpr int RSTEP = THREADS / C4;
  static_assert(BLOCK_N % 4 == 0 && THREADS % C4 == 0, "reduction tiling");
  const int t = threadIdx.x;
  const int c = n0 + (t % C4) * 4;
  const int rg = t / C4;
  const float4 a = *reinterpret_cast<const float4 *>(alpha + c);
#pragma unroll
  for (int i = 0; i < (BLOCK_M + RSTEP - 1) / RSTEP; ++i) {
    const int r = rg + i * RSTEP;
    if (r < M) {
      float4 acc = __ldcg(reinterpret_cast<const float4 *>(P + (size_t)r * N + c));
#pragma unroll
      for (int s = 1; s < SPLIT; ++s) {
        const float4 v = __ldcg(reinterpret_cast<const float4 *>(P + ((size_t)s * M + r) * N + c));
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      }
      TY *y = Y + (size_t)r * N + c;
      y[0] = TY(acc.x * a.x);
      y[1] = TY(acc.y * a.y);
      y[2] = TY(acc.z * a.z);
      y[3] = TY(acc.w * a.w);
    }
  }
}
"""

# Syncs are placed by hand (one barrier per chunk); bounds are guaranteed by construction (prepare pads nothing: all
# shapes are exact multiples, A rows are clamped + zero-filled), so the automatic guards are disabled.
_PASS_CFG = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
             tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
             tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
             tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
             tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True}    # keep `flag` out of A_s


@tilelang.jit(pass_configs=_PASS_CFG)
def nvfp4_shuffled_gemm_kernel(x, Wp, Sp, alpha, Y, P, cnt, block_M: int = 16, n_warps: int = 4, n_per_warp: int = 32,
                               stages: int = 3, split_k: int = 1):
    """Y = bf16(alpha * (x @ Wdq^T)) for the CTA's block_N columns (split_k CTAs per column block along K).

    split_k == 1: each CTA writes Y directly (P / cnt unused).
    split_k  > 1: P is the (split_k * M, N) fp32 partial workspace, cnt the per-column-block ticket counters (zero)."""
    M, N, K, NW, NS, PR, PN, NC = T.dynamic("M N K NW NS PR PN NC")
    x: T.Tensor[(M, K), T.bfloat16]
    Wp: T.Tensor[(NW, 4), T.uint32]        # (N*K/32, 4): per (n128, k128, n8 tile, lane) 4 twiddled words
    Sp: T.Tensor[(NS, 2), T.uint32]        # (N*K/128, 2): per (n128, k128, n8 tile, row) 8 scale codes
    alpha: T.Tensor[(N,), T.float32]       # per-column alpha * 128
    Y: T.Tensor[(M, N), T.bfloat16]
    P: T.Tensor[(PR, PN), T.float32]
    cnt: T.Tensor[(NC,), T.int32]

    assert block_M % 16 == 0 and 16 <= block_M <= 64
    assert n_per_warp % 16 == 0 and stages >= 2
    threads = n_warps * 32
    block_N = n_warps * n_per_warp
    warp_rows = block_M // 16
    warp_cols = n_per_warp // 16          # n16 tiles per warp (emitter granularity)
    n8_per_warp = n_per_warp // 8
    assert (block_M * 16) % threads == 0
    a_pieces = block_M * 16 // threads    # 16-byte cp.async pieces of A per thread per chunk
    assert block_N % 4 == 0 and threads % (block_N // 4) == 0    # reduction tiling (float4 columns x row groups)

    emitter = TensorCoreIntrinEmitter(a_dtype="bfloat16", b_dtype="bfloat16", accum_dtype="float32",
                                      a_transposed=False, b_transposed=True, block_row_warps=1, block_col_warps=n_warps,
                                      warp_row_tiles=block_M, warp_col_tiles=n_per_warp, chunk=BK)

    with T.Kernel(N // block_N, split_k, threads=threads, prelude=_CUDA_PRELUDE) as (bx, bk):
        A_s = T.alloc_shared((2 * block_M, BKP), T.bfloat16)              # 2 stages of A (padded rows)
        flag = T.alloc_shared((1,), T.int32)
        A_l = T.alloc_local((warp_rows * 8,), T.bfloat16)                 # A fragments of one k16 step
        B0_l = T.alloc_local((warp_cols * 8,), T.bfloat16)                # dequantised B fragments, step 2w
        B1_l = T.alloc_local((warp_cols * 8,), T.bfloat16)                # dequantised B fragments, step 2w+1
        W_l = T.alloc_local((stages, n8_per_warp, 4), T.uint32)           # raw weight words, register-staged
        S_l = T.alloc_local((stages, n8_per_warp, 2), T.uint32)           # raw scale codes, register-staged
        C_l = T.alloc_local((warp_rows * warp_cols * 8,), T.float32)

        tx = T.get_thread_binding()
        lane = tx % 32
        warp = tx // 32
        KC = K // BK                       # chunks in K
        cps = KC // split_k                # chunks per split
        c0 = bk * cps
        n8g0 = bx * (block_N // 8) + warp * n8_per_warp   # first global n8 tile of this warp

        T.clear(C_l)

        # register prefetch of the first stages-1 chunks
        for s in T.unroll(stages - 1):
            if s < cps:
                for t in T.unroll(n8_per_warp):
                    n8 = n8g0 + t
                    tb = ((n8 // 16) * KC + (c0 + s)) * 16 + (n8 % 16)
                    for v in T.vectorized(4):
                        W_l[s, t, v] = Wp[tb * 32 + lane, v]
                    for v in T.vectorized(2):
                        S_l[s, t, v] = Sp[tb * 8 + lane // 4, v]
        # A chunk 0 -> shared stage 0. Rows >= M: address clamped to row M-1 (always valid), data zero-filled (no read).
        for p in T.unroll(a_pieces):
            idx = p * threads + tx
            T.ptx_cp_async(T.access_ptr(A_s[idx // 16, (idx % 16) * 8], "w", extent=8),
                           T.access_ptr(x[T.min(idx // 16, M - 1), c0 * BK + (idx % 16) * 8], "r", extent=8), 8,
                           predicate=idx // 16 < M)
        T.ptx_commit_group()

        for cc in T.serial(T.ceildiv(cps, stages)):
            for s in T.unroll(stages):
                c = cc * stages + s
                if c < cps:
                    # 1. keep the weight stream going: chunk c+stages-1 into the register stage consumed last iteration
                    if c + (stages - 1) < cps:
                        for t in T.unroll(n8_per_warp):
                            n8 = n8g0 + t
                            tb = ((n8 // 16) * KC + (c0 + c + stages - 1)) * 16 + (n8 % 16)
                            for v in T.vectorized(4):
                                W_l[(s + stages - 1) % stages, t, v] = Wp[tb * 32 + lane, v]
                            for v in T.vectorized(2):
                                S_l[(s + stages - 1) % stages, t, v] = Sp[tb * 8 + lane // 4, v]
                    # 2. A chunk c has landed for this thread; the barrier makes every thread's piece visible and
                    #    guarantees all warps finished reading the other stage (chunk c-1) before it is overwritten
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    if c + 1 < cps:
                        for p in T.unroll(a_pieces):
                            idx = p * threads + tx
                            T.ptx_cp_async(
                                T.access_ptr(A_s[((c + 1) % 2) * block_M + idx // 16, (idx % 16) * 8], "w", extent=8),
                                T.access_ptr(x[T.min(idx // 16, M - 1), (c0 + c + 1) * BK + (idx % 16) * 8], "r",
                                             extent=8), 8, predicate=idx // 16 < M)
                    T.ptx_commit_group()
                    # 3. compute chunk c: 4 words = 8 k16 steps
                    # (the A region is built inline: assigning a BufferRegion to a name would let-bind it to a Var)
                    for w in T.unroll(4):
                        for t in T.unroll(n8_per_warp):
                            T.call_extern("handle", "tl_k2_dq_word", W_l[s, t, w], S_l[s, t, w // 2], (w % 2) * 2,
                                          T.access_ptr(B0_l[t * 4], "w", extent=4),
                                          T.access_ptr(B1_l[t * 4], "w", extent=4))
                        emitter.ldmatrix_a(A_l, BufferRegion(A_s, [Range.from_min_extent((c % 2) * block_M, block_M),
                                                                   Range.from_min_extent(0, BK)]), 2 * w)
                        emitter.mma(A_l, B0_l, C_l)
                        emitter.ldmatrix_a(A_l, BufferRegion(A_s, [Range.from_min_extent((c % 2) * block_M, block_M),
                                                                   Range.from_min_extent(0, BK)]), 2 * w + 1)
                        emitter.mma(A_l, B1_l, C_l)

        # epilogue: mma C fragment layout -> (row, col); rows >= M are dropped. Fragment element l = q*2 + e holds
        # row i*16 + 8*(q%2) + lane/4, column (n8 half q//2) + (lane%4)*2 + e: the e pair is stored as one vector.
        if split_k == 1:
            for i in T.unroll(warp_rows):
                for j in T.unroll(warp_cols):
                    for q in T.unroll(4):
                        row = i * 16 + 8 * (q % 2) + lane // 4
                        n = bx * block_N + warp * n_per_warp + j * 16 + 8 * (q // 2) + (lane % 4) * 2
                        if row < M:
                            for e in T.vectorized(2):
                                Y[row, n + e] = T.cast(C_l[i * warp_cols * 8 + j * 8 + q * 2 + e] * alpha[n + e],
                                                       T.bfloat16)
        else:
            for i in T.unroll(warp_rows):
                for j in T.unroll(warp_cols):
                    for q in T.unroll(4):
                        row = i * 16 + 8 * (q % 2) + lane // 4
                        n = bx * block_N + warp * n_per_warp + j * 16 + 8 * (q // 2) + (lane % 4) * 2
                        if row < M:
                            for e in T.vectorized(2):
                                P[bk * M + row, n + e] = C_l[i * warp_cols * 8 + j * 8 + q * 2 + e]
            # publish the partial tile, then take a ticket; the last CTA of this column block reduces in fixed order
            T.call_extern("handle", "__threadfence")
            T.sync_threads()
            if tx == 0:
                flag[0] = T.call_extern("int32", "tl_k2_ticket", T.address_of(cnt[bx]), split_k)
            T.sync_threads()
            if flag[0] != 0:
                T.call_extern("handle", f"tl_k2_reduce<{split_k}, {block_M}, {block_N}, {threads}>",
                              T.address_of(P[0, 0]), T.address_of(Y[0, 0]), T.address_of(alpha[0]), M, N,
                              bx * block_N)


# ------------------------------------------------------------------------------------------------------------------
# host side
# ------------------------------------------------------------------------------------------------------------------
@dataclass
class Prepared:
    Wp: object       # uint32 (N*K/32, 4)
    Sp: object       # uint32 (N*K/128, 2)
    alpha: object    # fp32 (N,) = alpha * 128
    cnt: object      # int32 (N/16,) split-K ticket counters (zero between launches)
    dummy: object    # fp32 (1, 1) stand-in workspace for split_k == 1
    N: int
    K: int


def _to_int32(v):
    """int64 tensor with values in [0, 2^32) -> int32 with the same bits (signed wrap, robust across torch versions)."""
    import torch
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def _shuffle_rows(packed_rows, scale_rows, KC):
    """Kernel layout for a block of rows (a multiple of 128): returns (words int64 (R*K/32, 4), scale words (R*K/128, 2))."""
    import torch
    u = packed_rows.view(torch.uint8)
    R = u.shape[0]
    K = u.shape[1] * 2
    nib = torch.stack([u & 0xF, u >> 4], dim=-1).reshape(R, K).to(torch.int64)      # nib[n, k]
    sc = scale_rows.float()                                                          # (R, K/16)
    zero_blk = sc == 0
    if bool(zero_blk.any()):
        # a block with scale 0 is identically 0: zero its codes so the substitute scale code cannot leak anything
        nib = nib.masked_fill(zero_blk.repeat_interleave(16, dim=1), 0)
    # multiplier bf16(scale * 2^119): exact (<= 4 significant bits), exponent 237..254 -> bits = 0x7000 | code << 4
    mult = (sc * (2.0 ** 119)).to(torch.bfloat16)
    bits = mult.view(torch.int16).to(torch.int64) & 0xFFFF
    ok = ((bits & 0xF00F) == 0x7000) | zero_blk
    assert bool(ok.all()), "unexpected ue4m3 scale (nan?)"
    scode = ((bits >> 4) & 0xFF).masked_fill(zero_blk, 0)                           # (R, K/16) 8-bit codes

    nb = R // 128
    # n = nb*128 + tt*8 + r ; k = kc*128 + w*32 + sw*16 + hi*8 + q*2 + e   (lane = r*4 + q, slot j = sw*4 + hi*2 + e)
    t = nib.reshape(nb, 16, 8, KC, 4, 2, 2, 4, 2).permute(0, 3, 1, 2, 7, 4, 5, 6, 8)
    t = t.reshape(nb * KC * 16 * 32, 4, 8)
    word = torch.zeros(t.shape[0], 4, dtype=torch.int64, device=t.device)
    for j, (ps, pe1, pe0, pm) in enumerate(_NIB_POS):
        v = t[:, :, j]
        word |= (((v >> 3) & 1) << ps) | (((v >> 2) & 1) << pe1) | (((v >> 1) & 1) << pe0) | ((v & 1) << pm)

    s = scode.reshape(nb, 16, 8, KC, 8).permute(0, 3, 1, 2, 4).reshape(nb * KC * 16 * 8, 2, 4)
    sw = s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)
    return word, sw


def prepare(packed_int8, scale_fp8, alpha_fp32) -> Prepared:
    """Checkpoint layout -> shuffled kernel layout (untimed, done once at load time).

    packed_int8 int8 (N, K/2) two e2m1 codes per byte (low nibble = even k); scale_fp8 e4m3 (N, K/16); alpha fp32 (N,).
    Processed in row blocks so the int64 temporaries stay small (~100 MB) even for the 34816-row gate_up.
    """
    import torch
    N, K2 = packed_int8.shape
    K = K2 * 2
    assert N % 128 == 0 and K % 256 == 0, (N, K)
    dev = packed_int8.device
    KC = K // BK
    Wp = torch.empty(N * K // 32, 4, dtype=torch.int32, device=dev)      # filled as int32, viewed as uint32 below
    Sp = torch.empty(N * K // 128, 2, dtype=torch.int32, device=dev)
    rows = 128 * max(1, (1 << 22) // (128 * K))          # ~4M weights per block
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        word, sw = _shuffle_rows(packed_int8[r0:r1], scale_fp8[r0:r1], KC)
        Wp[r0 * K // 32:r1 * K // 32] = _to_int32(word)
        Sp[r0 * K // 128:r1 * K // 128] = _to_int32(sw)
    alpha2 = (alpha_fp32.float() * 128.0).contiguous()
    cnt = torch.zeros(N // 16, dtype=torch.int32, device=dev)
    dummy = torch.zeros(1, 1, dtype=torch.float32, device=dev)
    return Prepared(Wp.view(torch.uint32), Sp.view(torch.uint32), alpha2, cnt, dummy, N, K)


# tunables (override for experiments): block_M is bucketed from M, the rest come from here
TUNE = {"n_warps": 4, "n_per_warp": 32, "stages": 3, "split_k": None, "max_split": 16}
_CFG_CACHE = {}

# split-K model constants (SM120 RTX PRO 6000; only used to rank split_k choices, never measured here)
_DRAM_BW = 1.63e12                 # B/s, measured DRAM read bandwidth
_TC_PER_SM = 285e12 / NUM_SMS      # BF16 dense tensor FLOP/s per SM
_BYTES_IN_FLIGHT = 4.0e6           # DRAM saturation: ~BW x 2.5 us loaded latency (conservative)
_RESIDENT = {16: 3, 32: 3, 48: 2, 64: 2}   # CTAs/SM by registers (136 / 168 / 194 / 228 regs x 128 threads)


def _split_time(s, M, N, K, block_M, block_N, n_warps, n_per_warp, stages):
    """Rough kernel time for split_k = s: max(DRAM stream, busiest SM's MMA work) + split-K tail."""
    ctas = (N // block_N) * s
    resident = min(ctas, _RESIDENT[block_M] * NUM_SMS)
    inflight = resident * (stages - 1) * n_warps * (n_per_warp // 8) * 576        # register-staged bytes
    bw = _DRAM_BW * min(1.0, inflight / _BYTES_IN_FLIGHT)
    part = 0.0 if s == 1 else s * M * N * 4.0                                        # fp32 partials (write + read)
    t_mem = (N * K * 0.5625 + part) / bw
    t_mma = math.ceil(ctas / NUM_SMS) * 2.0 * block_M * block_N * (K // s) / _TC_PER_SM
    t_red = 0.0 if s == 1 else 1e-6 + s * M * block_N * 4.0 / 100e9                 # last CTA's fixed-order sum
    return max(t_mem, t_mma) + t_red


def _pick_config(M, N, K):
    key = (M, N, K, tuple(sorted(TUNE.items())))
    cfg = _CFG_CACHE.get(key)
    if cfg is not None:
        return cfg
    block_M = 16 if M <= 16 else (32 if M <= 32 else (48 if M <= 48 else 64))
    n_warps, n_per_warp, stages = TUNE["n_warps"], TUNE["n_per_warp"], TUNE["stages"]
    block_N = n_warps * n_per_warp
    assert N % block_N == 0, (N, block_N)
    KC = K // BK
    split_k = TUNE["split_k"]
    if split_k is None:
        # deterministic split-K: s divides the chunk count. Small M is DRAM-bound (s only has to keep enough bytes in
        # flight); larger M is MMA-bound per SM, so s also balances the CTAs over the 188 SMs.
        cands = [s for s in range(1, TUNE["max_split"] + 1) if KC % s == 0]
        split_k = min(cands, key=lambda s: (_split_time(s, M, N, K, block_M, block_N, n_warps, n_per_warp, stages),
                                            s))
    assert KC % split_k == 0, (K, split_k)
    cfg = dict(block_M=block_M, n_warps=n_warps, n_per_warp=n_per_warp, stages=stages, split_k=split_k)
    _CFG_CACHE[key] = cfg
    return cfg


def linear(x, prep: Prepared, **override):
    """y (M, N) bf16 = x (M, K) bf16 @ Wdq^T, 1 <= M <= 64. Timed path; CUDA-graph capturable (one kernel launch)."""
    import torch
    M, K = x.shape
    assert K == prep.K and 1 <= M <= SUPPORTS_M, (x.shape, prep.K)
    N = prep.N
    cfg = dict(_pick_config(M, N, K))
    cfg.update(override)
    split_k = cfg["split_k"]
    assert (K // BK) % split_k == 0, (K, split_k)
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    P = prep.dummy if split_k == 1 else torch.empty(split_k * M, N, dtype=torch.float32, device=x.device)
    nvfp4_shuffled_gemm_kernel(x, prep.Wp, prep.Sp, prep.alpha, y, P, prep.cnt, block_M=cfg["block_M"],
                               n_warps=cfg["n_warps"], n_per_warp=cfg["n_per_warp"], stages=cfg["stages"],
                               split_k=split_k)
    return y
