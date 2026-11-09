# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 candidate "twiddle": SM120 NVFP4 W4A16 decode GEMM, register dequantisation with the LOP3/PRMT bit-trick.

Contract: see decode_contract.py (prepare / linear / SUPPORTS_M).

Why this streams weights faster than v0/v1
------------------------------------------
v0/v1 dequantised every FP4 into a BF16 *shared-memory* tile (scalar cvt chain, ~10 instructions per weight, one
16-bit smem write + one ldmatrix read per weight) and only then ran T.gemm. Here the 4-bit words go
global -> shared (cp.async, multi-stage T.Pipelined, 16 B per thread) -> registers (16 B ld.shared per thread per
n16 x k64) and are converted *in registers* straight into the mma.sync B fragment:

* prepare() permutes the nibbles ONCE into a "twiddled, fragment-major" layout: one 32-bit word holds exactly
  the 8 FP4 codes that one lane needs for one m16n8k16 x2 (n16 x k16) B fragment, in the bit order that the
  Triton/TileLang `decode_fp4_to_bf16_twiddling` trick expects (16-bit half h holds nibbles v0..v3 as
  s0@15 e0@8:7 m0@6 | s1@12 e1@5:4 m1@3 | s2@9 e2@2:1 m2@0 | s3@14 e3@11:10 m3@13). The two 16-bit halves are
  the even-k / odd-k values, so the four `and/shl/mul.bf16x2` results ARE the four mma B registers: no prmt,
  no repack. The layout is k64-granular, so any block_K that is a multiple of 64 streams the same tensors.
* The UE4M3 block scale is folded into the trick's final `mul.bf16x2`: prepare() re-encodes each scale byte
  as biased E5M3 so that `0x7000 | byte << 4` is exactly the BF16 for scale * 2^SCALE_C (SCALE_C = 119, the
  largest value keeping the exponent field <= 254 for scale = 448; normal AND subnormal UE4M3 scales, since
  the exponent is re-normalised on the host). In the kernel that BF16x2 word is ONE prmt (scale byte plus the
  constant 0x07 byte from a register) and ONE shift. Multiplying the raw twiddled word (whose BF16 exponent
  field holds the 2-bit FP4 exponent, i.e. value fp4 * 2^-126) by it gives fp4 * scale * 2^-7 exactly in BF16
  (<= 6 significant bits). The 2^7 is folded into the fp32 alpha of the epilogue, so the numerics are
  bit-for-bit the "exact class" (bf16 dequant, fp32 accumulate, one final rounding).
* Cost per 8 weights (static SASS of the compiled main loop, sm_120a): 19 ops = 11 LOP3/SHF/IMAD.SHL for the
  FP4 bit placement + 2 PRMT + 2 IMAD.SHL for the two scale words + 4 HMUL2.BF16_V2 (2.375 per weight). The
  whole steady-state loop is 415 SASS per 128x128 k-tile at block_M = 16 (3.24 per weight incl. 32 HMMA,
  8 LDSM, 6 LDS.128, 7 LDGSTS, 2 BAR). The hardware path (cvt.rn.f16x2.e2m1x2 = F2FP.F16.E2M1.UNPACK_B,
  which is what TileLang's fp4 casts lower to) costs 20 ops to scaled *f16x2* but 32 ops to scaled *bf16x2*,
  because SM120 has neither an e2m1x2 -> bf16x2 nor an f16x2 -> bf16x2 cvt (8 HADD2.F32 + 4 F2FP.BF16 extra).
  So for this BF16 GEMM the twiddle is ~40% cheaper; for an F16 MMA the two would be on par.

Tiles: block_N = 128 (4 warps x 32 columns, every weight dequantised exactly once), block_M in {16, 32, 64}
by M bucket, block_K / stages per bucket (_CFG) sized for 2 CTAs per SM (<= ~50 KB smem each). Split-K spreads
narrow layers over the SMs; each split writes an fp32 partial in the mma fragment layout and the LAST-arriving CTA
of an n-tile (int32 ticket, threadfence-reduction pattern, no fp32 atomics) sums the partials in fixed order
0..split_k-1, so results are bit-identical run to run and CUDA-graph replayable (the ticket resets itself).
Rows of x at or beyond M are never addressed: the A-tile load clamps the row index to M - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout

SUPPORTS_M = 64

BLOCK_N = 128
THREADS = 128
WARPS = THREADS // 32               # 4 warps, all along N (each weight is dequantised exactly once)
WARP_N = BLOCK_N // WARPS           # 32 output columns per warp
WARP_COLS = WARP_N // 16            # 2 n16 fragments per warp
GROUP_WORDS = BLOCK_N * 64 // 8     # 1024 int32 words of 8 FP4 codes per (n128 x k64) group
SGROUP_WORDS = BLOCK_N * 64 // 64   # 128 int32 words of 4 scale bytes per (n128 x k64) group
SCALE_C = 119                           # kernel weights are fp4 * scale * 2^(SCALE_C - 126); 119 keeps the bf16
                                        # exponent field of scale*2^SCALE_C <= 254 for scale <= 448 (E5M3 field <= 30)
ALPHA_COMP = 2.0 ** (126 - SCALE_C)     # folded into alpha (fp32, exact)
NUM_SMS = 188
SMEM_PER_SM = 100 * 1024                # SM120: 100 KB shared memory per SM, 99 KB per block, 1 KB reserved per CTA

# block_M -> (block_K, num_stages): 2 CTAs per SM in every bucket (see _smem_bytes), (stages - 1) tiles in flight
_CFG = {16: (128, 3), 32: (64, 4), 64: (64, 3)}

# split_k per (N, K, block_M); anything missing uses _pick_split()
SPLIT_K_OVERRIDE: dict[tuple[int, int, int], int] = {}

# ---------------------------------------------------------------------------------------------------------------
# Device code: the twiddling dequant (algorithm of examples/dequantize_gemm/quantize/mxfp.py, no prmt because the
# word is stored pre-swapped, and the bias multiply replaced by the folded block-scale multiply).
TWIDDLE_SRC = r"""
// w : one twiddled 32-bit word = the 8 FP4 codes of this lane's n16 x k16 B fragment (layout made by prepare()).
// sw: 4 biased-E5M3 scale bytes = (n=g,kb) (n=g+8,kb) (n=g,kb+1) (n=g+8,kb+1); sel = 0x4040 (kb) or 0x4242 (kb+1).
// out[0..3] (bf16x2) = mma.m16n8k16 B registers: [0],[1] -> column g (k 2t..2t+1 | 2t+8..2t+9), [2],[3] -> column g+8.
__device__ __forceinline__ void tl_twiddle_dequant(unsigned w, unsigned sw, unsigned sel, void *out) {
  // prmt puts the scale byte in bits 7:0 and the constant 0x07 in bits 15:8 of both halves (0x07bb07bb);
  // << 4 turns each half into 0x7bb0 = 0x7000 | byte << 4 = bf16(scale * 2^SCALE_C).
  const unsigned s0 = __byte_perm(sw, 0x07070707u, sel) << 4;            // column g   in both halves
  const unsigned s1 = __byte_perm(sw, 0x07070707u, sel + 0x0101u) << 4;  // column g+8 in both halves
  unsigned r0, r1, r2, r3;
  asm("{\n\t"
      ".reg .b32 t1, t2;\n\t"
      "and.b32 %0, %4, 0x81C081C0;\n\t"
      "shl.b32 %1, %4, 3;\n\t"
      "and.b32 %1, %1, 0x81C081C0;\n\t"
      "shl.b32 %2, %4, 6;\n\t"
      "and.b32 %2, %2, 0x81C081C0;\n\t"
      "shl.b32 t1, %4, 1;\n\t"
      "and.b32 %3, t1, 0x80008000;\n\t"
      "shr.b32 t2, %4, 3;\n\t"
      "and.b32 t2, t2, 0x01800180;\n\t"
      "or.b32 %3, %3, t2;\n\t"
      "shr.b32 t2, %4, 7;\n\t"
      "and.b32 t2, t2, 0x00400040;\n\t"
      "or.b32 %3, %3, t2;\n\t"
      "mul.rn.bf16x2 %0, %0, %5;\n\t"
      "mul.rn.bf16x2 %1, %1, %5;\n\t"
      "mul.rn.bf16x2 %2, %2, %6;\n\t"
      "mul.rn.bf16x2 %3, %3, %6;\n\t"
      "}"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(w), "r"(s0), "r"(s1));
  unsigned *o = reinterpret_cast<unsigned *>(out);
  o[0] = r0;
  o[1] = r1;
  o[2] = r2;
  o[3] = r3;
}

// split-K reduction step: acc[0..1] = P[p..p+1] (add == 0) or acc[0..1] += P[p..p+1], read through L2 (__ldcg:
// the partials were written by other CTAs, so a stale L1 line must not be used).
__device__ __forceinline__ void tl_ldcg_acc2(float *acc, const float *p, int add) {
  const float2 v = __ldcg(reinterpret_cast<const float2 *>(p));
  if (add) {
    acc[0] += v.x;
    acc[1] += v.y;
  } else {
    acc[0] = v.x;
    acc[1] = v.y;
  }
}
"""


# ---------------------------------------------------------------------------------------------------------------
# Host-side re-layout (prepare)

def _e5m3_scale_bytes(scale_abs: torch.Tensor) -> torch.Tensor:
    """|UE4M3| block scales (fp32) -> bytes b with bf16_bits(0x7000 | b << 4) == scale * 2^SCALE_C exactly.

    Normal and subnormal UE4M3 values both have <= 4 significant bits, so frexp re-normalises them exactly into
    a 5-bit exponent + 3-bit mantissa. Zero scales get byte 0 (their weights are zeroed in twiddle_weights)."""
    sf = scale_abs
    mant, exp = torch.frexp(sf)                          # sf = mant * 2^exp, mant in [0.5, 1)
    m3 = torch.round((mant * 2.0 - 1.0) * 8.0).to(torch.int32)   # 3 significant mantissa bits, exact
    ep = (exp - 1).to(torch.int32) + (SCALE_C - 97)      # exponent field = 224 + ep  ->  2^(ep + 97) * (1 + m3/8)
    pos = sf > 0
    if bool(pos.any()):
        lo, hi = int(ep[pos].min()), int(ep[pos].max())
        assert 0 <= lo and hi <= 30, f"block scale out of range for the E5M3 trick: exponent field {lo}..{hi}"
        assert bool(torch.isfinite(sf).all()), "non-finite block scale"
    b = torch.where(pos, (ep << 3) | m3, torch.zeros_like(ep))
    return b.to(torch.uint8)


def _twiddle16(v: torch.Tensor) -> torch.Tensor:
    """v (..., 4) int32 FP4 codes -> (...) int32 16-bit twiddled half-word (bit order of the PTX trick)."""
    s = (v >> 3) & 1
    e = (v >> 1) & 3
    m = v & 1
    w0 = (s[..., 0] << 15) | (e[..., 0] << 7) | (m[..., 0] << 6)
    w1 = (s[..., 1] << 12) | (e[..., 1] << 4) | (m[..., 1] << 3)
    w2 = (s[..., 2] << 9) | (e[..., 2] << 1) | m[..., 2]
    w3 = (s[..., 3] << 14) | (e[..., 3] << 10) | (m[..., 3] << 13)
    return w0 | w1 | w2 | w3


_PREP_ROWS = 2048       # rows per prepare() chunk (bounds the int32 temporaries to ~0.2 GB for K = 5120)


def twiddle_weights(packed_int8: torch.Tensor, zero_blocks: torch.Tensor | None = None,
                    neg_blocks: torch.Tensor | None = None) -> torch.Tensor:
    """(N, K/2) packed FP4 (low nibble = even k) -> group-major twiddled int32 words (N*K/8,).

    Word index = (((bn * KG + kg) * WARP_COLS + j) * THREADS + tx) * 4 + s  (KG = K/64 k64 groups), lane
    (g = lane//4, t = lane%4) of warp w = tx//32 holding, for n16 fragment j and k16 step s of k64 group kg:
      half r (0 = even k, 1 = odd k) = twiddle16 of [(n=g, k=2t+r), (n=g, k=2t+8+r), (n=g+8, k=2t+r), (n=g+8, k=2t+8+r)]
    with n relative to bn*128 + w*32 + j*16 and k relative to kg*64 + s*16. Blocks with a zero scale get zero codes,
    blocks with a negative scale get their sign bits flipped (the scale is then encoded as |scale|).
    """
    N, K2 = packed_int8.shape
    K = K2 * 2
    assert N % BLOCK_N == 0 and K % 64 == 0, (N, K)
    if N > _PREP_ROWS:      # group-major layout: row chunks (multiples of BLOCK_N) are independent and contiguous
        parts = []
        for r0 in range(0, N, _PREP_ROWS):
            zb = None if zero_blocks is None else zero_blocks[r0:r0 + _PREP_ROWS]
            nb = None if neg_blocks is None else neg_blocks[r0:r0 + _PREP_ROWS]
            parts.append(twiddle_weights(packed_int8[r0:r0 + _PREP_ROWS], zb, nb))
        return torch.cat(parts)
    u = packed_int8.view(torch.uint8)
    nib = torch.empty(N, K, dtype=torch.uint8, device=u.device)
    nib[:, 0::2] = u & 0xF
    nib[:, 1::2] = u >> 4
    blk = nib.view(N, K // 16, 16)
    if neg_blocks is not None and bool(neg_blocks.any()):
        blk[neg_blocks] ^= 8
    if zero_blocks is not None and bool(zero_blocks.any()):
        blk[zero_blocks] = 0
    n_tiles, KG = N // BLOCK_N, K // 64
    # n = bn*128 + w*32 + j*16 + h*8 + g ;  k = kg*64 + s*16 + p*8 + t*2 + r
    v = nib.view(n_tiles, WARPS, WARP_COLS, 2, 8, KG, 4, 2, 4, 2)
    #            0        1      2          3  4  5   6  7  8  9
    v = v.permute(0, 5, 2, 1, 4, 8, 6, 9, 3, 7).contiguous()      # (bn, kg, j, w, g, t, s, r, h, p)
    v = v.view(n_tiles, KG, WARP_COLS, THREADS, 4, 2, 4).to(torch.int32)   # (.., tx, s, r, v = h*2 + p)
    h16 = _twiddle16(v).to(torch.int64)                              # (bn, kg, j, tx, s, r)
    word = h16[..., 0] | (h16[..., 1] << 16)
    word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return word.reshape(-1).contiguous()


def twiddle_scales(scale_fp8: torch.Tensor) -> torch.Tensor:
    """(N, K/16) UE4M3 -> group-major biased-E5M3 bytes packed 4 per int32 (N*K/64,).

    Byte index in the (bn, kg) group = (((w * 8 + g) * WARP_COLS + j) * 4 + kb) * 2 + h  (kb = k16 step in the k64
    group, h: n = g / g+8), so one 16-byte load gives a lane both n16 fragments' scales for the whole k64 group.
    """
    N, KB = scale_fp8.shape
    K = KB * 16
    if N > _PREP_ROWS:
        return torch.cat([twiddle_scales(scale_fp8[r0:r0 + _PREP_ROWS]) for r0 in range(0, N, _PREP_ROWS)])
    n_tiles, KG = N // BLOCK_N, K // 64
    b = _e5m3_scale_bytes(scale_fp8.float().abs())
    sv = b.view(n_tiles, WARPS, WARP_COLS, 2, 8, KG, 4)
    #           0        1      2          3  4  5   6
    sv = sv.permute(0, 5, 1, 4, 2, 6, 3).contiguous()           # (bn, kg, w, g, j, kb, h)
    return sv.reshape(-1).view(torch.int32).contiguous()


@dataclass
class TwiddledWeight:
    Wq: torch.Tensor        # int32 (N*K/8,)   twiddled fragment-major FP4 words
    Sq: torch.Tensor        # int32 (N*K/64,)  biased-E5M3 scale bytes
    alpha: torch.Tensor     # fp32 (N,)        alpha * 2^(126 - SCALE_C)
    counters: torch.Tensor  # int32 (N/128,)   split-K arrival tickets (self-resetting)
    N: int
    K: int


def prepare(packed_int8: torch.Tensor, scale_fp8: torch.Tensor, alpha_fp32: torch.Tensor) -> TwiddledWeight:
    N, K = packed_int8.shape[0], packed_int8.shape[1] * 2
    assert scale_fp8.shape == (N, K // 16), (scale_fp8.shape, N, K)
    assert N % BLOCK_N == 0 and K % 64 == 0, (N, K)
    sf = scale_fp8.float()
    zero_blocks = sf == 0            # a zero block scale zeroes its 16 weights (scale byte is then arbitrary)
    neg_blocks = sf < 0              # UE4M3 should be >= 0; a negative one is folded into the FP4 sign bits
    Wq = twiddle_weights(packed_int8, zero_blocks, neg_blocks)
    Sq = twiddle_scales(scale_fp8)
    alpha = (alpha_fp32.float() * ALPHA_COMP).contiguous()
    counters = torch.zeros(N // BLOCK_N, dtype=torch.int32, device=packed_int8.device)
    return TwiddledWeight(Wq, Sq, alpha, counters, N, K)


# ---------------------------------------------------------------------------------------------------------------
# Kernels

def _make_emitter(block_M: int, block_K: int) -> TensorCoreIntrinEmitter:
    return TensorCoreIntrinEmitter(
        a_dtype=T.bfloat16, b_dtype=T.bfloat16, accum_dtype=T.float32,
        a_transposed=False, b_transposed=True,
        block_row_warps=1, block_col_warps=WARPS,
        warp_row_tiles=block_M, warp_col_tiles=WARP_N, chunk=block_K,
    )


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def twiddle_gemm_kernel(x, Wq, Sq, alpha, y, P, counter, block_M: int = 16, block_K: int = 128, split_k: int = 1,
                        num_stages: int = 3):
    M = T.dynamic("M")
    N = T.const("N")
    K = T.const("K")
    x: T.Tensor[(M, K), T.bfloat16]
    Wq: T.Tensor[(N * K // 8,), T.int32]
    Sq: T.Tensor[(N * K // 64,), T.int32]
    alpha: T.Tensor[(N,), T.float32]
    y: T.Tensor[(M, N), T.bfloat16]
    P: T.Tensor[(split_k, M, N) if split_k > 1 else (1, 1, 1), T.float32]   # split-K fp32 partials (dummy if 1)
    counter: T.Tensor[(N // BLOCK_N,), T.int32]

    KQ = block_K // 64                        # k64 groups per pipeline tile
    KG = K // 64                               # k64 groups in K
    n_tiles = N // BLOCK_N
    k_tiles = K // block_K
    kt_per_split = k_tiles // split_k          # linear() guarantees k_tiles % split_k == 0
    warp_rows = block_M // 16
    mma_emitter = _make_emitter(block_M, block_K)

    with T.Kernel(n_tiles, split_k, threads=THREADS) as (bx, bk):
        A_s = T.alloc_shared((block_M, block_K), T.bfloat16)
        W_s = T.alloc_shared((KQ * GROUP_WORDS,), T.int32)
        S_s = T.alloc_shared((KQ * SGROUP_WORDS,), T.int32)
        flag_s = T.alloc_shared((1,), T.int32)
        A_local = T.alloc_local((warp_rows * 8,), T.bfloat16)
        B_local = T.alloc_local((WARP_COLS * 8,), T.bfloat16)
        C_local = T.alloc_local((warp_rows * WARP_COLS * 8,), T.float32)
        W_local = T.alloc_local((WARP_COLS * 4,), T.int32)
        S_local = T.alloc_local((4,), T.int32)
        T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})
        T.import_source(TWIDDLE_SRC)

        tx = T.get_thread_binding()
        warp = tx // 32
        lane = tx % 32
        g = lane // 4
        t4 = lane % 4

        T.clear(C_local)
        for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
            kg = bk * kt_per_split + kt                 # k tile index in K
            grp = bx * KG + kg * KQ                     # first (n128 x k64) group of this tile
            # A tile: rows >= M re-read row M-1 (valid memory, L2 hits), so no address past the end of x is ever
            # formed; those accumulator rows are never stored. (A T.if_then_else zero-fill here is NOT lowered to
            # cp.async: it becomes a synchronous ld.global + st.shared and un-pipelines A, so the clamp is used.)
            for i, kk in T.Parallel(block_M, block_K):
                A_s[i, kk] = x[T.min(i, M - 1), kg * block_K + kk]
            T.copy(Wq[grp * GROUP_WORDS], W_s)
            T.copy(Sq[grp * SGROUP_WORDS], S_s)
            # one consumer statement for all three stage buffers -> one wait + one barrier pair per tile
            for q in T.unroll(KQ):
                for v in T.vectorized(4):
                    S_local[v] = S_s[q * SGROUP_WORDS + (warp * 8 + g) * 4 + v]
                for j in T.unroll(WARP_COLS):
                    for v in T.vectorized(4):
                        W_local[j * 4 + v] = W_s[q * GROUP_WORDS + (j * THREADS + tx) * 4 + v]
                for s in T.unroll(4):
                    mma_emitter.ldmatrix_a(A_local, A_s, q * 4 + s)
                    for j in T.unroll(WARP_COLS):
                        T.call_extern(
                            "handle", "tl_twiddle_dequant",
                            W_local[j * 4 + s],
                            S_local[j * 2 + s // 2],
                            0x4040 + (s % 2) * 0x0202,
                            T.access_ptr(B_local[j * 8], "w", 8),
                        )
                    mma_emitter.mma(A_local, B_local, C_local)

        # epilogue: mma m16n8 C fragment -> (row g / g+8, cols 2t..2t+1 of each n8 half of the n16 tile)
        if split_k == 1:
            for i in T.unroll(warp_rows):
                for j in T.unroll(WARP_COLS):
                    for l2 in T.unroll(4):
                        m = i * 16 + g + 8 * (l2 % 2)
                        n = bx * BLOCK_N + warp * WARP_N + j * 16 + 8 * (l2 // 2) + t4 * 2
                        ci = (i * WARP_COLS + j) * 8 + l2 * 2
                        if m < M:
                            for v in T.vectorized(2):
                                y[m, n + v] = T.cast(C_local[ci + v] * alpha[n + v], T.bfloat16)
        else:
            for i in T.unroll(warp_rows):
                for j in T.unroll(WARP_COLS):
                    for l2 in T.unroll(4):
                        m = i * 16 + g + 8 * (l2 % 2)
                        n = bx * BLOCK_N + warp * WARP_N + j * 16 + 8 * (l2 // 2) + t4 * 2
                        ci = (i * WARP_COLS + j) * 8 + l2 * 2
                        if m < M:
                            for v in T.vectorized(2):
                                P[bk, m, n + v] = C_local[ci + v]
            # threadfence reduction: the last CTA of this n-tile sums the partials in fixed order 0..split_k-1
            T.call_extern("handle", "__threadfence")
            T.sync_threads()
            if tx == 0:
                flag_s[0] = T.atomic_add(counter[bx], 1, return_prev=True)
            T.sync_threads()
            if flag_s[0] == split_k - 1:
                T.call_extern("handle", "__threadfence")
                # fixed order 0..split_k-1 (C_local is reused as the accumulator; this CTA's own partial is re-read
                # from P like the others). Not unrolled over splits: each step has warp_rows*8 independent float2
                # loads in flight, and the register count stays that of the main loop.
                for sk in T.unroll(split_k, unroll_factor=1):
                    for i in T.unroll(warp_rows):
                        for j in T.unroll(WARP_COLS):
                            for l2 in T.unroll(4):
                                m = i * 16 + g + 8 * (l2 % 2)
                                n = bx * BLOCK_N + warp * WARP_N + j * 16 + 8 * (l2 // 2) + t4 * 2
                                ci = (i * WARP_COLS + j) * 8 + l2 * 2
                                if m < M:
                                    T.call_extern("handle", "tl_ldcg_acc2", T.access_ptr(C_local[ci], "rw", 2),
                                                  T.access_ptr(P[sk, m, n], "r", 2), sk)
                for i in T.unroll(warp_rows):
                    for j in T.unroll(WARP_COLS):
                        for l2 in T.unroll(4):
                            m = i * 16 + g + 8 * (l2 % 2)
                            n = bx * BLOCK_N + warp * WARP_N + j * 16 + 8 * (l2 // 2) + t4 * 2
                            ci = (i * WARP_COLS + j) * 8 + l2 * 2
                            if m < M:
                                for v in T.vectorized(2):
                                    y[m, n + v] = T.cast(C_local[ci + v] * alpha[n + v], T.bfloat16)
                if tx == 0:
                    counter[bx] = 0


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def twiddle_debug_kernel(Wq, Sq, Wd):
    """Validation only: run the same dequant on every (n128 x k64) group and scatter the bf16 weights to (N, K)."""
    N = T.const("N")
    K = T.const("K")
    Wq: T.Tensor[(N * K // 8,), T.int32]
    Sq: T.Tensor[(N * K // 64,), T.int32]
    Wd: T.Tensor[(N, K), T.bfloat16]
    n_tiles = N // BLOCK_N
    KG = K // 64

    with T.Kernel(n_tiles, KG, threads=THREADS) as (bx, bq):
        W_s = T.alloc_shared((GROUP_WORDS,), T.int32)
        S_s = T.alloc_shared((SGROUP_WORDS,), T.int32)
        B_local = T.alloc_local((WARP_COLS * 8,), T.bfloat16)
        W_local = T.alloc_local((WARP_COLS * 4,), T.int32)
        S_local = T.alloc_local((4,), T.int32)
        T.import_source(TWIDDLE_SRC)
        tx = T.get_thread_binding()
        warp = tx // 32
        lane = tx % 32
        g = lane // 4
        t4 = lane % 4
        grp = bx * KG + bq
        T.copy(Wq[grp * GROUP_WORDS], W_s)
        T.copy(Sq[grp * SGROUP_WORDS], S_s)
        T.sync_threads()
        for v in T.vectorized(4):
            S_local[v] = S_s[(warp * 8 + g) * 4 + v]
        for j in T.unroll(WARP_COLS):
            for v in T.vectorized(4):
                W_local[j * 4 + v] = W_s[(j * THREADS + tx) * 4 + v]
        for s in T.unroll(4):
            for j in T.unroll(WARP_COLS):
                T.call_extern(
                    "handle", "tl_twiddle_dequant",
                    W_local[j * 4 + s],
                    S_local[j * 2 + s // 2],
                    0x4040 + (s % 2) * 0x0202,
                    T.access_ptr(B_local[j * 8], "w", 8),
                )
                for l in T.unroll(8):
                    n = bx * BLOCK_N + warp * WARP_N + j * 16 + g + 8 * (l // 4)
                    k = bq * 64 + s * 16 + 2 * t4 + (l % 2) + 8 * ((l % 4) // 2)
                    Wd[n, k] = B_local[j * 8 + l]


# ---------------------------------------------------------------------------------------------------------------
# Host entry points

def _smem_bytes(block_M: int) -> int:
    block_K, stages = _CFG[block_M]
    return stages * (block_M * block_K * 2 + block_K * BLOCK_N // 2 + block_K * BLOCK_N // 16) + 16


def _ctas_per_sm(block_M: int) -> int:
    return max(1, min(4, SMEM_PER_SM // (_smem_bytes(block_M) + 1024)))


_SPLIT_CACHE: dict[tuple[int, int, int], int] = {}


def _pick_split(N: int, K: int, block_M: int) -> int:
    key = (N, K, block_M)
    if key in SPLIT_K_OVERRIDE:
        return SPLIT_K_OVERRIDE[key]
    if key in _SPLIT_CACHE:
        return _SPLIT_CACHE[key]
    block_K, stages = _CFG[block_M]
    n_tiles, k_tiles = N // BLOCK_N, K // block_K
    divs = [s for s in range(1, 17) if k_tiles % s == 0 and k_tiles // s >= stages] or [1]
    if block_M <= 32:
        # memory-bound: the smallest split that gives every SM a CTA (every CTA keeps (stages-1) tiles in flight,
        # enough for Little's law); fewer splits = less partial traffic. All CTAs stay co-resident.
        cap = NUM_SMS * _ctas_per_sm(block_M)
        fits = [s for s in divs if n_tiles * s <= cap] or [1]
        cover = [s for s in fits if n_tiles * s >= NUM_SMS]
        best = min(cover) if cover else max(fits)
    else:
        # tensor-core bound at M = 64 (BF16 mma ~285 TFLOPS): per-SM MMA time of the busiest SM, plus the split-K
        # partials (write + L2 read, pessimistically priced at the DRAM rate since they may be written back)
        t_kt = 2.0 * BLOCK_N * block_K * block_M / (285e12 / NUM_SMS) * 1e6        # us per k-tile per SM

        def cost(s):
            per_sm = -(-n_tiles * s // NUM_SMS) * (k_tiles // s)
            return per_sm * t_kt + (0.0 if s == 1 else s * block_M * N * 4 / 1.63e6)
        best = min(divs, key=lambda s: (cost(s), s))
    _SPLIT_CACHE[key] = best
    return best


_DUMMY_P: dict[torch.device, torch.Tensor] = {}


def linear(x: torch.Tensor, prep: TwiddledWeight) -> torch.Tensor:
    """y = x @ W^T (bf16) for 1 <= M <= 64. Deterministic and CUDA-graph capturable. The split-K tickets live in
    `prep`, so one prepared weight must not be used by two concurrently running streams."""
    M, K = x.shape
    assert K == prep.K, (K, prep.K)
    assert 1 <= M <= SUPPORTS_M, M
    assert x.dtype == torch.bfloat16, x.dtype
    x = x.contiguous()
    if x.data_ptr() % 16:                       # cp.async needs 16-byte aligned rows (K % 8 == 0 keeps rows aligned)
        x = x.clone()
    block_M = 16 if M <= 16 else (32 if M <= 32 else 64)
    block_K, stages = _CFG[block_M]
    split_k = _pick_split(prep.N, K, block_M)
    assert (K // block_K) % split_k == 0, (K, block_K, split_k)
    y = torch.empty(M, prep.N, dtype=torch.bfloat16, device=x.device)
    if split_k > 1:
        P = torch.empty(split_k, M, prep.N, dtype=torch.float32, device=x.device)
    else:
        P = _DUMMY_P.get(x.device)
        if P is None:
            P = _DUMMY_P[x.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=x.device)
    twiddle_gemm_kernel(x, prep.Wq, prep.Sq, prep.alpha, y, P, prep.counters,
                        block_M=block_M, block_K=block_K, split_k=split_k, num_stages=stages)
    return y


def dequantize_debug(prep: TwiddledWeight) -> torch.Tensor:
    """(N, K) fp32 weights as the kernel sees them (already divided by 2^(126 - SCALE_C); alpha not applied)."""
    Wd = torch.empty(prep.N, prep.K, dtype=torch.bfloat16, device=prep.Wq.device)
    twiddle_debug_kernel(prep.Wq, prep.Sq, Wd)
    return Wd.float() * ALPHA_COMP
