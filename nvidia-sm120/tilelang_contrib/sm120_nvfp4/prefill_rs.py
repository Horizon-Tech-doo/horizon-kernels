# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K3 prefill on upstream #3257 (SM120 block-scaled GEMM with A from register fragments): the residual NVFP4 GEMM with
the WEIGHTS as the register operand, read straight from decode_regfrag's Wg (one packed weight copy per layer), with no
shared-memory round trip for the weights. Experimental (27-09-2026): see our run notes, "#3257".

Why the weights become operand A (swap-AB): #3257 lets mma.sync ... kind::mxf4nvf4.block_scale take A from registers
(tilelang/cuda/op/gemm/gemm_mma_sm120.py, is_gemm_rs), B must stay in shared memory. So the kernel computes
C^T (N x M) = W (N x K) . X^T, i.e. per CTA a 128 (weight rows) x 128 (activation rows) tile, and stores it transposed.

Register layout #3257 expects for A (mma_sm120_macro_generator.make_mma_load_layout, = CuTe SM120 ALayout): per 16x64
atom, lane l = 4g + t holds 4 words: r0 = row g, k 8t..8t+7 | r1 = row g+8, same k | r2 = row g, k 32+8t.. | r3 = row
g+8, k 32+8t.. (8 e2m1 nibbles per word, element i in nibble i). regfrag's Wg (decode_regfrag.prepare) already puts
rows g / g+8 of each 16-row half on lane quad g (the bf16 m16n8k16 A layout), so no lane ever needs another row.
It differs in two ways:
  (1) k distribution: regfrag lane t holds k = 16st + 2t + {0,1,8,9}; the FP4 atom wants 8 consecutive k per lane;
  (2) bit order: every nibble's 4 bits are scattered (decode_regfrag._SLOT_POS) so regfrag's bf16 unpack is cheap.
(2) is unavoidable ALU work (a bit gather per weight). (1) is avoided WITHOUT shuffles: lane t loads its own chunk's
word (st = t // 2) and the same word of lane t ^ 2; together lanes t and t ^ 1 then hold all 16 values of one 16-k
scale group, in an order fixed by the untwiddle below. The MMA's per-scale-group sum is exact, so the same in-group
permutation applied to the activations (the quantizer reads x at permuted positions, free) gives bit-identical
results (checked on the GPU, e3257_exact.py E2).

Untwiddle (hz_rs_unit): X, Y (lane t's and lane t ^ 2's word) -> Ra (row n_a), Rb (row n_b): 2 PRMT gather the four
16-bit halves' byte 0s / byte 1s (same bit structure), then 5 + 5 shift-and-merge field moves (7 distinct shifts):
19 instructions per 16 weights. Output byte L (L = 2 * isY + h) = code(k) | code(k + 8) << 4 of slot a / c.

conv=0 is a measurement arm: the same kernel with weights already in fragment order and standard nibbles (a second
copy, NOT the one-copy design): it isolates the cost of the untwiddle from the cost of the #3257 register path.
conv=2 (no weight loads, no untwiddle) and conv=3 (the weight loads, no untwiddle) are timing-only arms (27-09-2026):
their results are wrong by design; they bound what the untwiddle and the weight loads cost. conv=4 reads a weight copy
stored natively in the #3257 A layout (per warp and K block: [k atom][row atom][lane] 16-byte MMA quads), i.e. the prefill
side of the "native layout" option; timed on Wg's bytes (wrong results) until a native copy is built.
"""

from dataclasses import dataclass

import tilelang
import tilelang.language as T

from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import TensorCoreIntrinEmitterSM120

from .prefill_rf import B_TILE_SRC

__all__ = ["NVFP4WeightRS", "prepare_weight_rs", "prepare_layer_rs", "quantize_rs", "linear_rs", "get_kernel",
           "nvfp4_gemm_rs_kernel", "nvfp4_quant_rs_kernel", "perm_rel", "fragment_order_weights", "native_order_weights",
           "prepare_weight_native", "linear_native", "NATIVE_OPTS"]

RS_SRC = B_TILE_SRC + r"""
// X, Y: twiddled Wg words (same st, j) of lanes t and t^2 -> Ra (row n_a), Rb (row n_b), standard e2m1 nibbles.
// byte lanes L: 0 = X.h0, 1 = X.h1, 2 = Y.h0, 3 = Y.h1; output byte L = code(slot a|c) | code(slot b|d) << 4.
// twiddled half (bit: field): 0 c.m 1 c.e0 2 c.e1 3 b.m 4 b.e0 5 b.e1 6 a.m 7 a.e0 | 8 a.e1 9 c.s 10 d.e0 11 d.e1
//                             12 b.s 13 d.m 14 d.s 15 a.s   (t0 = the 4 low bytes, t1 = the 4 high bytes)
__device__ __forceinline__ void hz_rs_unit(unsigned X, unsigned Y, unsigned &ra, unsigned &rb) {
  const unsigned t0 = __byte_perm(X, Y, 0x6420), t1 = __byte_perm(X, Y, 0x7531);
  const unsigned s2 = t1 << 2, s3 = t1 << 3;
  unsigned a = (t0 >> 6) & 0x03030303u;              // a.m a.e0   6,7 -> 0,1
  a |= s2 & 0x04040404u;                             // a.e1       t1.0 -> 2
  a |= (t1 >> 4) & 0x08080808u;                      // a.s        t1.7 -> 3
  a |= (t0 << 1) & 0x70707070u;                      // b.m..b.e1  3..5 -> 4..6
  a |= s3 & 0x80808080u;                             // b.s        t1.4 -> 7
  unsigned b = t0 & 0x07070707u;                     // c.m..c.e1  0..2
  b |= s2 & 0x08080808u;                             // c.s        t1.1 -> 3
  b |= (t1 >> 1) & 0x10101010u;                      // d.m        t1.5 -> 4
  b |= s3 & 0x60606060u;                             // d.e0 d.e1  t1.2,3 -> 5,6
  b |= (t1 << 1) & 0x80808080u;                      // d.s        t1.6 -> 7
  ra = a; rb = b;
}
// tw=1: the same bit gather as hz_rs_unit, written as bit-select chains (hz_sel = one LOP3 each): the first select merges
// two sources and every later one overwrites only its field, so each output word needs 4 LOP3 instead of 5:
// 2 PRMT + 7 shifts + 8 LOP3 = 17 instructions per 16 weights (hz_rs_unit: 19)
__device__ __forceinline__ void hz_rs_unit2(unsigned X, unsigned Y, unsigned &ra, unsigned &rb) {
  const unsigned t0 = __byte_perm(X, Y, 0x6420), t1 = __byte_perm(X, Y, 0x7531);
  const unsigned s2 = t1 << 2, s3 = t1 << 3;
  unsigned a = hz_sel(t0 >> 6, 0x03030303u, s2);     // a.m a.e0 6,7 -> 0,1 | a.e1 t1.0 -> 2 (s2)
  a = hz_sel(t1 >> 4, 0x08080808u, a);               // a.s        t1.7 -> 3
  a = hz_sel(t0 << 1, 0x70707070u, a);               // b.m..b.e1  3..5 -> 4..6
  a = hz_sel(s3, 0x80808080u, a);                    // b.s        t1.4 -> 7
  unsigned b = hz_sel(t0, 0x07070707u, s2);          // c.m..c.e1  0..2 | c.s t1.1 -> 3 (s2)
  b = hz_sel(t1 >> 1, 0x10101010u, b);               // d.m        t1.5 -> 4
  b = hz_sel(s3, 0x60606060u, b);                    // d.e0 d.e1  t1.2,3 -> 5,6
  b = hz_sel(t1 << 1, 0x80808080u, b);               // d.s        t1.6 -> 7
  ra = a; rb = b;
}
template <int TW> __device__ __forceinline__ void hz_rs_u(unsigned X, unsigned Y, unsigned &ra, unsigned &rb) {
  if (TW) hz_rs_unit2(X, Y, ra, rb); else hz_rs_unit(X, Y, ra, rb);
}
// One K block's weights for this lane: W_s holds the warp's 8 Wg blocks (32 rows x 32 k, 512 B each, as in Wg) in
// shared memory (bulk-copied by the producer warps). Per block kc: conv=1 reads its own word pair (st, j = 0, 1) and
// lane t^2's (two 8-byte LDS), untwiddles them into rows n_a / n_b; conv=0 reads 16 ready bytes. The #3257 fragment of
// k atom ka = kc / 2 is a[ka]: atom (row half j) at word j * 4, + 2 for the second 32-k half, + 0 / 1 for rows g / g+8.
template <int CONV, int TWV = 0, typename TW, typename TA>
__device__ __forceinline__ void hz_rs_fill(const TW *wp, TA *a0p, TA *a1p, TA *a2p, TA *a3p, int lane) {
  const unsigned *w = reinterpret_cast<const unsigned *>(wp);
  unsigned *a[4] = {reinterpret_cast<unsigned *>(a0p), reinterpret_cast<unsigned *>(a1p),
                    reinterpret_cast<unsigned *>(a2p), reinterpret_cast<unsigned *>(a3p)};
  const int t = lane & 3, g = lane >> 2, st = t >> 1;
  const int own = 4 * (4 * g + t) + 2 * st, par = 4 * (4 * g + (t ^ 2)) + 2 * st;
  if (CONV == 4) {                   // native #3257 layout: one 16-byte LDS per MMA A quad (row atom i, k atom ka)
#pragma unroll
    for (int q = 0; q < 8; ++q)
      *reinterpret_cast<uint4 *>(a[q >> 1] + 4 * (q & 1)) = *reinterpret_cast<const uint4 *>(w + q * 128 + 4 * lane);
    return;
  }
#pragma unroll
  for (int kc = 0; kc < 8; ++kc) {
    unsigned *f = a[kc >> 1];
    const int h = (kc & 1) * 2;
    if (CONV == 2) {                 // measurement only: no weight traffic, no untwiddle (wrong results)
      f[h] = 0u; f[h + 1] = 0u; f[4 + h] = 0u; f[4 + h + 1] = 0u;
    } else if (CONV == 3) {          // measurement only: the same shared-memory loads, no untwiddle (wrong results)
      const uint2 x = *reinterpret_cast<const uint2 *>(w + kc * 128 + own);
      const uint2 y = *reinterpret_cast<const uint2 *>(w + kc * 128 + par);
      f[h] = x.x; f[h + 1] = y.x; f[4 + h] = x.y; f[4 + h + 1] = y.y;
    } else if (CONV) {
      const uint2 x = *reinterpret_cast<const uint2 *>(w + kc * 128 + own);
      const uint2 y = *reinterpret_cast<const uint2 *>(w + kc * 128 + par);
      hz_rs_u<TWV>(x.x, y.x, f[h], f[h + 1]);
      hz_rs_u<TWV>(x.y, y.y, f[4 + h], f[4 + h + 1]);
    } else {
      const uint4 v = *reinterpret_cast<const uint4 *>(w + kc * 128 + 4 * lane);
      f[h] = v.x; f[h + 1] = v.y; f[4 + h] = v.z; f[4 + h + 1] = v.w;
    }
  }
}
// kcall=1: the four atoms live in ONE (128 x 256) fragment, local order [k atom][row atom][4 words] (the order
// TensorCoreIntrinEmitterSM120.mma indexes: k_inner * warp_rows * 4 + i * 4 words)
template <int CONV, int TWV = 0, typename TW, typename TA>
__device__ __forceinline__ void hz_rs_fill_one(const TW *wp, TA *ap, int lane) {
  unsigned *b = reinterpret_cast<unsigned *>(ap);
  hz_rs_fill<CONV, TWV>(wp, b, b + 8, b + 16, b + 24, lane);
}
// ilv=1: one 64-k atom (Wg blocks kc = 2 ka, 2 ka + 1) into fragment a, right before that atom's pass-0 GEMM, so
// the untwiddle interleaves with the MMA issue stream instead of preceding the whole K block
template <int CONV, int TWV = 0, typename TW, typename TA>
__device__ __forceinline__ void hz_rs_fill_atom(const TW *wp, TA *ap, int ka, int lane) {
  const unsigned *w = reinterpret_cast<const unsigned *>(wp);
  unsigned *f = reinterpret_cast<unsigned *>(ap);
  const int t = lane & 3, g = lane >> 2, st = t >> 1;
  const int own = 4 * (4 * g + t) + 2 * st, par = 4 * (4 * g + (t ^ 2)) + 2 * st;
#pragma unroll
  for (int hk = 0; hk < 2; ++hk) {
    const int kc = 2 * ka + hk, h = hk * 2;
    if (CONV) {
      const uint2 x = *reinterpret_cast<const uint2 *>(w + kc * 128 + own);
      const uint2 y = *reinterpret_cast<const uint2 *>(w + kc * 128 + par);
      hz_rs_u<TWV>(x.x, y.x, f[h], f[h + 1]);
      hz_rs_u<TWV>(x.y, y.y, f[4 + h], f[4 + h + 1]);
    } else {
      const uint4 v = *reinterpret_cast<const uint4 *>(w + kc * 128 + 4 * lane);
      f[h] = v.x; f[h + 1] = v.y; f[4 + h] = v.z; f[4 + h + 1] = v.w;
    }
  }
}
// pu=1: the PRODUCER warps untwiddle one staged Wg tile in place into the native #3257 order (conv=4 layout): producer
// thread ptid plays consumer lane (ptid & 31) of consumer warp (ptid >> 5): the same hz_rs_fill into registers, then
// (after every producer thread has read its words: named barrier 2, producers only) one 16-byte store per MMA quad.
template <int TWV, typename TW>
__device__ __forceinline__ void hz_rs_prod_untwiddle(TW *wp, int ptid) {
  unsigned *w = reinterpret_cast<unsigned *>(wp) + (ptid >> 5) * 1024;
  const int l = ptid & 31;
  unsigned a[4][8];
  hz_rs_fill<1, TWV>(w, a[0], a[1], a[2], a[3], l);
  asm volatile("bar.sync 2, 128;" ::: "memory");
#pragma unroll
  for (int q = 0; q < 8; ++q)
    *reinterpret_cast<uint4 *>(w + q * 128 + 4 * l) = *reinterpret_cast<const uint4 *>(a[q >> 1] + 4 * (q & 1));
}
// the 4 consumer warps only (named barrier 1): before the epilogue writes C_s, every consumer warp must have finished
// its last ldmatrix from X_s, because TileLang's shared-memory merge may place C_s on top of an X_s stage (a_stages=3)
__device__ __forceinline__ void hz_rs_consumer_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
// producer warp 0 of the CTA: the 32 Wg blocks of one K block (4 row blocks x 8 k blocks, 16 KB) by 1D bulk copies,
// counted on bar (expect_tx issued by lane 0)
template <typename TG, typename TD, typename TB>
__device__ __forceinline__ void hz_rs_w_issue(const TG *src, TD *dst, TB *bar, int nb_stride_bytes, int lane) {
  const unsigned b = hz_saddr(bar), d = hz_saddr(dst);
  const char *gp = reinterpret_cast<const char *>(src);
  if (lane == 0) asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(16384) : "memory");
  __syncwarp();
  const int nb = lane >> 3, kc = lane & 7;
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];"
               :: "r"(d + (nb * 8 + kc) * 512), "l"(gp + (long long)nb * nb_stride_bytes + kc * 512), "r"(b) : "memory");
}
"""

SF_GRANULARITY_K = 16


def perm_rel(glane: int, e: int, st: int) -> int:
    """Logical k (0..15, inside a 16-k scale group) held at physical position p = 2 * glane + e of that group, where
    st = group index parity inside its 32-k block. Matches hz_rs_unit's output order (see module doc)."""
    return 2 * (glane >> 2) + (glane & 1) + 8 * e + 4 * (((glane >> 1) & 1) ^ st)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def nvfp4_quant_rs_kernel(x, rs, passes: int = 2, rows_per_cta: int = 32, num_threads: int = 512):
    """kernels.nvfp4_quant_kernel with (a) the in-group k permutation of hz_rs_unit (reads x at permuted positions:
    each 16-value group is the same set, so scales and codes are identical, only their order in the group changes) and
    (b) tile-contiguous row-major scale words: SX[p, (mt * K/256 + kb) * 128 + r, w] (one 2 KB tile per (mt, kb))."""
    from ._device import encode_ue4m3_ceil, decode_ue4m3, ue4m3_inverse, FP4_E2M1_MAX, CHUNK_WORDS
    M = T.dynamic("M")
    K = T.const("K")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    kbn = K // 256
    f32 = T.float32
    x: T.Tensor[(M, K), T.bfloat16]
    rs: T.Tensor[(M,), T.float32]
    A = T.empty((M, passes * (K // 2)), T.int8)
    SX = T.empty((passes, M * kbn, 4), T.uint32)

    with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        x_s = T.alloc_shared((rows_per_cta, 256), T.bfloat16)
        codes = T.alloc_shared((passes, rows_per_cta, 16), T.int32)
        T.copy(x[pid_m * rows_per_cta, pid_k256 * 256], x_s, prefer_instruction="tma")
        T.sync_threads()
        glane = lane % 8
        st = (lane // 8) % 2
        rel0 = 2 * (glane // 4) + (glane % 2) + 4 * (((glane // 2) % 2) ^ st)
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            grow = pid_m * rows_per_cta + row
            for k64 in T.serial(CHUNK_WORDS):
                gbase = k64 * 64 + (lane // 8) * 16
                byte = pid_k256 * 128 + k64 * 32 + lane
                group = k64 * 4 + (lane // 8)
                r0 = T.alloc_var(f32, init=T.cast(x_s[row, gbase + rel0], f32) * rs[grow])
                r1 = T.alloc_var(f32, init=T.cast(x_s[row, gbase + rel0 + 8], f32) * rs[grow])
                for p in T.unroll(passes):
                    am = T.alloc_var(f32, init=T.max(T.abs(r0), T.abs(r1)))
                    am = T.max(am, T.shfl_xor(am, 4, width=8))
                    am = T.max(am, T.shfl_xor(am, 2, width=8))
                    am = T.max(am, T.shfl_xor(am, 1, width=8))
                    code = T.alloc_var(T.int32, init=0)
                    sc = T.alloc_var(f32, init=0.0)
                    inv = T.alloc_var(f32, init=0.0)
                    if glane == 0:
                        code = encode_ue4m3_ceil(am / FP4_E2M1_MAX)
                        sc = decode_ue4m3(code)
                        inv = ue4m3_inverse(code)
                        codes[p, row, group] = code
                    sc = T.shfl_sync(sc, 0, width=8)
                    inv = T.shfl_sync(inv, 0, width=8)
                    f0 = T.cast(r0 * inv, T.float4_e2m1fn)
                    f1 = T.cast(r1 * inv, T.float4_e2m1fn)
                    A[grow, p * (K // 2) + byte] = T.cast(T.reinterpret(f0, T.uint8) | (T.reinterpret(f1, T.uint8) << 4), T.int8)
                    r0 = r0 - T.cast(f0, f32) * sc
                    r1 = r1 - T.cast(f1, f32) * sc
        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            grow = pid_m * rows_per_cta + row
            if lane < CHUNK_WORDS:
                for p in T.unroll(passes):
                    SX[p, ((grow // 128) * kbn + pid_k256) * 128 + grow % 128, lane] = (
                        T.cast(codes[p, row, lane * 4], T.uint32)
                        | (T.cast(codes[p, row, lane * 4 + 1], T.uint32) << 8)
                        | (T.cast(codes[p, row, lane * 4 + 2], T.uint32) << 16)
                        | (T.cast(codes[p, row, lane * 4 + 3], T.uint32) << 24))

    return A, SX


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def nvfp4_gemm_rs_kernel(N: int, K: int, passes: int = 2, out_dtype=T.bfloat16, conv: int = 1, a_stages: int = 2,
                         ilv: int = 0, tw: int = 0, epi: int = 0, kcall: int = 0, roll: int = 0, pu: int = 0):
    """C (M, N) = alpha[n] * rsinv[m] * sum_p dq(X_p) @ dq(W)^T with W as the #3257 register operand (A), from regfrag's
    Wg (conv=1) or a fragment-ordered copy (conv=0). M dynamic, multiple of 128. 256 threads: 128 consumers (4 warps x
    32 weight rows: fill the A fragments from the staged Wg tile, then T.gemm_blockscaled per 64-k atom and pass),
    128 producers (TMA of the activation tiles; bulk copies of the Wg tile and of both scale tiles).
    tw=1: the 17-instruction untwiddle (hz_rs_unit2). epi=1: alpha / rsinv staged once per CTA in shared memory by the
    producers and applied to the accumulators in registers (the same fp32 products in the same order: bit-identical),
    instead of two global loads per output element inside the transposing store. kcall=1: ONE T.gemm_blockscaled per
    K block and pass on a (128 x 256) A fragment (needs the unrolled register-A lowering), so the lowering can prefetch
    the next atom's B fragments across atoms. roll=1 (passes=2): software-pipelined fill: K block kb+1's weights are
    untwiddled atom by atom during pass 1 of K block kb, each right after the MMAs that last read that atom's fragment,
    so the ALU work overlaps the tensor pipe instead of stalling it (no extra registers: the fragment is refilled in
    place; the MMA order and so the result are unchanged). pu=1: the untwiddle moves to the producer warps: after issuing
    K block kb's last pass, they wait for its raw Wg tile, convert it in place into the native #3257 order and signal
    wn_full; the consumers then fill each MMA quad with one 16-byte shared load (the conv=4 path) from regfrag's Wg,
    still ONE weight copy in global memory. pu=2: the same, but K block kb+1's raw Wg tile is fetched one pass earlier
    (with pass 1 of kb, on its own barrier w_full) and untwiddled right after issuing (kb+1, pass 0), giving the producer
    two passes of lead instead of one (needs a_stages=3)."""
    bw, bm, bk = 128, 128, 256
    assert N % bw == 0 and K % bk == 0, (N, K)
    assert a_stages <= passes + 1        # W_s / SW_s (per K block, 2 buffers) are refilled only after their last use
    kbn = K // bk
    kc32 = K // 32
    M = T.dynamic("M")
    fp4, i32, u32, f32 = T.float4_e2m1fn, T.int32, T.uint32, T.float32
    # the #3257 A-fragment layout T.gemm_blockscaled infers for a 128 x 64 A tile (FullRow: 4 warps x 32 weight rows),
    # pinned so the extern fill (writes the fragment registers through a pointer) and the GEMM agree on it
    emitter = TensorCoreIntrinEmitterSM120(a_dtype=fp4, b_dtype=fp4, accum_dtype=f32, a_transposed=False,
                                           b_transposed=True, block_row_warps=4, block_col_warps=1, warp_row_tiles=32,
                                           warp_col_tiles=bm, chunk=64, num_elems_per_byte=2, is_blockscaled=True)
    emitter_k = TensorCoreIntrinEmitterSM120(a_dtype=fp4, b_dtype=fp4, accum_dtype=f32, a_transposed=False,
                                             b_transposed=True, block_row_warps=4, block_col_warps=1, warp_row_tiles=32,
                                             warp_col_tiles=bm, chunk=bk, num_elems_per_byte=2, is_blockscaled=True)

    @T.macro
    def gemm_pass_fill(A0, A1, A2, A3, X_s, stage, C_l, SW_s, wslot, SX_s, W_s, warp, lane):
        # ilv=1, pass 0: convert each k atom right before its GEMM (same accumulation order as K3)
        for ka, Af in ((0, A0), (1, A1), (2, A2), (3, A3)):
            T.call_extern("handle", f"hz_rs_fill_atom<{conv}, {tw}>", T.access_ptr(W_s[wslot, warp * 1024], "r"),
                          T.access_ptr(Af[0, 0], "w"), ka, lane)
            T.gemm_blockscaled(Af, X_s[stage, :, ka * 64:(ka + 1) * 64], C_l, SW_s[wslot, :, :], SX_s[stage, :, :],
                               transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=False, k_start=ka * 64,
                               sf_a_granularity_k=SF_GRANULARITY_K, sf_b_granularity_k=SF_GRANULARITY_K,
                               sf_layout="rowmajor")

    @T.macro
    def gemm_pass_roll(A0, A1, A2, A3, X_s, stage, C_l, SW_s, wslot, SX_s, W_s, warp, lane, kb, bars, nxt, nxt_par):
        # last pass of K block kb: MMAs of atom ka, then refill atom ka with K block kb+1 (its W tile arrived with the
        # producer's (kb+1, pass 0) stage: wait for it once, before the first refill)
        for ka, Af in ((0, A0), (1, A1), (2, A2), (3, A3)):
            T.gemm_blockscaled(Af, X_s[stage, :, ka * 64:(ka + 1) * 64], C_l, SW_s[wslot, :, :], SX_s[stage, :, :],
                               transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=False, k_start=ka * 64,
                               sf_a_granularity_k=SF_GRANULARITY_K, sf_b_granularity_k=SF_GRANULARITY_K,
                               sf_layout="rowmajor")
            if kb + 1 < kbn:
                if ka == 0:
                    T.barrier_wait(bars[nxt], nxt_par)
                T.call_extern("handle", f"hz_rs_fill_atom<{conv}, {tw}>",
                              T.access_ptr(W_s[1 - wslot, warp * 1024], "r"), T.access_ptr(Af[0, 0], "w"), ka, lane)

    @T.macro
    def gemm_pass_k(Ak, X_s, stage, C_l, SW_s, wslot, SX_s):
        T.gemm_blockscaled(Ak, X_s[stage, :, :], C_l, SW_s[wslot, :, :], SX_s[stage, :, :], transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow, clear_accum=False, k_start=0,
                           sf_a_granularity_k=SF_GRANULARITY_K, sf_b_granularity_k=SF_GRANULARITY_K, sf_layout="rowmajor")

    @T.macro
    def gemm_pass(A0, A1, A2, A3, X_s, stage, C_l, SW_s, wslot, SX_s):
        # one 64-k atom per call: the fragment is indexed statically (the #3257 rs lowering loops over k atoms with a
        # runtime index, which would put a 256-k fragment in local memory); pass/k order = K3's accumulation order
        for ka, Af in ((0, A0), (1, A1), (2, A2), (3, A3)):
            T.gemm_blockscaled(Af, X_s[stage, :, ka * 64:(ka + 1) * 64], C_l, SW_s[wslot, :, :], SX_s[stage, :, :], transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow, clear_accum=False, k_start=ka * 64,
                               sf_a_granularity_k=SF_GRANULARITY_K, sf_b_granularity_k=SF_GRANULARITY_K,
                               sf_layout="rowmajor")

    @T.prim_func
    def main(
        X: T.Tensor((M, passes * K), fp4),
        W: T.Tensor((N // 32, kc32, 128), i32),
        SX: T.Tensor((passes, M * kbn, 4), u32),
        SW: T.Tensor((N * kbn, 4), u32),
        alpha: T.Tensor((N,), f32),
        rsinv: T.Tensor((M,), f32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(N // bw, T.ceildiv(M, bm), threads=256, prelude=RS_SRC) as (bx, by):
            X_s = T.alloc_shared((a_stages, bm, bk), fp4)
            SX_s = T.alloc_shared((a_stages, bm, 4), u32)
            SW_s = T.alloc_shared((2, bw, 4), u32)
            W_s = T.alloc_shared((2, 4096), i32)                   # the CTA's Wg tile of one K block (16 KB) x 2
            C_s = T.alloc_shared((bm, bw), out_dtype)
            al_s = T.alloc_shared((bw,), f32)
            rs_s = T.alloc_shared((bm,), f32)
            A0 = T.alloc_fragment((bw, 64), fp4)
            A1 = T.alloc_fragment((bw, 64), fp4)
            A2 = T.alloc_fragment((bw, 64), fp4)
            A3 = T.alloc_fragment((bw, 64), fp4)
            Ak = T.alloc_fragment((bw, bk), fp4)
            C_l = T.alloc_fragment((bw, bm), f32)
            a_full = T.alloc_barrier([128] * a_stages)
            a_empty = T.alloc_barrier([128] * a_stages)
            wn_full = T.alloc_barrier([128] * 2)
            w_full = T.alloc_barrier([1] * 2)
            T.annotate_layout({A0: emitter.make_mma_load_layout(A0, "A"), A1: emitter.make_mma_load_layout(A1, "A"),
                               A2: emitter.make_mma_load_layout(A2, "A"), A3: emitter.make_mma_load_layout(A3, "A"),
                               Ak: emitter_k.make_mma_load_layout(Ak, "A")})
            tx = T.get_thread_binding()
            ptid = tx - 128
            warp = tx // 32
            lane = tx % 32
            with T.ws(0):
                T.clear(C_l)
            if epi:
                with T.ws(1):
                    # visible to the consumers through the a_full mbarrier (arrive = release, wait = acquire)
                    al_s[ptid] = alpha[bx * bw + ptid]
                    rs_s[ptid] = T.if_then_else(by * bm + ptid < M, rsinv[by * bm + ptid], T.float32(0))
            for kb in T.serial(kbn):
                with T.ws(1):
                    for p in T.unroll(passes):
                        it = kb * passes + p
                        T.barrier_wait(a_empty[it % a_stages], ((it // a_stages) % 2) ^ 1)
                        T.tma_copy(X[by * bm:(by + 1) * bm, p * K + kb * bk:p * K + (kb + 1) * bk],
                                   X_s[it % a_stages, :, :], barrier=a_full[it % a_stages])
                        if ptid == 0:
                            T.call_extern("handle", "hz_bulk_tx", T.access_ptr(SX[p, (by * kbn + kb) * bm, 0], "r"),
                                          T.access_ptr(SX_s[it % a_stages, 0, 0], "w"),
                                          T.access_ptr(a_full[it % a_stages], "rw"), bm * 16)
                        if pu == 2:
                            # raw Wg tile of K block kb + 1 fetched with pass 1 of kb (K block 0: with its pass 0), on
                            # its own barrier w_full
                            if p == 1:
                                if kb + 1 < kbn:
                                    if ptid < 32:
                                        T.call_extern("handle", "hz_rs_w_issue", T.access_ptr(W[bx * 4, (kb + 1) * 8, 0], "r"),
                                                      T.access_ptr(W_s[(kb + 1) % 2, 0], "w"),
                                                      T.access_ptr(w_full[(kb + 1) % 2], "rw"), kc32 * 512, ptid)
                                    if ptid == 0:
                                        T.barrier_arrive(w_full[(kb + 1) % 2])
                            else:
                                if kb == 0:
                                    if ptid < 32:
                                        T.call_extern("handle", "hz_rs_w_issue", T.access_ptr(W[bx * 4, 0, 0], "r"),
                                                      T.access_ptr(W_s[0, 0], "w"), T.access_ptr(w_full[0], "rw"),
                                                      kc32 * 512, ptid)
                                    if ptid == 0:
                                        T.barrier_arrive(w_full[0])
                        if p == 0:
                            if ptid == 0:
                                T.call_extern("handle", "hz_bulk_tx", T.access_ptr(SW[(bx * kbn + kb) * bw, 0], "r"),
                                              T.access_ptr(SW_s[kb % 2, 0, 0], "w"),
                                              T.access_ptr(a_full[it % a_stages], "rw"), bw * 16)
                            if pu != 2:
                                if ptid < 32:
                                    T.call_extern("handle", "hz_rs_w_issue", T.access_ptr(W[bx * 4, kb * 8, 0], "r"),
                                                  T.access_ptr(W_s[kb % 2, 0], "w"), T.access_ptr(a_full[it % a_stages], "rw"),
                                                  kc32 * 512, ptid)
                        T.barrier_arrive(a_full[it % a_stages])
                        if pu == 2 and p == 0:
                            # untwiddle K block kb's tile (fetched one pass earlier) in place, two passes before use
                            T.barrier_wait(w_full[kb % 2], (kb // 2) % 2)
                            T.call_extern("handle", f"hz_rs_prod_untwiddle<{tw}>", T.access_ptr(W_s[kb % 2, 0], "rw"),
                                          ptid)
                            T.barrier_arrive(wn_full[kb % 2])
                    if pu == 1:
                        # K block kb's raw Wg tile landed with a_full(kb, pass 0): untwiddle it in place, then release it
                        # to the consumers (they need it at (kb, pass 0), i.e. about one pass from now)
                        T.barrier_wait(a_full[(kb * passes) % a_stages], ((kb * passes) // a_stages) % 2)
                        T.call_extern("handle", f"hz_rs_prod_untwiddle<{tw}>", T.access_ptr(W_s[kb % 2, 0], "rw"), ptid)
                        T.barrier_arrive(wn_full[kb % 2])
                with T.ws(0):
                    for p in T.unroll(passes):
                        it = kb * passes + p
                        T.barrier_wait(a_full[it % a_stages], (it // a_stages) % 2)
                        if pu:
                            if p == 0:
                                T.barrier_wait(wn_full[kb % 2], (kb // 2) % 2)
                                T.call_extern("handle", "hz_rs_fill<4, 0>", T.access_ptr(W_s[kb % 2, warp * 1024], "r"),
                                              T.access_ptr(A0[0, 0], "w"), T.access_ptr(A1[0, 0], "w"),
                                              T.access_ptr(A2[0, 0], "w"), T.access_ptr(A3[0, 0], "w"), lane)
                            gemm_pass(A0, A1, A2, A3, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s)
                        elif roll:
                            if p == 0:
                                if kb == 0:
                                    T.call_extern("handle", f"hz_rs_fill<{conv}, {tw}>",
                                                  T.access_ptr(W_s[0, warp * 1024], "r"),
                                                  T.access_ptr(A0[0, 0], "w"), T.access_ptr(A1[0, 0], "w"),
                                                  T.access_ptr(A2[0, 0], "w"), T.access_ptr(A3[0, 0], "w"), lane)
                                gemm_pass(A0, A1, A2, A3, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s)
                            else:
                                gemm_pass_roll(A0, A1, A2, A3, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s, W_s, warp,
                                               lane, kb, a_full, (it + 1) % a_stages, ((it + 1) // a_stages) % 2)
                        elif kcall:
                            if p == 0:
                                T.call_extern("handle", f"hz_rs_fill_one<{conv}, {tw}>",
                                              T.access_ptr(W_s[kb % 2, warp * 1024], "r"), T.access_ptr(Ak[0, 0], "w"), lane)
                            gemm_pass_k(Ak, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s)
                        elif ilv and p == 0:
                            gemm_pass_fill(A0, A1, A2, A3, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s, W_s, warp, lane)
                        else:
                            if p == 0:
                                # this K block's weights: staged Wg words -> the four A fragments (registers only)
                                T.call_extern("handle", f"hz_rs_fill<{conv}, {tw}>", T.access_ptr(W_s[kb % 2, warp * 1024], "r"),
                                              T.access_ptr(A0[0, 0], "w"), T.access_ptr(A1[0, 0], "w"),
                                              T.access_ptr(A2[0, 0], "w"), T.access_ptr(A3[0, 0], "w"), lane)
                            gemm_pass(A0, A1, A2, A3, X_s, it % a_stages, C_l, SW_s, kb % 2, SX_s)
                        T.barrier_arrive(a_empty[it % a_stages])
            with T.ws(0):
                T.call_extern("handle", "hz_rs_consumer_sync")
                if epi:
                    for i, j in T.Parallel(bw, bm):
                        C_l[i, j] = C_l[i, j] * (al_s[i] * rs_s[j])
                    for i, j in T.Parallel(bw, bm):
                        C_s[j, i] = T.cast(C_l[i, j], out_dtype)
                else:
                    for i, j in T.Parallel(bw, bm):
                        C_s[j, i] = T.cast(C_l[i, j] * (alpha[bx * bw + i] * rsinv[by * bm + j]), out_dtype)
            T.sync_threads()
            T.copy(C_s, C[by * bm, bx * bw])

    return main


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class NVFP4WeightRS:
    w: object           # int32 (N/32, K/32, 128): regfrag's Wg (conv=1, shared, not a copy) or fragment-ordered (conv=0)
    sw: object          # uint32 (N/128 * K/256 * 128, 4): |scale| words, row-major per 128 x 256 tile
    alpha: object       # fp32 (N,)
    n: int
    k: int
    conv: int


def _scale_tiles(weight_scale):
    import torch
    from . import _pack_scale_bytes
    n, k = weight_scale.shape[0], weight_scale.shape[1] * 16
    words = _pack_scale_bytes(weight_scale.view(torch.uint8) & 0x7F)            # (N, K/64) row-major
    return words.view(n // 128, 128, k // 256, 4).permute(0, 2, 1, 3).contiguous().view(-1, 4)


def _alpha(weight_global_scale, n, shard_sizes, device):
    import torch
    gs = weight_global_scale.float().reshape(-1)
    if gs.numel() == 1:
        return (1.0 / gs).expand(n).contiguous().to(device)
    sizes = list(shard_sizes) if shard_sizes is not None else [n // gs.numel()] * gs.numel()
    return torch.cat([(1.0 / g).expand(sz) for g, sz in zip(gs, sizes)]).contiguous().to(device)


def prepare_weight_rs(wg, weight_scale, weight_global_scale, shard_sizes=None) -> NVFP4WeightRS:
    """One-copy handle: the weights ARE regfrag's Wg (decode_regfrag(_xl).prepare(...).Wg)."""
    n, k = weight_scale.shape[0], weight_scale.shape[1] * 16
    return NVFP4WeightRS(wg, _scale_tiles(weight_scale), _alpha(weight_global_scale, n, shard_sizes, wg.device), n, k, 1)


def prepare_layer_rs(weight_packed, weight_scale, weight_global_scale, shard_sizes=None):
    """(decode_regfrag_xl Prepared, NVFP4WeightRS sharing its Wg): ONE packed weight copy for decode and prefill."""
    import torch
    from . import decode_regfrag_xl
    n = weight_packed.shape[0]
    alpha = _alpha(weight_global_scale, n, shard_sizes, weight_packed.device)
    prep = decode_regfrag_xl.prepare(weight_packed.view(torch.int8), weight_scale, alpha)
    return prep, NVFP4WeightRS(prep.Wg, _scale_tiles(weight_scale), alpha, n, weight_packed.shape[1] * 2, 1)


def fragment_order_weights(weight_packed, weight_scale):
    """conv=0 measurement arm: the fragment-ordered standard-nibble copy hz_rs_convert<1> would produce from Wg
    (codes folded like decode_regfrag.prepare: zero scale -> 0, negative scale -> flipped sign)."""
    import torch
    u = weight_packed.view(torch.uint8)
    n, k = u.shape[0], u.shape[1] * 2
    nib = torch.stack(((u & 0xF), (u >> 4)), dim=-1).reshape(n, k)
    s = weight_scale.view(torch.uint8)
    nib = nib ^ (((s >> 7) & 1).repeat_interleave(16, 1) << 3)
    nib = nib.masked_fill((s & 0x7F).repeat_interleave(16, 1) == 0, 0)
    dev = u.device
    lane = torch.arange(32, device=dev).view(32, 1, 1, 1)      # [lane][j][row a|b][nibble]
    j = torch.arange(2, device=dev).view(1, 2, 1, 1)
    rb = torch.arange(2, device=dev).view(1, 1, 2, 1)
    q = torch.arange(8, device=dev).view(1, 1, 1, 8)
    g, t = lane // 4, lane % 4
    st, u_ = t // 2, t % 2
    L, e = q // 2, q % 2
    rel = 2 * u_ + (L % 2) + 8 * e + 4 * ((L // 2) ^ st)
    row = 16 * j + g + 8 * rb                                   # row inside the 32-row block
    kin = 16 * st + rel                                         # k inside the 32-k block
    blk = nib.view(n // 32, 32, k // 32, 32).permute(0, 2, 1, 3)   # (nb, kc, row, k)
    codes = blk[:, :, row.expand(32, 2, 2, 8), kin.expand(32, 2, 2, 8)].to(torch.int64)  # (nb, kc, 32, 2, 2, 8)
    words = (codes << (4 * torch.arange(8, device=dev))).sum(-1)                     # (nb, kc, 32, 2, 2)
    words = words.reshape(n // 32, k // 32, 128)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).contiguous()


def native_order_weights(weight_packed, weight_scale, slab_rows: int = 0):
    """conv=4 (27-09-2026, "native layout" evaluation): the weights stored in the #3257 A-fragment order, per 32-row
    block and K block [k atom][row atom][lane][r0 r1 r2 r3], so each MMA's A quad is ONE 16-byte shared load and no
    untwiddle is needed. Same codes and in-group k permutation as fragment_order_weights (only the word order differs),
    same (N/32, K/32, 128) int32 shape as regfrag's Wg. NOT regfrag's layout: decode would need its own kernel.
    slab_rows > 0: convert that many rows at a time into a preallocated result (every 32-row block is independent), so
    the int64 temporaries stay bounded (the unslabbed form needs ~9 bytes per weight of scratch: 1.6 GB for gate_up)."""
    import torch
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    if slab_rows and n > slab_rows:
        assert slab_rows % 32 == 0, slab_rows
        out = torch.empty(n // 32, k // 32, 128, dtype=torch.int32, device=weight_packed.device)
        for n0 in range(0, n, slab_rows):
            n1 = min(n, n0 + slab_rows)
            out[n0 // 32:n1 // 32] = native_order_weights(weight_packed[n0:n1], weight_scale[n0:n1])
        return out
    c0 = fragment_order_weights(weight_packed, weight_scale)
    nb, kc32 = c0.shape[0], c0.shape[1]
    c0 = c0.view(nb, kc32 // 8, 4, 2, 32, 2, 2)                  # nb, kb, ka, kc parity, lane, row atom j, rb
    return c0.permute(0, 1, 2, 5, 4, 3, 6).contiguous().view(nb, kc32, 128)


# Native-order prefill (HORIZON_NATIVE_PREFILL, 27-09-2026): the kernel options per pass count that won race r5/r6
# (rsN3: conv=4, epi=1, a_stages=3; bit-identical to K3, 0.93-0.95 of its GEMM time layer-weighted). a_stages is capped
# at passes + 1 by the kernel, so one pass (lossy fp4x1) runs with 2 stages.
NATIVE_OPTS = {2: {"epi": 1, "a_stages": 3}, 1: {"epi": 1, "a_stages": 2}}


def prepare_weight_native(weight_packed, weight_scale, weight_global_scale=None, shard_sizes=None, alpha=None,
                          slab_rows: int | None = None) -> NVFP4WeightRS:
    """compressed-tensors NVFP4 tensors -> a native #3257-order handle for linear_native: the SAME bytes as K3's packed
    copy (N * K/2) plus the same scale bytes (N * K/16), so it replaces sm120_nvfp4.prepare_weight's handle at equal
    memory. alpha (fp32 (N,)) may be given directly (e.g. the one decode_regfrag already uses); otherwise it is built
    from weight_global_scale / shard_sizes exactly like prepare_weight. The checkpoint tensors are not referenced
    afterwards, so the caller may free them."""
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    assert n % 128 == 0 and k % 256 == 0, (n, k)
    if alpha is None:
        alpha = _alpha(weight_global_scale, n, shard_sizes, weight_packed.device)
    if slab_rows is None:
        slab_rows = max(32, ((1 << 22) // k) // 32 * 32)          # ~4M codes per slab, like decode_regfrag.prepare
    w = native_order_weights(weight_packed, weight_scale, slab_rows=slab_rows)
    return NVFP4WeightRS(w, _scale_tiles(weight_scale), alpha.contiguous(), n, k, 4)


def linear_native(x, w: NVFP4WeightRS, passes: int = 2, out_dtype=None):
    """sm120_nvfp4.linear(x, w_k3, passes) on a native-order handle (prepare_weight_native): bit-identical output."""
    assert w.conv == 4, w.conv
    return linear_rs(x, w, passes, out_dtype, **NATIVE_OPTS[passes])


def quantize_rs(x, passes: int = 2):
    """(M, K) bf16 -> (X int8 (Mp, passes*K/2) in the in-group permuted order, SX uint32 tiles, rsinv (Mp,))."""
    import torch
    from . import _ROW_TARGET, _ROWS
    assert x.dim() == 2 and x.shape[1] % 256 == 0, x.shape
    m = x.shape[0]
    mp = (m + _ROWS - 1) // _ROWS * _ROWS
    if mp != m:
        x = torch.nn.functional.pad(x, (0, 0, 0, mp - m))
    x = x.contiguous()
    amax = x.abs().amax(dim=1).float()
    ok = amax > 0
    rs = torch.where(ok, _ROW_TARGET / amax.clamp_min(1e-30), torch.ones_like(amax))
    rsinv = torch.where(ok, amax / _ROW_TARGET, torch.ones_like(amax))
    X, SX = nvfp4_quant_rs_kernel(x, rs, passes=passes)
    return X, SX, rsinv


_KERNELS = {}


def get_kernel(n, k, passes=2, out_dtype=None, conv=1, **opts):
    import torch
    odt = torch.float32 if out_dtype == torch.float32 else torch.bfloat16
    key = (n, k, passes, odt, conv) + tuple(sorted(opts.items()))
    kern = _KERNELS.get(key)
    if kern is None:
        kern = _KERNELS[key] = nvfp4_gemm_rs_kernel(n, k, passes, T.float32 if odt == torch.float32 else T.bfloat16,
                                                    conv, **opts)
    return kern


def linear_rs(x, w: NVFP4WeightRS, passes: int = 2, out_dtype=None, **opts):
    """Same contract as sm120_nvfp4.linear(x, w_checkpoint, passes); bit-identical output is the goal."""
    import torch
    assert x.dim() == 2 and x.shape[1] == w.k, (tuple(x.shape), w.k)
    m = x.shape[0]
    X, SX, rsinv = quantize_rs(x, passes)
    mp = X.shape[0]
    kern = get_kernel(w.n, w.k, passes, out_dtype, w.conv, **opts)
    C = torch.empty(mp, w.n, dtype=torch.float32 if out_dtype == torch.float32 else torch.bfloat16, device=x.device)
    kern(X, w.w, SX, w.sw, w.alpha, rsinv, C)
    return C[:m]
