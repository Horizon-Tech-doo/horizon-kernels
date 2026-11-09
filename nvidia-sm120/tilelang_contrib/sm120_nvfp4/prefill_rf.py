# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K3 on regfrag's weight copy: the residual NVFP4 prefill GEMM (kernels.nvfp4_gemm_kernel) reading its B operand from
decode_regfrag's fragment-ordered weights (Prepared.Wg), so one NVFP4 layer needs ONE packed 4-bit weight copy on the GPU.

Bit-exact with kernels.nvfp4_gemm_kernel (same MMA, same tile, same scale words, same accumulation order, same
epilogue); only where the B tile comes from differs.

Layout facts this relies on (checked against the code and the generated CUDA, 26-09-2026):
* decode_regfrag.prepare: Wg int32 (N/32, K/32, 128); each 32x32 block is 512 bytes, lane l = 4g + t owns bytes
  [16l, 16l+16) = 4 words w = 2*st + j, and word w holds the 8 codes of rows n_a = 16j + g, n_b = n_a + 8 at
  k = 16st + 2t + {0, 1, 8, 9} (decode_regfrag._fragment_maps). Code (s, e1, e0, m) of slot r / half h sits at bit
  positions _SLOT_POS[r][h]; h = 1 is h = 0 plus 16. Slots: r=0 "a" (n_a, k0+h), r=1 "b" (n_a, k0+8+h),
  r=2 "c" (n_b, k0+h), r=3 "d" (n_b, k0+8+h), k0 = 16st + 2t.
  So the quad's 4 lanes hold row n_a's 16 bytes (k 0..31 of the block) as a 4x4 byte transpose: lane t has bytes
  t, t+4 (st 0) and 8+t, 12+t (st 1).
* Codes are the checkpoint's codes except in blocks whose UE4M3 scale is zero (codes set to 0: 0 * 0 = 0, same
  product) or negative (code sign flipped: the FP4 path then uses |scale|, (-c) * |s| == c * s exactly). Qwen3.8-27B-
  NVFP4 has neither (all 1.52e9 block scales positive, scanned 26-09-2026), so there Wg is a pure bit permutation.
* K3's B_s (kernels.nvfp4_gemm_kernel, lowered): 128 x 256 fp4 per stage = 128-byte rows, 16-byte chunk c of row n
  stored at chunk c ^ (n & 7) (TMA SWIZZLE_128B, read back by ldmatrix with the same XOR); low nibble = even k.

Kernel: 256 threads, hand-written warp specialisation (the auto-WS K3 has the same split: 128 consumer + 128
producer threads). Consumer threads 0..127: exactly K3's T.mma_gemm_blockscaled on (A_s, B_s, SFA_s, SFB_s) and K3's
epilogue. Producer threads 128..255 (idle in K3 apart from issuing TMA) build B_s from Wg once per 256-wide K block
(K3 copies B once per residual pass):
* wg_tma=1 (default): the tile's 16 KB of Wg arrive by 32 bulk copies (cp.async.bulk, 512 B each, one mbarrier with
  expect_tx) into a staging buffer padded to 528 B per 32x32 block; thread (g = tid & 7, kc = (g >> 1) ^ (tid >> 3 & 7))
  reads its 4 lane chunks with conflict-free LDS.128. wg_tma=0: per-thread 16-byte global loads one K block ahead,
  K loop unrolled by 2 (measured 5-8 % slower: the LDG path costs more than the TMA path).
* hz_rf_unit: per lane quad and fragment word, a 4x4 byte transpose (8 PRMT) then every output bit field is one shift +
  one select over the 4 byte lanes: 38 instructions per 32 weights (the untwiddle-first order needed 80).
* stores: whole 16-byte B_s chunks at chunk kc ^ (row & 7) (conflict-free with the mapping above).
* sc_tma=1 (default): SFA / SFB tiles (2 KB each) by bulk copy on the same full barriers (expect_tx), no register or
  store traffic for scales. sc_tma=0: through registers.
* b_first=1: convert B(kb) before issuing that K block's A tiles (two consumer iterations of slack instead of one).
Measured cost drivers (26-09-2026, RTX PRO 6000 Max-Q, power-capped at 300 W): producer ALU work slows the MMA warps on
the same SM sub-partitions by roughly 1.3 % per 100 instructions per K block; the Wg load path and the B_s stores cost
a few % more. A 2-CTA cluster sharing one converted tile over DSMEM was tried and is 32-46 % slower (lockstep + DSMEM).
"""

from dataclasses import dataclass

import tilelang
import tilelang.language as T

from .kernels import SF_GRANULARITY_K

__all__ = ["NVFP4WeightRF", "prepare_layer", "prepare_weight_rf", "linear_rf", "get_kernel", "nvfp4_gemm_rf_kernel",
           "B_TILE_SRC"]

# ---------------------------------------------------------------------------------------------------------------
# Producer-side B tile builder (CUDA, injected as the kernel prelude): the untwiddle/transpose unit, the B_s store
# pattern and the bulk-copy / mbarrier helpers.
B_TILE_SRC = r"""
__device__ __forceinline__ unsigned hz_sel(unsigned t, unsigned m, unsigned r) { return (t & m) | (r & ~m); }
// One lane quad's word w (x_t = word w of lane 4g+t, t = 0..3) -> 4 canonical B_s words: slot a / b (row n_a, k0 and
// k0+8 pairs) and c / d (row n_b) of lanes t = 0..3, i.e. byte t = code(k0+2t) | code(k0+2t+1) << 4.
// First a 4x4 byte transpose (8 PRMT): tB holds byte B of x_0..x_3. Byte B of a twiddled word (bit i of the byte):
//   byte 0/2 (half h = 0/1): c.m c.e0 c.e1 b.m b.e0 b.e1 a.m a.e0      byte 1/3: a.e1 c.s d.e0 d.e1 b.s d.m d.s a.s
// then every output bit field is ONE shift + ONE select over all 4 byte lanes (38 instructions per 32 weights).
__device__ __forceinline__ void hz_rf_unit(unsigned x0, unsigned x1, unsigned x2, unsigned x3,
                                           unsigned &ta, unsigned &tb, unsigned &tc, unsigned &td) {
  const unsigned p = __byte_perm(x0, x1, 0x5140), q = __byte_perm(x2, x3, 0x5140);
  const unsigned p2 = __byte_perm(x0, x1, 0x7362), q2 = __byte_perm(x2, x3, 0x7362);
  const unsigned t0 = __byte_perm(p, q, 0x5410), t1 = __byte_perm(p, q, 0x7632);
  const unsigned t2 = __byte_perm(p2, q2, 0x5410), t3 = __byte_perm(p2, q2, 0x7632);
  unsigned a = t3;                                   // a.s(h1): byte bit 7 -> 7
  a = hz_sel(t0 >> 6, 0x03030303u, a);               // a.m a.e0 (h0)  6,7 -> 0,1
  a = hz_sel(t1 << 2, 0x04040404u, a);               // a.e1 (h0)      0   -> 2
  a = hz_sel(t1 >> 4, 0x08080808u, a);               // a.s (h0)       7   -> 3
  a = hz_sel(t2 >> 2, 0x30303030u, a);               // a.m a.e0 (h1)  6,7 -> 4,5
  a = hz_sel(t3 << 6, 0x40404040u, a);               // a.e1 (h1)      0   -> 6
  unsigned b = t0 >> 3;                              // b.m b.e0 b.e1 (h0) 3..5 -> 0..2
  b = hz_sel(t1 >> 1, 0x08080808u, b);               // b.s (h0)       4   -> 3
  b = hz_sel(t2 << 1, 0x70707070u, b);               // b.em (h1)      3..5 -> 4..6
  b = hz_sel(t3 << 3, 0x80808080u, b);               // b.s (h1)       4   -> 7
  unsigned c = t0;                                   // c.em (h0)      0..2 -> 0..2
  c = hz_sel(t1 << 2, 0x08080808u, c);               // c.s (h0)       1   -> 3
  c = hz_sel(t2 << 4, 0x70707070u, c);               // c.em (h1)      0..2 -> 4..6
  c = hz_sel(t3 << 6, 0x80808080u, c);               // c.s (h1)       1   -> 7
  unsigned d = t1 >> 5;                              // d.m (h0)       5   -> 0
  d = hz_sel(t1 >> 1, 0x06060606u, d);               // d.e0 d.e1 (h0) 2,3 -> 1,2
  d = hz_sel(t1 >> 3, 0x08080808u, d);               // d.s (h0)       6   -> 3
  d = hz_sel(t3 >> 1, 0x10101010u, d);               // d.m (h1)       5   -> 4
  d = hz_sel(t3 << 3, 0x60606060u, d);               // d.e0 d.e1 (h1) 2,3 -> 5,6
  d = hz_sel(t3 << 1, 0x80808080u, d);               // d.s (h1)       6   -> 7
  ta = a; tb = b; tc = c; td = d;
}
// q: this thread's 32 Wg words of one K block (2 groups x 4 lanes x 4 words); bs: B_s stage base (128 rows x 128
// bytes, 128B-swizzled). Producer thread tid (0..127) owns weight blocks nb = tid >> 6 and nb + 2, lane quad g, k32 block kc.
template <typename TQ, typename TB>
__device__ __forceinline__ void hz_rf_build_b(TQ *qp, TB *bp, int tid) {
  const unsigned *q = reinterpret_cast<const unsigned *>(qp);
  unsigned char *bs = reinterpret_cast<unsigned char *>(bp);
  // g = position in the quarter-warp, kc = (g >> 1) ^ quarter: the 8 threads of a quarter-warp hit 8 different bank
  // groups both in B_s (chunk kc ^ g) and in the padded Wg staging buffer (block stride 528 B: group kc + 4g + t)
  const int g = tid & 7, kc = (g >> 1) ^ ((tid >> 3) & 7), nb0 = tid >> 6;
  const int sw = (kc ^ g) << 4;
#pragma unroll
  for (int grp = 0; grp < 2; ++grp) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      unsigned ra[4], rb[4];
#pragma unroll
      for (int st = 0; st < 2; ++st) {
        const int w = 2 * st + j;
        hz_rf_unit(q[grp * 16 + 0 + w], q[grp * 16 + 4 + w], q[grp * 16 + 8 + w], q[grp * 16 + 12 + w],
                   ra[2 * st], ra[2 * st + 1], rb[2 * st], rb[2 * st + 1]);
      }
      const int na = (nb0 + 2 * grp) * 32 + 16 * j + g;
      *reinterpret_cast<uint4 *>(bs + na * 128 + sw) = make_uint4(ra[0], ra[1], ra[2], ra[3]);
      *reinterpret_cast<uint4 *>(bs + (na + 8) * 128 + sw) = make_uint4(rb[0], rb[1], rb[2], rb[3]);
    }
  }
}

// ---- wg_tma=1: Wg tile by 1D bulk TMA into a 16 KB staging buffer (one mbarrier, expect_tx), read back with LDS ----
__device__ __forceinline__ unsigned hz_saddr(const void *p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }
template <typename TB> __device__ __forceinline__ void hz_mbar_init(TB *bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(hz_saddr(bar)), "r"(count) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
template <typename TB> __device__ __forceinline__ void hz_mbar_wait(TB *bar, int parity) {
  const unsigned a = hz_saddr(bar);
  unsigned done;
  do {
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t}" : "=r"(done) : "r"(a), "r"(parity) : "memory");
  } while (!done);
}
// the 128 producer threads only (named barrier 5; TileLang itself uses barrier 0 here)
__device__ __forceinline__ void hz_bar_producers() { asm volatile("bar.sync 5, 128;" ::: "memory"); }
// producer warp 0: lane 0 expects 16384 bytes, then every lane copies one 512-byte weight block (nb = lane / 8,
// kc = lane % 8) into the staging buffer at (nb * 8 + kc) * 528 (16-byte pad per block: conflict-free LDS.128 reads)
template <typename TG, typename TW, typename TB>
__device__ __forceinline__ void hz_wg_issue(TG *src, TW *dst, TB *bar, int nb_stride_bytes, int lane) {
  const unsigned b = hz_saddr(bar), d = hz_saddr(dst);
  const char *g = reinterpret_cast<const char *>(src);
  if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(16384) : "memory");
  __syncwarp();
  const int nb = lane >> 3, kc = lane & 7;
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];"
               :: "r"(d + (nb * 8 + kc) * 528), "l"(g + (long long)nb * nb_stride_bytes + kc * 512), "r"(b) : "memory");
}
// one thread: +bytes expected on bar (no arrive), then a 1D bulk copy global -> shared that completes them
template <typename TS, typename TD, typename TB>
__device__ __forceinline__ void hz_bulk_tx(TS *src, TD *dst, TB *bar, int bytes) {
  const unsigned b = hz_saddr(bar);
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(hz_saddr(dst)), "l"(src), "r"(bytes), "r"(b) : "memory");
}
"""


# The producer/consumer hand-off is entirely mbarrier-based: TileLang's automatic __syncthreads insertion (which does
# not model mbarriers and would serialise the two warp groups every K block) is switched off.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def nvfp4_gemm_rf_kernel(N: int, K: int, passes: int = 2, out_dtype=T.bfloat16, a_stages: int = 2,
                          wg_tma: int = 1, sc_tma: int = 1, b_first: int = 0):
    """C (M, N) = alpha[n] * rsinv[m] * sum_p dq(A_p) @ dq(W)^T, W from regfrag's Wg. M dynamic, multiple of 128."""
    block_M = block_N = 128
    block_K = 256
    assert N % block_N == 0 and K % block_K == 0, (N, K)
    assert not sc_tma or wg_tma, "sc_tma needs wg_tma"
    words = block_K // 64
    k_blocks = K // block_K
    kc32 = K // 32
    M = T.dynamic("M")
    fp4, i32, u32, f32 = T.float4_e2m1fn, T.int32, T.uint32, T.float32

    @T.macro
    def load(Wg, SFA, SFB, kb, q, sfa, sfb, bx, by, g, kc, nb0, ptid):
        """Producer: K block kb's Wg words, A-scale rows and B-scale row of this thread into registers."""
        for grp in T.unroll(2):
            for t in T.unroll(4):
                for v in T.vectorized(4):
                    q[grp * 16 + t * 4 + v] = Wg[bx * 4 + nb0 + 2 * grp, kb * 8 + kc, g * 16 + t * 4 + v]
        for p in T.unroll(passes):
            for v in T.vectorized(words):
                sfa[p * words + v] = SFA[p, (by * k_blocks + kb) * block_M + ptid, v]
        for v in T.vectorized(words):
            sfb[v] = SFB[(bx * k_blocks + kb) * block_N + ptid, v]

    @T.macro
    def load_scales(SFA, SFB, kb, sfa, sfb, bx, by, ptid):
        for p in T.unroll(passes):
            for v in T.vectorized(words):
                sfa[p * words + v] = SFA[p, (by * k_blocks + kb) * block_M + ptid, v]
        for v in T.vectorized(words):
            sfb[v] = SFB[(bx * k_blocks + kb) * block_N + ptid, v]

    @T.macro
    def b_tile(SFB, B_s, SFB_s, b_full, b_empty, kb, q, sfb, bx, ptid):
        """Producer: B tile of K block kb (used by every pass) + its scale rows."""
        T.barrier_wait(b_empty[kb % 2], ((kb // 2) % 2) ^ 1)
        if sc_tma:
            if ptid == 0:
                T.call_extern("handle", "hz_bulk_tx", T.access_ptr(SFB[(bx * k_blocks + kb) * block_N, 0], "r"),
                              T.access_ptr(SFB_s[kb % 2, 0, 0], "w"), T.access_ptr(b_full[kb % 2], "rw"),
                              block_N * words * 4)
        T.call_extern("handle", "hz_rf_build_b", T.access_ptr(q, "r"), T.access_ptr(B_s[kb % 2, 0, 0], "w"), ptid)
        if not sc_tma:
            for v in T.vectorized(words):
                SFB_s[kb % 2, ptid, v] = sfb[v]
        T.barrier_arrive(b_full[kb % 2])

    @T.macro
    def produce(A, SFA, SFB, A_s, B_s, SFA_s, SFB_s, a_full, a_empty, b_full, b_empty, kb, q, sfa, sfb, bx, by, ptid):
        """Producer: A tile of pass 0 by TMA, then the B tile of K block kb (used by every pass), then the other
        passes' A tiles. Scale rows from registers, or (sc_tma) by bulk copy counted on the same full barrier."""
        if b_first:
            b_tile(SFB, B_s, SFB_s, b_full, b_empty, kb, q, sfb, bx, ptid)
        for p in T.unroll(passes):
            it = kb * passes + p
            T.barrier_wait(a_empty[it % a_stages], ((it // a_stages) % 2) ^ 1)
            T.tma_copy(A[by * block_M:(by + 1) * block_M, p * K + kb * block_K:p * K + (kb + 1) * block_K],
                       A_s[it % a_stages, :, :], barrier=a_full[it % a_stages])
            if sc_tma:
                if ptid == 0:
                    T.call_extern("handle", "hz_bulk_tx", T.access_ptr(SFA[p, (by * k_blocks + kb) * block_M, 0], "r"),
                                  T.access_ptr(SFA_s[it % a_stages, 0, 0], "w"),
                                  T.access_ptr(a_full[it % a_stages], "rw"), block_M * words * 4)
            else:
                for v in T.vectorized(words):
                    SFA_s[it % a_stages, ptid, v] = sfa[p * words + v]
            T.barrier_arrive(a_full[it % a_stages])
            if p == 0 and not b_first:
                b_tile(SFB, B_s, SFB_s, b_full, b_empty, kb, q, sfb, bx, ptid)

    @T.macro
    def consume(A_s, B_s, SFA_s, SFB_s, C_l, a_full, a_empty, b_full, b_empty, kb):
        """Consumer: exactly K3's block-scaled MMA on (A_s, B_s, SFA_s, SFB_s), pass order 0, 1, ..."""
        T.barrier_wait(b_full[kb % 2], (kb // 2) % 2)
        for p in T.unroll(passes):
            it = kb * passes + p
            T.barrier_wait(a_full[it % a_stages], (it // a_stages) % 2)
            T.mma_gemm_blockscaled(A_s[it % a_stages, :, :], B_s[kb % 2, :, :], C_l, SFA_s[it % a_stages, :, :],
                                   SFB_s[kb % 2, :, :], transpose_B=True, clear_accum=False, k_start=kb * block_K,
                                   sf_a_granularity_k=SF_GRANULARITY_K, sf_b_granularity_k=SF_GRANULARITY_K,
                                   sf_layout="blockscaled_chunk_kmajor")
            T.barrier_arrive(a_empty[it % a_stages])
        T.barrier_arrive(b_empty[kb % 2])

    @T.prim_func
    def main(
        A: T.Tensor((M, passes * K), fp4),
        Wg: T.Tensor((N // 32, kc32, 128), i32),
        SFA: T.Tensor((passes, M * k_blocks, words), u32),
        SFB: T.Tensor((N * k_blocks, words), u32),
        alpha: T.Tensor((N,), f32),
        rsinv: T.Tensor((M,), f32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(N // block_N, T.ceildiv(M, block_M), threads=256, prelude=B_TILE_SRC) as (bx, by):
            A_s = T.alloc_shared((a_stages, block_M, block_K), fp4)
            B_s = T.alloc_shared((2, block_N, block_K), fp4)
            SFA_s = T.alloc_shared((a_stages, block_M, words), u32)
            SFB_s = T.alloc_shared((2, block_N, words), u32)
            W_s = T.alloc_shared((4224 if wg_tma else 4,), i32)           # wg_tma: Wg staging (32 x 528 B)
            wbar = T.alloc_shared((1,), T.uint64)
            C_l = T.alloc_fragment((block_M, block_N), f32)
            # wg_tma=0: two register sets (K blocks alternate between them: loop unrolled by 2, no register copies);
            # wg_tma=1: q0 only (read from the staging buffer); sfa / sfb only without sc_tma
            q0 = T.alloc_local((32,), i32)
            q1 = T.alloc_local((32,), i32)
            sfa0 = T.alloc_local((passes * words,), u32)
            sfa1 = T.alloc_local((passes * words,), u32)
            sfb0 = T.alloc_local((words,), u32)
            sfb1 = T.alloc_local((words,), u32)
            a_full = T.alloc_barrier([128] * a_stages)
            a_empty = T.alloc_barrier([128] * a_stages)
            b_full = T.alloc_barrier([128, 128])
            b_empty = T.alloc_barrier([128, 128])

            tx = T.get_thread_binding()
            ptid = tx - 128                     # producer-local thread id
            g = ptid % 8                        # same thread mapping as hz_rf_build_b
            kc = (g // 2) ^ ((ptid // 8) % 8)
            nb0 = ptid // 64

            if wg_tma:
                if tx == 128:
                    T.call_extern("handle", "hz_mbar_init", T.access_ptr(wbar, "w"), 1)
                T.sync_threads()
                with T.ws(1):
                    if ptid < 32:
                        T.call_extern("handle", "hz_wg_issue", T.access_ptr(Wg[bx * 4, 0, 0], "r"),
                                      T.access_ptr(W_s, "w"), T.access_ptr(wbar, "rw"), kc32 * 512, ptid)
                    if not sc_tma:
                        load_scales(SFA, SFB, 0, sfa1, sfb1, bx, by, ptid)
                with T.ws(0):
                    T.clear(C_l)
                for kb in T.serial(k_blocks):
                    with T.ws(1):
                        # this K block's Wg tile: wait for the staging copy, read it, release it, start the next one
                        T.call_extern("handle", "hz_mbar_wait", T.access_ptr(wbar, "rw"), kb % 2)
                        for grp in T.unroll(2):
                            for t in T.unroll(4):
                                for v in T.vectorized(4):
                                    q0[grp * 16 + t * 4 + v] = W_s[((nb0 + 2 * grp) * 8 + kc) * 132 + g * 16 + t * 4 + v]
                        if not sc_tma:
                            for i in T.unroll(passes * words):
                                sfa0[i] = sfa1[i]
                            for i in T.unroll(words):
                                sfb0[i] = sfb1[i]
                        T.call_extern("handle", "hz_bar_producers")
                        if ptid < 32 and kb + 1 < k_blocks:
                            T.call_extern("handle", "hz_wg_issue", T.access_ptr(Wg[bx * 4, (kb + 1) * 8, 0], "r"),
                                          T.access_ptr(W_s, "w"), T.access_ptr(wbar, "rw"), kc32 * 512, ptid)
                        if not sc_tma:
                            load_scales(SFA, SFB, T.min(kb + 1, k_blocks - 1), sfa1, sfb1, bx, by, ptid)
                        produce(A, SFA, SFB, A_s, B_s, SFA_s, SFB_s, a_full, a_empty, b_full, b_empty, kb, q0, sfa0, sfb0, bx,
                                by, ptid)
                    with T.ws(0):
                        consume(A_s, B_s, SFA_s, SFB_s, C_l, a_full, a_empty, b_full, b_empty, kb)
            else:
                with T.ws(1):
                    load(Wg, SFA, SFB, 0, q0, sfa0, sfb0, bx, by, g, kc, nb0, ptid)
                with T.ws(0):
                    T.clear(C_l)
                # K block pairs: even blocks use register set 0, odd blocks set 1; each producer step first prefetches
                # the next block into the other set (clamped at the end: re-reads the last block, never past it)
                for kp in T.serial(k_blocks // 2):
                    with T.ws(1):
                        load(Wg, SFA, SFB, 2 * kp + 1, q1, sfa1, sfb1, bx, by, g, kc, nb0, ptid)
                        produce(A, SFA, SFB, A_s, B_s, SFA_s, SFB_s, a_full, a_empty, b_full, b_empty, 2 * kp, q0, sfa0, sfb0, bx,
                                by, ptid)
                    with T.ws(0):
                        consume(A_s, B_s, SFA_s, SFB_s, C_l, a_full, a_empty, b_full, b_empty, 2 * kp)
                    with T.ws(1):
                        load(Wg, SFA, SFB, T.min(2 * kp + 2, k_blocks - 1), q0, sfa0, sfb0, bx, by, g, kc, nb0, ptid)
                        produce(A, SFA, SFB, A_s, B_s, SFA_s, SFB_s, a_full, a_empty, b_full, b_empty, 2 * kp + 1, q1, sfa1, sfb1,
                                bx, by, ptid)
                    with T.ws(0):
                        consume(A_s, B_s, SFA_s, SFB_s, C_l, a_full, a_empty, b_full, b_empty, 2 * kp + 1)
                if k_blocks % 2:
                    with T.ws(1):
                        produce(A, SFA, SFB, A_s, B_s, SFA_s, SFB_s, a_full, a_empty, b_full, b_empty, k_blocks - 1, q0, sfa0, sfb0,
                                bx, by, ptid)
                    with T.ws(0):
                        consume(A_s, B_s, SFA_s, SFB_s, C_l, a_full, a_empty, b_full, b_empty, k_blocks - 1)

            with T.ws(0):
                for i, j in T.Parallel(block_M, block_N):
                    C_l[i, j] = C_l[i, j] * (alpha[bx * block_N + j] * rsinv[by * block_M + i])
                T.copy(C_l, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class NVFP4WeightRF:
    wg: object          # int32 (N/32, K/32, 128): decode_regfrag Prepared.Wg (shared, not a copy)
    sfb: object         # uint32 (N * K/256, 4), swizzled |scale| words (as kernels' prepare_weight)
    alpha: object       # fp32 (N,)
    n: int
    k: int


def _scales_alpha(weight_scale, weight_global_scale, shard_sizes, device):
    """K3's swizzled |scale| words and per-column alpha (as kernels' prepare_weight, sign bit dropped)."""
    import torch
    from . import _pack_scale_bytes, swizzle_scale_words
    n, k = weight_scale.shape[0], weight_scale.shape[1] * 16
    assert n % 128 == 0 and k % 256 == 0, (n, k)
    sfb = swizzle_scale_words(_pack_scale_bytes(weight_scale.view(torch.uint8) & 0x7F)).reshape(-1, 4)
    gs = weight_global_scale.float().reshape(-1)
    if gs.numel() == 1:
        alpha = (1.0 / gs).expand(n).contiguous()
    else:
        sizes = list(shard_sizes) if shard_sizes is not None else [n // gs.numel()] * gs.numel()
        assert sum(sizes) == n and len(sizes) == gs.numel(), (sizes, n, gs.numel())
        alpha = torch.cat([(1.0 / g).expand(sz) for g, sz in zip(gs, sizes)]).contiguous()
    return sfb, alpha.to(device)


def prepare_weight_rf(wg, weight_scale, weight_global_scale, shard_sizes=None) -> NVFP4WeightRF:
    """K3 weight handle WITHOUT packed weights of its own: B comes from regfrag's Wg (pass prep.Wg).

    Scales and alpha exactly as prepare_weight(), except that the scale sign bit is dropped: Wg already carries a
    negative block scale as flipped code signs (decode_regfrag.prepare), so (-c) * |s| == c * s."""
    import torch
    n, k = weight_scale.shape[0], weight_scale.shape[1] * 16
    assert tuple(wg.shape) == (n // 32, k // 32, 128) and wg.dtype == torch.int32, (tuple(wg.shape), n, k)
    sfb, alpha = _scales_alpha(weight_scale, weight_global_scale, shard_sizes, wg.device)
    return NVFP4WeightRF(wg, sfb, alpha, n, k)


def prepare_layer(weight_packed, weight_scale, weight_global_scale, shard_sizes=None):
    """ONE packed weight copy for decode and prefill: (decode_regfrag_xl Prepared, NVFP4WeightRF sharing its Wg).

    Same alpha as prepare_weight() (regfrag folds it the same way as before). Nothing returned references
    weight_packed, so the caller may free the checkpoint tensor afterwards."""
    import torch
    from . import decode_regfrag_xl
    n, k = weight_packed.shape[0], weight_packed.shape[1] * 2
    sfb, alpha = _scales_alpha(weight_scale, weight_global_scale, shard_sizes, weight_packed.device)
    prep = decode_regfrag_xl.prepare(weight_packed.view(torch.int8), weight_scale, alpha)
    return prep, NVFP4WeightRF(prep.Wg, sfb, alpha, n, k)


_KERNELS = {}


def get_kernel(n, k, passes=2, out_dtype=None, **opts):
    """Compiled nvfp4_gemm_rf_kernel for one layer shape (cached). opts: kernel options (a_stages, wg_tma, sc_tma,
    b_first); the defaults are the measured best."""
    import torch
    odt = torch.float32 if out_dtype == torch.float32 else torch.bfloat16
    key = (n, k, passes, odt) + tuple(sorted(opts.items()))
    kern = _KERNELS.get(key)
    if kern is None:
        kern = _KERNELS[key] = nvfp4_gemm_rf_kernel(n, k, passes, T.float32 if odt == torch.float32 else T.bfloat16,
                                                    **opts)
    return kern


def linear_rf(x, w: NVFP4WeightRF, passes: int = 2, out_dtype=None, **opts):
    """Same contract and bit-identical output as sm120_nvfp4.linear(x, w_checkpoint, passes)."""
    import torch
    from . import quantize
    assert x.dim() == 2 and x.shape[1] == w.k, (tuple(x.shape), w.k)
    m = x.shape[0]
    A, SFA, rsinv = quantize(x, passes)
    mp = A.shape[0]
    kern = get_kernel(w.n, w.k, passes, out_dtype, **opts)
    C = torch.empty(mp, w.n, dtype=torch.float32 if out_dtype == torch.float32 else torch.bfloat16, device=x.device)
    kern(A, w.wg, SFA.reshape(passes, mp * (w.k // 256), 4), w.sfb, w.alpha, rsinv, C)
    return C[:m]
