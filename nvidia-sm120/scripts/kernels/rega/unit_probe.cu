// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0

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

extern "C" __global__ void u19(const uint4 *__restrict__ in, uint2 *out) {
  uint4 v = in[threadIdx.x]; unsigned a, b; hz_rs_unit(v.x, v.y, a, b); unsigned c, d; hz_rs_unit(v.z, v.w, c, d);
  out[threadIdx.x] = make_uint2(a ^ c, b ^ d);
}
extern "C" __global__ void u17(const uint4 *__restrict__ in, uint2 *out) {
  uint4 v = in[threadIdx.x]; unsigned a, b; hz_rs_unit2(v.x, v.y, a, b); unsigned c, d; hz_rs_unit2(v.z, v.w, c, d);
  out[threadIdx.x] = make_uint2(a ^ c, b ^ d);
}
