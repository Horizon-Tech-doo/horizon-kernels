# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 decode-GEMM candidate "v2smem": the shared-memory design (packed ring -> bit-trick dequant -> bf16 chunk -> T.gemm).

Contract: see decode_contract.py.  W4A16, NVFP4 weights, 1 <= M <= 64, exact class (bf16 dequant, fp32 accumulate,
one rounding), deterministic (split-K partials are summed in a fixed order by the last-arriving CTA of each N tile;
the only atomic is an int32 arrival ticket), CUDA-graph capturable (no host syncs; the ticket counters reset
themselves; workspaces come from torch.empty).

Design (one CTA = one 128-row weight tile x one K range, ALL tokens, 256 threads = 8 warps):

  orientation  y^T = W x^T: the dequantised weight chunk is the MMA *A* operand (m16 = 16 weight rows) and the tokens
               are the MMA N dimension, so the token tile Mt steps in 8s (8, 16, ..., 64).  M <= 8 issues half the
               mma.sync of a 16-row token tile, M = 33 costs 40/64 of a 64-row tile (tensor work matters from M ~ 40).
  prepare()    re-lays the checkpoint ONCE (untimed) so each 64-k pipeline stage of a CTA is one contiguous chunk:
               Wp[tile, stage] = 256 threads x 16 B (thread t: weight row t//2, k half t%2, 32 fp4) = 4 KB, with the
               nibbles twiddled into the bit positions the dequant below extracts; Sp[tile, stage] = 128 rows x 4 block
               scales (512 B) re-encoded as biased E5M3 so that 0x7000 | b << 4 is exactly bf16(scale * 2^119).
  pipeline     hand-written cp.async ring of NS stages (W 4 KB + S 512 B + x slice Mt x 64 bf16 per stage), ONE commit
               group per stage; automatic barrier insertion is disabled and the barriers are placed by hand:
               - 2 CTAs/SM tiles (Mt <= 48, <= 49 KB smem, <= 128 regs): NS-1 stages in flight, 2 barriers per stage:
                   wait(i) ; bar ; issue i+NS-1 into the slot of i-1 ; dequant i -> Wb ; bar ; gemm(Wb, x_s[i])
                 (the other CTA on the SM overlaps its dequant with this CTA's mma and vice versa)
               - 1 CTA/SM tiles (Mt >= 56, tensor-bound): double-buffered Wb, NS-2 stages in flight, 1 barrier per stage:
                   wait(i) ; bar ; issue i+NS-2 into the slot of i-2 ; gemm(Wb[(i-1)%2], x_s[i-1]) ; dequant i -> Wb[i%2]
                 (mma of one stage and dequant of the next share one basic block, so ALU and tensor work interleave)
  dequant      per thread per stage: one 16-B lds + one 4-B lds, 11 integer ops per 8 weights (shift / LOP3, the
               sign and the 3 exponent+mantissa bits of each E2M1 code land on bf16 bits 15 and 8..6, i.e. fp4 * 2^-126
               exactly, subnormal 0.5 included) + one mul.rn.bf16x2 per 2 weights by the scale (fp4 * scale * 2^-7,
               exact: <= 6 significant bits), 6 ops per 2 block scales, then four 16-B sts into the swizzled 128 x 64
               bf16 chunk that T.gemm reads with ldmatrix.  The 2^7 is folded into alpha (fp32, exact).
  mma          T.gemm(Wb[128 x 64], x_s[Mt x 64], C[128 x Mt], transpose_B) -> mma.sync m16n8k16 bf16 -> fp32,
               FullRow policy: warp w owns weight rows 16w..16w+15 for all Mt tokens (Wb is read exactly once).
  epilogue     split_k == 1: y[m, n] = bf16(C[n, m] * alpha[n]).  split_k > 1: fp32 partials P[s, m, n]; the last
               CTA of the N tile (int32 ticket) sums P[0..S-1] in fixed order and writes bf16.
  split-K      chosen per (N, K, Mt) by choose_split() (precomputed in prepare()); uneven splits are allowed.
"""

import torch
import tilelang
import tilelang.language as T

SUPPORTS_M = 64
BLOCK_N = 128           # weight rows per CTA
BLOCK_K = 64            # k per pipeline stage
THREADS = 256
NUM_SMS = 188

# token tile Mt -> ring stages, CTAs per SM, double-buffered bf16 chunk.
#   smem = NS * (4608 + 128 * Mt) + (1 or 2) * 16384 bytes.  SM120: 99 KB per CTA, 100 KB per SM (1 KB reserved per
#   CTA), so 2 CTAs/SM need <= 49 KB each (and <= 128 registers/thread: __launch_bounds__(256, 2)).
MT_CFG = {
    8: dict(ns=5, ctas_per_sm=2, db=False),     # 43.5 KB, 4 stages in flight per CTA
    16: dict(ns=5, ctas_per_sm=2, db=False),    # 48.5 KB, 4
    24: dict(ns=4, ctas_per_sm=2, db=False),    # 46.0 KB, 3
    32: dict(ns=3, ctas_per_sm=2, db=False),    # 41.5 KB, 2
    40: dict(ns=3, ctas_per_sm=2, db=False),    # 44.5 KB, 2
    48: dict(ns=3, ctas_per_sm=2, db=False),    # 47.5 KB, 2
    56: dict(ns=5, ctas_per_sm=1, db=True),     # 89.5 KB, 3
    64: dict(ns=5, ctas_per_sm=1, db=True),     # 94.5 KB, 3
}
SCALE_C = 7                      # dequantised weights are fp4 * scale * 2^-SCALE_C; alpha carries 2^SCALE_C
SPLIT_OVERRIDE: dict = {}        # (N, K, Mt) -> split_k, for tuning without code changes

_DQ_SRC = r"""
// bf16x2 multiply (round to nearest; bf16 arithmetic keeps subnormal inputs).
__device__ __forceinline__ unsigned v2s_bmul(unsigned a, unsigned b) {
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// 4 twiddled words (32 E2M1 codes of one weight row, 32 consecutive k) + the thread's two block scales
// (E5M3 bytes 2h, 2h+1 of sw) -> 16 bf16x2 = fp4 * scale * 2^-7 exactly, k-ordered (low half = even k).
// Word layout (per 16-bit half; the high half is the same +16 and holds the odd k):
//   out j=0: s@15 e1@8  e0@7  m@6     -> v & M
//   out j=1: s@12 e1@5  e0@4  m@3     -> (v << 3) & M
//   out j=2: s@14 e1@11 e0@10 m@9     -> ((v << 1) & S) | ((v >> 3) & G)
//   out j=3: s@13 e1@2  e0@1  m@0     -> ((v << 2) & S) | ((v << 6) & G)
// With sign at bit 15 and e1 e0 m at bits 8..6 the bf16 is fp4 * 2^-126 (e = 0 gives the subnormal 0.5 * m).
template <typename TW, typename TO>
__device__ __forceinline__ void v2s_dq32(const TW *w_in, TO *out, unsigned sw, int h) {
  const unsigned *w = reinterpret_cast<const unsigned *>(w_in);
  unsigned *o = reinterpret_cast<unsigned *>(out);
  const unsigned sel = 0x4040u + (unsigned)h * 0x0202u;
  const unsigned s0 = (__byte_perm(sw, 0u, sel) << 4) | 0x70007000u;           // bf16x2 (scale0 * 2^119)
  const unsigned s1 = (__byte_perm(sw, 0u, sel + 0x0101u) << 4) | 0x70007000u;  // bf16x2 (scale1 * 2^119)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned v = w[i];
    const unsigned s = (i < 2) ? s0 : s1;
    const unsigned r0 = v & 0x81C081C0u;
    const unsigned r1 = (v << 3) & 0x81C081C0u;
    const unsigned r2 = ((v << 1) & 0x80008000u) | ((v >> 3) & 0x01C001C0u);
    const unsigned r3 = ((v << 2) & 0x80008000u) | ((v << 6) & 0x01C001C0u);
    o[4 * i + 0] = v2s_bmul(r0, s);
    o[4 * i + 1] = v2s_bmul(r1, s);
    o[4 * i + 2] = v2s_bmul(r2, s);
    o[4 * i + 3] = v2s_bmul(r3, s);
  }
}
"""

_PASS_CFG = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,   # barriers are placed by hand (see the loops)
    # every global access is in bounds by construction (x rows >= M are predicated off AND clamped), so the automatic
    # bounds predicates on the cp.async sources / epilogue stores are pure overhead
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}


@tilelang.jit(pass_configs=_PASS_CFG)
def v2smem_kernel(Mt: int, NS: int, db: bool = False, min_blocks: int = 1, threads: int = THREADS):
    assert Mt % 8 == 0 and 8 <= Mt <= 64 and threads == 256
    assert NS >= (4 if db else 3)
    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")
    NW = T.dynamic("NW")      # rows of the prepared weight / scale tensors (= N/128 * K/64)
    CT = T.dynamic("CT")      # N / 128 ticket counters
    S = T.dynamic("S")        # split_k (P.shape[0]); P is a (1, 1, 1) dummy when S == 1
    PM = T.dynamic("PM")
    PN = T.dynamic("PN")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    D = NS - 2 if db else NS - 1                      # stages in flight ahead of the one being dequantised
    NB = 2 if db else 1                               # bf16 chunk buffers
    x_chunks = Mt * 8                                 # 16-B chunks of one x slice (Mt rows x 128 B)
    x_rounds = (x_chunks + threads - 1) // threads
    out_rounds = (Mt * BLOCK_N) // threads
    policy = T.GemmWarpPolicy.FullRow

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Wp: T.Tensor((NW, 1024), i32),
        Sp: T.Tensor((NW, 128), i32),
        alpha: T.Tensor((N,), f32),
        counters: T.Tensor((CT,), i32),
        Y: T.Tensor((M, N), bf16),
        P: T.Tensor((S, PM, PN), f32),
    ):
        with T.Kernel(N // BLOCK_N, S, threads=threads) as (bx, bk):
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            T.import_source(_DQ_SRC)
            W_s = T.alloc_shared((NS, 1024), i32)
            S_s = T.alloc_shared((NS, 128), i32)
            X_s = T.alloc_shared((NS, Mt, BLOCK_K), bf16)
            Wb = T.alloc_shared((NB, BLOCK_N, BLOCK_K), bf16)
            ticket = T.alloc_shared((1,), i32)
            w_l = T.alloc_local((4,), i32)
            o_l = T.alloc_local((32,), bf16)
            C = T.alloc_fragment((BLOCK_N, Mt), f32)
            tx = T.get_thread_binding()

            kst = K // BLOCK_K                            # k stages of the whole row
            cnt = ((bk + 1) * kst) // S - (bk * kst) // S  # this CTA's stage count (uneven splits allowed)
            # per-CTA scalars live in registers (as expressions they were re-derived, with a 64-bit division, in
            # every address computation of the loop)
            kb0 = T.alloc_var(i32, init=(bk * kst) // S)  # first k stage of this CTA
            wrow0 = T.alloc_var(i32, init=bx * kst + kb0)  # prepared chunk row of stage kb0
            T.clear(C)

            # ---- prologue: stages 0 .. D-1 (one commit group each, possibly empty)
            for p in T.unroll(D):
                if p < cnt:
                    T.ptx_cp_async(T.access_ptr(W_s[p, tx * 4], "w", 4),
                                   T.access_ptr(Wp[wrow0 + p, tx * 4], "r", 4), 4)
                    if tx < 32:
                        T.ptx_cp_async(T.access_ptr(S_s[p, tx * 4], "w", 4),
                                       T.access_ptr(Sp[wrow0 + p, tx * 4], "r", 4), 4)
                    for c in T.unroll(x_rounds):
                        if c * threads + tx < x_chunks:
                            T.ptx_cp_async(
                                T.access_ptr(X_s[p, (c * threads + tx) // 8, ((c * threads + tx) % 8) * 8], "w", 8),
                                T.access_ptr(x[T.min((c * threads + tx) // 8, M - 1),
                                               (kb0 + p) * BLOCK_K + ((c * threads + tx) % 8) * 8], "r", 8),
                                8, predicate=(c * threads + tx) // 8 < M)
                T.ptx_commit_group()

            if db:
                # ---- 1 barrier per stage: iteration i multiplies stage i-1 and dequantises stage i into Wb[i%2].
                # peeled iteration 0 (dequant only)
                T.ptx_wait_group(D - 1)
                T.sync_threads()
                if D < cnt:
                    T.ptx_cp_async(T.access_ptr(W_s[D % NS, tx * 4], "w", 4),
                                   T.access_ptr(Wp[wrow0 + D, tx * 4], "r", 4), 4)
                    if tx < 32:
                        T.ptx_cp_async(T.access_ptr(S_s[D % NS, tx * 4], "w", 4),
                                       T.access_ptr(Sp[wrow0 + D, tx * 4], "r", 4), 4)
                    for c in T.unroll(x_rounds):
                        if c * threads + tx < x_chunks:
                            T.ptx_cp_async(
                                T.access_ptr(X_s[D % NS, (c * threads + tx) // 8, ((c * threads + tx) % 8) * 8], "w", 8),
                                T.access_ptr(x[T.min((c * threads + tx) // 8, M - 1),
                                               (kb0 + D) * BLOCK_K + ((c * threads + tx) % 8) * 8], "r", 8),
                                8, predicate=(c * threads + tx) // 8 < M)
                T.ptx_commit_group()
                for v in T.vectorized(4):
                    w_l[v] = W_s[0, tx * 4 + v]
                T.call_extern("handle", "v2s_dq32", T.access_ptr(w_l, "r"), T.access_ptr(o_l, "w"),
                              S_s[0, tx // 2], tx % 2)
                for j in T.unroll(4):
                    for v in T.vectorized(8):
                        Wb[0, tx // 2, (tx % 2) * 32 + j * 8 + v] = o_l[j * 8 + v]

                for i in T.serial(1, cnt):
                    # stage i landed (D-1 younger groups may be in flight) and is visible; every thread finished
                    # iteration i-1, i.e. dequant(i-1) wrote Wb[(i-1)%2] and gemm(i-2) stopped reading Wb[i%2] and
                    # x_s[(i-2)%NS] (the slot refilled now: (i+D)%NS == (i-2)%NS).
                    T.ptx_wait_group(D - 1)
                    T.sync_threads()
                    if i + D < cnt:
                        T.ptx_cp_async(T.access_ptr(W_s[(i + D) % NS, tx * 4], "w", 4),
                                       T.access_ptr(Wp[wrow0 + i + D, tx * 4], "r", 4), 4)
                        if tx < 32:
                            T.ptx_cp_async(T.access_ptr(S_s[(i + D) % NS, tx * 4], "w", 4),
                                           T.access_ptr(Sp[wrow0 + i + D, tx * 4], "r", 4), 4)
                        for c in T.unroll(x_rounds):
                            if c * threads + tx < x_chunks:
                                T.ptx_cp_async(
                                    T.access_ptr(X_s[(i + D) % NS, (c * threads + tx) // 8,
                                                     ((c * threads + tx) % 8) * 8], "w", 8),
                                    T.access_ptr(x[T.min((c * threads + tx) // 8, M - 1),
                                                   (kb0 + i + D) * BLOCK_K + ((c * threads + tx) % 8) * 8], "r", 8),
                                    8, predicate=(c * threads + tx) // 8 < M)
                    T.ptx_commit_group()
                    # mma of stage i-1 first: nvcc cannot prove that the Wb[i%2] stores below do not alias these
                    # ldmatrix reads of Wb[(i-1)%2], so in the other order every HMMA would wait for the stores; in
                    # this order the dequant ALU work of stage i fills the HMMA latencies.
                    T.gemm(Wb[(i - 1) % 2, :, :], X_s[(i - 1) % NS, :, :], C, transpose_B=True, policy=policy)
                    for v in T.vectorized(4):
                        w_l[v] = W_s[i % NS, tx * 4 + v]
                    T.call_extern("handle", "v2s_dq32", T.access_ptr(w_l, "r"), T.access_ptr(o_l, "w"),
                                  S_s[i % NS, tx // 2], tx % 2)
                    for j in T.unroll(4):
                        for v in T.vectorized(8):
                            Wb[i % 2, tx // 2, (tx % 2) * 32 + j * 8 + v] = o_l[j * 8 + v]
                T.sync_threads()                          # Wb[(cnt-1)%2] complete
                T.gemm(Wb[(cnt - 1) % 2, :, :], X_s[(cnt - 1) % NS, :, :], C, transpose_B=True, policy=policy)
            else:
                # ---- 2 barriers per stage, NS-1 stages in flight
                for i in T.serial(cnt):
                    # stage i has landed (NS-2 younger groups may still be in flight) and is visible to every thread;
                    # every thread also finished stage i-1 (its dequant read W_s/S_s[(i-1)%NS], its gemm read Wb and
                    # X_s[(i-1)%NS]), so that slot and Wb may be overwritten.
                    T.ptx_wait_group(D - 1)
                    T.sync_threads()
                    if i + D < cnt:
                        T.ptx_cp_async(T.access_ptr(W_s[(i + D) % NS, tx * 4], "w", 4),
                                       T.access_ptr(Wp[wrow0 + i + D, tx * 4], "r", 4), 4)
                        if tx < 32:
                            T.ptx_cp_async(T.access_ptr(S_s[(i + D) % NS, tx * 4], "w", 4),
                                           T.access_ptr(Sp[wrow0 + i + D, tx * 4], "r", 4), 4)
                        for c in T.unroll(x_rounds):
                            if c * threads + tx < x_chunks:
                                T.ptx_cp_async(
                                    T.access_ptr(X_s[(i + D) % NS, (c * threads + tx) // 8,
                                                     ((c * threads + tx) % 8) * 8], "w", 8),
                                    T.access_ptr(x[T.min((c * threads + tx) // 8, M - 1),
                                                   (kb0 + i + D) * BLOCK_K + ((c * threads + tx) % 8) * 8], "r", 8),
                                    8, predicate=(c * threads + tx) // 8 < M)
                    T.ptx_commit_group()
                    # dequant: thread tx = weight row tx//2, k half tx%2 of this stage
                    for v in T.vectorized(4):
                        w_l[v] = W_s[i % NS, tx * 4 + v]
                    T.call_extern("handle", "v2s_dq32", T.access_ptr(w_l, "r"), T.access_ptr(o_l, "w"),
                                  S_s[i % NS, tx // 2], tx % 2)
                    for j in T.unroll(4):
                        for v in T.vectorized(8):
                            Wb[0, tx // 2, (tx % 2) * 32 + j * 8 + v] = o_l[j * 8 + v]
                    T.sync_threads()                      # Wb complete
                    T.gemm(Wb[0, :, :], X_s[i % NS, :, :], C, transpose_B=True, policy=policy)

            if S == 1:
                for n, m in T.Parallel(BLOCK_N, Mt):
                    if m < M:
                        Y[m, bx * BLOCK_N + n] = T.cast(C[n, m] * alpha[bx * BLOCK_N + n], bf16)
            else:
                for n, m in T.Parallel(BLOCK_N, Mt):
                    if m < M:
                        P[bk, m, bx * BLOCK_N + n] = C[n, m]
                # last-arriving CTA of this N tile reduces the partials in fixed split order
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tx == 0:
                    ticket[0] = T.atomic_add(counters[bx], 1, return_prev=True)
                T.sync_threads()
                if ticket[0] == S - 1:
                    T.call_extern("handle", "__threadfence")
                    for e in T.unroll(out_rounds):
                        if (e * threads + tx) // BLOCK_N < M:
                            acc = T.alloc_var(f32, init=T.call_extern(
                                "float32", "__ldcg",
                                T.access_ptr(P[0, (e * threads + tx) // BLOCK_N,
                                               bx * BLOCK_N + (e * threads + tx) % BLOCK_N], "r")))
                            for s in T.serial(1, S):
                                acc = acc + T.call_extern(
                                    "float32", "__ldcg",
                                    T.access_ptr(P[s, (e * threads + tx) // BLOCK_N,
                                                   bx * BLOCK_N + (e * threads + tx) % BLOCK_N], "r"))
                            Y[(e * threads + tx) // BLOCK_N, bx * BLOCK_N + (e * threads + tx) % BLOCK_N] = T.cast(
                                acc * alpha[bx * BLOCK_N + (e * threads + tx) % BLOCK_N], bf16)
                    if tx == 0:
                        counters[bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# prepare (untimed): twiddled weight chunks, E5M3 scale chunks, alpha * 2^7, ticket counters, split-K table

# (sign, e1, e0, m) bit positions in the 16-bit half for output j of a word (see _DQ_SRC)
_POS = ((15, 8, 7, 6), (12, 5, 4, 3), (14, 11, 10, 9), (13, 2, 1, 0))
_PREP_ROWS = 2048


def _twiddle_rows(nib: torch.Tensor) -> torch.Tensor:
    """nib (R, K) uint8 E2M1 codes (R % 128 == 0, K % 64 == 0) -> (R/128 * K/64, 1024) int32 chunks."""
    R, K = nib.shape
    nt, ks = R // BLOCK_N, K // BLOCK_K
    # k = stage*64 + h*32 + i*8 + j*2 + p ; row = tile*128 + r
    v = nib.view(nt, BLOCK_N, ks, 2, 4, 4, 2).to(torch.int32)          # (nt, r, ks, h, i, j, p)
    half = torch.zeros(nt, BLOCK_N, ks, 2, 4, 2, dtype=torch.int32, device=nib.device)
    for j, (ps, pe1, pe0, pm) in enumerate(_POS):
        c = v[:, :, :, :, :, j, :]                                     # (nt, r, ks, h, i, p)
        half |= (((c >> 3) & 1) << ps) | (((c >> 2) & 1) << pe1) | (((c >> 1) & 1) << pe0) | ((c & 1) << pm)
    word = half[..., 0] | (half[..., 1] << 16)                         # (nt, r, ks, h, i) ; int32 wraps as bits
    word = word.permute(0, 2, 1, 3, 4).contiguous()                    # (nt, ks, r, h, i): thread t = 2r + h
    return word.view(nt * ks, 1024)


def _e5m3(scale_abs: torch.Tensor) -> torch.Tensor:
    """|UE4M3| scales (float32, exact) -> uint8 b with bf16_bits(0x7000 | b << 4) == scale * 2^119 (0 for scale 0)."""
    b16 = (scale_abs * (2.0 ** (126 - SCALE_C))).to(torch.bfloat16).view(torch.int16).to(torch.int32)
    pos = scale_abs > 0
    if bool(pos.any()):
        bp = b16[pos]
        assert int(bp.min()) >= 0x7000 and int(bp.max()) <= 0x7FF0 and int((bp & 0xF).max()) == 0, \
            "block scale outside the E5M3 re-encoding range"
    b = torch.where(pos, (b16 - 0x7000) >> 4, torch.zeros_like(b16))
    return b.to(torch.uint8)


class Prepared:
    __slots__ = ("N", "K", "Wp", "Sp", "alpha", "counters", "split")

    def __init__(self, N, K, Wp, Sp, alpha, counters, split):
        self.N, self.K, self.Wp, self.Sp, self.alpha, self.counters, self.split = N, K, Wp, Sp, alpha, counters, split


def prepare(packed_int8, scale_fp8, alpha_fp32) -> Prepared:
    N, K = packed_int8.shape[0], packed_int8.shape[1] * 2
    assert N % BLOCK_N == 0 and K % 256 == 0, (N, K)
    assert scale_fp8.shape == (N, K // 16), (scale_fp8.shape, N, K)
    dev = packed_int8.device
    u = packed_int8.contiguous().view(torch.uint8)
    sf = scale_fp8.float()
    Wp = torch.empty(N // BLOCK_N * (K // BLOCK_K), 1024, dtype=torch.int32, device=dev)
    per = K // BLOCK_K
    for r0 in range(0, N, _PREP_ROWS):
        r1 = min(N, r0 + _PREP_ROWS)
        ub = u[r0:r1]
        nib = torch.stack((ub & 0xF, ub >> 4), dim=-1).reshape(r1 - r0, K)     # low nibble = even k
        blk = nib.view(r1 - r0, K // 16, 16)
        s = sf[r0:r1]
        blk[s == 0] = 0                                  # zero block scale -> zero weights (scale byte is then free)
        neg = s < 0                                      # (UE4M3 is unsigned; tolerate a sign bit by flipping codes)
        if bool(neg.any()):
            blk[neg] ^= 0x8
        Wp[(r0 // BLOCK_N) * per:(r1 // BLOCK_N) * per] = _twiddle_rows(nib)
    b = _e5m3(sf.abs())                                                        # (N, K/16) uint8
    Sp = b.view(N // BLOCK_N, BLOCK_N, K // BLOCK_K, 4).permute(0, 2, 1, 3).contiguous()
    Sp = Sp.view(N // BLOCK_N * (K // BLOCK_K), 512).view(torch.int32)
    alpha = (alpha_fp32.float() * (2.0 ** SCALE_C)).contiguous()
    assert alpha.shape == (N,), alpha.shape
    counters = torch.zeros(N // BLOCK_N, dtype=torch.int32, device=dev)   # split-K tickets (self-resetting; one
    split = {mt: choose_split(N, K, mt) for mt in MT_CFG}                  # linear() in flight per Prepared at a time)
    _dummy(dev)                              # created here so a first call under CUDA-graph capture allocates nothing
    return Prepared(N, K, Wp, Sp, alpha, counters, split)


# ---------------------------------------------------------------------------------------------------------------
# dispatch

def token_tile(M: int) -> int:
    return (M + 7) // 8 * 8


# mma.sync m16n8k16 per 64-k stage of one CTA = 4 * Mt; at 285 TFLOPS that is ~26 * Mt SM cycles, against ~1220 cycles
# to stream the stage's 4.6 KB at the SM's share of 1.63 TB/s -> tensor/memory time ratio per stage:
_TENSOR_PER_STAGE = 26.0 / 1220.0


def choose_split(N: int, K: int, Mt: int) -> int:
    """split_k per (layer shape, token tile), uneven splits allowed, >= 8 k stages per split.

    Memory-bound tiles (Mt <= 32, 2 CTAs/SM, every CTA of the grid resident at once): split only until all SMs have
    work (a narrow layer on 40 tiles would otherwise leave 148 SMs idle and under-fill the memory pipe).
    Tensor-heavier tiles: minimise, in units of one stage streamed at the SM's DRAM share,
        max(memory: tiles*stages/188, tensor: most-loaded SM = ceil(CTAs/188) * stages/split * r)
        + 2 per residency wave (pipeline fill) + split-K reduction (2 + split*Mt/64)."""
    key = (N, K, Mt)
    if key in SPLIT_OVERRIDE:
        return SPLIT_OVERRIDE[key]
    tiles, stages = N // BLOCK_N, K // BLOCK_K
    per_sm = MT_CFG[Mt]["ctas_per_sm"]
    max_split = max(1, min(16, stages // 8))
    if Mt <= 32:
        if tiles >= NUM_SMS:
            return 1
        return max(1, min(NUM_SMS * per_sm // tiles, max_split))
    r = _TENSOR_PER_STAGE * Mt
    mem = tiles * stages / NUM_SMS
    best, best_t = 1, None
    for s in range(1, max_split + 1):
        ctas = tiles * s
        tensor = -(-ctas // NUM_SMS) * -(-stages // s) * r
        waves = -(-ctas // (NUM_SMS * per_sm))
        t = max(mem, tensor) + 2.0 * waves + (0.0 if s == 1 else 2.0 + s * Mt / 64.0)
        if best_t is None or t < best_t - 1e-9:
            best, best_t = s, t
    return best


_KERNELS: dict = {}
_DUMMY: dict = {}


def _dummy(dev):
    """(1, 1, 1) fp32 stand-in for the split-K workspace when split_k == 1 (the kernel never touches it)."""
    dev = torch.device(dev)
    P = _DUMMY.get(dev)
    if P is None:
        P = _DUMMY[dev] = torch.zeros(1, 1, 1, dtype=torch.float32, device=dev)
    return P


def _kernel(Mt: int):
    k = _KERNELS.get(Mt)
    if k is None:
        cfg = MT_CFG[Mt]
        k = _KERNELS[Mt] = v2smem_kernel(Mt, cfg["ns"], cfg["db"], cfg["ctas_per_sm"])
    return k


def linear(x: torch.Tensor, p: Prepared, split_k: int = None) -> torch.Tensor:
    """y = x @ W^T (bf16 (M, N)) for 1 <= M <= 64."""
    M, K = x.shape
    assert K == p.K and 1 <= M <= SUPPORTS_M, (tuple(x.shape), p.K)
    Mt = token_tile(M)
    s = split_k if split_k is not None else p.split[Mt]
    Y = torch.empty(M, p.N, dtype=torch.bfloat16, device=x.device)
    P = torch.empty(s, M, p.N, dtype=torch.float32, device=x.device) if s > 1 else _dummy(x.device)
    if not x.is_contiguous() or x.data_ptr() % 16:        # cp.async moves 16-B chunks of x rows
        x = x.contiguous().clone()
    _kernel(Mt)(x, p.Wp, p.Sp, p.alpha, p.counters, Y, P)
    return Y
