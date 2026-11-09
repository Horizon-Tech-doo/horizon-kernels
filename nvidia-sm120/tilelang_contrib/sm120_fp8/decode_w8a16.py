# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""W8A16 GEMM for SM120: y = x @ W^T with BF16 activations x (M, K) and FP8 e4m3 weights W (N, K), 1 <= M <= 1024.

Purpose: FP8 checkpoints (NVIDIA ModelOpt per-tensor FP8 attention / GDN projections, per-channel FP8) at decode batch
sizes, where the GEMM is weight-stream bound. vLLM's path on this GPU is W8A8 (an activation-quantisation kernel, then a
CUTLASS FP8 GEMM); this kernel reads the SAME weight tensor (the checkpoint's row-major (N, K) e4m3 bytes: no second
copy, no re-layout) and keeps the activations in BF16 (no activation quantisation: more accurate than W8A8).

Numerics ("exact class"): every e4m3 weight is converted to BF16 exactly (e4m3 has 4 significant bits and exponents
2^-9 .. 2^8, all representable in BF16), products accumulate in fp32 on the tensor cores (mma.sync m16n8k16 bf16), and
one fp32 multiplier per output column (alpha: per-tensor, per-shard or per-channel weight scale) is applied before the
single BF16 rounding. Deterministic (fixed-order split-K reduction, no fp32 atomics), CUDA-graph capturable.

Data path per 32(n) x 32(k) weight block of a warp:
* cp.async streams the weight rows (row-major e4m3, 16-byte chunks) into a shared tile padded by 16 bytes per row, and
  the A rows into a BF16 tile padded by 16 elements per row (both paddings make the reads below bank-conflict free).
* Two ldmatrix.x4 (b16 view of the byte tile) give lane (g, t) the 4 bytes k = 4t .. 4t+3 of rows n = g (+8) for k
  halves 0..15 and 16..31: the FP8 m16n8k32 B-fragment order, NOT the BF16 m16n8k16 order (k = 2t, 2t+1, 2t+8, 2t+9).
  The k index of a GEMM can be permuted freely as long as A and B use the same permutation: for BF16 k16 step s the
  logical k (2t, 2t+1, 2t+8, 2t+9) is mapped to the physical k (16s+4t, +1, +2, +3). B needs no shuffle (bytes 0,1 ->
  register b0, bytes 2,3 -> b1) and A is read with two 64-bit shared loads per m16 tile (rows g and g+8, k 16s+4t..+3)
  instead of ldmatrix.
* e4m3 -> BF16 in registers, per two weights: PRMT (bytes to the high byte of each 16-bit half), shift + 2 masks (sign
  stays at bit 15, exponent/mantissa land at bits 10..4: a BF16 whose value is e4m3 * 2^-120, a BF16 subnormal for e4m3
  subnormals) and one mul.rn.bf16x2 by 2^120 (exact: power-of-two scaling of a <= 4-significant-bit value into the
  normal range). ~2.5 instructions per weight, 1 byte per weight streamed.
* Grid (m tiles, n tiles, split_k) as decode_regfrag_xl's tile kernel: TileLang T.Pipelined cp.async pipeline, the CTA
  covers block_M = warps_m * WM rows x block_N = warps_n * 32 * nb columns, deterministic split-K (fp32 partials +
  ticket, last CTA sums in fixed order), padding rows wrap to distinct valid rows (never stored).
"""

import tilelang
import tilelang.language as T

SUPPORTS_M = 1024
_CNT = 1 << 14          # split-K ticket counters per device
_NUM_SMS = 188
APAD = 16               # BF16 elements of padding per A row (64-bit A loads: rows g, g+8 hit distinct banks)
WPAD = 16               # bytes of padding per weight row (ldmatrix: 8 row addresses on distinct 16-byte slots)

_W8_SRC = r"""
__device__ __forceinline__ unsigned tl_w8_bf(unsigned y) {
  // y: two e4m3 codes in bits 8..15 and 24..31 -> bf16x2 of the exact e4m3 values
  const unsigned r = (y & 0x80008000u) | ((y >> 4) & 0x07F007F0u);   // value * 2^-120 (subnormal for e4m3 subnormals)
  unsigned d;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(r), "r"(0x7B807B80u));   // * 2^120, exact
  return d;
}
// B fragments of one warp's 32(n) x 32(k) block: wp = &W_s[n0, k0] (row stride `stride` bytes).
// out[8*s + 4*j + r] (16 bf16x2): k16 step s, n16 tile j; r = 0 (n_a, logical k 2t..2t+1), 1 (n_a, 2t+8..2t+9),
// 2 (n_b, 2t..), 3 (n_b, 2t+8..), n_a = 16j + lane/4, n_b = n_a + 8 -- decode_regfrag's register order.
template <typename TW, typename TO>
__device__ __forceinline__ void tl_w8_ldb(TW *wp, int stride, int lane, TO *op) {
  const unsigned char *w = reinterpret_cast<const unsigned char *>(wp);
  unsigned *out = reinterpret_cast<unsigned *>(op);
  const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const unsigned char *p = w + (16 * j + 8 * (mi >> 1) + rr) * stride + 16 * (mi & 1);
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    unsigned d0, d1, d2, d3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
    // d0 (n_a, k 4t..4t+3), d1 (n_a, k 16+4t..), d2 (n_b, k 4t..), d3 (n_b, k 16+4t..)
    out[4 * j + 0] = tl_w8_bf(__byte_perm(d0, 0u, 0x1404u));
    out[4 * j + 1] = tl_w8_bf(__byte_perm(d0, 0u, 0x3424u));
    out[4 * j + 2] = tl_w8_bf(__byte_perm(d2, 0u, 0x1404u));
    out[4 * j + 3] = tl_w8_bf(__byte_perm(d2, 0u, 0x3424u));
    out[8 + 4 * j + 0] = tl_w8_bf(__byte_perm(d1, 0u, 0x1404u));
    out[8 + 4 * j + 1] = tl_w8_bf(__byte_perm(d1, 0u, 0x3424u));
    out[8 + 4 * j + 2] = tl_w8_bf(__byte_perm(d3, 0u, 0x1404u));
    out[8 + 4 * j + 3] = tl_w8_bf(__byte_perm(d3, 0u, 0x3424u));
  }
}
// A fragment of one m16 tile for k16 step at ap = &A_s[row0, k0 + 16 s] (bf16, row stride `stride` elements), in the
// permuted k order above: a0 = (g, k 4t..4t+1), a1 = (g+8, 4t..), a2 = (g, 4t+2..4t+3), a3 = (g+8, 4t+2..).
template <typename TA, typename TO>
__device__ __forceinline__ void tl_w8_lda(TA *ap, int stride, int lane, TO *op) {
  const unsigned short *a = reinterpret_cast<const unsigned short *>(ap);
  unsigned *out = reinterpret_cast<unsigned *>(op);
  const int g = lane >> 2, t = lane & 3;
  const uint2 v0 = *reinterpret_cast<const uint2 *>(a + g * stride + 4 * t);
  const uint2 v1 = *reinterpret_cast<const uint2 *>(a + (g + 8) * stride + 4 * t);
  out[0] = v0.x; out[1] = v1.x; out[2] = v0.y; out[3] = v1.y;
}
"""


def smem_shapes(warps_m, WM, warps_n, nb, block_K):
    """Shared buffers the main loop needs (one pipeline stage each; T.Pipelined multi-buffers them):
    A_s (block_M, block_K + APAD) bf16 and W_s (block_N, block_K + WPAD) int8."""
    return (warps_m * WM, block_K + APAD), (warps_n * 32 * nb, block_K + WPAD)


def w8_mainloop(K: int, warps_m: int, WM: int, warps_n: int, nb: int, block_K: int, num_stages: int, k_tiles: int):
    """T.macro body(x, W, M, m0, n0, kt0, tid, A_s, W_s, C_l): C_l (fp32 local, mt*nb*16 per thread) += the
    (block_M x block_N) tile at rows m0.., cols n0.. over k tiles [kt0, kt0 + k_tiles) (block_K each).
    K4 contract: executed by all warps_m*warps_n*32 threads of one CTA (tid = thread index); A_s / W_s from
    smem_shapes(); no grid-level assumptions; ends after the pipeline drains (the caller syncs before reusing A_s /
    W_s for another task). Padding rows (>= M) read row (m0 + i) % M and must not be stored by the caller."""
    mt = WM // 16
    WN = 32 * nb
    kchunks = block_K // 32
    block_M = warps_m * WM
    block_N = warps_n * WN
    AS = block_K + APAD
    WS = block_K + WPAD
    bf16 = T.bfloat16

    @T.macro
    def body(x, W, M, m0, n0, kt0, tid, A_s, W_s, C_l):
        B_l = T.alloc_local((nb * 32,), bf16)
        A_l = T.alloc_local((mt * 8,), bf16)
        warp = tid // 32
        lane = tid % 32
        wm = warp % warps_m
        wn = warp // warps_m
        for kt in T.Pipelined(k_tiles, num_stages=num_stages):
            k0 = (kt0 + kt) * block_K
            for i, kk in T.Parallel(block_M, block_K):
                A_s[i, kk] = x[(m0 + i) % M, k0 + kk]
            for i, kk in T.Parallel(block_N, block_K):
                W_s[i, kk] = W[n0 + i, k0 + kk]
            for c in T.unroll(kchunks):
                for b in T.unroll(nb):
                    T.call_extern("handle", "tl_w8_ldb", T.access_ptr(W_s[wn * WN + b * 32, c * 32], "r"), WS,
                                  lane, T.access_ptr(B_l[b * 32], "w", extent=32))
                for s in T.unroll(2):
                    for i in T.unroll(mt):
                        T.call_extern("handle", "tl_w8_lda",
                                      T.access_ptr(A_s[wm * WM + i * 16, c * 32 + s * 16], "r"), AS, lane,
                                      T.access_ptr(A_l[i * 8], "w", extent=8))
                    for i in T.unroll(mt):
                        for b in T.unroll(nb):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8, C_l.data,
                                          ((i * nb + b) * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, b * 32 + (s * 2 + j) * 8 + 4, C_l.data,
                                          ((i * nb + b) * 2 + j) * 8 + 4, T.bool(False))

    return body


def w8_store(warps_m: int, WM: int, warps_n: int, nb: int, partial: bool = False):
    """T.macro store(Y, alpha, C_l, M, m0, n0, tid, sk): Y[row, col] = bf16(C_l * alpha[col]) for the tile's rows < M;
    partial=True: Y is the fp32 split-K buffer (split_k, M, N) and the tile goes to Y[sk, row, col] unrounded."""
    mt = WM // 16
    WN = 32 * nb

    @T.macro
    def store(Y, alpha, C_l, M, m0, n0, tid, sk):
        warp = tid // 32
        lane = tid % 32
        wm = warp % warps_m
        wn = warp // warps_m
        # C_l[((i*nb+b)*2+j)*8 + r] sits at row 16i + 8*((r%4)//2) + lane//4,
        #                                   col 32b + 16j + 8*(r//4) + 2*(lane%4) + r%2   (warp-relative)
        for i in T.unroll(mt):
            for b in T.unroll(nb):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = m0 + wm * WM + i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = n0 + wn * WN + b * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M:
                            if partial:
                                for v in T.vectorized(2):
                                    Y[sk, grow, gcol + v] = C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]
                            else:
                                for v in T.vectorized(2):
                                    Y[grow, gcol + v] = T.cast(
                                        C_l[((i * nb + b) * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], T.bfloat16)

    return store


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True})
def fp8_w8a16_kernel(N: int, K: int, warps_m: int, WM: int, warps_n: int, nb: int, block_K: int, split_k: int,
                     num_stages: int, min_blocks: int = 1):
    """Standalone kernel: grid (m tiles, n tiles, split_k) around w8_mainloop / w8_store; deterministic split-K."""
    assert WM % 16 == 0 and nb in (1, 2) and block_K % 32 == 0
    block_M = warps_m * WM
    block_N = warps_n * 32 * nb
    threads = 32 * warps_m * warps_n
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    mt = WM // 16
    a_shape, w_shape = smem_shapes(warps_m, WM, warps_n, nb, block_K)
    mainloop = w8_mainloop(K, warps_m, WM, warps_n, nb, block_K, num_stages, kt_per_split)
    store = w8_store(warps_m, WM, warps_n, nb, partial=False)
    store_p = w8_store(warps_m, WM, warps_n, nb, partial=True)
    M = T.dynamic("M")
    bf16, f32, i32, i8 = T.bfloat16, T.float32, T.int32, T.int8
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
        W: T.Tensor((N, K), i8),
        alpha: T.Tensor((N,), f32),
        P: T.Tensor(P_shape, f32),
        counters: T.Tensor((_CNT,), i32),
    ):
        with T.Kernel(T.ceildiv(M, block_M), n_tiles, split_k, threads=threads, prelude=_W8_SRC) as (bm, bx, bk):
            if min_blocks > 1:
                T.annotate_min_blocks_per_sm(min_blocks)
            A_s = T.alloc_shared(a_shape, bf16)
            W_s = T.alloc_shared(w_shape, i8)
            ticket_s = T.alloc_shared((1,), i32)
            C_l = T.alloc_local((mt * nb * 2 * 8,), f32)
            tid = T.get_thread_binding()
            m0 = bm * block_M
            T.clear(C_l)
            mainloop(x, W, M, m0, bx * block_N, bk * kt_per_split, tid, A_s, W_s, C_l)
            if split_k == 1:
                store(Y, alpha, C_l, M, m0, bx * block_N, tid, 0)
            else:
                store_p(P, alpha, C_l, M, m0, bx * block_N, tid, bk)
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
# Host side.
class Prepared:
    """A W8A16 weight handle: W is a VIEW of the checkpoint's (N, K) e4m3 tensor as int8 (no copy)."""
    __slots__ = ("N", "K", "W", "alpha", "device")

    def __init__(self, N, K, W, alpha):
        self.N, self.K, self.W, self.alpha = N, K, W, alpha
        self.device = W.device


_COUNTERS = {}
_DUMMY = {}
_KERNELS = {}


def _counters(device):
    import torch
    c = _COUNTERS.get(device)
    if c is None:
        c = _COUNTERS[device] = torch.zeros(_CNT, dtype=torch.int32, device=device)
    return c


def prepare(weight_fp8, scale):
    """weight_fp8: (N, K) torch.float8_e4m3fn (row-major, contiguous; NOT copied); scale: fp32 scalar (per tensor) or
    (N,) (per output column: per channel, or a per-shard scale expanded to its columns). y = (x @ w_e4m3^T) * scale."""
    import torch
    assert weight_fp8.dtype == torch.float8_e4m3fn and weight_fp8.dim() == 2, (weight_fp8.dtype, weight_fp8.shape)
    w = weight_fp8 if weight_fp8.is_contiguous() else weight_fp8.contiguous()
    N, K = w.shape
    assert N % 128 == 0 and K % 64 == 0, (N, K)
    s = torch.as_tensor(scale, dtype=torch.float32, device=w.device).reshape(-1)
    alpha = (s.expand(N) if s.numel() == 1 else s).to(torch.float32).contiguous()
    assert alpha.numel() == N, (alpha.shape, N)
    _counters(w.device)
    return Prepared(N, K, w.view(torch.int8), alpha)


def dequantize_reference(prep):
    """Exact fp32 dequantised weight (N, K)."""
    import torch
    return prep.W.view(torch.float8_e4m3fn).float() * prep.alpha[:, None]


def _choose_split(n_tiles, k_tiles, max_ctas=2 * _NUM_SMS, min_ctas=_NUM_SMS):
    divs = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0]
    ok = [d for d in divs if n_tiles * d <= max_ctas]
    if not ok:
        return 1
    enough = [d for d in ok if n_tiles * d >= min_ctas]
    return enough[0] if enough else ok[-1]


def smem_bytes(cfg):
    warps_m, WM, warps_n, nb, block_K, split_k, num_stages = cfg[:7]
    return num_stages * (warps_m * WM * (block_K + APAD) * 2 + warps_n * 32 * nb * (block_K + WPAD)) + 4


# Tuned buckets: (N, K) -> [(M_max, cfg), ...] ascending (filled from w8_tune sweeps); others use _generic_config.
_TUNED = {}


def _generic_config(N, K, M):
    """(warps_m, WM, warps_n, nb, block_K, split_k, num_stages[, min_blocks])."""
    if M <= 16:
        warps_m, WM = 1, 16
    elif M <= 32:
        warps_m, WM = 1, 32
    elif M <= 64:
        warps_m, WM = 1, 64
    else:
        warps_m, WM = 2, 64
    warps_n, nb = 4, 1
    block_K = 128 if M <= 32 and K % 128 == 0 else 64
    block_N = warps_n * 32 * nb
    if N % block_N:
        warps_n = 2
        block_N = 64
    cfg0 = (warps_m, WM, warps_n, nb, block_K, 1, 1)
    per_stage = smem_bytes(cfg0)
    num_stages = max(2, min(4, (48 * 1024) // per_stage)) if M <= 32 else max(2, min(4, (96 * 1024) // per_stage))
    tiles = -(-M // (warps_m * WM)) * (N // block_N)
    split_k = _choose_split(tiles, K // block_K)
    return (warps_m, WM, warps_n, nb, block_K, split_k, num_stages)


def config(N, K, M):
    for m_max, cfg in _TUNED.get((N, K), ()):
        if M <= m_max:
            return cfg
    return _generic_config(N, K, M)


def linear(x, prep, cfg=None):
    """y = x @ W^T * alpha (bf16 out) for 1 <= M <= 1024 (larger M in 1024-row chunks); deterministic, capturable.

    Split-K launches use a per-device ticket buffer: launches of one device must not run concurrently on two streams."""
    import torch
    M, K = x.shape
    assert K == prep.K, (x.shape, prep.N, prep.K)
    if M > SUPPORTS_M:
        return torch.cat([linear(x[i:i + SUPPORTS_M], prep, cfg) for i in range(0, M, SUPPORTS_M)])
    N = prep.N
    if cfg is None:
        cfg = config(N, K, M)
    warps_m, WM, warps_n, nb, block_K, split_k, num_stages = cfg[:7]
    min_blocks = cfg[7] if len(cfg) > 7 else 1
    key = (N, K) + tuple(cfg[:7]) + (min_blocks,)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = fp8_w8a16_kernel(N, K, warps_m, WM, warps_n, nb, block_K, split_k, num_stages,
                                                  min_blocks)
    x = x.contiguous()
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if split_k > 1:
        assert -(-M // (warps_m * WM)) * (N // (warps_n * 32 * nb)) <= _CNT
        P = torch.empty(split_k, M, N, dtype=torch.float32, device=x.device)
    else:
        P = _DUMMY.get(x.device)
        if P is None:
            P = _DUMMY[x.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=x.device)
    kernel(x, y, prep.W, prep.alpha, P, _counters(x.device))
    return y
