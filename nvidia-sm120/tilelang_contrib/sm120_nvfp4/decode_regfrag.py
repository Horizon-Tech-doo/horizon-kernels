# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 candidate "regfrag": W4A16 decode GEMM (BF16 x NVFP4, M <= 64) with dequantisation into REGISTER B fragments.

Design (contract: decode_contract.py):

* prepare() re-lays the packed FP4 weights ONCE into 32(n) x 32(k) blocks of 512 bytes ordered like the mma.sync
  m16n8k16 B fragments of one warp: lane l owns 16 consecutive bytes = the 32 weights it feeds to the tensor core for
  two n16 tiles x two k16 steps. Inside each 32-bit word the 8 nibbles are "twiddled" (bits scattered, see _SLOT_POS)
  so that three of the four bf16x2 fragment registers come out of ONE shift + ONE mask (x & M, (x<<3) & M, (x<<6) & M,
  M = 0x81C081C0) and the fourth out of three shifts + three LOP3. Each result is a bf16x2 whose value is exactly
  fp4 * 2^-126 (e2m1 codes with e=0 become bf16 subnormals, i.e. 0 and 0.5*2^-126).
* The UE4M3 block scales are re-encoded ONCE as one byte b with bf16(0x7000 | b << 4) == scale * 2^119 (S0E5M3 style,
  exact for every e4m3 value incl. subnormals), stored WITHOUT replication (64 bytes per 32x32 block: the NVFP4
  payload, 12.5% on top of the nibbles). Per lane, one PRMT + one IMAD turn a scale word into bf16x2 pairs
  {scale(n_a), scale(n_b)}; ONE mul.rn.bf16x2 per fragment register (the pair half is broadcast by the HMUL2 operand
  swizzle .H0_H0/.H1_H1, free) gives fp4 * scale * 2^-7 exactly (<= 6 significant bits, >= 2^-17: normal BF16).
  alpha * 2^7 is applied once in the fp32 epilogue. Zero scales zero their codes, negative scales flip code signs.
* The kernel streams packed blocks + scales + the A tile into shared memory with cp.async inside T.Pipelined
  (num_stages deep) and never writes a BF16 weight tile anywhere: each lane does one 128-bit ld.shared of its 16 weight
  bytes and one 64-bit ld.shared of its 8 scale bytes (quad-broadcast), dequantises in registers and issues mma.sync
  with that B fragment. A (all M rows) is read with ldmatrix from a swizzled tile; rows >= M re-read row M-1 (clamped
  index), so no address past the end of x is ever formed; those accumulator rows are simply never stored.
* Narrow layers use split-K over grid.y. Each split writes an fp32 partial tile; the LAST CTA to finish a tile (atomic
  ticket, threadfence-reduction pattern) sums the split_k partials in fixed order 0..split_k-1 and writes BF16:
  deterministic, no fp32 atomics on data; tickets reset themselves, so the launch is CUDA-graph replayable.

Measured SASS (sm_120a, steady-state k-tile loop, per lane per 64 weights): block_M=16: 208 instructions = 3.25 per
weight, of which dequant 136 = 2.1 per weight (32 HMUL2.BF16_V2, ~48 LOP3, ~48 SHF/IMAD.SHL, 8 PRMT + 8 LEA for scale
pairs), 16 HMMA, 4 LDS, 4 LDSM, 4 LDGSTS, 2 BAR, ~40 address/loop. block_M=64: 305 (64 HMMA). No spills (72 regs at
block_M 16/32, 148 at 48, 204 at 64). The block_M=16 loop of the previous version of this file spent ~4.1 instructions
per weight on dequant alone and streamed 4x replicated scales (33% more DRAM bytes).

Why T.ptx_mma + emitter.ldmatrix_a rather than T.gemm(A_shared, B_fragment): it is the same is_gemm_sr data path
(A via ldmatrix from shared, B from registers), but the B fragment can then be produced by packed-word bit tricks
(the per-element T.Parallel fill of a fragment costs ~10 instructions per weight).
"""

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

SUPPORTS_M = 64

# ---------------------------------------------------------------------------------------------------------------
# Register dequantisation of one lane's chunk: 4 twiddled weight words + 2 scale words -> 16 bf16x2 B registers.
#   weight word w = 2*st + j: k16 step st (0,1), n16 tile j (0,1) of the lane's 32x32 block.
#   out[4*w + r]: r=0 (n_a, k0..k0+1)  r=1 (n_a, k0+8..k0+9)  r=2 (n_b, k0..k0+1)  r=3 (n_b, k0+8..k0+9)
#       with n_a = 16j + lane/4, n_b = n_a + 8, k0 = 16*st + 2*(lane%4)  (mma.m16n8k16 B layout, two n8 tiles)
#   scale word st, byte 2*x + j = scale code of (n = 16j + 8x + lane/4, k16 step st); p_j = {code(j,a), code(j,b)}.
_DQ_SRC = r"""
__device__ __forceinline__ unsigned tl_rf_mul_lo(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {l, l};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
__device__ __forceinline__ unsigned tl_rf_mul_hi(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {h, h};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
template <typename TQ, typename TS, typename TO>
__device__ __forceinline__ void tl_rf_dq_chunk(TQ *qp, TS *sp, TO *op) {
  const unsigned *q = reinterpret_cast<const unsigned *>(qp);
  const unsigned *s = reinterpret_cast<const unsigned *>(sp);
  unsigned *out = reinterpret_cast<unsigned *>(op);
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const unsigned sw = s[st];
    // bf16x2 {scale(n_a), scale(n_b)} * 2^119 for n16 tile j = 0 (bytes 0, 2) and j = 1 (bytes 1, 3)
    const unsigned p0 = __byte_perm(sw, 0u, 0x4240u) * 16u + 0x70007000u;
    const unsigned p1 = __byte_perm(sw, 0u, 0x4341u) * 16u + 0x70007000u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned x = q[2 * st + j];
      const unsigned p = j ? p1 : p0;
      const unsigned ra = x & 0x81C081C0u;
      const unsigned rb = (x << 3) & 0x81C081C0u;
      const unsigned rc = (x << 6) & 0x81C081C0u;
      const unsigned rd = ((x << 1) & 0x80008000u) | ((x >> 3) & 0x01800180u) | ((x >> 7) & 0x00400040u);
      out[8 * st + 4 * j + 0] = tl_rf_mul_lo(ra, p);
      out[8 * st + 4 * j + 1] = tl_rf_mul_lo(rb, p);
      out[8 * st + 4 * j + 2] = tl_rf_mul_hi(rc, p);
      out[8 * st + 4 * j + 3] = tl_rf_mul_hi(rd, p);
    }
  }
}
"""

# Bit positions (sign, e1, e0, m) of the nibble feeding fragment register r (slot a/b/c/d above), half h (0 = low
# bf16 = even k, 1 = high bf16 = odd k). Slots a, b, c: target positions shifted by 0, 3, 6; slot d: the rest.
_SLOT_POS = {
    0: ((15, 8, 7, 6), (31, 24, 23, 22)),
    1: ((12, 5, 4, 3), (28, 21, 20, 19)),
    2: ((9, 2, 1, 0), (25, 18, 17, 16)),
    3: ((14, 11, 10, 13), (30, 27, 26, 29)),
}
_ALPHA_FOLD = 128.0     # the kernel accumulates fp4 * scale * 2^-7


# ---------------------------------------------------------------------------------------------------------------
# Kernel. Grid = (N / block_N, split_k); block_N = 32 * n_warps; one warp owns 32 columns for all M rows.
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def nvfp4_regfrag_kernel(N: int, K: int, block_M: int, n_warps: int, block_K: int, split_k: int, num_stages: int):
    assert block_M in (16, 32, 48, 64) and n_warps in (1, 2, 4, 8) and block_K % 32 == 0
    block_N = 32 * n_warps
    threads = 32 * n_warps
    kchunks = block_K // 32          # 32-wide k chunks per k tile (one 16-byte fragment load per lane each)
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    warp_rows = block_M // 16
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=1, block_col_warps=n_warps,
                                      warp_row_tiles=block_M, warp_col_tiles=32, chunk=block_K)
    P_shape = (split_k, M, N) if split_k > 1 else (1, 1, 1)

    def _partial_sum(P, i, col):
        """Fixed-order (0, 1, ..., split_k-1) fp32 sum of the split-K partials: deterministic."""
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
        counters: T.Tensor((N // 32,), i32),
    ):
        with T.Kernel(n_tiles, split_k, threads=threads, prelude=_DQ_SRC) as (bx, bk):
            T.assume(M >= 1)  # v0.1.15 (#3294): M >= 1 is the launch contract; keeps the unguarded loads
            A_s = T.alloc_shared((block_M, block_K), bf16)
            Wq_s = T.alloc_shared((n_warps, kchunks, 128), i32)
            S_s = T.alloc_shared((n_warps, kchunks, 16), i32)
            ticket_s = T.alloc_shared((1,), i32)
            q_l = T.alloc_local((4,), i32)
            s_l = T.alloc_local((2,), i32)
            B_l = T.alloc_local((32,), bf16)                    # 2 k16 steps x 2 n16 tiles x 8 values
            A_l = T.alloc_local((warp_rows * 8,), bf16)         # one k16 step of A for all m16 tiles
            C_l = T.alloc_local((warp_rows * 2 * 8,), f32)      # warp_rows m16 x 2 n16 accumulators
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})

            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            T.clear(C_l)

            for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
                kc0 = (bk * kt_per_split + kt) * kchunks
                # A tile: rows >= M re-read row M-1 (clamped: no address past the end of x is ever formed); those
                # accumulator rows are never stored. A pure global->shared store keeps this an async (cp.async) stage.
                for i, kk in T.Parallel(block_M, block_K):
                    A_s[i, kk] = x[T.min(i, M - 1), kc0 * 32 + kk]
                T.copy(Wg[bx * n_warps, kc0, 0], Wq_s)
                T.copy(Sg[bx * n_warps, kc0, 0], S_s)
                for c in T.unroll(kchunks):
                    for v in T.vectorized(4):
                        q_l[v] = Wq_s[warp, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[v] = S_s[warp, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l, "r"), T.access_ptr(s_l, "r"),
                                  T.access_ptr(B_l, "w"))
                    for s in T.unroll(2):
                        emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                        for i in T.unroll(warp_rows):
                            for j in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, (s * 2 + j) * 8, C_l.data, (i * 2 + j) * 8, T.bool(False))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data,
                                          i * 8, B_l.data, (s * 2 + j) * 8 + 4, C_l.data, (i * 2 + j) * 8 + 4,
                                          T.bool(False))

            # epilogue: C_l[(i*2+j)*8 + r] sits at row 16i + 8*((r%4)//2) + lane//4, col 16j + 8*(r//4) + 2*(lane%4) + r%2
            for i in T.unroll(warp_rows):
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        grow = i * 16 + 8 * (r2 % 2) + lane // 4
                        gcol = bx * block_N + warp * 32 + j * 16 + 8 * (r2 // 2) + 2 * (lane % 4)
                        if grow < M:
                            if split_k == 1:
                                for v in T.vectorized(2):
                                    Y[grow, gcol + v] = T.cast(C_l[(i * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v], bf16)
                            else:
                                for v in T.vectorized(2):
                                    P[bk, grow, gcol + v] = C_l[(i * 2 + j) * 8 + r2 * 2 + v] * alpha[gcol + v]

            if split_k > 1:
                # threadfence-reduction: the last CTA of this N tile reduces the partials in fixed order.
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(counters[bx], 1, return_prev=True)
                T.sync_threads()
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    for i, j in T.Parallel(block_M, block_N):
                        if i < M:
                            Y[i, bx * block_N + j] = T.cast(_partial_sum(P, i, bx * block_N + j), bf16)
                    if tid == 0:
                        counters[bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# Host side: layout preparation and dispatch.
_NUM_SMS = 188


def _to_int32(w64):
    """int64 holding uint32 bit patterns -> int32 with the same bits."""
    import torch
    return torch.where(w64 >= 2 ** 31, w64 - 2 ** 32, w64).to(torch.int32)


def _fragment_maps(device):
    """Index maps of one 32x32 block.

    w_idx (1024,): nibble stream order [lane][word][element e] -> n_in * 32 + k_in  (e = 2r + h, see _DQ_SRC)
    s_idx (64,):   scale byte order [g][st][x][j]               -> n_in * 2 + kb"""
    import torch
    l = torch.arange(32).view(32, 1, 1)
    w = torch.arange(4).view(1, 4, 1)
    e = torch.arange(8).view(1, 1, 8)
    g, t = l // 4, l % 4
    st, j = w // 2, w % 2
    r, h = e // 2, e % 2
    n_in = 16 * j + g + 8 * (r // 2)
    k_in = 16 * st + 2 * t + h + 8 * (r % 2)
    w_idx = (n_in * 32 + k_in).reshape(-1)
    b = torch.arange(64)
    gb, sb, xb, jb = b // 8, (b // 4) % 2, (b // 2) % 2, b % 2
    s_idx = (16 * jb + 8 * xb + gb) * 2 + sb
    return w_idx.to(device), s_idx.to(device)


class Prepared:
    __slots__ = ("N", "K", "Wg", "Sg", "alpha", "counters", "device")

    def __init__(self, N, K, Wg, Sg, alpha, counters):
        self.N, self.K, self.Wg, self.Sg, self.alpha, self.counters = N, K, Wg, Sg, alpha, counters
        self.device = Wg.device


def _encode_scales(scale_fp8):
    """UE4M3 (N, K/16) -> (byte codes b with bf16(0x7000 | b<<4) == |scale| * 2^119, zero mask, negative mask)."""
    import torch
    sc = scale_fp8.float()
    assert bool(torch.isfinite(sc).all()), "NaN/Inf block scales are not supported"
    zero = sc == 0
    neg = torch.signbit(sc) & ~zero
    bits = (sc.abs() * (2.0 ** 119)).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    nz = ~zero
    assert bool(((bits >> 12)[nz] == 7).all()) and bool(((bits & 0xF)[nz] == 0).all()), "scale re-encoding not exact"
    code = ((bits >> 4) & 0xFF).masked_fill(zero, 0).to(torch.uint8)
    return code, zero, neg


def prepare(packed_int8, scale_fp8, alpha_fp32):
    """Re-layout packed FP4 (N, K/2) + UE4M3 scales (N, K/16) into warp-fragment-ordered 32x32 blocks (untimed)."""
    import torch
    N, K = packed_int8.shape[0], packed_int8.shape[1] * 2
    assert N % 128 == 0 and K % 256 == 0, (N, K)
    assert tuple(scale_fp8.shape) == (N, K // 16), scale_fp8.shape
    dev = packed_int8.device
    w_idx, s_idx = _fragment_maps(dev)
    scode, zero, neg = _encode_scales(scale_fp8)

    Wg = torch.empty(N // 32, K // 32, 128, dtype=torch.int32, device=dev)
    rows = max(32, ((1 << 22) // K) // 32 * 32)              # bounded temporaries (~4M codes per slab)
    u8 = packed_int8.view(torch.uint8)
    for n0 in range(0, N, rows):
        n1 = min(N, n0 + rows)
        u = u8[n0:n1]
        nib = torch.stack(((u & 0xF), (u >> 4)), dim=-1).reshape(n1 - n0, K)     # nib[n, k] = e2m1 code
        # exact scale folding: zero-scale blocks -> code 0, negative scales -> flipped code sign
        nib = nib ^ (neg[n0:n1].repeat_interleave(16, 1).to(torch.uint8) << 3)
        nib = nib.masked_fill(zero[n0:n1].repeat_interleave(16, 1), 0)
        nb = (n1 - n0) // 32
        blk = nib.view(nb, 32, K // 32, 32).permute(0, 2, 1, 3).reshape(nb, K // 32, 1024)
        stream = blk[:, :, w_idx].view(nb, K // 32, 32, 4, 8).to(torch.int64)       # [lane][word][e]
        word = torch.zeros(nb, K // 32, 32, 4, dtype=torch.int64, device=dev)
        for e in range(8):
            ps, p1, p0, pm = _SLOT_POS[e // 2][e % 2]
            c = stream[..., e]
            word |= ((c >> 3) & 1) << ps
            word |= ((c >> 2) & 1) << p1
            word |= ((c >> 1) & 1) << p0
            word |= (c & 1) << pm
        Wg[n0 // 32:n1 // 32] = _to_int32(word).view(nb, K // 32, 128)
    sblk = scode.view(N // 32, 32, K // 32, 2).permute(0, 2, 1, 3).reshape(N // 32, K // 32, 64)
    Sg = sblk[:, :, s_idx].contiguous().view(torch.int32).contiguous()            # (N/32, K/32, 16)
    counters = torch.zeros(N // 32, dtype=torch.int32, device=dev)
    alpha = (alpha_fp32.to(torch.float32) * _ALPHA_FOLD).contiguous()
    return Prepared(N, K, Wg.contiguous(), Sg, alpha, counters)


def _choose_split(n_tiles, k_tiles, max_ctas=2 * _NUM_SMS, min_ctas=_NUM_SMS):
    """Smallest divisor of k_tiles giving >= min_ctas CTAs, else the largest one keeping <= max_ctas (else 1)."""
    divs = [d for d in range(1, k_tiles + 1) if k_tiles % d == 0]
    ok = [d for d in divs if n_tiles * d <= max_ctas]
    if not ok:
        return 1
    enough = [d for d in ok if n_tiles * d >= min_ctas]
    return enough[0] if enough else ok[-1]


def _choose_split_balanced(n_tiles, k_tiles, max_split=8, max_ctas=4 * _NUM_SMS):
    """Tensor-bound buckets: the divisor of k_tiles that best balances CTAs over the SMs.

    Every CTA does the same work, so the busiest SM runs ceil(C / SMs) of them: efficiency (C/SMs) / ceil(C/SMs).
    Ties go to the smaller split (less fp32 partial traffic)."""
    import math
    best, best_eff = 1, -1.0
    for d in range(1, min(max_split, k_tiles) + 1):
        c = n_tiles * d
        if k_tiles % d or (d > 1 and c > max_ctas):
            continue
        eff = (c / _NUM_SMS) / math.ceil(c / _NUM_SMS)
        if eff > best_eff + 1e-9:
            best, best_eff = d, eff
    return best


def config(N, K, M):
    """(block_M, n_warps, block_K, split_k, num_stages) for one layer shape and token count.

    M <= 32 is weight-stream bound: >= 188 co-resident CTAs (one wave), 256 columns per CTA on wide layers, 3 k tiles
    in flight (4 stages; 3 for block_M 32 with 8 warps so two 39 KB CTAs fit per SM). M > 32 is tensor bound (the
    MMA work is block_M rows whatever M is): 128 columns per CTA and a split-K that balances CTAs over the SMs."""
    block_M = 16 if M <= 16 else (32 if M <= 32 else (48 if M <= 48 else 64))
    block_K = 64
    W = N // 32
    if block_M <= 32:
        n_warps = 8 if W >= 256 else 4
        while W % n_warps:
            n_warps //= 2
        split_k = _choose_split(W // n_warps, K // block_K)
        num_stages = 3 if (block_M == 32 and n_warps == 8) else 4
    else:
        n_warps = 4
        while W % n_warps:
            n_warps //= 2
        split_k = _choose_split_balanced(W // n_warps, K // block_K)
        num_stages = 3
    return block_M, n_warps, block_K, split_k, num_stages


_DUMMY = {}
_KERNELS = {}


def linear(x, prep, cfg=None):
    """y = x @ W^T (bf16 out) for 1 <= M <= 64 tokens; deterministic, CUDA-graph capturable.

    cfg optionally overrides config(): (block_M, n_warps, block_K, split_k, num_stages).
    Note: split-K launches use prep.counters as tickets, so one Prepared must not run concurrently on two streams."""
    import torch
    M, K = x.shape
    N = prep.N
    assert K == prep.K and 1 <= M <= SUPPORTS_M, (x.shape, prep.N, prep.K)
    if cfg is None:
        cfg = config(N, K, M)
    block_M, n_warps, block_K, split_k, num_stages = cfg
    assert M <= block_M
    key = (N, K) + tuple(cfg)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = nvfp4_regfrag_kernel(N, K, block_M, n_warps, block_K, split_k, num_stages)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    if split_k > 1:
        P = torch.empty(split_k, M, N, dtype=torch.float32, device=x.device)
    else:
        P = _DUMMY.get(x.device)
        if P is None:
            P = _DUMMY[x.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=x.device)
    kernel(x.contiguous(), y, prep.Wg, prep.Sg, prep.alpha, P, prep.counters)
    return y
