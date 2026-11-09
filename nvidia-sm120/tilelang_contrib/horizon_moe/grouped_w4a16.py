# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Grouped W4A16 NVFP4 MoE GEMMs for SM120 decode (MoE step 4): BF16 activations x NVFP4 expert weights, one launch per
projection for ALL active experts, CUDA-graph safe (device-side work list, static worst-case grid).

Work list: Marlin-MoE's alignment format with 16-row blocks (vllm moe_align_block_size(topk_ids, 16, E) or
horizon_moe.route.route_align(..., block_size=16)): sorted_ids (S,) = per expert a segment of flat pair indices
p = token * TOPK + slot, padded to 16 with the sentinel MP = M * TOPK; expert_ids (B,) = the expert of every 16-row
block; ntp (1,) = number of used rows (a multiple of 16). The grid covers the worst-case B for the captured M; blocks at
or after ntp exit at once.

Per block (one expert, <= 16 pair rows) the CTA runs decode_regfrag's main loop unchanged: the expert's weights in
regfrag's fragment order (32 x 32 blocks of 512 B + 64 B of re-encoded scales, ~2.1 instructions per weight to
dequantise into mma.sync B fragments), A rows gathered from x (up: row = p // TOPK, the token) or from the
intermediate (down: row = p), fp32 accumulation, deterministic split-K (fp32 partials + ticket, fixed-order sum).

  up   (w13, N = 2 * I, K = H): rows of w13 are interleaved in 8-row groups [g0..g7, u0..u7, g8..g15, u8..u15, ...]
       (prepare_experts(..., interleave_gate_up=True)), so in the m16n8k16 C fragment a lane holds gate column c (n8
       half 0) and up column c (n8 half 1) of the same row: the epilogue writes act[p, c] = silu(g) * u directly,
       vLLM's rounding: g, u rounded to bf16 (Marlin's w13 output), s = bf16(g / (1 + expf(-g))), bf16(s * u).
  down (w2, N = H, K = I): epilogue out[p, n] = bf16(acc * alpha_e * topk_weight[p]) (Marlin's mul_topk_weights);
       the TOPK slots of a token are then summed in fixed slot order (vLLM moe_sum, or fused later).

alpha: one fp32 multiplier per expert (ModelOpt weight_scale_2; gate/up tied), x 128 (regfrag's scale fold).

K4: `grouped_body(...)` is the per-(n tile, split, block) T.macro (grid indices and shared buffers passed in);
`moe_grouped_kernel` is the standalone launch that wraps it.
"""

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

from tilelang.contrib.sm120_nvfp4 import decode_regfrag as _rf

BS = 16                      # rows per expert block (one m16 tile)
_CNT = 1 << 15               # split-K ticket counters per device (blocks x n tiles of one launch)

_ACT_SRC = r"""
__device__ __forceinline__ float hz_bf16_round(float v) {   // round-to-nearest-even to bf16 (finite inputs)
  unsigned u = __float_as_uint(v);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  return __uint_as_float(u);
}
// vLLM act_and_mul(silu): x, y are bf16 (Marlin's w13 output); silu(x) = x / (1 + expf(-x)) rounded to bf16; * y -> bf16
__device__ __forceinline__ float hz_silu_mul(float g, float u) {
  const float gq = hz_bf16_round(g), uq = hz_bf16_round(u);
  const float s = hz_bf16_round(gq / (1.0f + expf(-gq)));
  return s * uq;
}
"""


def grouped_body(mode: str, N: int, K: int, n_warps: int, block_K: int, split_k: int, num_stages: int, topk: int):
    """T.macro for one (n tile bx, split bk, block bb): see the module docstring. Shared buffers are allocated by the
    caller (K4 arena or the standalone kernel)."""
    assert mode in ("up", "down") and n_warps in (1, 2, 4, 8) and block_K % 32 == 0
    block_N = 32 * n_warps
    kchunks = block_K // 32
    n_tiles = N // block_N
    k_tiles = K // block_K
    assert N % block_N == 0 and K % block_K == 0 and k_tiles % split_k == 0, (N, K, block_N, block_K, split_k)
    kt_per_split = k_tiles // split_k
    NW = N // 32
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    emitter = TensorCoreIntrinEmitter(a_dtype=bf16, b_dtype=bf16, accum_dtype=f32, a_transposed=False,
                                      b_transposed=True, block_row_warps=1, block_col_warps=n_warps,
                                      warp_row_tiles=BS, warp_col_tiles=32, chunk=block_K)

    def _psum(P, row, col):
        """fixed-order (0, 1, ..., split_k-1) fp32 sum of the split-K partials: deterministic"""
        acc = P[0, row, col]
        for sk in range(1, split_k):
            acc = acc + P[sk, row, col]
        return acc

    @T.macro
    def body(x, sorted_ids, expert_ids, ntp, Wg, Sg, alpha, tw, Y, P, counters, MP, bx, bk, bb, tid,
             A_s, Wq_s, S_s, prow_s, arow_s, ticket_s):
        q_l = T.alloc_local((4,), i32)
        s_l = T.alloc_local((2,), i32)
        B_l = T.alloc_local((32,), bf16)
        A_l = T.alloc_local((8,), bf16)
        C_l = T.alloc_local((16,), f32)
        e = T.alloc_var(i32)
        a = T.alloc_var(f32)
        warp = tid // 32
        lane = tid % 32
        if bb * BS < ntp[0]:
            e = expert_ids[bb]
            a = alpha[e]
            if tid < BS:
                p = sorted_ids[bb * BS + tid]
                prow_s[tid] = p
                if mode == "up":
                    arow_s[tid] = T.if_then_else(p < MP, p // topk, 0)
                else:
                    arow_s[tid] = T.if_then_else(p < MP, p, 0)
            T.sync_threads()
            T.clear(C_l)
            for kt in T.Pipelined(kt_per_split, num_stages=num_stages):
                kc0 = (bk * kt_per_split + kt) * kchunks
                for i, kk in T.Parallel(BS, block_K):
                    A_s[i, kk] = x[arow_s[i], kc0 * 32 + kk]
                T.copy(Wg[e * NW + bx * n_warps, kc0, 0], Wq_s)
                T.copy(Sg[e * NW + bx * n_warps, kc0, 0], S_s)
                for c in T.unroll(kchunks):
                    for v in T.vectorized(4):
                        q_l[v] = Wq_s[warp, c, lane * 4 + v]
                    for v in T.vectorized(2):
                        s_l[v] = S_s[warp, c, (lane // 4) * 2 + v]
                    T.call_extern("handle", "tl_rf_dq_chunk", T.access_ptr(q_l, "r"), T.access_ptr(s_l, "r"),
                                  T.access_ptr(B_l, "w"))
                    for s in T.unroll(2):
                        emitter.ldmatrix_a(A_l, A_s, c * 2 + s)
                        for j in T.unroll(2):
                            T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data, 0,
                                      B_l.data, (s * 2 + j) * 8, C_l.data, j * 8, T.bool(False))
                            T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", A_l.data, 0,
                                      B_l.data, (s * 2 + j) * 8 + 4, C_l.data, j * 8 + 4, T.bool(False))
            # C_l[j*8 + r] sits at row 8*((r%4)//2) + lane//4, col 16j + 8*(r//4) + 2*(lane%4) + r%2 (warp-relative)
            if split_k == 1:
                for j in T.unroll(2):
                    for h in T.unroll(2):              # row half
                        row = 8 * h + lane // 4
                        p = prow_s[row]
                        if p < MP:
                            if mode == "up":           # gate at r = 2h + v, up at r = 4 + 2h + v: same row, same c
                                oc = (bx * block_N + warp * 32 + 16 * j) // 2 + 2 * (lane % 4)
                                for v in T.unroll(2):
                                    Y[p, oc + v] = T.cast(T.call_extern(
                                        f32, "hz_silu_mul", C_l[j * 8 + 2 * h + v] * a, C_l[j * 8 + 4 + 2 * h + v] * a),
                                        bf16)
                            else:
                                for g2 in T.unroll(2):
                                    col = bx * block_N + warp * 32 + 16 * j + 8 * g2 + 2 * (lane % 4)
                                    for v in T.unroll(2):
                                        Y[p, col + v] = T.cast(C_l[j * 8 + 4 * g2 + 2 * h + v] * a * tw[p], bf16)
            else:
                for j in T.unroll(2):
                    for r2 in T.unroll(4):
                        srow = bb * BS + 8 * (r2 % 2) + lane // 4
                        col = bx * block_N + warp * 32 + 16 * j + 8 * (r2 // 2) + 2 * (lane % 4)
                        for v in T.vectorized(2):
                            P[bk, srow, col + v] = C_l[j * 8 + r2 * 2 + v] * a
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    ticket_s[0] = T.atomic_add(counters[bb * n_tiles + bx], 1, return_prev=True)
                T.sync_threads()
                if ticket_s[0] == split_k - 1:
                    T.call_extern("handle", "__threadfence")
                    if mode == "up":
                        for i, cc in T.Parallel(BS, block_N // 2):
                            p = prow_s[i]
                            if p < MP:
                                gc = bx * block_N + (cc // 8) * 16 + cc % 8
                                Y[p, bx * (block_N // 2) + cc] = T.cast(T.call_extern(
                                    f32, "hz_silu_mul", _psum(P, bb * BS + i, gc), _psum(P, bb * BS + i, gc + 8)), bf16)
                    else:
                        for i, cc in T.Parallel(BS, block_N):
                            p = prow_s[i]
                            if p < MP:
                                Y[p, bx * block_N + cc] = T.cast(_psum(P, bb * BS + i, bx * block_N + cc) * tw[p], bf16)
                    if tid == 0:
                        counters[bb * n_tiles + bx] = 0

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def moe_grouped_kernel(mode: str, E: int, N: int, K: int, topk: int, n_warps: int, block_K: int, split_k: int,
                       num_stages: int):
    block_N = 32 * n_warps
    kchunks = block_K // 32
    n_tiles = N // block_N
    NO = N // 2 if mode == "up" else N
    M = T.dynamic("M")          # rows of x: tokens (up) or pairs (down)
    MP = T.dynamic("MP")        # pairs = tokens * topk (rows of Y; also the sentinel value)
    S = T.dynamic("S")
    B = T.dynamic("B")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32
    P_shape = (split_k, S, N) if split_k > 1 else (1, 1, 1)
    body = grouped_body(mode, N, K, n_warps, block_K, split_k, num_stages, topk)

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        sorted_ids: T.Tensor((S,), i32),
        expert_ids: T.Tensor((B,), i32),
        ntp: T.Tensor((1,), i32),
        Wg: T.Tensor((E * (N // 32), K // 32, 128), i32),
        Sg: T.Tensor((E * (N // 32), K // 32, 16), i32),
        alpha: T.Tensor((E,), f32),
        tw: T.Tensor((MP,), f32),
        Y: T.Tensor((MP, NO), bf16),
        P: T.Tensor(P_shape, f32),
        counters: T.Tensor((_CNT,), i32),
    ):
        with T.Kernel(n_tiles, split_k, B, threads=32 * n_warps, prelude=_rf._DQ_SRC + _ACT_SRC) as (bx, bk, bb):
            A_s = T.alloc_shared((BS, block_K), bf16)
            Wq_s = T.alloc_shared((n_warps, kchunks, 128), i32)
            S_s = T.alloc_shared((n_warps, kchunks, 16), i32)
            prow_s = T.alloc_shared((BS,), i32)
            arow_s = T.alloc_shared((BS,), i32)
            ticket_s = T.alloc_shared((1,), i32)
            T.annotate_layout({A_s: make_mma_swizzle_layout(A_s)})
            tid = T.get_thread_binding()
            body(x, sorted_ids, expert_ids, ntp, Wg, Sg, alpha, tw, Y, P, counters, MP, bx, bk, bb, tid,
                 A_s, Wq_s, S_s, prow_s, arow_s, ticket_s)

    return main


# ---------------------------------------------------------------------------------------------------------------
# Host side
class GroupedWeights:
    """One projection of all experts in regfrag's fragment order, concatenated: Wg (E * N/32, K/32, 128) int32,
    Sg (E * N/32, K/32, 16) int32, alpha (E,) fp32 (x 128)."""
    __slots__ = ("E", "N", "K", "Wg", "Sg", "alpha", "interleaved")

    def __init__(self, E, N, K, Wg, Sg, alpha, interleaved):
        self.E, self.N, self.K, self.Wg, self.Sg, self.alpha, self.interleaved = E, N, K, Wg, Sg, alpha, interleaved


def interleave_gate_up(t):
    """(E, 2I, ...) rows [gate 0..I-1 | up 0..I-1] -> [g0..g7, u0..u7, g8..g15, u8..u15, ...]."""
    E, N2 = t.shape[0], t.shape[1]
    I = N2 // 2
    assert I % 8 == 0
    g, u = t[:, :I], t[:, I:]
    import torch
    return torch.stack([g.reshape(E, I // 8, 8, *t.shape[2:]), u.reshape(E, I // 8, 8, *t.shape[2:])], 2) \
        .reshape(t.shape).contiguous()


def prepare_experts(packed_u8, scale_fp8, alpha_fp32, interleave=False):
    """packed_u8 (E, N, K/2) uint8 e2m1 pairs (low nibble = even k), scale_fp8 (E, N, K/16) e4m3 (row-major, the
    checkpoint layout), alpha_fp32 (E,) per-expert multiplier -> GroupedWeights (untimed; ~one expert at a time)."""
    import torch
    E, N, K2 = packed_u8.shape
    K = K2 * 2
    if interleave:
        packed_u8 = interleave_gate_up(packed_u8)
        scale_fp8 = interleave_gate_up(scale_fp8.view(torch.uint8)).view(torch.float8_e4m3fn)
    NW = N // 32
    dev = packed_u8.device
    Wg = torch.empty(E * NW, K // 32, 128, dtype=torch.int32, device=dev)
    Sg = torch.empty(E * NW, K // 32, 16, dtype=torch.int32, device=dev)
    one = torch.ones(N, dtype=torch.float32, device=dev)
    for e in range(E):
        p = _rf.prepare(packed_u8[e].view(torch.int8), scale_fp8[e], one)
        Wg[e * NW:(e + 1) * NW] = p.Wg
        Sg[e * NW:(e + 1) * NW] = p.Sg
    alpha = (alpha_fp32.to(torch.float32).reshape(E) * _rf._ALPHA_FOLD).contiguous().to(dev)
    return GroupedWeights(E, N, K, Wg, Sg, alpha, interleave)


_KERNELS = {}
_COUNTERS = {}
_DUMMY = {}


def _counters(dev):
    c = _COUNTERS.get(dev)
    if c is None:
        import torch
        c = _COUNTERS[dev] = torch.zeros(_CNT, dtype=torch.int32, device=dev)
    return c


def config(mode, N, K, M):
    """(n_warps, block_K, split_k, num_stages) per projection and token count (first guess; the race tunes it).
    Decode is weight-stream bound: few 16-row blocks, so split K until ~1-2 waves of CTAs."""
    n_warps, block_K = 4, 64
    k_tiles = K // block_K
    active = min(9 * M, 257)                       # distinct experts ~ blocks at decode
    ctas = (N // (32 * n_warps)) * active
    split = 1
    while ctas * split * 2 <= 2 * 188 and k_tiles % (split * 2) == 0 and split < 8:
        split *= 2
    return n_warps, block_K, split, 4


def run(mode, x, align, w, tw, topk, out=None, cfg=None):
    """One grouped projection. x: (M, K) bf16 (tokens for up, pair rows for down); align = (sorted_ids, expert_ids,
    ntp) with 16-row blocks; tw: (MP,) fp32 slot weights (down; ignored for up). Returns Y (MP, N/2 or N) bf16."""
    import torch
    sorted_ids, expert_ids, ntp = align
    rows = x.shape[0]
    MP = tw.shape[0]
    NO = w.N // 2 if mode == "up" else w.N
    m_tokens = MP // topk
    if cfg is None:
        cfg = config(mode, w.N, w.K, m_tokens)
    n_warps, block_K, split_k, num_stages = cfg
    key = (mode, w.E, w.N, w.K, topk) + tuple(cfg)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = moe_grouped_kernel(mode, w.E, w.N, w.K, topk, n_warps, block_K, split_k, num_stages)
    y = out if out is not None else torch.empty(MP, NO, dtype=torch.bfloat16, device=x.device)
    if split_k > 1:
        P = torch.empty(split_k, sorted_ids.shape[0], w.N, dtype=torch.float32, device=x.device)
    else:
        P = _DUMMY.get(x.device)
        if P is None:
            P = _DUMMY[x.device] = torch.zeros(1, 1, 1, dtype=torch.float32, device=x.device)
    assert expert_ids.shape[0] * (w.N // (32 * n_warps)) <= _CNT
    k(x.contiguous(), sorted_ids, expert_ids, ntp, w.Wg, w.Sg, w.alpha, tw, y, P, _counters(x.device))
    return y


def moe_forward(x, topk_weights, align, w13, w2, cfg_up=None, cfg_down=None, moe_sum=None):
    """Full MoE for x (M, H) bf16, topk_weights (M, TOPK) fp32, align (16-row blocks over the ids): up (+SiLU*up),
    down (x slot weight), then the fixed-order slot sum. Returns (M, H) bf16."""
    import torch
    M, H = x.shape
    topk = topk_weights.shape[1]
    tw = topk_weights.reshape(-1).contiguous()
    act = run("up", x, align, w13, tw, topk, cfg=cfg_up)
    y2 = run("down", act, align, w2, tw, topk, cfg=cfg_down)
    if moe_sum is not None:
        out = torch.empty(M, H, dtype=x.dtype, device=x.device)
        moe_sum(y2.view(M, topk, H), out)
        return out
    return y2.view(M, topk, H).float().sum(1).to(x.dtype)
