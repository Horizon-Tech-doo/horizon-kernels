"""GQA flash-decoding (one new token per sequence) for gfx90a (MI250X), TileLang.

Decode reads the whole KV cache once per step, so it is memory-bandwidth bound; the metric is GB/s vs the GCD's HBM
peak (~1.6 TB/s), plus speed vs torch SDPA (enable_gqa=True).
  kernel 1: grid (splits, B * Hkv). A block takes one kv head, its G query heads (rows padded to 16 for MFMA) and one
            slice of the cache; each K/V tile is read once for the G heads. Writes partial O and log-sum-exp per split.
  kernel 2: combines the splits per (batch, head).
Layout: Q [B, H, D], K/V cache [B, Lmax, Hkv, D] fp16, seqlens [B] int32 (variable lengths), O [B, H, D] fp16.
Usage: python gqa_decode.py   (correctness vs float64, speed vs SDPA, achieved bandwidth)
"""
import json
import math
import sys

import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.transform import PassConfigKey


@tilelang.jit(out_idx=[4, 5])
def decode_partial(B, Lmax, Hkv, G, D, S, bN, threads, policy="row", bM=16):
    H = Hkv * G
    # G query heads padded to bM rows. bM 16 = one MFMA tile for one wavefront; bM 64 with 256 threads gives four
    # wavefronts the same K/V tile (the extra rows are wasted MFMA work, cheap in a bandwidth-bound kernel) so more
    # loads are in flight per workgroup (5 Nov, after vLLM's Triton kernel reached 58-59% of HBM at batch 32-64).
    chunk = T.ceildiv(Lmax, S)
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32
    pol = GemmWarpPolicy.FullRow if policy == "row" else GemmWarpPolicy.FullCol

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K: T.Tensor((B, Lmax, Hkv, D), dtype), V: T.Tensor((B, Lmax, Hkv, D), dtype),
             Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc), Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(S, B * Hkv, threads=threads) as (s, byz):
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            k_lo = s * chunk
            k_hi = T.min(k_lo + chunk, n)
            K_s = T.alloc_shared((bN, D), dtype)
            V_s = T.alloc_shared((bN, D), dtype)
            q = T.alloc_fragment((bM, D), dtype)
            acc_s = T.alloc_fragment((bM, bN), acc)
            p = T.alloc_fragment((bM, bN), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            rs = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            for i, d in T.Parallel(bM, D):
                q[i, d] = T.if_then_else(i < G, Q[b, hk * G + i, d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.ceildiv(T.max(k_hi - k_lo, 0), bN)):
                k0 = k_lo + t * bN
                for j, d in T.Parallel(bN, D):
                    K_s[j, d] = T.if_then_else(k0 + j < k_hi, K[b, k0 + j, hk, d], T.cast(0, dtype))
                    V_s[j, d] = T.if_then_else(k0 + j < k_hi, V[b, k0 + j, hk, d], T.cast(0, dtype))
                for i, j in T.Parallel(bM, bN):
                    acc_s[i, j] = T.if_then_else(k0 + j < k_hi, 0, -T.infinity(acc))
                T.gemm(q, K_s, acc_s, transpose_B=True, policy=pol)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, bN):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, rs, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += rs[i]
                T.copy(acc_s, p)
                T.gemm(p, V_s, acc_o, policy=pol)
            for i, d in T.Parallel(bM, D):
                if i < G:
                    Op[b, hk * G + i, s, d] = T.if_then_else(l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0)
            for i in T.Parallel(bM):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


@tilelang.jit(out_idx=[5, 6])
def decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, S, threads=64, PAGE=16, bM=16):
    """Paged KV cache in vLLM's NHD layout: K/V [num_blocks, PAGE, Hkv, D], block table [B, MAXP] (page ids), lengths [B].
    One 16-key tile = one page, so the tile is a plain copy of page BT[b, p]. Splits are whole pages."""
    H = Hkv * G
    pages_per_split = T.ceildiv(MAXP, S)
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K: T.Tensor((NBLK, PAGE, Hkv, D), dtype), V: T.Tensor((NBLK, PAGE, Hkv, D), dtype),
             BT: T.Tensor((B, MAXP), T.int32), Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc),
             Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(S, B * Hkv, threads=threads) as (s, byz):
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            p_lo = s * pages_per_split
            p_hi = T.min(p_lo + pages_per_split, T.ceildiv(n, PAGE))
            K_s = T.alloc_shared((PAGE, D), dtype)
            V_s = T.alloc_shared((PAGE, D), dtype)
            q = T.alloc_fragment((bM, D), dtype)
            acc_s = T.alloc_fragment((bM, PAGE), acc)
            p = T.alloc_fragment((bM, PAGE), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            rs = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            for i, d in T.Parallel(bM, D):
                q[i, d] = T.if_then_else(i < G, Q[b, hk * G + T.min(i, G - 1), d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.max(p_hi - p_lo, 0)):
                pg = p_lo + t
                blk = BT[b, pg]
                k0 = pg * PAGE
                T.copy(K[blk, 0:PAGE, hk, :], K_s)
                T.copy(V[blk, 0:PAGE, hk, :], V_s)
                for j, d in T.Parallel(PAGE, D):                  # stale finite data past n is safe; NaN in K past n is not (N3, rel 0.26)
                    if k0 + j >= n:
                        V_s[j, d] = T.cast(0, dtype)
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.if_then_else(k0 + j < n, 0, -T.infinity(acc))
                T.gemm(q, K_s, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, rs, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += rs[i]
                T.copy(acc_s, p)
                T.gemm(p, V_s, acc_o, policy=GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(bM, D):
                if i < G:
                    Op[b, hk * G + i, s, d] = T.if_then_else(l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0)
            for i in T.Parallel(bM):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


def _bf16_bits(x):
    """fp32 to bf16 by bits, round to nearest even, no branch: what Triton's truncf compiles to, less its NaN select (N3 round 4). hip_bfloat16's own conversion tests for inf and NaN behind an exec-mask
    branch per element. NaN and inf stay NaN and inf; only a NaN payload of all ones would wrap, and exp and division do not produce one."""
    b = T.reinterpret(x, T.uint32)
    return T.reinterpret(T.cast((b + ((b >> 16) & 1) + 0x7fff) >> 16, T.uint16), T.bfloat16)


def _kv_layout(packed, NBLK, Hkv, PAGE, D, Nb=None):
    """Physical KV cache layouts of vLLM 0.30 (N3, 6 Nov), as (name, K shape, V shape, V_s shape, V_s transposed). Nb is the cache's page size in tokens, a multiple of the kernel's
    16-token tile; a hybrid model such as Qwen3.8 has large pages (read from a live start). NBLK counts those pages.
    False "nhd": separate K, V tensors [NBLK, Nb, Hkv, D]; not a vLLM layout, kept for round 1 comparisons.
    True or "lbhnc": one tensor [NBLK, Hkv, Nb, 2D], K then V in each row. TRITON_ATTN with VLLM_KV_CACHE_LAYOUT=HND. Pass the same tensor as K and V.
    "lbnhc": one tensor [NBLK, Nb, Hkv, 2D], K then V in each row. TRITON_ATTN's default, on hybrid models too. Pass the same tensor as K and V.
    "lhbnc": K [NBLK, Hkv, D / 8, Nb, 8] (x-packed), V [NBLK, Hkv, D, Nb] (D-major), each group whole, from PagedAttention.split_kv_cache. ROCM_ATTN's default on a uniform model.
    "rocm_hybrid": K and V as in "lhbnc" but side by side in each page, physical [NBLK, 2, Nb, Hkv * D]. What ROCM_ATTN, the default backend, gets on a hybrid model: vLLM calls it LBHNC,
    with two head slots. Pass views K [NBLK, 2, Hkv, D / 8, Nb, 8] and V [NBLK, 2, Hkv, D, Nb] of the same memory."""
    Nb = Nb or PAGE
    layout = {False: "nhd", True: "lbhnc"}.get(packed, packed)
    if layout == "lhbnc":
        assert D % 8 == 0
        return layout, (NBLK, Hkv, D // 8, Nb, 8), (NBLK, Hkv, D, Nb), (D, PAGE), True
    if layout == "rocm_hybrid":
        assert D % 8 == 0
        return layout, (NBLK, 2, Hkv, D // 8, Nb, 8), (NBLK, 2, Hkv, D, Nb), (D, PAGE), True
    if layout == "nhd":
        return layout, (NBLK, Nb, Hkv, D), (NBLK, Nb, Hkv, D), (PAGE, D), False
    shape = {"lbhnc": (NBLK, Hkv, Nb, 2 * D), "lbnhc": (NBLK, Nb, Hkv, 2 * D)}[layout]
    return layout, shape, shape, (PAGE, D), False


@tilelang.jit(out_idx=[5, 6], pass_configs={PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
def decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, threads=128, PAGE=16, bM=32, qform="if", swap=False, rc=False, taper=None, tail="kvg", packed=False, stage="joint", kord="jx", Nb=None, dt="fp16", pc="bits"):
    """Paged decode with one global-load round per page (N3, 6 Nov): K and V are loaded in one loop, no T.copy (T.copy puts a
    barrier after each copy), so the wave's 16 dwordx4 loads are issued before the first LDS write. Needs bounds checks off,
    since a checked load is its own branch and its own round. BT must hold valid page ids; there is no bounds check on it.
    rc: divide once per row and multiply, instead of one fp32 division per output element (the division is 640 of the block's VALU).
    taper (S1, A, a): the first S1 splits hold A pages, the rest a pages. With swap the grid walks s slowest, so the last blocks
    launched are short and the kernel's tail is short.
    tail: what happens to the slots past n in the last page, which may hold stale data including NaN. "kvg" zeroes K and V and sets the
    tail scores to -inf behind one test, so only the last page runs the masks; it is NaN safe and no slower than "kv" (N3, jobs 23504499
    and 23504570). "kv" is the same without the test. "v" zeroes V only and is kept as the measured baseline: NaN in K's tail reaches every
    score of the row, since NaN + -inf is NaN, and the whole output is NaN.
    packed: which physical cache layout K and V are read from, see _kv_layout. The layout is a deployment choice: ROCM_ATTN is vLLM 0.30's
    default on gfx90a and uses "lhbnc"; TRITON_ATTN defaults to "lbnhc"; True ("lbhnc") is only reached with VLLM_KV_CACHE_LAYOUT=HND.
    stage: "joint" issues K and V loads in one loop, "k_v" in two successive loops, K first (always the case on "lhbnc", where K and V have
    different loop shapes; "v_k" loads V first there). kord: thread order of the K loop on "lhbnc" and "rocm_hybrid", "jx" or "xj" (address order). Nb: see _kv_layout; BT then holds one entry per Nb tokens, MAXP stays in tiles.
    dt: "fp16" or "bf16". vLLM's cache takes the model's dtype, and Qwen3.8's is BF16 (read from a live start). pc: how probabilities become bf16 for the V product, "bits" (_bf16_bits; gains 1-3% over "hip", jobs 23508935, 23508973) or "hip" (hip_bfloat16's conversion). No effect on fp16."""
    H = Hkv * G
    layout, Kshape, Vshape, Vsshape, transV = _kv_layout(packed, NBLK, Hkv, PAGE, D, Nb)
    Mt = (Nb or PAGE) // PAGE
    assert (Nb or PAGE) % PAGE == 0
    pages_per_split = T.ceildiv(MAXP, S)
    if taper is not None:
        S1, A, a = taper
        assert S1 * A + (S - S1) * a >= MAXP
    scale = (1.0 / D) ** 0.5
    dtype, acc = {"fp16": T.float16, "bf16": T.bfloat16}[dt], T.float32

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K: T.Tensor(Kshape, dtype), V: T.Tensor(Vshape, dtype),
             BT: T.Tensor((B, -(-MAXP // Mt)), T.int32), Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc),
             Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(B * Hkv, S, threads=threads) if swap else T.Kernel(S, B * Hkv, threads=threads) as grid:
            byz, s = grid if swap else (grid[1], grid[0])
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            if taper is None:
                p_lo = s * pages_per_split
                p_hi = T.min(p_lo + pages_per_split, T.ceildiv(n, PAGE))
            else:
                p_lo = T.min(s, S1) * A + T.max(s - S1, 0) * a
                p_hi = T.min(p_lo + T.if_then_else(s < S1, A, a), T.ceildiv(n, PAGE))
            K_s = T.alloc_shared((PAGE, D), dtype)
            V_s = T.alloc_shared(Vsshape, dtype)
            pad_c2p = T.alloc_shared((512,), T.float16)
            T.fill(pad_c2p, 0)
            q = T.alloc_fragment((bM, D), dtype)
            acc_s = T.alloc_fragment((bM, PAGE), acc)
            p = T.alloc_fragment((bM, PAGE), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            rs = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            rl = T.alloc_fragment((bM,), acc)
            if qform == "if":
                for i, d in T.Parallel(bM, D):
                    q[i, d] = T.if_then_else(i < G, Q[b, hk * G + T.min(i, G - 1), d], T.cast(0, dtype))
            elif qform == "plain":                                # unconditional load, zero after: rows >= G hold head G-1 for one step
                for i, d in T.Parallel(bM, D):
                    q[i, d] = Q[b, hk * G + T.min(i, G - 1), d]
                for i, d in T.Parallel(bM, D):
                    if i >= G:
                        q[i, d] = T.cast(0, dtype)
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.max(p_hi - p_lo, 0)):
                pg = p_lo + t
                blk = BT[b, pg // Mt]
                j0 = (pg % Mt) * PAGE
                k0 = pg * PAGE
                if layout == "lhbnc" or layout == "rocm_hybrid":      # K x-packed, V D-major: contiguous along e and along j, so V_s stays [D, PAGE]
                    if stage == "v_k":
                        if layout == "lhbnc":
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, hk, d, j0 + j]
                        else:
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, 1, hk, d, j0 + j]
                    if layout == "lhbnc":
                        if kord == "xj":                              # threads walk K in address order
                            for x8, j, e in T.Parallel(D // 8, PAGE, 8):
                                K_s[j, x8 * 8 + e] = K[blk, hk, x8, j0 + j, e]
                        else:
                            for j, x8, e in T.Parallel(PAGE, D // 8, 8):
                                K_s[j, x8 * 8 + e] = K[blk, hk, x8, j0 + j, e]
                    else:
                        if kord == "xj":
                            for x8, j, e in T.Parallel(D // 8, PAGE, 8):
                                K_s[j, x8 * 8 + e] = K[blk, 0, hk, x8, j0 + j, e]
                        else:
                            for j, x8, e in T.Parallel(PAGE, D // 8, 8):
                                K_s[j, x8 * 8 + e] = K[blk, 0, hk, x8, j0 + j, e]
                    if stage != "v_k":
                        if layout == "lhbnc":
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, hk, d, j0 + j]
                        else:
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, 1, hk, d, j0 + j]
                elif stage == "k_v":
                    if layout == "lbhnc":
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, hk, j0 + j, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, hk, j0 + j, D + d]
                    elif layout == "lbnhc":
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, j0 + j, hk, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, j0 + j, hk, D + d]
                    else:
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, j0 + j, hk, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, j0 + j, hk, d]
                elif layout == "lbhnc":
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, hk, j0 + j, d]
                        V_s[j, d] = V[blk, hk, j0 + j, D + d]
                elif layout == "lbnhc":
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, j0 + j, hk, d]
                        V_s[j, d] = V[blk, j0 + j, hk, D + d]
                else:
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, j0 + j, hk, d]
                        V_s[j, d] = V[blk, j0 + j, hk, d]
                if tail == "kvg":
                    T.fill(acc_s, 0)
                    if k0 + PAGE > n:
                        if transV:
                            for j, d in T.Parallel(PAGE, D):
                                if k0 + j >= n:
                                    K_s[j, d] = T.cast(0, dtype)
                            for d, j in T.Parallel(D, PAGE):
                                if k0 + j >= n:
                                    V_s[d, j] = T.cast(0, dtype)
                        else:
                            for j, d in T.Parallel(PAGE, D):
                                if k0 + j >= n:
                                    K_s[j, d] = T.cast(0, dtype)
                                    V_s[j, d] = T.cast(0, dtype)
                        for i, j in T.Parallel(bM, PAGE):
                            if k0 + j >= n:
                                acc_s[i, j] = -T.infinity(acc)
                else:
                    for j, d in T.Parallel(PAGE, D):
                        if k0 + j >= n:
                            V_s[j, d] = T.cast(0, dtype)
                            if tail == "kv":
                                K_s[j, d] = T.cast(0, dtype)
                    for i, j in T.Parallel(bM, PAGE):
                        acc_s[i, j] = T.if_then_else(k0 + j < n, 0, -T.infinity(acc))
                T.gemm(q, K_s, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, rs, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += rs[i]
                if pc == "bits" and dt == "bf16":
                    for i, j in T.Parallel(bM, PAGE):
                        p[i, j] = _bf16_bits(acc_s[i, j])
                else:
                    T.copy(acc_s, p)
                T.gemm(p, V_s, acc_o, transpose_B=transV, policy=GemmWarpPolicy.FullRow)
            if rc:
                for i in T.Parallel(bM):
                    rl[i] = T.if_then_else(l_i[i] > 0, 1.0 / l_i[i], 0.0)
                for i, d in T.Parallel(bM, D):
                    if i < G:
                        Op[b, hk * G + i, s, d] = acc_o[i, d] * rl[i]
            else:
                for i, d in T.Parallel(bM, D):
                    if i < G:
                        Op[b, hk * G + i, s, d] = T.if_then_else(l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0)
            for i in T.Parallel(bM):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


@tilelang.jit(out_idx=[5, 6], pass_configs={PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True})
def decode_partial_paged_v3(B, NBLK, MAXP, Hkv, G, D, S1, S2, threads=128, PAGE=16, bM=32, frac=86, tail="kvg", packed=False, stage="joint", kord="jx", Nb=None, dt="fp16", pc="bits"):
    """decode_partial_paged_v2 with the swap grid and rc, and split ranges computed from each sequence's own page count (N3, 6 Nov).
    frac percent of a sequence's pages go in the first S1 splits and the rest in the other S2, so the taper scales with the sequence
    instead of a short sequence filling a few long splits and leaving the short ones empty. S1 * A + S2 * a >= pages holds for every length.
    tail, packed and stage as in decode_partial_paged_v2."""
    H = Hkv * G
    S = S1 + S2
    layout, Kshape, Vshape, Vsshape, transV = _kv_layout(packed, NBLK, Hkv, PAGE, D, Nb)
    Mt = (Nb or PAGE) // PAGE
    assert (Nb or PAGE) % PAGE == 0
    assert S1 >= 1 and S2 >= 1
    scale = (1.0 / D) ** 0.5
    dtype, acc = {"fp16": T.float16, "bf16": T.bfloat16}[dt], T.float32

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K: T.Tensor(Kshape, dtype), V: T.Tensor(Vshape, dtype),
             BT: T.Tensor((B, -(-MAXP // Mt)), T.int32), Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc),
             Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(B * Hkv, S, threads=threads) as (byz, s):
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            pages = T.ceildiv(n, PAGE)
            A = T.max(T.ceildiv(pages * frac // 100, S1), 1)
            a = T.max(T.ceildiv(T.max(pages - S1 * A, 0), S2), 1)
            p_lo = T.min(s, S1) * A + T.max(s - S1, 0) * a
            p_hi = T.min(p_lo + T.if_then_else(s < S1, A, a), pages)
            K_s = T.alloc_shared((PAGE, D), dtype)
            V_s = T.alloc_shared(Vsshape, dtype)
            pad_c2p = T.alloc_shared((512,), T.float16)
            T.fill(pad_c2p, 0)
            q = T.alloc_fragment((bM, D), dtype)
            acc_s = T.alloc_fragment((bM, PAGE), acc)
            p = T.alloc_fragment((bM, PAGE), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            rs = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            rl = T.alloc_fragment((bM,), acc)
            for i, d in T.Parallel(bM, D):
                q[i, d] = T.if_then_else(i < G, Q[b, hk * G + T.min(i, G - 1), d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.max(p_hi - p_lo, 0)):
                pg = p_lo + t
                blk = BT[b, pg // Mt]
                j0 = (pg % Mt) * PAGE
                k0 = pg * PAGE
                if layout == "lhbnc" or layout == "rocm_hybrid":      # K x-packed, V D-major: contiguous along e and along j, so V_s stays [D, PAGE]
                    if stage == "v_k":
                        if layout == "lhbnc":
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, hk, d, j0 + j]
                        else:
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, 1, hk, d, j0 + j]
                    if layout == "lhbnc":
                        if kord == "xj":                              # threads walk K in address order
                            for x8, j, e in T.Parallel(D // 8, PAGE, 8):
                                K_s[j, x8 * 8 + e] = K[blk, hk, x8, j0 + j, e]
                        else:
                            for j, x8, e in T.Parallel(PAGE, D // 8, 8):
                                K_s[j, x8 * 8 + e] = K[blk, hk, x8, j0 + j, e]
                    else:
                        if kord == "xj":
                            for x8, j, e in T.Parallel(D // 8, PAGE, 8):
                                K_s[j, x8 * 8 + e] = K[blk, 0, hk, x8, j0 + j, e]
                        else:
                            for j, x8, e in T.Parallel(PAGE, D // 8, 8):
                                K_s[j, x8 * 8 + e] = K[blk, 0, hk, x8, j0 + j, e]
                    if stage != "v_k":
                        if layout == "lhbnc":
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, hk, d, j0 + j]
                        else:
                            for d, j in T.Parallel(D, PAGE):
                                V_s[d, j] = V[blk, 1, hk, d, j0 + j]
                elif stage == "k_v":
                    if layout == "lbhnc":
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, hk, j0 + j, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, hk, j0 + j, D + d]
                    elif layout == "lbnhc":
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, j0 + j, hk, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, j0 + j, hk, D + d]
                    else:
                        for j, d in T.Parallel(PAGE, D):
                            K_s[j, d] = K[blk, j0 + j, hk, d]
                        for j, d in T.Parallel(PAGE, D):
                            V_s[j, d] = V[blk, j0 + j, hk, d]
                elif layout == "lbhnc":
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, hk, j0 + j, d]
                        V_s[j, d] = V[blk, hk, j0 + j, D + d]
                elif layout == "lbnhc":
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, j0 + j, hk, d]
                        V_s[j, d] = V[blk, j0 + j, hk, D + d]
                else:
                    for j, d in T.Parallel(PAGE, D):
                        K_s[j, d] = K[blk, j0 + j, hk, d]
                        V_s[j, d] = V[blk, j0 + j, hk, d]
                if tail == "kvg":
                    T.fill(acc_s, 0)
                    if k0 + PAGE > n:
                        if transV:
                            for j, d in T.Parallel(PAGE, D):
                                if k0 + j >= n:
                                    K_s[j, d] = T.cast(0, dtype)
                            for d, j in T.Parallel(D, PAGE):
                                if k0 + j >= n:
                                    V_s[d, j] = T.cast(0, dtype)
                        else:
                            for j, d in T.Parallel(PAGE, D):
                                if k0 + j >= n:
                                    K_s[j, d] = T.cast(0, dtype)
                                    V_s[j, d] = T.cast(0, dtype)
                        for i, j in T.Parallel(bM, PAGE):
                            if k0 + j >= n:
                                acc_s[i, j] = -T.infinity(acc)
                else:
                    for j, d in T.Parallel(PAGE, D):
                        if k0 + j >= n:
                            K_s[j, d] = T.cast(0, dtype)
                            V_s[j, d] = T.cast(0, dtype)
                    for i, j in T.Parallel(bM, PAGE):
                        acc_s[i, j] = T.if_then_else(k0 + j < n, 0, -T.infinity(acc))
                T.gemm(q, K_s, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, rs, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += rs[i]
                if pc == "bits" and dt == "bf16":
                    for i, j in T.Parallel(bM, PAGE):
                        p[i, j] = _bf16_bits(acc_s[i, j])
                else:
                    T.copy(acc_s, p)
                T.gemm(p, V_s, acc_o, transpose_B=transV, policy=GemmWarpPolicy.FullRow)
            for i in T.Parallel(bM):
                rl[i] = T.if_then_else(l_i[i] > 0, 1.0 / l_i[i], 0.0)
            for i, d in T.Parallel(bM, D):
                if i < G:
                    Op[b, hk * G + i, s, d] = acc_o[i, d] * rl[i]
            for i in T.Parallel(bM):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


@tilelang.jit(out_idx=[8, 9])
def decode_partial_paged_fp8(B, NBLK, MAXP, Hkv, G, D, S, threads=128, PAGE=16, bM=32):
    """Simulated FP8 KV cache on gfx90a (no FP8 hardware): K/V stored as FP8 e4m3 bit patterns (uint8) with one scale
    per tensor, decoded in the kernel through a 256-entry fp16 lookup table in LDS. Half the bytes of fp16."""
    H = Hkv * G
    pages_per_split = T.ceildiv(MAXP, S)
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K8: T.Tensor((NBLK, PAGE, Hkv, D // 4), T.int32), V8: T.Tensor((NBLK, PAGE, Hkv, D // 4), T.int32),
             Lut: T.Tensor((256,), dtype), Ksc: T.Tensor((1,), acc), Vsc: T.Tensor((1,), acc),
             BT: T.Tensor((B, MAXP), T.int32), Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc),
             Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(S, B * Hkv, threads=threads) as (s, byz):
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            p_lo = s * pages_per_split
            p_hi = T.min(p_lo + pages_per_split, T.ceildiv(n, PAGE))
            lut = T.alloc_shared((256,), dtype)
            K8s = T.alloc_shared((PAGE, D // 4), T.int32)    # 4 FP8 bytes per word: TileLang's vectorised uint8 copy
            V8s = T.alloc_shared((PAGE, D // 4), T.int32)    # emits a broken make_uint4 call on HIP (5 Nov)
            K_s = T.alloc_shared((PAGE, D), dtype)
            V_s = T.alloc_shared((PAGE, D), dtype)
            q = T.alloc_fragment((bM, D), dtype)
            acc_s = T.alloc_fragment((bM, PAGE), acc)
            p = T.alloc_fragment((bM, PAGE), dtype)
            acc_o = T.alloc_fragment((bM, D), acc)
            m_i = T.alloc_fragment((bM,), acc)
            m_prev = T.alloc_fragment((bM,), acc)
            l_i = T.alloc_fragment((bM,), acc)
            rs = T.alloc_fragment((bM,), acc)
            sf = T.alloc_fragment((bM,), acc)
            T.copy(Lut, lut)
            ks = Ksc[0]
            vs = Vsc[0]
            for i, d in T.Parallel(bM, D):
                q[i, d] = T.if_then_else(i < G, Q[b, hk * G + T.min(i, G - 1), d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.max(p_hi - p_lo, 0)):
                pg = p_lo + t
                blk = BT[b, pg]
                k0 = pg * PAGE
                T.copy(K8[blk, 0:PAGE, hk, :], K8s)
                T.copy(V8[blk, 0:PAGE, hk, :], V8s)
                for j, d in T.Parallel(PAGE, D):
                    K_s[j, d] = lut[(K8s[j, d // 4] >> ((d % 4) * 8)) & 255]
                    V_s[j, d] = T.if_then_else(k0 + j < n, T.cast(T.cast(lut[(V8s[j, d // 4] >> ((d % 4) * 8)) & 255], acc) * vs, dtype),
                                               T.cast(0, dtype))
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.if_then_else(k0 + j < n, 0, -T.infinity(acc))
                T.gemm(q, K_s, acc_s, transpose_B=True, policy=GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(bM, PAGE):                 # K scale applied to the scores (one multiply per score)
                    acc_s[i, j] = acc_s[i, j] * ks
                T.copy(m_i, m_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for i in T.Parallel(bM):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(bM):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, d in T.Parallel(bM, D):
                    acc_o[i, d] *= sf[i]
                for i, j in T.Parallel(bM, PAGE):
                    acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(acc_s, rs, dim=1)
                for i in T.Parallel(bM):
                    l_i[i] += rs[i]
                T.copy(acc_s, p)
                T.gemm(p, V_s, acc_o, policy=GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(bM, D):
                if i < G:
                    Op[b, hk * G + i, s, d] = T.if_then_else(l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0)
            for i in T.Parallel(bM):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


def paged_v2_config(MAXP):
    """Best measured N3 config, batch 8 to 64 at about 1 GB of cache: 86% of pages in splits of 44, the rest in splits of 12, so
    about 1650 long blocks and 1000 short ones are launched in that order. S = S1 + S2 for the Op/Lse shapes and for decode_combine."""
    S1 = max(1, MAXP * 86 // (100 * 44))
    S2 = -(-max(MAXP - S1 * 44, 0) // 12)
    return S1 + S2, (S1, 44, 12)


def paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE=16, packed=True, stage=None, kord=None, Nb=None, dt="fp16", pc="bits"):
    """Best measured decode for this shape (N3, 6 Nov; evidence/N3): returns (k1, k2). Whole-cache reads, so block count and block length are what matter.
    packed picks the physical cache layout (_kv_layout), stage the load order, kord the K thread order; the selection below was tuned on "lbhnc".
    Defaults by measurement on the vLLM image stack: K then V on "lbhnc" (joint is 0.88 to 1.0 ms and unstable there, K then V a steady 0.88, jobs 23508495, 23508529), joint on "lbnhc",
    K in address order and K before V on "lhbnc" and "rocm_hybrid" (V first flips by job).
    Under about 700 blocks with paged_v2_config the CUs are not full (110 CUs at 3 blocks each): uniform splits of up to 16 pages, at least 512 blocks where pages allow.
    Long sequences (256 pages or more): v3 with v2's split counts. Below 256 pages v3's per-sequence ranges make blocks too short: v2's taper.
    Batch 16 x 1024 pages: uniform 27 without the swap; it beats the taper on base, full and ties on the held-out part and skew lengths."""
    H = Hkv * G
    if stage is None:
        stage = "k_v" if packed is True or packed == "lbhnc" else "joint"
    if kord is None:
        kord = "xj" if packed in ("lhbnc", "rocm_hybrid") else "jx"
    Sv, tp = paged_v2_config(MAXP)
    if B == 16 and MAXP == 1024:
        k1 = decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, 27, 128, PAGE, 32, "if", False, True, None, "kvg", packed, stage, kord, Nb, dt, pc)
        return k1, decode_combine_v2(B, H, 27, D, 256, dt, pc)
    if B * Hkv * Sv < 700:
        P = min(16, max(4, -(-B * MAXP * Hkv // 512)))
        S = -(-MAXP // P)
        k1 = decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", True, True, None, "kvg", packed, stage, kord, Nb, dt, pc)
        return k1, decode_combine_v2(B, H, S, D, 256, dt, pc)
    if MAXP >= 256:
        S1 = tp[0]
        S2 = max(Sv - S1, 1)
        k1 = decode_partial_paged_v3(B, NBLK, MAXP, Hkv, G, D, S1, S2, 128, PAGE, 32, 86, "kvg", packed, stage, kord, Nb, dt, pc)
        return k1, decode_combine_v2(B, H, S1 + S2, D, 256, dt, pc)
    k1 = decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, Sv, 128, PAGE, 32, "if", True, True, tp, "kvg", packed, stage, kord, Nb, dt, pc)
    return k1, decode_combine_v2(B, H, Sv, D, 256, dt, pc)


def paged_uniform(B, NBLK, MAXP, Hkv, G, D, S, PAGE=16, packed=True, stage=None, kord=None, Nb=None, dt="fp16", pc="bits"):
    """Uniform splits, S of them, swap grid: returns (k1, k2)."""
    if stage is None:
        stage = "k_v" if packed is True or packed == "lbhnc" else "joint"
    if kord is None:
        kord = "xj" if packed in ("lhbnc", "rocm_hybrid") else "jx"
    k1 = decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32, "if", True, True, None, "kvg", packed, stage, kord, Nb, dt, pc)
    return k1, decode_combine_v2(B, Hkv * G, S, D, 256, dt, pc)


# Dispatch-time pick on BF16, page 784, layout LBNHC: (B * Hkv, MAXP) -> ((lower bound on r, uniform split count or 0 for paged_kernels), ...), first match wins. Fit on seeds 940-947, jobs 23508974, 23508975.
_PICK_BF16_784 = {(128, 64): ((0.82, 5), (0.56, 3), (0, 4)), (256, 64): ((0.70, 0), (0.58, 3), (0, 4)), (128, 128): ((0.82, 5), (0.70, 0), (0.58, 3), (0, 4)),
                  (256, 128): ((0.70, 0), (0.58, 6), (0, 4)), (64, 256): ((0.84, 10), (0.58, 6), (0, 8)), (32, 512): ((0.82, 10), (0.72, 12), (0.5, 0), (0, 24)),
                  (64, 64): ((0.66, 5), (0, 0)), (64, 128): ((0.66, 5), (0, 0)), (32, 256): ((0.66, 10), (0, 0)), (16, 512): ((0.66, 20), (0, 0))}  # these four fit at Hkv 2 (tensor parallel 2), seeds 948-950, jobs 23509050, 23509051


def paged_picker(B, NBLK, MAXP, Hkv, G, D, PAGE=16, packed=True, stage=None, kord=None, Nb=None, dt="fp16", pc="bits"):
    """Dispatch-time pick (N3, 6 Nov; evidence/N3): returns pick(sum_lens, max_lens) -> (k1, k2). The host knows the lengths, so no GPU read is needed. r = sum / (B * max).
    BF16, page 784, LBNHC, on a shape in _PICK_BF16_784: the table. The best split count moves with the active block count, not monotonically: 320, 640 and 1280 blocks beat 384, 512 and 768 by up to 17% (jobs 23508974, 23508975).
    Elsewhere (FP16, other pages and layouts), under 256 pages at batch 32 or more: uniform 4 splits for r below 0.62, uniform 3 for r below 0.8, else paged_kernels (the taper wins near-full batches,
    uniform 3 loses up to 21% there, jobs 23508631, 23508632), thresholds picked on FP16 seeds 900-923. Otherwise paged_kernels."""
    kk = paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, packed, stage, kord, Nb, dt, pc)
    tb = _PICK_BF16_784.get((B * Hkv, MAXP)) if dt == "bf16" and packed == "lbnhc" and (Nb or PAGE) == 784 else None
    if tb:
        F = {S: (paged_uniform(B, NBLK, MAXP, Hkv, G, D, S, PAGE, packed, stage, kord, Nb, dt, pc) if S else kk) for _, S in tb}
        return lambda sum_lens, max_lens: F[next(S for lo, S in tb if sum_lens >= lo * B * max_lens)]
    if MAXP >= 256 or B < 32 or Hkv != 4:      # at Hkv 2 the FP16 rule below loses to paged_kernels by up to 28% (32 x 2k skew, 23509050)
        return lambda sum_lens, max_lens: kk
    k3 = paged_uniform(B, NBLK, MAXP, Hkv, G, D, 3, PAGE, packed, stage, kord, Nb, dt, pc)
    k4 = paged_uniform(B, NBLK, MAXP, Hkv, G, D, 4, PAGE, packed, stage, kord, Nb, dt, pc)
    return lambda sum_lens, max_lens: k4 if sum_lens < 0.62 * B * max_lens else (k3 if sum_lens < 0.8 * B * max_lens else kk)


def run_fp8():
    """FP8 (e4m3, per-tensor scale) KV vs fp16 KV on the same data: bandwidth on the bytes actually read and accuracy
    vs float64 of the ORIGINAL fp16 cache (total cost) and of the dequantised cache (kernel error alone)."""
    from tilelang.profiler import do_bench
    torch.manual_seed(0)
    Hkv, G, D, PAGE = 4, 6, 256, 16
    H = Hkv * G
    fp8 = torch.float8_e4m3fn
    lut = torch.arange(256, dtype=torch.uint8, device="cuda").view(fp8).to(torch.float16)
    lut = torch.nan_to_num(lut, nan=0.0)
    for B, L in [(1, 131072), (8, 32768), (32, 8192), (64, 4096)]:
        lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
        if B > 1:
            lens[1::2] = L * 3 // 4
        MAXP = L // PAGE
        NBLK = B * MAXP
        Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
        Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
        BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
        ksc = (Kc.abs().amax().float() / 448.0).view(1)
        vsc = (Vc.abs().amax().float() / 448.0).view(1)
        K8 = (Kc.float() / ksc).to(fp8).view(torch.uint8)
        V8 = (Vc.float() / vsc).to(fp8).view(torch.uint8)
        Kdq = (K8.view(fp8).float() * ksc).half()
        Vdq = (V8.view(fp8).float() * vsc).half()
        small = B * L <= 32 * 8192
        R = ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G) if small else None
        Rq = ref64(Q, Kdq[BT.long()].view(B, L, Hkv, D), Vdq[BT.long()].view(B, L, Hkv, D), lens, G) if small else None
        nbytes8 = 2 * 1 * Hkv * D * int(lens.sum())
        S = max(1, min(512, 110 * 12 // (B * Hkv)))
        k2 = decode_combine(B, H, S, D)
        r = {"impl": "tilelang_decode_paged16_fp8", "B": B, "L": L, "splits": S}
        try:
            k1 = decode_partial_paged_fp8(B, NBLK, MAXP, Hkv, G, D, S)
            K32, V32 = K8.view(torch.int32), V8.view(torch.int32)    # same bytes, little-endian words
            run = lambda: k2(*k1(Q, K32, V32, lut, ksc, vsc, BT, lens))
            O = run()
            if small:
                r["rel_vs_fp64_original"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
                r["rel_vs_fp64_dequantised"] = round(((O.double() - Rq).norm() / Rq.norm()).item(), 6)
            ms = do_bench(run, warmup=10, rep=50)
            r.update(ms=round(ms, 4), gbps_fp8_bytes=round(nbytes8 / ms / 1e6, 1), pct_hbm=round(100 * nbytes8 / ms / 1e6 / 1600, 1),
                     fp16_equiv_gbps=round(2 * nbytes8 / ms / 1e6, 1))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            errs = [ln for ln in msg.splitlines() if "error" in ln.lower()][:4]
            r["err"] = (type(e).__name__ + ": " + " | ".join(errs) if errs else msg[-220:])[:900]
            print(json.dumps(r), flush=True)
            return
        print(json.dumps(r), flush=True)
        k1h = decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, S, 128, PAGE, 32)
        msh = do_bench(lambda: k2(*k1h(Q, Kc, Vc, BT, lens)), warmup=10, rep=50)
        print(json.dumps({"impl": "tilelang_decode_paged16_fp16", "B": B, "L": L, "splits": S, "ms": round(msh, 4)}), flush=True)


def run_paged():
    """Paged decode on layout LBHNC, the layout vLLM 0.30 uses only with VLLM_KV_CACHE_LAYOUT=HND (page 16, pages shuffled so the table matters):
    landed kernels per shape. vLLM's default on gfx90a is ROCM_ATTN on LHBNC; see _kv_layout."""
    from tilelang.profiler import do_bench
    torch.manual_seed(0)
    Hkv, G, D, PAGE = 4, 6, 256, 16
    H = Hkv * G
    for B, L in [(8, 32768), (16, 16384), (32, 8192), (64, 4096), (128, 2048), (64, 1024), (32, 1024)]:
        lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
        if B > 1:
            lens[1::2] = L * 3 // 4
        MAXP = L // PAGE
        NBLK = B * MAXP
        Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
        Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
        KV = torch.empty(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16)   # LBHNC: K then V in each row
        KV[..., :D] = Kc.transpose(1, 2)
        KV[..., D:] = Vc.transpose(1, 2)
        BT = torch.randperm(NBLK, device="cuda", dtype=torch.int64).to(torch.int32).view(B, MAXP)
        R = ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G) if B * L <= 32 * 8192 else None
        nbytes = 2 * 2 * Hkv * D * int(lens.sum())
        # N3 (6 Nov): landed form per shape; raw output in evidence/N3/raw
        k1, k2 = paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE)
        run = lambda: k2(*k1(Q, KV, KV, BT, lens))
        r = {"impl": "tilelang_decode_paged16_landed", "B": B, "L": L}
        O = run()
        if R is not None:
            r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
        ms = do_bench(run, warmup=10, rep=50)
        r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), pct_hbm=round(100 * nbytes / ms / 1e6 / 1600, 1))
        print(json.dumps(r), flush=True)


@tilelang.jit(out_idx=[4, 5])
def decode_partial_simt(B, Lmax, Hkv, G, D, S, bN, threads):
    """Same contract as decode_partial, without matrix cores: decode does ~2 FLOP per byte, so plain vector FMAs suffice
    and the block can use many threads (all of them load K/V tiles, which is what decode is limited by).
    Rows padded to GP = 8 (power of two) for layout inference."""
    H = Hkv * G
    GP = 8
    chunk = T.ceildiv(T.ceildiv(Lmax, S), bN) * bN               # chunks are whole tiles
    scale = (1.0 / D) ** 0.5
    dtype, acc = T.float16, T.float32

    @T.prim_func
    def main(Q: T.Tensor((B, H, D), dtype), K: T.Tensor((B, Lmax, Hkv, D), dtype), V: T.Tensor((B, Lmax, Hkv, D), dtype),
             Lens: T.Tensor((B,), T.int32), Op: T.Tensor((B, H, S, D), acc), Lse: T.Tensor((B, H, S), acc)):
        with T.Kernel(S, B * Hkv, threads=threads) as (s, byz):
            b = byz // Hkv
            hk = byz % Hkv
            n = Lens[b]
            k_lo = s * chunk
            k_hi = T.min(k_lo + chunk, n)
            q_s = T.alloc_shared((GP, D), dtype)
            K_s = T.alloc_shared((bN, D), dtype)
            V_s = T.alloc_shared((bN, D), dtype)
            sc = T.alloc_fragment((GP, bN), acc)
            acc_o = T.alloc_fragment((GP, D), acc)
            m_i = T.alloc_fragment((GP,), acc)
            m_prev = T.alloc_fragment((GP,), acc)
            l_i = T.alloc_fragment((GP,), acc)
            rs = T.alloc_fragment((GP,), acc)
            sf = T.alloc_fragment((GP,), acc)
            for i, d in T.Parallel(GP, D):
                q_s[i, d] = T.if_then_else(i < G, Q[b, hk * G + T.min(i, G - 1), d], T.cast(0, dtype))
            T.fill(acc_o, 0)
            T.fill(m_i, -T.infinity(acc))
            T.fill(l_i, 0)
            for t in T.serial(T.ceildiv(T.max(k_hi - k_lo, 0), bN)):
                k0 = k_lo + t * bN
                T.copy(K[b, k0:k0 + bN, hk, :], K_s)
                T.copy(V[b, k0:k0 + bN, hk, :], V_s)
                for j, d in T.Parallel(bN, D):                    # rows past the sequence end may hold anything (NaN)
                    if k0 + j >= k_hi:
                        V_s[j, d] = T.cast(0, dtype)
                for i, j in T.Parallel(GP, bN):
                    a = T.alloc_var(acc)
                    a = 0.0
                    for d in T.serial(D):
                        a += T.cast(q_s[i, d], acc) * T.cast(K_s[j, d], acc)
                    sc[i, j] = T.if_then_else(k0 + j < k_hi, a, -T.infinity(acc))
                T.copy(m_i, m_prev)
                T.reduce_max(sc, m_i, dim=1, clear=False)
                for i in T.Parallel(GP):
                    m_i[i] = T.max(m_i[i], m_prev[i])
                for i in T.Parallel(GP):
                    sf[i] = T.exp(m_prev[i] * scale - m_i[i] * scale)
                    l_i[i] *= sf[i]
                for i, j in T.Parallel(GP, bN):
                    sc[i, j] = T.exp(sc[i, j] * scale - m_i[i] * scale)
                T.reduce_sum(sc, rs, dim=1)
                for i in T.Parallel(GP):
                    l_i[i] += rs[i]
                for i, d in T.Parallel(GP, D):
                    o = T.alloc_var(acc)
                    o = acc_o[i, d] * sf[i]
                    for j in T.serial(bN):
                        o += sc[i, j] * T.cast(V_s[j, d], acc)
                    acc_o[i, d] = o
            for i, d in T.Parallel(GP, D):
                if i < G:
                    Op[b, hk * G + i, s, d] = T.if_then_else(l_i[i] > 0, acc_o[i, d] / l_i[i], 0.0)
            for i in T.Parallel(GP):
                if i < G:
                    Lse[b, hk * G + i, s] = T.if_then_else(l_i[i] > 0, m_i[i] * scale + T.log(l_i[i]), -T.infinity(acc))

    return main


@tilelang.jit(out_idx=[2])
def decode_combine(B, H, S, D, threads=256):
    acc = T.float32

    @T.prim_func
    def main(Op: T.Tensor((B, H, S, D), acc), Lse: T.Tensor((B, H, S), acc), O: T.Tensor((B, H, D), T.float16)):
        with T.Kernel(B * H, threads=threads) as bh:
            b = bh // H
            h = bh % H
            mx = T.alloc_var(acc)
            den = T.alloc_var(acc)
            mx = -T.infinity(acc)
            for s in T.serial(S):
                mx = T.max(mx, Lse[b, h, s])
            den = 0.0
            for s in T.serial(S):
                den += T.if_then_else(Lse[b, h, s] > -T.infinity(acc), T.exp(Lse[b, h, s] - mx), 0.0)
            for d in T.Parallel(D):
                o = T.alloc_var(acc)
                o = 0.0
                for s in T.serial(S):
                    o += T.if_then_else(Lse[b, h, s] > -T.infinity(acc), T.exp(Lse[b, h, s] - mx) * Op[b, h, s, d], 0.0)
                O[b, h, d] = T.cast(o / den, T.float16)

    return main


@tilelang.jit(out_idx=[2])
def decode_combine_v2(B, H, S, D, threads=256, dt="fp16", pc="bits"):
    """decode_combine with one exp per split per thread instead of two (N3)."""
    acc = T.float32
    dtype = {"fp16": T.float16, "bf16": T.bfloat16}[dt]

    @T.prim_func
    def main(Op: T.Tensor((B, H, S, D), acc), Lse: T.Tensor((B, H, S), acc), O: T.Tensor((B, H, D), dtype)):
        with T.Kernel(B * H, threads=threads) as bh:
            b = bh // H
            h = bh % H
            mx = T.alloc_var(acc)
            mx = -T.infinity(acc)
            for s in T.serial(S):
                mx = T.max(mx, Lse[b, h, s])
            for d in T.Parallel(D):
                o = T.alloc_var(acc)
                den = T.alloc_var(acc)
                w = T.alloc_var(acc)
                o = 0.0
                den = 0.0
                for s in T.serial(S):
                    w = T.if_then_else(Lse[b, h, s] > -T.infinity(acc), T.exp(Lse[b, h, s] - mx), 0.0)
                    den += w
                    o += w * Op[b, h, s, d]
                O[b, h, d] = _bf16_bits(o / den) if pc == "bits" and dt == "bf16" else T.cast(o / den, dtype)

    return main


def ref64(Q, K, V, lens, G):
    B, H, D = Q.shape
    out = torch.empty(B, H, D, dtype=torch.float64, device=Q.device)
    for b in range(B):
        n = int(lens[b])
        k = K[b, :n].double().repeat_interleave(G, 1)           # [n, H, D]
        v = V[b, :n].double().repeat_interleave(G, 1)
        s = torch.einsum("hd,nhd->hn", Q[b].double(), k) / math.sqrt(D)
        out[b] = torch.einsum("hn,nhd->hd", F.softmax(s, -1), v)
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fp8":
    run_fp8()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "paged":
    run_paged()
    sys.exit(0)

if __name__ == "__main__":
    from tilelang.profiler import do_bench
    torch.manual_seed(0)
    Hkv, G, D = 4, 6, 256
    H = Hkv * G
    cases = [(1, 131072), (8, 32768), (32, 8192), (64, 4096)]
    # round 1 (5 Nov): only (64 threads, bN 32, MFMA row) worked: exact, 1.06-7.2x SDPA, but 8-13% of HBM bandwidth.
    # round 2: the SIMT kernel was 10-50x slower (TileLang lowers the inner dot-product loops poorly); MFMA stays.
    # LDS per block = 2 * bN * D * 2 bytes; MI250X has 64 KB per CU, so bN 16 lets 4 blocks share a CU.
    # round 3 best: (64 thr, bN 16, 4-8 blocks/CU). Round 4: padded rows bM 32/64 with 128/256 threads.
    configs = [(64, 16, "row", 8, 16), (256, 16, "row", 4, 64), (256, 32, "row", 2, 64), (256, 32, "row", 4, 64),
               (128, 32, "row", 4, 32), (128, 16, "row", 8, 32), (256, 16, "row", 8, 64)]
    if len(sys.argv) > 1:
        configs = [tuple(json.loads(sys.argv[1]))]
    for B, L in cases:
        lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
        if B > 1:
            lens[1::2] = L * 3 // 4                                  # variable lengths
        Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
        R = ref64(Q, K, V, lens, G) if B * L <= 32 * 8192 else None
        nbytes = 2 * 2 * Hkv * D * int(lens.sum())                    # K + V actually read, fp16
        # SDPA baseline: uniform length (it has no per-sequence lengths without a mask), so it reads >= our bytes
        qt, kt, vt = Q[:, :, None, :], K.transpose(1, 2), V.transpose(1, 2)
        try:
            sms = do_bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True), warmup=10, rep=50)
            sd = {"ms": round(sms, 4), "gbps_on_our_bytes": round(nbytes / sms / 1e6, 1)}
        except Exception as e:  # noqa: BLE001
            sms, sd = None, {"err": str(e)[-120:]}
        print(json.dumps({"impl": "torch_sdpa", "B": B, "L": L, **sd}), flush=True)
        for cfg in configs:
            th, bN, pol, per_cu = cfg[:4]
            bMr = cfg[4] if len(cfg) > 4 else 16
            S = max(1, min(512, 110 * per_cu // (B * Hkv)))          # 110 CUs per GCD
            r = {"impl": "tilelang_decode", "B": B, "L": L, "splits": S, "threads": th, "bN": bN, "bM": bMr, "kernel": pol}
            try:
                k1 = decode_partial_simt(B, L, Hkv, G, D, S, bN, th) if pol == "simt" else decode_partial(B, L, Hkv, G, D, S, bN, th, pol, bMr)
                k2 = decode_combine(B, H, S, D)
                run = lambda: k2(*k1(Q, K, V, lens))
                O = run()
                if R is not None:
                    r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
                ms = do_bench(run, warmup=10, rep=50)
                r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), pct_hbm=round(100 * nbytes / ms / 1e6 / 1600, 1))
                if sms:
                    r["vs_sdpa"] = round(sms / ms, 2)
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
            print(json.dumps(r), flush=True)
