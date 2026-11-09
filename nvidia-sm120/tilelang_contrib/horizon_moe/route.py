# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""MoE router for Qwen3-Next-style blocks (softmax top-k over E routed experts + one shared expert gated by a sigmoid),
in ONE launch, optionally fused with Marlin-MoE's block alignment (vLLM's moe_align_block_size).

Input:  logits (M, LD) bf16, row-major; columns 0..E-1 are the routed-expert logits, column E the shared-expert gate
        logit (the router gate and the shared-expert gate computed as one GEMM); columns > E are ignored (padding).
Output: weights (M, K+1) fp32, ids (M, K+1) int32. Slots 0..K-1 = softmax top-K (optionally renormalised) over the E
        routed logits; slot K = (E, sigmoid(logit E) rounded to bf16).
route_align additionally writes, for the E+1 experts and Marlin's block size BS, exactly what
        vllm.model_executor.layers.fused_moe.moe_align_block_size returns: sorted_token_ids (S,) (per expert a segment
        of the flat pair indices t*(K+1)+slot routed to it, padded to a multiple of BS with the sentinel M*(K+1)),
        expert_ids (B,) (the expert of every BS-block, -1 after the last used block) and num_tokens_post_padded (1,).
        Pair order inside an expert's segment follows shared-memory atomics (as vLLM's own count_and_sort kernel); the
        MoE GEMMs compute every pair row independently, so their output does not depend on it.

route_shared replaces, per MoE layer and step: a contiguous copy of the routed logits, vLLM's topkGating kernel,
torch.sigmoid, a fill and two concatenations (6 launches -> 1). route_align (one CTA, M <= ALIGN_MAX_M) also replaces
moe_align_block_size_kernel + count_and_sort_expert_tokens_kernel (8 launches -> 1).

Numerics: bit-identical to vLLM 0.29's topk_softmax (csrc/libtorch_stable/moe/topk_softmax_kernels.cu, topkGating
<VPT=8, EXPERTS=256, BYTES_PER_LDG=16, WARP=32>) plus torch.sigmoid on bf16:
  - one warp per row; lane t owns experts [VPT*t, VPT*t + VPT) in ascending order;
  - row max: lane max, then butterfly max (xor 16, 8, 4, 2, 1);
  - e = expf(x - max) with the PRECISE expf (an extern call: TVM lowers T.exp to __expf), lane sum in ascending
    order starting from 0, butterfly sum (xor 16 .. 1); p = e * (1 / sum); NaN/Inf -> 0;
  - K rounds of argmax: lane argmax with strict '>' in ascending order, butterfly argmax where a larger value wins and
    ties go to the lower expert id; the winner is replaced by -10000 in its lane;
  - renormalise: selected_sum = sum of the K winners in selection order; scale = 1 / (sum > 0 ? sum : 1); w = p * scale;
  - shared slot: 1 / (1 + expf(-x)) in fp32, rounded to bf16 (torch.sigmoid on a bf16 tensor), then back to fp32.
Deterministic results, CUDA-graph capturable (dynamic sizes, no workspace, no global atomics), batch-invariant weights/ids.
"""

import tilelang
import tilelang.language as T

_KERNELS = {}
WARPS = 4
ALIGN_THREADS = 1024
ALIGN_MAX_M = 128


def _route_row(E: int, K: int, renorm: bool, with_sids: bool):
    """T.macro: route one row with one warp (lane = lane index). Writes Wt/Ids (and sids[row*(K+1)+k] if with_sids)."""
    VPT = E // 32
    f32, i32 = T.float32, T.int32

    @T.macro
    def body(logits, Wt, Ids, sids, row, lane):
        v = T.alloc_local((VPT,), f32)
        ch = T.alloc_local((VPT,), f32)
        ow = T.alloc_local((K,), f32)
        oe = T.alloc_local((K,), i32)
        mx = T.alloc_var(f32)
        s = T.alloc_var(f32)
        bv = T.alloc_var(f32)
        bw = T.alloc_var(f32)
        be = T.alloc_var(i32)
        xv = T.alloc_var(f32)
        xw = T.alloc_var(f32)
        xe = T.alloc_var(i32)
        sel = T.alloc_var(f32)
        scale = T.alloc_var(f32)
        for i in T.unroll(VPT):
            v[i] = T.cast(logits[row, lane * VPT + i], f32)
        mx = v[0]
        for i in T.unroll(VPT - 1):
            mx = T.max(mx, v[i + 1])
        for r in T.unroll(5):
            mx = T.max(mx, T.shfl_xor(mx, 16 >> r))
        s = T.float32(0)
        for i in T.unroll(VPT):
            v[i] = T.call_extern(f32, "expf", v[i] - mx)
            s = s + v[i]
        for r in T.unroll(5):
            s = s + T.shfl_xor(s, 16 >> r)
        s = T.float32(1) / s
        for i in T.unroll(VPT):
            v[i] = v[i] * s
            v[i] = T.if_then_else(T.Or(T.isnan(v[i]), T.isinf(v[i])), T.float32(0), v[i])
            ch[i] = v[i]
        sel = T.float32(0)
        for k in T.unroll(K):
            bv = ch[0]
            bw = v[0]
            be = lane * VPT
            for i in T.unroll(VPT - 1):
                if ch[i + 1] > bv:
                    bv = ch[i + 1]
                    bw = v[i + 1]
                    be = lane * VPT + i + 1
            for r in T.unroll(5):
                xv = T.shfl_xor(bv, 16 >> r)
                xw = T.shfl_xor(bw, 16 >> r)
                xe = T.shfl_xor(be, 16 >> r)
                if T.Or(xv > bv, T.And(xv == bv, xe < be)):
                    bv = xv
                    bw = xw
                    be = xe
            ow[k] = bw
            oe[k] = be
            sel = sel + bw
            if be // VPT == lane:
                for i in T.unroll(VPT):
                    if i == be % VPT:
                        ch[i] = T.float32(-10000)
        scale = T.float32(1)
        if renorm:
            scale = scale / T.if_then_else(sel > T.float32(0), sel, T.float32(1))
        if lane == 0:
            for k in T.unroll(K):
                Wt[row, k] = ow[k] * scale
                Ids[row, k] = oe[k]
                if with_sids:
                    sids[row * (K + 1) + k] = oe[k]
            xv = T.cast(logits[row, E], f32)
            xw = T.float32(1) / (T.float32(1) + T.call_extern(f32, "expf", -xv))
            Wt[row, K] = T.cast(T.cast(xw, T.bfloat16), f32)
            Ids[row, K] = E
            if with_sids:
                sids[row * (K + 1) + K] = E

    return body


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def route_shared_kernel(E: int, LD: int, K: int, renorm: bool, warps: int = WARPS):
    assert E % 32 == 0 and 1 <= E // 32 <= 16 and LD >= E + 1 and 1 <= K <= E, (E, LD, K)
    VPT = E // 32
    M = T.dynamic("M")
    body = _route_row(E, K, renorm, False)

    @T.prim_func
    def main(logits: T.Tensor((M, LD), T.bfloat16), Wt: T.Tensor((M, K + 1), T.float32),
             Ids: T.Tensor((M, K + 1), T.int32)):
        with T.Kernel(T.ceildiv(M, warps), threads=32 * warps) as bx:
            tid = T.get_thread_binding()
            row = bx * warps + tid // 32
            dummy = T.alloc_shared((1,), T.int32)
            if row < M:
                body(logits, Wt, Ids, dummy, row, tid % 32)

    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def route_align_kernel(E: int, LD: int, K: int, renorm: bool, BS: int, threads: int = ALIGN_THREADS,
                       max_m: int = ALIGN_MAX_M):
    """One CTA: route every row (warp w: rows w, w + W, ...), then count pairs per expert (E + 1 experts), exclusive
    scan of the BS-padded counts, expert_ids per block, sentinel fill, scatter of the pair indices."""
    assert E % 32 == 0 and 1 <= E // 32 <= 16 and LD >= E + 1 and 1 <= K <= E, (E, LD, K)
    VPT = E // 32
    EA = E + 1                                  # experts incl. the shared one
    W = threads // 32
    NW = (EA + 31) // 32                        # warps that hold an expert in the scan
    assert NW <= 32 and NW <= W and threads % 32 == 0
    MAXP = max_m * (K + 1)
    M = T.dynamic("M")
    S = T.dynamic("S")
    B = T.dynamic("B")
    i32 = T.int32
    body = _route_row(E, K, renorm, True)

    @T.prim_func
    def main(logits: T.Tensor((M, LD), T.bfloat16), Wt: T.Tensor((M, K + 1), T.float32),
             Ids: T.Tensor((M, K + 1), i32), sorted_ids: T.Tensor((S,), i32), expert_ids: T.Tensor((B,), i32),
             ntp: T.Tensor((1,), i32)):
        with T.Kernel(1, threads=threads) as bx:
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            sids = T.alloc_shared((MAXP,), i32)
            cnt = T.alloc_shared((NW * 32,), i32)
            cum = T.alloc_shared((NW * 32 + 1,), i32)
            cur = T.alloc_shared((NW * 32,), i32)
            wsum = T.alloc_shared((32,), i32)
            wpre = T.alloc_shared((32,), i32)
            x = T.alloc_var(i32)
            y = T.alloc_var(i32)
            c = T.alloc_var(i32)
            t = T.alloc_var(i32)
            z = T.alloc_var(i32)
            numel = M * (K + 1)
            if tid < NW * 32:
                cnt[tid] = 0
            # --- 1. route (warp-uniform row loop)
            for j in T.serial(T.ceildiv(M, W)):
                if j * W + warp < M:
                    body(logits, Wt, Ids, sids, j * W + warp, lane)
            T.sync_threads()
            # --- 2. count pairs per expert
            for j in T.serial(T.ceildiv(numel, threads)):
                if j * threads + tid < numel:
                    T.atomic_add(cnt[sids[j * threads + tid]], 1)
            T.sync_threads()
            # --- 3. exclusive scan of the padded counts (warp scans + a scan of the warp totals)
            c = 0
            if tid < EA:
                c = T.ceildiv(cnt[tid], BS) * BS
            x = c
            for r in T.unroll(5):
                y = T.shfl_up(x, 1 << r)
                if lane >= (1 << r):
                    x = x + y
            if T.And(lane == 31, warp < NW):
                wsum[warp] = x
            T.sync_threads()
            if warp == 0:           # (x holds this thread's prefix: use z as the shuffle temporary)
                t = 0
                if lane < NW:
                    t = wsum[lane]
                y = t
                for r in T.unroll(5):
                    z = T.shfl_up(y, 1 << r)
                    if lane >= (1 << r):
                        y = y + z
                wpre[lane] = y - t
            T.sync_threads()
            if tid < EA:
                x = wpre[warp] + x - c
            if tid < EA:
                cum[tid] = x
                cur[tid] = x
                if tid == EA - 1:
                    cum[EA] = x + c
                    ntp[0] = x + c
            T.sync_threads()
            # --- 4. expert id per block, -1 after the last used block; sentinel fill of sorted_ids
            x = 0
            y = 0
            if tid < EA:
                x = cum[tid] // BS
                y = cum[tid + 1] // BS
            t = cum[EA] // BS
            for j in T.serial(y - x):
                expert_ids[x + j] = tid
            for j in T.serial(T.ceildiv(B, threads)):
                if T.And(j * threads + tid >= t, j * threads + tid < B):
                    expert_ids[j * threads + tid] = -1
            for j in T.serial(T.ceildiv(S, threads)):
                if j * threads + tid < S:
                    sorted_ids[j * threads + tid] = numel
            T.sync_threads()
            # --- 5. scatter the pair indices into their expert's segment
            for j in T.serial(T.ceildiv(numel, threads)):
                if j * threads + tid < numel:
                    y = T.atomic_add(cur[sids[j * threads + tid]], 1, return_prev=True)
                    sorted_ids[y] = j * threads + tid

    return main


def get_kernel(E, LD, K, renorm):
    key = ("route", E, LD, K, bool(renorm))
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = route_shared_kernel(E, LD, K, bool(renorm))
    return k


def get_align_kernel(E, LD, K, renorm, BS):
    key = ("align", E, LD, K, bool(renorm), BS)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = route_align_kernel(E, LD, K, bool(renorm), BS)
    return k


def supported(E, LD, K):
    return E % 32 == 0 and 1 <= E // 32 <= 16 and LD >= E + 1 and 1 <= K <= E


def _as_rows(logits):
    M, LD = logits.shape
    if logits.stride(1) != 1 or logits.stride(0) != LD:
        logits = logits.contiguous()
    return logits


def route_shared(logits, E, K, renorm=True):
    """logits (M, LD >= E + 1) bf16 -> (weights (M, K+1) fp32, ids (M, K+1) int32)."""
    import torch
    logits = _as_rows(logits)
    M, LD = logits.shape
    w = torch.empty(M, K + 1, dtype=torch.float32, device=logits.device)
    ids = torch.empty(M, K + 1, dtype=torch.int32, device=logits.device)
    if M > 0:
        get_kernel(E, LD, K, renorm)(logits, w, ids)
    return w, ids


def align_sizes(numel, num_experts, block_size):
    """vLLM 0.29 moe_align_block_size's buffer sizes (pad_sorted_ids=False): (len(sorted_ids), len(expert_ids))."""
    s = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        s = min(numel * block_size, s)
    return s, (s + block_size - 1) // block_size


def route_align(logits, E, K, renorm, block_size):
    """route_shared + Marlin block alignment over E + 1 experts (M <= ALIGN_MAX_M).
    -> (weights, ids, (sorted_token_ids, expert_ids, num_tokens_post_padded))."""
    import torch
    logits = _as_rows(logits)
    M, LD = logits.shape
    assert 1 <= M <= ALIGN_MAX_M, M
    dev = logits.device
    s, b = align_sizes(M * (K + 1), E + 1, block_size)
    w = torch.empty(M, K + 1, dtype=torch.float32, device=dev)
    ids = torch.empty(M, K + 1, dtype=torch.int32, device=dev)
    sorted_ids = torch.empty(s, dtype=torch.int32, device=dev)
    expert_ids = torch.empty(b, dtype=torch.int32, device=dev)
    ntp = torch.empty(1, dtype=torch.int32, device=dev)
    get_align_kernel(E, LD, K, renorm, block_size)(logits, w, ids, sorted_ids, expert_ids, ntp)
    return w, ids, (sorted_ids, expert_ids, ntp)


def warm(E, LD, K, renorm, device, block_sizes=(8,)):
    """compile + load outside CUDA-graph capture (one binary for every M per kernel)."""
    import torch
    x = torch.zeros(4, LD, dtype=torch.bfloat16, device=device)
    route_shared(x, E, K, renorm)
    for bs in block_sizes:
        route_align(x, E, K, renorm, bs)
