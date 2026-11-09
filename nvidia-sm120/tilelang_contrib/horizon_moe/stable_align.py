# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Stable MoE alignment: ascending flattened token-slot index within each expert.

Integer histogram atomics count memberships only; scheduling never selects an
output position. This standalone candidate does not change route selection or
weights. IDs outside [0, experts) are ignored. No serving integration yet.
"""
import tilelang
import tilelang.language as T

_SRC = r'''
__device__ __forceinline__ int horizon_pair_rank(int expert) {
  unsigned peers = __match_any_sync(0xffffffffu, expert);
  unsigned lane = threadIdx.x & 31;
  return __popc(peers & ((1u << lane) - 1u));
}
'''


def geometry(pairs, experts, block_size):
    if pairs < 1 or experts < 1 or block_size not in (8, 16, 32, 64):
        raise ValueError('positive pairs/experts and block size 8/16/32/64 required')
    chunks = (pairs + 31) // 32
    shared_bytes = 4 * (chunks * experts + 2 * experts + 1)
    if shared_bytes > 95 * 1024:
        raise ValueError('stable alignment exceeds the SM120 shared-memory budget')
    size = pairs + experts * (block_size - 1)
    if pairs < experts:
        size = min(pairs * block_size, size)
    return chunks, size, (size + block_size - 1) // block_size, shared_bytes


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def stable_align_kernel(P: int, E: int, BS: int, threads: int = 256):
    C, S, B, _ = geometry(P, E, BS)
    assert threads % 32 == 0
    W = threads // 32
    i32 = T.int32

    @T.prim_func
    def main(ids: T.Tensor((P,), i32), sorted_ids: T.Tensor((S,), i32),
             expert_ids: T.Tensor((B,), i32), ntp: T.Tensor((1,), i32)):
        with T.Kernel(1, threads=threads, prelude=_SRC):
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            hist = T.alloc_shared((C, E), i32)
            counts = T.alloc_shared((E,), i32)
            offsets = T.alloc_shared((E + 1,), i32)
            ex = T.alloc_var(i32)
            rank = T.alloc_var(i32)
            total = T.alloc_var(i32)
            for j in T.serial(T.ceildiv(C * E, threads)):
                h = j * threads + tid
                if h < C * E:
                    hist[h // E, h % E] = 0
            T.sync_threads()
            for j in T.serial(T.ceildiv(P, threads)):
                p = j * threads + tid
                if p < P:
                    ex = ids[p]
                    if T.And(ex >= 0, ex < E):
                        T.atomic_add(hist[p // 32, ex], 1)
            T.sync_threads()
            for j in T.serial(T.ceildiv(E, threads)):
                e = j * threads + tid
                if e < E:
                    total = 0
                    for c in T.serial(C):
                        total += hist[c, e]
                    counts[e] = T.ceildiv(total, BS) * BS
            T.sync_threads()
            if tid == 0:
                total = 0
                for e in T.serial(E):
                    offsets[e] = total
                    total += counts[e]
                offsets[E] = total
                ntp[0] = total
            for p in T.Parallel(S):
                sorted_ids[p] = P
            for b in T.Parallel(B):
                expert_ids[b] = -1
            T.sync_threads()
            for j in T.serial(T.ceildiv(E, threads)):
                e = j * threads + tid
                if e < E:
                    total = counts[e] // BS
                    for b in T.serial(total):
                        expert_ids[offsets[e] // BS + b] = e
            # Every physical warp executes the match intrinsic with all lanes,
            # including lanes beyond P in the last logical chunk.
            for j in T.serial(T.ceildiv(C, W)):
                chunk = j * W + warp
                p = chunk * 32 + lane
                ex = E
                if p < P:
                    ex = ids[p]
                rank = T.call_extern(i32, 'horizon_pair_rank', ex)
                if T.And(p < P, T.And(ex >= 0, ex < E)):
                    # Keep a uniform compile-time loop bound. A thread-varying
                    # T.serial(chunk) was mis-lowered for C=2 into an unconditional
                    # hist[0] addition, including logical chunk zero.
                    for c in T.serial(C):
                        if c < chunk:
                            rank += hist[c, ex]
                    sorted_ids[offsets[ex] + rank] = p
    return main


def align(ids, experts, block_size):
    """CUDA int32 contiguous expert IDs -> vLLM-compatible alignment buffers."""
    import torch
    if ids.dtype != torch.int32 or not ids.is_cuda or not ids.is_contiguous():
        raise ValueError('contiguous CUDA int32 IDs required')
    p = ids.numel()
    _, s, b, _ = geometry(p, experts, block_size)
    result = (torch.empty(s, device=ids.device, dtype=torch.int32),
              torch.empty(b, device=ids.device, dtype=torch.int32),
              torch.empty(1, device=ids.device, dtype=torch.int32))
    stable_align_kernel(p, experts, block_size)(ids.reshape(-1), *result)
    return result
