"""Fused mixture-of-experts layer (grouped GEMM over expert-sorted tokens) for gfx90a (MI250X), TileLang.

Default shape: hidden 2048, expert intermediate 512, 256 experts, top-8 routing, read from the Qwen/Qwen3.6-35B-A3B config.
fp16 weights and activations, fp32 accumulate.

FusedMoE is the implementation to use. The layout came out of the E2 and N1 experiments (evidence/E2, evidence/N1):
  align : two kernels. Count, prefix sum and Bexp in LDS, then claim one slot per pair with a global atomic. At T <= 24 on
          clang >= 20 one kernel instead (align_fused: parallel scan, LDS slot claim), with FullCol decode tiles (evidence/MOE-MAX).
  k1    : X @ W_gate^T and X @ W_up^T, SiLU(gate) * up. Grid is (columns, blocks), so all columns of one block run in
          order. Row indices live in LDS, so the X load is branch-free. Padding rows read the last real row of X, so X
          needs no copy, and they are not written.
  k2    : Inter @ W_down^T, then weight. Sid and Wt are read once per thread. Padding rows go to a dump row Y[N].
  sum   : one kernel for the top-k reduction, no atomics.
Safe-memory legalization is off for every kernel, so every tile must divide the matrix (checked in FusedMoE).

Tiles depend on T and on the compiler. They were tuned on clang 18 (ROCm 6.2.4), and tiles_for has a separate clang >= 20
rule measured on clang 22 (the vLLM image, ROCm 7.2.3). The gate/up flags are compiler-specific. Clang 18 gets unroll 20000,
which groups the global loads of a K step; on clang 22 that flag serialises them and k1 gets about 35% slower. Clang 22's
default scheduler also serialises the small tile, and -misched=gcn-max-ilp groups it again (20 loads as one run at T = 16).
Same code object, same result in either runtime (evidence/N1), so this is scheduling, not stack. Any change to the gate/up
source can move this, so re-time it after editing that kernel. Use -misched=, not -amdgpu-sched-strategy=: that one is ignored
for gcn-max-* values and aborts hipcc on clang 18.

The large tile on clang >= 20 also partitions gate/up warps Square (2 x 2, 32 x 32 per warp, the shape of the 32x32x8 MFMA), not FullRow: 4 to 5%
less k1 at T = 8192 on synthetic data and 0.17 to 0.20 ms on real layers. On clang 18 it gains too little to adopt, and Square on the down
kernel or on a 128 x 128 tile loses on both (evidence/N1).

Round 2 on clang >= 20 (evidence/MOE-MAX): T = 1 runs GEMV kernels per pair (no align, no MFMA), T = 3 and 4 GEMV kernels per
expert block over a one-kernel align. The 96-row band path ('dual') and the parallel-scan align (al='ps') are opt-in only. Every
other T, clang 18 and unknown clang keep the kernels above unchanged.

Round 3 on clang >= 20 (evidence/MOE-MAX): T = 1 fuses the top-k sum into the GEMV down kernel (2 launches); T = 2 runs GEMV
kernels without align, each expert's first pair computing every pair on that expert (dedup). The gate/up with two tile shapes
in one launch (gate_up_mix: an expert with 65 to 96 rows left takes one 96-row block, every other block the 64-row path; layout,
align, down and sum unchanged) is opt-in only since round 5 (cfg={"gu_mix": 96}): it lost on skewed routing (see _MIX_RANGE_C22).

moe_gate_up, moe_down and moe_align_gpu are the earlier forms, kept for scaling/ep_moe.py.

Usage: python moe_fused.py             correctness against an fp32 reference and timing at each T
"""
import argparse
import contextlib
import json
import math
import os
import re
import subprocess

import torch
import torch.nn.functional as F
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.transform import PassConfigKey as PK

DTYPE, ACC = T.float16, T.float32
POL = {"row": GemmWarpPolicy.FullRow, "square": GemmWarpPolicy.Square, "col": GemmWarpPolicy.FullCol}
NO_SAFE = {PK.TL_DISABLE_SAFE_MEMORY_ACCESS: True}
_SMALL = dict(bM=16, bN=32, bK=128, th=64, dN=64, dK=128, dth=64, kp=2, al_th=1024, sum_th=256, gu_unroll=20000)
_MID = dict(bM=64, bN=64, bK=128, th=512, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=None)
_LARGE = dict(bM=64, bN=64, bK=64, th=256, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=20000)
_ILP = ("-mllvm", "-misched=gcn-max-ilp")
_SMALL_C22 = dict(_SMALL, gu_unroll=None, gu_flags=_ILP)
_LARGE_C22 = dict(_LARGE, gu_unroll=None, gu_flags=_ILP, gu_pol="square")
_SMALL_MAX_C22 = 512
# Decode size on clang >= 20 (evidence/MOE-MAX): one-kernel align, and 16-row tiles shared by 4 warps (FullCol). T = 1 and 7 to 24
# take both the FullCol gate/up and down (except _DEC_NOT_C22, round 6); T = 2 to 6 keep the 1-warp gate/up (FullCol gate/up
# loses there, and on zipf2 at 5 to 6) and take only the FullCol down. T > 24 unchanged: the FullCol gate/up loses 2 to 8% on skewed routing at T = 32 to 256.
_DEC_C22 = dict(_SMALL_C22, bN=64, th=256, gu_pol="col", dth=256, dn_pol="col", al="fused")
_DEC26_C22 = dict(_SMALL_C22, dth=256, dn_pol="col", al="fused")
_DEC_MAX_C22 = 24
# Round 2 (evidence/MOE-MAX, clang >= 20). T = 1: GEMV gate/up and down per pair, no align (top-k experts of one token are
# distinct, so no weight is read twice). T = 3 and 4: one-kernel align with Tn slots per expert block and GEMV kernels per expert
# block (weights read once per expert on any routing). Opt-in only, not dispatched: path 'dual' (cfg={"path": "dual", "bA": 96})
# runs gate/up on 96-row blocks written into the landed 64-row layout, 3 to 10% faster at T = 2048 to 3000 on uniform routing but
# 0.3 to 2.6% slower on zipf1/zipf2; al='ps' (parallel-scan align, 2 to 5 us less per call) wins eager but reads 0.5 to 1% slower
# under graph replay at T >= 2400.
_GEMV1_C22 = dict(_DEC_C22, path="gemv", cpg=1, rpg=4)
_GEMVR_C22 = dict(_DEC_C22, path="gemv_rows", cpg=1, rpg=4)
_GEMVR_MIN_C22, _GEMVR_MAX_C22 = 3, 4   # T = 2 lost 0.5 to 1.5% on zipf1 (eager) and zipf2 (graph) in both verify jobs: not taken
_DUAL_C22 = dict(_LARGE_C22, path="dual", bA=96)
_DUAL_RANGE_C22 = None   # not dispatched (see above)
# Round 3 (evidence/MOE-MAX, clang >= 20).
_GEMV1S_C22 = dict(_DEC_C22, path="gemv_s", cpg=1, rpg=4)    # T = 1: down + top-k sum in one kernel
_GEMVDD_C22 = dict(_DEC_C22, path="gemv_dd", cpg=1, rpg=4)   # T = 2: GEMV with per-expert dedup, no align
_GEMVDD_TS_C22 = (2,)
# Opt-in only, not dispatched (round 5): gate/up with 64- and 96-row blocks in one launch (cfg={"gu_mix": 96} on the large tile).
# Against the 64-row-only kernel at T = 2200 to 4096, 2 opposite-order jobs x 2 routing seeds: uniform 0.90 to 0.99 (parity at
# 3600 to 3700, 1.004 to 1.011 at 3300), but zipf1/zipf2 lost beyond the identical-code spread in both jobs at 2200 zipf1 (1.010 to
# 1.011, seed 8), 2700 zipf2 (1.009 to 1.011, seed 7) and 3300 zipf2 (1.006 to 1.008, seed 7); round 3 saw up to 1.016 (2875
# zipf1, graph) and 1.012 (2400 zipf2). No T range wins on every routing and dispatch cannot see routing, so it stays off.
_MIX_RANGE_C22 = None
_LARGE_MIX_C22 = dict(_LARGE_C22, gu_mix=96)
# Round 4 (evidence/MOE-MAX, clang >= 20). Small T: gate/up GEMV then gemv_dn_sumT (down + top-k sum in one kernel), 2 launches.
# T = 2 (dedup gate/up, 2 output rows per down block): eager 0.91 to 0.96 of the 3-launch dedup path, graph 0.94 to 0.96.
# T = 3: eager 0.83 to 0.98, graph 0.97 to 0.98 of the per-expert-block GEMV path. T = 4: mixed (graph 0.99 to 1.03), not taken.
_GEMVDS_C22 = dict(_DEC_C22, path="gemv_ds", gu="dd", cpg=1, rpg=2)
_GEMVDS_TS_C22 = (2, 3)
# Round 6 (evidence/MOE-MAX, clang >= 20). Raced against the pre-workflow module (9ad0c345, _SMALL_C22 with the two-kernel align)
# under graph replay, 2 opposite-order jobs x seeds 7 and 8 x 3 routings. The FullCol decode tile lost or tied on zipf2 at these T
# (landed/head 0.981-1.011 on zipf2; at each of these T at least one zipf2 cell fails head/landed > 1 + the identical-code
# spread, 1.34% and 0.97% in the two jobs), so they take the pre-workflow tile again. Its five kernel sources equal 9ad0c345's
# (md5 per kernel, both jobs). T = 1-8, 20, 23 and 24 keep the round 1-4 paths: 0.731-0.976 of head in every cell, fp32 too.
_DEC_NOT_C22 = (13, 15, 16, 17, 19, 21)


def tiles_for(Tn, clang):
    """Measured at the T values in evidence/N1. Clang >= 20 uses the small tile only with its scheduler flag: without it the
    tile is 2x slower in k1. Clang unknown (0) takes the unflagged mid and large rule, which holds on both compilers."""
    if clang == 0:
        return dict(_MID if Tn <= 1024 else _LARGE)
    if clang >= 20:
        if Tn == 1:
            return dict(_GEMV1S_C22)
        if Tn in _GEMVDS_TS_C22:
            return dict(_GEMVDS_C22)
        if Tn in _GEMVDD_TS_C22:
            return dict(_GEMVDD_C22)
        if _GEMVR_MIN_C22 <= Tn <= _GEMVR_MAX_C22:
            return dict(_GEMVR_C22)
        if Tn in _DEC_NOT_C22:
            return dict(_SMALL_C22)
        if Tn <= _DEC_MAX_C22:
            return dict(_DEC26_C22 if 2 <= Tn <= 6 else _DEC_C22)
        if _DUAL_RANGE_C22 and _DUAL_RANGE_C22[0] <= Tn <= _DUAL_RANGE_C22[1]:
            return dict(_DUAL_C22)
        if _MIX_RANGE_C22 and _MIX_RANGE_C22[0] <= Tn <= _MIX_RANGE_C22[1] and Tn > 1024:
            return dict(_LARGE_MIX_C22)
        return dict(_SMALL_C22 if Tn <= _SMALL_MAX_C22 else _MID if Tn <= 1024 else _LARGE_C22)
    return dict(_SMALL if Tn <= 256 else _MID if Tn <= 1024 else _LARGE)


def clang_major():
    """Major version of the clang behind hipcc, or 0 if unknown."""
    from tilelang.contrib.rocm import find_hipcc
    out = subprocess.run([find_hipcc(), "--version"], capture_output=True, text=True).stdout
    m = re.search(r"clang version (\d+)", out)
    return int(m.group(1)) if m else 0


def pmax_rows(N, E, bM):
    """Upper bound on padded rows. At most min(N, E) experts are hit, so small T launches far fewer blocks."""
    return (N + min(N, E) * (bM - 1) + bM - 1) // bM * bM


def align_count(N, E, bM, threads=1024):
    """Block 0 counts experts in LDS, writes each group's first slot to Cur and fills Bexp. Block 1 fills Sid with the
    sentinel N. Unused blocks get expert -1."""
    P = pmax_rows(N, E, bM)
    NB = P // bM

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(2, threads=threads) as b:
            cnt = T.alloc_shared((E,), T.int32)
            start = T.alloc_shared((E + 1,), T.int32)
            tx = T.get_thread_binding()
            if b == 1:
                for r in T.serial(T.ceildiv(P, threads)):
                    if r * threads + tx < P:
                        Sid[r * threads + tx] = N
            else:
                for r in T.serial(T.ceildiv(E, threads)):
                    if r * threads + tx < E:
                        cnt[r * threads + tx] = 0
                for r in T.serial(T.ceildiv(NB, threads)):
                    if r * threads + tx < NB:
                        Bexp[r * threads + tx] = -1
                T.sync_threads()
                for r in T.serial(T.ceildiv(N, threads)):
                    if r * threads + tx < N:
                        T.atomic_add(cnt[Ids[r * threads + tx]], 1)
                T.sync_threads()
                if tx == 0:
                    acc_ = T.alloc_var(T.int32)
                    acc_ = 0
                    for e in T.serial(E):
                        start[e] = acc_
                        Cur[e] = acc_
                        acc_ += (cnt[e] + bM - 1) // bM * bM
                    start[E] = acc_
                T.sync_threads()
                for r in T.serial(T.ceildiv(E, threads)):
                    e = r * threads + tx
                    if e < E:
                        nb = T.alloc_var(T.int32)
                        nb = (start[e + 1] - start[e]) // bM
                        for blk in T.serial(nb):
                            Bexp[start[e] // bM + blk] = e

    return main


def align_count_ps(N, E, bM, threads=1024):
    """align_count with the prefix over E as a parallel scan (Hillis-Steele) instead of one thread's serial loop
    (about 8 us at E = 256, profiled in evidence/MOE-MAX). Same outputs; align_slots runs after it unchanged."""
    P = pmax_rows(N, E, bM)
    NB = P // bM
    assert E <= threads
    LOG = max(1, (E - 1).bit_length())

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(2, threads=threads) as b:
            cnt = T.alloc_shared((E,), T.int32)
            sc = T.alloc_shared((E,), T.int32)
            tmp = T.alloc_shared((E,), T.int32)
            tx = T.get_thread_binding()
            if b == 1:
                for r in T.serial(T.ceildiv(P, threads)):
                    if r * threads + tx < P:
                        Sid[r * threads + tx] = N
            if b == 0:
                if tx < E:
                    cnt[tx] = 0
                for r in T.serial(T.ceildiv(NB, threads)):
                    if r * threads + tx < NB:
                        Bexp[r * threads + tx] = -1
                T.sync_threads()
                for r in T.serial(T.ceildiv(N, threads)):
                    if r * threads + tx < N:
                        T.atomic_add(cnt[Ids[r * threads + tx]], 1)
                T.sync_threads()
                if tx < E:
                    sc[tx] = (cnt[tx] + bM - 1) // bM * bM
                T.sync_threads()
                for d in T.serial(LOG):
                    if tx < E:
                        tmp[tx] = T.if_then_else(tx >= T.shift_left(1, d), sc[T.max(tx - T.shift_left(1, d), 0)], 0)
                    T.sync_threads()
                    if tx < E:
                        sc[tx] = sc[tx] + tmp[tx]
                    T.sync_threads()
                if tx < E:
                    nb = T.alloc_var(T.int32)
                    nb = (cnt[tx] + bM - 1) // bM
                    Cur[tx] = sc[tx] - nb * bM
                    for blk in T.serial(nb):
                        Bexp[(sc[tx] - nb * bM) // bM + blk] = tx

    return main


def align_fused(N, E, bM, threads=1024):
    """One-block align for decode size: count, parallel scan of the padded counts (Hillis-Steele over E), Bexp, sentinel fill,
    slot claim with LDS atomics. The two-kernel align above prefixes E on one thread, about 8 us at T = 1 (profiled,
    evidence/MOE-MAX). Pair order inside a group is not fixed, as in the two-kernel form."""
    P = pmax_rows(N, E, bM)
    NB = P // bM
    assert E <= threads
    LOG = max(1, (E - 1).bit_length())
    RE = T.ceildiv(E, threads)

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32)):
        with T.Kernel(1, threads=threads) as _:
            cnt = T.alloc_shared((E,), T.int32)
            sc = T.alloc_shared((E,), T.int32)
            tmp = T.alloc_shared((E,), T.int32)
            cur = T.alloc_shared((E,), T.int32)
            tx = T.get_thread_binding()
            for r in T.serial(RE):
                if r * threads + tx < E:
                    cnt[r * threads + tx] = 0
            for r in T.serial(T.ceildiv(P, threads)):
                if r * threads + tx < P:
                    Sid[r * threads + tx] = N
            for r in T.serial(T.ceildiv(NB, threads)):
                if r * threads + tx < NB:
                    Bexp[r * threads + tx] = -1
            T.sync_threads()
            for r in T.serial(T.ceildiv(N, threads)):
                if r * threads + tx < N:
                    T.atomic_add(cnt[Ids[r * threads + tx]], 1)
            T.sync_threads()
            for r in T.serial(RE):
                if r * threads + tx < E:
                    sc[r * threads + tx] = (cnt[r * threads + tx] + bM - 1) // bM * bM
            T.sync_threads()
            for d in T.serial(LOG):
                for r in T.serial(RE):
                    if r * threads + tx < E:
                        tmp[r * threads + tx] = T.if_then_else(r * threads + tx >= T.shift_left(1, d), sc[T.max(r * threads + tx - T.shift_left(1, d), 0)], 0)
                T.sync_threads()
                for r in T.serial(RE):
                    if r * threads + tx < E:
                        sc[r * threads + tx] = sc[r * threads + tx] + tmp[r * threads + tx]
                T.sync_threads()
            for r in T.serial(RE):
                e = r * threads + tx
                if e < E:
                    nb = T.alloc_var(T.int32)
                    nb = (cnt[e] + bM - 1) // bM
                    cur[e] = sc[e] - nb * bM
                    for blk in T.serial(nb):
                        Bexp[(sc[e] - nb * bM) // bM + blk] = e
            T.sync_threads()
            for r in T.serial(T.ceildiv(N, threads)):
                i = r * threads + tx
                if i < N:
                    slot = T.atomic_add(cur[Ids[i]], 1, return_prev=True)
                    Sid[slot] = i

    return main


def align_slots(N, E, bM, threads=256):
    """Claim a slot per pair with a global atomic."""
    P = pmax_rows(N, E, bM)

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as b:
            tx = T.get_thread_binding()
            i = b * threads + tx
            if i < N:
                slot = T.atomic_add(Cur[Ids[i]], 1, return_prev=True)
                Sid[slot] = i

    return main


def gate_up(Tn, topk, H, I, E, P, bM, bN, bK, th, kp, x="last", pol="row"):
    """Where padding rows read X from:
      x='zero': a zero row Tn. X is [Tn + 1, H], so the caller or FusedMoE must provide that row.
      x='last': the last real row. X is [Tn, H].
    In both, padding rows are not written to Inter."""
    NB = P // bM
    N = Tn * topk
    Xrows = Tn + 1 if x == "zero" else Tn

    @T.prim_func
    def main(X: T.Tensor((Xrows, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Inter: T.Tensor((P, I), DTYPE)):
        with T.Kernel(T.ceildiv(I, bN), NB, threads=th) as (bn, bm):
            rows = T.alloc_shared((bM,), T.int32)
            if x != "zero":
                real = T.alloc_shared((bM,), T.int32)
            xs = T.alloc_shared((bM, bK), DTYPE)
            gs = T.alloc_shared((bN, bK), DTYPE)
            us = T.alloc_shared((bN, bK), DTYPE)
            ag = T.alloc_fragment((bM, bN), ACC)
            au = T.alloc_fragment((bM, bN), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                for i in T.Parallel(bM):
                    if x == "zero":
                        rows[i] = T.min(Sid[bm * bM + i], N) // topk
                    else:
                        rows[i] = T.min(Sid[bm * bM + i], N - 1) // topk
                        real[i] = Sid[bm * bM + i]
                T.clear(ag)
                T.clear(au)
                for k in T.serial(T.ceildiv(H, bK)):
                    for i, kk in T.Parallel(bM, bK):
                        xs[i, kk] = X[rows[i], k * bK + kk]
                    T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                    T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                    T.gemm(xs, gs, ag, transpose_B=True, k_pack=kp, policy=POL[pol])
                    T.gemm(xs, us, au, transpose_B=True, k_pack=kp, policy=POL[pol])
                for i, j in T.Parallel(bM, bN):
                    g = ag[i, j]
                    if x == "zero":
                        if rows[i] < Tn:
                            Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], DTYPE)
                    else:
                        if real[i] < N:
                            Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], DTYPE)

    return main


def down(Tn, topk, H, I, E, P, bM, bN, bK, th, kp, pol="row"):
    """Y is [N + 1, H]; row N is a dump row for padding, so the write loop has no branch per element. pol 'col' lets 4 warps
    share a 16-row decode tile."""
    NB = P // bM
    N = Tn * topk

    @T.prim_func
    def main(Inter: T.Tensor((P, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Wt: T.Tensor((N,), ACC), Y: T.Tensor((N + 1, H), DTYPE)):
        with T.Kernel(T.ceildiv(H, bN), NB, threads=th) as (bn, bm):
            a_s = T.alloc_shared((bM, bK), DTYPE)
            w_s = T.alloc_shared((bN, bK), DTYPE)
            ac = T.alloc_fragment((bM, bN), ACC)
            sf = T.alloc_fragment((bM,), T.int32)
            wf = T.alloc_fragment((bM,), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                T.clear(ac)
                for k in T.serial(T.ceildiv(I, bK)):
                    T.copy(Inter[bm * bM:bm * bM + bM, k * bK:k * bK + bK], a_s)
                    T.copy(W2[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], w_s)
                    T.gemm(a_s, w_s, ac, transpose_B=True, k_pack=kp, policy=POL[pol])
                for i in T.Parallel(bM):
                    s0 = Sid[bm * bM + i]
                    sf[i] = T.min(s0, N)
                    wf[i] = T.if_then_else(s0 < N, Wt[T.min(s0, N - 1)], T.cast(0, ACC))
                for i, j in T.Parallel(bM, bN):
                    Y[sf[i], bn * bN + j] = T.cast(ac[i, j] * wf[i], DTYPE)

    return main


def gemv_gu(Tn, topk, H, I, E, cpg=1, th=256):
    N = Tn * topk
    G = th // 32
    CJ = G * cpg
    VK = H // 256
    assert I % CJ == 0 and H % 256 == 0 and th % 32 == 0
    R = 2 * cpg

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Ids: T.Tensor((N,), T.int32),
             Inter: T.Tensor((N, I), DTYPE)):
        with T.Kernel(I // CJ, N, threads=th) as (bj, p):
            tx = T.get_thread_binding()
            xf = T.alloc_local((VK * 8,), ACC)
            wb = T.alloc_local((R * VK * 8,), ACC)
            acc = T.alloc_local((R * 8,), ACC)
            red = T.alloc_local((R,), ACC)
            e = Ids[p]
            t = p // topk
            l = tx % 32
            c0 = bj * CJ + (tx // 32) * cpg
            for v in T.unroll(VK):
                for j in T.vectorized(8):
                    xf[v * 8 + j] = T.cast(X[t, (v * 32 + l) * 8 + j], ACC)
            for r in T.unroll(R):
                for v in T.unroll(VK):
                    for j in T.vectorized(8):
                        wb[(r * VK + v) * 8 + j] = T.cast(W13[e, (r % 2) * I + c0 + r // 2, (v * 32 + l) * 8 + j], ACC)
            for q in T.unroll(R * 8):
                acc[q] = T.cast(0, ACC)
            for r in T.unroll(R):
                for v in T.unroll(VK):
                    for j in T.unroll(8):
                        acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VK + v) * 8 + j] * xf[v * 8 + j]
            for r in T.unroll(R):
                red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                red[r] = T.warp_reduce_sum(red[r])
            if l == 0:
                for c in T.unroll(cpg):
                    g = red[2 * c]
                    Inter[p, c0 + c] = T.cast(g / (1.0 + T.exp(-g)) * red[2 * c + 1], DTYPE)

    return main

def gemv_dn(Tn, topk, H, I, E, rpg=4, th=256):
    N = Tn * topk
    G = th // 32
    CH = G * rpg
    VI = I // 256
    assert H % CH == 0 and I % 256 == 0

    @T.prim_func
    def main(Inter: T.Tensor((N, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Ids: T.Tensor((N,), T.int32),
             Wt: T.Tensor((N,), ACC), Y: T.Tensor((N, H), DTYPE)):
        with T.Kernel(H // CH, N, threads=th) as (bh, p):
            tx = T.get_thread_binding()
            af = T.alloc_local((VI * 8,), ACC)
            wb = T.alloc_local((rpg * VI * 8,), ACC)
            acc = T.alloc_local((rpg * 8,), ACC)
            red = T.alloc_local((rpg,), ACC)
            e = Ids[p]
            l = tx % 32
            h0 = bh * CH + (tx // 32) * rpg
            for v in T.unroll(VI):
                for j in T.vectorized(8):
                    af[v * 8 + j] = T.cast(Inter[p, (v * 32 + l) * 8 + j], ACC)
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.vectorized(8):
                        wb[(r * VI + v) * 8 + j] = T.cast(W2[e, h0 + r, (v * 32 + l) * 8 + j], ACC)
            for q in T.unroll(rpg * 8):
                acc[q] = T.cast(0, ACC)
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.unroll(8):
                        acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VI + v) * 8 + j] * af[v * 8 + j]
            for r in T.unroll(rpg):
                red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                red[r] = T.warp_reduce_sum(red[r])
            if l == 0:
                w = Wt[p]
                for r in T.unroll(rpg):
                    Y[p, h0 + r] = T.cast(red[r] * w, DTYPE)

    return main

def gemv_gu_rows(Tn, topk, H, I, E, P, R, cpg=1, th=256):
    """GEMV gate/up over the align layout with R slots per block (R = Tn: one block per hit expert), so each expert's
    weights are read once whatever the routing; each lane loads its weight vectors once and applies them to every real row."""
    N = Tn * topk
    NB = P // R
    G = th // 32
    CJ = G * cpg
    VK = H // 256
    C2 = 2 * cpg

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Inter: T.Tensor((P, I), DTYPE)):
        with T.Kernel(I // CJ, NB, threads=th) as (bj, bm):
            tx = T.get_thread_binding()
            xf = T.alloc_local((VK * 8,), ACC)
            wb = T.alloc_local((C2 * VK * 8,), ACC)
            acc = T.alloc_local((C2 * 8,), ACC)
            red = T.alloc_local((C2,), ACC)
            l = tx % 32
            c0 = bj * CJ + (tx // 32) * cpg
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                for r in T.unroll(C2):
                    for v in T.unroll(VK):
                        for j in T.vectorized(8):
                            wb[(r * VK + v) * 8 + j] = T.cast(W13[e, (r % 2) * I + c0 + r // 2, (v * 32 + l) * 8 + j], ACC)
                for i in T.unroll(R):
                    s = Sid[bm * R + i]
                    if s < N:
                        for v in T.unroll(VK):
                            for j in T.vectorized(8):
                                xf[v * 8 + j] = T.cast(X[s // topk, (v * 32 + l) * 8 + j], ACC)
                        for q in T.unroll(C2 * 8):
                            acc[q] = T.cast(0, ACC)
                        for r in T.unroll(C2):
                            for v in T.unroll(VK):
                                for j in T.unroll(8):
                                    acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VK + v) * 8 + j] * xf[v * 8 + j]
                        for r in T.unroll(C2):
                            red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                            red[r] = T.warp_reduce_sum(red[r])
                        if l == 0:
                            for c in T.unroll(cpg):
                                g = red[2 * c]
                                Inter[bm * R + i, c0 + c] = T.cast(g / (1.0 + T.exp(-g)) * red[2 * c + 1], DTYPE)

    return main

def gemv_dn_rows(Tn, topk, H, I, E, P, R, rpg=4, th=256):
    N = Tn * topk
    NB = P // R
    G = th // 32
    CH = G * rpg
    VI = I // 256

    @T.prim_func
    def main(Inter: T.Tensor((P, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Wt: T.Tensor((N,), ACC), Y: T.Tensor((N, H), DTYPE)):
        with T.Kernel(H // CH, NB, threads=th) as (bh, bm):
            tx = T.get_thread_binding()
            af = T.alloc_local((VI * 8,), ACC)
            wb = T.alloc_local((rpg * VI * 8,), ACC)
            acc = T.alloc_local((rpg * 8,), ACC)
            red = T.alloc_local((rpg,), ACC)
            l = tx % 32
            h0 = bh * CH + (tx // 32) * rpg
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                for r in T.unroll(rpg):
                    for v in T.unroll(VI):
                        for j in T.vectorized(8):
                            wb[(r * VI + v) * 8 + j] = T.cast(W2[e, h0 + r, (v * 32 + l) * 8 + j], ACC)
                for i in T.unroll(R):
                    s = Sid[bm * R + i]
                    if s < N:
                        for v in T.unroll(VI):
                            for j in T.vectorized(8):
                                af[v * 8 + j] = T.cast(Inter[bm * R + i, (v * 32 + l) * 8 + j], ACC)
                        for q in T.unroll(rpg * 8):
                            acc[q] = T.cast(0, ACC)
                        for r in T.unroll(rpg):
                            for v in T.unroll(VI):
                                for j in T.unroll(8):
                                    acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VI + v) * 8 + j] * af[v * 8 + j]
                        for r in T.unroll(rpg):
                            red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                            red[r] = T.warp_reduce_sum(red[r])
                        if l == 0:
                            w = Wt[s]
                            for r in T.unroll(rpg):
                                Y[s, h0 + r] = T.cast(red[r] * w, DTYPE)

    return main

def gate_up_mix(Tn, topk, H, I, E, P, bN, bK, th, kp, pol="square", big=96, pol_big=None, alias=True, bM=64):
    """Landed gate/up (64-row blocks, x='last') with a second tile shape in the same launch. Rows left = Cur[e] - bM * bm
    (Cur[e] is the end of expert e's real rows after align_slots). 64 < rows left <= big: this block runs big rows, its own
    and the first big - 64 rows of the expert's last block (which then exits). Otherwise the landed bM-row path. Same layout,
    align, down and sum as the landed path. alias: the bM-row path uses rows 0..bM-1 of the big X tile (one LDS buffer, so
    LDS stays at 2 blocks per CU)."""
    NB = P // bM
    N = Tn * topk
    pb = POL[pol_big or pol]
    assert bM < big <= 2 * bM

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Cur: T.Tensor((E,), T.int32), Inter: T.Tensor((P, I), DTYPE)):
        with T.Kernel(T.ceildiv(I, bN), NB, threads=th) as (bn, bm):
            rows = T.alloc_shared((big,), T.int32)
            real = T.alloc_shared((big,), T.int32)
            xb = T.alloc_shared((big, bK), DTYPE)
            if not alias:
                xm = T.alloc_shared((bM, bK), DTYPE)
            gs = T.alloc_shared((bN, bK), DTYPE)
            us = T.alloc_shared((bN, bK), DTYPE)
            agb = T.alloc_fragment((big, bN), ACC)
            aub = T.alloc_fragment((big, bN), ACC)
            agm = T.alloc_fragment((bM, bN), ACC)
            aum = T.alloc_fragment((bM, bN), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                rem = Cur[e] - bm * bM
                if rem > bM and rem <= big:
                    for i in T.Parallel(big):
                        rows[i] = T.min(Sid[bm * bM + i], N - 1) // topk
                        real[i] = Sid[bm * bM + i]
                    T.clear(agb)
                    T.clear(aub)
                    for k in T.serial(T.ceildiv(H, bK)):
                        for i, kk in T.Parallel(big, bK):
                            xb[i, kk] = X[rows[i], k * bK + kk]
                        T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                        T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                        T.gemm(xb, gs, agb, transpose_B=True, k_pack=kp, policy=pb)
                        T.gemm(xb, us, aub, transpose_B=True, k_pack=kp, policy=pb)
                    for i, j in T.Parallel(big, bN):
                        g = agb[i, j]
                        if real[i] < N:
                            Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * aub[i, j], DTYPE)
                else:
                    if rem > big - bM or bm == 0 or Bexp[T.max(bm - 1, 0)] != e:
                        for i in T.Parallel(bM):
                            rows[i] = T.min(Sid[bm * bM + i], N - 1) // topk
                            real[i] = Sid[bm * bM + i]
                        T.clear(agm)
                        T.clear(aum)
                        for k in T.serial(T.ceildiv(H, bK)):
                            if alias:
                                for i, kk in T.Parallel(bM, bK):
                                    xb[i, kk] = X[rows[i], k * bK + kk]
                            else:
                                for i, kk in T.Parallel(bM, bK):
                                    xm[i, kk] = X[rows[i], k * bK + kk]
                            T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                            T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                            if alias:
                                T.gemm(xb[0:bM, :], gs, agm, transpose_B=True, k_pack=kp, policy=POL[pol])
                                T.gemm(xb[0:bM, :], us, aum, transpose_B=True, k_pack=kp, policy=POL[pol])
                            else:
                                T.gemm(xm, gs, agm, transpose_B=True, k_pack=kp, policy=POL[pol])
                                T.gemm(xm, us, aum, transpose_B=True, k_pack=kp, policy=POL[pol])
                        for i, j in T.Parallel(bM, bN):
                            g = agm[i, j]
                            if real[i] < N:
                                Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * aum[i, j], DTYPE)

    return main


def gemv_gu_dd(Tn, topk, H, I, E, cpg=1, th=256):
    N = Tn * topk
    G = th // 32
    CJ = G * cpg
    VK = H // 256
    C2 = 2 * cpg

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Ids: T.Tensor((N,), T.int32),
             Inter: T.Tensor((N, I), DTYPE)):
        with T.Kernel(I // CJ, N, threads=th) as (bj, p):
            tx = T.get_thread_binding()
            xf = T.alloc_local((VK * 8,), ACC)
            wb = T.alloc_local((C2 * VK * 8,), ACC)
            acc = T.alloc_local((C2 * 8,), ACC)
            red = T.alloc_local((C2,), ACC)
            dup = T.alloc_var(T.int32)
            e = Ids[p]
            l = tx % 32
            c0 = bj * CJ + (tx // 32) * cpg
            dup = 0
            for q in T.serial(p):
                if Ids[q] == e:
                    dup = 1
            if dup == 0:
                for r in T.unroll(C2):
                    for v in T.unroll(VK):
                        for j in T.vectorized(8):
                            wb[(r * VK + v) * 8 + j] = T.cast(W13[e, (r % 2) * I + c0 + r // 2, (v * 32 + l) * 8 + j], ACC)
                for q in T.serial(N - p):
                    if Ids[p + q] == e:
                        for v in T.unroll(VK):
                            for j in T.vectorized(8):
                                xf[v * 8 + j] = T.cast(X[(p + q) // topk, (v * 32 + l) * 8 + j], ACC)
                        for z in T.unroll(C2 * 8):
                            acc[z] = T.cast(0, ACC)
                        for r in T.unroll(C2):
                            for v in T.unroll(VK):
                                for j in T.unroll(8):
                                    acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VK + v) * 8 + j] * xf[v * 8 + j]
                        for r in T.unroll(C2):
                            red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                            red[r] = T.warp_reduce_sum(red[r])
                        if l == 0:
                            for c in T.unroll(cpg):
                                g = red[2 * c]
                                Inter[p + q, c0 + c] = T.cast(g / (1.0 + T.exp(-g)) * red[2 * c + 1], DTYPE)

    return main


def gemv_dn_dd(Tn, topk, H, I, E, rpg=4, th=256):
    N = Tn * topk
    G = th // 32
    CH = G * rpg
    VI = I // 256

    @T.prim_func
    def main(Inter: T.Tensor((N, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Ids: T.Tensor((N,), T.int32),
             Wt: T.Tensor((N,), ACC), Y: T.Tensor((N, H), DTYPE)):
        with T.Kernel(H // CH, N, threads=th) as (bh, p):
            tx = T.get_thread_binding()
            af = T.alloc_local((VI * 8,), ACC)
            wb = T.alloc_local((rpg * VI * 8,), ACC)
            acc = T.alloc_local((rpg * 8,), ACC)
            red = T.alloc_local((rpg,), ACC)
            dup = T.alloc_var(T.int32)
            e = Ids[p]
            l = tx % 32
            h0 = bh * CH + (tx // 32) * rpg
            dup = 0
            for q in T.serial(p):
                if Ids[q] == e:
                    dup = 1
            if dup == 0:
                for r in T.unroll(rpg):
                    for v in T.unroll(VI):
                        for j in T.vectorized(8):
                            wb[(r * VI + v) * 8 + j] = T.cast(W2[e, h0 + r, (v * 32 + l) * 8 + j], ACC)
                for q in T.serial(N - p):
                    if Ids[p + q] == e:
                        for v in T.unroll(VI):
                            for j in T.vectorized(8):
                                af[v * 8 + j] = T.cast(Inter[p + q, (v * 32 + l) * 8 + j], ACC)
                        for z in T.unroll(rpg * 8):
                            acc[z] = T.cast(0, ACC)
                        for r in T.unroll(rpg):
                            for v in T.unroll(VI):
                                for j in T.unroll(8):
                                    acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VI + v) * 8 + j] * af[v * 8 + j]
                        for r in T.unroll(rpg):
                            red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                            red[r] = T.warp_reduce_sum(red[r])
                        if l == 0:
                            w = Wt[p + q]
                            for r in T.unroll(rpg):
                                Y[p + q, h0 + r] = T.cast(red[r] * w, DTYPE)

    return main


def gemv_dn_sum(Tn, topk, H, I, E, rpg=4):
    """T = 1 down GEMV with the top-k sum fused: one block per rpg output rows, warp g (32 lanes) computes pair g's dot
    products for those rows, partial results go to LDS and threads 0..rpg-1 sum the topk pairs in fixed order and write O
    (fp16). Same weight bytes and block count as gemv_dn; no Y round trip and no sum launch."""
    assert Tn == 1
    N = topk
    th = topk * 32
    VI = I // 256
    assert H % rpg == 0 and I % 256 == 0 and rpg <= th

    @T.prim_func
    def main(Inter: T.Tensor((N, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Ids: T.Tensor((N,), T.int32),
             Wt: T.Tensor((N,), ACC), O: T.Tensor((Tn, H), DTYPE)):
        with T.Kernel(H // rpg, threads=th) as bh:
            tx = T.get_thread_binding()
            part = T.alloc_shared((topk * rpg,), ACC)
            af = T.alloc_local((VI * 8,), ACC)
            wb = T.alloc_local((rpg * VI * 8,), ACC)
            acc = T.alloc_local((rpg * 8,), ACC)
            red = T.alloc_local((rpg,), ACC)
            sm = T.alloc_var(ACC)
            g = tx // 32
            l = tx % 32
            e = Ids[g]
            h0 = bh * rpg
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.vectorized(8):
                        wb[(r * VI + v) * 8 + j] = T.cast(W2[e, h0 + r, (v * 32 + l) * 8 + j], ACC)
            for v in T.unroll(VI):
                for j in T.vectorized(8):
                    af[v * 8 + j] = T.cast(Inter[g, (v * 32 + l) * 8 + j], ACC)
            for z in T.unroll(rpg * 8):
                acc[z] = T.cast(0, ACC)
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.unroll(8):
                        acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VI + v) * 8 + j] * af[v * 8 + j]
            for r in T.unroll(rpg):
                red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                red[r] = T.warp_reduce_sum(red[r])
            if l == 0:
                w = Wt[g]
                for r in T.unroll(rpg):
                    part[g * rpg + r] = red[r] * w
            T.sync_threads()
            if tx < rpg:
                sm = part[tx]
                for q in T.unroll(topk - 1):
                    sm = sm + part[(q + 1) * rpg + tx]
                O[0, h0 + tx] = T.cast(sm, DTYPE)

    return main


def gemv_dn_sumT(Tn, topk, H, I, E, rpg=4):
    """Down GEMV with the top-k sum fused, for Tn * topk * 32 <= 1024 threads (gemv_dn_sum is the T = 1 form). One block per rpg
    output rows; 32-lane group g computes pair g, so pairs that share an expert read the same weight rows at the same time
    (the second read is served by cache). Partials go to LDS and Tn * rpg threads sum each token's topk pairs in fixed order
    and write O (fp16). No Y round trip and no sum launch."""
    N = Tn * topk
    th = N * 32
    VI = I // 256
    assert H % rpg == 0 and I % 256 == 0 and th <= 1024 and Tn * rpg <= th

    @T.prim_func
    def main(Inter: T.Tensor((N, I), DTYPE), W2: T.Tensor((E, H, I), DTYPE), Ids: T.Tensor((N,), T.int32),
             Wt: T.Tensor((N,), ACC), O: T.Tensor((Tn, H), DTYPE)):
        with T.Kernel(H // rpg, threads=th) as bh:
            tx = T.get_thread_binding()
            part = T.alloc_shared((N * rpg,), ACC)
            af = T.alloc_local((VI * 8,), ACC)
            wb = T.alloc_local((rpg * VI * 8,), ACC)
            acc = T.alloc_local((rpg * 8,), ACC)
            red = T.alloc_local((rpg,), ACC)
            sm = T.alloc_var(ACC)
            g = tx // 32
            l = tx % 32
            e = Ids[g]
            h0 = bh * rpg
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.vectorized(8):
                        wb[(r * VI + v) * 8 + j] = T.cast(W2[e, h0 + r, (v * 32 + l) * 8 + j], ACC)
            for v in T.unroll(VI):
                for j in T.vectorized(8):
                    af[v * 8 + j] = T.cast(Inter[g, (v * 32 + l) * 8 + j], ACC)
            for z in T.unroll(rpg * 8):
                acc[z] = T.cast(0, ACC)
            for r in T.unroll(rpg):
                for v in T.unroll(VI):
                    for j in T.unroll(8):
                        acc[r * 8 + j] = acc[r * 8 + j] + wb[(r * VI + v) * 8 + j] * af[v * 8 + j]
            for r in T.unroll(rpg):
                red[r] = ((acc[r * 8] + acc[r * 8 + 1]) + (acc[r * 8 + 2] + acc[r * 8 + 3])) + ((acc[r * 8 + 4] + acc[r * 8 + 5]) + (acc[r * 8 + 6] + acc[r * 8 + 7]))
                red[r] = T.warp_reduce_sum(red[r])
            if l == 0:
                w = Wt[g]
                for r in T.unroll(rpg):
                    part[g * rpg + r] = red[r] * w
            T.sync_threads()
            if tx < Tn * rpg:
                t = tx // rpg
                r0 = tx % rpg
                sm = part[(t * topk) * rpg + r0]
                for q in T.unroll(topk - 1):
                    sm = sm + part[(t * topk + q + 1) * rpg + r0]
                O[t, h0 + r0] = T.cast(sm, DTYPE)

    return main


def gate_up_x(Tn, topk, H, I, E, PA, PB, bM, bN, bK, th, kp, pol="square"):
    """Landed gate/up (x='last') on the bM-row layout A, writing each real row to its slot in layout B (PosB), so the
    landed down kernel runs unchanged on layout B."""
    NB = PA // bM
    N = Tn * topk

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), W13: T.Tensor((E, 2 * I, H), DTYPE), Sid: T.Tensor((PA,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), PosB: T.Tensor((N,), T.int32), Inter: T.Tensor((PB, I), DTYPE)):
        with T.Kernel(T.ceildiv(I, bN), NB, threads=th) as (bn, bm):
            rows = T.alloc_shared((bM,), T.int32)
            real = T.alloc_shared((bM,), T.int32)
            dst = T.alloc_shared((bM,), T.int32)
            xs = T.alloc_shared((bM, bK), DTYPE)
            gs = T.alloc_shared((bN, bK), DTYPE)
            us = T.alloc_shared((bN, bK), DTYPE)
            ag = T.alloc_fragment((bM, bN), ACC)
            au = T.alloc_fragment((bM, bN), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                for i in T.Parallel(bM):
                    rows[i] = T.min(Sid[bm * bM + i], N - 1) // topk
                    real[i] = Sid[bm * bM + i]
                    dst[i] = PosB[T.min(Sid[bm * bM + i], N - 1)]
                T.clear(ag)
                T.clear(au)
                for k in T.serial(T.ceildiv(H, bK)):
                    for i, kk in T.Parallel(bM, bK):
                        xs[i, kk] = X[rows[i], k * bK + kk]
                    T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                    T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                    T.gemm(xs, gs, ag, transpose_B=True, k_pack=kp, policy=POL[pol])
                    T.gemm(xs, us, au, transpose_B=True, k_pack=kp, policy=POL[pol])
                for i, j in T.Parallel(bM, bN):
                    g = ag[i, j]
                    if real[i] < N:
                        Inter[dst[i], bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], DTYPE)

    return main


def align_dual(N, E, bA, bB, PA, PB, threads=1024):
    """Count once, then prefix the padded counts for block sizes bA (gate/up layout A) and bB (down layout B) with one
    parallel scan (Hillis-Steele over E; threads [0, E) scan A, [E, 2E) scan B). Start*/Cur* hold each expert's first slot,
    Bexp* the expert of each block. Blocks 1 and 2 fill SidA and SidB with the sentinel N."""
    NA, NBB = PA // bA, PB // bB
    assert 2 * E <= threads
    LOG = max(1, (E - 1).bit_length())

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), SidA: T.Tensor((PA,), T.int32), BexpA: T.Tensor((NA,), T.int32),
             CurA: T.Tensor((E,), T.int32), StartA: T.Tensor((E,), T.int32), SidB: T.Tensor((PB,), T.int32),
             BexpB: T.Tensor((NBB,), T.int32), StartB: T.Tensor((E,), T.int32)):
        with T.Kernel(3, threads=threads) as b:
            cnt = T.alloc_shared((E,), T.int32)
            sc = T.alloc_shared((2 * E,), T.int32)
            tmp = T.alloc_shared((2 * E,), T.int32)
            tx = T.get_thread_binding()
            if b == 1:
                for r in T.serial(T.ceildiv(PA, threads)):
                    if r * threads + tx < PA:
                        SidA[r * threads + tx] = N
            if b == 2:
                for r in T.serial(T.ceildiv(PB, threads)):
                    if r * threads + tx < PB:
                        SidB[r * threads + tx] = N
            if b == 0:
                if tx < E:
                    cnt[tx] = 0
                for r in T.serial(T.ceildiv(NA, threads)):
                    if r * threads + tx < NA:
                        BexpA[r * threads + tx] = -1
                for r in T.serial(T.ceildiv(NBB, threads)):
                    if r * threads + tx < NBB:
                        BexpB[r * threads + tx] = -1
                T.sync_threads()
                for r in T.serial(T.ceildiv(N, threads)):
                    if r * threads + tx < N:
                        T.atomic_add(cnt[Ids[r * threads + tx]], 1)
                T.sync_threads()
                if tx < E:
                    sc[tx] = (cnt[tx] + bA - 1) // bA * bA
                if E <= tx and tx < 2 * E:
                    sc[tx] = (cnt[tx - E] + bB - 1) // bB * bB
                T.sync_threads()
                for d in T.serial(LOG):
                    if tx < 2 * E:
                        tmp[tx] = T.if_then_else(tx % E >= T.shift_left(1, d), sc[T.max(tx - T.shift_left(1, d), 0)], 0)
                    T.sync_threads()
                    if tx < 2 * E:
                        sc[tx] = sc[tx] + tmp[tx]
                    T.sync_threads()
                if tx < E:
                    na = T.alloc_var(T.int32)
                    na = (cnt[tx] + bA - 1) // bA
                    CurA[tx] = sc[tx] - na * bA
                    StartA[tx] = sc[tx] - na * bA
                    for blk in T.serial(na):
                        BexpA[(sc[tx] - na * bA) // bA + blk] = tx
                if E <= tx and tx < 2 * E:
                    nb = T.alloc_var(T.int32)
                    nb = (cnt[tx - E] + bB - 1) // bB
                    StartB[tx - E] = sc[tx] - nb * bB
                    for blk in T.serial(nb):
                        BexpB[(sc[tx] - nb * bB) // bB + blk] = tx - E

    return main


def align_dual_slots(N, E, PA, PB, threads=256):
    """One atomic per pair claims its slot in layout A; its rank inside the expert's group is the same in layout B, so the B
    slot is derived (StartB - StartA offset). PosB[i] is pair i's row in layout B."""

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), SidA: T.Tensor((PA,), T.int32), CurA: T.Tensor((E,), T.int32),
             StartA: T.Tensor((E,), T.int32), SidB: T.Tensor((PB,), T.int32), StartB: T.Tensor((E,), T.int32),
             PosB: T.Tensor((N,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as b:
            tx = T.get_thread_binding()
            i = b * threads + tx
            if i < N:
                e = Ids[i]
                sl = T.atomic_add(CurA[e], 1, return_prev=True)
                SidA[sl] = i
                sl2 = sl - StartA[e] + StartB[e]
                SidB[sl2] = i
                PosB[i] = sl2

    return main


def topk_sum(Tn, topk, H, th=256, otype=None):
    """Each token is a block and each thread owns H // th columns. One writer per output, so no atomics. Larger th means
    fewer contiguous bytes per thread per row: 512 beat 256 at T = 4096, and 128 and below were far slower."""
    otype = otype or ACC
    per = H // th

    @T.prim_func
    def main(Y: T.Tensor((Tn * topk, H), DTYPE), O: T.Tensor((Tn, H), otype)):
        with T.Kernel(Tn, threads=th) as t:
            tx = T.get_thread_binding()
            s = T.alloc_local((per,), ACC)
            for j in T.vectorized(per):
                s[j] = T.cast(Y[t * topk, tx * per + j], ACC)
            for k in T.serial(1, topk):
                for j in T.vectorized(per):
                    s[j] = s[j] + T.cast(Y[t * topk + k, tx * per + j], ACC)
            for j in T.vectorized(per):
                O[t, tx * per + j] = T.cast(s[j], otype)

    return main


@contextlib.contextmanager
def _hipcc_flags(flags):
    """TileLang 0.1.15 ignores compile_flags on the HIP path, so flags reach hipcc through its own env var. They still go
    to tilelang.compile, because that is what keeps them in the kernel cache key."""
    old = os.environ.get("HIPCC_COMPILE_FLAGS_APPEND")
    if flags:
        os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = " ".join(flags)
    try:
        yield
    finally:
        if flags:
            if old is None:
                del os.environ["HIPCC_COMPILE_FLAGS_APPEND"]
            else:
                os.environ["HIPCC_COMPILE_FLAGS_APPEND"] = old


def _build(fn, out_idx=None, flags=None):
    with _hipcc_flags(flags):
        return tilelang.compile(fn, out_idx=out_idx, pass_configs=NO_SAFE, compile_flags=list(flags) if flags else None)


class FusedMoE:
    """One layer call for a fixed T. Kernels are specialised on T, so keep one instance per token count.
    cfg: overrides for tiles_for(Tn). gu_unroll in it applies on clang older than 20 only; gu_flags are extra hipcc flags for
         gate/up on any compiler, and gu_pol its warp policy; both are dropped if cfg changes the gate/up tile without naming them; dn_unroll is off by default (no gain at T = 4096 once down uses 64x64x128, and slower at
         T = 1024 with the old tile).
    x: 'last' (default, no copy of X) or 'zero' as in gate_up. With 'zero', X is copied into a kept [Tn + 1, H] buffer on
       each call, unless the caller already passes [Tn + 1, H] with row Tn zero, or x_copy='cat' asks for a new torch.cat
       per call. Both copies cost more than they save: evidence/N1.
    out: 'fp16' matches vLLM's output type, 'fp32' keeps the sum's precision and writes twice the bytes."""

    def __init__(self, Tn, topk=8, H=2048, I=512, E=256, cfg=None, x="last", x_copy="buffer", out="fp16", device="cuda"):
        clang = clang_major()
        c = dict(tiles_for(Tn, clang), **(cfg or {}))
        if cfg and "gu_flags" not in cfg and any(k in cfg for k in ("bM", "bN", "bK", "th")):
            c["gu_flags"] = None      # the table's flags were timed on the table's tile only
        if cfg and "gu_pol" not in cfg and any(k in cfg for k in ("bM", "bN", "bK", "th")):
            c.pop("gu_pol", None)     # same for the warp policy
        if cfg and "dn_pol" not in cfg and any(k in cfg for k in ("bM", "dN", "dK", "dth")):
            c.pop("dn_pol", None)     # and for the down policy; the one-kernel align holds for any tile
        if (cfg and "path" not in cfg and any(k in cfg for k in ("bM", "bN", "bK", "th", "dN", "dK", "dth"))) or x != "last":
            c.pop("path", None)       # the decode and band paths were timed on their own kernels, x='last' only
        if (cfg and "gu_mix" not in cfg and any(k in cfg for k in ("bM", "bN", "bK", "th", "al"))) or x != "last":
            c.pop("gu_mix", None)     # the mixed gate/up was timed on the landed 64-row tile with the two-kernel align only
        if c.get("path") == "gemv_s" and out != "fp16":
            c["path"] = "gemv"        # the fused sum writes fp16 only
        if c.get("path") == "gemv_ds" and out != "fp16":
            c = dict(c, **(_GEMVDD_C22 if Tn in _GEMVDD_TS_C22 else _GEMVR_C22))   # same: the round-3 kernels for this T
        self.path = c.get("path", "mfma")
        self.mix = False
        self.Tn, self.topk, self.H, self.I, self.E, self.cfg, self.x, self.x_copy = Tn, topk, H, I, E, c, x, x_copy
        self.N = N = Tn * topk
        if H % c["bK"] or I % c["bN"] or I % c["dK"] or H % c["dN"] or H % c["sum_th"]:
            raise ValueError(f"tiles must divide the matrix (safe-memory is off): {c}")
        self.P = P = pmax_rows(N, E, c["bM"])
        self.NB = P // c["bM"]
        self.unroll = c["gu_unroll"] if c["gu_unroll"] and 0 < clang < 20 else None
        self.gu_flags = tuple(c.get("gu_flags") or ())
        flags = lambda u: ("-mllvm", f"-unroll-threshold={u}") if u else None
        self.fused_align = c.get("al") == "fused"
        self.dev = device
        otype = DTYPE if out == "fp16" else ACC
        if self.path in ("gemv", "gemv_rows", "gemv_s", "gemv_dd", "gemv_ds"):
            ilp = self.gu_flags or None
            if self.path == "gemv_ds":
                self.ka = self.kb = self.ks = None
                self.k1 = _build((gemv_gu_dd if c.get("gu", "dd") == "dd" else gemv_gu)(Tn, topk, H, I, E, c["cpg"]), [3], ilp)
                self.k2 = _build(gemv_dn_sumT(Tn, topk, H, I, E, c["rpg"]), [4], ilp)
                return
            if self.path == "gemv_s":
                self.ka = self.kb = self.ks = None
                self.k1 = _build(gemv_gu(Tn, topk, H, I, E, c["cpg"]), [3], ilp)
                self.k2 = _build(gemv_dn_sum(Tn, topk, H, I, E, c["rpg"]), [4], ilp)
                return
            if self.path == "gemv_dd":
                self.ka = self.kb = None
                self.k1 = _build(gemv_gu_dd(Tn, topk, H, I, E, c["cpg"]), [3], ilp)
                self.k2 = _build(gemv_dn_dd(Tn, topk, H, I, E, c["rpg"]), [4], ilp)
            elif self.path == "gemv":
                self.ka = self.kb = None
                self.k1 = _build(gemv_gu(Tn, topk, H, I, E, c["cpg"]), [3], ilp)
                self.k2 = _build(gemv_dn(Tn, topk, H, I, E, c["rpg"]), [4], ilp)
            else:
                self.P = pmax_rows(N, E, Tn)
                self.NB = self.P // Tn
                self.ka, self.kb = _build(align_fused(N, E, Tn, c["al_th"])), None
                self.k1 = _build(gemv_gu_rows(Tn, topk, H, I, E, self.P, Tn, c["cpg"]), [4], ilp)
                self.k2 = _build(gemv_dn_rows(Tn, topk, H, I, E, self.P, Tn, c["rpg"]), [5], ilp)
            self.ks = _build(topk_sum(Tn, topk, H, 256, otype), [1])
            return
        if self.path == "dual":
            bA = c["bA"]
            self.PA = pmax_rows(N, E, bA)
            self.NA = self.PA // bA
            self.ka = _build(align_dual(N, E, bA, c["bM"], self.PA, P, c["al_th"]))
            self.kb = _build(align_dual_slots(N, E, self.PA, P, min(256, c["al_th"])))
            self.k1 = _build(gate_up_x(Tn, topk, H, I, E, self.PA, P, bA, c["bN"], c["bK"], c["th"], c["kp"], c.get("gu_pol", "row")), [5], self.gu_flags or None)
            self.k2 = _build(down(Tn, topk, H, I, E, P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"], c.get("dn_pol", "row")), [5], flags(c.get("dn_unroll")))
            self.ks = _build(topk_sum(Tn, topk, H, c["sum_th"], otype), [1])
            return
        if self.fused_align:
            self.ka, self.kb = _build(align_fused(N, E, c["bM"], c["al_th"])), None
        else:
            self.ka = _build((align_count_ps if c.get("al") == "ps" else align_count)(N, E, c["bM"], c["al_th"]))
            self.kb = _build(align_slots(N, E, c["bM"], min(256, c["al_th"])))
        self.mix = bool(c.get("gu_mix")) and not self.fused_align and x == "last"
        if self.mix:
            self.k1 = _build(gate_up_mix(Tn, topk, H, I, E, P, c["bN"], c["bK"], c["th"], c["kp"], c.get("gu_pol", "row"), c["gu_mix"], None, True, c["bM"]), [5], (flags(self.unroll) or ()) + self.gu_flags or None)
        else:
            self.k1 = _build(gate_up(Tn, topk, H, I, E, P, c["bM"], c["bN"], c["bK"], c["th"], c["kp"], x, c.get("gu_pol", "row")), [4], (flags(self.unroll) or ()) + self.gu_flags or None)
        self.k2 = _build(down(Tn, topk, H, I, E, P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"], c.get("dn_pol", "row")), [5], flags(c.get("dn_unroll")))
        self.ks = _build(topk_sum(Tn, topk, H, c["sum_th"], otype), [1])
        self._zr = torch.zeros(1, H, dtype=torch.float16, device=device)
        self._Xz = None
        if x == "zero" and x_copy == "buffer":
            self._Xz = torch.zeros(Tn + 1, H, dtype=torch.float16, device=device)

    def align(self, idf):
        s = torch.empty((self.P,), dtype=torch.int32, device=self.dev)
        b = torch.empty((self.NB,), dtype=torch.int32, device=self.dev)
        if self.fused_align:
            self.ka(idf, s, b)
            return s, b
        c = torch.empty((self.E,), dtype=torch.int32, device=self.dev)
        self.ka(idf, s, b, c)
        self.kb(idf, s, c)
        self._cur = c
        return s, b

    def _x(self, X):
        if self.x != "zero" or X.shape[0] == self.Tn + 1:
            return X
        if self.x_copy == "cat":
            return torch.cat([X, self._zr])
        self._Xz[: self.Tn].copy_(X)
        return self._Xz

    def __call__(self, X, W13, W2, ids, wts, rec=None):
        """X [Tn, H] fp16, W13 [E, 2I, H], W2 [E, H, I], ids [Tn, topk] int32, wts [Tn, topk] fp32. rec(i) marks stages."""
        rec = rec or (lambda i: None)
        idf, wf = ids.flatten(), wts.flatten()
        rec(0)
        if self.path in ("gemv_s", "gemv_ds"):
            rec(1)
            inter = self.k1(X, W13, idf)
            rec(2)
            o = self.k2(inter, W2, idf, wf)
            rec(3)
            rec(4)
            return o
        if self.path in ("gemv", "gemv_dd"):
            rec(1)
            inter = self.k1(X, W13, idf)
            rec(2)
            y = self.k2(inter, W2, idf, wf)
            rec(3)
            o = self.ks(y)
            rec(4)
            return o
        if self.path == "dual":
            e = lambda n: torch.empty((n,), dtype=torch.int32, device=self.dev)
            sa, ba, ca, sta, sb, bb, stb, pos = e(self.PA), e(self.NA), e(self.E), e(self.E), e(self.P), e(self.NB), e(self.E), e(self.N)
            self.ka(idf, sa, ba, ca, sta, sb, bb, stb)
            self.kb(idf, sa, ca, sta, sb, stb, pos)
            rec(1)
            inter = self.k1(X, W13, sa, ba, pos)
            rec(2)
            y = self.k2(inter, W2, sb, bb, wf)
            rec(3)
            o = self.ks(y[: self.N])
            rec(4)
            return o
        xx = self._x(X)
        s, be = self.align(idf)
        rec(1)
        inter = self.k1(xx, W13, s, be, self._cur) if self.mix else self.k1(xx, W13, s, be)
        rec(2)
        y = self.k2(inter, W2, s, be, wf)
        rec(3)
        o = self.ks(y[: self.N])
        rec(4)
        return o


@tilelang.jit(out_idx=[1, 2])
def moe_align_gpu(N, E, bM, threads=1024):
    """Earlier one-kernel align with the fixed bound N + E * (bM - 1). The order of pairs inside a group is not fixed;
    every output row is computed independently, so the output does not depend on it. Unused blocks get expert -1."""
    Pmax = (N + E * (bM - 1) + bM - 1) // bM * bM
    NBmax = Pmax // bM

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((Pmax,), T.int32), Bexp: T.Tensor((NBmax,), T.int32)):
        with T.Kernel(1, threads=threads) as _:
            cnt = T.alloc_shared((E,), T.int32)
            cur = T.alloc_shared((E,), T.int32)
            start = T.alloc_shared((E + 1,), T.int32)
            tx = T.get_thread_binding()
            for r in T.serial(T.ceildiv(E, threads)):
                if r * threads + tx < E:
                    cnt[r * threads + tx] = 0
            for r in T.serial(T.ceildiv(Pmax, threads)):
                if r * threads + tx < Pmax:
                    Sid[r * threads + tx] = N
            for r in T.serial(T.ceildiv(NBmax, threads)):
                if r * threads + tx < NBmax:
                    Bexp[r * threads + tx] = -1
            T.sync_threads()
            for r in T.serial(T.ceildiv(N, threads)):
                if r * threads + tx < N:
                    T.atomic_add(cnt[Ids[r * threads + tx]], 1)
            T.sync_threads()
            if tx == 0:
                acc_ = T.alloc_var(T.int32)
                acc_ = 0
                for e in T.serial(E):
                    start[e] = acc_
                    cur[e] = acc_
                    acc_ += (cnt[e] + bM - 1) // bM * bM
                start[E] = acc_
            T.sync_threads()
            for r in T.serial(T.ceildiv(N, threads)):
                i = r * threads + tx
                if i < N:
                    slot = T.atomic_add(cur[Ids[i]], 1, return_prev=True)
                    Sid[slot] = i
            for r in T.serial(T.ceildiv(E, threads)):
                e = r * threads + tx
                if e < E:
                    for blk in T.serial(N // bM + 1):
                        if start[e] + blk * bM < start[e + 1]:
                            Bexp[start[e] // bM + blk] = e

    return main


@tilelang.jit(out_idx=[4])
def moe_gate_up(Tn, topk, H, I, E, P, bM, bN, bK, threads, policy="row", kp=1):
    dtype, acc = T.float16, T.float32
    pol = GemmWarpPolicy.FullRow if policy == "row" else GemmWarpPolicy.FullCol
    NB = P // bM

    @T.prim_func
    def main(X: T.Tensor((Tn, H), dtype), W13: T.Tensor((E, 2 * I, H), dtype), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Inter: T.Tensor((P, I), dtype)):
        with T.Kernel(NB, T.ceildiv(I, bN), threads=threads) as (bm, bn):
            xs = T.alloc_shared((bM, bK), dtype)
            gs = T.alloc_shared((bN, bK), dtype)
            us = T.alloc_shared((bN, bK), dtype)
            ag = T.alloc_fragment((bM, bN), acc)
            au = T.alloc_fragment((bM, bN), acc)
            if Bexp[bm] >= 0:                                    # blocks past the used rows exit (fixed-size grid)
                e = Bexp[bm]
                T.clear(ag)
                T.clear(au)
                for k in T.serial(T.ceildiv(H, bK)):
                    for i, kk in T.Parallel(bM, bK):
                        sid = Sid[bm * bM + i]
                        xs[i, kk] = T.if_then_else(sid < Tn * topk, X[T.min(sid, Tn * topk - 1) // topk, k * bK + kk], T.cast(0, dtype))
                    T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                    T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                    T.gemm(xs, gs, ag, transpose_B=True, k_pack=kp, policy=pol)
                    T.gemm(xs, us, au, transpose_B=True, k_pack=kp, policy=pol)
                for i, j in T.Parallel(bM, bN):
                    g = ag[i, j]
                    Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], dtype)

    return main


@tilelang.jit(out_idx=[5])
def moe_down(Tn, topk, H, I, E, P, bM, bN, bK, threads, policy="row", kp=1):
    dtype, acc = T.float16, T.float32
    pol = GemmWarpPolicy.FullRow if policy == "row" else GemmWarpPolicy.FullCol
    NB = P // bM

    @T.prim_func
    def main(Inter: T.Tensor((P, I), dtype), W2: T.Tensor((E, H, I), dtype), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Wt: T.Tensor((Tn * topk,), acc), Y: T.Tensor((Tn * topk, H), dtype)):
        with T.Kernel(NB, T.ceildiv(H, bN), threads=threads) as (bm, bn):
            a_s = T.alloc_shared((bM, bK), dtype)
            w_s = T.alloc_shared((bN, bK), dtype)
            ac = T.alloc_fragment((bM, bN), acc)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                T.clear(ac)
                for k in T.serial(T.ceildiv(I, bK)):
                    T.copy(Inter[bm * bM:bm * bM + bM, k * bK:k * bK + bK], a_s)
                    T.copy(W2[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], w_s)
                    T.gemm(a_s, w_s, ac, transpose_B=True, k_pack=kp, policy=pol)
                for i, j in T.Parallel(bM, bN):
                    sid = Sid[bm * bM + i]
                    if sid < Tn * topk:
                        Y[sid, bn * bN + j] = T.cast(ac[i, j] * Wt[sid], dtype)

    return main


def reference(X, W13, W2, ids, wts):
    """float32 per-expert loop (exact math, slow)."""
    Tn, topk = ids.shape
    I = W2.shape[2]
    out = torch.zeros(Tn, X.shape[1], dtype=torch.float32, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].float() @ W13[e].float().T
        a = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (a @ W2[e].float().T) * wts[tok, kk, None])
    return out


if __name__ == "__main__":
    from tilelang.profiler import do_bench
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2048); ap.add_argument("--I", type=int, default=512)
    ap.add_argument("--E", type=int, default=256); ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", default="1,16,64,256,1024,4096")
    a = ap.parse_args()
    H, I, E, topk = a.H, a.I, a.E, a.topk
    torch.manual_seed(0)
    W13 = (torch.randn(E, 2 * I, H, device="cuda") * (1 / math.sqrt(H))).half()
    W2 = (torch.randn(E, H, I, device="cuda") * (1 / math.sqrt(I))).half()
    for Tn in [int(x) for x in a.tokens.split(",")]:
        X = torch.randn(Tn, H, device="cuda").half()
        wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, device="cuda"), -1), topk, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float()
        ids = ids.int()
        R = reference(X, W13, W2, ids, wts)
        m = FusedMoE(Tn, topk, H, I, E)
        O = m(X, W13, W2, ids, wts)
        print(json.dumps({"T": Tn, "cfg": m.cfg, "x": m.x, "unroll": m.unroll, "gu_flags": m.gu_flags, "rel_vs_fp32": round(((O.float() - R).norm() / R.norm()).item(), 7),
                          "ms": round(do_bench(lambda: m(X, W13, W2, ids, wts), warmup=10, rep=50), 4)}), flush=True)
