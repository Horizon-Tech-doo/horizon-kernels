"""Fused mixture-of-experts layer (grouped GEMM over expert-sorted tokens) for gfx90a (MI250X), TileLang.

Default shape: hidden 2048, expert intermediate 512, 256 experts, top-8 routing, read from the Qwen/Qwen3.6-35B-A3B config.
fp16 weights and activations, fp32 accumulate.

FusedMoE is the implementation to use. The layout came out of the E2 and N1 experiments (evidence/E2, evidence/N1):
  align : two kernels. Count, prefix sum and Bexp in LDS, then claim one slot per pair with a global atomic.
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
ROW = GemmWarpPolicy.FullRow
POL = {"row": GemmWarpPolicy.FullRow, "square": GemmWarpPolicy.Square, "col": GemmWarpPolicy.FullCol}
NO_SAFE = {PK.TL_DISABLE_SAFE_MEMORY_ACCESS: True}
_SMALL = dict(bM=16, bN=32, bK=128, th=64, dN=64, dK=128, dth=64, kp=2, al_th=1024, sum_th=256, gu_unroll=20000)
_MID = dict(bM=64, bN=64, bK=128, th=512, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=None)
_LARGE = dict(bM=64, bN=64, bK=64, th=256, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=20000)
_ILP = ("-mllvm", "-misched=gcn-max-ilp")
_SMALL_C22 = dict(_SMALL, gu_unroll=None, gu_flags=_ILP)
_LARGE_C22 = dict(_LARGE, gu_unroll=None, gu_flags=_ILP)
_SMALL_MAX_C22 = 512


def tiles_for(Tn, clang):
    """Measured at the T values in evidence/N1. Clang >= 20 uses the small tile only with its scheduler flag: without it the
    tile is 2x slower in k1. Clang unknown (0) takes the unflagged mid and large rule, which holds on both compilers."""
    if clang == 0:
        return dict(_MID if Tn <= 1024 else _LARGE)
    if clang >= 20:
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
    """Y is [N + 1, H]; row N is a dump row for padding, so the write loop has no branch per element."""
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
         gate/up on any compiler, dropped if cfg changes the gate/up tile without naming them; dn_unroll is off by default (no gain at T = 4096 once down uses 64x64x128, and slower at
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
        self.Tn, self.topk, self.H, self.I, self.E, self.cfg, self.x, self.x_copy = Tn, topk, H, I, E, c, x, x_copy
        self.N = N = Tn * topk
        if H % c["bK"] or I % c["bN"] or I % c["dK"] or H % c["dN"] or H % c["sum_th"]:
            raise ValueError(f"tiles must divide the matrix (safe-memory is off): {c}")
        self.P = P = pmax_rows(N, E, c["bM"])
        self.NB = P // c["bM"]
        self.unroll = c["gu_unroll"] if c["gu_unroll"] and 0 < clang < 20 else None
        self.gu_flags = tuple(c.get("gu_flags") or ())
        flags = lambda u: ("-mllvm", f"-unroll-threshold={u}") if u else None
        self.ka = _build(align_count(N, E, c["bM"], c["al_th"]))
        self.kb = _build(align_slots(N, E, c["bM"], min(256, c["al_th"])))
        self.k1 = _build(gate_up(Tn, topk, H, I, E, P, c["bM"], c["bN"], c["bK"], c["th"], c["kp"], x, c.get("gu_pol", "row")), [4], (flags(self.unroll) or ()) + self.gu_flags or None)
        self.k2 = _build(down(Tn, topk, H, I, E, P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"], c.get("dn_pol", "row")), [5], flags(c.get("dn_unroll")))
        self.ks = _build(topk_sum(Tn, topk, H, c["sum_th"], DTYPE if out == "fp16" else ACC), [1])
        self.dev = device
        self._zr = torch.zeros(1, H, dtype=torch.float16, device=device)
        self._Xz = None
        if x == "zero" and x_copy == "buffer":
            self._Xz = torch.zeros(Tn + 1, H, dtype=torch.float16, device=device)

    def align(self, idf):
        s = torch.empty((self.P,), dtype=torch.int32, device=self.dev)
        b = torch.empty((self.NB,), dtype=torch.int32, device=self.dev)
        c = torch.empty((self.E,), dtype=torch.int32, device=self.dev)
        self.ka(idf, s, b, c)
        self.kb(idf, s, c)
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
        xx = self._x(X)
        s, be = self.align(idf)
        rec(1)
        inter = self.k1(xx, W13, s, be)
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
