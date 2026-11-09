"""E2 kernels: working copy of moe_fused.py's gate/up and down with one switch per change, plus a top-k sum kernel."""
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.transform import PassConfigKey as PK
dtype, acc = T.float16, T.float32
ROW = GemmWarpPolicy.FullRow


def pc(ns):
    return {PK.TL_DISABLE_SAFE_MEMORY_ACCESS: True} if ns else None


def gate_up(Tn, topk, H, I, E, P, bM, bN, bK, th, kp, x="if", mask=False, swap=False):
    """x='if': moe_fused.moe_gate_up X path (T.if_then_else per element). x='v2': rows in LDS, zero row Tn for padding.
    mask=True: padding rows are not written to Inter (v2 only)."""
    NB = P // bM
    N = Tn * topk
    Xrows = Tn + 1 if x == "v2" else Tn
    v = x in ("v2", "v2c")
    grid = (T.ceildiv(I, bN), NB) if swap else (NB, T.ceildiv(I, bN))

    @T.prim_func
    def main(X: T.Tensor((Xrows, H), dtype), W13: T.Tensor((E, 2 * I, H), dtype), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Inter: T.Tensor((P, I), dtype)):
        with T.Kernel(*grid, threads=th) as (b0, b1):
            bm, bn = (b1, b0) if swap else (b0, b1)
            if v:
                rows = T.alloc_shared((bM,), T.int32)
            xs = T.alloc_shared((bM, bK), dtype)
            gs = T.alloc_shared((bN, bK), dtype)
            us = T.alloc_shared((bN, bK), dtype)
            ag = T.alloc_fragment((bM, bN), acc)
            au = T.alloc_fragment((bM, bN), acc)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                if v:
                    for i in T.Parallel(bM):
                        rows[i] = T.min(Sid[bm * bM + i], N if x == "v2" else N - 1) // topk
                T.clear(ag)
                T.clear(au)
                for k in T.serial(T.ceildiv(H, bK)):
                    for i, kk in T.Parallel(bM, bK):
                        if v:
                            xs[i, kk] = X[rows[i], k * bK + kk]
                        else:
                            sid = Sid[bm * bM + i]
                            xs[i, kk] = T.if_then_else(sid < N, X[T.min(sid, N - 1) // topk, k * bK + kk], T.cast(0, dtype))
                    T.copy(W13[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], gs)
                    T.copy(W13[e, I + bn * bN:I + bn * bN + bN, k * bK:k * bK + bK], us)
                    T.gemm(xs, gs, ag, transpose_B=True, k_pack=kp, policy=ROW)
                    T.gemm(xs, us, au, transpose_B=True, k_pack=kp, policy=ROW)
                for i, j in T.Parallel(bM, bN):
                    g = ag[i, j]
                    if mask:
                        if rows[i] < Tn:
                            Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], dtype)
                    else:
                        Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * au[i, j], dtype)

    return main


def down(Tn, topk, H, I, E, P, bM, bN, bK, th, kp, hoist=False, mask=False, swap=False, dump=False):
    """hoist: Sid and Wt read once per thread into 1-D fragments. mask: padding rows of Inter are not loaded."""
    NB = P // bM
    N = Tn * topk
    grid = (T.ceildiv(H, bN), NB) if swap else (NB, T.ceildiv(H, bN))

    @T.prim_func
    def main(Inter: T.Tensor((P, I), dtype), W2: T.Tensor((E, H, I), dtype), Sid: T.Tensor((P,), T.int32),
             Bexp: T.Tensor((NB,), T.int32), Wt: T.Tensor((N,), acc), Y: T.Tensor((N + 1 if dump else N, H), dtype)):
        with T.Kernel(*grid, threads=th) as (b0, b1):
            bm, bn = (b1, b0) if swap else (b0, b1)
            a_s = T.alloc_shared((bM, bK), dtype)
            w_s = T.alloc_shared((bN, bK), dtype)
            ac = T.alloc_fragment((bM, bN), acc)
            if mask:
                rows = T.alloc_shared((bM,), T.int32)
            if hoist:
                sf = T.alloc_fragment((bM,), T.int32)
                wf = T.alloc_fragment((bM,), acc)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                T.clear(ac)
                if mask:
                    for i in T.Parallel(bM):
                        rows[i] = Sid[bm * bM + i]
                for k in T.serial(T.ceildiv(I, bK)):
                    if mask:
                        for i, kk in T.Parallel(bM, bK):
                            a_s[i, kk] = T.if_then_else(rows[i] < N, Inter[bm * bM + i, k * bK + kk], T.cast(0, dtype))
                    else:
                        T.copy(Inter[bm * bM:bm * bM + bM, k * bK:k * bK + bK], a_s)
                    T.copy(W2[e, bn * bN:bn * bN + bN, k * bK:k * bK + bK], w_s)
                    T.gemm(a_s, w_s, ac, transpose_B=True, k_pack=kp, policy=ROW)
                if hoist:
                    for i in T.Parallel(bM):
                        s0 = Sid[bm * bM + i]
                        sf[i] = T.min(s0, N) if dump else s0
                        wf[i] = T.if_then_else(s0 < N, Wt[T.min(s0, N - 1)], T.cast(0, acc))
                    for i, j in T.Parallel(bM, bN):
                        if dump:
                            Y[sf[i], bn * bN + j] = T.cast(ac[i, j] * wf[i], dtype)
                        else:
                            if sf[i] < N:
                                Y[sf[i], bn * bN + j] = T.cast(ac[i, j] * wf[i], dtype)
                else:
                    for i, j in T.Parallel(bM, bN):
                        sid = Sid[bm * bM + i]
                        if sid < N:
                            Y[sid, bn * bN + j] = T.cast(ac[i, j] * Wt[sid], dtype)

    return main


def topk_sum(Tn, topk, H, th=256, otype=None, vec=True):
    """Coalesced reduction: each token is one block; each thread owns H/th columns. No atomics: one writer per output."""
    otype = otype or acc
    per = H // th

    @T.prim_func
    def main(Y: T.Tensor((Tn * topk, H), dtype), O: T.Tensor((Tn, H), otype)):
        with T.Kernel(Tn, threads=th) as t:
            tx = T.get_thread_binding()
            s = T.alloc_local((per,), acc)
            for j in T.vectorized(per):
                s[j] = T.cast(Y[t * topk, tx * per + j], acc)
            for k in T.serial(1, topk):
                for j in T.vectorized(per):
                    s[j] = s[j] + T.cast(Y[t * topk + k, tx * per + j], acc)
            for j in T.vectorized(per):
                O[t, tx * per + j] = T.cast(s[j], otype)

    return main


def align_tight(N, E, bM, threads=1024):
    """moe_align_gpu with the Bexp fill bounded by the expert's own block count (was N // bM + 1 tests per thread)."""
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
                    nb = T.alloc_var(T.int32)
                    nb = (start[e + 1] - start[e]) // bM
                    for blk in T.serial(nb):
                        Bexp[start[e] // bM + blk] = e

    return main


def align3_a(N, E, bM, threads=1024):
    """Two blocks like vLLM's moe_align: block 0 counts experts, writes group starts to Cur and fills Bexp; block 1 fills Sid."""
    Pmax = (N + E * (bM - 1) + bM - 1) // bM * bM
    NBmax = Pmax // bM

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((Pmax,), T.int32), Bexp: T.Tensor((NBmax,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(2, threads=threads) as b:
            cnt = T.alloc_shared((E,), T.int32)
            start = T.alloc_shared((E + 1,), T.int32)
            tx = T.get_thread_binding()
            if b == 1:
                for r in T.serial(T.ceildiv(Pmax, threads)):
                    if r * threads + tx < Pmax:
                        Sid[r * threads + tx] = N
            else:
                for r in T.serial(T.ceildiv(E, threads)):
                    if r * threads + tx < E:
                        cnt[r * threads + tx] = 0
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


def align3_b(N, E, bM, threads=256):
    """Claim a slot per pair with a global atomic across many blocks."""
    Pmax = (N + E * (bM - 1) + bM - 1) // bM * bM

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((Pmax,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as b:
            tx = T.get_thread_binding()
            i = b * threads + tx
            if i < N:
                slot = T.atomic_add(Cur[Ids[i]], 1, return_prev=True)
                Sid[slot] = i

    return main


def build(kind, tgt=None, ns=False, out=True, flags=None, **kw):
    """kind in gu/dn/sum. Returns a JIT kernel."""
    if kind == "gu":
        f, oi = gate_up(**kw), [4]
    elif kind == "dn":
        f, oi = down(**kw), [5]
    elif kind == "al3a":
        f, oi = align3_a(**kw), None
    elif kind == "al3b":
        f, oi = align3_b(**kw), None
    elif kind == "al":
        f, oi = align_tight(**kw), [1, 2]
    else:
        f, oi = topk_sum(**kw), [1]
    return tilelang.compile(f, out_idx=oi if out else None, target=tgt, pass_configs=pc(ns), compile_flags=flags)
