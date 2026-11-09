"""Isolation arms for the grid's k1 gain over the landed per-T kernels. Experiment only, landed moe_fused.py is imported unedited.
Differences between landed k1 at T and grid k1 at G = next power of two: (a) padding rows read a zero row instead of a real one,
(b) X is copied per call into a kept buffer just before k1, (c) launch, Sid, Bexp and Inter are sized for G.
  ship      landed.
  ship_z0   landed kernel, row T-1 of X set to zero: (a) value only, same code. Its err is wrong on that one token by construction.
  x_once    landed kernel, X in a kept buffer filled once: placement of X.
  x_copy    landed kernel, X copied per call into a kept buffer before align: (b).
  xz_row    x='zero' kernel, [T+1] X prefilled, row T zero: (a) with the zero kernel's code.
  xz_dirty  same kernel, row T holds real data: value against address.
  xz_buf    x='zero' kernel with its per-call buffer copy.
  pad2      landed form, launch, Sid, Bexp and Inter sized for G, padding reads real row T-1, no copy: (c).
  gridt     control, the T-row-sum grid with per-call copy, as n1grid.GridMoE.
  g_once    grid kernels, X and ids and weights filled once: grid without (b).
  g_dirty   grid kernels, per-call copy, row G-1 holds real data: grid without (a).
"""
import torch
import tilelang.rocm.language as T
import n1grid

MEMO = {}
H, I, E, topk = 2048, 512, 256, 8


def align_count_p(N, E_, bM, P, threads):
    NB = P // bM

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32), Cur: T.Tensor((E_,), T.int32)):
        with T.Kernel(2, threads=threads) as b:
            cnt = T.alloc_shared((E_,), T.int32)
            start = T.alloc_shared((E_ + 1,), T.int32)
            tx = T.get_thread_binding()
            if b == 1:
                for r in T.serial(T.ceildiv(P, threads)):
                    if r * threads + tx < P:
                        Sid[r * threads + tx] = N
            else:
                for r in T.serial(T.ceildiv(E_, threads)):
                    if r * threads + tx < E_:
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
                    for e in T.serial(E_):
                        start[e] = acc_
                        Cur[e] = acc_
                        acc_ += (cnt[e] + bM - 1) // bM * bM
                    start[E_] = acc_
                T.sync_threads()
                for r in T.serial(T.ceildiv(E_, threads)):
                    e = r * threads + tx
                    if e < E_:
                        nb = T.alloc_var(T.int32)
                        nb = (start[e + 1] - start[e]) // bM
                        for blk in T.serial(nb):
                            Bexp[start[e] // bM + blk] = e

    return main


def align_slots_p(N, E_, P, threads):
    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Cur: T.Tensor((E_,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as b:
            tx = T.get_thread_binding()
            i = b * threads + tx
            if i < N:
                slot = T.atomic_add(Cur[Ids[i]], 1, return_prev=True)
                Sid[slot] = i

    return main


def _mem(key, f):
    if key not in MEMO:
        MEMO[key] = f()
    return MEMO[key]


def _pad2(M, Tn, G):
    m = M.FusedMoE(Tn)
    c = m.cfg
    P = M.pmax_rows(G * topk, E, c["bM"])
    N = Tn * topk
    m.P, m.NB = P, P // c["bM"]
    m.ka = M._build(align_count_p(N, E, c["bM"], P, c["al_th"]))
    m.kb = M._build(align_slots_p(N, E, P, min(256, c["al_th"])))
    m.k1 = M._build(M.gate_up(Tn, topk, H, I, E, P, c["bM"], c["bN"], c["bK"], c["th"], c["kp"], "last", c.get("gu_pol", "row")), [4], tuple(c.get("gu_flags") or ()) or None)
    m.k2 = M._build(M.down(Tn, topk, H, I, E, P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"]), [5])
    return m


def _grid(g, X, W13, W2, ids, wts, copy=True, dirty=False):
    Tn, G, m = X.shape[0], g.G, g.m
    Xb = torch.zeros(G, H, dtype=torch.float16, device=X.device)
    Ib = torch.full((G, topk), E, dtype=torch.int32, device=X.device)
    Wb = torch.zeros(G, topk, dtype=torch.float32, device=X.device)
    if dirty:
        Xb[G - 1] = X[Tn - 1]
    if not copy:
        Xb[:Tn] = X
        Ib[:Tn] = ids
        Wb[:Tn] = wts

    def fn(rec):
        rec = rec or (lambda i: None)
        rec(0)
        if copy:
            Xb[:Tn].copy_(X); Ib[:Tn].copy_(ids); Wb[:Tn].copy_(wts)
            Ib[Tn:] = E
        idf, wf = Ib.flatten(), Wb.flatten()
        s = torch.empty((m.P,), dtype=torch.int32, device=m.dev)
        b = torch.empty((m.NB,), dtype=torch.int32, device=m.dev)
        c = torch.empty((E,), dtype=torch.int32, device=m.dev)
        g.ka(idf, s, b, c)
        g.kb(idf, s, c)
        rec(1)
        inter = m.k1(Xb, W13, s, b)
        rec(2)
        y = m.k2(inter, W2, s, b, wf)
        rec(3)
        o = g.ks(y[: m.N], Tn)
        rec(4)
        return o[:Tn]

    return fn


def arms(spec, M, Tn, X, W13, W2, ids, wts, emit):
    G = n1grid.next_pow2(Tn)
    out = {}
    for arm in spec.split(","):
        if arm in ("ship", "ship_z0", "x_once", "x_copy"):
            m = _mem((Tn, "ship"), lambda: M.FusedMoE(Tn))
            if arm == "ship":
                emit(T=Tn, kind="ship_cfg", cfg=m.cfg, gu_flags=m.gu_flags, x="last", G=G)
                fn = lambda rec, m=m: m(X, W13, W2, ids, wts, rec)
            elif arm == "ship_z0":
                X0 = X.clone(); X0[Tn - 1] = 0
                fn = lambda rec, m=m, X0=X0: m(X0, W13, W2, ids, wts, rec)
            elif arm == "x_once":
                Xk = X.clone()
                fn = lambda rec, m=m, Xk=Xk: m(Xk, W13, W2, ids, wts, rec)
            else:
                Xk = torch.zeros_like(X)
                def fn(rec, m=m, Xk=Xk):
                    Xk.copy_(X)
                    return m(Xk, W13, W2, ids, wts, rec)
        elif arm in ("xz_row", "xz_dirty", "xz_buf"):
            m = _mem((Tn, "xz"), lambda: M.FusedMoE(Tn, x="zero"))
            if arm == "xz_buf":
                fn = lambda rec, m=m: m(X, W13, W2, ids, wts, rec)
            else:
                Xz = torch.cat([X, torch.zeros(1, H, device=X.device, dtype=X.dtype) if arm == "xz_row" else X[:1]])
                fn = lambda rec, m=m, Xz=Xz: m(Xz, W13, W2, ids, wts, rec)
        elif arm == "pad2":
            m = _mem((Tn, "pad2"), lambda: _pad2(M, Tn, G))
            emit(T=Tn, kind="pad2_cfg", G=G, P=m.P, NB=m.NB)
            fn = lambda rec, m=m: m(X, W13, W2, ids, wts, rec)
        elif arm in ("gridt", "g_once", "g_dirty"):
            g = _mem(("grid", G), lambda: n1grid.GridMoE(G, sum_mode="gridt"))
            if arm == "gridt":
                emit(T=Tn, kind="grid_cfg", G=G, P=g.m.P, NB=g.m.NB, cfg=g.m.cfg)
            fn = _grid(g, X, W13, W2, ids, wts, copy=arm != "g_once", dirty=arm == "g_dirty")
        else:
            raise ValueError(arm)
        out[arm] = (fn, 4)
    return out
