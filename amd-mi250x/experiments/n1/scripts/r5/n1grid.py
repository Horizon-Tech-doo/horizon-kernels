"""T grid prototype, not landed. One FusedMoE kernel set per grid size G >= T serves any T: X, ids and weights are copied into kept
[G, ...] buffers, padded tokens carry expert id E and get no slots, so their rows stay unwritten and their blocks exit early. Align is the
landed align plus one branch on the expert id. The sum still runs over G rows; the output is the first T rows."""
import torch
import tilelang
import tilelang.rocm.language as T
import moe_fused as M


def align_count(N, E, bM, threads=1024):
    P = M.pmax_rows(N, E, bM)
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
                        if Ids[r * threads + tx] < E:
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
    P = M.pmax_rows(N, E, bM)

    @T.prim_func
    def main(Ids: T.Tensor((N,), T.int32), Sid: T.Tensor((P,), T.int32), Cur: T.Tensor((E,), T.int32)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as b:
            tx = T.get_thread_binding()
            i = b * threads + tx
            if i < N:
                if Ids[i] < E:
                    slot = T.atomic_add(Cur[Ids[i]], 1, return_prev=True)
                    Sid[slot] = i

    return main


def next_pow2(Tn):
    g = 1
    while g < Tn:
        g *= 2
    return g


class GridMoE:
    def __init__(self, G, topk=8, H=2048, I=512, E=256, device="cuda"):
        self.m = M.FusedMoE(G, topk, H, I, E, device=device)
        c = self.m.cfg
        self.G, self.topk, self.H, self.E = G, topk, H, E
        self.ka = M._build(align_count(self.m.N, E, c["bM"], c["al_th"]))
        self.kb = M._build(align_slots(self.m.N, E, c["bM"], min(256, c["al_th"])))
        self.Xb = torch.zeros(G, H, dtype=torch.float16, device=device)
        self.Ib = torch.full((G, topk), E, dtype=torch.int32, device=device)
        self.Wb = torch.zeros(G, topk, dtype=torch.float32, device=device)

    def __call__(self, X, W13, W2, ids, wts, rec=None):
        rec = rec or (lambda i: None)
        m, Tn = self.m, X.shape[0]
        rec(0)
        self.Xb[:Tn].copy_(X); self.Ib[:Tn].copy_(ids); self.Wb[:Tn].copy_(wts)
        self.Ib[Tn:] = self.E
        idf, wf = self.Ib.flatten(), self.Wb.flatten()
        s = torch.empty((m.P,), dtype=torch.int32, device=m.dev)
        b = torch.empty((m.NB,), dtype=torch.int32, device=m.dev)
        c = torch.empty((self.E,), dtype=torch.int32, device=m.dev)
        self.ka(idf, s, b, c)
        self.kb(idf, s, c)
        rec(1)
        inter = m.k1(self.Xb, W13, s, b)
        rec(2)
        y = m.k2(inter, W2, s, b, wf)
        rec(3)
        o = m.ks(y[: m.N])
        rec(4)
        return o[:Tn]
