# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Reference (torch, any device) of G3v: the G3 replay ring on vLLM 0.30's state layout (30-09-2026).

Mirrors tilelang.contrib.horizon_ops.gdn_g3v bookkeeping exactly (ring positions in the conv regions of window slots
1..SW-1, meta per (request, head, chunk) with nbuf | npend << 8 | half << 16, entry flush when n_acc > 1, window mode
when a ring slot is invalid), plus a small simulator of vLLM 0.30's align-mode machinery (V2 runner):
  preprocess_state : window column c = (num_computed + T - 1) // B; crossing -> num_accepted = 1 and the pre-copy
                     state[bt[c_old + n_acc - 1]] -> state[bt[c_new]]
  forward          : pure speculative batch -> the fused decode (vLLM window kernel, or G3v); mixed batch -> FLA
                     (same arithmetic as the window kernel)
  postprocess_state: num_computed += n_acc; align post-copy (postprocess_mamba_fused_kernel's decision logic)
The plugin's hooks (expand before a crossing, before a non-pure batch, before a post-copy; meta reset at
add_request) are modelled by World(g3=True). Arithmetic is gdn_state_ref's (identical in both protocols).
"""
import torch

import gdn_state_ref as R

DK = DV = 128
CH = 32


def rec_floats(HV, H):
    return HV * DV + H * DK + HV


def ring_geometry(HV, H, RF, SW):
    rec = rec_floats(HV, H)
    pps = RF // rec
    return pps, (pps * (SW - 1)) // 2


class Store:
    """KV pool of one GDN layer: ssm state per block + the block's conv region (ring storage for G3v)"""

    def __init__(self, NB, HV, RF):
        self.S = torch.zeros(NB, HV, DV, DK)
        self.ring = torch.full((NB, RF), float("nan"))     # never-written ring floats are NaN: misuse shows


def _pos(sidx_row, p, PPS, REC):
    return int(sidx_row[1 + p // PPS]), (p % PPS) * REC


def g3v_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, ridx, st: Store, meta, z, norm_w, out, HALF, PPS,
             scale=DK ** -0.5, eps=1e-6, act="silu", stats=None):
    HV = st.S.shape[1]
    H = (qkv.shape[1] - HV * DV) // (2 * DK)
    G = HV // H
    REC = rec_floats(HV, H)
    OK_, OG_ = HV * DV, HV * DV + H * DK
    SW = sidx.shape[1]
    q, k, v, g, beta = R._prep(qkv, a, b, A_log, dt_bias, HV, H, scale)
    for r in range(sidx.shape[0]):
        bos, eos = int(cu[r]), int(cu[r + 1])
        nt = eos - bos
        if nt <= 0:
            continue
        acc, rq, s0 = int(nacc[r]), int(ridx[r]), int(sidx[r, 0])
        vsrc = int(sidx[r, acc - 1]) if 0 < acc <= SW else 0
        if vsrc <= 0 or nt > R.T_MAX:
            out[bos:eos] = 0
            continue
        ringok = rq >= 0 and nt <= HALF and all(int(sidx[r, j]) > 0 for j in range(1, SW))
        m = meta[rq] if ringok else torch.zeros(HV, DV // CH, dtype=torch.int32)
        assert bool((m == m[0, 0]).all())
        m = int(m[0, 0])
        nbuf, npend, half = m % 256, (m // 256) % 256, (m // 65536) % 2
        ncom, fl, rbase, wbase, src = 0, False, 0, 0, vsrc
        if ringok:
            if npend == 0:
                half = 0
                if vsrc != s0:
                    fl, wbase = True, HALF
            else:
                src = s0
                ncom = nbuf + max(min(acc, npend), 0)
                rbase = half * HALF
                wbase = rbase + ncom
                if ncom + nt > HALF:
                    fl, wbase = True, (1 - half) * HALF
            rd, wr = set(range(rbase, rbase + ncom)), set(range(wbase, wbase + nt))
            assert not (rd & wr) and max(rd | wr) < 2 * HALF <= PPS * (SW - 1)
        h = st.S[src].clone()
        for j in range(ncom):
            blk, off = _pos(sidx[r], rbase + j, PPS, REC)
            rec = st.ring[blk, off: off + REC]
            kk = rec[OK_: OK_ + H * DK].view(H, DK).repeat_interleave(G, 0)
            h = h * rec[OG_: OG_ + HV][:, None, None]
            h = h + kk[:, None, :] * rec[:HV * DV].view(HV, DV)[:, :, None]
        if fl:
            st.S[s0] = h
            if stats is not None:
                stats["flushes"] += 1
        o_all = []
        for t in range(nt):
            kk = k[bos + t].repeat_interleave(G, 0)
            qq = q[bos + t].repeat_interleave(G, 0)
            h, delta, o = R._token(h, kk, qq, v[bos + t], g[bos + t], beta[bos + t])
            o_all.append(o.to(torch.bfloat16).float())
            if ringok:
                blk, off = _pos(sidx[r], wbase + t, PPS, REC)
                st.ring[blk, off: off + HV * DV] = delta.reshape(-1)
                st.ring[blk, off + OK_: off + OK_ + H * DK] = k[bos + t].reshape(-1)
                st.ring[blk, off + OG_: off + OG_ + HV] = g[bos + t]
            else:
                dst = int(sidx[r, t])
                if dst > 0:
                    st.S[dst] = h
        for t in range(nt):
            out[bos + t] = R._norm(o_all[t], z[bos + t], norm_w, eps, act)
        if ringok:
            meta[rq] = (nt * 256 + (1 - half) * 65536) if fl else (ncom + nt * 256 + half * 65536)
        if stats is not None:
            stats["steps"] += 1
            stats["replayed"] += ncom
            stats["window_mode"] += 0 if ringok else 1


def g3v_expand(sidx, ridx, mask, st: Store, meta, HALF, PPS, H):
    HV = st.S.shape[1]
    G = HV // H
    REC = rec_floats(HV, H)
    OK_, OG_ = HV * DV, HV * DV + H * DK
    n = 0
    for r in range(sidx.shape[0]):
        rq, s0 = int(ridx[r]), int(sidx[r, 0])
        if not int(mask[r]) or rq < 0 or s0 <= 0:
            continue
        m = int(meta[rq][0, 0])
        nbuf, npend, half = m % 256, (m // 256) % 256, (m // 65536) % 2
        if npend == 0:
            continue
        h = st.S[s0].clone()
        for j in range(nbuf + npend):
            blk, off = _pos(sidx[r], half * HALF + j, PPS, REC)
            rec = st.ring[blk, off: off + REC]
            kk = rec[OK_: OK_ + H * DK].view(H, DK).repeat_interleave(G, 0)
            h = h * rec[OG_: OG_ + HV][:, None, None]
            h = h + kk[:, None, :] * rec[:HV * DV].view(HV, DV)[:, :, None]
            if j >= nbuf:
                dst = int(sidx[r, j - nbuf])
                if dst > 0:
                    st.S[dst] = h
        meta[rq] = 0
        n += 1
    return n


class World:
    """one GDN layer, nr requests, vLLM 0.30 align-mode bookkeeping; g3=True adds the G3v kernel + plugin hooks"""

    def __init__(self, nr, HV, H, B, RF, g3, seed, SW=4, nblocks_per_req=64, init=None, dt=None):
        self.nr, self.HV, self.H, self.B, self.SW, self.g3 = nr, HV, H, B, SW, g3
        self.PPS, self.HALF = ring_geometry(HV, H, RF, SW)
        self.st = Store(1 + nr * nblocks_per_req, HV, RF)
        # block tables: unique block ids per request (block 0 = vLLM's null block)
        self.bt = [list(range(1 + i * nblocks_per_req, 1 + (i + 1) * nblocks_per_req)) for i in range(nr)]
        self.req_idx = list(range(nr))               # vLLM request-state index (stable per request)
        self.meta = torch.zeros(nr + 2, HV, DV // CH, dtype=torch.int32)
        self.computed = [5 + 3 * i for i in range(nr)]
        self.col = [(c - 1) // B for c in self.computed]
        self.acc = [1] * nr
        for i in range(nr):
            self.st.S[self.bt[i][self.col[i]]] = init[i]
        self.stats = {"flushes": 0, "steps": 0, "replayed": 0, "window_mode": 0, "expanded": 0}
        self.observed = []                          # (step, what, request, block): states vLLM read through copies

    def window(self, i, col):
        return [self.bt[i][col + k] if col + k < len(self.bt[i]) else 0 for k in range(self.SW)]

    def _expand(self, rows, cols):
        if not self.g3 or not rows:
            return
        sidx = torch.tensor([self.window(i, cols[i]) for i in rows], dtype=torch.int32)
        ridx = torch.tensor([self.req_idx[i] for i in rows], dtype=torch.int32)
        self.stats["expanded"] += g3v_expand(sidx, ridx, torch.ones(len(rows), dtype=torch.int32), self.st,
                                             self.meta, self.HALF, self.PPS, self.H)

    def preprocess(self, T, mixed):
        new = [(self.computed[i] + T[i] - 1) // self.B for i in range(self.nr)]
        cross = [i for i in range(self.nr) if new[i] != self.col[i]]
        # plugin H1: rows crossing a block boundary, and every row when the batch is not pure speculative decode
        self._expand(list(range(self.nr)) if mixed else cross, self.col)
        for i in cross:
            src = self.bt[i][self.col[i] + self.acc[i] - 1]
            self.st.S[self.bt[i][new[i]]] = self.st.S[src]
            self.observed.append(("precopy", i, src, self.st.S[src].clone()))
            self.acc[i] = 1
        self.col = new

    def forward(self, T, inputs, A_log, dt_bias, norm_w, mixed):
        qkv, a, b, z, cu = inputs
        sidx = torch.tensor([self.window(i, self.col[i]) for i in range(self.nr)], dtype=torch.int32)
        nacc = torch.tensor(self.acc, dtype=torch.int32)
        out = torch.full((qkv.shape[0], self.HV, DV), 7.0, dtype=torch.bfloat16)
        if self.g3 and not mixed:
            ridx = torch.tensor(self.req_idx, dtype=torch.int32)
            g3v_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, ridx, self.st, self.meta, z, norm_w, out, self.HALF,
                     self.PPS, stats=self.stats)
        else:
            R.window_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, self.st.S, z, norm_w, out)
        return out

    def postprocess(self, acc_new):
        rows, cp = [], []
        for i in range(self.nr):
            self.acc[i] = acc_new[i]
            self.computed[i] += acc_new[i]
            new_c = self.computed[i]
            running = new_c - acc_new[i] + 1
            aligned = (new_c // self.B) * self.B
            if aligned >= running:
                bias, dest = aligned - running, aligned // self.B - 1
                rows.append(i)
                cp.append((i, bias, dest))
        self._expand(rows, self.col)                  # plugin H2
        for i, bias, dest in cp:
            src_col = self.col[i]
            if src_col == dest:
                self.acc[i] = 1
            if src_col == dest and bias == 0:
                continue
            src = self.bt[i][src_col + bias]
            self.st.S[self.bt[i][dest]] = self.st.S[src]
            self.observed.append(("postcopy", i, src, self.st.S[src].clone()))

    def committed(self, i):
        return self.st.S[self.bt[i][self.col[i] + self.acc[i] - 1]]
