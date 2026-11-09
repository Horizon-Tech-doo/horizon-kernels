# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Reference (plain torch, any device) for the GDN decode state kernels in tilelang.contrib.horizon_ops.gdn_state.

window_step:  vLLM 0.29 fused_gdn_decode_post_conv_mtp semantics (source slot state_idx[r, n_acc - 1], a state written
              after every window token), with the stored state rounded to the state tensor's dtype (fp32 / bf16 RNE /
              fp16 RNE saturated to +-65504).
replay_step:  the G3 protocol (one checkpoint + a two-half ring of exact per-token records, meta per 32-row chunk),
              written to mirror the kernel's bookkeeping step by step, with assertions on the ring invariants
              (records read and written by one launch never overlap, all meta entries of a head agree).
materialize:  fold the committed ring tokens into the checkpoint and empty the ring (same half).

Arithmetic: fp32, per element h = h * g; hk = sum(h k); delta = (v - hk) * beta; h = h + k * delta; o = sum(h q).
It is the same in window_step and replay_step (the replay re-applies h * g and h + k * delta with the SAVED delta),
so on any device the two protocols must agree bit for bit with a float32 checkpoint. It is not bit-identical to the
CUDA kernels (those use fast-math FMAs and approximate exp / rsqrt); the GPU test compares kernels to vLLM's kernel.
"""
import torch

DK = DV = 128
CH = 32
T_MAX = 8


def store_round(h: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp32 -> state dtype, exactly as the kernels store (RNE; fp16 saturated to the finite range first)."""
    if dtype == torch.float32:
        return h.clone()
    if dtype == torch.float16:
        return h.clamp(-65504.0, 65504.0).to(torch.float16)
    return h.to(dtype)


def bf16_rne_bits(x: torch.Tensor) -> torch.Tensor:
    """independent bit-level RNE fp32 -> bf16 (finite inputs), used to cross-check torch's conversion"""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    u = u & 0xFFFF
    u = torch.where(u >= 0x8000, u - 0x10000, u)
    return u.to(torch.int16).view(torch.bfloat16)


def _prep(qkv, a, b, A_log, dt_bias, HV, H, scale):
    """per token: q, k (normalised, per key head), v, decay g, beta (all fp32)"""
    L = qkv.shape[0]
    x = qkv.float()
    q = x[:, : H * DK].view(L, H, DK)
    k = x[:, H * DK: 2 * H * DK].view(L, H, DK)
    v = x[:, 2 * H * DK: 2 * H * DK + HV * DV].view(L, HV, DV)
    q = q * (torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    xa = a.float() + dt_bias.float()[None, :]
    sp = torch.where(xa > 20.0, xa, torch.log1p(torch.exp(torch.clamp(xa, max=20.0))))
    g = torch.exp(-torch.exp(A_log.float())[None, :] * sp)
    beta = torch.sigmoid(b.float())
    return q, k, v, g, beta


def _norm(o, z, norm_w, eps, act):
    rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
    zf = z.float()
    gate = torch.sigmoid(zf) if act == "sigmoid" else zf * torch.sigmoid(zf)
    return (o * rstd * norm_w.float()[None, :] * gate).to(torch.bfloat16)


def _token(h, kk, qq, vv, g, beta):
    """one token of the recurrence on h (HV, DV, DK) fp32; returns (h, delta (HV, DV), o (HV, DV))"""
    h = h * g[:, None, None]
    hk = (h * kk[:, None, :]).sum(-1)
    delta = (vv - hk) * beta[:, None]
    h = h + kk[:, None, :] * delta[:, :, None]
    o = (h * qq[:, None, :]).sum(-1)
    return h, delta, o


def window_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out, scale=DK ** -0.5, eps=1e-6,
                act="silu"):
    """vLLM semantics. state (NS, HV, DV, DK) of dtype fp32/bf16/fp16, modified in place; out (L, HV, DV) bf16."""
    HV = state.shape[1]
    H = (qkv.shape[1] - HV * DV) // (2 * DK)
    G = HV // H
    q, k, v, g, beta = _prep(qkv, a, b, A_log, dt_bias, HV, H, scale)
    W = sidx.shape[1]
    for r in range(sidx.shape[0]):
        bos, eos = int(cu[r]), int(cu[r + 1])
        nt = eos - bos
        if nt <= 0:
            continue
        acc = int(nacc[r])
        src = int(sidx[r, acc - 1]) if 0 < acc <= W else 0
        if src <= 0 or nt > T_MAX:
            out[bos:eos] = 0
            continue
        h = state[src].float()
        o_all = []
        for t in range(nt):
            kk = k[bos + t].repeat_interleave(G, 0)
            qq = q[bos + t].repeat_interleave(G, 0)
            h, _, o = _token(h, kk, qq, v[bos + t], g[bos + t], beta[bos + t])
            o_all.append(o.to(torch.bfloat16).float())
            dst = int(sidx[r, t])
            if dst > 0:
                state[dst] = store_round(h, state.dtype)
        for t in range(nt):
            out[bos + t] = _norm(o_all[t], z[bos + t], norm_w, eps, act)
    return out


class Ring:
    """G3 ring for NG ring indices: k (NG, 2 HALF, H, DK), d (NG, 2 HALF, HV, DV), g (NG, 2 HALF, HV), meta (NG, HV, 4)"""

    def __init__(self, NG, HV, H, half=16, device="cpu"):
        self.half = half
        self.k = torch.zeros(NG, 2 * half, H, DK, device=device)
        self.d = torch.zeros(NG, 2 * half, HV, DV, device=device)
        self.g = torch.zeros(NG, 2 * half, HV, device=device)
        self.meta = torch.zeros(NG, HV, DV // CH, dtype=torch.int32, device=device)
        self.stats = {"flushes": 0, "steps": 0, "replayed": 0, "window_records": 0}

    def __getitem__(self, i):          # tuple view (ring_k, ring_d, ring_g, meta), like gdn_state.ring_buffers()
        return (self.k, self.d, self.g, self.meta)[i]

    def __iter__(self):
        return iter((self.k, self.d, self.g, self.meta))

    def fields(self, rg):
        m = self.meta[rg]
        assert bool((m == m[0, 0]).all()), "meta entries of a ring disagree"
        m = int(m[0, 0])
        return m % 256, (m // 256) % 256, (m // 65536) % 2

    def set(self, rg, nbuf, npend, half):
        self.meta[rg] = nbuf + npend * 256 + half * 65536


def replay_step(qkv, a, b, A_log, dt_bias, cidx, ridx, cu, nacc, fflush, state, ring: Ring, z, norm_w, out,
                scale=DK ** -0.5, eps=1e-6, act="silu"):
    """G3 verify step, mirroring gdn_replay_kernel. Returns a list of per-request dicts (for the traffic model)."""
    HV = state.shape[1]
    H = (qkv.shape[1] - HV * DV) // (2 * DK)
    G = HV // H
    HALF = ring.half
    q, k, v, g, beta = _prep(qkv, a, b, A_log, dt_bias, HV, H, scale)
    info = []
    for r in range(cidx.shape[0]):
        bos, eos = int(cu[r]), int(cu[r + 1])
        nt = eos - bos
        if nt <= 0:
            continue
        src, rg = int(cidx[r]), int(ridx[r])
        if src <= 0 or rg < 0 or nt > T_MAX:
            out[bos:eos] = 0
            continue
        nbuf, npend, half = ring.fields(rg)
        acc = max(min(int(nacc[r]), npend), 0)
        ncom = nbuf + acc
        fl = bool(int(fflush[r]) != 0 or ncom + nt > HALF)
        rbase = half * HALF
        wbase = (1 - half) * HALF if fl else half * HALF + ncom
        read = set(range(rbase, rbase + ncom))
        write = set(range(wbase, wbase + nt))
        assert not (read & write), ("ring read/write overlap", r, read, write)
        assert max(read | write) < 2 * HALF
        h = state[src].float()
        for j in range(ncom):
            p = rbase + j
            kk = ring.k[rg, p].repeat_interleave(G, 0)
            h = h * ring.g[rg, p][:, None, None]
            h = h + kk[:, None, :] * ring.d[rg, p][:, :, None]
        if fl:
            state[src] = store_round(h, state.dtype)
            h = state[src].float()
            ring.stats["flushes"] += 1
        o_all = []
        for t in range(nt):
            kk = k[bos + t].repeat_interleave(G, 0)
            qq = q[bos + t].repeat_interleave(G, 0)
            h, delta, o = _token(h, kk, qq, v[bos + t], g[bos + t], beta[bos + t])
            o_all.append(o.to(torch.bfloat16).float())
            ring.d[rg, wbase + t] = delta
            ring.k[rg, wbase + t] = k[bos + t]
            ring.g[rg, wbase + t] = g[bos + t]
        for t in range(nt):
            out[bos + t] = _norm(o_all[t], z[bos + t], norm_w, eps, act)
        if fl:
            ring.set(rg, 0, nt, 1 - half)
        else:
            ring.set(rg, ncom, nt, half)
        ring.stats["steps"] += 1
        ring.stats["replayed"] += ncom
        ring.stats["window_records"] += nt
        info.append({"r": r, "ncom": ncom, "flush": fl, "nt": nt})
    return info


def materialize(cidx, ridx, nacc, state, ring: Ring):
    HV = state.shape[1]
    G = HV // ring.k.shape[2]
    HALF = ring.half
    for r in range(cidx.shape[0]):
        src, rg = int(cidx[r]), int(ridx[r])
        if src <= 0 or rg < 0:
            continue
        nbuf, npend, half = ring.fields(rg)
        ncom = nbuf + max(min(int(nacc[r]), npend), 0)
        if ncom > 0:
            h = state[src].float()
            for j in range(ncom):
                p = half * HALF + j
                kk = ring.k[rg, p].repeat_interleave(G, 0)
                h = h * ring.g[rg, p][:, None, None]
                h = h + kk[:, None, :] * ring.d[rg, p][:, :, None]
            state[src] = store_round(h, state.dtype)
        ring.set(rg, 0, 0, half)


def traffic_bytes(nr, T, HV=48, H=16, state_bytes=4, acc_mean=None, half=16):
    """paper model, bytes per layer and decode step for nr requests with windows of T tokens.
    vLLM / G1 / G1S: read 1 state + write T states. G3: read the checkpoint, the committed records (~acc_mean + the
    carried buffer, average HALF / 2), write T records, and a checkpoint write every (HALF - T) / acc_mean steps."""
    S = HV * DV * DK * state_bytes
    rec_tok = HV * DV * 4 + H * DK * 4 + HV * 4      # delta + k + g per token (fp32)
    window = nr * (S + T * S)
    if acc_mean is None:
        acc_mean = T
    ncom_avg = (half - T) / 2.0 + acc_mean / 2.0
    flush_rate = min(1.0, acc_mean / max(1.0, half - T))
    g3 = nr * (S + ncom_avg * rec_tok + T * rec_tok + flush_rate * S)
    return {"window": window, "g3": g3, "state": S, "record_per_token": rec_tok}
