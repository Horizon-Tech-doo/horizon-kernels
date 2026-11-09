"""Simulated FP8 (E4M3) and FP4 (E2M1, NVFP4 style) weight paths on gfx90a, which has no FP8 or FP4 hardware. Prototype, not landed; imports
the landed moe_fused. Weights are stored as packed bit patterns in int32 words and decoded to fp16 in LDS through fp16 lookup tables in the
kernel, then fed to the same fp16 MFMA. FP8: one fp32 scale per expert per matrix. FP4: one E4M3 scale per 16 elements along K, plus one fp32
scale per expert per matrix. Align and sum are the landed kernels.
Usage: n1fp.py --data D --cases T:route,... [--rounds N] [--rep N] [--stack S] [--build-only] [--compile-only T]"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilelang
import tilelang.rocm.language as T
from tilelang.tileop.base import GemmWarpPolicy
import moe_fused as M

DTYPE, ACC = M.DTYPE, M.ACC
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
LUT4 = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float16)
LUT8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float16)
LUT8 = torch.nan_to_num(LUT8, nan=0.0)


def gate_up_q(fmt, Tn, topk, H, I, E, P, bM, bN, bK, th, kp, pol="row"):
    NB, N = P // bM, Tn * topk
    per = 4 if fmt == "fp8" else 8
    Hw, Kw = H // per, bK // per

    @T.prim_func
    def main(X: T.Tensor((Tn, H), DTYPE), Wq: T.Tensor((E, 2 * I, Hw), T.int32), Ws: T.Tensor((E, 2 * I, H // 64), T.int32), L8: T.Tensor((256,), DTYPE), L4: T.Tensor((16,), DTYPE),
             Sc: T.Tensor((E, 2), ACC), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32), Inter: T.Tensor((P, I), DTYPE)):
        with T.Kernel(T.ceildiv(I, bN), NB, threads=th) as (bn, bm):
            rows = T.alloc_shared((bM,), T.int32)
            real = T.alloc_shared((bM,), T.int32)
            lut = T.alloc_shared((256,), DTYPE)
            l4 = T.alloc_shared((16,), DTYPE)
            xs = T.alloc_shared((bM, bK), DTYPE)
            gs = T.alloc_shared((bN, bK), DTYPE)
            us = T.alloc_shared((bN, bK), DTYPE)
            gw = T.alloc_shared((bN, Kw), T.int32)
            uw = T.alloc_shared((bN, Kw), T.int32)
            gb = T.alloc_shared((bN, bK // 64), T.int32)
            ub = T.alloc_shared((bN, bK // 64), T.int32)
            ag = T.alloc_fragment((bM, bN), ACC)
            au = T.alloc_fragment((bM, bN), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                sg = Sc[e, 0]
                su = Sc[e, 1]
                if fmt == "fp8" or fmt == "fp4":
                    T.copy(L8, lut)
                if fmt == "fp4":
                    T.copy(L4, l4)
                for i in T.Parallel(bM):
                    rows[i] = T.min(Sid[bm * bM + i], N - 1) // topk
                    real[i] = Sid[bm * bM + i]
                T.clear(ag)
                T.clear(au)
                for k in T.serial(T.ceildiv(H, bK)):
                    for i, kk in T.Parallel(bM, bK):
                        xs[i, kk] = X[rows[i], k * bK + kk]
                    T.copy(Wq[e, bn * bN:bn * bN + bN, k * Kw:k * Kw + Kw], gw)
                    T.copy(Wq[e, I + bn * bN:I + bn * bN + bN, k * Kw:k * Kw + Kw], uw)
                    if fmt == "fp4":
                        Kb = bK // 64
                        T.copy(Ws[e, bn * bN:bn * bN + bN, k * Kb:k * Kb + Kb], gb)
                        T.copy(Ws[e, I + bn * bN:I + bn * bN + bN, k * Kb:k * Kb + Kb], ub)
                    if fmt == "fp8":
                        for j, kk in T.Parallel(bN, bK):
                            gs[j, kk] = lut[(gw[j, kk // 4] >> ((kk % 4) * 8)) & 255]
                    else:
                        for j, kk in T.Parallel(bN, bK):
                            gs[j, kk] = l4[(gw[j, kk // 8] >> ((kk % 8) * 4)) & 15] * lut[(gb[j, kk // 64] >> (((kk // 16) % 4) * 8)) & 255]
                    if fmt == "fp8":
                        for j, kk in T.Parallel(bN, bK):
                            us[j, kk] = lut[(uw[j, kk // 4] >> ((kk % 4) * 8)) & 255]
                    else:
                        for j, kk in T.Parallel(bN, bK):
                            us[j, kk] = l4[(uw[j, kk // 8] >> ((kk % 8) * 4)) & 15] * lut[(ub[j, kk // 64] >> (((kk // 16) % 4) * 8)) & 255]
                    T.gemm(xs, gs, ag, transpose_B=True, k_pack=kp, policy=M.POL[pol])
                    T.gemm(xs, us, au, transpose_B=True, k_pack=kp, policy=M.POL[pol])
                for i, j in T.Parallel(bM, bN):
                    g = ag[i, j] * sg
                    if real[i] < N:
                        Inter[bm * bM + i, bn * bN + j] = T.cast(g / (1.0 + T.exp(-g)) * (au[i, j] * su), DTYPE)

    return main


def down_q(fmt, Tn, topk, H, I, E, P, bM, bN, bK, th, kp):
    NB, N = P // bM, Tn * topk
    per = 4 if fmt == "fp8" else 8
    Iw, Kw = I // per, bK // per

    @T.prim_func
    def main(Inter: T.Tensor((P, I), DTYPE), Wq: T.Tensor((E, H, Iw), T.int32), Ws: T.Tensor((E, H, I // 64), T.int32), L8: T.Tensor((256,), DTYPE), L4: T.Tensor((16,), DTYPE),
             Sc: T.Tensor((E,), ACC), Sid: T.Tensor((P,), T.int32), Bexp: T.Tensor((NB,), T.int32), Wt: T.Tensor((N,), ACC), Y: T.Tensor((N + 1, H), DTYPE)):
        with T.Kernel(T.ceildiv(H, bN), NB, threads=th) as (bn, bm):
            lut = T.alloc_shared((256,), DTYPE)
            l4 = T.alloc_shared((16,), DTYPE)
            a_s = T.alloc_shared((bM, bK), DTYPE)
            w_s = T.alloc_shared((bN, bK), DTYPE)
            ww = T.alloc_shared((bN, Kw), T.int32)
            wb = T.alloc_shared((bN, bK // 64), T.int32)
            ac = T.alloc_fragment((bM, bN), ACC)
            sf = T.alloc_fragment((bM,), T.int32)
            wf = T.alloc_fragment((bM,), ACC)
            if Bexp[bm] >= 0:
                e = Bexp[bm]
                sc = Sc[e]
                T.copy(L8, lut)
                if fmt == "fp4":
                    T.copy(L4, l4)
                T.clear(ac)
                for k in T.serial(T.ceildiv(I, bK)):
                    T.copy(Inter[bm * bM:bm * bM + bM, k * bK:k * bK + bK], a_s)
                    T.copy(Wq[e, bn * bN:bn * bN + bN, k * Kw:k * Kw + Kw], ww)
                    if fmt == "fp4":
                        Kb = bK // 64
                        T.copy(Ws[e, bn * bN:bn * bN + bN, k * Kb:k * Kb + Kb], wb)
                    if fmt == "fp8":
                        for j, kk in T.Parallel(bN, bK):
                            w_s[j, kk] = lut[(ww[j, kk // 4] >> ((kk % 4) * 8)) & 255]
                    else:
                        for j, kk in T.Parallel(bN, bK):
                            w_s[j, kk] = l4[(ww[j, kk // 8] >> ((kk % 8) * 4)) & 15] * lut[(wb[j, kk // 64] >> (((kk // 16) % 4) * 8)) & 255]
                    T.gemm(a_s, w_s, ac, transpose_B=True, k_pack=kp, policy=GemmWarpPolicy.FullRow)
                for i in T.Parallel(bM):
                    s0 = Sid[bm * bM + i]
                    sf[i] = T.min(s0, N)
                    wf[i] = T.if_then_else(s0 < N, Wt[T.min(s0, N - 1)], T.cast(0, ACC)) * sc
                for i, j in T.Parallel(bM, bN):
                    Y[sf[i], bn * bN + j] = T.cast(ac[i, j] * wf[i], DTYPE)

    return main


# ---- encoding, CPU, one matrix at a time. Returns words, block scales (fp4), matrix scales, dequantised fp16-free fp32 weights.
_F8 = torch.arange(128, dtype=torch.uint8).view(torch.float8_e4m3fn).float()   # positive codes
_F8[127] = _F8[126]                                                            # 0x7f is NaN in e4m3fn


def enc_fp8(W):
    """W [..., K] float. Scale maps the matrix maximum to 448."""
    sc = W.abs().amax().clamp_min(1e-30) / 448.0
    q = (W / sc).to(torch.float8_e4m3fn)
    dq = q.float() * sc
    return q.view(torch.uint8), None, sc, dq


def pack(u8, per):
    """Little-endian packing into int32 words. per = elements per word, 4 (bytes) or 8 (nibbles), or byte array for 4."""
    bits = 32 // per
    n = u8.shape[-1] // per
    v = u8.reshape(*u8.shape[:-1], n, per).long()
    w = sum(v[..., i] << (i * bits) for i in range(per))
    return (w & 0xffffffff).to(torch.int64).where(w < 2**31, w - 2**32).to(torch.int32)


def enc_fp4(W):
    """W [R, K] float, K % 64 == 0. Block scale: the E4M3 value at or above block max / 6, after a matrix fp32 scale that puts the largest at 448."""
    R, K = W.shape
    b = W.reshape(R, K // 16, 16)
    bmax = b.abs().amax(-1)
    sc = (bmax.amax().clamp_min(1e-30) / 6.0) / 448.0
    want = bmax / 6.0 / sc
    idx = torch.searchsorted(_F8[:127], want.contiguous()).clamp_max(126)
    sv = _F8[idx]                                                                # E4M3 value, rounded up
    nz = sv > 0
    r = b.abs() / (sv * sc).clamp_min(1e-30).unsqueeze(-1)                       # block value in raw units; dropping sc here zeroed every code
    tab = torch.tensor(E2M1)
    code = (r.unsqueeze(-1) - tab).abs().argmin(-1)                             # nearest E2M1 magnitude
    code = torch.where(nz.unsqueeze(-1), code, torch.zeros_like(code))
    sign = (b < 0).long()
    dq = tab[code] * (1 - 2 * sign) * sv.unsqueeze(-1) * sc
    nib = (code | (sign << 3)).reshape(R, K).to(torch.uint8)
    return nib, idx.to(torch.uint8), sc, dq.reshape(R, K)


def encode(fmt, W13, W2, dev):
    """Per expert: gate and up get their own scale, W2 its own. Returns device tensors and dequantised fp32 CPU weights."""
    E, I2, H = W13.shape
    I = I2 // 2
    enc = enc_fp8 if fmt == "fp8" else enc_fp4
    per = 4 if fmt == "fp8" else 8
    Q13, B13, S13, D13, Q2, B2, S2, D2 = [], [], [], [], [], [], [], []
    for e in range(E):
        rq, rb, rs, rd = [], [], [], []
        for part in (W13[e, :I].float(), W13[e, I:].float()):
            q, b, s, d = enc(part)
            rq.append(q); rb.append(b); rs.append(s); rd.append(d)
        Q13.append(pack(torch.cat(rq), per)); S13.append(torch.stack(rs)); D13.append(torch.cat(rd))
        if fmt == "fp4":
            B13.append(pack(torch.cat(rb), 4))
        q, b, s, d = enc(W2[e].float())
        Q2.append(pack(q, per)); S2.append(s); D2.append(d)
        if fmt == "fp4":
            B2.append(pack(b, 4))
    z13 = torch.zeros(E, I2, H // 64, dtype=torch.int32) if fmt == "fp8" else torch.stack(B13)
    z2 = torch.zeros(E, H, W2.shape[2] // 64, dtype=torch.int32) if fmt == "fp8" else torch.stack(B2)
    out = dict(Q13=torch.stack(Q13).to(dev), B13=z13.to(dev), S13=torch.stack(S13).float().to(dev), Q2=torch.stack(Q2).to(dev), B2=z2.to(dev), S2=torch.stack(S2).float().to(dev))
    return out, torch.stack(D13), torch.stack(D2)


class QMoE:
    """Landed align and sum, decoded gate/up and down, landed tiles for T on this compiler."""

    def __init__(self, fmt, Tn, Q, dev="cuda", topk=8, H=2048, I=512, E=256):
        self.m = m = M.FusedMoE(Tn, topk, H, I, E, device=dev)
        c = m.cfg
        self.Q, self.fmt = Q, fmt
        # Mid tile on gate/up, three fp16 x 16 KB plus two packed x 8 KB, overruns the 64 KB of LDS (launch fails at T = 1024, 23525758). Same bM, so align holds: use the large gate/up.
        g = dict(M._LARGE_C22) if (c["bM"], c["bK"]) == (64, 128) else c
        self.k1 = M._build(gate_up_q(fmt, Tn, topk, H, I, E, m.P, g["bM"], g["bN"], g["bK"], g["th"], g["kp"], g.get("gu_pol", "row")), [8], tuple(g.get("gu_flags") or ()) or None)
        self.k2 = M._build(down_q(fmt, Tn, topk, H, I, E, m.P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"]), [9])
        self.L8, self.L4 = LUT8.to(dev), LUT4.to(dev)

    def __call__(self, X, ids, wts, rec=None):
        rec = rec or (lambda i: None)
        m, q = self.m, self.Q
        idf, wf = ids.flatten(), wts.flatten()
        rec(0)
        s, be = m.align(idf)
        rec(1)
        inter = self.k1(X, q["Q13"], q["B13"], self.L8, self.L4, q["S13"], s, be)
        rec(2)
        y = self.k2(inter, q["Q2"], q["B2"], self.L8, self.L4, q["S2"], s, be, wf)
        rec(3)
        o = m.ks(y[: m.N])
        rec(4)
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--cases", default=""); ap.add_argument("--rounds", type=int, default=4); ap.add_argument("--rep", type=int, default=25); ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--stack", default="image"); ap.add_argument("--fmts", default="fp8,fp4"); ap.add_argument("--layer", default="syn"); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--vllm", action="store_true")
    a = ap.parse_args()
    H, I, E, topk = 2048, 512, 256, 8
    D = torch.load(a.data)
    W13c, W2c = D["W13"], D["W2"]
    W13, W2 = W13c.cuda(), W2c.cuda()
    meta = dict(stack=a.stack, torch=torch.__version__, layer=a.layer, jobid=os.environ.get("SLURM_JOB_ID"), set="fp")
    emit = lambda **k: print(json.dumps({**meta, **k}), flush=True)
    wstats = dict(w13_max=W13c.abs().float().max().item(), w13_rms=W13c.float().pow(2).mean().sqrt().item(), w2_max=W2c.abs().float().max().item(), w2_rms=W2c.float().pow(2).mean().sqrt().item())
    emit(kind="weights", **wstats)
    QS, DQ = {}, {}
    for fmt in a.fmts.split(","):
        t0 = time.time()
        QS[fmt], d13, d2 = encode(fmt, W13c, W2c, "cuda")
        DQ[fmt] = (d13.cuda(), d2.cuda())
        emit(kind="encoded", fmt=fmt, s=round(time.time() - t0, 1),
             wrel13=round(((d13 - W13c.float()).norm() / W13c.float().norm()).item(), 6), wrel2=round(((d2 - W2c.float()).norm() / W2c.float().norm()).item(), 6),
             bytes_per_expert=QS[fmt]["Q13"][0].numel() * 4 + QS[fmt]["Q2"][0].numel() * 4 + (0 if fmt == "fp8" else QS[fmt]["B13"][0].numel() * 4 + QS[fmt]["B2"][0].numel() * 4))
    def ref(X, ids, wts, w13, w2):
        out = torch.zeros(ids.shape[0], H, dtype=torch.float32, device=X.device)
        for e in ids.unique().tolist():
            tok, kk = (ids == e).nonzero(as_tuple=True)
            h = X[tok].float() @ w13[e].float().T
            ac = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
            out.index_add_(0, tok, (ac @ w2[e].float().T) * wts[tok, kk, None])
        return out
    EV = lambda: torch.cuda.Event(enable_timing=True)
    def bench(fn):
        for _ in range(a.warm):
            fn(None)
        evs = []
        for _ in range(a.rep):
            ev = [EV() for _ in range(5)]
            fn(lambda i, ev=ev: ev[i].record())
            evs.append(ev)
        torch.cuda.synchronize()
        t = torch.tensor([[x[i].elapsed_time(x[i + 1]) for i in range(4)] for x in evs], dtype=torch.float64)
        return dict(stages=[round(v, 4) for v in t.median(0).values.tolist()], st_tot=round(t.sum(1).median().item(), 4), st_min=round(t.sum(1).min().item(), 4))
    def plain(fn):
        for _ in range(a.warm):
            fn()
        p = []
        for _ in range(a.rep):
            s, e = EV(), EV()
            s.record(); fn(); e.record()
            p.append((s, e))
        torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) for s, e in p], dtype=torch.float64)
        return dict(tot_med=round(t.median().item(), 4), tot_min=round(t.min().item(), 4))
    for case in a.cases.split(","):
        Tn, route = int(case.split(":")[0]), case.split(":")[1]
        d = D[(Tn, route)]
        X, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
        R = ref(X, ids, wts, W13, W2)
        hits = ids.flatten().bincount(minlength=E)
        emit(T=Tn, route=route, kind="routing", experts_hit=int((hits > 0).sum()))
        ship = M.FusedMoE(Tn)
        arms = {"fp16": (lambda rec=None: ship(X, W13, W2, ids, wts, rec))}
        for fmt in a.fmts.split(","):
            try:
                q = QMoE(fmt, Tn, QS[fmt])
            except Exception as ex:
                emit(T=Tn, route=route, kind="arm_failed", arm=fmt, err=(type(ex).__name__ + ": " + str(ex))[-500:])
                continue
            arms[fmt] = (lambda rec=None, q=q: q(X, ids, wts, rec))
            O = q(X, ids, wts)
            Rq = ref(X, ids, wts, *DQ[fmt])
            emit(T=Tn, route=route, kind="err", arm=fmt, rel_vs_fp16w=round(((O.float() - R).norm() / R.norm()).item(), 6),
                 kernel_vs_dq=round(((O.float() - Rq).norm() / Rq.norm()).item(), 7), dq_vs_fp16w=round(((Rq - R).norm() / R.norm()).item(), 6),
                 nonfinite=int((~torch.isfinite(O.float())).sum()), cos=round(torch.nn.functional.cosine_similarity(O.float().flatten(), R.flatten(), dim=0).item(), 6))
        O = ship(X, W13, W2, ids, wts)
        emit(T=Tn, route=route, kind="err", arm="fp16", rel_vs_fp16w=round(((O.float() - R).norm() / R.norm()).item(), 6), nonfinite=int((~torch.isfinite(O.float())).sum()))
        order = list(arms)[::-1] if a.reverse else list(arms)
        for r in range(a.rounds):
            sh = r * len(order) // a.rounds
            rot = order[sh:] + order[:sh]
            emit(T=Tn, route=route, kind="order", round=r, order=rot)
            for n in rot:
                res = plain(arms[n]); res.update(bench(arms[n]))
                emit(T=Tn, route=route, kind="time", arm=n, round=r, **res)


if __name__ == "__main__":
    main()
