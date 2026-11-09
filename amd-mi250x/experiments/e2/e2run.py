"""E2 timing. One process per stack. All arms use one bench: per-iteration events around the stages, with and without an
L2-flushing cache.zero_() before each iteration (tilelang do_bench flushes; the old vllm_moe_baseline.py did not)."""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--tokens", default="4096,1024"); ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--rep", type=int, default=30); ap.add_argument("--warm", type=int, default=5); ap.add_argument("--stack")
ap.add_argument("--only", default=""); ap.add_argument("--no-tl", action="store_true"); ap.add_argument("--no-vllm", action="store_true")
a = ap.parse_args()
H, I, E, topk = 2048, 512, 256, 8
bM, bN, bK, dN, th, kp = 64, 64, 64, 128, 256, 2
D = torch.load(a.data)
W13, W2 = D["W13"].cuda(), D["W2"].cuda()
meta = {"stack": a.stack, "torch": torch.__version__, "hip": getattr(torch.version, "hip", None)}
EV = lambda: torch.cuda.Event(enable_timing=True)
FL = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")


def bench(fn, nst, flush):
    """fn(ev) records nst+1 events. Returns per-stage median ms, total mean/median/min."""
    for _ in range(a.warm):
        fn([EV() for _ in range(nst + 1)])
    evs = []
    for _ in range(a.rep):
        if flush:
            FL.zero_()
        ev = [EV() for _ in range(nst + 1)]
        fn(ev)
        evs.append(ev)
    torch.cuda.synchronize()
    t = torch.tensor([[e[i].elapsed_time(e[i + 1]) for i in range(nst)] for e in evs], dtype=torch.float64)
    tot = t.sum(1)
    return {"stages": [round(x, 4) for x in t.median(0).values.tolist()], "tot_med": round(tot.median().item(), 4),
            "tot_mean": round(tot.mean().item(), 4), "tot_min": round(tot.min().item(), 4)}


def reference(X, ids, wts):
    Tn, _ = ids.shape
    out = torch.zeros(Tn, H, dtype=torch.float32, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].float() @ W13[e].float().T
        ac = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (ac @ W2[e].float().T) * wts[tok, kk, None])
    return out


def emit(**k):
    print(json.dumps({**meta, **k}), flush=True)


def tl_arms(Tn, X, ids, wts, R):
    import e2k
    from moe_fused import moe_align_gpu
    N = Tn * topk
    idf, wf = ids.flatten().contiguous(), wts.flatten().contiguous()
    Xz = torch.cat([X, torch.zeros(1, H, device=X.device, dtype=X.dtype)])
    kc = {}
    def align(bm, tight):
        key = ("al", bm, tight)
        if key not in kc:
            if tight == 3:
                ka3a = e2k.build("al3a", ns=True, out=False, N=N, E=E, bM=bm, threads=1024)
                ka3b = e2k.build("al3b", ns=True, out=False, N=N, E=E, bM=bm, threads=256)
                Pm_ = pmax(bm)
                def al3(ii):
                    s_ = torch.empty((Pm_,), dtype=torch.int32, device="cuda")
                    b_ = torch.empty((Pm_ // bm,), dtype=torch.int32, device="cuda")
                    c_ = torch.empty((E,), dtype=torch.int32, device="cuda")
                    ka3a(ii, s_, b_, c_)
                    ka3b(ii, s_, c_)
                    return s_, b_
                kc[key] = al3
            else:
                kc[key] = e2k.build("al", ns=True, N=N, E=E, bM=bm, threads=1024) if tight else moe_align_gpu(N, E, bm)
        return kc[key]
    def pmax(bm):
        return (N + E * (bm - 1) + bm - 1) // bm * bm
    def gk(c, P):
        key = ("gu", P, tuple(sorted(c.items())))
        if key not in kc:
            c2 = dict(c); ns = c2.pop("ns", False); out = c2.pop("out", True); fl = c2.pop("flags", None)
            fl = list(fl) if fl else None
            kc[key] = e2k.build("gu", ns=ns, out=out, flags=fl, **dict(dict(Tn=Tn, topk=topk, H=H, I=I, E=E, P=P, bM=bM, bN=bN, bK=bK, th=th, kp=kp), **c2))
        return kc[key]
    def dk(c, P):
        key = ("dn", P, tuple(sorted(c.items())))
        if key not in kc:
            c2 = dict(c); ns = c2.pop("ns", False); out = c2.pop("out", True); fl = c2.pop("flags", None)
            fl = list(fl) if fl else None
            kc[key] = e2k.build("dn", ns=ns, out=out, flags=fl, **dict(dict(Tn=Tn, topk=topk, H=H, I=I, E=E, P=P, bM=bM, bN=dN, bK=bK, th=th, kp=kp), **c2))
        return kc[key]
    sk = e2k.build("sum", ns=True, Tn=Tn, topk=topk, H=H, th=256)
    sk16 = e2k.build("sum", ns=True, Tn=Tn, topk=topk, H=H, th=256, otype=e2k.dtype)
    tsum = lambda y: y.view(Tn, topk, H).sum(1, dtype=torch.float32)
    v2 = dict(x="v2", ns=True); v2c = dict(x="v2c", ns=True); hns = dict(hoist=True, ns=True); nsd = dict(ns=True)
    SW = dict(swap=True)
    A = {
      "base": ({}, {}, tsum), "gu_ns": (dict(ns=True), {}, tsum), "dn_ns": ({}, nsd, tsum), "both_ns": (dict(ns=True), nsd, tsum),
      "dn_h": ({}, dict(hoist=True), tsum), "all": (dict(ns=True), hns, tsum), "v2": (dict(x="v2"), {}, tsum), "v2ns": (v2, {}, tsum),
      "all_v2": (v2, hns, tsum), "all_v2_sum": (v2, hns, sk), "all_v2_sum16": (v2, hns, sk16),
      "base_sum": ({}, {}, sk),
      "swgu": (dict(v2, **SW), hns, sk), "swdn": (v2, dict(hns, **SW), sk), "swboth": (dict(v2, **SW), dict(hns, **SW), sk),
      "swboth_nohoist": (dict(v2, **SW), dict(nsd, **SW), sk), "swboth_v2c": (dict(v2c, **SW), dict(hns, **SW), sk),
      "swboth_v2c_nh": (dict(v2c, **SW), dict(nsd, **SW), sk),
      "maskgu": (dict(v2, mask=True), hns, sk), "maskdn": (v2, dict(hns, mask=True), sk), "maskboth": (dict(v2, mask=True), dict(hns, mask=True), sk),
      "swmask": (dict(v2, mask=True, **SW), dict(hns, mask=True, **SW), sk),
      "swmaskdn_nh": (dict(v2c, **SW), dict(nsd, mask=True, **SW), sk),
      "swmaskdn": (dict(v2c, **SW), dict(hns, mask=True, **SW), sk),
      "dump": (dict(v2c, **SW), dict(hns, dump=True, **SW), sk),
      # tile sweep on the swapped grid
      "t_g64k32_d64k32": (dict(v2c, bK=32, **SW), dict(nsd, bK=32, **SW), sk),
      "t_gN32": (dict(v2c, bN=32, **SW), dict(nsd, **SW), sk),
      "t_dN64": (dict(v2c, **SW), dict(nsd, bN=64, **SW), sk),
      "t_dN256": (dict(v2c, **SW), dict(nsd, bN=256, bK=32, **SW), sk),
      "t_gN128": (dict(v2c, bN=128, bK=32, **SW), dict(nsd, **SW), sk),
      "t_kp1": (dict(v2c, kp=1, **SW), dict(nsd, kp=1, **SW), sk),
      "t_bM128": (dict(v2c, bM=128, bN=64, bK=32, **SW), dict(nsd, bM=128, bN=128, bK=32, **SW), sk, 128),
      "t_bM128_512": (dict(v2c, bM=128, bN=64, bK=64, th=512, **SW), dict(nsd, bM=128, bN=128, bK=64, th=512, **SW), sk, 128),
      "t_bM32": (dict(v2c, bM=32, bN=64, th=128, **SW), dict(nsd, bM=32, bN=128, th=128, **SW), sk, 32),
    }
    gsw = dict(v2, **SW); gmsw = dict(v2, mask=True, **SW); ncw = dict(nsd, **SW); hsw = dict(hns, **SW); dsw = dict(hns, dump=True, **SW)
    A.update({
      "F1": (gsw, ncw, sk, bM, 3), "F2_xcat": (gsw, ncw, sk, bM, 3, True), "F3_dump": (gsw, dsw, sk, bM, 3), "F3_dump_xcat": (gsw, dsw, sk, bM, 3, True),
      "F4_mg": (gmsw, ncw, sk, bM, 3), "F5_mg_dump": (gmsw, dsw, sk, bM, 3), "F6_mgd": (gmsw, dict(hsw, mask=True), sk, bM, 3),
      "U1": (gmsw, dict(dsw, flags=("-mllvm", "-unroll-threshold=2000")), sk, bM, 3),
      "U2": (dict(gmsw, flags=("-mllvm", "-unroll-threshold=20000")), dict(dsw, flags=("-mllvm", "-unroll-threshold=2000")), sk, bM, 3),
      "U3": (gmsw, dict(ncw, flags=("-mllvm", "-unroll-threshold=2000")), sk, bM, 3),
      "F7_hoist": (gsw, hsw, sk, bM, 3), "F8_v2c_dump": (dict(v2c, **SW), dsw, sk, bM, 3),
    })
    if a.only:
        A = {k: v for k, v in A.items() if k in a.only.split(",")}
    def mkarm(g, d, sf, bm=bM, tight=False, P=None, trunc=None, xcat=False):
        Pm = P or pmax(bm)
        ka = align(bm, tight)
        k1, k2 = gk(g, Pm), dk(d, Pm)
        x = Xz if g.get("x") == "v2" else X
        dump = d.get("dump", False)
        zr = torch.zeros(1, H, device=X.device, dtype=X.dtype)
        def fn(ev):
            ev[0].record()
            xx = torch.cat([X, zr]) if xcat else x
            s, be = ka(idf)
            if trunc:
                s, be = s[:trunc[0]], be[:trunc[1]]
            ev[1].record(); inter = k1(xx, W13, s, be)
            ev[2].record(); y = k2(inter, W2, s, be, wf)
            ev[3].record(); o = sf(y[:N] if dump else y)
            ev[4].record()
            return o
        return fn
    arms = {}
    # geometry per block size
    for bm in (64,):
        s0, be0 = align(bm, False)(idf)
        Pm = s0.numel()
        valid = s0[s0 < N].long(); blk = torch.nonzero(s0 < N).flatten() // bm
        ok = bool(torch.equal(valid.sort().values, torch.arange(N, device=valid.device)) and torch.equal(be0[blk].long(), idf[valid].long()))
        NBr = int((be0 >= 0).sum()); Pr = NBr * bm
        emit(T=Tn, kind="geometry", bM=bm, Pmax=Pm, NBmax=Pm // bm, P_used=Pr, NB_used=NBr, pad_rows=Pr - N, align_ok=ok, experts_hit=int(ids.unique().numel()))
    s1, be1 = align(bM, True)(idf)
    s0, be0 = align(bM, False)(idf)
    emit(T=Tn, kind="align_tight_check", bexp_equal=bool(torch.equal(be0, be1)), sid_set_equal=bool(torch.equal(s0.sort().values, s1.sort().values)))
    s3, be3 = align(bM, 3)(idf)
    emit(T=Tn, kind="align3_check", bexp_equal=bool(torch.equal(be0, be3)), sid_set_equal=bool(torch.equal(s0.sort().values, s3.sort().values)))
    def add(n, *arg, **kw):
        try:
            fn = mkarm(*arg, **kw)
            O = fn([EV() for _ in range(5)])
            emit(T=Tn, kind="err", arm=n, rel_vs_fp32=round(((O.float() - R).norm() / R.norm()).item(), 7))
            arms[n] = fn
        except Exception as ex:  # noqa: BLE001
            emit(T=Tn, kind="arm_failed", arm=n, err=(type(ex).__name__ + ": " + str(ex))[-400:])
    for n, sp in A.items():
        add(n, sp[0], sp[1], sp[2], bm=(sp[3] if len(sp) > 3 else bM), tight=(sp[4] if len(sp) > 4 else False), xcat=(sp[5] if len(sp) > 5 else False))
    if not a.only:
        add("swboth_v2c_nh_al2", dict(v2c, **SW), dict(nsd, **SW), sk, tight=True)
        add("swboth_v2c_al2", dict(v2c, **SW), dict(hns, **SW), sk, tight=True)
        add("align2_only", {}, {}, tsum, tight=True)
        add("swboth_v2c_nh_al3", dict(v2c, **SW), dict(nsd, **SW), sk, tight=3)
        add("swboth_v2c_al3", dict(v2c, **SW), dict(hns, **SW), sk, tight=3)
        add("trunc", v2, hns, sk, P=Pr, trunc=(Pr, NBr))
        k1p, k2p = gk(dict(v2, out=False), Pm), dk(dict(hns, out=False), Pm)
        skp = e2k.build("sum", ns=True, out=False, Tn=Tn, topk=topk, H=H, th=256)
        ka = align(bM, False)
        Ip, Yp, Op = torch.empty((Pm, I), dtype=torch.float16, device="cuda"), torch.empty((N, H), dtype=torch.float16, device="cuda"), torch.empty((Tn, H), dtype=torch.float32, device="cuda")
        def fpre(ev):
            ev[0].record(); s, be = ka(idf)
            ev[1].record(); k1p(Xz, W13, s, be, Ip)
            ev[2].record(); k2p(Ip, W2, s, be, wf, Yp)
            ev[3].record(); skp(Yp, Op)
            ev[4].record()
            return Op
        O = fpre([EV() for _ in range(5)]); arms["prealloc"] = fpre
        emit(T=Tn, kind="err", arm="prealloc", rel_vs_fp32=round(((O.float() - R).norm() / R.norm()).item(), 7))
    return arms


def vllm_arms(Tn, X, ids, wts, R):
    import functools
    from vllm.triton_utils import tl
    from vllm.model_executor.layers.fused_moe import fused_moe as FM
    from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation, MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm import _custom_ops as ops
    idi = ids.int()
    act = MoEActivation.from_str("silu")
    dt = FM._get_config_dtype_str(use_fp8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, dtype=X.dtype)
    cfg = FM.try_get_optimal_moe_config(W13.size(), W2.size(), topk, dt, Tn, block_shape=None)
    emit(T=Tn, kind="vllm_config", cfg={k: v for k, v in cfg.items()})
    N1 = W13.size(1)
    KW = dict(compute_type=tl.float16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False,
              per_channel_quant=False, block_shape=None)
    def mk(cf):
        def fn(ev):
            c13 = torch.empty(Tn * topk * max(N1, H), device=X.device, dtype=X.dtype)
            c1, c3 = c13[: Tn * topk * N1].view(Tn, topk, N1), c13[: Tn * topk * H].view(Tn, topk, H)
            c2 = torch.empty((Tn * topk, I), device=X.device, dtype=X.dtype)
            o = torch.empty_like(X)
            ev[0].record()
            sid, eid, npp = moe_align_block_size(idi, cf["BLOCK_SIZE_M"], E, None, ignore_invalid_experts=True)
            ev[1].record()
            FM.dispatch_fused_moe_kernel(X, W13, c1, None, None, None, wts, sid, eid, npp, False, topk, cf, B_bias=None, **KW)
            ev[2].record()
            apply_moe_activation(act, c2, c1.view(-1, N1))
            ev[3].record()
            FM.dispatch_fused_moe_kernel(c2, W2, c3, None, None, None, wts, sid, eid, npp, True, 1, cf, B_bias=None, **KW)
            ev[4].record()
            ops.moe_sum(c3.view(*c3.size()), o)
            ev[5].record()
            return o
        return fn
    fn = mk(cfg)
    O = fn([EV() for _ in range(6)])
    emit(T=Tn, kind="err", arm="vllm_stages", rel_vs_fp32=round(((O.float() - R).norm() / R.norm()).item(), 7))
    cf64 = dict(cfg, BLOCK_SIZE_M=64)
    fn64 = mk(cf64)
    O = fn64([EV() for _ in range(6)])
    emit(T=Tn, kind="err", arm="vllm_bm64", rel_vs_fp32=round(((O.float() - R).norm() / R.norm()).item(), 7))
    def whole(ev):
        ev[0].record()
        o = FM.fused_experts(X, W13, W2, wts, idi)
        ev[1].record()
        return o
    O = whole([EV(), EV()])
    emit(T=Tn, kind="err", arm="vllm_total", rel_vs_fp32=round(((O.float() - R).norm() / R.norm()).item(), 7))
    return {"vllm_stages": (fn, 5), "vllm_bm64": (fn64, 5), "vllm_total": (whole, 1)}


for Tn in [int(x) for x in a.tokens.split(",")]:
    d = D[Tn]
    X, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
    R = reference(X, ids, wts)
    arms = {}
    if not a.no_vllm and a.stack == "vllm":
        arms.update(vllm_arms(Tn, X, ids, wts, R))
    if not a.no_tl:
        arms.update({k: (v, 4) for k, v in tl_arms(Tn, X, ids, wts, R).items()})
    order = list(arms)
    for r in range(a.rounds):
        rot = order[r % len(order):] + order[:r % len(order)]
        for n in rot:
            fn, nst = arms[n]
            for fl in (0, 1):
                t0 = time.time()
                res = bench(fn, nst, bool(fl))
                emit(T=Tn, kind="time", arm=n, flush=fl, round=r, **res, wall_s=round(time.time() - t0, 2))
