"""N1 bench. One process per stack. Plain total: only two events around the whole call, for ours and for vLLM; stage events
in a separate pass. Arm order rotates between rounds.
--set a: X-path and lever arms at T=4096,1024 uniform. b: tiles at all T, 3 routings. c: chosen arms at all T, 3 routings.
u: unroll trace."""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--set", default="a"); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--rep", type=int, default=50); ap.add_argument("--warm", type=int, default=5); ap.add_argument("--stack")
ap.add_argument("--tokens", default=""); ap.add_argument("--routes", default=""); ap.add_argument("--arms", default="")
ap.add_argument("--cases", default=""); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--dump", default="")
ap.add_argument("--codir", default=""); ap.add_argument("--no-tl", action="store_true"); ap.add_argument("--no-vllm", action="store_true"); ap.add_argument("--cfgs", default=""); ap.add_argument("--build-only", action="store_true"); ap.add_argument("--percall", action="store_true")
a = ap.parse_args()
H, I, E, topk = 2048, 512, 256, 8
D = torch.load(a.data)
W13, W2 = D["W13"].cuda(), D["W2"].cuda()
meta = {"stack": a.stack, "torch": torch.__version__, "set": a.set, "jobid": os.environ.get("SLURM_JOB_ID")}
EV = lambda: torch.cuda.Event(enable_timing=True)
def emit(**k):
    print(json.dumps({**meta, **k}), flush=True)

def reference(X, ids, wts):
    Tn = ids.shape[0]
    out = torch.zeros(Tn, H, dtype=torch.float32, device=X.device)
    for e in ids.unique().tolist():
        tok, kk = (ids == e).nonzero(as_tuple=True)
        h = X[tok].float() @ W13[e].float().T
        ac = F.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, tok, (ac @ W2[e].float().T) * wts[tok, kk, None])
    return out

def bench_plain(fn):
    for _ in range(a.warm):
        fn(None)
    pairs = []
    for _ in range(a.rep):
        s, e = EV(), EV()
        s.record(); fn(None); e.record()
        pairs.append((s, e))
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) for s, e in pairs], dtype=torch.float64)
    r = dict(tot_med=round(t.median().item(), 4), tot_mean=round(t.mean().item(), 4), tot_min=round(t.min().item(), 4))
    if a.percall:
        r["calls"] = [round(x, 4) for x in t.tolist()]
    return r

def bench_stages(fn, nst):
    for _ in range(a.warm):
        fn(None)
    evs = []
    for _ in range(a.rep):
        ev = [EV() for _ in range(nst + 1)]
        fn(lambda i, ev=ev: ev[i].record())
        evs.append(ev)
    torch.cuda.synchronize()
    t = torch.tensor([[e[i].elapsed_time(e[i + 1]) for i in range(nst)] for e in evs], dtype=torch.float64)
    return dict(stages=[round(x, 4) for x in t.median(0).values.tolist()], st_tot=round(t.sum(1).median().item(), 4))

def errchk(n, fn, R, Tn, route, X, ids):
    O = fn(None)
    bad = int((~torch.isfinite(O.float())).sum())
    d = (O.float() - R)
    emit(T=Tn, route=route, kind="err", arm=n, rel_vs_fp32=round((d.norm() / R.norm()).item(), 7), max_abs=round(d.abs().max().item(), 6), nonfinite=bad)

def vllm_arms(Tn, X, ids, wts):
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
    KW = dict(compute_type=tl.float16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False, block_shape=None)
    def stg(rec):
        rec = rec or (lambda i: None)
        c13 = torch.empty(Tn * topk * max(N1, H), device=X.device, dtype=X.dtype)
        c1, c3 = c13[: Tn * topk * N1].view(Tn, topk, N1), c13[: Tn * topk * H].view(Tn, topk, H)
        c2 = torch.empty((Tn * topk, I), device=X.device, dtype=X.dtype)
        o = torch.empty_like(X)
        rec(0); sid, eid, npp = moe_align_block_size(idi, cfg["BLOCK_SIZE_M"], E, None, ignore_invalid_experts=True)
        rec(1); FM.dispatch_fused_moe_kernel(X, W13, c1, None, None, None, wts, sid, eid, npp, False, topk, cfg, B_bias=None, **KW)
        rec(2); apply_moe_activation(act, c2, c1.view(-1, N1))
        rec(3); FM.dispatch_fused_moe_kernel(c2, W2, c3, None, None, None, wts, sid, eid, npp, True, 1, cfg, B_bias=None, **KW)
        rec(4); ops.moe_sum(c3.view(*c3.size()), o)
        rec(5)
        return o
    def whole(rec):
        return FM.fused_experts(X, W13, W2, wts, idi)
    return {"vllm_total": (whole, None), "vllm_stages": (stg, 5)}

CF = {"d": dict(bM=64, bN=64, dN=128, bK=64, th=256, kp=2),
      "m32": dict(bM=32, bN=64, dN=128, bK=64, th=128, kp=2),
      "m16": dict(bM=16, bN=64, dN=128, bK=64, th=64, kp=2),
      "m16k": dict(bM=16, bN=32, dN=64, bK=128, th=64, kp=2),
      "m16n": dict(bM=16, bN=16, dN=64, bK=128, th=64, kp=2),
      "m8": dict(bM=8, bN=64, dN=128, bK=64, th=64, kp=2),
      "n32": dict(bM=64, bN=32, dN=128, bK=64, th=256, kp=2),
      "m32n": dict(bM=32, bN=32, dN=64, bK=128, th=128, kp=2)}

# set s: small tiles under explicit gate/up flags, bypassing FusedMoE's clang >= 20 gate as set u does.
SMALL = dict(kp=2, dN=64, dK=128, dth=64, al_th=1024, sum_th=256, gu_unroll=None)
TILES = {"A": dict(SMALL, bM=16, bN=32, bK=128, th=64), "D": dict(SMALL, bM=16, bN=64, bK=128, th=128)}
XTILES = {"E": dict(SMALL, bM=32, bN=32, bK=128, th=128), "F": dict(SMALL, bM=32, bN=64, bK=64, th=128), "G": dict(SMALL, bM=64, bN=64, bK=64, th=256)}
FLAGS = {"none": None, "u0": ("-mllvm", "-unroll-threshold=0"), "nounroll": ("-fno-unroll-loops",), "u20k": ("-mllvm", "-unroll-threshold=20000"), "ilp": ("-mllvm", "-misched=gcn-max-ilp"), "mmc": ("-mllvm", "-misched=gcn-max-memory-clause")}
MEMO = {}
_I = ("-mllvm", "-misched=gcn-max-ilp"); _U = ("-mllvm", "-unroll-threshold=20000")
RF = {"none": (None, None), "ilp": (_I, None), "mmc": (FLAGS["mmc"], None), "u20k": (_U, None), "ilpu": (_I + _U, None), "ilpd": (_I, _I), "ilpud": (_I + _U, _I)}
_A = TILES["A"]
SWEEP = {"S1": dict(_A, bK=256), "S2": dict(_A, th=128), "S3": dict(_A, bN=64, th=64), "S4": dict(_A, bN=64, bK=64, th=64), "S5": dict(_A, bM=32, th=64), "S6": dict(_A, bM=32, bN=64, th=128),
         "S7": dict(_A, bM=8, th=64), "S8": dict(_A, bN=16, bK=256, th=64), "Ad1": dict(_A, dK=256), "Ad2": dict(_A, dN=32), "Ad3": dict(_A, dN=128, dth=128), "Ad4": dict(_A, dth=128), "Ad5": dict(_A, dth=256), "Ad6": dict(_A, dK=64)}
_L = dict(bM=64, bN=64, bK=64, th=256, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=None)
SWEEP.update({"B1": dict(_L, bN=128, th=512), "B2": dict(_L, bM=128, th=512), "B3": dict(_L, bN=128), "B4": dict(_L, bK=128, th=512), "B5": dict(_L, bM=32, th=128), "B6": dict(_L, bM=128, bN=128, th=512), "B7": dict(_L, bM=128, bN=128, th=512, dN=128, dK=64, dth=512),
              "C1": dict(_L, dN=128, dK=64), "C2": dict(_L, dth=512), "C3": dict(_L, dK=64, dth=256), "C4": dict(_L, dN=128, dK=128, dth=512), "C5": dict(_L, sum_th=1024), "C6": dict(_L, bM=128, dN=64, dK=128, dth=256, th=256)})

_B6 = SWEEP["B6"]
SWEEP.update({"B6S": dict(_B6, gu_pol="square"), "B6SS": dict(_B6, gu_pol="square", dn_pol="square"), "LS": dict(_L, gu_pol="square"), "LSS": dict(_L, gu_pol="square", dn_pol="square"), "LDS": dict(_L, dn_pol="square"),
              "LKS": dict(_L, bK=128, gu_pol="square"), "MS": dict(TILES.get("M", {}), bM=64, bN=64, bK=128, th=512, dN=64, dK=128, dth=256, kp=2, al_th=1024, sum_th=512, gu_unroll=None, gu_pol="square")})

SWEEP.update({"W512": dict(_L, gu_pol="square", th=512, gu_flags=_I), "W128": dict(_L, gu_pol="square", th=128, gu_flags=_I)})

def _kdirs():
    import glob
    return {os.path.dirname(p) for p in glob.glob(os.environ["TILELANG_CACHE_DIR"] + "/**/device_kernel.cu", recursive=True)}

def dump_gu(new, name):
    """Disassembly of the gate/up code object among the cache dirs this arm's build created. Nothing new = cache hit."""
    import re, struct, subprocess
    os.makedirs(a.dump, exist_ok=True)
    gu = [d for d in new if "half_t* __restrict__ W13" in open(d + "/device_kernel.cu").read()]
    emit(kind="dump_map", arm=name, new_dirs=len(new), gate_up_dirs=len(gu))
    for dd in gu:
        data = open(dd + "/executable.so", "rb").read()
        i = data.find(b"__CLANG_OFFLOAD_BUNDLE__")
        while i >= 0:
            c = struct.unpack_from("<Q", data, i + 24)[0]; p = i + 32
            for _ in range(c):
                off, sz, tl = struct.unpack_from("<QQQ", data, p); p += 24
                tr = data[p:p + tl].decode(errors="replace"); p += tl
                if "gfx90a" in tr and sz > 0:
                    tmp = f"{a.dump}/{name}.co"; open(tmp, "wb").write(data[i + off:i + off + sz])
                    dis = subprocess.run(["/opt/rocm/llvm/bin/llvm-objdump", "-d", tmp], capture_output=True, text=True).stdout
                    open(f"{a.dump}/{name}.dis", "w").write(dis)
                    open(f"{a.dump}/{name}.json", "w").write(json.dumps({"arm": name, "res": json.load(open(dd + "/resource_usage.json"))}))
                    os.remove(tmp)
            i = data.find(b"__CLANG_OFFLOAD_BUNDLE__", i + 24)

def small_moe(M, Tn, cfg, fl, name, dfl=None):
    key = (Tn, name)
    if key not in MEMO:
        before = _kdirs() if a.dump else set()
        m = M.FusedMoE(Tn, cfg=cfg)
        c = m.cfg
        if fl:
            m.k1 = M._build(M.gate_up(Tn, topk, H, I, E, m.P, c["bM"], c["bN"], c["bK"], c["th"], c["kp"], "last", c.get("gu_pol", "row")), [4], fl)
        if dfl:
            m.k2 = M._build(M.down(Tn, topk, H, I, E, m.P, c["bM"], c["dN"], c["dK"], c["dth"], c["kp"], c.get("dn_pol", "row")), [5], dfl)
        if a.dump:
            dump_gu(_kdirs() - before, f"T{Tn}_{name}")
        MEMO[key] = m
    return MEMO[key]

def tl_arms(Tn, X, ids, wts, route):
    import moe_fused as M
    arms = {}
    def F5(n, **kw):
        try:
            m = M.FusedMoE(Tn, **kw)
            arms[n] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
        except Exception as ex:
            emit(T=Tn, route=route, kind="arm_failed", arm=n, err=(type(ex).__name__ + ": " + str(ex))[-500:])
    if a.set == "a":
        F5("F5x_cat", x="zero", x_copy="cat", out="fp32")     # asked for first: F5 with the X copy timed
        F5("F5x_buf", x="zero", x_copy="buffer", out="fp32")
        F5("F5l", x="last", out="fp32"); F5("F5s", x="self", out="fp32")
        m = M.FusedMoE(Tn, x="zero", out="fp32")
        Xz = torch.cat([X, torch.zeros(1, H, device=X.device, dtype=X.dtype)])
        arms["F5_prebuilt"] = (lambda rec, m=m, Xz=Xz: m(Xz, W13, W2, ids, wts, rec), 4)
        Xd = torch.cat([X, X[:1]])                            # same kernel, spare row holds real data: value vs address
        arms["F5_dirtyrow"] = (lambda rec, m=m, Xd=Xd: m(Xd, W13, W2, ids, wts, rec), 4)
        F5("F5l16", x="last", out="fp16")
        for th in (128, 64, 32):
            F5(f"F5l16_s{th}", x="last", out="fp16", sum_th=th)
        import e2k
        arms.update(ablation(Tn, X, ids, wts, e2k, M))
    elif a.set == "b":
        for c in a.cfgs.split(","):
            F5(f"l_{c}", x="last", out="fp16", cfg=CF[c])
    elif a.set == "c":
        for spec in a.arms.split(","):
            c, xp = spec.split(":")
            F5(f"{xp}_{c}", x=xp, out="fp16", cfg=CF[c])
    elif a.set == "k":
        base = dict(bM=64, bN=64, dN=128, bK=64, th=256, kp=2)
        K = {"base": {}, "d256": dict(dN=256), "d256t512": dict(dN=256, dth=512), "d128k128": dict(dK=128), "d64k128": dict(dN=64, dK=128),
             "dcol": dict(dpol="col"), "dkp1": dict(dkp=1), "dkp4": dict(dkp=4), "d128t512": dict(dth=512), "d128t128": dict(dth=128),
             "g32k128": dict(bN=32, bK=128, dK=64), "gt512": dict(bN=128, th=512, dth=256)}
        for n, c in K.items():
            F5(f"k_{n}", x="last", out="fp16", cfg=dict(base, **c))
        for n, kw in {"s512": dict(sum_th=512), "s1024": dict(sum_th=1024), "st8": dict(sum_tok=8), "st64": dict(sum_tok=64), "s1024t8": dict(sum_th=1024, sum_tok=8)}.items():
            F5(f"k_{n}", x="last", out="fp16", cfg=base, **kw)
    elif a.set == "j":
        for n, kw in json.load(open(a.arms)).items():
            kw = dict(kw); kw.setdefault("x", "last"); kw.setdefault("out", "fp16")
            kw["cfg"] = dict(dict(bM=64, bN=64, dN=64, dK=128, bK=64, th=256, kp=2), **kw.get("cfg", {}))
            F5(n, **kw)
    elif a.set == "s":
        for tn, cfg in TILES.items():
            for fn_, fl in FLAGS.items():
                m = small_moe(M, Tn, cfg, fl, f"{tn}_{fn_}")
                arms[f"{tn}_{fn_}"] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
        m = small_moe(M, Tn, dict(M._MID), None, "mid")
        emit(T=Tn, kind="ship_cfg", arm="mid", cfg=m.cfg, unroll=m.unroll, x=m.x)
        arms["mid"] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
        if a.dump and Tn == 16:      # no timing: static state of the tiles that showed no clang gap
            for tn, cfg in XTILES.items():
                small_moe(M, Tn, cfg, None, f"{tn}_none")
    elif a.set == "r4":
        # arms: mid | <tile>_<flag> | <tile>_ilpd | ship | co:<tag>. co arms load prebuilt code objects through hipModule (n1cold.py).
        import n1cold
        T_ = dict(TILES, M=dict(M._MID, gu_unroll=None), L=dict(M._LARGE, gu_unroll=None), **SWEEP)
        for spec in a.arms.split(","):
            arm, _, rng = spec.partition("@")
            if rng and not int(rng.split("-")[0]) <= Tn <= int(rng.split("-")[1]):
                continue
            if arm.startswith("co:"):
                try:
                    if (Tn, arm) not in MEMO:
                        MEMO[(Tn, arm)] = n1cold.COMoE(os.path.join(a.codir, f"T{Tn}_{arm[3:]}"))
                    m = MEMO[(Tn, arm)]
                except FileNotFoundError:
                    continue
            elif arm == "ship":
                if (Tn, arm) not in MEMO:
                    MEMO[(Tn, arm)] = M.FusedMoE(Tn); emit(T=Tn, kind="ship_cfg", cfg=MEMO[(Tn, arm)].cfg, unroll=MEMO[(Tn, arm)].unroll, gu_flags=MEMO[(Tn, arm)].gu_flags, x="last")
                m = MEMO[(Tn, arm)]
            elif arm == "row_large":       # control: the landed clang >= 20 large tile with the earlier FullRow gate/up
                if (Tn, arm) not in MEMO:
                    MEMO[(Tn, arm)] = M.FusedMoE(Tn, cfg=dict(M._LARGE_C22, gu_pol="row")); emit(T=Tn, kind="ship_cfg", arm=arm, cfg=MEMO[(Tn, arm)].cfg, gu_flags=MEMO[(Tn, arm)].gu_flags)
                m = MEMO[(Tn, arm)]
            elif arm == "mid":
                m = small_moe(M, Tn, dict(M._MID), None, "mid"); emit(T=Tn, kind="ship_cfg", arm="mid", cfg=m.cfg, unroll=m.unroll, x=m.x)
            else:
                tn, fn_ = arm.split("_")
                fl, dfl = RF[fn_]
                m = small_moe(M, Tn, T_[tn], fl, arm, dfl)
            arms[arm] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
    elif a.set == "g":
        import n1grid
        for spec in a.arms.split(","):
            if spec == "ship":
                if (Tn, spec) not in MEMO:
                    MEMO[(Tn, spec)] = M.FusedMoE(Tn); emit(T=Tn, kind="ship_cfg", cfg=MEMO[(Tn, spec)].cfg, gu_flags=MEMO[(Tn, spec)].gu_flags)
                m = MEMO[(Tn, spec)]
            else:          # grid<factor>: G is the next power of two times the factor
                mode = spec if spec in ("gridt", "gride") else "grid"
                G = n1grid.next_pow2(Tn) * (int(spec[4:]) if mode == "grid" and spec[4:] else 1)
                if (mode, G) not in MEMO:
                    MEMO[(mode, G)] = n1grid.GridMoE(G, sum_mode=mode); emit(T=Tn, kind="grid_cfg", G=G, mode=mode, cfg=MEMO[(mode, G)].m.cfg)
                m = MEMO[(mode, G)]
            arms[spec] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
    elif a.set == "f":
        m = M.FusedMoE(Tn)
        emit(T=Tn, kind="ship_cfg", cfg=m.cfg, unroll=m.unroll, x=m.x)
        arms["ship"] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
    elif a.set == "u":
        for u in a.arms.split(","):
            g, d = u.split(":")
            F5(f"z_g{g}_d{d}", x="zero", out="fp32", gu_unroll=int(g) or None, dn_unroll=int(d) or None)
            F5(f"l_g{g}_d{d}", x="last", out="fp32", gu_unroll=int(g) or None, dn_unroll=int(d) or None)
            import e2k
            m = M.FusedMoE(Tn, x="zero", out="fp32", dn_unroll=int(d) or None)
            P, c = m.P, m.cfg
            fl = ("-mllvm", f"-unroll-threshold={g}") if int(g) else None
            m.k1 = M._build(e2k.gate_up(Tn, topk, H, I, E, P, c["bM"], c["bN"], c["bK"], c["th"], c["kp"], x="v2", mask=False, swap=True), [4], fl)
            arms[f"nm_g{g}_d{d}"] = (lambda rec, m=m: m(X, W13, W2, ids, wts, rec), 4)
    return arms

def ablation(Tn, X, ids, wts, e2k, M):
    """Leave-one-out from F5 with the X copy timed; E2's kernels, E2's align at P = fixed bound."""
    bM, bN, bK, dN, th, kp = 64, 64, 64, 128, 256, 2
    N = Tn * topk
    idf, wf = ids.flatten().contiguous(), wts.flatten().contiguous()
    Pm = (N + E * (bM - 1) + bM - 1) // bM * bM
    al_o = M.moe_align_gpu(N, E, bM)
    ka3a = e2k.build("al3a", ns=True, out=False, N=N, E=E, bM=bM, threads=1024)
    ka3b = e2k.build("al3b", ns=True, out=False, N=N, E=E, bM=bM, threads=256)
    def al3(ii):
        s_ = torch.empty((Pm,), dtype=torch.int32, device="cuda"); b_ = torch.empty((Pm // bM,), dtype=torch.int32, device="cuda"); c_ = torch.empty((E,), dtype=torch.int32, device="cuda")
        ka3a(ii, s_, b_, c_); ka3b(ii, s_, c_)
        return s_, b_
    sk = e2k.build("sum", ns=True, Tn=Tn, topk=topk, H=H, th=256)
    tsum = lambda y: y.view(Tn, topk, H).sum(1, dtype=torch.float32)
    zr = torch.zeros(1, H, device=X.device, dtype=X.dtype)
    v2 = dict(x="v2", ns=True); SW = dict(swap=True)
    full = (dict(v2, mask=True, **SW), dict(hoist=True, ns=True, dump=True, **SW), sk, al3)
    A = {"a_full": full,
         "a_noswap": (dict(v2, mask=True), dict(hoist=True, ns=True, dump=True), sk, al3),
         "a_nomask": (dict(v2, **SW), full[1], sk, al3),
         "a_nov2ns": (dict(x="if", **SW), full[1], sk, al3),
         "a_nohoistdump": (full[0], dict(ns=True, **SW), sk, al3),
         "a_oldalign": (full[0], full[1], sk, al_o),
         "a_torchsum": (full[0], full[1], tsum, al3),
         "a_base": ({}, {}, tsum, al_o),
         "a_base_swap": (dict(SW), dict(SW), tsum, al_o)}
    out = {}
    D0 = dict(Tn=Tn, topk=topk, H=H, I=I, E=E, P=Pm, bM=bM, bK=bK, th=th, kp=kp)
    for n, (g, d, sf, al) in A.items():
        g2 = dict(g); d2 = dict(d); ns1 = g2.pop("ns", False); ns2 = d2.pop("ns", False)
        k1 = e2k.build("gu", ns=ns1, **dict(D0, bN=bN, **g2)); k2 = e2k.build("dn", ns=ns2, **dict(D0, bN=dN, **d2))
        zero = g.get("x") == "v2"; dump = d.get("dump", False)
        def fn(rec, k1=k1, k2=k2, sf=sf, al=al, zero=zero, dump=dump):
            rec = rec or (lambda i: None)
            rec(0); xx = torch.cat([X, zr]) if zero else X
            s, be = al(idf)
            rec(1); inter = k1(xx, W13, s, be)
            rec(2); y = k2(inter, W2, s, be, wf)
            rec(3); o = sf(y[:N] if dump else y)
            rec(4)
            return o
        out[n] = (fn, 4)
    return out

toks = [int(t) for t in a.tokens.split(",")] if a.tokens else ([4096, 1024] if a.set == "a" else [1, 16, 64, 256, 1024, 4096])
routes = a.routes.split(",") if a.routes else (["uniform"] if a.set in ("a", "u") else ["uniform", "zipf1", "zipf2"])
cases = [(int(c.split(":")[0]), c.split(":")[1]) for c in a.cases.split(",")] if a.cases else [(t, r) for t in toks for r in routes]
for Tn, route in cases:
    if True:
        d = D[(Tn, route)]
        X, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
        R = reference(X, ids, wts)
        hits = ids.flatten().bincount(minlength=E)
        emit(T=Tn, route=route, kind="routing", experts_hit=int((hits > 0).sum()), top1_share=round(hits.max().item() / hits.sum().item(), 4), blocks64=int(((hits + 63) // 64).sum()), blocks16=int(((hits + 15) // 16).sum()))
        arms = {}
        if not a.no_vllm and a.stack == "vllm":
            arms.update(vllm_arms(Tn, X, ids, wts))
        if not a.no_tl:
            arms.update(tl_arms(Tn, X, ids, wts, route))
        if a.build_only:
            emit(T=Tn, kind="prebuilt", arms=list(arms))
            continue
        for n, (fn, nst) in arms.items():
            try:
                errchk(n, fn, R, Tn, route, X, ids)
            except Exception as ex:
                emit(T=Tn, route=route, kind="err_failed", arm=n, err=(type(ex).__name__ + ": " + str(ex))[-400:])
        order = list(arms)[::-1] if a.reverse else list(arms)
        for r in range(a.rounds):
            sh = r * len(order) // max(a.rounds, 1)      # rotation spreads arms evenly, whatever the round count
            rot = order[sh:] + order[:sh]
            emit(T=Tn, route=route, kind="order", round=r, order=rot)
            for n in rot:
                fn, nst = arms[n]
                t0 = time.time()
                res = bench_plain(fn)
                if nst:
                    res.update(bench_stages(fn, nst))
                emit(T=Tn, route=route, kind="time", arm=n, round=r, **res, wall_s=round(time.time() - t0, 2))
