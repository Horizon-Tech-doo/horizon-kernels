"""N5 timing and correctness. One flavour per process.
Usage: n5_run.py --re REGEX --tag T [--inputs a,b] [--check] [--ck] [--rep N]
All arms of a (case, input) pair run in this process; CK and SDPA run in this process too when --ck is given (image only)."""
import argparse, json, os, re, time
import torch
import torch.nn.functional as F
from bench_common import bench
import patch
patch.install()
import n5lib
from n5lib import CASES, inputs, ref, build
from n5_arms import ARMS, SEL
ap = argparse.ArgumentParser()
ap.add_argument("--re", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--inputs", default="randn0")
ap.add_argument("--check", action="store_true"); ap.add_argument("--ck", action="store_true"); ap.add_argument("--sdpa", action="store_true"); ap.add_argument("--rep", type=int, default=100)
ap.add_argument("--gemm", action="store_true")
a = ap.parse_args()
FL, STACK = os.environ.get("N5_PATCH", ""), os.environ.get("N5_STACK", "")
pat = re.compile(SEL.get(a.re, a.re))
sel = [c for c in ARMS if pat.search(c["name"])]
print(json.dumps(dict(info=1, torch=torch.__version__, hip=getattr(torch.version, "hip", None), stack=STACK, patch=FL, tag=a.tag, n=len(sel))), flush=True)
rot = sum(map(ord, a.tag))
for case in sorted({c["case"] for c in sel}):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    rep = a.rep if L <= 8192 else max(20, a.rep // 4)
    for inp in a.inputs.split(","):
        Q, K, V = inputs(inp, s)
        R, Rs = None, None
        if a.check:
            if L <= 8192:
                R = ref(Q, K, V, causal, G)
        qt, kt, vt = Q.transpose(1, 2), K.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        base = dict(case=case, input=inp, tag=a.tag, stack=STACK, patch=FL)
        if a.sdpa or a.ck:
            r = dict(base, name=case + "_sdpa", impl="sdpa")
            So = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2)
            if a.check:
                try:
                    Rs = F.scaled_dot_product_attention(qt.float(), kt.float(), vt.float(), is_causal=causal).transpose(1, 2)
                except Exception as e:  # noqa: BLE001
                    r["sdpa32_err"] = str(e)[-120:]
                if R is not None:
                    r["rel_vs_fp64"] = round(((So.double() - R).norm() / R.norm()).item(), 7)
                if Rs is not None:
                    r["rel_vs_sdpa32"] = round(((So.float() - Rs).norm() / Rs.norm()).item(), 7)
            ms = bench(lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal), warmup=5, rep=max(10, rep // 2))
            r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            print(json.dumps(r), flush=True)
        else:
            So = None
        if a.ck:
            import flash_attn
            from flash_attn import flash_attn_func
            r = dict(base, name=case + "_ck", impl="ck", flash_attn=flash_attn.__version__)
            O = flash_attn_func(Q, K, V, causal=causal)
            if a.check:
                if R is not None:
                    r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
                if So is not None:
                    r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                if Rs is not None:
                    r["rel_vs_sdpa32"] = round(((O.float() - Rs).norm() / Rs.norm()).item(), 7)
            del O
            ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=rep)
            r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            print(json.dumps(r), flush=True)
        cs = [c for c in sel if c["case"] == case]
        rr = rot % max(1, len(cs))
        for c in cs[rr:] + cs[:rr]:
            r = dict(base, name=c["name"], impl="tl", P=c["P"], bN=c["bN"], th=c["th"], mode=c["mode"], form=c["form"], vp=c["vp"])
            t0 = time.time()
            patch.CTX.update(swap=c.get("swap", False), rev=c.get("rev", False), rows=c.get("rows", 0), bN=c["bN"], D=D, th=c["th"], L=L, Hkv=Hkv, P=c["P"], G=G, B=B)
            try:
                k = build(c)
                O = k(Q, K, V)
                if a.check and c["mode"] == "full" and "S" not in FL:
                    if R is not None:
                        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
                    if So is not None:
                        r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7)
                    if Rs is not None:
                        r["rel_vs_sdpa32"] = round(((O.float() - Rs).norm() / Rs.norm()).item(), 7)
                    same = 0
                    nrep = 30 if L <= 8192 else 10
                    for _ in range(nrep):
                        O2 = k(Q, K, V)
                        same += int(torch.equal(O2, O))
                        del O2
                    r["repeat_equal"] = f"{same}/{nrep}"
                del O
                if "S" in FL:
                    # stamps per wave in a copy of Q: (r0, r1, c0, c1); realtime 25 MHz, s_memtime follows the SIMD clock (E1)
                    import statistics
                    ts, last = [], None
                    for it in range(8):
                        Qc = Q.clone()
                        torch.cuda.synchronize()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record(); O = k(Qc, K, V); e1.record(); torch.cuda.synchronize()
                        if it >= 3:
                            ts.append(e0.elapsed_time(e1))
                        del O
                        last = Qc
                    ms = statistics.median(ts)
                    r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
                    w = c["th"] // 64
                    rows_ = c["rows"] or c["P"] * G
                    NB = -(-L * G // rows_)
                    rec = []
                    for bk in range(NB):
                        for hk in range(Hkv):
                            for wv in range(w):
                                fr = bk * rows_ + wv
                                if fr // G < L:
                                    rec.append((bk, hk, wv, fr // G, hk * G + fr % G))
                    x = last[0, [t[3] for t in rec], [t[4] for t in rec], :16].detach().cpu().numpy().tobytes()
                    t = torch.frombuffer(bytearray(x), dtype=torch.int64).view(-1, 4).double()
                    bid = torch.tensor([t2[0] * Hkv + t2[1] for t2 in rec])
                    dr, dc = t[:, 1] - t[:, 0], t[:, 3] - t[:, 2]
                    span = (t[:, 1].max() - t[:, 0].min()).item() / 25e3  # ms
                    bl = torch.zeros(NB * Hkv, dtype=torch.float64).scatter_reduce(0, bid, t[:, 1], "amax") - t[:, 0].new_full((NB * Hkv,), float("inf")).scatter_reduce(0, bid, t[:, 0], "amin")
                    r.update(st_span_ms=round(span, 4), st_mhz=round((dc.sum() / dr.sum()).item() * 25, 1), st_wave_ms_sum=round(dr.sum().item() / 25e3, 3),
                             st_block_ms_sum=round(bl.sum().item() / 25e3, 3), st_cu_busy_110=round(bl.sum().item() / 25e3 / span / 110, 4),
                             st_mcyc_block_sum=round(bl.sum().item() / 25 * (dc.sum() / dr.sum()).item() * 25 / 1e6 / 1e6 * 1e6, 3))
                else:
                    ms = bench(lambda: k(Q, K, V), warmup=10, rep=rep)
                    r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
            except Exception as e:  # noqa: BLE001
                r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
            r["wall_s"] = round(time.time() - t0, 1)
            print(json.dumps(r), flush=True)
        del Q, K, V, R, Rs, So, qt, kt, vt
        torch.cuda.empty_cache()
if a.gemm:
    x, y = torch.randn(8192, 8192, device="cuda", dtype=torch.float16), torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    ms = bench(lambda: x @ y.T, warmup=3, rep=20)
    print(json.dumps(dict(name="gemm8192", tag=a.tag, stack=STACK, ms=round(ms, 4), tflops=round(2 * 8192**3 / ms * 1e-9, 2))), flush=True)
