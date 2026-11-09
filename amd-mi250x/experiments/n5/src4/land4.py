"""Landed module after round 4, with TF. Same process: SDPA pre, arm, SDPA post, one per case. Guard cases must raise. --compile: build only, no GPU. Ratio: ours / same-process SDPA, TF, higher is better."""
import json, os, sys, time
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
comp = len(sys.argv) > 1 and sys.argv[1] == "--compile"
if comp:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn.functional as F
import gqa_fa
from bench_common import bench
from n5lib import CASES, inputs, ref
from n5lib_r4 import extend
extend(CASES)
c18 = gqa_fa.clang_major() == 18
print(json.dumps(dict(info=1, clang=gqa_fa.clang_major(), torch=torch.__version__)), flush=True)
# guard
for name, kw, must in [("lb_P32_t768", dict(P=32, th=768, G=6, rows=0, lb=4), True), ("lb_P192_mha", dict(P=192, th=768, G=1, rows=0, lb=4), True), ("sched_rows128", dict(P=0, th=512, G=6, rows=128, lb=4, sched="gcn-max-ilp"), True), ("lb_rows128", dict(P=0, th=512, G=6, rows=128, lb=4), not c18), ("lb_mha_P128", dict(P=128, th=512, G=1, rows=0, lb=4), not c18), ("sched_mha_P160", dict(P=160, th=640, G=1, rows=0, lb=4, sched="gcn-max-ilp"), not c18)]:
    try:
        gqa_fa._check_tuning(kw["G"], kw["P"], kw["th"], kw["rows"], kw.get("lb", 0), kw.get("sched", ""))
        raised = False
    except ValueError:
        raised = True
    print(json.dumps(dict(guard=name, raised=raised, ok=raised == must)), flush=True)
rule = [(L, B, gqa_fa.mha_tile(L, B)) for L, B in [(2048, 1), (4096, 1), (4320, 1), (4608, 1), (6144, 1), (8192, 1), (16384, 1), (1024, 4), (1024, 16), (4096, 4), (8192, 2)]]
print(json.dumps(dict(rule=[(L, B, t["P"], t.get("lb", 0), t.get("sched", "")) for L, B, t in rule])), flush=True)
A = [("d128_4096", dict(P=0, bN=32, threads=512, rows=128, rev=True, swap=True, recip=True)), ("d128_4096", dict(P=32, bN=96, threads=768, rev=True, swap=True, recip=True)), ("d256_8192", dict(P=32, bN=32, threads=768, rev=True, swap=True, recip=True))]
for case, L, B in [("mha_4096", 4096, 1), ("mha_8192", 8192, 1), ("mha_16384", 16384, 1), ("mhaB4_4096", 4096, 4)]:
    A.append((case, gqa_fa.mha_tile(L, B)))
if c18:
    A.append(("d128_16384", dict(P=0, bN=32, threads=512, rows=128, rev=True, swap=True, recip=True, lb=4)))
    A.append(("mha_4096", dict(P=160, bN=32, threads=640, rev=False, swap=False, recip=True, lb=4, sched="gcn-max-ilp")))   # untimed combination: decides P160's sched
    A.append(("mha_8192", dict(P=128, bN=32, threads=512, rev=False, swap=False, recip=True, lb=4)))
    A.append(("mha_16384", dict(P=128, bN=32, threads=512, rev=False, swap=False, recip=True, lb=4)))
for rnd in range(1, 3):
    for case, t in A:
        s = CASES[case]
        B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
        t0 = time.time()
        kw = dict(t)
        k = gqa_fa.gqa_flash_fast(B, L, Hkv, G, D, causal, kw.pop("P"), kw.pop("bN"), kw.pop("threads"), 2, True, kw.pop("rev"), kw.pop("recip"), kw.pop("swap"), kw.pop("rows", 0), kw.pop("lb", 0), kw.pop("sched", ""))
        r = dict(case=case, P=t["P"], bN=t["bN"], th=t["threads"], rows=t.get("rows", 0), lb=t.get("lb", 0), sched=t.get("sched", ""), round=rnd, stack=os.environ.get("N5_STACK"))
        if comp:
            r["t"] = round(time.time() - t0, 1); print(json.dumps(r), flush=True)
            continue
        Q, K_, V = inputs("randn0", s)
        flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
        rep = 50 if B * L <= 8192 else 20
        qt, kt, vt = Q.transpose(1, 2), K_.transpose(1, 2).repeat_interleave(G, 1), V.transpose(1, 2).repeat_interleave(G, 1)
        sd = lambda: F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)  # noqa: E731
        s0 = flops / bench(sd, warmup=5, rep=max(10, rep // 2)) * 1e-9
        O = k(Q, K_, V)
        if B * L <= 8192:
            R = ref(Q, K_, V, causal, G)
            r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7); del R
        So = sd().transpose(1, 2)
        r["rel_vs_sdpa"] = round(((O.float() - So.float()).norm() / So.float().norm()).item(), 7); del So
        r["repeat_equal"] = f"{sum(int(torch.equal(k(Q, K_, V), O)) for _ in range(20))}/20"
        r["tflops"] = round(flops / bench(lambda: k(Q, K_, V), warmup=10, rep=rep) * 1e-9, 2)
        s1 = flops / bench(sd, warmup=5, rep=max(10, rep // 2)) * 1e-9
        r.update(sdpa_pre=round(s0, 2), sdpa_post=round(s1, 2), vs_sdpa=round(r["tflops"] / ((s0 + s1) / 2), 3))
        print(json.dumps(r), flush=True)
        del Q, K_, V, O, qt, kt, vt
        torch.cuda.empty_cache()
