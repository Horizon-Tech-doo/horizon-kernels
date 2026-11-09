"""E3 second kernel set timing. Usage: run_tlb.py <case> <regex> <tag> [--check]."""
import json, re, sys, time
import torch
import torch.nn.functional as F
from bench_common import bench
import patch
patch.install()
from gqa_e3b import fwd
from cfgs_e import CASES, CONFIGS
import os
FL = os.environ.get('E3_PATCH', '')
case, pat, tag = sys.argv[1], re.compile(sys.argv[2]), sys.argv[3]
check = "--check" in sys.argv
s = CASES[case]
B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
torch.manual_seed(0)
Q = torch.randn(B, L, Hkv * G, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, L, Hkv, D, device="cuda", dtype=torch.float16)
flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
R = None
if check:
    q, k_, v = Q.double(), K.double().repeat_interleave(G, 2), V.double().repeat_interleave(G, 2)
    s_ = torch.einsum("bqhd,bkhd->bhqk", q, k_) / q.size(-1) ** 0.5
    if causal:
        n = s_.size(-1)
        s_ = s_.masked_fill(torch.ones(n, n, device=s_.device).tril() == 0, float("-inf"))
    R = torch.einsum("bhqk,bkhd->bqhd", F.softmax(s_, -1), v)
    del q, k_, v, s_
cfgs = [c for c in CONFIGS if c["case"] == case and pat.search(c["name"])]
rot = sum(map(ord, tag)) % max(1, len(cfgs))
cfgs = cfgs[rot:] + cfgs[:rot]
for c in cfgs:
    r = dict(name=c["name"] + ("_p" + FL if FL else ""), case=case, patch=FL, tag=tag, P=c["P"], bN=c["bN"], th=c["th"], mode=c["mode"], edge=c["edge"], rev=c["rev"], recip=c["recip"], vt=c["vt"])
    t0 = time.time()
    patch.CTX.update(bN=c["bN"], D=D, th=c["th"], L=L, Hkv=Hkv)
    try:
        k = fwd(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], 2, c["edge"], c["recip"], c["mode"], c["vt"], c["v_pack"], c["rev"])
        O = k(Q, K, V, 0)
        if check and c["mode"] == "full":
            r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 6)
            # repeat check: barrier removal could race, so 30 more calls must stay bitwise equal to the first
            same = 0
            for _ in range(30):
                O2 = k(Q, K, V, 0)
                same += int(torch.equal(O2, O))
                del O2
            r["repeat_equal_of_30"] = same
        del O
        ms = bench(lambda: k(Q, K, V, 0), warmup=10, rep=100)
        r.update(ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))
    except Exception as e:  # noqa: BLE001
        r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
    r["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(r), flush=True)
