# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# GDN chunked prefill gate: horizon_ops.gdn_prefill vs vLLM's FLA chunk_gated_delta_rule vs an fp64 token-by-token
# reference (outputs and final state), varlen batches with initial states; speed at 8k / 16k-token prefills.
import importlib, json, math, sys
import torch
import torch.nn.functional as F
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule as fla

OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
gp = importlib.import_module(f"tilelang.contrib.{PKG}.gdn_prefill")
dev = torch.device("cuda")
H, HV, K, V = 16, 48, 128, 128
G = HV // H


def log(r):
    print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")


def inputs(lens, gen, with_h0=True):
    TT = sum(lens)
    q = F.normalize(torch.randn(1, TT, H, K, device=dev, generator=gen), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, TT, H, K, device=dev, generator=gen), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, TT, HV, V, device=dev, generator=gen).to(torch.bfloat16)
    a = torch.randn(1, TT, HV, device=dev, generator=gen)
    g = -torch.exp(torch.randn(HV, device=dev, generator=gen) * 0.5) * F.softplus(a)
    beta = torch.sigmoid(torch.randn(1, TT, HV, device=dev, generator=gen))
    h0 = (torch.randn(len(lens), HV, V, K, device=dev, generator=gen) * 0.1) if with_h0 else \
        torch.zeros(len(lens), HV, V, K, device=dev)
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32); cu[1:] = torch.cumsum(torch.tensor(lens), 0)
    return q, k, v, g.float(), beta.float(), h0, cu.to(dev)


def reference(q, k, v, g, beta, h0, cu, rows_per_seq=4):
    """fp64 recurrence; returns {(seq, t): o[t]} for sampled t and final states"""
    outs, finals = {}, []
    scale = K ** -0.5
    for n in range(cu.shape[0] - 1):
        b0, b1 = int(cu[n]), int(cu[n + 1])
        S = h0[n].double()                                 # (HV, V, K)
        pick = set(torch.linspace(0, b1 - b0 - 1, rows_per_seq).long().tolist())
        for t in range(b1 - b0):
            qt = q[0, b0 + t].double().repeat_interleave(G, 0) * scale
            kt = k[0, b0 + t].double().repeat_interleave(G, 0)
            vt = v[0, b0 + t].double()
            S = S * torch.exp(g[0, b0 + t].double())[:, None, None]
            d = (vt - torch.einsum("hvk,hk->hv", S, kt)) * beta[0, b0 + t].double()[:, None]
            S = S + torch.einsum("hv,hk->hvk", d, kt)
            if t in pick:
                outs[(n, t)] = torch.einsum("hvk,hk->hv", S, qt)
        finals.append(S)
    return outs, torch.stack(finals)


gen = torch.Generator(device=dev); gen.manual_seed(0)
for lens in ([64], [100], [300, 17, 1000], [2048]):
    q, k, v, g, beta, h0, cu = inputs(lens, gen)
    o_f, ht_f = fla(q, k, v, g, beta, initial_state=h0.clone(), output_final_state=True, cu_seqlens=cu)
    o_h, ht_h = gp.chunk_gated_delta_rule(q, k, v, g, beta, h0.clone(), cu)
    o_h2, _ = gp.chunk_gated_delta_rule(q, k, v, g, beta, h0.clone(), cu)
    torch.cuda.synchronize()
    ref, fin = reference(q, k, v, g, beta, h0, cu)
    ef = max(float((o_f[0, int(cu[n]) + t].double() - r).abs().max()) for (n, t), r in ref.items())
    eh = max(float((o_h[0, int(cu[n]) + t].double() - r).abs().max()) for (n, t), r in ref.items())
    log({"test": "gdnpf_correct", "lens": lens, "fla_max_abs": ef, "ours_max_abs": eh,
         "fla_state_rel": float((ht_f.double() - fin).norm() / fin.norm()),
         "ours_state_rel": float((ht_h.double() - fin).norm() / fin.norm()),
         "ref_absmax": max(float(r.abs().max()) for r in ref.values()), "deterministic": bool(torch.equal(o_h, o_h2))})
for lens in ([8192], [16384], [4096] * 4):
    q, k, v, g, beta, h0, cu = inputs(lens, gen)
    ts = {}
    for name, fn in (("fla", lambda: fla(q, k, v, g, beta, initial_state=h0, output_final_state=True, cu_seqlens=cu)),
                     ("ours", lambda: gp.chunk_gated_delta_rule(q, k, v, g, beta, h0, cu))):
        fn(); torch.cuda.synchronize()
        best = 1e30
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize(); best = min(best, e0.elapsed_time(e1))
        ts[name + "_ms"] = round(best, 3)
    log({"test": "gdnpf_bench", "lens": lens, **ts})
print("TEST_GDNPF_DONE", flush=True)
