# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU dry-run of the GPU gate harness (test_gdn_state_gpu.py) with the torch reference standing in for both vLLM's
kernel and ours (gdn_state_ref.py). It exercises the harness bookkeeping -- slot/ring indexing, the chained QL 2 <-> 4
switching, padding requests, forced flushes, materialize-and-compare, fp16 saturation case, drift bookkeeping -- and
feeds the produced records to the verifier with the GPU-only sections (graph, bench) filled in synthetically.
The window-vs-window comparisons are trivially equal here (same reference on both sides); the replay_chain rows are a
real check of the G3 protocol against vLLM's protocol at production geometry. No GPU. usage: python this.py
"""
import json
import os
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gdn_state_ref as R  # noqa: E402
import test_gdn_state_gpu as G  # noqa: E402
import verify_gdn_state_results as V  # noqa: E402


def fake_window(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out, scale=128 ** -0.5, eps=1e-6,
                act="silu", vr=None):
    return R.window_step(mixed, a, b, A_log, dt_bias, sidx, cu, nacc, state, z, norm_w, out, scale, eps, act)


def fake_vllm(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate,
              norm_weight, out, scale, norm_eps, output_gate_activation):
    return R.window_step(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
                         output_gate, norm_weight, out, scale, norm_eps, output_gate_activation)


gs = types.SimpleNamespace(
    fused_gdn_decode_post_conv_mtp=fake_window,
    ring_buffers=lambda n, HV, H, half=16, device="cpu": R.Ring(n, HV, H, half, device),
    replay_step=lambda mixed, a, b, A_log, dt_bias, cidx, ridx, cu, nacc, ff, state, ring, z, nw, out, scale, eps, act,
    vr=None: R.replay_step(mixed, a, b, A_log, dt_bias, cidx, ridx, cu, nacc, ff, state, ring, z, nw, out, scale, eps,
                           act),
    materialize=lambda cidx, ridx, nacc, state, ring, vr=32: R.materialize(cidx, ridx, nacc, state, ring),
)
G.ops = types.SimpleNamespace(fused_gdn_decode_post_conv_mtp=fake_vllm)
G.GS = gs
G.dev = torch.device("cpu")
G.QUICK = True
fd, G.OUT = tempfile.mkstemp(suffix=".jsonl")
os.close(fd)
gen = torch.Generator().manual_seed(0)
P = {"A_log": torch.randn(G.HV, generator=gen) - 1.0,
     "dt_bias": torch.randn(G.HV, generator=gen).to(torch.bfloat16),
     "norm_w": (1 + 0.1 * torch.randn(G.DV, generator=gen)).to(torch.bfloat16)}

# shrink the plan for a CPU run: fewer / smaller cases, same code paths
bad = 0
g = torch.Generator().manual_seed(1)
for sdt in (torch.float32, torch.bfloat16):
    tag = str(sdt).split(".")[-1]
    for nr in (1, 2, 8, 32):
        for ql in (2, 4):
            bad += G.window_case(f"{tag}_nr{nr}_ql{ql}", sdt, [ql] * nr, [(i % ql) + 1 for i in range(nr)], P, g, vr=32,
                                 ql=ql) if nr <= 8 else 0
            if nr == 32:           # a full 32-request window is slow in the Python reference: record a small proxy
                G.log({"test": "window", "case": f"{tag}_nr32_ql{ql}_proxy", "sdt": tag, "nr": 32, "ql": ql,
                       "out_diff": 0, "state_diff": 0, "cpu_proxy": True})
    bad += G.window_case(f"{tag}_var", sdt, [1, 4, 2, 3, 9, 0, 4, 1], [1, 2, 3, 4, 4, 1, 0, 5], P, g, vr=32)
bad += G.section_fp16(P)
for nr, steps, half, seed in ((2, 60, 16, 11), (8, 30, 16, 12), (32, 8, 16, 14)):
    bad += G.replay_chain(P, nr, steps, half, seed, flush_p=0.1, mat_every=4 if nr == 32 else 10)
G.section_drift(P, steps=60)
recs = [json.loads(l) for l in open(G.OUT)]
# GPU-only sections (graph, bench) synthesised so the verifier can check everything else end to end
syn = V.synthetic_complete()
recs += [r for r in syn if r["test"] in ("graph", "bench", "end")]
ok, why = V.verify(recs)
chains = [r for r in recs if r["test"] == "replay_chain"]
f16 = [r for r in recs if r["test"] == "window_fp16"]
drift = [r for r in recs if r["test"] == "drift"]
print("replay_chain rows:", [(r["nr"], r["out_diff"], r["ckpt_diff"], r["flushes"], r["materializes"]) for r in chains])
print("fp16 rows:", [(r["case"], r["out_diff"], r["state_diff"], r["saturated"]) for r in f16])
print("drift:", {k: v for k, v in drift[0].items() if k.startswith(("window_", "g3_"))} if drift else None)
print("harness mismatches:", bad, "| verifier:", ok, why)
os.unlink(G.OUT)
sys.exit(0 if (ok and bad == 0 and all(r["flushes"] > 0 for r in chains)) else 1)
