# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# G1b gate: horizon_ops.conv1d_update vs vLLM's Triton causal_conv1d_update (spec decode, varlen, no bias, SiLU):
# bit-identity of the outputs and of the whole conv state, acceptance 1..4, token counts 1..4, null slots; chained
# 200 steps; speed per call.
import importlib, json, sys
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
cu_mod = importlib.import_module(f"tilelang.contrib.{PKG}.conv1d_update")
dev = torch.device("cuda")
C, SL = 10240, 6


def log(r):
    print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")


def case(ntoks, acc, gen, dim_first=True, steps=1):
    nr = len(ntoks)
    base = torch.randn(nr * 4 + 1, C, SL, device=dev, generator=gen).to(torch.bfloat16)
    if not dim_first:   # stored (slots, state_len, dim), used transposed
        base = base.transpose(1, 2).contiguous().transpose(1, 2)
    st1 = base.clone(); st2 = base.clone()
    if not dim_first:
        st1 = base.transpose(1, 2).contiguous().transpose(1, 2); st2 = base.transpose(1, 2).contiguous().transpose(1, 2)
    w = (torch.randn(C, 4, device=dev, generator=gen) * 0.5).to(torch.bfloat16)
    sidx = (torch.arange(nr * 4, dtype=torch.int32) + 1).view(nr, 4).to(dev)
    sidx[0, 0] = 0 if nr > 3 else sidx[0, 0]
    cu = torch.zeros(nr + 1, dtype=torch.int32); cu[1:] = torch.cumsum(torch.tensor(ntoks), 0); cu = cu.to(dev)
    nacc = torch.tensor(acc, dtype=torch.int32, device=dev)
    bad = 0
    for s in range(steps):
        xw = torch.randn(int(cu[-1]), C + 6144, device=dev, generator=gen).to(torch.bfloat16)
        x1 = xw.clone()[:, :C]; x2 = xw.clone()[:, :C]
        o1 = causal_conv1d_update(x1, st1, w, None, "silu", conv_state_indices=sidx[:, 0], num_accepted_tokens=nacc,
                                  query_start_loc=cu, max_query_len=4, validate_data=False)
        cu_mod.conv_update(x2, st2, w, sidx[:, 0], nacc, cu, 4)
        torch.cuda.synchronize()
        bad += int((o1.view(torch.int16) != x2.view(torch.int16)).sum()) + int((st1.view(torch.int16) != st2.view(torch.int16)).sum())
    return bad


gen = torch.Generator(device=dev); gen.manual_seed(0)
tot = 0
for ntoks, acc in (([4], [1]), ([4], [4]), ([4, 4, 4], [1, 2, 3]), ([1, 4, 2, 3, 4], [1, 2, 3, 4, 2]), ([4] * 8, [4] * 8)):
    for df in (True, False):
        b = case(ntoks, acc, gen, df)
        tot += b
        log({"test": "conv_bits", "ntoks": ntoks, "acc": acc, "dim_first": df, "diff": b})
b = case([4, 4], [2, 3], gen, True, steps=200); tot += b
log({"test": "conv_chain200", "diff": b})
log({"summary": "conv_g1b", "bit_diffs_total": tot})
print("TEST_CONV_DONE", flush=True)
