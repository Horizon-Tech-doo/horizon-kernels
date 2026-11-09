# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""analyze_real_kv.py DUMP_DIR [PATH_TO_horizon_ops] (29-09-2026, CPU only)
Real K / V / Q of Qwen3.8-27B full-attention layers 3, 31, 63 (bf16, first long prefill, dumped by horizon_kvfp4 with
HORIZON_KVFP4_DUMP) -> per layer: round-trip relative L2 of K and V for FP8 e4m3 (scale 1, today), NVFP4 plain,
NVFP4 + scale search, NVFP4 + search + K Hadamard; and the error of the causal attention output of the last 256
query tokens (all 24 heads, GQA 6) over the whole dumped context, each format vs bf16 K / V. Decides the Hadamard
default on real data (the synthetic tests cannot)."""
import importlib.util
import json
import os
import sys

import torch

D = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    "$HK_SRC/wt-kvfp4/tilelang/contrib/horizon_ops")
spec = importlib.util.spec_from_file_location("kv_fp4_ref", os.path.join(PKG, "kv_fp4_ref.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
torch.set_num_threads(8)
sg = R.hadamard_signs(256)


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def fp8(x):
    return x.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()


def nv(x, search=True, rot=None):
    y = R.rotate(x, rot) if rot is not None else x.float()
    d, s = R.quantize_rows(y, 1.0, search)
    deq = R.dequantize_rows(d, s)
    if rot is not None:          # back to the original space (orthogonal rotation)
        deq = R.fwht(deq / R.fwht_scale(256) / 256) * rot
    return deq


def attn(q, k, v, nq=256):
    """causal attention of the last nq query tokens; q (T, 24, 256), k / v (T, 4, 256) -> (nq, 24, 256) fp64"""
    T = k.shape[0]
    qq = q[T - nq:].double().view(nq, 4, 6, 256)
    s = torch.einsum("thgd,lhd->thgl", qq, k.double()) / 16.0
    pos = torch.arange(T)
    lim = (T - nq + torch.arange(nq)).view(nq, 1, 1, 1)
    s = s.masked_fill(pos.view(1, 1, 1, T) > lim, float("-inf"))
    return torch.einsum("thgl,lhd->thgd", torch.softmax(s, -1), v.double()).reshape(nq, 24, 256)


out = {}
for layer in (3, 31, 63):
    f = {n: os.path.join(D, f"layer{layer}_{n}.pt") for n in "kvq"}
    if not all(os.path.exists(p) for p in f.values()):
        continue
    k, v, q = (torch.load(f[n]) for n in "kvq")
    T = min(k.shape[0], q.shape[0], 8192)
    k, v, q = k[:T].view(T, 4, 256), v[:T].view(T, 4, 256), q[:T].view(T, 24, 256)
    ref = attn(q, k, v)
    fm = {"fp8_e4m3 (today)": (fp8(k), fp8(v)), "nvfp4 plain": (nv(k, False), nv(v, False)),
          "nvfp4 + search": (nv(k), nv(v)), "nvfp4 + search + K Hadamard": (nv(k, rot=sg), nv(v))}
    res = {"tokens": T, "k_absmax": round(k.float().abs().max().item(), 2), "v_absmax": round(v.float().abs().max().item(), 2)}
    for name, (kq, vq) in fm.items():
        res[name] = {"k_rel_l2": round(rel(kq, k), 5), "v_rel_l2": round(rel(vq, v), 5),
                     "attn_rel_l2": round(rel(attn(q, kq, vq), ref), 5)}
    out[f"layer{layer}"] = res
    print(json.dumps({f"layer{layer}": res}), flush=True)
json.dump(out, open(os.path.join(D, "analysis.json"), "w"), indent=1)
