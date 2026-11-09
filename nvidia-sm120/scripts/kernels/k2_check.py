# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Correctness check for one K2 decode candidate against the exact reference (contract: decode_contract.py).
#   k2_check.py <module>            e.g. k2_check.py tilelang.contrib.sm120_nvfp4.decode_gemv
# Checks a real checkpoint layer plus all 5 synthetic shapes, M in {1, 3, 8, 16, 33, 64} (capped at SUPPORTS_M),
# determinism (two calls bit-identical) and CUDA-graph capturability. Prints one JSON line per case; exit 1 on fail.
import importlib, json, sys, torch
from safetensors import safe_open
from tilelang.contrib.sm120_nvfp4.decode_contract import SHAPES, dequantize_reference, synthetic_layer

mod = importlib.import_module(sys.argv[1])
SUP = getattr(mod, "SUPPORTS_M", 64)
TOL = 1.5 * 0.00166
ok_all = True


def check(tag, packed, scale, alpha, Ms):
    global ok_all
    N, K = packed.shape[0], packed.shape[1] * 2
    W = dequantize_reference(packed, scale, alpha)
    prep = mod.prepare(packed, scale, alpha)
    for M in Ms:
        if M > SUP:
            continue
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ W.T
        try:
            y = mod.linear(x, prep)
            y2 = mod.linear(x, prep)
            torch.cuda.synchronize()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            det = torch.equal(y, y2)
            shape_ok = tuple(y.shape) == (M, N) and y.dtype == torch.bfloat16
            passed = rel <= TOL and det and shape_ok
        except Exception as ex:  # noqa: BLE001
            rel, det, shape_ok, passed = None, None, None, False
            print(json.dumps({"case": tag, "M": M, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}), flush=True)
        ok_all &= passed
        print(json.dumps({"case": tag, "N": N, "K": K, "M": M, "rel_err": rel, "deterministic": det, "shape_ok": shape_ok, "pass": passed}), flush=True)
    # CUDA-graph capture at M=8 (or SUP)
    M = min(8, SUP)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    try:
        mod.linear(x, prep); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = mod.linear(x, prep)
        x.copy_(torch.randn(M, K, device="cuda", dtype=torch.bfloat16)); g.replay(); torch.cuda.synchronize()
        rel = ((y.float() - x.float() @ W.T).norm() / (x.float() @ W.T).norm()).item()
        cap = rel <= TOL
    except Exception as ex:  # noqa: BLE001
        cap = False; rel = f"{type(ex).__name__}: {str(ex)[:200]}"
    ok_all &= cap
    print(json.dumps({"case": tag, "cuda_graph_replay_ok": cap, "rel_err_after_replay": rel}), flush=True)


with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    pre = "model.language_model.layers.3.mlp.down_proj."
    wp = f.get_tensor(pre + "weight_packed").view(torch.int8); ws = f.get_tensor(pre + "weight_scale")
    al = (1.0 / f.get_tensor(pre + "weight_global_scale").float()).expand(wp.shape[0]).contiguous()
check("real:layers.3.mlp.down_proj", wp, ws, al, (1, 3, 8, 16, 33, 64))
for name, (N, K) in SHAPES.items():
    check(f"synthetic:{name}", *synthetic_layer(N, K, seed=1), (1, 16, 64))
print(json.dumps({"module": sys.argv[1], "ALL_PASS": ok_all}))
sys.exit(0 if ok_all else 1)
