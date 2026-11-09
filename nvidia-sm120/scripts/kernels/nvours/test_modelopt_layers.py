# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Standalone verification (27-09-2026) of horizon_kernels' ModelOpt adapter (HORIZON_MODELOPT=1) on REAL layers of
# nvidia/Qwen3.8-27B-NVFP4: every distinct NVFP4 shape (fused mlp.gate_up 34816x5120 with two shard scales, mlp.down
# 5120x17408, lm_head 248320x5120), through vLLM's own ModelOptNvFp4LinearMethod (create_weights ->
# process_weights_after_loading -> apply) with the plugin installed, against
#   ref    exact fp32 dequantisation (decode_contract.dequantize_reference, alpha = weight_scale_2),
#   marlin vLLM's stock path on the same weights (MarlinNvFp4LinearKernel, plugin hooks bypassed),
# plus the same statistics on our own compressed-tensors checkpoint (the tolerance we accepted there), bit-identity of
# the adapter vs direct kernel calls, determinism, CUDA-graph thread_local capture/replay, and the weak layer registry.
#   $HK_WORK/gpu_lock.sh nvours-test $HK_WORK/nvours/run_test.sh      (one JSON line per case; ~3 min)
import gc
import json
import os
import sys
import time

import torch
from safetensors import safe_open

T0 = time.time()
NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
OURS = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
TOL = 1.5 * 0.00166          # exact class (decode_contract / k2_check_xl): regfrag vs the fp32 reference
FP4_TOL = 0.02               # fp4x2 calibration gate (rel_tol of the variant)
MS = (1, 2, 3, 4, 7, 8, 16, 17, 32, 33, 48, 64, 65, 96, 128, 129, 200, 256, 257, 512, 513, 1000, 1024)
MS_PREFILL = (1025, 2048, 4096)
GRAPH_MS = (4, 32, 128, 512)
ok_all = True


def out(d):
    print(json.dumps(d), flush=True)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def row_rel(a, b):
    return ((a.float() - b.float()).norm(dim=1) / b.float().norm(dim=1).clamp_min(1e-30)).max().item()


# ------------------------------------------------------------------------------------------------ plugin + vLLM
import vllm.model_executor.parameter as _vparam  # noqa: E402

# one GPU, no torch.distributed group: vLLM's parameter classes only need the TP rank / world size
_vparam.get_tensor_model_parallel_rank = lambda: 0
_vparam.get_tensor_model_parallel_world_size = lambda: 1
from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel as MK  # noqa: E402
from vllm.model_executor.layers.quantization import modelopt as mo  # noqa: E402

STOCK = {"marlin_pw": MK.process_weights_after_loading, "marlin_apply": MK.apply_weights,
         "mo_pw": mo.ModelOptNvFp4LinearMethod.process_weights_after_loading}
from horizon_kernels import vllm_plugin as vp  # noqa: E402

vp._install()
HOOKED = {"marlin_pw": MK.process_weights_after_loading, "marlin_apply": MK.apply_weights}
assert HOOKED["marlin_pw"] is not STOCK["marlin_pw"] and mo.ModelOptNvFp4LinearMethod.process_weights_after_loading \
    is not STOCK["mo_pw"], "plugin hooks not installed (HORIZON_NVFP4_ADAPTIVE / HORIZON_MODELOPT?)"
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import prefill_rs  # noqa: E402
from tilelang.contrib.sm120_nvfp4.decode_contract import dequantize_reference  # noqa: E402
import tilelang.contrib.sm120_nvfp4 as nv  # noqa: E402

CFG = mo.ModelOptNvFp4Config(quant_method="NVFP4", is_checkpoint_nvfp4_serialized=True, kv_cache_quant_algo=None,
                             exclude_modules=[], group_size=16)


def nv_tensors(prefixes):
    idx = json.load(open(f"{NV}/model.safetensors.index.json"))["weight_map"]
    res = []
    for p in prefixes:
        d = {}
        for s in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            with safe_open(f"{NV}/{idx[p + '.' + s]}", "pt", device="cuda") as f:
                d[s] = f.get_tensor(p + "." + s)
        res.append(d)
    return res


def build_layer(shards, stock):
    """A vLLM ModelOpt NVFP4 linear layer loaded with the checkpoint shards, processed stock or through the plugin."""
    method = mo.ModelOptNvFp4LinearMethod(CFG)
    layer = torch.nn.Module()
    sizes = [s["weight"].shape[0] for s in shards]
    k = shards[0]["weight"].shape[1] * 2
    with torch.device("cuda"):
        method.create_weights(layer, k, sizes, k, sum(sizes), torch.bfloat16, weight_loader=lambda *a, **kw: None)
    layer.params_dtype = torch.bfloat16
    layer.weight.data.copy_(torch.cat([s["weight"] for s in shards]))
    layer.weight_scale.data.copy_(torch.cat([s["weight_scale"] for s in shards]))
    layer.weight_scale_2.data.copy_(torch.stack([s["weight_scale_2"].float().reshape(()) for s in shards]))
    layer.input_scale.data.copy_(torch.stack([s["input_scale"].float().reshape(()) for s in shards]))
    if stock:
        MK.process_weights_after_loading, MK.apply_weights = STOCK["marlin_pw"], STOCK["marlin_apply"]
        try:
            STOCK["mo_pw"](method, layer)
        finally:
            MK.process_weights_after_loading, MK.apply_weights = HOOKED["marlin_pw"], HOOKED["marlin_apply"]
        fwd = lambda x: STOCK["marlin_apply"](method.kernel, layer, x)  # noqa: E731
    else:
        method.process_weights_after_loading(layer)
        fwd = lambda x: method.apply(layer, x)  # noqa: E731
    return layer, fwd


def ref_matmul(x, packed, scale, alpha, chunk=16384):
    """x @ W^T in fp32 with W dequantised exactly, in row chunks (the lm_head's full fp32 W would be 5 GB)."""
    outs = []
    for n0 in range(0, packed.shape[0], chunk):
        W = dequantize_reference(packed[n0:n0 + chunk].view(torch.int8), scale[n0:n0 + chunk], alpha[n0:n0 + chunk])
        outs.append(x.float() @ W.T)
        del W
    return torch.cat(outs, 1)


def check_layer(tag, prefixes, ms, exact_only=False):
    global ok_all
    shards = nv_tensors(prefixes)
    packed = torch.cat([s["weight"] for s in shards]); scale = torch.cat([s["weight_scale"] for s in shards])
    ws2 = [s["weight_scale_2"].float().reshape(()) for s in shards]
    alpha = torch.cat([a.expand(s["weight"].shape[0]) for a, s in zip(ws2, shards)]).contiguous()
    N, K = packed.shape[0], packed.shape[1] * 2
    m0 = torch.cuda.memory_allocated()
    layer, ours = build_layer(shards, stock=False)
    torch.cuda.synchronize(); m1 = torch.cuda.memory_allocated()
    slayer, marlin = build_layer(shards, stock=True)
    torch.cuda.synchronize(); m2 = torch.cuda.memory_allocated()
    reg = vp._LAYERS[layer.horizon_idx]() is layer
    out({"case": tag, "N": N, "K": K, "shards": len(shards), "weight_scale_2": [float(a) for a in ws2],
         "shards_tied": len(set(float(a) for a in ws2)) == 1, "registered": reg, "has_marlin_copy": layer.horizon_has_marlin,
         "checkpoint_weight_freed": layer.weight.numel() == 0,
         "alpha_equals_ws2": bool(torch.equal(layer.horizon_nvfp4.alpha, alpha)),
         "mem_ours_MiB": round((m1 - m0) / 2**20), "mem_stock_marlin_MiB": round((m2 - m1) / 2**20)})
    ok_all &= reg and not layer.horizon_has_marlin and bool(torch.equal(layer.horizon_nvfp4.alpha, alpha))
    if exact_only:   # what the plugin sets for a real ParallelLMHead (HORIZON_LMHEAD_FP4=0): exact class only
        layer.horizon_exact_only = True
    op = vp._op()

    def pick(M):
        v = op.select((N, K), M)
        return op.default if layer.horizon_exact_only and "exact" not in v.tags else v.name

    direct_rf = xl.prepare(packed.view(torch.int8), scale, alpha)
    direct_nat = prefill_rs.prepare_weight_native(packed, scale, alpha=alpha) if N % 128 == 0 else None
    for M in ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        v = pick(M)
        y = ours(x); y2 = ours(x); ym = marlin(x)
        ref = ref_matmul(x, packed, scale, alpha)
        torch.cuda.synchronize()
        if v == "regfrag":
            d = xl.linear(x, direct_rf) if M <= xl.SUPPORTS_M else \
                torch.cat([xl.linear(x[i:i + 1024], direct_rf) for i in range(0, M, 1024)])
        else:
            d = prefill_rs.linear_native(x, direct_nat, passes=2 if v == "fp4x2" else 1)
        r, rr, rm = rel(y, ref), row_rel(y, ref), rel(ym, ref)
        tol = TOL if v == "regfrag" else FP4_TOL
        passed = (r <= tol and (v != "regfrag" or rr <= 4 * TOL) and torch.equal(y, y2) and torch.equal(y, d)
                  and bool(torch.isfinite(y).all()) and tuple(y.shape) == (M, N))
        ok_all &= passed
        out({"case": tag, "M": M, "variant": v, "rel_ours_vs_ref": round(r, 6), "max_row_rel": round(rr, 6),
             "rel_marlin_vs_ref": round(rm, 6), "rel_ours_vs_marlin": round(rel(y, ym), 6),
             "deterministic": torch.equal(y, y2), "bit_identical_to_direct_kernel": torch.equal(y, d), "pass": passed})
        del x, y, y2, ym, ref, d
    if True:           # CUDA-graph capture (thread_local, as vLLM runs with HORIZON_CAPTURE_MODE) + replay
        for M in GRAPH_MS:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            v = pick(M)
            ours(x); torch.cuda.synchronize()
            cap = False; rels = []
            for attempt in range(3):   # gpushare can break a capture at random: retry
                try:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, capture_error_mode="thread_local"):
                        yg = ours(x)
                    cap = True
                    break
                except RuntimeError as ex:  # noqa: PERF203
                    torch.cuda.synchronize(); rels.append(f"capture attempt {attempt}: {str(ex)[:120]}")
            good = cap
            if cap:
                for _ in range(2):
                    x.copy_(torch.randn(M, K, device="cuda", dtype=torch.bfloat16)); g.replay(); torch.cuda.synchronize()
                    e = ours(x); torch.cuda.synchronize()
                    rr = rel(yg, ref_matmul(x, packed, scale, alpha))
                    rels.append(round(rr, 6))
                    good &= torch.equal(yg, e) and rr <= (TOL if v == "regfrag" else FP4_TOL)
            ok_all &= good
            out({"case": tag, "M": M, "variant": v, "cuda_graph_thread_local_replay_ok": good, "rel_after_replay": rels})
    idx = layer.horizon_idx
    del layer, slayer, ours, marlin, direct_rf, direct_nat, shards, packed, scale
    gc.collect(); torch.cuda.synchronize()
    freed = vp._LAYERS[idx]() is None
    ok_all &= freed
    out({"case": tag, "registry_entry_released_after_del": freed, "mem_after_del_MiB": round((torch.cuda.memory_allocated() - m0) / 2**20)})
    torch.cuda.empty_cache()


def ours_checkpoint_reference(tag, prefixes, n_ms=(1, 4, 32, 128, 512, 1024)):
    """Same statistics on our own compressed-tensors checkpoint (the tolerance production accepted); fused shards are
    concatenated with vLLM's stock collapse (alpha = 1 / max(gs)), as production runs (HORIZON_EXACT_SHARD_SCALES=0)."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (apply_fp4_marlin_linear,
                                                                               prepare_fp4_layer_for_marlin)
    import types
    with safe_open(OURS, "pt", device="cuda") as f:
        wp = torch.cat([f.get_tensor(p + ".weight_packed") for p in prefixes])
        ws = torch.cat([f.get_tensor(p + ".weight_scale") for p in prefixes])
        gs = torch.cat([f.get_tensor(p + ".weight_global_scale").float().reshape(-1) for p in prefixes])
    N, K = wp.shape[0], wp.shape[1] * 2
    alpha = (1.0 / gs.max()).expand(N).contiguous()
    prep = xl.prepare(wp.view(torch.int8), ws, alpha)
    l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16,
                              weight=wp.clone(), weight_scale=ws.clone(), weight_global_scale=(1.0 / gs.max()).clone())
    prepare_fp4_layer_for_marlin(l)
    for M in n_ms:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        ref = ref_matmul(x, wp, ws, alpha)
        y = xl.linear(x, prep)
        ym = apply_fp4_marlin_linear(x, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K)
        out({"case": tag, "M": M, "rel_regfrag_vs_ref": round(rel(y, ref), 6), "max_row_rel": round(row_rel(y, ref), 6),
             "rel_marlin_vs_ref": round(rel(ym, ref), 6), "rel_regfrag_vs_marlin": round(rel(y, ym), 6)})
    del prep, l
    torch.cuda.empty_cache()


out({"env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("HORIZON_", "TILELANG_"))},
     "table": str(vp._op().table_path), "table_has_lm_head": vp._op().has_key((248320, 5120))})
L = "model.language_model.layers"
if os.environ.get("NVOURS_LMHEAD_ONLY") == "1":
    check_layer("nvidia:lm_head(exact-only)", ["lm_head"], MS, exact_only=True)
    out({"ALL_PASS": ok_all, "secs": round(time.time() - T0)})
    sys.exit(0 if ok_all else 1)
for li in (0, 31, 63):
    check_layer(f"nvidia:layers.{li}.mlp.gate_up", [f"{L}.{li}.mlp.gate_proj", f"{L}.{li}.mlp.up_proj"], MS + MS_PREFILL)
    check_layer(f"nvidia:layers.{li}.mlp.down", [f"{L}.{li}.mlp.down_proj"], MS + MS_PREFILL)
check_layer("nvidia:lm_head", ["lm_head"], MS)
check_layer("nvidia:lm_head(exact-only)", ["lm_head"], MS, exact_only=True)
ours_checkpoint_reference("ours:layers.3.mlp.gate_up(CT)", [f"{L}.3.mlp.gate_proj", f"{L}.3.mlp.up_proj"])
ours_checkpoint_reference("ours:layers.3.mlp.down(CT)", [f"{L}.3.mlp.down_proj"])
out({"ALL_PASS": ok_all, "secs": round(time.time() - T0)})
sys.exit(0 if ok_all else 1)
