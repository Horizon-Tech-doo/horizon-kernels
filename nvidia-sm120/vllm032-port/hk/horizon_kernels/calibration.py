# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Calibration of horizon_kernels' adaptive NVFP4 linear for ANY layer shape (N, K), used by
#   - the vLLM plugin at load: a model switch brings new shapes, which are calibrated before serving
#     (HORIZON_AUTO_CALIBRATE, default 1), so a new model gets the fastest kernel per shape and batch size;
#   - scripts/kernels/calibrate_nvfp4_linear.py (offline, e.g. after a driver or GPU change).
# Timings use random weights of the real shape (speed does not depend on the values) and enough weight copies that
# memory-bound small-M timings are not flattered by the 128 MB L2; the correctness gate compares every variant with
# the reference variant on the same random data.
import types

import torch


def make_layer(N: int, K: int, seed: int, with_marlin: bool):
    """A stand-in NVFP4 layer with every prepared weight form the variants need."""
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl
    g = torch.Generator(device="cuda").manual_seed(seed)
    wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
    wgs = torch.tensor([448.0 * 6], device="cuda")
    l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16,
                              bias=None, weight=wp.clone(), weight_scale=ws.clone(),
                              weight_global_scale=(1.0 / wgs).clone(), horizon_marlin_corr=None)
    from horizon_kernels.nvfp4_linear import prepare_prefill_weight
    l.horizon_nvfp4 = prepare_prefill_weight(wp, ws, wgs, None)   # K3's handle, or the native-order copy
    l.horizon_regfrag = decode_regfrag_xl.prepare(wp.view(torch.int8), ws, l.horizon_nvfp4.alpha)
    if with_marlin:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin
        prepare_fp4_layer_for_marlin(l)
    return l


def calibrate_shape(op, N: int, K: int, buckets=None, log=print):
    """Calibrate one (N, K) into op's table (saved to disk). Returns the number of weight copies used."""
    from horizon_kernels.nvfp4_linear import drop_marlin
    copies = max(2, int(512e6 // (N * K // 2)) + 1)          # > 512 MB of weights per timing round
    layers = [make_layer(N, K, s, with_marlin=not drop_marlin()) for s in range(copies)]

    def make_args(m):
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        return [(x, l) for l in layers]

    op.calibrate(key=(N, K), make_args=make_args, ms=buckets, iters=5, log=log)
    del layers
    torch.cuda.empty_cache()
    return copies
