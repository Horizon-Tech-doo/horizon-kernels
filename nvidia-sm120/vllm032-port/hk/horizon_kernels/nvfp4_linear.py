# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Adaptive NVFP4 linear for Qwen3.8-27B on the test server RTX PRO 6000 (SM120, driver 570).
# Picks per (layer shape, token-count bucket) between:
#   marlin   W4A16 (vLLM Marlin): exact reference numerics, the default
#   regfrag  W4A16 (horizon-tilelang decode_regfrag_xl, "our own Marlin"): exact class, every M (1..1024 natively,
#            larger M in 1024-row chunks); dequantises straight into the tensor-core register fragment (~2 instructions
#            per weight). 26-09-2026 race: faster than Marlin at every M 1..32 on all 5 shapes (-3..-4% layer-weighted),
#            within 1% at 33..64, 1.5% faster on average at 96..1024 (worst cells -3..-6%).
# HORIZON_DROP_MARLIN=1: Marlin is not a variant (and the plugin never builds Marlin's weight copy: -12.8 GiB);
#            regfrag is the exact default/reference, calibrated into its own table (op name ..._nomarlin).
#   fp4x2    residual two-pass NVFP4 on FP4 tensor cores (~1% GEMM error)  -- compute-bound prefill / big batches
#   fp4x1    one-pass W4A4 (~10% GEMM error), tagged "lossy", excluded unless HORIZON_ALLOW_LOSSY=1
# HORIZON_NATIVE_PREFILL=1 (27-09-2026, default 0): fp4x2 / fp4x1 run on prefill_rs's native #3257 register-A order
#            (prefill_rs.linear_native: conv=4, epi=1, a_stages=3 for two passes) instead of K3 (kernels.py) on the
#            checkpoint layout. layer.horizon_nvfp4 is then a prefill_rs.NVFP4WeightRS built by
#            prefill_rs.prepare_weight_native (same bytes as K3's copy, which is freed). Bit-identical to K3; decode
#            (regfrag) is untouched. Calibrate into its own table dir (TILELANG_ADAPTIVE_DIR): fp4 timings change.
# The choice comes from tilelang.adaptive calibration on this GPU (scripts/kernels/calibrate_nvfp4_linear.py):
# exact variants (marlin, regfrag) replace each other on a >= 2% win; fp4x2 only replaces them on a >= 10% win at
# both edges of a token-count bucket, and only if its output stays within 2% (relative error) of Marlin's.
import os
import torch
from tilelang.adaptive import AdaptiveOp, Variant
from tilelang.contrib import sm120_nvfp4 as nv

XL_MAX_M = 1024   # decode_regfrag_xl.SUPPORTS_M


def drop_marlin() -> bool:
    return os.environ.get("HORIZON_DROP_MARLIN") == "1"


def native_prefill() -> bool:
    return os.environ.get("HORIZON_NATIVE_PREFILL", "0") == "1"


def _marlin(x, layer):
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import apply_fp4_marlin_linear
    # vLLM 0.32 (port 24-10-2026): prepare_fp4_layer_for_marlin no longer creates layer.workspace and Marlin's own
    # apply passes workspace=None (the marlin_gemm op allocates it); 0.30 layers still carry it. Pass whichever exists.
    return apply_fp4_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_global_scale,
                                   getattr(layer, "workspace", None),
                                   layer.output_size_per_partition, layer.input_size_per_partition)


def _regfrag(x, layer):
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
    if x.shape[0] <= XL_MAX_M:
        return xl.linear(x, layer.horizon_regfrag)
    # exact path for big prefill batches when Marlin is dropped (normally fp4x2 is chosen there)
    return torch.cat([xl.linear(x[i:i + XL_MAX_M], layer.horizon_regfrag) for i in range(0, x.shape[0], XL_MAX_M)])


def _fp4x2(x, layer):
    return nv.linear(x, layer.horizon_nvfp4, passes=2)


def _fp4x1(x, layer):
    return nv.linear(x, layer.horizon_nvfp4, passes=1)


def _fp4x2_native(x, layer):
    from tilelang.contrib.sm120_nvfp4 import prefill_rs
    return prefill_rs.linear_native(x, layer.horizon_nvfp4, passes=2)


def _fp4x1_native(x, layer):
    from tilelang.contrib.sm120_nvfp4 import prefill_rs
    return prefill_rs.linear_native(x, layer.horizon_nvfp4, passes=1)


def prepare_prefill_weight(weight_packed, weight_scale, gs, shard_sizes, alpha=None):
    """The FP4 prefill variants' weight handle: K3's (checkpoint layout, shares weight_packed's storage) or, with
    HORIZON_NATIVE_PREFILL=1, the native-order copy (prefill_rs.prepare_weight_native; weight_packed may be freed).

    gs: compressed-tensors global scales (divisors; alpha = 1/gs per shard). alpha (fp32 (N,), optional): the
    per-output-column multiplier given directly (ModelOpt checkpoints store weight_scale_2 = alpha, and 1/(1/a) is not
    always a bit-exact round trip in fp32); gs is then ignored."""
    if native_prefill():
        from tilelang.contrib.sm120_nvfp4 import prefill_rs
        return prefill_rs.prepare_weight_native(weight_packed, weight_scale, gs, shard_sizes=shard_sizes, alpha=alpha)
    if alpha is not None:
        import dataclasses
        one = torch.ones(1, dtype=torch.float32, device=weight_packed.device)
        w = nv.prepare_weight(weight_packed, weight_scale, one)
        return dataclasses.replace(w, alpha=alpha.to(torch.float32).contiguous())
    return nv.prepare_weight(weight_packed, weight_scale, gs, shard_sizes=shard_sizes)


def make_op(allow_lossy: bool | None = None) -> AdaptiveOp:
    if allow_lossy is None:
        allow_lossy = os.environ.get("HORIZON_ALLOW_LOSSY") == "1"
    drop = drop_marlin()
    variants = [] if drop else [Variant("marlin", _marlin, rel_tol=0.0, tags={"exact"})]
    if drop or os.environ.get("HORIZON_REGFRAG", "1") == "1":
        variants.append(Variant("regfrag", _regfrag, rel_tol=0.0 if drop else 0.0025, tags={"exact"}))
    nat = native_prefill()
    variants += [Variant("fp4x2", _fp4x2_native if nat else _fp4x2, rel_tol=0.02),
                 Variant("fp4x1", _fp4x1_native if nat else _fp4x1, tags={"lossy"}, rel_tol=0.15)]
    base = "regfrag" if drop else "marlin"
    return AdaptiveOp(
        "qwen3_8_nvfp4_linear" + ("_nomarlin" if drop else ""), variants,
        exclude_tags=() if allow_lossy else {"lossy"},
        default=base, reference=base, margin=0.10, exact_margin=0.02,
    )
