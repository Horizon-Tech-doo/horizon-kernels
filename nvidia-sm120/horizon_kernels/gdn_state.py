# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels GDN state (29-09-2026, default off): the GDN recurrent-state dtype and our 16-bit-capable decode core.
#
# HORIZON_GDN_STATE=bf16|fp16 (unset or fp32 = unchanged): the recurrent (SSM) state of every Qwen3.5/3.8 GDN layer is
#   stored as bfloat16 / float16 instead of the model config's float32 (mamba_ssm_dtype). Halves the state bytes read
#   and written per decode step (48 layers x 48 heads x 128 x 128 per request and slot) and halves the per-request GDN
#   cache memory (more concurrent requests in the same KV budget). Arithmetic stays fp32 inside every kernel; only the
#   carried state is rounded (once per decode step; once per prefill chunk). NUMERICS CHANGE: gated (long-context
#   prose/code NLL at 16k/32k, needles to 250k, long-generation evals), see docs/gdn-state-design.md.
#   * bf16: vLLM supports it end to end (its fused CUDA decode kernel has a bf16 instantiation, FLA prefill and the
#     mixed-batch Triton path cast). Equivalent to vLLM's --mamba-ssm-cache-dtype bfloat16, set from the plugin
#     because the service passes fixed CLI flags.
#   * fp16: 3 more mantissa bits than bf16 (CPU study: ~8x smaller output error at the same bytes); vLLM's fused CUDA
#     kernel has no fp16 instantiation, so fp16 REQUIRES HORIZON_GDN2=1 (our kernel) and adds float16 to vLLM's list
#     of fused-decode state dtypes. Stores saturate at +-65504 (never inf).
# HORIZON_GDN2=1: the speculative GDN decode core (vLLM's torch.ops._C.fused_gdn_decode_post_conv_mtp) runs on
#   tilelang.contrib.horizon_ops.gdn_state (G1S): G1's arithmetic and row split, state fp32 / bf16 / fp16. fp32 and bf16
#   are bit-identical to vLLM's kernel (standalone GPU gate pending); calls it does not cover fall back to vLLM's op
#   (fp32 / bf16) or raise (fp16: vLLM has no kernel for it; a silent fallback would corrupt the state).
#
# Activation markers: logged at install AND at the first warm-up. The warm-up line is preceded by a newline on stderr:
#   it is printed during CUDA-graph capture, right after tqdm's carriage-return progress bar, and the service log stores such
#   an entry as binary ("[N blob data]"), which a plain log grep never shows (why the 27-09 G1 marker looked
#   missing). Always search logs with a mode that prints binary entries.
# Not compatible with HORIZON_GDN_RECOVER=1 (G2 keeps an fp32 checkpoint of its own): install() refuses the pair.
import os
import sys

import torch

_DTYPES = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16}
_WARM: set = set()
_WARM_ALWAYS_OK = False          # tests: the fake kernel needs no compile
_ORIG = {}


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _announce(msg, *args):
    try:                                  # end any pending tqdm line so the service log stores the marker as clean text
        sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    _log().info(msg, *args)


def requested_dtype():
    v = os.environ.get("HORIZON_GDN_STATE", "").strip().lower()
    if v in ("", "0", "fp32", "float32", "auto"):
        return None
    if v not in _DTYPES:
        raise ValueError(f"HORIZON_GDN_STATE={v!r}: expected fp32, bf16 or fp16")
    return _DTYPES[v]


def _install_state_dtype(dt):
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C
    if getattr(C.gated_delta_net_state_dtype, "_horizon", False):
        return
    orig = C.gated_delta_net_state_dtype

    def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype="auto"):
        conv, _ = orig(model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype)
        return (conv, dt)

    gated_delta_net_state_dtype._horizon = True
    C.gated_delta_net_state_dtype = classmethod(gated_delta_net_state_dtype)
    if dt == torch.float16:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
        if torch.float16 not in Q.FUSED_GDN_STATE_DTYPES:
            Q.FUSED_GDN_STATE_DTYPES = (*Q.FUSED_GDN_STATE_DTYPES, torch.float16)
    _announce("horizon_kernels GDN state: recurrent state stored as %s (HORIZON_GDN_STATE)", str(dt).split(".")[-1])


def _warm(gs, mixed_qkv, a, b, dt_bias, state, output_gate, norm_weight, act):
    """compile + load every row split for this stride / dtype set, outside CUDA-graph capture (G1's pattern)"""
    if _WARM_ALWAYS_OK:
        return True
    key = (mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0), state.stride(0), state.dtype,
           dt_bias.dtype, norm_weight.dtype, act, mixed_qkv.device)
    if key in _WARM:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    HV = state.shape[1]
    dev = mixed_qkv.device
    sidx = torch.zeros(1, 4, dtype=torch.int32, device=dev)      # zero path: no state read or write
    cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    nacc = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.empty(1, HV, 128, dtype=torch.bfloat16, device=dev)
    for vr in (32, 64, 128):
        gs.fused_gdn_decode_post_conv_mtp(mixed_qkv[:1], a[:1], b[:1], torch.zeros(HV, device=dev), dt_bias, sidx, cu,
                                          nacc, state, output_gate[:1], norm_weight, out, 0.1, 1e-6, act, vr=vr)
    torch.cuda.synchronize()
    _WARM.add(key)
    _announce("horizon_kernels GDN2: GDN decode core on horizon_ops.gdn_state (state %s, HV=%d, strides qkv=%d ab=%d "
              "z=%d state=%d)", str(state.dtype).split(".")[-1], HV, mixed_qkv.stride(0), a.stride(0),
              output_gate.stride(0), state.stride(0))
    return True


def _install_op():
    from vllm import _custom_ops as ops
    if getattr(ops.fused_gdn_decode_post_conv_mtp, "_horizon_gdn2", False):
        return
    try:                                  # never stack on top of G1's wrapper: fall back to vLLM's own op
        from horizon_kernels import ops as hops
        orig = hops._ORIG.get("gdn", ops.fused_gdn_decode_post_conv_mtp)
    except Exception:  # noqa: BLE001
        orig = ops.fused_gdn_decode_post_conv_mtp
    _ORIG["gdn"] = orig

    def fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
                                       state, output_gate, norm_weight, out=None, scale=128 ** -0.5, norm_eps=1e-5,
                                       output_gate_activation="silu"):
        from tilelang.contrib.horizon_ops import gdn_state as gs
        ok = (gs.supported(mixed_qkv, state, output_gate, a, b) and A_log.dtype == torch.float32
              and state_indices.shape[1] <= gs.T_MAX and output_gate_activation in ("silu", "sigmoid")
              and dt_bias.dtype in (torch.float32, torch.bfloat16) and norm_weight.dtype in (torch.float32, torch.bfloat16)
              and (out is None or out.is_contiguous()))
        if ok:
            ok = _warm(gs, mixed_qkv, a, b, dt_bias, state, output_gate, norm_weight, output_gate_activation)
        if not ok:
            if state.dtype == torch.float16:
                raise RuntimeError("horizon_kernels GDN2: float16 GDN state but this decode call is not covered by "
                                   "horizon_ops.gdn_state, and vLLM has no float16 kernel")
            _log().warning_once("horizon_kernels GDN2: GDN decode call not covered, using vLLM's kernel")
            return orig(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
                        output_gate, norm_weight, out, scale, norm_eps, output_gate_activation)
        return gs.fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                                                 num_accepted_tokens, state, output_gate, norm_weight, out, scale,
                                                 norm_eps, output_gate_activation)

    fused_gdn_decode_post_conv_mtp._horizon = True        # G1's install() sees it and does not wrap again
    fused_gdn_decode_post_conv_mtp._horizon_gdn2 = True
    ops.fused_gdn_decode_post_conv_mtp = fused_gdn_decode_post_conv_mtp
    _announce("horizon_kernels GDN2: decode-core hook installed (horizon_ops.gdn_state; marker at first warm-up)")


def install():
    dt = requested_dtype()
    gdn2 = os.environ.get("HORIZON_GDN2", "0") == "1"
    if dt is None and not gdn2:
        return
    if os.environ.get("HORIZON_GDN_RECOVER", "0") == "1" and dt is not None:
        raise ValueError("HORIZON_GDN_STATE with HORIZON_GDN_RECOVER=1: G2 keeps an fp32 checkpoint; pick one")
    if dt == torch.float16 and not gdn2:
        raise ValueError("HORIZON_GDN_STATE=fp16 needs HORIZON_GDN2=1 (vLLM has no fp16 GDN decode kernel)")
    if gdn2:
        _install_op()
    if dt is not None:
        _install_state_dtype(dt)
