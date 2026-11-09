# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels quantized MTP draft layer (formats track S6, 27-09-2026): HORIZON_MTP_QUANT=fp8 (default off).
#
# The MTP drafter's linear layers (mtp.fc, mtp.layers.*.self_attn.*, mtp.layers.*.mlp.*) are BF16 in our checkpoint
# (0.85 GB read per draft token, 3 drafts per decode step). With HORIZON_MTP_QUANT=fp8 each one is quantized ONCE at
# load to FP8 e4m3 per output row (amax / 448) and runs on our W8A16 kernel (tilelang.contrib.sm120_fp8.decode_w8a16:
# BF16 activations, exact e4m3 -> BF16, fp32 accumulation): 0.43 GB per draft instead of 0.85 GB.
# Output quality cannot change: drafts are only proposals, and the target model verifies every token with its own
# (unchanged) weights. Only the acceptance rate can move; that is the gate (acc_measure, +-0.02 per category).
#   HORIZON_MTP_QUANT_FREE_BF16 (default 1): free the BF16 weight (our kernel serves every M, prefill in 1,024-row
#   chunks), so memory drops by ~0.4 GB instead of growing.
# Hook: vLLM's UnquantizedLinearMethod (the checkpoint ignores ^mtp.*), for layers whose prefix is under "mtp.".
import os
import re
import weakref

import torch

_LAYERS: list = []
_MTP = re.compile(r"(^|\.)mtp\.")


def enabled() -> bool:
    return os.environ.get("HORIZON_MTP_QUANT", "") in ("fp8", "nvfp4")


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _impl(x: torch.Tensor, idx: int) -> torch.Tensor:
    layer = _LAYERS[idx]()
    p = layer.horizon_mtpq
    x2 = x.reshape(-1, p.K)
    if getattr(layer, "horizon_mtpq_kind", "fp8") == "nvfp4":
        from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
        if x2.shape[0] <= xl.SUPPORTS_M:
            y = xl.linear(x2, p)
        else:
            y = torch.cat([xl.linear(x2[i:i + xl.SUPPORTS_M], p) for i in range(0, x2.shape[0], xl.SUPPORTS_M)])
    else:
        from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
        y = w8.linear(x2, p)
    return y.reshape(*x.shape[:-1], p.N)


def _impl_direct(layer, x):
    _LAYERS.append(weakref.ref(layer))
    try:
        return _impl(x, len(_LAYERS) - 1)
    finally:
        _LAYERS.pop()


def _fake(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _LAYERS[idx]().horizon_mtpq.N))


def quantize_rows(w_bf16):
    """(N, K) bf16 -> ((N, K) e4m3, (N,) fp32 scale), per-row amax / 448, round to nearest."""
    wf = w_bf16.float()
    s = wf.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    return (wf / s).to(torch.float8_e4m3fn).contiguous(), s.reshape(-1).contiguous()


_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_nvfp4_rtn(w_bf16):
    """(N, K) bf16 -> (packed int8 (N, K/2) e2m1 codes, low nibble = even k; e4m3 block scales (N, K/16); fp32 alpha (N,))
    round-to-nearest-even with block scale amax/6 (llm-compressor style), global multiplier amax_tensor / (6 * 448)."""
    wf = w_bf16.float()
    N, K = wf.shape
    # A zero matrix otherwise produces 0/0 block scales. Keep the global
    # multiplier times the smallest E4M3 block scale (2**-9) normal FP32.
    # Clamping gs alone to tiny still makes sf subnormal: with denormal
    # flushing, zero blocks then divide by zero and acquire nonzero codes.
    min_global_scale = torch.finfo(torch.float32).tiny * (2 ** 9)
    gs = (wf.abs().max() / (6.0 * 448.0)).clamp_min(min_global_scale)
    blk = wf.reshape(N, K // 16, 16)
    s = (blk.abs().amax(-1, keepdim=True) / 6.0 / gs).clamp(min=2 ** -9, max=448.0).to(torch.float8_e4m3fn)
    sf = s.float() * gs
    a = (blk / sf).abs().clamp(max=6.0)
    pos = torch.tensor(_E2M1_POS, device=wf.device)
    mids = (pos[1:] + pos[:-1]) / 2
    idx = torch.bucketize(a, mids)
    tie = (a.unsqueeze(-1) == mids).any(-1)
    idx = idx + (tie & (idx % 2 == 1)).long()
    code = idx | ((blk < 0) & (idx > 0)).long() << 3
    code = code.reshape(N, K).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    alpha = torch.full((N,), float(gs), device=wf.device, dtype=torch.float32)
    return packed.view(torch.int8), s.reshape(N, K // 16).contiguous(), alpha


def install():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if getattr(UnquantizedLinearMethod.apply, "_horizon_mtpq", False):
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "mtp_w8_linear")):
        op = torch.library.custom_op("horizon::mtp_w8_linear", mutates_args=())(_impl)
        op.register_fake(_fake)
    orig_pw = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply
    free = os.environ.get("HORIZON_MTP_QUANT_FREE_BF16", "1") == "1"

    def pw(self, layer):
        r = orig_pw(self, layer)
        layer.horizon_mtpq_idx = None
        prefix = getattr(layer, "prefix", "") or ""
        w = getattr(layer, "weight", None)
        if not _MTP.search(prefix) or w is None or w.dim() != 2 or w.dtype != torch.bfloat16 or not w.is_cuda:
            return r
        N, K = w.shape
        if N % 128 or K % 64 or getattr(layer, "tp_size", 1) != 1:
            _log().info("horizon_kernels MTP quant: %s (%dx%d) left in BF16", prefix, N, K)
            return r
        kind = os.environ.get("HORIZON_MTP_QUANT", "fp8")
        layer.horizon_mtpq_kind = kind
        if kind == "nvfp4":
            if K % 256:
                _log().info("horizon_kernels MTP quant: %s (%dx%d) left in BF16 (NVFP4 needs K %% 256 == 0)", prefix, N, K)
                return r
            from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
            packed, sc, alpha = quantize_nvfp4_rtn(w.data)
            layer.horizon_mtpq = xl.prepare(packed, sc, alpha)
        else:
            from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
            q, s = quantize_rows(w.data)
            layer.horizon_mtpq = w8.prepare(q, s)
            layer.horizon_mtpq_w = q                   # keep the e4m3 storage alive (prepare views it)
        from horizon_kernels.fp8_linear import _capture_max
        for m in range(1, max(1024, _capture_max()) + 1):   # every M, so nothing compiles while serving / capturing
            _impl_direct(layer, torch.zeros(m, K, device=w.device, dtype=torch.bfloat16))
        torch.cuda.synchronize()
        if free:
            layer.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        layer.horizon_mtpq_idx = len(_LAYERS)
        _LAYERS.append(weakref.ref(layer))
        _log().info("horizon_kernels MTP quant: %s %dx%d -> %s (BF16 %s)", prefix, N, K,
                    "FP8 per-row W8A16" if kind == "fp8" else "NVFP4 RTN on regfrag_xl", "freed" if free else "kept")
        return r

    def apply(self, layer, x, bias=None):
        idx = getattr(layer, "horizon_mtpq_idx", None)
        if idx is None or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        y = torch.ops.horizon.mtp_w8_linear(x, idx)
        return y + bias if bias is not None else y

    apply._horizon_mtpq = True
    UnquantizedLinearMethod.process_weights_after_loading = pw
    UnquantizedLinearMethod.apply = apply
