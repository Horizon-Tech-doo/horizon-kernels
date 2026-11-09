# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels K5 (26-09-2026): small-op work in the decode step, switched by HORIZON_K5=1 (default off).
#
# K5a "skinny": the GDN in_proj_ba projection (BF16, N = 2 x 48 = 96 outputs, K = 5120) runs on our
#   tilelang.contrib.horizon_fused.skinny_bf16 kernel for decode batches (M <= 128) instead of cuBLAS, which picks a
#   16x16-tile kernel with only N/16 = 6 CTAs (~31 us for 1 MB of weights, 48 calls per step). Larger M (prefill) keeps
#   cuBLAS. Rounding: fp32 accumulation + one bf16 rounding, like cuBLAS; only the summation order differs (quality
#   proven with logprob_eval + paired eval, see our run notes "K5").
#   Sub-switch HORIZON_K5_SKINNY (default 1 when HORIZON_K5=1).
#
# K5b "norm": residual add + Gemma RMSNorm (every decoder / MTP norm with hidden 5120) runs on
#   tilelang.contrib.horizon_fused.add_rmsnorm (one CTA of 640 threads per row, everything in registers) instead of
#   Inductor's Triton reduction. Same fp32 elementwise maths as vLLM's native op; the sum of squares is added in a
#   different fixed order (rstd may differ in its last bit). Sub-switch HORIZON_K5_NORM (default 1).
#
# K5c "silu" (27-09-2026, K4 phase 1): SiLU(gate) * up (vLLM's SiluAndMul, the MLP activation between gate_up and down)
#   runs on tilelang.contrib.horizon_fused.silu_mul in its Inductor-exact mode (the same instruction sequence as the
#   Triton kernel Inductor generates: a / (1 + expf(-a)) with div.full.f32, one bf16 rounding; exhaustively
#   bit-tested, k4/silu_bits.py), so gate_up -> silu -> down is ours -> ours and PDL covers both boundaries.
#   Sub-switch HORIZON_K5_SILU (default 0, unlike the others: added later). HORIZON_K5_SILU_MODE picks
#   the silu_mul mode (default 1).
#
# Everything here is installed from register() (vllm_plugin.py) in every vLLM process, needs no CUDA at import time,
# and falls back to the stock op for any layer or batch it does not handle.
import os

import torch
import torch.nn.functional as F

_SK_LAYERS: list = []
_PI_BA = os.environ.get("HORIZON_PI", "0") == "1" and os.environ.get("HORIZON_PI_BA", "1") == "1"


def _pi_prefill_step() -> bool:
    from horizon_kernels.prefill_invariant import prefill_step
    return prefill_step()


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def enabled(name: str, default: str = "1") -> bool:
    return os.environ.get("HORIZON_K5", "0") == "1" and os.environ.get(f"HORIZON_K5_{name}", default) == "1"


# ---------------------------------------------------------------------------------------------------------------
# K5a: skinny BF16 GEMM for in_proj_ba
def _skinny_impl(x: torch.Tensor, idx: int) -> torch.Tensor:
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    w = _SK_LAYERS[idx].weight
    k = w.shape[1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if 1 <= m <= sk.SUPPORTS_M and x2.dtype == torch.bfloat16:
        y = sk.linear(x2.contiguous(), w)
    elif _PI_BA and m >= 1 and x2.dtype == torch.bfloat16 and _pi_prefill_step():
        # HORIZON_PI: in a step with prefill, every row count stays on the batch-invariant skinny kernel
        y = sk.linear(x2.contiguous(), w)
    else:
        y = F.linear(x2, w)
    return y.reshape(*x.shape[:-1], w.shape[0])


def _skinny_fake(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _SK_LAYERS[idx].weight.shape[0]))


def _install_skinny():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    if getattr(UnquantizedLinearMethod.apply, "_horizon_k5", False):
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "skinny_linear")):
        op = torch.library.custom_op("horizon::skinny_linear", mutates_args=())(_skinny_impl)
        op.register_fake(_skinny_fake)
    orig_pw = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply

    def pw(self, layer):
        r = orig_pw(self, layer)
        w = getattr(layer, "weight", None)
        prefix = getattr(layer, "prefix", "") or ""
        layer.horizon_k5_sk = None
        if (prefix.endswith("in_proj_ba") and w is not None and w.dim() == 2 and w.dtype == torch.bfloat16
                and w.is_cuda and getattr(layer, "bias", None) is None):
            from tilelang.contrib.horizon_fused import skinny_bf16 as sk
            n, k = w.shape
            if sk.supported(n, k):
                layer.horizon_k5_sk = len(_SK_LAYERS)
                _SK_LAYERS.append(layer)
                # compile + load every config the dispatcher can pick, outside any CUDA-graph capture
                for cfg in sk.all_configs(n, k):
                    for m in (1, cfg[0], sk.SUPPORTS_M):
                        sk.linear(torch.zeros(m, k, device=w.device, dtype=torch.bfloat16), w.data, cfg=cfg)
                if _pdl_policy():   # HORIZON_PDL: the (config, flags) pair depends on M -> run every decode M once
                    for m in range(1, sk.SUPPORTS_M + 1):
                        sk.linear(torch.zeros(m, k, device=w.device, dtype=torch.bfloat16), w.data)
                torch.cuda.synchronize()
                if len(_SK_LAYERS) == 1:
                    _log().info("horizon_kernels K5: skinny BF16 GEMM active for %s-like layers (N=%d, K=%d, M<=%d)",
                                prefix, n, k, sk.SUPPORTS_M)
        return r

    def apply(self, layer, x, bias=None):
        idx = getattr(layer, "horizon_k5_sk", None)
        if idx is None or bias is not None:
            return orig_apply(self, layer, x, bias)
        return torch.ops.horizon.skinny_linear(x, idx)

    apply._horizon_k5 = True
    UnquantizedLinearMethod.process_weights_after_loading = pw
    UnquantizedLinearMethod.apply = apply


# ---------------------------------------------------------------------------------------------------------------
# K5b: add + Gemma RMSNorm
_NORM_H = 5120


def _norm_ok(x):
    from tilelang.contrib.horizon_fused import add_rmsnorm as an
    return x.dtype == torch.bfloat16 and x.dim() >= 1 and x.shape[-1] % 1024 == 0 and an.supported(x.shape[-1])


_NORM_WARM: set = set()


def _warm(x, eps):
    """first call per (device, H, eps) compiles/loads the TileLang kernels: never inside a CUDA-graph capture"""
    key = (x.device, x.shape[-1], float(eps))
    if key not in _NORM_WARM:
        if torch.cuda.is_current_stream_capturing():
            _log().warning_once("horizon_kernels K5: norm kernel first used inside CUDA-graph capture")
        else:
            warm_norm(x.device, x.shape[-1], eps); torch.cuda.synchronize(); _NORM_WARM.add(key)


def _add_rmsnorm_impl(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    from tilelang.contrib.horizon_fused import add_rmsnorm as an
    _warm(x, eps)
    h = x.shape[-1]
    x2 = x.reshape(-1, h).contiguous(); r2 = r.reshape(-1, h).contiguous()
    if x2.shape[0] == 0:
        return torch.empty_like(x), torch.empty_like(x)
    y, ro = an.add_rmsnorm(x2, r2, w, eps)
    return y.view(x.shape), ro.view(x.shape)


def _add_rmsnorm_fake(x, r, w, eps):
    return torch.empty_like(x), torch.empty_like(x)


def _rmsnorm_impl(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    from tilelang.contrib.horizon_fused import add_rmsnorm as an
    _warm(x, eps)
    h = x.shape[-1]
    x2 = x.reshape(-1, h).contiguous()
    if x2.shape[0] == 0:
        return torch.empty_like(x)
    return an.rmsnorm(x2, w, eps).view(x.shape)


def _rmsnorm_fake(x, w, eps):
    return torch.empty_like(x)


def _pdl_policy() -> bool:
    return os.environ.get("HORIZON_PDL", "0").strip() not in ("", "0")


def _warm_ms(default, max_m=1024):
    """row counts to warm: `default`, plus (with HORIZON_PDL) one M per PDL flag value the policy can pick"""
    if not _pdl_policy():
        return default
    from tilelang.contrib.horizon_k4 import pdl
    return sorted(set(default) | set(pdl.warm_ms(max_m)))


def warm_norm(device, h=_NORM_H, eps=1e-6):
    """compile + load both norm kernels outside CUDA-graph capture (dynamic M: one binary each per PDL flag)."""
    from tilelang.contrib.horizon_fused import add_rmsnorm as an
    for m in _warm_ms((4,)):
        x = torch.zeros(m, h, device=device, dtype=torch.bfloat16)
        an.add_rmsnorm(x, x, x[0], eps); an.rmsnorm(x, x[0], eps)


def _install_norm():
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    if getattr(GemmaRMSNorm.forward_native, "_horizon_k5", False):
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "gemma_add_rmsnorm")):
        op = torch.library.custom_op("horizon::gemma_add_rmsnorm", mutates_args=())(_add_rmsnorm_impl)
        op.register_fake(_add_rmsnorm_fake)
        op = torch.library.custom_op("horizon::gemma_rmsnorm", mutates_args=())(_rmsnorm_impl)
        op.register_fake(_rmsnorm_fake)
    orig = GemmaRMSNorm.forward_native

    def forward_native(self, x, residual=None):
        w = self.weight
        if not (_norm_ok(x) and w.dtype == torch.bfloat16 and w.dim() == 1 and w.shape[0] == x.shape[-1]
                and (residual is None or residual.dtype == x.dtype)):
            return orig(self, x, residual)
        if residual is None:
            return torch.ops.horizon.gemma_rmsnorm(x, w, self.variance_epsilon)
        return torch.ops.horizon.gemma_add_rmsnorm(x, residual, w, self.variance_epsilon)

    forward_native._horizon_k5 = True
    GemmaRMSNorm.forward_native = forward_native
    GemmaRMSNorm.forward_cuda = lambda self, x, residual=None: self.forward_native(x, residual)
    _log().info("horizon_kernels K5: add+Gemma-RMSNorm kernel active (bf16, hidden a multiple of 1024)")


# ---------------------------------------------------------------------------------------------------------------
# K5c: SiLU * mul
_SILU_MAX_M = 1024
_SILU_WARM: set = set()


def _silu_mode() -> int:
    return int(os.environ.get("HORIZON_K5_SILU_MODE", "1"))


def _silu_impl(x: torch.Tensor) -> torch.Tensor:
    from tilelang.contrib.horizon_fused import silu_mul as sm
    i2 = x.shape[-1]
    x2 = x.reshape(-1, i2).contiguous()
    m = x2.shape[0]
    if m == 0:
        return x.new_empty((*x.shape[:-1], i2 // 2))
    key = (x.device, i2)
    if key not in _SILU_WARM:
        if torch.cuda.is_current_stream_capturing():
            _log().warning_once("horizon_kernels K5: silu kernel first used inside CUDA-graph capture")
        else:
            warm_silu(x.device, i2); torch.cuda.synchronize(); _SILU_WARM.add(key)
    y = sm.silu_mul(x2, mode=_silu_mode())
    return y.view(*x.shape[:-1], i2 // 2)


def _silu_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def warm_silu(device, i2):
    """compile + load the silu kernel for every PDL flag the policy can pick (dynamic M: one binary per flag)."""
    from tilelang.contrib.horizon_fused import silu_mul as sm
    for m in _warm_ms((1, _SILU_MAX_M), _SILU_MAX_M):
        sm.silu_mul(torch.zeros(m, i2, device=device, dtype=torch.bfloat16), mode=_silu_mode())


def _silu_ok(x):
    from tilelang.contrib.horizon_fused import silu_mul as sm
    return (x.dtype == torch.bfloat16 and x.dim() >= 1 and x.shape[-1] % (2 * sm.VEC) == 0
            and x.numel() // x.shape[-1] <= _SILU_MAX_M)


def _install_silu():
    from vllm.model_executor.layers.activation import SiluAndMul
    if getattr(SiluAndMul.forward_native, "_horizon_k5", False):
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "silu_mul")):
        op = torch.library.custom_op("horizon::silu_mul", mutates_args=())(_silu_impl)
        op.register_fake(_silu_fake)
    orig = SiluAndMul.forward_native

    def forward_native(x):
        if not _silu_ok(x):
            return orig(x)
        return torch.ops.horizon.silu_mul(x)

    forward_native._horizon_k5 = True
    SiluAndMul.forward_native = staticmethod(forward_native)
    _log().info("horizon_kernels K5: SiLU*mul kernel active (bf16, mode %d, M <= %d)", _silu_mode(), _SILU_MAX_M)


def install():
    if os.environ.get("HORIZON_K5", "0") != "1":
        return
    if enabled("SKINNY"):
        _install_skinny()
    if enabled("NORM"):
        _install_norm()
    if enabled("SILU", "0"):
        _install_silu()
