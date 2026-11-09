# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels: NVIDIA ModelOpt NVFP4 checkpoints (27-09-2026), switched by HORIZON_MODELOPT=1 (default 0).
#
# 1. Linear layers. vLLM 0.29 loads a ModelOpt "NVFP4" layer (ModelOptNvFp4LinearMethod, quantization/modelopt.py) as
#    weight uint8 (N, K/2) (e2m1 pairs, low nibble = even k), weight_scale fp8 e4m3 (N, K/16) (row-major, NOT swizzled:
#    swizzling only happens inside the FlashInfer/CUTLASS kernels' own process_weights_after_loading), weight_scale_2
#    fp32 (one per shard) and input_scale fp32 (one per shard). Its process_weights_after_loading collapses both with
#    max(), renames weight_scale_2 -> weight_global_scale (a MULTIPLIER, not reciprocated) and hands the layer to the
#    selected NVFP4 kernel (MarlinNvFp4LinearKernel on this box). That is the same hand-off point as the
#    compressed-tensors scheme, so the plugin's Marlin hook serves both; the only thing it needs from here is the
#    per-shard multiplier before the collapse (layer.horizon_shard_alpha) and the format tag.
#    W4A16-declared layers (ModelOptNvFp4W4A16LinearMethod) are hooked the same way.
#    ModelOpt's static input_scale is not used: our decode path is W4A16 and our FP4 prefill quantises activations
#    dynamically per row (as for compressed-tensors, whose input_global_scale we ignore too). Stock vLLM ignores it as
#    well on this box (Marlin is W4A16).
#
# 2. K1a draft head on a quantised lm_head. The vLLM patch (patches/vllm029-qwen3_5_mtp-horizon-draft-head.patch)
#    builds the fused FP8 draft head from self.lm_head.weight, which here is packed uint8 (or an empty tensor after
#    HORIZON_NATIVE_PREFILL frees it), and it casts HORIZON_DRAFT_HEAD_WEIGHTS to that dtype: silently garbage drafts.
#    install_draft_head_guard() pre-builds the head from the draft-head files as BF16 before the patched method runs,
#    so the patch finds a head and never touches the quantised lm_head. Without a weights file it refuses loudly.
#    Build a draft head from NVIDIA's own lm_head with scripts/kernels/nvours/make_nvidia_draft_head.py.
import os

import torch

_HOOKED = False


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def install_linear_hooks():
    """Record ModelOpt's per-shard weight_scale_2 (multipliers) before vLLM's max() collapse."""
    global _HOOKED
    if _HOOKED:
        return
    from vllm.model_executor.layers.quantization import modelopt as mo
    if not hasattr(mo, "ModelOptNvFp4LinearMethod"):
        # vLLM 0.30 merged the ModelOpt linear methods into one ModelOptLinearMethod (not ported, 29-09-2026):
        # leave ModelOpt layers on stock kernels rather than fail the start
        _log().warning("horizon_kernels: HORIZON_MODELOPT=1 is not ported to this vLLM (no ModelOptNvFp4LinearMethod); "
                       "ModelOpt NVFP4 layers stay stock")
        _HOOKED = True
        return
    classes = [mo.ModelOptNvFp4LinearMethod]
    if hasattr(mo, "ModelOptNvFp4W4A16LinearMethod"):
        classes.append(mo.ModelOptNvFp4W4A16LinearMethod)
    for cls in classes:
        orig = cls.process_weights_after_loading

        def pw(self, layer, _orig=orig):
            ws2 = getattr(layer, "weight_scale_2", None)
            if ws2 is not None:
                layer.horizon_shard_alpha = ws2.detach().float().reshape(-1).clone()
                layer.horizon_fmt = "modelopt"
            install_draft_head_guard()
            return _orig(self, layer)

        cls.process_weights_after_loading = pw
    _HOOKED = True


_GUARD = False


def install_draft_head_guard():
    """Wrap Qwen3_5MTP.get_top_tokens (the K1a-patched method) so a quantised lm_head never feeds the draft head."""
    global _GUARD
    if _GUARD or os.environ.get("HORIZON_FP8_DRAFT_HEAD") != "1":
        return
    try:
        from vllm.model_executor.models import qwen3_5_mtp as m
    except Exception as ex:  # noqa: BLE001
        _log().warning("horizon_kernels: draft-head guard not installed (%s)", ex)
        return
    cls = m.Qwen3_5MTP
    orig = cls.get_top_tokens
    if getattr(orig, "_horizon_modelopt", False):
        _GUARD = True
        return

    def get_top_tokens(self, hidden_states):
        if getattr(self, "_horizon_draft_head", None) is None and getattr(self.lm_head, "tp_size", 1) == 1:
            w = getattr(self.lm_head, "weight", None)
            if w is not None and not w.is_floating_point():
                _build_head(self, w.device)
        return orig(self, hidden_states)

    get_top_tokens._horizon_modelopt = True
    cls.get_top_tokens = get_top_tokens
    _GUARD = True


def _build_head(model, device):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("horizon_kernels draft head must be built before CUDA graph capture")
    vpath = os.environ.get("HORIZON_DRAFT_VOCAB")
    wpath = os.environ.get("HORIZON_DRAFT_HEAD_WEIGHTS")
    if not (vpath and wpath):
        raise RuntimeError("horizon_kernels: HORIZON_FP8_DRAFT_HEAD=1 on a quantised (NVFP4) lm_head needs "
                           "HORIZON_DRAFT_VOCAB and HORIZON_DRAFT_HEAD_WEIGHTS (a BF16 (V, hidden) head, e.g. from "
                           "make_nvidia_draft_head.py); or set HORIZON_FP8_DRAFT_HEAD=0")
    from safetensors.torch import load_file
    from horizon_kernels.draft_head import DraftHeadFP8
    vocab_ids = torch.load(vpath, map_location=device).to(torch.long)
    weight = load_file(wpath, device=str(device))["weight"].to(torch.bfloat16)
    hidden = model.config.hidden_size
    assert weight.shape == (vocab_ids.numel(), hidden), (weight.shape, vocab_ids.numel(), hidden)
    act = os.environ.get("HORIZON_DRAFT_HEAD_ACT", "fp8")
    head = model._horizon_draft_head = DraftHeadFP8(weight, act=act, vocab_ids=vocab_ids)
    _log().info("horizon_kernels: fused FP8 draft head active (%s x %s, act=%s, draft vocab=%s, trained head=%s, "
                "lm_head quantised: head from %s)", head.N, head.K, act, vocab_ids.numel(), True, wpath)
