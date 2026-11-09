# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels vLLM plugin (entry point group "vllm.general_plugins", loaded in every vLLM process).
#
# HORIZON_NVFP4_ADAPTIVE=1 makes vLLM's MarlinNvFp4LinearKernel adaptive: per call it runs Marlin W4A16 (decode,
# small batches) or the residual two-pass NVFP4 GEMM on the SM120 FP4 tensor cores (prefill, large batches), as
# chosen by the tilelang.adaptive calibration table for this GPU (scripts/kernels/calibrate_nvfp4_linear.py).
# The choice is made from the whole batch's token count: in a step that mixes a prefill chunk with decode tokens,
# the decode rows also go through fp4x2 (about 1% GEMM error) -- this is part of what the quality evals measure.
#
# HORIZON_EXACT_SHARD_SCALES (default 0 = vLLM behaviour): with 1, fused layers whose shards carry different NVFP4
# global scales (Qwen3.5/3.8 linear attention in_proj_qkv|in_proj_z) are dequantised per shard on both paths instead
# of vLLM's max() collapse. Per-shard is what the checkpoint maths says (verified against the official FP8 weights),
# BUT the 26-09-2026 long-prompt test showed it breaks long-context recall inside vLLM (perplexity x6.3 on 32k-token
# prose) while the collapse is stable. Under investigation; keep 0 until explained.
#
# Memory: each NVFP4 layer keeps the checkpoint layout (for FP4) next to Marlin's repacked copy (+12.8 GiB) and,
# with HORIZON_REGFRAG=1 (default), regfrag's fragment-ordered copy (+12.8 GiB): memory vLLM's profiler does not
# attribute to weights. Fine with --kv-cache-memory-bytes on a 96 GB GPU.
# HORIZON_DROP_MARLIN=1 (26-09-2026): regfrag (decode_regfrag_xl, every M) replaces Marlin as the exact path and
# Marlin's copy is never built (-12.8 GiB); calibrated into its own table (qwen3_8_nvfp4_linear_nomarlin).
#
# HORIZON_NATIVE_PREFILL=1 (27-09-2026, default 0; needs HORIZON_DROP_MARLIN=1): the FP4 prefill variants (fp4x2,
# fp4x1) read a weight copy stored in the #3257 register-A order (prefill_rs.prepare_weight_native, rsN3 kernel)
# instead of K3's checkpoint-layout copy. The native copy has exactly the bytes of the checkpoint weight it replaces,
# and the checkpoint weight (layer.weight) is freed once regfrag's copy is built, so memory is unchanged. Output is
# bit-identical to K3; decode (regfrag_xl on its own copy) is unchanged.
#
# HORIZON_MODELOPT=1 (27-09-2026, default 0; needs HORIZON_NVFP4_ADAPTIVE=1): NVIDIA ModelOpt NVFP4 linear layers
# (ModelOptNvFp4LinearMethod / ModelOptNvFp4W4A16LinearMethod, e.g. nvidia/Qwen3.8-27B-NVFP4's MLPs and lm_head) take
# the same paths as compressed-tensors layers (regfrag_xl decode, native-order FP4 prefill, drop-Marlin). Both formats
# store uint8 (N, K/2) e2m1 codes (low nibble = even k) and row-major fp8 (N, K/16) block scales before the kernel's
# own process_weights_after_loading; they differ only in the global scale: compressed-tensors keeps a divisor
# (weight_global_scale, alpha = 1/gs), ModelOpt a multiplier (weight_scale_2 = alpha). vLLM collapses a fused layer's
# per-shard scales with max() in both cases, which for ModelOpt means alpha = max(weight_scale_2) (for
# compressed-tensors alpha = 1/max(gs)); HORIZON_EXACT_SHARD_SCALES=0 reproduces each format's stock collapse.
# See horizon_kernels/modelopt.py (also the K1a draft-head guard for a quantised lm_head).
#
# HORIZON_AUTO_CALIBRATE (default 1, 26-09-2026): a layer shape missing from the table (new model, new GPU/driver)
# is calibrated at load, once, and saved (horizon_kernels/calibration.py), so a model switch keeps the fastest
# kernel per shape and batch size without a manual step.
#
# HORIZON_PDL (27-09-2026, K4 phase 1, default 0 = off): Programmatic Dependent Launch in every Horizon TileLang kernel
# (regfrag / regfrag_xl decode GEMMs, native FP4 prefill, K5 skinny / norm / silu): the next kernel is scheduled while
# the current one finishes and prefetches its weights into L2 before waiting for the previous kernel's outputs.
# Value = per-batch-size policy for tilelang.contrib.horizon_k4.pdl.set_policy: "3@8,5@32" = flags 3 (early trigger +
# L2 weight prefetch) for M <= 8 rows, 5 (late trigger) for M <= 32, off above. Only boundaries between two of our
# kernels gain; foreign kernels in between are waited for as before. Output is bit-identical to HORIZON_PDL=0 (the
# kernels' maths is unchanged; docs/k4-design.md section 9).
#
# register() always runs (vLLM imports every general plugin): it only adds the HORIZON_* settings and a hash of the
# horizon_kernels sources to vLLM's torch.compile cache key, so toggling a feature or editing a kernel never loads
# a stale compiled graph. The custom op and the hooks are installed only when HORIZON_NVFP4_ADAPTIVE=1.
import hashlib
import os
import weakref
from pathlib import Path

import torch

_LAYERS: list = []   # layer registry indexed by the custom op's int argument (weak references: a layer vLLM drops, e.g.
                     # the MTP drafter's own lm_head once it is replaced by the target's, frees its prepared copies)
_OP = None
_WARM: set = set()   # (N, K, variant) whose TileLang kernels already ran once outside CUDA-graph capture
_WARM_M: set = set()  # (N, K, variant, M) seen outside capture. Inside CUDA-graph capture a TileLang variant may
                      # only run for an (N, K, M) it already ran eagerly: some compile a new variant per M (regfrag
                      # picks its tile/split config per M; the JIT may specialise others), and a JIT compile or module
                      # load inside capture invalidates the graph. Anything unseen falls back to (exact) Marlin.
_TABLE_CHECKED: set = set()
_STATE: dict = {}    # one-shot flags (kernel-choice check)
# HORIZON_PI (30-09-2026, horizon_kernels/prefill_invariant.py, default off): chunk-invariant prefill
_PI_NVFP4 = os.environ.get("HORIZON_PI", "0") == "1" and os.environ.get("HORIZON_PI_NVFP4", "1") == "1"
_PI_PROBE = bool(os.environ.get("HORIZON_PI_PROBE", ""))


def _op():
    global _OP
    if _OP is None:
        from horizon_kernels.nvfp4_linear import make_op
        _OP = make_op()
    return _OP


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _marlin_exact(x2, layer):
    from horizon_kernels.nvfp4_linear import _marlin
    y = _marlin(x2, layer)
    if layer.horizon_marlin_corr is not None:   # fp32 per-column shard correction, then one bf16 rounding
        y = (y.float() * layer.horizon_marlin_corr).to(y.dtype)
    return y


def _nvfp4_linear_impl(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    layer = _LAYERS[layer_idx]()
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    x2 = x.reshape(-1, k)
    if _PI_PROBE:                        # HORIZON_PI_PROBE: measure-only op probe, once, at the first eager call
        from horizon_kernels import prefill_invariant as _pi
        _pi.maybe_probe(_LAYERS, _op())
    if _PI_NVFP4 and layer.horizon_nvfp4 is not None and not layer.horizon_exact_only:
        from horizon_kernels.prefill_invariant import prefill_step   # HORIZON_PI: a step with prefill -> fp4x2 always
        if prefill_step():
            var = _op().variants.get("fp4x2")
            if var is not None:
                return var.fn(x2, layer).reshape(*x.shape[:-1], n).to(x.dtype)
    v = _op().select((n, k), x2.shape[0])
    if layer.horizon_exact_only and "exact" not in v.tags:   # LM heads: logits stay on the exact class by default
        v = _op().variants[_op().default]
    if layer.horizon_has_marlin and (v.name == "marlin" or layer.horizon_nvfp4 is None
                                     or (v.name == "regfrag" and layer.horizon_regfrag is None)):
        y = _marlin_exact(x2, layer)
    elif torch.cuda.is_current_stream_capturing() and (n, k, v.name, x2.shape[0]) not in _WARM_M:
        # a TileLang kernel's first call (JIT compile, module load, smem attribute) is illegal inside capture.
        # With Marlin: use it for this captured size. Without (HORIZON_DROP_MARLIN=1) the load-time warm-up covers
        # every vLLM capture size, so this should never happen; run the kernel anyway (worst case: vLLM retries).
        if layer.horizon_has_marlin:
            y = _marlin_exact(x2, layer)
        else:
            _log().warning_once("horizon_kernels: unwarmed %s M=%d inside CUDA-graph capture", v.name, x2.shape[0])
            y = v.fn(x2, layer)
    else:
        y = v.fn(x2, layer)
        _WARM.add((n, k, v.name))
        if not torch.cuda.is_current_stream_capturing():
            _WARM_M.add((n, k, v.name, x2.shape[0]))
    return y.reshape(*x.shape[:-1], n).to(x.dtype)


def _capture_max() -> int:
    """Largest token count vLLM records as a CUDA graph (0 if unknown)."""
    try:
        from vllm.config import get_current_vllm_config
        sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes or []
        return int(max(sizes)) if sizes else 0
    except Exception:  # noqa: BLE001
        return 0


def _regfrag_warm_ms(n: int, k: int) -> range:
    """Every M from 1 to 1024 (and to the largest capture size): decode_regfrag_xl picks a config per M bucket, and
    above 64 rows the "rf" path runs 64-row passes whose tail pass (M mod 64 rows) picks its own sub-config, so only
    running every M guarantees no compile or module load happens while serving. ~1-2 s per layer shape once compiled."""
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
    return range(1, max(xl.SUPPORTS_M, _capture_max()) + 1)


def _nvfp4_linear_fake(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _LAYERS[layer_idx]().output_size_per_partition))


def _register_op():
    if hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "nvfp4_linear"):
        return
    op = torch.library.custom_op("horizon::nvfp4_linear", mutates_args=())(_nvfp4_linear_impl)
    op.register_fake(_nvfp4_linear_fake)


def _source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    try:
        import tilelang.adaptive as ad, tilelang.contrib.sm120_nvfp4 as nv
        mods = [ad, nv]
        if os.environ.get("HORIZON_K5", "0") == "1" or os.environ.get("HORIZON_MOE_GATE", "0") == "1":
            import tilelang.contrib.horizon_fused as hf
            mods.append(hf)
        if os.environ.get("HORIZON_MOE_ROUTE", "0") == "1":
            import tilelang.contrib.horizon_moe as hm
            mods.append(hm)
        if os.environ.get("HORIZON_PDL", "0") not in ("", "0") or os.environ.get("HORIZON_K4_SEGMENT", "0") == "1":
            import tilelang.contrib.horizon_k4 as hk
            mods.append(hk)
        if any(os.environ.get(k, "0") == "1" for k in ("HORIZON_GDN", "HORIZON_GDN_RECOVER", "HORIZON_ATTN",
                                                        "HORIZON_PREFILL_RS", "HORIZON_GDN_CONV", "HORIZON_QKNORM",
                                                        "HORIZON_GDN_PREFILL", "HORIZON_KVW")):
            import tilelang.contrib.horizon_ops as ho
            mods.append(ho)
        if os.environ.get("HORIZON_FP8", "0") == "1" or os.environ.get("HORIZON_MTP_QUANT") \
                or os.environ.get("HORIZON_LMHEAD_QUANT") == "fp8":
            import tilelang.contrib.sm120_fp8 as hf8
            mods.append(hf8)
        for m in mods:
            for p in sorted(Path(m.__file__).parent.glob("*.py")):
                h.update(p.name.encode()); h.update(p.read_bytes())
    except Exception:  # noqa: BLE001
        pass
    return h.hexdigest()[:16]


def _extend_compile_cache_key():
    import vllm.envs as envs
    if getattr(envs.compile_factors, "_horizon", False):
        return
    orig = envs.compile_factors
    extra = {k: v for k, v in sorted(os.environ.items()) if k.startswith("HORIZON_")}
    extra["horizon_kernels_src"] = _source_hash()

    def compile_factors():
        f = orig()
        f["HORIZON_KERNELS"] = extra
        return f

    compile_factors._horizon = True
    envs.compile_factors = compile_factors


def _install_capture_mode():
    """HORIZON_CAPTURE_MODE=thread_local|relaxed: record CUDA graphs in that cudaStreamCaptureMode instead of "global".
    gpushare (libgpushare.so, preloaded into every process) runs a background thread that syncs the context when it
    releases its GPU lock, and it only knows a capture is active once it has seen a kernel launch on the capturing
    stream. Under "global", that sync from another thread invalidates vLLM's capture at random (26-09-2026). The
    captured graph is identical; only which threads may make unsafe CUDA calls during capture changes. vLLM never
    asks for a mode explicitly, so only the default "global" is overridden. Patching the method needs no CUDA."""
    mode = os.environ.get("HORIZON_CAPTURE_MODE", "")
    if mode not in ("thread_local", "relaxed"):
        return
    import torch.cuda.graphs as g
    orig = g.CUDAGraph.capture_begin
    if getattr(orig, "_horizon", False):
        return

    def capture_begin(self, pool=None, capture_error_mode="global", **kw):
        return orig(self, pool=pool, capture_error_mode=mode if capture_error_mode == "global" else capture_error_mode, **kw)

    capture_begin._horizon = True
    g.CUDAGraph.capture_begin = capture_begin


def _column_alpha(layer, exact):
    """(alpha fp32 (N,) per output column, alpha Marlin uses, gs for the compressed-tensors prepare path or None) for a
    registered-format NVFP4 layer, or None if the layer is neither a compressed-tensors nor a ModelOpt NVFP4 layer.
    exact=False reproduces vLLM's stock collapse of a fused layer's shard scales for that format."""
    n = layer.output_size_per_partition
    sizes = list(getattr(layer, "logical_widths", [n]))
    gs = getattr(layer, "horizon_shard_gs", None)            # compressed-tensors: divisors, alpha = 1 / gs
    if gs is not None:
        if gs.numel() != len(sizes):
            gs, sizes = gs.max().reshape(1), [n]
        gs_marlin = gs.max()                                  # vLLM: weight_global_scale = 1 / max(gs)
        if not exact:
            gs = gs_marlin.expand(len(sizes)).clone()
        return None, 1.0 / gs_marlin, gs, sizes
    ms = getattr(layer, "horizon_shard_alpha", None)          # ModelOpt: weight_scale_2 is the multiplier itself
    if ms is None:
        return None
    if ms.numel() != len(sizes):
        ms, sizes = ms.max().reshape(1), [n]
    a_marlin = ms.max()                                       # vLLM: weight_global_scale = max(weight_scale_2)
    if not exact:
        ms = a_marlin.expand(len(sizes)).clone()
    alpha = torch.cat([a.expand(s) for a, s in zip(ms, sizes)]).to(layer.weight.device, torch.float32).contiguous()
    return alpha, a_marlin, None, sizes


def _install():
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import compressed_tensors_w4a4_nvfp4 as ct
    from tilelang.contrib import sm120_nvfp4 as nv

    _register_op()
    exact = os.environ.get("HORIZON_EXACT_SHARD_SCALES", "0") == "1"
    from horizon_kernels.nvfp4_linear import XL_MAX_M, drop_marlin, native_prefill, prepare_prefill_weight
    drop = drop_marlin()
    native = native_prefill()
    modelopt = os.environ.get("HORIZON_MODELOPT", "0") == "1"
    if native and not drop:
        raise RuntimeError("horizon_kernels: HORIZON_NATIVE_PREFILL=1 needs HORIZON_DROP_MARLIN=1 (Marlin reads the "
                           "checkpoint weight that the native prefill copy replaces)")
    scheme = ct.CompressedTensorsW4A4Fp4
    orig_scheme_pw = scheme.process_weights_after_loading
    orig_marlin_pw = MarlinNvFp4LinearKernel.process_weights_after_loading
    orig_marlin_apply = MarlinNvFp4LinearKernel.apply_weights

    def scheme_pw(self, layer):
        # vLLM 0.32 port (24-10-2026): our hooks live on MarlinNvFp4LinearKernel. If vLLM picked another NVFP4 kernel
        # (one that sits ahead of Marlin in its priority list and is not in VLLM_DISABLED_KERNELS), none of the Horizon
        # NVFP4 paths run and nothing else says so. Log it once, loudly; HORIZON_STRICT_KERNELS=1 refuses to serve.
        kname = type(getattr(self, "kernel", None)).__name__
        if kname != "MarlinNvFp4LinearKernel" and "kernel_checked" not in _STATE:
            _STATE["kernel_checked"] = True
            msg = ("horizon_kernels: NVFP4 layers are on %s, NOT MarlinNvFp4LinearKernel: the Horizon NVFP4 kernels are "
                   "INACTIVE (check VLLM_DISABLED_KERNELS against this vLLM's NVFP4 priority list)" % kname)
            if os.environ.get("HORIZON_STRICT_KERNELS", "0") == "1":
                raise RuntimeError(msg)
            _log().error(msg)
        elif kname == "MarlinNvFp4LinearKernel" and "kernel_checked" not in _STATE:
            _STATE["kernel_checked"] = True
            _log().info("horizon_kernels: vLLM %s selected MarlinNvFp4LinearKernel for NVFP4 GEMM: Horizon hooks attach",
                        ".".join(map(str, _vllm_version())))
        # keep the per-shard global scales (compressed-tensors divisors) before vLLM collapses them to max()
        layer.horizon_shard_gs = layer.weight_global_scale.detach().float().reshape(-1).clone()
        return orig_scheme_pw(self, layer)

    def marlin_pw(self, layer):
        col = _column_alpha(layer, exact)
        if col is None or layer.weight.dtype != torch.uint8 or layer.weight_scale.dtype != torch.float8_e4m3fn:
            # not a compressed-tensors / (HORIZON_MODELOPT=1) ModelOpt NVFP4 layer as expected: stock Marlin, unchanged
            layer.horizon_idx = None
            layer.horizon_has_marlin = True
            _log().warning_once("horizon_kernels: unexpected NVFP4 layer format, leaving it on stock Marlin")
            return orig_marlin_pw(self, layer)
        alpha, alpha_marlin, gs, sizes = col
        n, k = layer.output_size_per_partition, layer.input_size_per_partition
        # layer.weight is still the checkpoint's packed NVFP4 (N, K/2); weight_scale is fp8 (N, K/16)
        if n % 128 == 0 and k % 256 == 0:
            # K3's handle (a view of layer.weight + swizzled scales) or, with HORIZON_NATIVE_PREFILL=1, the native-order
            # copy + scale tiles (same bytes); same per-column alpha either way
            layer.horizon_nvfp4 = prepare_prefill_weight(layer.weight.data, layer.weight_scale.data, gs, sizes,
                                                         alpha=alpha)
        else:  # shape the FP4 tensor-core kernel cannot tile: this layer always uses Marlin
            layer.horizon_nvfp4 = None
        # our own W4A16 decode kernel (regfrag): its own fragment-ordered weight copy, same per-column alpha as FP4
        layer.horizon_regfrag = None
        if layer.horizon_nvfp4 is not None and "regfrag" in _op().variants:
            from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl
            layer.horizon_regfrag = decode_regfrag_xl.prepare(layer.weight.data.view(torch.int8),
                                                              layer.weight_scale.data, layer.horizon_nvfp4.alpha)
        # Marlin uses one multiplier for the whole layer; a per-column fp32 correction makes it exact per shard
        corr = None
        if exact and gs is not None and gs.numel() > 1 and torch.unique(gs).numel() > 1:   # compressed-tensors
            corr = torch.cat([(gs.max() / g).expand(s) for g, s in zip(gs, sizes)]).to(layer.weight.device, torch.float32)
        elif exact and alpha is not None and torch.unique(alpha).numel() > 1:               # ModelOpt
            corr = (alpha / alpha_marlin.to(alpha.device, torch.float32)).contiguous()
        layer.horizon_marlin_corr = corr
        # bias (none in Qwen3.8's NVFP4 layers) is kept unpermuted: Marlin permutes layer.bias for its own epilogue
        b = getattr(layer, "bias", None)
        layer.horizon_bias = b.detach().clone() if b is not None else None
        # HORIZON_DROP_MARLIN=1: a layer our kernels fully serve never gets Marlin's repacked copy (-12.8 GiB overall);
        # layer.weight stays the checkpoint layout, which the FP4 path (horizon_nvfp4) references anyway
        layer.horizon_has_marlin = not (drop and layer.horizon_nvfp4 is not None and layer.horizon_regfrag is not None)
        if layer.horizon_has_marlin:
            orig_marlin_pw(self, layer)
        elif native:
            # the native copy replaced K3's view of the checkpoint weight: nothing reads layer.weight any more (FP4 uses
            # the native copy, regfrag its own Wg, Marlin is never built), so free it (keeps weights at today's size)
            layer.weight.data = torch.empty(0, dtype=layer.weight.dtype, device=layer.weight.device)
        layer.horizon_idx = len(_LAYERS)
        _LAYERS.append(weakref.ref(layer))
        # an LM head only ever sees one row per sampled position (<= max-num-seqs x (1 + drafts)): calibrate it up to
        # XL_MAX_M only (a 16k-row calibration probe would need ~8 GB of logits per call)
        lm_head = _is_lm_head(layer)
        big = getattr(layer, "horizon_fmt", "") == "modelopt" and lm_head
        # the LM head's output is the logits (verification + sampling): keep it on the exact class (regfrag) unless
        # HORIZON_LMHEAD_FP4=1. The calibration still records where FP4 would be faster (27-09-2026: lm_head
        # 248320x5120 from 65 rows up, because regfrag is untuned for that shape above 64 rows).
        layer.horizon_exact_only = lm_head and os.environ.get("HORIZON_LMHEAD_FP4", "0") != "1"
        op = _op()
        if (n, k) not in _TABLE_CHECKED:
            _TABLE_CHECKED.add((n, k))
            if not op.has_key((n, k)) and layer.horizon_nvfp4 is not None \
                    and os.environ.get("HORIZON_AUTO_CALIBRATE", "1") == "1":
                # a new model (or GPU/driver) brings shapes the table does not know: calibrate them now, once, so the
                # fastest kernel per batch size is used from the first request (result is saved in the table)
                import time
                from horizon_kernels.calibration import calibrate_shape
                t0 = time.time()
                _log().info("horizon_kernels: new NVFP4 shape %s: calibrating once (about 1-2 min)", (n, k))
                buckets = [b for b in op.buckets if b <= XL_MAX_M] if big else None
                calibrate_shape(op, n, k, buckets=buckets, log=lambda msg: _log().info("horizon_kernels: %s", msg))
                _log().info("horizon_kernels: shape %s calibrated in %.0f s -> %s", (n, k), time.time() - t0,
                            op.table_path)
            elif not op.has_key((n, k)):
                _log().warning("horizon_kernels: no calibration entry for NVFP4 shape %s in %s -> default variant "
                               "(%s) only for it", (n, k), op.table_path, op.default)
        # warm up (JIT-compile) every selectable non-Marlin variant now, outside any CUDA-graph capture.
        # regfrag compiles one kernel per M bucket / split config, so every bucket edge is exercised once per shape.
        if layer.horizon_nvfp4 is not None:
            for name, var in op.variants.items():
                if name == "marlin" or (n, k, name) in _WARM or (var.tags & op.exclude_tags):
                    continue
                if name == "regfrag" and layer.horizon_regfrag is None:
                    continue
                if layer.horizon_exact_only and "exact" not in var.tags:
                    continue
                # regfrag chooses its tile/split config per M bucket: run every M that vLLM may capture (and at
                # least 1..64), so no capture size JIT-compiles inside capture, plus one M per distinct config up to
                # 1024, so no kernel compiles or loads while serving (measured 26-09-2026: without this, the first
                # 8-user run after a start was 4-8% slower with HORIZON_DROP_MARLIN=1)
                ms = _regfrag_warm_ms(n, k) if name == "regfrag" else (128,)
                for m in ms:
                    var.fn(torch.zeros(m, k, device=layer.weight.device, dtype=torch.bfloat16), layer)
                    _WARM_M.add((n, k, name, m))
                _WARM.add((n, k, name))
            torch.cuda.synchronize()
        if len(_LAYERS) == 1:
            _log().info("horizon_kernels: adaptive NVFP4 linear active (exact_shard_scales=%s, drop_marlin=%s, "
                        "native_prefill=%s, modelopt=%s, capture_max=%d, table=%s)", exact, drop, native, modelopt,
                        _capture_max(), op.table_path)
        if big:
            _log().info("horizon_kernels: ModelOpt NVFP4 LM head %s on horizon kernels (alpha=%.6g, exact class "
                        "only=%s)", (n, k), float(alpha_marlin), layer.horizon_exact_only)

    def marlin_apply(self, layer, x, bias=None):
        if getattr(layer, "horizon_idx", None) is None:
            return orig_marlin_apply(self, layer, x, bias)
        y = torch.ops.horizon.nvfp4_linear(x, layer.horizon_idx)
        if layer.horizon_bias is not None:
            y = y + layer.horizon_bias
        return y

    scheme.process_weights_after_loading = scheme_pw
    MarlinNvFp4LinearKernel.process_weights_after_loading = marlin_pw
    MarlinNvFp4LinearKernel.apply_weights = marlin_apply
    if modelopt:
        from horizon_kernels import modelopt as mo
        mo.install_linear_hooks()


def _is_lm_head(layer) -> bool:
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        return isinstance(layer, ParallelLMHead)
    except Exception:  # noqa: BLE001
        return False


def _install_pdl():
    spec = os.environ.get("HORIZON_PDL", "0").strip()
    if spec in ("", "0"):
        return
    try:
        from tilelang.contrib.horizon_k4 import pdl
    except Exception as e:  # noqa: BLE001  (a prefix without horizon_k4: stay off, loudly)
        _log().warning("horizon_kernels PDL: HORIZON_PDL=%s but tilelang.contrib.horizon_k4 is missing (%s): off",
                       spec, e)
        return
    pdl.set_policy(spec)
    _log().info("horizon_kernels PDL: policy %s active", pdl.policy_str())


def _vllm_version() -> tuple:
    try:
        from importlib.metadata import version
        return tuple(int(p) for p in version("vllm").split("+")[0].split(".")[:3])
    except Exception:  # noqa: BLE001
        return (0, 0, 0)


# vLLM 0.30 (29-09-2026 port): new NVFP4 linear kernels that sit AHEAD of MarlinNvFp4LinearKernel in vLLM's CUDA
# priority list and would take SM12x layers away from Marlin, so none of the hooks above would ever run (all of
# K3 / regfrag / native prefill silently off). <env-file> lists them in VLLM_DISABLED_KERNELS; this is a backstop.
# Re-verified for vLLM 0.31/0.32 (24-10-2026, static read of kernels/linear/__init__.py): _POSSIBLE_NVFP4_KERNELS[CUDA] is
# FlashInferCuteDsl, FlashInferCutlass, FlashInferB12x, Cutlass, FlashInferCuteDslW4A16, Marlin, ... in the same order as
# 0.30; 0.32 only adds B12xNvFp4W4A16LinearKernel (after Marlin) and a Humming priority bump that applies to SM90 only.
_V030_AHEAD_OF_MARLIN = ("FlashInferCuteDslNvFp4W4A16LinearKernel",)


def _ensure_marlin_nvfp4_path():
    """Plugins load in the engine-core process before its worker processes start and before vLLM caches its env
    (enable_envs_cache), so adding a missing name here reaches the kernel choice in every worker."""
    if _vllm_version() < (0, 30, 0):
        return                                            # 0.29: nothing ahead of Marlin that is not disabled
    cur = [k for k in os.environ.get("VLLM_DISABLED_KERNELS", "").split(",") if k]
    missing = [k for k in _V030_AHEAD_OF_MARLIN if k not in cur]
    if missing:
        os.environ["VLLM_DISABLED_KERNELS"] = ",".join(cur + missing)
        _log().warning("horizon_kernels: vLLM %s: added %s to VLLM_DISABLED_KERNELS (it precedes Marlin on SM12x; "
                       "without it the NVFP4 hooks never run). Put it in the env file.",
                       ".".join(map(str, _vllm_version())), ",".join(missing))


def _feature(name, fn):
    """Install one optional feature; a failure logs an error and leaves that feature off, so one broken patch cannot
    skip every later install (vLLM swallows plugin exceptions). The core NVFP4 install above stays fatal."""
    try:
        fn()
    except Exception as e:
        _log().error("horizon_kernels: feature %s FAILED to install (%s: %s); continuing without it",
                     name, type(e).__name__, e, exc_info=True)


def register():
    """vLLM general-plugin entry point (runs in every vLLM process, including the API server: no CUDA here)."""
    _extend_compile_cache_key()
    _install_capture_mode()
    _install_pdl()      # before any kernel is built or warmed up, so every warm-up compiles the PDL variants
    if os.environ.get("HORIZON_NVFP4_ADAPTIVE") == "1":
        _ensure_marlin_nvfp4_path()
        _install()
    elif os.environ.get("HORIZON_MODELOPT", "0") == "1":
        _log().warning("horizon_kernels: HORIZON_MODELOPT=1 needs HORIZON_NVFP4_ADAPTIVE=1; ModelOpt layers stay stock")
    if os.environ.get("HORIZON_PI", "0") == "1":   # chunk-invariant prefill (horizon_kernels/prefill_invariant.py)
        def _f():
            from horizon_kernels import prefill_invariant
            prefill_invariant.install()
        _feature("prefill_invariant.install", _f)
    if os.environ.get("HORIZON_GDN_CKPT", "0") == "1":   # GDN in-step prefill checkpoint (horizon_kernels/gdn_ckpt.py)
        def _f():
            from horizon_kernels import gdn_ckpt
            gdn_ckpt.install()
        _feature("gdn_ckpt.install", _f)
    if os.environ.get("HORIZON_TAIL_MIN_BLOCK", "0") == "1":   # no tail stop for prompts <= one block (gdn_ckpt.py)
        def _f():
            from horizon_kernels import gdn_ckpt as _gc
            _gc.install_tail()
        _feature("gdn_ckpt.install_tail", _f)
    if os.environ.get("HORIZON_K5", "0") == "1":   # K5 small-op kernels (horizon_kernels/k5.py), default off
        def _f():
            from horizon_kernels import k5
            k5.install()
        _feature("k5.install", _f)
    if os.environ.get("HORIZON_K4_SEGMENT", "0") == "1":   # K4 phase 2 segment kernel (horizon_kernels/k4seg.py)
        def _f():
            from horizon_kernels import k4seg
            k4seg.install()
        _feature("k4seg.install", _f)
    if os.environ.get("HORIZON_MOE", "0") == "1":  # MoE features (horizon_kernels/moe.py), default off
        def _f():
            from horizon_kernels import moe
            moe.install()
        _feature("moe.install", _f)
    if any(os.environ.get(k, "0") == "1" for k in ("HORIZON_GDN", "HORIZON_ATTN", "HORIZON_PREFILL_RS",
                                                    "HORIZON_GDN_CONV", "HORIZON_QKNORM", "HORIZON_GDN_PREFILL",
                                                    "HORIZON_KVW")):  # ops track (ops.py)
        def _f():
            from horizon_kernels import ops
            ops.install()
        _feature("ops.install", _f)
    if os.environ.get("HORIZON_GDN_RECOVER", "0") == "1":  # G2: one GDN checkpoint per request (gdn_recover.py)
        def _f():
            from horizon_kernels import gdn_recover
            gdn_recover.install()
        _feature("gdn_recover.install", _f)
    if (os.environ.get("HORIZON_GDN_STATE", "") not in ("", "0", "fp32", "float32", "auto")
            or os.environ.get("HORIZON_GDN2", "0") == "1"):
        def _f():
            from horizon_kernels import gdn_state
            gdn_state.install()
        _feature("gdn_state.install", _f)
    if os.environ.get("HORIZON_GDN_G3", "0") == "1":  # G3v replay-ring GDN decode (gdn_g3.py)
        def _f():
            from horizon_kernels import gdn_g3
            gdn_g3.install()
        _feature("gdn_g3.install", _f)
    if os.environ.get("HORIZON_FP8", "0") == "1" or os.environ.get("HORIZON_FP8_PERCOL", "0") == "1":  # fp8_linear.py
        def _f():
            from horizon_kernels import fp8_linear
            fp8_linear.install()
        _feature("fp8_linear.install", _f)
    if os.environ.get("HORIZON_LMHEAD_QUANT") or os.environ.get("HORIZON_DUMP_LMHEAD_H"):  # horizon_kernels/lmhead.py
        def _f():
            from horizon_kernels import lmhead
            lmhead.install()
        _feature("lmhead.install", _f)
    if os.environ.get("HORIZON_MTP_QUANT", ""):   # quantized MTP draft layer (horizon_kernels/mtp_quant.py)
        def _f():
            from horizon_kernels import mtp_quant
            mtp_quant.install()
        _feature("mtp_quant.install", _f)
    if os.environ.get("HORIZON_MIX_MLP_FROM", "") or os.environ.get("HORIZON_MIX_NVFP4_FROM_CT", ""):  # ckpt_mix.py
        def _f():
            from horizon_kernels import ckpt_mix
            ckpt_mix.install()
        _feature("ckpt_mix.install", _f)
    if os.environ.get("HORIZON_PMU_MTP_FIX", "0") not in ("", "0"):  # prefix-match-unit + MTP fix (pmu_mtp_fix.py)
        def _f():
            from horizon_kernels import pmu_mtp_fix
            pmu_mtp_fix.install()
        _feature("pmu_mtp_fix.install", _f)
