# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels FP8 linear (formats track, 27-09-2026): HORIZON_FP8=1 (default 0).
#
# FP8 e4m3 linear layers with a per-tensor or per-channel weight scale (NVIDIA ModelOpt "FP8" attention / GDN
# projections in nvidia/Qwen3.8-27B-NVFP4 and nvidia/Qwen3.6-35B-A3B-NVFP4; compressed-tensors / fp8 per-tensor or
# per-channel checkpoints) run, per batch size, on the faster of
#   w8a16    our W8A16 GEMM (tilelang.contrib.sm120_fp8.decode_w8a16): BF16 activations x the layer's own e4m3 weight
#            bytes (no copy), exact e4m3 -> BF16 in registers, fp32 accumulation, per-column fp32 scale, one BF16
#            rounding. No activation quantisation (more accurate than W8A8). Deterministic, graph-capturable.
#   cutlass  vLLM's stock path on this GPU: static FP8 activation quantisation + CUTLASS sm120 W8A8 GEMM.
# as chosen by a tilelang.adaptive table (op "horizon_fp8_linear", same TILELANG_ADAPTIVE_DIR as the NVFP4 table),
# calibrated at load for any shape the table does not know (HORIZON_AUTO_CALIBRATE, default 1).
#   HORIZON_FP8_FORCE=w8a16|cutlass   one variant for every batch size (A/B arms)
#   HORIZON_FP8_W8A16_MAX_M=<m>       w8a16 up to m rows, cutlass above (no table; 27-09-2026 race: the two are at parity
#                                     at M <= 16, CUTLASS's FP8 tensor cores win from M = 32 up)
# Hook point: vLLM's CutlassFP8ScaledMMLinearKernel (process_weights_after_loading / apply_weights), i.e. every FP8
# per-tensor / per-channel linear that vLLM put on CUTLASS, after vLLM's own weight processing (a fused layer with
# unequal shard scales has already been re-quantised to the max scale by vLLM: the weights are exactly the ones stock
# vLLM multiplies, so the only numerics change is W8A8 -> W8A16).
#   HORIZON_FP8_PERCOL=1 (27-09-2026, independent of HORIZON_FP8): a fused ModelOpt FP8 layer whose shards carry
#                          different per-tensor weight scales is NOT re-quantised to the max scale (vLLM's
#                          requantize_with_max_scale rounds the smaller-scale shard a second time); the original e4m3
#                          weights are kept and the shard scales become one per-output-column scale (N,), which both
#                          vLLM's CUTLASS scaled_mm (numpy-broadcast scale_b) and our W8A16 kernel accept.
# Installed from vllm_plugin.register(); needs no CUDA at import time.
import os
import weakref

import torch

_LAYERS: list = []       # weak references, indexed by the custom op's int argument
_OP = None
_WARM_M: set = set()     # (N, K, M) our kernel already ran eagerly (outside graph capture)
_TABLE_CHECKED: set = set()
MAX_M_NATIVE = 1024


def enabled() -> bool:
    return os.environ.get("HORIZON_FP8", "0") == "1"


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _w8():
    from tilelang.contrib.sm120_fp8 import decode_w8a16
    return decode_w8a16


def _v_w8a16(x2, layer):
    return _w8().linear(x2, layer.horizon_w8)


def _v_cutlass(x2, layer):
    return layer.horizon_fp8_stock(x2)


def make_op():
    from tilelang.adaptive import AdaptiveOp, Variant
    variants = [Variant("cutlass", _v_cutlass, rel_tol=0.10),                 # stock W8A8 (activation quant error)
                Variant("w8a16", _v_w8a16, rel_tol=0.0, tags={"exact"})]      # the accurate one: the reference
    return AdaptiveOp("horizon_fp8_linear", variants, default="cutlass", reference="w8a16", margin=0.10,
                      exact_margin=0.02)


def _op():
    global _OP
    if _OP is None:
        _OP = make_op()
    return _OP


def _fixed_policy() -> bool:
    return bool(os.environ.get("HORIZON_FP8_FORCE") or os.environ.get("HORIZON_FP8_W8A16_MAX_M"))


def _select(n, k, m):
    force = os.environ.get("HORIZON_FP8_FORCE", "")
    if force:
        return _op().variants[force]
    max_m = int(os.environ.get("HORIZON_FP8_W8A16_MAX_M", "0") or 0)
    if max_m > 0:
        return _op().variants["w8a16" if m <= max_m else "cutlass"]
    return _op().select((n, k), m)


def _impl(x: torch.Tensor, idx: int) -> torch.Tensor:
    layer = _LAYERS[idx]()
    p = layer.horizon_w8
    x2 = x.reshape(-1, p.K)
    m = x2.shape[0]
    v = _select(p.N, p.K, m)
    if v.name == "w8a16" and torch.cuda.is_current_stream_capturing() and (p.N, p.K, m) not in _WARM_M \
            and m <= MAX_M_NATIVE:
        _log().warning_once("horizon_kernels FP8: unwarmed w8a16 M=%d inside CUDA-graph capture", m)
    y = v.fn(x2, layer)
    if v.name == "w8a16" and not torch.cuda.is_current_stream_capturing():
        _WARM_M.add((p.N, p.K, m))
    return y.reshape(*x.shape[:-1], p.N).to(x.dtype)


def _fake(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _LAYERS[idx]().horizon_w8.N))


def _register_op():
    if hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "fp8_linear"):
        return
    op = torch.library.custom_op("horizon::fp8_linear", mutates_args=())(_impl)
    op.register_fake(_fake)


def _capture_max() -> int:
    try:
        from vllm.config import get_current_vllm_config
        sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes or []
        return int(max(sizes)) if sizes else 0
    except Exception:  # noqa: BLE001
        return 0


def stock_fn(w_nk, w_scale, in_scale):
    """vLLM's CUTLASS W8A8 path for a stand-in layer (calibration): static per-tensor activation quant + GEMM."""
    from vllm import _custom_ops as ops
    b = w_nk.t()
    ws = w_scale.reshape(-1).float()
    ws = ws if ws.numel() > 1 else ws.reshape(1)
    xs = in_scale.reshape(1).float()

    def f(x2):
        xq, s = ops.scaled_fp8_quant(x2, xs)
        return ops.cutlass_scaled_mm(xq, b, out_dtype=torch.bfloat16, scale_a=s,
                                     scale_b=ws.reshape(-1, 1) if ws.numel() > 1 else ws)
    return f


def calibrate_shape(op, N, K, buckets=None, log=print):
    """Calibrate one FP8 (N, K) into op's table with random e4m3 weights (> 512 MB of rotated copies)."""
    import types
    w8 = _w8()
    copies = max(2, int(512e6 // (N * K)) + 1)
    layers = []
    for s in range(copies):
        g = torch.Generator(device="cuda").manual_seed(s)
        w = (torch.randn(N, K, device="cuda", generator=g) * 64).to(torch.float8_e4m3fn)
        ws = torch.tensor(1e-3, device="cuda")
        l = types.SimpleNamespace(horizon_w8=w8.prepare(w, ws), horizon_fp8_stock=stock_fn(w, ws,
                                                                                            torch.tensor(4.0 / 448, device="cuda")))
        l._w = w
        layers.append(l)

    def make_args(m):
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        return [(x, l) for l in layers]

    op.calibrate(key=(N, K), make_args=make_args, ms=buckets, iters=5, log=log)
    del layers
    torch.cuda.empty_cache()
    return copies


def install_percol():
    """HORIZON_FP8_PERCOL=1: per-column weight scales instead of vLLM's re-quantisation of fused FP8 shards."""
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization import modelopt as mo
    cls = getattr(mo, "ModelOptFp8LinearMethod", None)
    if cls is None:
        # vLLM >= 0.30 merged the ModelOpt linear methods into one ModelOptLinearMethod (not ported): leave the FP8
        # layers on stock requantisation rather than crash register() (this raised AttributeError before 24-10-2026)
        _log().warning("horizon_kernels FP8: HORIZON_FP8_PERCOL=1 is not ported to this vLLM (no "
                       "ModelOptFp8LinearMethod); fused FP8 layers keep vLLM's requantisation")
        return
    if getattr(cls.process_weights_after_loading, "_horizon_percol", False):
        return
    orig = cls.process_weights_after_loading
    stats = {"layers": 0}

    def pw(self, layer):
        ws = getattr(layer, "weight_scale", None)
        widths = list(getattr(layer, "logical_widths", []) or [])
        if ws is None or ws.numel() <= 1 or ws.numel() != len(widths) or bool((ws == ws.reshape(-1)[0]).all()):
            return orig(self, layer)
        col = torch.cat([ws.reshape(-1)[i].float().expand(w) for i, w in enumerate(widths)]).reshape(-1, 1)
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight.input_dim = 0
        layer.weight.output_dim = 1
        layer.weight_scale = Parameter(col.contiguous(), requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        self.fp8_linear.process_weights_after_loading(layer)
        stats["layers"] += 1
        if stats["layers"] == 1:
            _log().info("horizon_kernels FP8: per-column shard scales (no re-quantisation) for fused FP8 layers "
                        "(HORIZON_FP8_PERCOL=1)")

    pw._horizon_percol = True
    cls.process_weights_after_loading = pw


def install():
    if os.environ.get("HORIZON_FP8_PERCOL", "0") == "1":
        install_percol()
    if os.environ.get("HORIZON_FP8", "0") != "1":
        return
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel as C
    if getattr(C.apply_weights, "_horizon_fp8", False):
        return
    _register_op()
    orig_pw = C.process_weights_after_loading
    orig_apply = C.apply_weights

    def pw(self, layer):
        r = orig_pw(self, layer)
        layer.horizon_fp8_idx = None
        try:
            w_name, ws_name, xs_name, _ = self.layer_param_names
            w = getattr(layer, w_name)
            ws = getattr(layer, ws_name)
            wt = w.t()                                    # vLLM stores B as (K, N) column-major: .t() is (N, K)
            N, K = wt.shape
            ok = (w.dtype == torch.float8_e4m3fn and w.dim() == 2 and wt.is_contiguous() and N % 128 == 0
                  and K % 64 == 0 and ws is not None and ws.numel() in (1, N)
                  and getattr(self, "logical_output_size", N) == N)
        except Exception as ex:  # noqa: BLE001
            _log().warning_once("horizon_kernels FP8: layer not handled (%s)", ex)
            ok = False
        if not ok:
            return r
        w8 = _w8()
        layer.horizon_w8 = w8.prepare(wt, ws.detach().float().reshape(-1))
        layer.horizon_fp8_stock = (lambda x2, _k=self, _l=weakref.ref(layer): orig_apply(_k, _l(), x2))
        layer.horizon_fp8_idx = len(_LAYERS)
        _LAYERS.append(weakref.ref(layer))
        op = _op()
        if (N, K) not in _TABLE_CHECKED:
            _TABLE_CHECKED.add((N, K))
            if not op.has_key((N, K)) and not _fixed_policy() \
                    and os.environ.get("HORIZON_AUTO_CALIBRATE", "1") == "1":
                import time
                t0 = time.time()
                _log().info("horizon_kernels FP8: new shape %s: calibrating once", (N, K))
                calibrate_shape(op, N, K, log=lambda msg: _log().info("horizon_kernels: %s", msg))
                _log().info("horizon_kernels FP8: shape %s calibrated in %.0f s -> %s", (N, K), time.time() - t0,
                            op.table_path)
            # warm (JIT-compile + load) our kernel for every M it can see, outside any CUDA-graph capture
            for m in range(1, max(MAX_M_NATIVE, _capture_max()) + 1):
                if _select(N, K, m).name == "w8a16" and (N, K, m) not in _WARM_M:
                    w8.linear(torch.zeros(m, K, device=wt.device, dtype=torch.bfloat16), layer.horizon_w8)
                    _WARM_M.add((N, K, m))
            torch.cuda.synchronize()
        if len(_LAYERS) == 1:
            _log().info("horizon_kernels FP8: W8A16 kernel active (force=%s, table=%s)",
                        os.environ.get("HORIZON_FP8_FORCE", "") or (("w8a16 up to M=" + os.environ["HORIZON_FP8_W8A16_MAX_M"])
                                                         if os.environ.get("HORIZON_FP8_W8A16_MAX_M") else "adaptive"),
            op.table_path)
        return r

    def apply_weights(self, layer, x, bias=None):
        idx = getattr(layer, "horizon_fp8_idx", None)
        if idx is None or bias is not None or not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        return torch.ops.horizon.fp8_linear(x, idx)

    apply_weights._horizon_fp8 = True
    C.process_weights_after_loading = pw
    C.apply_weights = apply_weights
