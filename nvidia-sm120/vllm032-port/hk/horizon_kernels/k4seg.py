# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels K4 phase 2 (27-09-2026): the K4 segment kernel inside vLLM. Switch HORIZON_K4_SEGMENT=1 (default 0).
#
# One persistent launch (tilelang.contrib.horizon_k4.segment) per layer boundary replaces the kernel sequence
#     out_proj(L) -> add+RMSNorm(L) -> gate_up(L) -> SiLU*mul -> down(L) -> add+RMSNorm(L+1) -> in_proj(L+1) [+ in_proj_ba]
# for decode-sized batches (1..32 tokens). The model's forward is restructured so that this stretch is ONE custom op,
# horizon::k4_segment(core, residual, cur, nxt) -> (in_proj output of L+1, in_proj_ba output of L+1, residual), between
# layer L's attention core (GDN core op / attention) and layer L+1's attention core. Batches outside the segment kernel's
# range (prefill, > 32 tokens) run the very same ops one by one inside the custom op (the adaptive NVFP4 linear, K5
# norm, K5 SiLU), so every batch size keeps production's numerics; the segment kernel itself is bit-identical to that
# sequence (seg_test.py, our run notes "K4 phase 2").
# Requires HORIZON_NVFP4_ADAPTIVE=1 + HORIZON_DROP_MARLIN=1 (regfrag weights), HORIZON_K5=1 with NORM and SILU
# (the op calls those kernels directly). Dense Qwen3.5/3.8 text models only; anything else keeps the stock forward.
# Timeout guard (a K4 launch whose counter wait times out has WRONG outputs and cannot stop the step):
#   * at load, every segment kernel runs once eagerly and synchronously (warm-up): a timeout there (e.g. the CTAs do not
#     all fit on the GPU at once) turns K4 off for this process (production's per-op path, same numerics) + an error log;
#   * every decode step: right before vLLM hands the sampled tokens back (AsyncOutput.get_output of the V2 model runner,
#     after its copy event has synchronised, so every kernel of the step has finished), the host-mapped error word is
#     read (a plain load). If it is set, the step raises instead of returning tokens (its requests fail with an error,
#     no wrong token reaches a user), the marker file is written and the engine stops;
#   * backstop: the kernel also writes its error flag into page-locked host memory (runtime.HostFlag); a watchdog
#     thread polls that word (no CUDA call, no sync) every 50 ms. On a trip it logs CRITICAL, writes the marker file
#     HORIZON_K4_DISABLE_MARKER (default ~/.cache/horizon_kernels/k4_disabled) and exits the engine process at once,
#     so vLLM restarts and the next start sees the marker and keeps K4 off. Delete the marker
#     to re-enable K4 after the cause is fixed.
#   Missing per-step output hooks or host-mapped error flags disable K4. A watchdog alone is insufficient to
#   prevent a failed step from being returned during its polling interval.
#   HORIZON_K4_TIMEOUT_MS (default 2000): counter-wait timeout inside the kernel.
# HORIZON_K4_GDN=1 (phase 3a, needs HORIZON_K4_SEGMENT=1; default 0): for GDN layers in a pure speculative-decode
#   step (vLLM's fused GDN MTP decode path) the layer's GDN core (conv1d update, recurrence, gated norm; the ops
#   track's horizon_ops bodies, bit-identical to vLLM's kernels) runs INSIDE the same persistent launch as the segment:
#   one launch per GDN layer = conv1d + core + gated norm + out_proj ... next layer's in_proj. Custom op
#   horizon::k4_gdn_layer (a torch.compile splitting op, like vLLM's own GDN op, because it reads the step's
#   metadata); recurrent state must be FP32. Other batches/state formats run vLLM's GDN op + horizon::k4_segment.
# HORIZON_K4_TUNE=<json> (from scripts/kernels/k4p1/seg_sweep.py (not in this release)): per-M segment configs {"M": {hw, stages, pf_kt,
#   ncta_per_sm} or {"off": 1}}; a batch of M rows uses the entry of the smallest key >= M ("off": per-op path).
# Sub-switches: HORIZON_K4_NCTA (default 2 per SM), HORIZON_K4_MAX_M (default 32), HORIZON_K4_HW (GEMM task pipeline:
# 0 TileLang-pipelined, 1 weights before the wait, 2 run ahead = default), HORIZON_K4_STAGES (4), HORIZON_K4_PF_KT
# (L2-prefetch depth before a wait, 0 = stages).
import os

import torch

_MODELS: list = []            # patched Qwen3_5Model instances (layers looked up by (model, layer index))
_SEGS: dict = {}              # (M, nq, device) -> Segment
_FAILED: set = set()
_WARMED: set = set()
_STATE = {"disabled": False, "watchdog": None, "flags": {}}


def _marker() -> str:
    return os.path.expanduser(os.environ.get("HORIZON_K4_DISABLE_MARKER", "~/.cache/horizon_kernels/k4_disabled"))


def _trip(name):
    """watchdog: a K4 launch timed out while serving -> outputs of that step are wrong: stop the engine now"""
    _write_marker(name)
    try:
        _log().critical("horizon_kernels K4: counter-wait TIMEOUT in the segment kernel (%s): outputs of the current "
                        "step are invalid; exiting the engine so vLLM restarts without K4 (marker %s)", name, _marker())
    finally:
        os._exit(70)


class K4TimeoutError(RuntimeError):
    pass


def _write_marker(name):
    import time
    try:
        os.makedirs(os.path.dirname(_marker()), exist_ok=True)
        with open(_marker(), "a") as f:
            f.write(f"{time.strftime('%d-%m-%Y %H:%M:%S')} horizon_kernels K4: segment kernel timed out ({name}); "
                    f"K4 disabled until this file is removed\n")
    except OSError:
        pass


def check_step(where: str = "step"):
    """per-step guard: plain loads of the host-mapped error words (no CUDA call). Raises K4TimeoutError if any is set,
    after writing the disable marker; call only where the step's GPU work is known to be complete."""
    for name, f in list(_STATE["flags"].items()):
        if f.get():
            _write_marker(f"{name}, {where}")
            _log().critical("horizon_kernels K4: segment kernel TIMEOUT detected at %s: this step's tokens are "
                            "discarded (requests fail); stopping the engine, K4 off after restart (marker %s)",
                            where, _marker())
            raise K4TimeoutError("horizon_kernels K4: segment kernel timed out; step discarded")


def _wrap_get_output(cls, where):
    """patch cls.get_output: run it (it synchronises the step's copy event), then check_step before returning"""
    if getattr(cls.get_output, "_horizon_k4", False):
        return
    orig = cls.get_output

    def get_output(self):
        out = orig(self)
        check_step(where)
        return out

    get_output._horizon_k4 = True
    cls.get_output = get_output


def _install_step_check():
    n = 0
    for mod, cls in (("vllm.v1.worker.gpu.async_utils", "AsyncOutput"),
                     ("vllm.v1.worker.gpu_model_runner", "AsyncGPUModelRunnerOutput")):
        try:
            import importlib
            c = getattr(importlib.import_module(mod), cls)
            _wrap_get_output(c, f"{cls}.get_output")
            n += 1
        except Exception:  # noqa: BLE001
            pass
    if n == 0:
        _STATE["disabled"] = True
        _log().warning("horizon_kernels K4: no model-runner output hook found; K4 stays OFF because the "
                       "watchdog alone cannot reject a failed step before tokens are returned")
        return False
    return True


def _watch(seg):
    if not getattr(seg.errh, "mapped", False):
        _STATE["disabled"] = True
        _log().warning("horizon_kernels K4: host-mapped error flag unavailable; K4 stays OFF because a "
                       "device-only flag cannot be checked before returning the step without GPU synchronization")
        return False
    _STATE["flags"][str(seg.errh.dptr)] = seg.errh
    if _STATE["watchdog"] is None:
        from tilelang.contrib.horizon_k4 import runtime as rt
        _STATE["watchdog"] = rt.Watchdog(_trip)
    _STATE["watchdog"].add(str(seg.errh.dptr), seg.errh)
    return True


def enabled() -> bool:
    return os.environ.get("HORIZON_K4_SEGMENT", "0") == "1"


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _max_m() -> int:
    return int(os.environ.get("HORIZON_K4_MAX_M", "32"))


def _is_gdn(layer) -> bool:
    return getattr(layer, "layer_type", "") == "linear_attention"


def _out_lin(layer):
    return layer.linear_attn.out_proj if _is_gdn(layer) else layer.self_attn.o_proj


def _in_lin(layer):
    return layer.linear_attn.in_proj_qkvz if _is_gdn(layer) else layer.self_attn.qkv_proj


def _layer(mi: int, li: int):
    return _MODELS[mi].layers[li]


# ---------------------------------------------------------------------------------------------------------------
# the custom op
def _lin(x, lin):
    """the plugin's adaptive NVFP4 linear (what vLLM calls for this layer)"""
    return torch.ops.horizon.nvfp4_linear(x, lin.horizon_idx)


def _fallback(core, residual, Lc, Ln):
    """the same ops production runs, one kernel each (K5 norm, K5 SiLU, adaptive NVFP4 linear, K5 skinny)"""
    pn, nn_ = Lc.post_attention_layernorm, Ln.input_layernorm
    o = _lin(core, _out_lin(Lc))
    y2, r2 = torch.ops.horizon.gemma_add_rmsnorm(o, residual, pn.weight, pn.variance_epsilon)
    a = torch.ops.horizon.silu_mul(_lin(y2, Lc.mlp.gate_up_proj))
    d = _lin(a, Lc.mlp.down_proj)
    y1, r1 = torch.ops.horizon.gemma_add_rmsnorm(d, r2, nn_.weight, nn_.variance_epsilon)
    q = _lin(y1, _in_lin(Ln))
    if _is_gdn(Ln):
        ba = Ln.linear_attn.in_proj_ba.quant_method.apply(Ln.linear_attn.in_proj_ba, y1)
    else:
        ba = y1.new_empty((y1.shape[0], 0))
    return q, ba, r1


_GDN_LAYERS: dict = {}


def _gdn_enabled() -> bool:
    return enabled() and os.environ.get("HORIZON_K4_GDN", "0") == "1"


def _gdn_inputs(attn, md):
    """the GDN state / metadata dict for segment.GdnLayer, or None when this batch is not the fused MTP decode"""
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
        if md is None or not attn._can_use_fused_gdn_mtp_decode(md):
            return None
        if getattr(md, "num_prefills", 0) or attn.conv1d.bias is not None or attn.tp_size != 1:
            return None
        # vLLM's fused predicate also accepts BF16 recurrent state. The
        # persistent kernel currently declares its SSM tensor as FP32 only.
        if attn.kv_cache[1].dtype != torch.float32:
            return None
        R = int(md.num_spec_decodes)
        conv = attn.kv_cache[0] if q.is_conv_state_dim_first() else attn.kv_cache[0].transpose(-1, -2)
        sidx = md.spec_state_indices_tensor[:R]
        return dict(conv_state=conv, conv_w=attn.conv1d.weight.view(attn.conv1d.weight.size(0), -1),
                    ssm=attn.kv_cache[1], sidx=sidx, cu=md.spec_query_start_loc[:R + 1],
                    nacc=md.num_accepted_tokens[:R], a_log=attn.A_log, dt_bias=attn.dt_bias, norm_w=attn.norm.weight,
                    act_sigmoid=int(getattr(attn.norm, "activation", "silu") == "sigmoid"), lmax=int(sidx.size(1)),
                    scale=attn.head_k_dim ** -0.5, eps=attn.layer_norm_epsilon, R=R)
    except Exception:  # noqa: BLE001
        return None


def _gdn_layer(M, R, nq, g, device):
    key = (M, R, nq, device, tuple(g["conv_state"].stride()), g["ssm"].stride(0), g["dt_bias"].dtype,
           g["norm_w"].dtype, g["act_sigmoid"], tuple(g["conv_state"].shape[2:]))
    lay = _GDN_LAYERS.get(key)
    if lay is None:
        # Captured calls must reuse a kernel and workspace prepared eagerly.
        # Constructor work includes compilation, host transfers and allocations.
        if torch.cuda.is_current_stream_capturing():
            _log().warning_once("horizon_kernels persistent GDN decode: no warmed kernel for M=%d R=%d; "
                                "capturing the per-op path", M, R)
            return None
        from tilelang.contrib.horizon_k4 import segment as sg
        ncta = int(os.environ.get("HORIZON_K4_NCTA", "0")) or None
        kw = dict(hw=int(os.environ.get("HORIZON_K4_HW", "2")), stages=int(os.environ.get("HORIZON_K4_STAGES", "4")),
                  pf_kt=int(os.environ.get("HORIZON_K4_PF_KT", "0")))
        lay = _GDN_LAYERS[key] = sg.GdnLayer(M, R, g, device=device, ncta=ncta, nq=nq,
                                             silu_mode=int(os.environ.get("HORIZON_K5_SILU_MODE", "1")),
                                             timeout_ms=int(os.environ.get("HORIZON_K4_TIMEOUT_MS", "2000")), **kw)
        if not _watch(lay):
            _GDN_LAYERS.pop(key, None)
            return None
    return lay


def _gdn_impl(proj: torch.Tensor, ba: torch.Tensor, residual: torch.Tensor, model: int, cur: int, nxt: int
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Lc, Ln = _layer(model, cur), _layer(model, nxt)
    attn = Lc.linear_attn
    M = proj.shape[0]
    g = None
    if M and _k4_ready(M, Lc, Ln):
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        md = md.get(attn.prefix) if isinstance(md, dict) else None
        g = _gdn_inputs(attn, md)
    if g is not None and g["R"] * g["lmax"] == M:
        nq = _in_lin(Ln).output_size_per_partition
        lay = _gdn_layer(M, g["R"], nq, g, proj.device)
        if lay is not None:
            lc, ln = _weights(Lc, Ln)
            q, ba2, r1 = lay(proj, ba, residual.contiguous(), g, lc, ln)
            if not _is_gdn(Ln):
                ba2 = proj.new_empty((M, 0))
            return q, ba2, r1
    core = _gdn_core(attn, proj, ba)
    return torch.ops.horizon.k4_segment(core, residual, model, cur, nxt)


def _gdn_fake(proj, ba, residual, model, cur, nxt):
    return _seg_fake(proj.new_empty((proj.shape[0], 6144)), residual, model, cur, nxt)


def _k4_ready(M, Lc, Ln) -> bool:
    """the segment kernel serves this batch: 1 <= M <= max, every GEMM on regfrag (what the adaptive table picks), K5
    kernels in place"""
    if _STATE["disabled"] or not (1 <= M <= _max_m()):
        return False
    tn = _tune(M)
    if tn is not None and tn.get("off"):
        return False
    from horizon_kernels.vllm_plugin import _op
    lins = [_out_lin(Lc), Lc.mlp.gate_up_proj, Lc.mlp.down_proj, _in_lin(Ln)]
    shapes = [(l.output_size_per_partition, l.input_size_per_partition) for l in lins]
    if shapes[:3] != [(5120, 6144), (34816, 5120), (5120, 17408)] or shapes[3] not in ((16384, 5120), (14336, 5120)):
        return False
    for lin in lins:
        if getattr(lin, "horizon_regfrag", None) is None or getattr(lin, "horizon_idx", None) is None:
            return False
        if _op().select((lin.output_size_per_partition, lin.input_size_per_partition), M).name != "regfrag":
            return False
    if _is_gdn(Ln) and getattr(Ln.linear_attn.in_proj_ba, "horizon_k5_sk", None) is None:
        return False
    return True


_TUNE: dict = {}


def _tune(M):
    """per-M config from HORIZON_K4_TUNE (smallest key >= M), or None"""
    path = os.environ.get("HORIZON_K4_TUNE", "")
    if not path:
        return None
    if "table" not in _TUNE:
        import json
        try:
            _TUNE["table"] = sorted((int(k), v) for k, v in json.load(open(path)).items())
        except Exception as e:  # noqa: BLE001
            _log().warning("horizon_kernels K4: cannot read HORIZON_K4_TUNE=%s (%s): defaults", path, e)
            _TUNE["table"] = []
    for m, v in _TUNE["table"]:
        if M <= m:
            return v
    return None


def _segment(M, nq, device):
    key = (M, nq, device)
    seg = _SEGS.get(key)
    if seg is None:
        from tilelang.contrib.horizon_k4 import segment as sg
        ncta = int(os.environ.get("HORIZON_K4_NCTA", "0")) or None
        mode = int(os.environ.get("HORIZON_K5_SILU_MODE", "1"))
        kw = dict(hw=int(os.environ.get("HORIZON_K4_HW", "2")), stages=int(os.environ.get("HORIZON_K4_STAGES", "4")),
                  pf_kt=int(os.environ.get("HORIZON_K4_PF_KT", "0")))
        tn = _tune(M)
        if tn:
            kw.update({k: int(tn[k]) for k in ("hw", "stages", "pf_kt") if k in tn})
            if "ncta_per_sm" in tn:
                ncta = int(tn["ncta_per_sm"]) * torch.cuda.get_device_properties(device).multi_processor_count
        seg = _SEGS[key] = sg.Segment(M, device=device, ncta=ncta, silu_mode=mode, nq=nq,
                                      timeout_ms=int(os.environ.get("HORIZON_K4_TIMEOUT_MS", "2000")), **kw)
    return seg


def _weights(Lc, Ln):
    lc = dict(out=_out_lin(Lc).horizon_regfrag, gu=Lc.mlp.gate_up_proj.horizon_regfrag,
              down=Lc.mlp.down_proj.horizon_regfrag, w2=Lc.post_attention_layernorm.weight)
    ln = dict(qkvz=_in_lin(Ln).horizon_regfrag, w1=Ln.input_layernorm.weight)
    if _is_gdn(Ln):
        ln["ba"] = Ln.linear_attn.in_proj_ba.weight
    return lc, ln


def _seg_impl(core: torch.Tensor, residual: torch.Tensor, model: int, cur: int, nxt: int
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Lc, Ln = _layer(model, cur), _layer(model, nxt)
    M = core.shape[0]
    if model not in _WARMED and not torch.cuda.is_current_stream_capturing():
        _WARMED.add(model)
        import time
        t0 = time.time()
        if warm(model, core.device):
            _log().info("horizon_kernels K4: segment kernels warmed for M 1..%d in %.0f s", _max_m(), time.time() - t0)
    if M and _k4_ready(M, Lc, Ln) and (M, cur) not in _FAILED:
        nq = _in_lin(Ln).output_size_per_partition
        if torch.cuda.is_current_stream_capturing() and (M, nq, core.device) not in _SEGS:
            _log().warning_once("horizon_kernels persistent decode: no warmed segment for M=%d; "
                                "capturing the per-op path", M)
            return _fallback(core, residual, Lc, Ln)
        seg = _segment(M, nq, core.device)
        lc, ln = _weights(Lc, Ln)
        q, ba, r1 = seg(core.contiguous(), residual.contiguous(), lc, ln)
        if not _is_gdn(Ln):
            ba = core.new_empty((M, 0))
        return q, ba, r1
    return _fallback(core, residual, Lc, Ln)


def _seg_fake(core, residual, model, cur, nxt):
    Ln = _layer(model, nxt)
    nq = _in_lin(Ln).output_size_per_partition
    nba = Ln.linear_attn.in_proj_ba.output_size_per_partition if _is_gdn(Ln) else 0
    return core.new_empty((core.shape[0], nq)), core.new_empty((core.shape[0], nba)), torch.empty_like(residual)


def warm(model_idx: int, device, max_m=None) -> bool:
    """compile + load the segment kernels and build every M's schedule/workspaces, outside capture; every kernel runs
    once and is checked for a timeout (co-residency). Returns False (and turns K4 off for this process) on a failure."""
    m = _MODELS[model_idx]
    pairs = {}
    for i in range(len(m.layers) - 1):
        pairs.setdefault(_in_lin(m.layers[i + 1]).output_size_per_partition, (m.layers[i], m.layers[i + 1]))
    for M in range(1, (max_m or _max_m()) + 1):
        for nq, (Lc, Ln) in pairs.items():
            if _k4_ready(M, Lc, Ln):
                seg = _segment(M, nq, device)
                lc, ln = _weights(Lc, Ln)
                seg(torch.zeros(M, 6144, dtype=torch.bfloat16, device=device),
                    torch.zeros(M, 5120, dtype=torch.bfloat16, device=device), lc, ln)
                torch.cuda.synchronize()
                if seg.failed():
                    seg.reset_after_failure()
                    _STATE["disabled"] = True
                    _log().error("horizon_kernels K4: segment kernel M=%d timed out during warm-up (CTAs not "
                                 "co-resident?): K4 OFF for this process, production's per-op path is used", M)
                    return False
                if not _watch(seg):
                    return False
    return True


# ---------------------------------------------------------------------------------------------------------------
# model forward: layer L's attention core -> k4_segment -> layer L+1's attention core
def _gdn_core(attn, mixed_qkvz, ba):
    """QwenGatedDeltaNetAttention.forward_cuda parts 2 + 3 without out_proj: returns the out_proj input."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name
    num_tokens = mixed_qkvz.size(0)
    core = torch.zeros((num_tokens, attn.num_v_heads // attn.tp_size, attn.head_v_dim), dtype=mixed_qkvz.dtype,
                       device=mixed_qkvz.device)
    if (attn.enable_fused_gdn_decode and mixed_qkvz.dtype == torch.bfloat16
            and attn.norm.weight.dtype in (torch.bfloat16, torch.float32)):
        torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(mixed_qkvz, ba, core,
                                                                 layer_name=_encode_layer_name(attn.prefix))
        return core.flatten(-2)
    assert not attn.gqa_interleaved_layout
    qkv_size = (attn.key_dim * 2 + attn.value_dim) // attn.tp_size
    z_size = attn.value_dim // attn.tp_size
    mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
    z = z.reshape(z.size(0), -1, attn.head_v_dim)
    b, a = attn.split_ba(ba)
    torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, b.contiguous(), a.contiguous(), core,
                                           layer_name=_encode_layer_name(attn.prefix))
    z_shape = z.shape
    c = attn.norm(core.reshape(-1, core.shape[-1]), z.reshape(-1, z.shape[-1]))
    return c.reshape(z_shape).flatten(-2)


def _attn_core(attn, qkv, positions):
    """Qwen3NextAttention.forward without qkv_proj / o_proj."""
    q, k, v, gate = attn._project_qkv_gate(qkv, positions)
    out = attn.attn(q, k, v)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return out


def _core_from(layer, proj, ba, positions):
    return _gdn_core(layer.linear_attn, proj, ba) if _is_gdn(layer) else _attn_core(layer.self_attn, proj, positions)


def _project(layer, h):
    if _is_gdn(layer):
        return layer.linear_attn.in_proj_qkvz(h)[0], layer.linear_attn.in_proj_ba(h)[0]
    return layer.self_attn.qkv_proj(h)[0], None


def _supported(model) -> bool:
    try:
        from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
        if get_tensor_model_parallel_world_size() != 1 or not get_pp_group().is_first_rank \
                or not get_pp_group().is_last_rank:
            return False
    except Exception:  # noqa: BLE001
        return False
    cfg = model.config
    if getattr(cfg, "model_type", "") != "qwen3_5_text" or getattr(model, "use_sequence_parallel", False):
        return False
    for layer in model.layers:
        if getattr(layer, "layer_scale", False) or getattr(layer, "use_attn_reduce_scatter_for_moe", False):
            return False
    return True


def _vllm_ok() -> bool:
    """k4seg copies Qwen3_5Model.forward and the GDN core split of vLLM 0.30. Not ported to 0.31/0.32 (the GDN layer, its
    metadata builder and the V2 runner output class changed): refuse there unless HORIZON_K4_FORCE_VERSION=1."""
    if os.environ.get("HORIZON_K4_FORCE_VERSION", "0") == "1":
        return True
    try:
        from importlib.metadata import version
        v = version("vllm").split("+")[0]
    except Exception:  # noqa: BLE001
        return True
    return v.startswith("0.30.") or v.startswith("0.29.")


def install():
    if not enabled():
        return
    if not _vllm_ok():
        _log().warning("horizon_kernels K4: HORIZON_K4_SEGMENT=1 but this vLLM version is not ported for k4seg: K4 stays OFF "
                       "(production's per-op path); set HORIZON_K4_FORCE_VERSION=1 to try anyway")
        return
    if os.path.exists(_marker()):
        _log().warning("horizon_kernels K4: HORIZON_K4_SEGMENT=1 but %s exists (an earlier timeout): K4 stays OFF",
                       _marker())
        return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
    if getattr(Qwen3_5Model.forward, "_horizon_k4", False):
        return
    if not _install_step_check():
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "k4_segment")):
        op = torch.library.custom_op("horizon::k4_segment", mutates_args=())(_seg_impl)
        op.register_fake(_seg_fake)
    if _gdn_enabled() and not hasattr(torch.ops.horizon, "k4_gdn_layer"):
        op = torch.library.custom_op("horizon::k4_gdn_layer", mutates_args=())(_gdn_impl)
        op.register_fake(_gdn_fake)
        from vllm.config.compilation import CompilationConfig
        if "horizon::k4_gdn_layer" not in CompilationConfig._attention_ops:
            CompilationConfig._attention_ops.append("horizon::k4_gdn_layer")
    orig_init, orig_forward = Qwen3_5Model.__init__, Qwen3_5Model.forward

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.horizon_k4_idx = None
        if _supported(self):
            self.horizon_k4_idx = len(_MODELS)
            _MODELS.append(self)
            _log().info("horizon_kernels K4: segment kernel active for %d layer boundaries (M <= %d)",
                        len(self.layers) - 1, _max_m())

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        mi = getattr(self, "horizon_k4_idx", None)
        if mi is None:
            return orig_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds)
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        n = len(self.layers)
        pend = None
        for i in range(n):
            layer = self.layers[i]
            if pend is not None:
                proj, ba, residual = pend
            else:
                if residual is None:
                    residual = hidden
                    h = layer.input_layernorm(hidden)
                else:
                    h, residual = layer.input_layernorm(hidden, residual)
                proj, ba = _project(layer, h)
            if i + 1 < n and _is_gdn(layer) and _gdn_enabled():
                q, ba2, residual = torch.ops.horizon.k4_gdn_layer(proj, ba, residual, mi, i, i + 1)
                pend = (q, ba2 if _is_gdn(self.layers[i + 1]) else None, residual)
                continue
            core = _core_from(layer, proj, ba, positions)
            if i + 1 < n:
                q, ba2, residual = torch.ops.horizon.k4_segment(core, residual, mi, i, i + 1)
                pend = (q, ba2 if _is_gdn(self.layers[i + 1]) else None, residual)
            else:
                hidden, _ = _out_lin(layer)(core)
                hidden, residual = layer.post_attention_layernorm(hidden, residual)
                hidden = layer.mlp(hidden)
        hidden, _ = self.norm(hidden, residual)
        return hidden

    __init__._horizon_k4 = True
    forward._horizon_k4 = True
    Qwen3_5Model.__init__ = __init__
    Qwen3_5Model.forward = forward
