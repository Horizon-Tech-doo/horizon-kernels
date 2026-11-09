# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels MoE (27-09-2026): Qwen3.5/3.6-MoE (Qwen3NextSparseMoeBlock) on the RTX PRO 6000, switched by
# HORIZON_MOE=1 (default 0). Every feature has its own switch (default 0) so each one is A/B-tested on its own; any
# layer or batch a feature does not handle stays on stock vLLM.
#
# HORIZON_MOE_SPEC_BACKEND=1 (step 0b): vLLM 0.29's V2 model runner loads the MTP drafter with the target's
#   kernel_config, so `speculative_config.moe_backend` is ignored (only the old V1 proposer applies it,
#   v1/spec_decode/llm_base_proposer.py:_create_draft_vllm_config) and `--moe-backend marlin` for the NVFP4 experts
#   reaches the drafter's BF16 experts ("moe_backend='marlin' is not supported for unquantized MoE"). This wraps the V2
#   loader (v1/worker/gpu/spec_decode/eagle/utils.py:load_eagle_model) to apply the drafter's moe_backend exactly as V1
#   does. Loader-only: the served model is unchanged.
#
# HORIZON_MOE_FSE=1 (step 1): the shared expert runs as routed expert #256 inside the same Marlin-MoE call (9 slots per
#   token; slot 9 has weight sigmoid(shared_expert_gate(x))). vLLM 0.29 already has the loader/runner machinery for
#   "fused shared experts" (built for ROCm AITER and switched off elsewhere): the checkpoint's mlp.shared_expert.* is
#   loaded as mlp.experts.256.*, the MoE layer allocates 257 experts, the runner gets the shared-expert gate. What is
#   missing on CUDA and added here:
#     - enabling it (resolve_layer_fused_shared_expert) for ModelOpt W4A16-NVFP4 experts + shared expert only (the
#       checkpoint's gate/up weight_scale_2 are tied for all 40 shared experts, so vLLM's fused-w13 scale is exact);
#     - a router: softmax top-8 + renormalise over the 256 routed logits (the stock topk_softmax op, as before) plus
#       slot 9 = (256, sigmoid(shared logit)) (the stock CUDA router would take the top-8 of all 257 logits);
#     - the router and shared-expert gates as ONE (257 x 2048) linear layer (the runner's own "fused gate" used a
#       separate F.linear; here the concatenated weight lives in the router gate layer, so HORIZON_MOE_GATE can serve
#       it);
#     - global_num_experts = 257 for Marlin's moe_align_block_size (vLLM passes 256, which would drop slot 9);
#     - the loader rename without vLLM's per-shard split (ModelOpt's 0-d weight_scale_2 / input_scale cannot be
#       sliced; with one shared expert the split is the identity anyway).
#   Numerics: the shared expert's product is weighted in fp32 inside Marlin's down-projection epilogue and summed with
#   the routed slots by moe_sum, instead of a bf16 multiply + add after it: exact-class, not bit-identical to stock.
#
# HORIZON_MOE_GATE=1 (step 2; needs HORIZON_MOE_FSE=1): the fused 257-row gate (zero-padded to 260 rows: skinny needs
#   N % 4 == 0) runs on tilelang.contrib.horizon_fused.skinny_bf16 (K5a) for M <= skinny.SUPPORTS_M, cuBLAS above.
#   The logits tensor is (M, 260); the router reads columns 0..256 only.
#
# HORIZON_MOE_ROUTE=1 (step 3; needs HORIZON_MOE_FSE=1): the router runs as one TileLang kernel
#   (tilelang.contrib.horizon_moe.route): softmax, top-8, renormalise, sigmoid slot, in one launch instead of
#   copy + topk_softmax + sigmoid + 2 concatenations; bit-identical to the torch router of HORIZON_MOE_FSE.
#   HORIZON_MOE_ROUTE_ALIGN (default 1): for M <= 128 the same launch (one CTA) also computes Marlin-MoE's block
#   alignment (sorted_token_ids / expert_ids / num_tokens_post_padded), which Marlin's moe_align_block_size call then
#   reuses (2 more launches saved per layer).
#
# HORIZON_MOE_GROUPED=1 (step 4; needs FSE + ROUTE; A/B only until the race passes): for M <= HORIZON_MOE_GROUPED_MAX_M
#   (default 128) the experts run on tilelang.contrib.horizon_moe.grouped_w4a16 (our grouped W4A16 NVFP4 GEMMs: up with
#   the SiLU*up epilogue, down with the slot weight, then vLLM's moe_sum) on the route kernel's 16-row alignment. Our
#   regfrag-layout copy of the experts is built BEFORE Marlin's repack; Marlin stays for larger M (prefill), so this
#   switch costs one more expert copy (~18 GB) until Marlin is dropped.
import os
import threading

import torch

_TL = threading.local()   # depth > 0 while a Qwen3NextSparseMoeBlock is being constructed
_GATES: list = []         # gate layers served by HORIZON_MOE_GATE (index = custom-op argument)
_GATE_MAX_M = [0]         # skinny for 1 <= M <= this (HORIZON_MOE_GATE_MAX_M, default skinny.SUPPORTS_M), cuBLAS above
_INSTALLED = False
_WARM_ROUTE: list = []


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def on(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


# ---------------------------------------------------------------------------------------------------------------
# step 0b: drafter moe_backend on the V2 runner
def _install_spec_moe_backend():
    import importlib
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eu
    orig = eu.load_eagle_model
    if getattr(orig, "_horizon", False):
        return

    def load_eagle_model(target_model, vllm_config):
        sc = vllm_config.speculative_config
        mb = getattr(sc, "moe_backend", None) if sc is not None else None
        if mb is not None and vllm_config.kernel_config.moe_backend != mb:
            from vllm.config import replace
            _log().info("horizon_kernels MoE: drafter moe_backend=%s (target %s), as the V1 proposer does",
                        mb, vllm_config.kernel_config.moe_backend)
            vllm_config = replace(vllm_config, kernel_config=replace(vllm_config.kernel_config, moe_backend=mb))
        return orig(target_model, vllm_config)

    load_eagle_model._horizon = True
    eu.load_eagle_model = load_eagle_model
    for name in ("mtp.speculator", "eagle.speculator", "multi_module_mtp.speculator"):
        try:
            m = importlib.import_module(f"vllm.v1.worker.gpu.spec_decode.{name}")
        except Exception:  # noqa: BLE001
            continue
        if getattr(m, "load_eagle_model", None) is orig:
            m.load_eagle_model = load_eagle_model


# ---------------------------------------------------------------------------------------------------------------
# step 1: shared expert as routed expert #256
def _fse_wanted(quant_config, prefix: str, shared_name: str) -> bool:
    if prefix.startswith("mtp") or ".mtp." in prefix:
        return False
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(quant_config, ModelOptMixedPrecisionConfig):
        return False
    algos = {quant_config._resolve_quant_algo(f"{prefix}.experts")}
    algos |= {quant_config._resolve_quant_algo(f"{prefix}.{shared_name}.{p}") for p in ("gate_proj", "up_proj",
                                                                                          "down_proj")}
    return algos == {"W4A16_NVFP4"}


def _rename_shared(orig):
    def maybe_fuse_shared_experts(weights, *, n_routed_experts, n_shared_experts, ckpt_prefix="mlp.shared_experts",
                                  enabled=None):
        if not enabled or n_shared_experts != 1 or not on("HORIZON_MOE_FSE"):
            yield from orig(weights, n_routed_experts=n_routed_experts, n_shared_experts=n_shared_experts,
                            ckpt_prefix=ckpt_prefix, enabled=enabled)
            return
        prefix = f"{ckpt_prefix}."
        for name, w in weights:
            yield (name.replace(prefix, f"mlp.experts.{n_routed_experts}.") if prefix in name else name), w
    maybe_fuse_shared_experts._horizon = True
    return maybe_fuse_shared_experts


_ALIGN = [None]   # (topk_ids, block_size, num_experts, (sorted_ids, expert_ids, num_tokens_post_padded)) of the last route
_ALIGN_STATS = {"reused": 0, "fallback": 0}
_ALIGN16 = [None]  # (topk_ids, (sorted_ids, expert_ids, ntp)) with 16-row blocks, for the grouped GEMMs
_GROUPED_MAX_M = [int(os.environ.get("HORIZON_MOE_GROUPED_MAX_M", "128"))]


def _marlin_block_size(m, topk, e):
    """block_size_m exactly as vLLM 0.29's fused_marlin_moe picks it (bf16 activations, global_num_experts = E)."""
    for bs in (8, 16, 32, 48, 64):
        if m * topk / e / bs < 0.9:
            break
    return bs


def _install_align_reuse():
    """Marlin-MoE calls moe_align_block_size(topk_ids, ...) on the ids our route kernel just wrote: return the alignment
    the same kernel computed (same contract), instead of 2 more launches. Any other call (other layers/models, other
    ids tensor, EP, padding) goes to vLLM's own op."""
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe as mm
    orig = mm.moe_align_block_size
    if getattr(orig, "_horizon", False):
        return

    def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False,
                             ignore_invalid_experts=False):
        st = _ALIGN[0]
        if (st is not None and st[0] is topk_ids and st[1] == block_size and st[2] == num_experts
                and expert_map is None and not pad_sorted_ids):
            _ALIGN[0] = None
            if _ALIGN_STATS["reused"] == 0:
                _log().info("horizon_kernels MoE: Marlin-MoE uses the route kernel's block alignment")
            _ALIGN_STATS["reused"] += 1
            return st[3]
        if st is not None and _ALIGN_STATS["fallback"] == 0:
            _log().warning("horizon_kernels MoE: route alignment not reused (ids tensor or sizes differ); stock align")
        _ALIGN_STATS["fallback"] += st is not None
        return orig(topk_ids, block_size, num_experts, expert_map, pad_sorted_ids=pad_sorted_ids,
                    ignore_invalid_experts=ignore_invalid_experts)

    moe_align_block_size._horizon = True
    mm.moe_align_block_size = moe_align_block_size


def _prepare_grouped(layer):
    """Our regfrag-layout copy of all experts (w13 with gate/up interleaved) from the checkpoint tensors, before
    Marlin's repack; then compile + load every grouped-kernel config the dispatcher can pick (outside capture)."""
    from tilelang.contrib.horizon_moe import grouped_w4a16 as gw
    w13 = layer.w13_weight.data.view(torch.uint8)
    w2 = layer.w2_weight.data.view(torch.uint8)
    s13 = layer.w13_weight_scale.data
    s2 = layer.w2_weight_scale.data
    if not (w13.dim() == 3 and s13.dtype == torch.float8_e4m3fn and s13.shape[2] * 8 == w13.shape[2]):
        _log().warning_once("horizon_kernels MoE: unexpected expert tensor layout %s / %s: grouped path off",
                            tuple(w13.shape), tuple(s13.shape))
        return
    a13 = layer.w13_weight_scale_2.data.float().reshape(w13.shape[0], -1)[:, 0].contiguous()
    a2 = layer.w2_weight_scale_2.data.float().reshape(-1).contiguous()
    layer.horizon_w13 = gw.prepare_experts(w13, s13, a13, interleave=True)
    layer.horizon_w2 = gw.prepare_experts(w2, s2, a2)
    layer.horizon_grouped = True
    E, H = w13.shape[0], w2.shape[1]
    dev = w13.device
    cfgs = sorted({(gw.config("up", layer.horizon_w13.N, layer.horizon_w13.K, m),
                    gw.config("down", layer.horizon_w2.N, layer.horizon_w2.K, m), m)
                   for m in range(1, _GROUPED_MAX_M[0] + 1)}, key=lambda c: c[2])
    seen = set()
    for cu, cd, m in cfgs:
        if (cu, cd) in seen or len(seen) > 64:
            continue
        seen.add((cu, cd))
        ids = torch.arange(m * 9, device=dev, dtype=torch.int32).remainder(E).view(m, 9)
        tw = torch.full((m, 9), 1.0 / 9, device=dev)
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        al = moe_align_block_size(ids, 16, E, None, ignore_invalid_experts=True)
        gw.moe_forward(torch.zeros(m, H, device=dev, dtype=torch.bfloat16), tw, al, layer.horizon_w13,
                       layer.horizon_w2, cfg_up=cu, cfg_down=cd)
    torch.cuda.synchronize()
    _log().info_once("horizon_kernels MoE: grouped W4A16 GEMMs active for M <= %d (%d kernel configs warmed)",
                     _GROUPED_MAX_M[0], len(seen))


class _Router:
    """Built lazily (BaseRouter is imported only once vLLM is up)."""
    cls = None


def _router_cls():
    if _Router.cls is not None:
        return _Router.cls
    from vllm.model_executor.layers.fused_moe.config import get_routing_method_type
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    class HorizonSharedGateRouter(BaseRouter):
        """softmax top-k (+ renormalise) over the first E logits, then slot k+1 = (E, sigmoid(logit E)).
        router_logits: (M, >= E + 1); columns beyond E + 1 (zero padding of the fused gate) are ignored."""

        def __init__(self, top_k, global_num_experts, eplb_state=None, renormalize=True):
            super().__init__(top_k=top_k, global_num_experts=global_num_experts, eplb_state=eplb_state)
            self.renormalize = renormalize
            self.scoring_func = "softmax"
            self.use_kernel = on("HORIZON_MOE_ROUTE")
            self.use_align = self.use_kernel and os.environ.get("HORIZON_MOE_ROUTE_ALIGN", "1") == "1"
            self.use_grouped = self.use_kernel and on("HORIZON_MOE_GROUPED")

        @property
        def routing_method_type(self):
            return get_routing_method_type(scoring_func="softmax", top_k=self.top_k, renormalize=self.renormalize,
                                           num_expert_group=None, has_e_score_bias=False)

        def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
            n = self.global_num_experts
            if self.use_kernel and router_logits.dtype == torch.bfloat16 and indices_type in (None, torch.int32):
                from tilelang.contrib.horizon_moe import route
                m = router_logits.shape[0]
                if self.use_grouped and 1 <= m <= min(_GROUPED_MAX_M[0], route.ALIGN_MAX_M):
                    w, ids, al = route.route_align(router_logits, n, self.top_k, self.renormalize, 16)
                    _ALIGN16[0] = (ids, al)            # consumed by m_apply's grouped path (same ids tensor)
                    return w, ids
                if self.use_align and 1 <= m <= route.ALIGN_MAX_M:
                    bs = _marlin_block_size(m, self.top_k + 1, n + 1)
                    w, ids, al = route.route_align(router_logits, n, self.top_k, self.renormalize, bs)
                    _ALIGN[0] = (ids, bs, n + 1, al)   # consumed by Marlin-MoE's moe_align_block_size (same ids)
                    return w, ids
                return route.route_shared(router_logits, n, self.top_k, self.renormalize)
            w, ids, _ = fused_topk(hidden_states, router_logits[:, :n].contiguous(), self.top_k, self.renormalize,
                                   indices_type, "softmax")
            # HF/stock: sigmoid(gate) on the bf16 gate output, rounded to bf16, then multiplied in
            sw = torch.sigmoid(router_logits[:, n:n + 1]).float()
            sid = torch.full_like(ids[:, :1], n)
            return torch.cat([w, sw], 1), torch.cat([ids, sid], 1)

    _Router.cls = HorizonSharedGateRouter
    return HorizonSharedGateRouter


def _install_fse():
    from vllm.model_executor.layers.fused_moe import layer as moe_layer
    from vllm.model_executor.layers.fused_moe.runner import moe_runner
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import modelopt as mo
    from vllm.model_executor.models import qwen3_next as q3n
    from vllm._aiter_ops import rocm_aiter_ops
    try:
        from vllm.model_executor.models import qwen3_5 as q35
    except Exception:  # noqa: BLE001
        q35 = None

    # (a) enable FSE for ModelOpt W4A16-NVFP4 MoE blocks
    orig_resolve = q3n.resolve_layer_fused_shared_expert

    def resolve_layer_fused_shared_expert(quant_config, prefix, shared_expert_name="shared_experts"):
        if orig_resolve(quant_config, prefix, shared_expert_name):
            return True
        ok = _fse_wanted(quant_config, prefix, shared_expert_name)
        if ok:
            _log().info_once("horizon_kernels MoE: shared expert runs as an extra routed expert inside the Marlin-MoE "
                             "call (HORIZON_MOE_FSE=1)")
        return ok

    q3n.resolve_layer_fused_shared_expert = resolve_layer_fused_shared_expert

    # (b) loader rename
    for mod in (q3n, q35):
        if mod is not None and hasattr(mod, "maybe_fuse_shared_experts") \
                and not getattr(mod.maybe_fuse_shared_experts, "_horizon", False):
            mod.maybe_fuse_shared_experts = _rename_shared(mod.maybe_fuse_shared_experts)

    # (c) our router for FSE layers built inside a Qwen3NextSparseMoeBlock
    blk = q3n.Qwen3NextSparseMoeBlock
    orig_init = blk.__init__

    def block_init(self, *a, **kw):
        _TL.depth = getattr(_TL, "depth", 0) + 1
        try:
            orig_init(self, *a, **kw)
        finally:
            _TL.depth -= 1

    blk.__init__ = block_init
    orig_cr = moe_layer.create_fused_moe_router

    def create_fused_moe_router(*a, **kw):
        if (not a and getattr(_TL, "depth", 0) > 0 and kw.get("num_fused_shared_experts", 0) == 1
                and kw.get("scoring_func", "softmax") == "softmax" and not kw.get("use_grouped_topk", False)
                and kw.get("custom_routing_function") is None and kw.get("e_score_correction_bias") is None
                and kw.get("zero_expert_type") is None and kw.get("hash_indices_table") is None
                and not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()):
            return _router_cls()(kw["top_k"], kw["global_num_experts"], eplb_state=kw.get("eplb_state"),
                                 renormalize=kw.get("renormalize", True))
        return orig_cr(*a, **kw)

    moe_layer.create_fused_moe_router = create_fused_moe_router

    # (d) router gate + shared-expert gate as one linear layer (the gate layer owns the concatenated weight)
    runner_cls = moe_runner.MoERunner
    orig_rinit = runner_cls.__init__

    def runner_init(self, *a, **kw):
        orig_rinit(self, *a, **kw)
        if self._fse_fuse_gate and self.gate is not None and isinstance(self.router, _router_cls()):
            self._fse_fuse_gate = False
            self.gate.horizon_extra_gate = (self.shared_expert_gate,)   # a tuple: not registered as a submodule

    runner_cls.__init__ = runner_init
    orig_upw = UnquantizedLinearMethod.process_weights_after_loading
    orig_uapply = UnquantizedLinearMethod.apply

    def u_pw(self, layer):
        r = orig_upw(self, layer)
        extra = getattr(layer, "horizon_extra_gate", None)
        if extra is not None and getattr(layer, "horizon_gate_n", None) is None:
            w = torch.cat([layer.weight.data] + [m.weight.data.to(layer.weight.dtype) for m in extra], 0)
            n = w.shape[0]
            pad = (-n) % 8
            if pad:   # zero rows: skinny needs N % 4 == 0, the route kernel reads 16-byte-aligned rows; never read
                w = torch.cat([w, w.new_zeros(pad, w.shape[1])], 0)
            layer.weight = torch.nn.Parameter(w.contiguous(), requires_grad=False)
            layer.horizon_gate_n = n
            layer.horizon_gate_idx = None
            if on("HORIZON_MOE_GATE"):
                _register_gate(layer)
        return r

    def u_apply(self, layer, x, bias=None):
        if getattr(layer, "horizon_gate_idx", None) is not None and bias is None:
            return torch.ops.horizon.moe_gate_linear(x, layer.horizon_gate_idx)
        return orig_uapply(self, layer, x, bias)

    UnquantizedLinearMethod.process_weights_after_loading = u_pw
    UnquantizedLinearMethod.apply = u_apply

    # (e) Marlin-MoE must count expert #256: global_num_experts = -1 -> E from the weights (257)
    cls = mo.ModelOptNvFp4FusedMoE
    orig_mpw = cls.process_weights_after_loading
    orig_mapply = cls.apply

    def m_pw(self, layer):
        e = layer.w13_weight.shape[0]
        layer.horizon_fse = e > layer.global_num_experts
        if layer.horizon_fse:
            _log().info_once("horizon_kernels MoE: FSE layer with %d experts (%d routed + shared)", e,
                             layer.global_num_experts)
            if on("HORIZON_MOE_ROUTE") and not _WARM_ROUTE:
                from tilelang.contrib.horizon_moe import route
                ld = layer.global_num_experts + 1
                ld += (-ld) % 8
                # the gate's logits are (M, E+1 padded to a multiple of 8); compile + load outside CUDA-graph capture
                route.warm(layer.global_num_experts, ld, layer.top_k, layer.renormalize, layer.w13_weight.device,
                           block_sizes=(8, 16))
                torch.cuda.synchronize()
                _WARM_ROUTE.append(ld)
            layer.horizon_grouped = False
            if on("HORIZON_MOE_GROUPED") and on("HORIZON_MOE_ROUTE"):
                _prepare_grouped(layer)
        return orig_mpw(self, layer)

    def m_apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if not getattr(layer, "horizon_fse", False):
            return orig_mapply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        if getattr(layer, "horizon_grouped", False) and x.shape[0] <= _GROUPED_MAX_M[0] and x.dtype == torch.bfloat16:
            from tilelang.contrib.horizon_moe import grouped_w4a16 as gw
            import vllm._custom_ops as ops
            st = _ALIGN16[0]
            if st is not None and st[0] is topk_ids:
                al = st[1]
                _ALIGN16[0] = None
            else:
                from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
                al = moe_align_block_size(topk_ids, 16, layer.horizon_w13.E, None, ignore_invalid_experts=True)
            return gw.moe_forward(x.contiguous(), topk_weights, al, layer.horizon_w13, layer.horizon_w2,
                                  moe_sum=lambda y, o: ops.moe_sum(y, o))
        return self.moe_kernel.apply(x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
                                     activation=layer.activation, global_num_experts=-1, expert_map=layer.expert_map,
                                     apply_router_weight_on_input=layer.apply_router_weight_on_input,
                                     shared_experts=shared_experts, shared_experts_input=shared_experts_input)

    cls.process_weights_after_loading = m_pw
    cls.apply = m_apply


# ---------------------------------------------------------------------------------------------------------------
# step 2: the fused gate on K5a skinny
def _gate_impl(x: torch.Tensor, idx: int) -> torch.Tensor:
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    w = _GATES[idx].weight
    k = w.shape[1]
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    if 1 <= m <= _GATE_MAX_M[0] and x2.dtype == torch.bfloat16:
        y = sk.linear(x2.contiguous(), w)
    else:
        y = torch.nn.functional.linear(x2, w)
    return y.reshape(*x.shape[:-1], w.shape[0])


def _gate_fake(x: torch.Tensor, idx: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], _GATES[idx].weight.shape[0]))


def _register_gate(layer):
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    w = layer.weight
    n, k = w.shape
    if not (w.dtype == torch.bfloat16 and w.is_cuda and sk.supported(n, k)):
        _log().warning_once("horizon_kernels MoE: gate %s not supported by skinny, stays on cuBLAS", (n, k))
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "moe_gate_linear")):
        op = torch.library.custom_op("horizon::moe_gate_linear", mutates_args=())(_gate_impl)
        op.register_fake(_gate_fake)
    layer.horizon_gate_idx = len(_GATES)
    _GATES.append(layer)
    # default: skinny's measured crossover vs cuBLAS; larger values keep the gate on the deterministic, batch-invariant
    # skinny kernel for bigger batches too (cuBLAS changes kernels, hence summation order, with M)
    _GATE_MAX_M[0] = int(os.environ.get("HORIZON_MOE_GATE_MAX_M", sk.SUPPORTS_M))
    # compile + load every config the dispatcher can pick, outside any CUDA-graph capture
    for cfg in sk.all_configs(n, k):
        for m in (1, cfg[0], sk.SUPPORTS_M):
            sk.linear(torch.zeros(m, k, device=w.device, dtype=torch.bfloat16), w.data, cfg=cfg)
    torch.cuda.synchronize()
    if len(_GATES) == 1:
        _log().info("horizon_kernels MoE: router + shared-expert gate (%d rows, padded to %d) on skinny BF16 for "
                    "M <= %d", layer.horizon_gate_n, n, _GATE_MAX_M[0])


# ---------------------------------------------------------------------------------------------------------------
def install():
    global _INSTALLED
    if _INSTALLED or not on("HORIZON_MOE"):
        return
    _INSTALLED = True
    feats = []
    if on("HORIZON_MOE_SPEC_BACKEND"):
        _install_spec_moe_backend(); feats.append("spec_backend")
    if on("HORIZON_MOE_FSE"):
        _install_fse(); feats.append("fse")
        if on("HORIZON_MOE_GATE"):
            feats.append("gate")
        if on("HORIZON_MOE_GROUPED"):
            feats.append("grouped")
        if on("HORIZON_MOE_ROUTE"):
            feats.append("route")
            if os.environ.get("HORIZON_MOE_ROUTE_ALIGN", "1") == "1":
                _install_align_reuse(); feats.append("route_align")
    elif on("HORIZON_MOE_GATE") or on("HORIZON_MOE_ROUTE"):
        _log().warning("horizon_kernels MoE: HORIZON_MOE_GATE / HORIZON_MOE_ROUTE need HORIZON_MOE_FSE=1; ignored")
    _log().info("horizon_kernels MoE installed: %s", ",".join(feats) or "nothing (all sub-switches 0)")
