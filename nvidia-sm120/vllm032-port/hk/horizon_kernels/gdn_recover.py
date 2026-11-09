# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels G2 (27-09-2026, ops track): one committed GDN state per decode step for Qwen3.5/3.8 (RecoverSSM).
#
# HORIZON_GDN_RECOVER=1 (default 0; needs the V2 model runner, which production uses, and speculative decoding).
# vLLM's speculative verify writes the GDN recurrent state after EVERY token of the window into its own mamba-cache slot
# (1 + drafts slots per request and layer, 3.1 MB each at Qwen3.8's 48 heads x 128 x 128 fp32), so that the next step
# can start from the last accepted token. With this switch:
#   * each request keeps ONE checkpoint slot (MambaSpec.num_speculative_blocks = 0, the mechanism vLLM 0.29 ships for
#     Kimi-K3 KDA as cache_config.use_kda_recoverssm): the mamba cache needs 3 fewer blocks per request at 3 drafts;
#   * the verify (tilelang.contrib.horizon_ops.gdn_recover.gdn_verify_kernel) reads the checkpoint and writes the same
#     outputs as G1, but no state; it saves the per-token update inputs (delta, normalised k, decay: fp32) instead;
#   * after sampling, the V2 runner's RecoverSSMState.commit_step calls this module's commit once per step: vLLM's
#     commit-plan and conv-compaction Triton kernels (bookkeeping and a copy, shared with Kimi-K3) and our
#     gdn_commit_kernel, which folds the accepted tokens into every layer's checkpoint in one launch with the same fp32
#     instructions the verify used (bit-identical to G1's per-token states).
# Pure speculative batches run the verify with the fused gated norm (like vLLM's fused CUDA path); speculative decodes
# mixed with prefills (vLLM: FLA's Triton recurrent kernel + RMSNormGated) run the verify without the norm.
# Off: HORIZON_GDN_RECOVER=0 and restart (the KV-cache layout changes with the switch, so a restart is required).
import os
from dataclasses import dataclass, field, fields

import torch

_LAYERS: dict = {}      # layer prefix -> index into the per-device buffers
_BUF: dict = {}         # device -> dict(corr, kn, dbuf, ...)
_BY_ALOG: dict = {}     # A_log data_ptr -> layer prefix


def enabled() -> bool:
    return os.environ.get("HORIZON_GDN_RECOVER", "0") == "1"


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _spec_tokens(vllm_config) -> int:
    sc = getattr(vllm_config, "speculative_config", None)
    return int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 0


# ------------------------------------------------------------------------------------------------ config / KV spec
def _install_flag():
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
    if getattr(MambaBase.get_kv_cache_spec, "_horizon", False):
        return
    orig_spec = MambaBase.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        if isinstance(self, QwenGatedDeltaNetAttention) and _spec_tokens(vllm_config) > 0:
            vllm_config.cache_config.use_kda_recoverssm = True
        return orig_spec(self, vllm_config)

    get_kv_cache_spec._horizon = True
    MambaBase.get_kv_cache_spec = get_kv_cache_spec

    orig_init = MambaHybridModelState.__init__

    def init(self, vllm_config, *a, **kw):
        if _spec_tokens(vllm_config) > 0:
            vllm_config.cache_config.use_kda_recoverssm = True
        return orig_init(self, vllm_config, *a, **kw)

    MambaHybridModelState.__init__ = init


# ------------------------------------------------------------------------------------------------ metadata + commit
def _metadata_cls():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata, RecoverSSMPostprocessMetadata

    @dataclass
    class HorizonGDNRecoverMetadata(GDNAttentionMetadata, RecoverSSMMetadata):
        hz_commit: dict | None = field(default=None, repr=False, compare=False)
        hz_ctx: object = field(default=None, repr=False, compare=False)

        def commit_recoverssm_state(self, num_accepted_tokens):
            c = self.hz_commit
            if c is None:
                return None
            n = self.num_spec_decodes
            self.hz_ctx.commit(num_accepted_tokens, c["state_indices"][:n, 0], c["query_start_loc"][: n + 1],
                               c["request_indices"], c.get("block_table"), c.get("num_computed_tokens"),
                               c.get("block_size"))
            if c.get("block_table") is None:
                return None
            return RecoverSSMPostprocessMetadata(num_spec_decodes=n, request_indices=c["request_indices"],
                                                 block_table=c["block_table"],
                                                 num_computed_tokens=c["num_computed_tokens"],
                                                 block_size=c["block_size"])

    return HorizonGDNRecoverMetadata


class _CommitContext:
    """per metadata builder (one GDN KV-cache group): layers, address tables, plan buffers"""

    def __init__(self, layers, spec_query_len, max_num_reqs, buf):
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        dev = layers[0].kv_cache[1].device
        convs = [l.kv_cache[0] if is_conv_state_dim_first() else l.kv_cache[0].transpose(-1, -2) for l in layers]
        states = [l.kv_cache[1] for l in layers]
        s0 = states[0]
        assert all(s.stride() == s0.stride() and s.shape[1:] == s0.shape[1:] for s in states), "GDN state layouts differ"
        self.HV = s0.shape[1]
        self.H = layers[0].num_k_heads // layers[0].tp_size
        self.slot_stride = s0.stride(0)
        self.state_addrs = torch.tensor([s.data_ptr() for s in states], dtype=torch.int64, device=dev)
        self.convs = convs
        self.conv_addrs = torch.tensor([c.data_ptr() for c in convs], dtype=torch.int64, device=dev)
        self.conv_block_strides = torch.tensor([c.stride(0) for c in convs], dtype=torch.int64, device=dev)
        self.conv_dim_strides = torch.tensor([c.stride(1) for c in convs], dtype=torch.int64, device=dev)
        self.conv_tok_strides = torch.tensor([c.stride(2) for c in convs], dtype=torch.int64, device=dev)
        self.conv_history_len = convs[0].shape[2] - spec_query_len + 1
        self.spec_query_len = spec_query_len
        self.commit_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=dev)
        self.final_idx = torch.zeros(max_num_reqs, dtype=torch.int32, device=dev)
        self.bnd_idx = torch.zeros(max_num_reqs, dtype=torch.int32, device=dev)
        self.bnd_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=dev)
        self.buf = buf

    def commit(self, num_accepted, state_indices, query_start_loc, request_indices, block_table, num_computed,
               block_size):
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel, _prepare_commit_plan_kernel
        from vllm.triton_utils import triton
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
        from tilelang.contrib.horizon_ops import gdn_recover as gr
        batch = state_indices.shape[0]
        if batch == 0:
            return
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted, request_indices, state_indices, query_start_loc, block_table, num_computed,
            self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len, NULL_BLOCK_ID, block_size or 1,
            block_table.shape[1] if block_table is not None else 1, num_accepted.stride(0),
            request_indices.stride(0) if request_indices is not None else 0, state_indices.stride(0),
            query_start_loc.stride(0), bt_stride[0], bt_stride[1],
            0 if num_computed is None else num_computed.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len, num_warps=1)
        conv_dim = self.convs[0].shape[1]
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, len(self.convs))](
            self.convs[0], self.conv_addrs, self.conv_block_strides, self.conv_dim_strides, self.conv_tok_strides,
            state_indices, self.commit_lens, self.final_idx, self.bnd_idx, self.bnd_len, NULL_BLOCK_ID, conv_dim,
            self.conv_history_len, state_indices.stride(0), BLOCK_D=256,
            BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len), ALIGN_MODE=block_table is not None,
            num_warps=4)
        b = self.buf
        gr.commit(self.state_addrs, state_indices.contiguous(), self.final_idx[:batch], self.bnd_idx[:batch],
                  self.bnd_len[:batch], self.commit_lens[:batch], b["corr"], b["kn"], b["dbuf"], self.HV, self.H,
                  self.slot_stride)


def _buffers(device, n_layers, max_reqs, HV, H, tb):
    b = _BUF.get(device)
    if b is None:
        b = _BUF[device] = {
            "corr": torch.zeros(n_layers, max_reqs, HV, tb, 128, dtype=torch.float32, device=device),
            "kn": torch.zeros(n_layers, max_reqs, H, tb, 128, dtype=torch.float32, device=device),
            "dbuf": torch.zeros(n_layers, max_reqs, HV, tb, dtype=torch.float32, device=device),
        }
        mb = sum(t.numel() * 4 for t in b.values()) / 2 ** 20
        _log().info("horizon_kernels G2: GDN RecoverSSM active (%d layers, %d requests, %.0f MiB side buffers)",
                    n_layers, max_reqs, mb)
    return b


def _install_builder():
    from vllm.v1.attention.backends import gdn_attn
    if getattr(gdn_attn.GDNAttentionBackend.get_builder_cls, "_horizon", False):
        return
    Meta = _metadata_cls()
    Base = gdn_attn.GDNAttentionMetadataBuilder

    class HorizonGDNRecoverBuilder(Base):
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            # GDNAttentionMetadataBuilder overrides __init__ without retaining
            # layer_names in vLLM 0.29. Own an immutable copy for commit context.
            self.layer_names = tuple(layer_names)
            self.hz_recover = bool(vllm_config.cache_config.use_kda_recoverssm) and self.num_spec > 0
            self.hz_ctx = None
            if self.hz_recover:
                tc = vllm_config.model_config.hf_text_config
                HV, H = tc.linear_num_value_heads, tc.linear_num_key_heads
                n_gdn = sum(1 for t in getattr(tc, "layer_types", []) if t == "linear_attention") or len(layer_names)
                self.hz_max_reqs = vllm_config.scheduler_config.max_num_seqs
                # the hybrid KV-cache manager splits the GDN layers into several groups (one builder each): global,
                # consecutive buffer indices per group, so a group's commit sees one contiguous slice
                self.hz_l0 = len(_LAYERS)
                for n in layer_names:
                    _LAYERS.setdefault(n, len(_LAYERS))
                full = _buffers(device, max(n_gdn, len(_LAYERS)), self.hz_max_reqs, HV, H, self.num_spec + 1)
                sl = slice(self.hz_l0, self.hz_l0 + len(layer_names))
                self.hz_buf = {k: v[sl] for k, v in full.items()}
                self.hz_ones = torch.ones(max(self.hz_max_reqs, self.decode_cudagraph_max_bs) + 1,
                                          dtype=torch.int32, device=device)

        def _ctx(self):
            fc = self.vllm_config.compilation_config.static_forward_context
            if self.hz_ctx is not None and fc[self.layer_names[0]].kv_cache[1].data_ptr() != self.hz_ctx_ptr:
                self.hz_ctx = None           # the KV cache was (re)allocated after the context was built
            if self.hz_ctx is None:
                layers = [fc[n] for n in self.layer_names]
                self.hz_ctx_ptr = layers[0].kv_cache[1].data_ptr()
                for n, l in zip(self.layer_names, layers):
                    _BY_ALOG[l.A_log.data_ptr()] = n
                self.hz_ctx = _CommitContext(layers, self.num_spec + 1, self.hz_max_reqs, self.hz_buf)
            return self.hz_ctx

        def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
                  num_decode_draft_tokens_cpu=None, fast_build=False):
            if not self.hz_recover:
                return super().build(common_prefix_len, common_attn_metadata, num_accepted_tokens,
                                     num_decode_draft_tokens_cpu, fast_build)
            m = common_attn_metadata
            if num_accepted_tokens is not None:
                # the previous step's commit folded the accepted tokens into the checkpoint and compacted the conv
                # window: every request starts at offset 0 (vLLM's "neutral" num_accepted_tokens = 1)
                num_accepted_tokens = self.hz_ones[: num_accepted_tokens.shape[0]]
            md = super().build(common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
            out = Meta(**{f.name: getattr(md, f.name) for f in fields(md)})
            si = out.spec_state_indices_tensor
            if si is not None and si.shape[1] != self.num_spec + 1:
                # vLLM's GDN code reads the window length from this tensor's width (causal_conv1d_update's
                # max_query_len): keep 1 + drafts columns, all = the checkpoint slot (our kernels read column 0)
                out.spec_state_indices_tensor = si[:, :1].expand(-1, self.num_spec + 1).contiguous()
            if md.num_spec_decodes > 0 and md.spec_state_indices_tensor is not None:
                self._ctx()
                req_idx = None
                if md.num_prefills > 0 or md.num_decodes > 0:
                    mask = (num_decode_draft_tokens_cpu >= 0).nonzero(as_tuple=True)[0].to(torch.int32)
                    req_idx = mask.to(m.query_start_loc.device, non_blocking=True)
                c = {"state_indices": md.spec_state_indices_tensor, "query_start_loc": md.spec_query_start_loc,
                     "request_indices": req_idx}
                if self.kv_cache_spec.mamba_cache_mode == "align":
                    c.update(block_table=m.block_table_tensor, num_computed_tokens=m.compute_num_computed_tokens(),
                             block_size=self.kv_cache_spec.block_size)
                out.hz_commit = c
                out.hz_ctx = self.hz_ctx
            return out

    orig = gdn_attn.GDNAttentionBackend.get_builder_cls

    def get_builder_cls():
        return HorizonGDNRecoverBuilder

    get_builder_cls._horizon = True
    gdn_attn.GDNAttentionBackend.get_builder_cls = staticmethod(get_builder_cls)
    _ORIG_BUILDER["cls"] = orig


_ORIG_BUILDER: dict = {}


# ------------------------------------------------------------------------------------------------ forward hooks
def _layer_buf(prefix):
    i = _LAYERS[prefix]
    b = next(iter(_BUF.values()))
    return b["corr"][i], b["kn"][i], b["dbuf"][i]


def _install_forward():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
    Cls = q.QwenGatedDeltaNetAttention
    if getattr(Cls._forward_core_decode_spec_post_conv_fused_norm, "_horizon", False):
        return
    orig_fused = Cls._forward_core_decode_spec_post_conv_fused_norm

    def fused(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):
        if self.prefix not in _LAYERS or self.kv_cache[1].dtype != torch.float32:
            return orig_fused(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata)
        from tilelang.contrib.horizon_ops import gdn_recover as gr
        n = attn_metadata.num_spec_decodes
        corr, kn, dbuf = _layer_buf(self.prefix)
        gr.verify(mixed_qkv, a, b, self.A_log, self.dt_bias, attn_metadata.spec_state_indices_tensor[:n],
                  attn_metadata.spec_query_start_loc[: n + 1], attn_metadata.num_accepted_tokens[:n],
                  self.kv_cache[1], output_gate, self.norm.weight, core_attn_out, corr, kn, dbuf,
                  self.head_k_dim ** -0.5, self.layer_norm_epsilon, self.norm.activation)

    fused._horizon = True
    Cls._forward_core_decode_spec_post_conv_fused_norm = fused

    orig_rec = q.fused_sigmoid_gating_delta_rule_update

    def rec(A_log, a, b, dt_bias, q, k, v, initial_state, inplace_final_state=True, cu_seqlens=None,
            ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False, **kw):
        prefix = _BY_ALOG.get(A_log.data_ptr())
        if (prefix is None or num_accepted_tokens is None or ssm_state_indices is None
                or ssm_state_indices.dim() != 2 or not use_qk_l2norm_in_kernel or initial_state.dtype != torch.float32):
            return orig_rec(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=initial_state,
                            inplace_final_state=inplace_final_state, cu_seqlens=cu_seqlens,
                            ssm_state_indices=ssm_state_indices, num_accepted_tokens=num_accepted_tokens,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        # speculative decodes inside a mixed batch: our verify without the norm (vLLM applies RMSNormGated after)
        from tilelang.contrib.horizon_ops import gdn_recover as gr
        L = q.shape[1]
        HV = v.shape[2]
        mixed = torch.cat([q.reshape(L, -1), k.reshape(L, -1), v.reshape(L, -1)], dim=1)
        out = torch.empty(L, HV, v.shape[3], dtype=q.dtype, device=q.device)
        n = cu_seqlens.shape[0] - 1
        corr, kn, dbuf = _layer_buf(prefix)
        gr.verify(mixed, a.contiguous(), b.contiguous(), A_log, dt_bias, ssm_state_indices[:n], cu_seqlens,
                  num_accepted_tokens[:n], initial_state, out, torch.ones(v.shape[-1], dtype=torch.float32,
                                                                           device=q.device),
                  out, corr, kn, dbuf, q.shape[-1] ** -0.5, 1e-6, "silu", with_norm=False)
        return out.unsqueeze(0), None

    q.fused_sigmoid_gating_delta_rule_update = rec


def install():
    if not enabled():
        return
    _install_flag()
    _install_builder()
    _install_forward()
    _log().info("horizon_kernels G2: GDN RecoverSSM hooks installed (one checkpoint slot per request)")
