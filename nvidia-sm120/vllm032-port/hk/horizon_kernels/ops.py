# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels ops track (27-09-2026): our own kernels for the non-GEMM ops of the decode step, each behind its own
# default-off switch, installed from register() (vllm_plugin.py) in every vLLM process without touching CUDA.
#
# HORIZON_GDN=1: the GDN (linear attention) decode core of the speculative verify path -- gating, delta-rule recurrent
#   update over the request's tokens, gated RMSNorm -- runs on tilelang.contrib.horizon_ops.gdn_decode instead of
#   vLLM's CUDA kernel torch.ops._C.fused_gdn_decode_post_conv_mtp. Same contract, same fp32 op order and scalar
#   helpers, compiled with --use_fast_math like vLLM's: bit-identical outputs and states (our run notes,
#   "G1"). A head's 128 value rows are spread over 1-4 CTAs (vLLM: one CTA per request x head, 48 CTAs at 1 user).
#   Anything the kernel does not cover (bf16 state, other head dims, odd strides) falls back to vLLM's op.
#
# HORIZON_ATTN=1: full-attention DECODE on tilelang.contrib.horizon_ops.attn_decode instead of FlashInfer's XQA
#   (vLLM's dedicated SM120 decode path): split-KV over the paged FP8 KV cache, splits sized per request inside the
#   kernel (one CUDA graph serves every context length), fixed-order combine (deterministic, batch-invariant). Not
#   bit-identical to XQA (different summation order): quality gate = decode fingerprint divergence + paired eval.
#   Covered: NHD layout, fp8 e4m3 KV, uniform q_len_per_req (1 = drafts, 1 + drafts = verify) with the causal draft
#   mask, no sinks / sliding window, float scales; anything else falls back to XQA.
#
# HORIZON_PREFILL_RS=1: the per-row activation scale of the row-scaled NVFP4 prefill (prefill_rs.quantize_rs: torch
#   abs -> full |x| copy, amax reduce, clamp, two where) in one TileLang kernel (horizon_ops.row_scale), bit-identical.
#
# HORIZON_GDN_CONV=1: the GDN conv1d state update of speculative decodes (vLLM's Triton _causal_conv1d_update_kernel)
#   on horizon_ops.conv1d_update, bit-identical (bf16 tap products, fp32 adds, the same ex2 / div.full SiLU).
#
# HORIZON_QKNORM=1: the full-attention q/k Gemma-RMSNorm + partial mRoPE + gate split (vLLM's Triton
#   _fused_qk_rmsnorm_rope_gate_kernel) on horizon_ops.qk_norm_rope, bit-identical (Triton's reduction tree emulated),
#   through the torch custom op horizon::qk_norm_rope_gate (the call sits inside vLLM's torch.compile region).
#
# HORIZON_GDN_PREFILL=1: the GDN chunked prefill (FLA chunk_gated_delta_rule: 6 Triton kernels) on
#   horizon_ops.gdn_prefill (a WY kernel + one sequential pass per value slice that keeps the state on chip). Not
#   bit-identical to FLA (summation order): long-prompt log-prob gate.
#
# HORIZON_KVW=1: the FP8 KV-cache write (vLLM's reshape_and_cache_flash) on horizon_ops.kv_write, bit-identical
#   (IEEE x / scale, satfinite e4m3), through FlashInferImpl.do_kv_cache_update.
import os

import torch  # noqa: F401  (custom-op schema annotations)

_ORIG = {}
_WARM: set = set()


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _gdn_warm(mixed_qkv, a, b, dt_bias, state, output_gate, norm_weight, act):
    """compile + load every row split the dispatcher can pick for this stride set, outside CUDA-graph capture"""
    import torch
    from tilelang.contrib.horizon_ops import gdn_decode as gd
    key = (mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0), state.stride(0), dt_bias.dtype,
           norm_weight.dtype, act, mixed_qkv.device)
    if key in _WARM:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    HV = state.shape[1]
    dev = mixed_qkv.device
    # one-request dummy launch on the zero path (num_accepted = 0: no state read or write) per row split
    sidx = torch.zeros(1, 4, dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    nacc = torch.zeros(1, dtype=torch.int32, device=dev)
    out = torch.empty(1, HV, 128, dtype=torch.bfloat16, device=dev)
    for vr in (32, 64, 128):
        gd.fused_gdn_decode_post_conv_mtp(mixed_qkv[:1], a[:1], b[:1], torch.zeros(HV, device=dev), dt_bias, sidx, cu,
                                          nacc, state, output_gate[:1], norm_weight, out, 0.1, 1e-6, act, vr=vr)
    torch.cuda.synchronize()
    _WARM.add(key)
    if len(_WARM) == 1:
        _log().info("horizon_kernels ops: GDN decode core on horizon_ops.gdn_decode (HV=%d, strides qkv=%d ab=%d z=%d "
                    "state=%d)", HV, mixed_qkv.stride(0), a.stride(0), output_gate.stride(0), state.stride(0))
    return True


def _install_gdn():
    import torch
    from vllm import _custom_ops as ops
    if getattr(ops.fused_gdn_decode_post_conv_mtp, "_horizon", False):
        return
    orig = ops.fused_gdn_decode_post_conv_mtp
    _ORIG["gdn"] = orig

    def fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
                                       state, output_gate, norm_weight, out=None, scale=128 ** -0.5, norm_eps=1e-5,
                                       output_gate_activation="silu"):
        from tilelang.contrib.horizon_ops import gdn_decode as gd
        ok = (gd.supported(mixed_qkv, state, output_gate, a, b) and A_log.dtype == torch.float32
              and state_indices.shape[1] <= gd.T_MAX and output_gate_activation in ("silu", "sigmoid")
              and dt_bias.dtype in (torch.float32, torch.bfloat16) and norm_weight.dtype in (torch.float32, torch.bfloat16)
              and (out is None or out.is_contiguous()))
        if ok:
            ok = _gdn_warm(mixed_qkv, a, b, dt_bias, state, output_gate, norm_weight, output_gate_activation)
        if not ok:
            _log().warning_once("horizon_kernels ops: GDN decode call not covered, using vLLM's kernel")
            return orig(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
                        output_gate, norm_weight, out, scale, norm_eps, output_gate_activation)
        return gd.fused_gdn_decode_post_conv_mtp(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens,
                                                 num_accepted_tokens, state, output_gate, norm_weight, out, scale,
                                                 norm_eps, output_gate_activation)

    fused_gdn_decode_post_conv_mtp._horizon = True
    ops.fused_gdn_decode_post_conv_mtp = fused_gdn_decode_post_conv_mtp


_ATTN_WARM: set = set()


def _install_attn():
    import torch
    from vllm.v1.attention.backends import flashinfer as fi
    if getattr(fi.flashinfer_xqa_batch_decode_with_kv_cache, "_horizon", False):
        return
    orig = fi.flashinfer_xqa_batch_decode_with_kv_cache
    _ORIG["xqa"] = orig

    def xqa(query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, bmm1_scale=1.0, bmm2_scale=1.0,
            window_left=-1, out=None, sinks=None, kv_layout="NHD", q_len_per_req=1, mask=None, q_cu_seq_lens=None,
            kv_cache_sf=None, **kw):
        # vLLM 0.32 passes kv_cache_sf= (NVFP4 KV block scales, None for fp8 KV): only a None one is ours to take
        ok = (kv_cache_sf is None and isinstance(kv_cache, tuple) and kv_layout in ("NHD", "HND") and sinks is None and window_left == -1
              and q_cu_seq_lens is None and not kw and not torch.is_tensor(bmm1_scale) and not torch.is_tensor(bmm2_scale)
              and query.dtype == torch.bfloat16 and kv_cache[0].dtype == torch.float8_e4m3fn
              and query.shape[-1] == kv_cache[0].shape[-1] and kv_cache[0].stride(-1) == 1)
        kvh = kv_cache[0].shape[1 if kv_layout == "HND" else 2] if ok else 1
        ok = (ok and query.shape[1] % kvh == 0 and (query.shape[1] // kvh) * int(q_len_per_req or 1) <= 32
              and seq_lens.dtype == torch.int32 and block_tables.dtype == torch.int32 and query.is_contiguous()
              and block_tables.is_contiguous() and seq_lens.is_contiguous() and (out is None or out.is_contiguous()))
        if ok:
            from tilelang.contrib.horizon_ops import attn_decode as ad
            key = (query.shape[1], kv_cache[0].shape[1:], kv_cache[0].stride(), kv_layout, int(q_len_per_req or 1),
                   query.device)
            if key not in _ATTN_WARM:
                if torch.cuda.is_current_stream_capturing():
                    ok = False
                else:
                    _ATTN_WARM.add(key)
                    if len(_ATTN_WARM) == 1:
                        _log().info("horizon_kernels ops: attention decode on horizon_ops.attn_decode (heads %d, "
                                    "page %d, kv heads %d)", query.shape[1], kv_cache[0].shape[1],
                                    kv_cache[0].shape[2])
        if not ok:
            _log().warning_once("horizon_kernels ops: attention decode call not covered, using XQA")
            return orig(query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer, block_tables=block_tables,
                        seq_lens=seq_lens, max_seq_len=max_seq_len, bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale,
                        window_left=window_left, out=out, sinks=sinks, kv_layout=kv_layout,
                        q_len_per_req=q_len_per_req, mask=mask, q_cu_seq_lens=q_cu_seq_lens,
                        **({"kv_cache_sf": kv_cache_sf} if kv_cache_sf is not None else {}), **kw)
        from tilelang.contrib.horizon_ops import attn_decode as ad
        return ad.decode(query, kv_cache[0], kv_cache[1], block_tables, seq_lens, max_seq_len, bmm1_scale,
                         bmm2_scale, int(q_len_per_req or 1), out=out, kv_layout=kv_layout)

    xqa._horizon = True
    fi.flashinfer_xqa_batch_decode_with_kv_cache = xqa


def _install_prefill_rs():
    from tilelang.contrib.sm120_nvfp4 import prefill_rs
    from tilelang.contrib.horizon_ops import row_scale
    if getattr(prefill_rs.quantize_rs, "_horizon", False):
        return
    fn = row_scale.quantize_rs
    fn._horizon = True
    prefill_rs.quantize_rs = fn
    _log().info("horizon_kernels ops: NVFP4 prefill row scales on horizon_ops.row_scale (one kernel)")


def _install_gdn_conv():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
    if getattr(q.causal_conv1d_update, "_horizon", False):
        return
    orig = q.causal_conv1d_update

    def conv(x, conv_state, weight, bias=None, activation=None, conv_state_indices=None, num_accepted_tokens=None,
             query_start_loc=None, max_query_len=-1, validate_data=False, out=None, **kw):
        ok = (bias is None and activation in ("silu", "swish", True) and num_accepted_tokens is not None
              and query_start_loc is not None and conv_state_indices is not None and not kw and out is None
              and x.dtype == conv_state.dtype and weight.shape[1] == 4 and 1 <= max_query_len <= 8
              and x.shape[1] % 256 == 0 and x.stride(1) == 1)
        if not ok:
            return orig(x, conv_state, weight, bias, activation, conv_state_indices=conv_state_indices,
                        num_accepted_tokens=num_accepted_tokens, query_start_loc=query_start_loc,
                        max_query_len=max_query_len, validate_data=validate_data, out=out, **kw)
        from tilelang.contrib.horizon_ops import conv1d_update as cu
        return cu.conv_update(x, conv_state, weight, conv_state_indices, num_accepted_tokens, query_start_loc,
                              max_query_len)

    conv._horizon = True
    q.causal_conv1d_update = conv
    _log().info("horizon_kernels ops: GDN conv1d spec update on horizon_ops.conv1d_update")


def _qk_impl(q_gate: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
             cos_sin_cache: torch.Tensor, positions: torch.Tensor, eps: float, num_q_heads: int,
             num_kv_heads: int, head_dim: int, rotary_dim: int, sec_h: int,
             sec_w: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch
    from tilelang.contrib.horizon_ops import qk_norm_rope as qk
    NT = q_gate.shape[0]
    qo = torch.empty(NT, num_q_heads * head_dim, dtype=q_gate.dtype, device=q_gate.device)
    ko = torch.empty(NT, num_kv_heads * head_dim, dtype=k.dtype, device=k.device)
    go = torch.empty_like(qo)
    if NT:
        kern = qk.qk_norm_rope_kernel(num_q_heads, num_kv_heads, head_dim, rotary_dim, q_gate.stride(0), k.stride(0),
                                      "bfloat16", sec_h, sec_w)
        kern(q_gate, k, qo, ko, go, q_weight, k_weight, cos_sin_cache, positions, eps)
    return qo, ko, go


def _qk_fake(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, num_q_heads, num_kv_heads, head_dim,
             rotary_dim, sec_h, sec_w):
    NT = q_gate.shape[0]
    qo = q_gate.new_empty((NT, num_q_heads * head_dim))
    return qo, k.new_empty((NT, num_kv_heads * head_dim)), q_gate.new_empty((NT, num_q_heads * head_dim))


def _install_qknorm():
    import torch
    from vllm.model_executor.models import qwen3_next as qn
    if getattr(qn.fused_qk_rmsnorm_rope_gate, "_horizon", False):
        return
    if not (hasattr(torch.ops, "horizon") and hasattr(torch.ops.horizon, "qk_norm_rope_gate")):
        op = torch.library.custom_op("horizon::qk_norm_rope_gate", mutates_args=())(_qk_impl)
        op.register_fake(_qk_fake)
    orig = qn.fused_qk_rmsnorm_rope_gate

    def fused(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, num_q_heads, num_kv_heads, head_dim,
              rotary_dim, mrope_section=None, norm_beta=0.0):
        ok = (norm_beta == 1.0 and head_dim == 256 and rotary_dim == 64 and positions.dim() == 2
              and mrope_section is not None and positions.dtype == torch.int64 and q_gate.dtype == torch.bfloat16
              and cos_sin_cache.dtype == torch.bfloat16 and q_weight.dtype == torch.bfloat16
              and q_gate.stride(1) == 1 and k.stride(1) == 1 and positions.stride(1) == 1)
        if not ok:
            return orig(q_gate, k, q_weight, k_weight, cos_sin_cache, positions, eps, num_q_heads, num_kv_heads,
                        head_dim, rotary_dim, mrope_section=mrope_section, norm_beta=norm_beta)
        return torch.ops.horizon.qk_norm_rope_gate(q_gate, k, q_weight, k_weight, cos_sin_cache, positions,
                                                   float(eps), num_q_heads, num_kv_heads, head_dim, rotary_dim,
                                                   int(mrope_section[1]), int(mrope_section[2]))

    fused._horizon = True
    qn.fused_qk_rmsnorm_rope_gate = fused
    _log().info("horizon_kernels ops: q/k norm + RoPE + gate on horizon_ops.qk_norm_rope")


def _install_gdn_prefill():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
    if getattr(q.fla_chunk_gated_delta_rule, "_horizon", False):
        return
    orig = q.fla_chunk_gated_delta_rule

    def chunk(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None,
              chunk_indices=None, chunk_offsets=None, use_qk_l2norm_in_kernel=False, core_attn_out=None):
        ok = (not use_qk_l2norm_in_kernel and initial_state is not None and output_final_state
              and cu_seqlens is not None and q.dim() == 4 and q.shape[0] == 1 and q.dtype == torch.bfloat16
              and q.shape[-1] == 128 and v.shape[-1] == 128 and v.shape[2] % q.shape[2] == 0
              and initial_state.dtype == torch.float32)
        if not ok:
            return orig(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                        output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                        chunk_offsets=chunk_offsets, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                        core_attn_out=core_attn_out)
        from tilelang.contrib.horizon_ops import gdn_prefill as gp
        o, ht = gp.chunk_gated_delta_rule(q, k, v, g, beta, initial_state, cu_seqlens, scale=scale,
                                          chunk_indices=chunk_indices)
        if core_attn_out is not None:
            co = core_attn_out.reshape(-1)
            co[: o.numel()].copy_(o.reshape(-1))
        return o, ht

    chunk._horizon = True
    q.fla_chunk_gated_delta_rule = chunk
    _log().info("horizon_kernels ops: GDN chunked prefill on horizon_ops.gdn_prefill")


def _install_kvw():
    from vllm.v1.attention.backends import flashinfer as fi
    Impl = fi.FlashInferImpl
    if getattr(Impl.do_kv_cache_update, "_horizon", False):
        return
    orig = Impl.do_kv_cache_update

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if (self.kv_sharing_target_layer_name is not None or getattr(self, "is_kvcache_nvfp4", False)
                or kv_cache.dtype not in (torch.float8_e4m3fn, torch.uint8) or self.cache_dtype not in ("fp8", "fp8_e4m3")
                or key.dtype != torch.bfloat16 or layer._k_scale.numel() != 1 or key.stride(-1) != 1):
            return orig(self, layer, key, value, kv_cache, slot_mapping)
        from tilelang.contrib.horizon_ops import kv_write
        k_cache, v_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        kv_write.write(key, value, k_cache, v_cache, slot_mapping, layer._k_scale, layer._v_scale)

    do_kv_cache_update._horizon = True
    Impl.do_kv_cache_update = do_kv_cache_update
    _log().info("horizon_kernels ops: FP8 KV-cache write on horizon_ops.kv_write")


def install():
    if os.environ.get("HORIZON_KVW", "0") == "1":
        _install_kvw()
    if os.environ.get("HORIZON_GDN_PREFILL", "0") == "1":
        _install_gdn_prefill()
    if os.environ.get("HORIZON_GDN_CONV", "0") == "1":
        _install_gdn_conv()
    if os.environ.get("HORIZON_QKNORM", "0") == "1":
        _install_qknorm()
    if os.environ.get("HORIZON_PREFILL_RS", "0") == "1":
        _install_prefill_rs()
    if os.environ.get("HORIZON_GDN", "0") == "1":
        _install_gdn()
    if os.environ.get("HORIZON_ATTN", "0") == "1":
        _install_attn()
