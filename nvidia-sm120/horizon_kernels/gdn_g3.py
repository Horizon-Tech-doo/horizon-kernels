# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels G3v (30-09-2026, default off): the G3 replay-ring GDN decode inside vLLM 0.30 (V2 model runner,
# prefix caching "align" mode, MTP speculative decoding incl. the 3/1 draft schedule).
#
# HORIZON_GDN_G3=1. vLLM's speculative GDN verify writes the recurrent state after EVERY window token into its own
# window slot (1 + drafts writes of 3.1 MB per request and layer); G3v writes the checkpoint only when a small ring
# of exact per-token records overflows (tilelang.contrib.horizon_ops.gdn_g3v). The ring lives in the conv regions of
# window slots 1..SW-1, which vLLM never uses: no extra GPU memory. Outputs and every state vLLM observes are
# bit-identical to vLLM's kernel by construction (vLLM's own fast-math helpers for the window arithmetic; the replay
# re-applies the window's two instructions with the saved fp32 delta). CPU protocol proof: test_g3v_cpu.py.
#
# vLLM's layout is restored ("expand": slot[t] = state after window token t, exactly vLLM's) before anything else
# reads the slots:
#   H1 MambaHybridModelState.preprocess_state  rows whose window moves (the align pre-copy), and every row when the
#                                              batch is not pure speculative decode (mixed batches go through FLA)
#   H2 MambaHybridModelState.postprocess_state rows the align post-copy saves (postprocess_mamba_fused_kernel's rule)
#   H3 MambaHybridModelState.add_request       the request-state index's ring is reset (meta = 0)
#   H4 MambaHybridModelState.prepare_attn      the vLLM request-state index of every batch row -> a static buffer the
#                                              (CUDA-graph captured) decode kernel reads
#   L  QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm -> gdn_g3v.step
# Everything H1-H4 does is asynchronous GPU work (no host sync); the decode kernel's inputs are static buffers.
# Requirements checked at context build, otherwise vLLM's kernel is used unchanged (logged): fp32 state, K = V = 128,
# align cache mode, the conv region of a state block contiguous and 4-byte aligned, a ring half >= 1 + drafts.
# Not compatible with HORIZON_GDN_RECOVER, HORIZON_GDN, HORIZON_GDN2 / HORIZON_GDN_STATE (refused at install); not yet
# modelled (G3v disables itself, vLLM's kernel runs): fine-grained mamba prefix cache / prefix_match_unit,
# HORIZON_GDN_CKPT, HORIZON_PI. 01-10-2026: those are now ALLOWED (decode-only protocol, see docs/gdn-state-design.md
# section 13; HORIZON_GDN_G3_COMPAT_STRICT=1 restores the refusal). HORIZON_GDN_G3_CHECK=1 adds runtime guards for them.
# G3v-sel (30-09-2026 evening, default on under HORIZON_GDN_G3=1; HORIZON_GDN_G3_SEL=0 = the per-group path): H1 and H2
# are ONE kernel launch each (gdn_g3v.expand_sel: row selection + expand of every layer of every group, and H1 also
# writes H4's request-index buffer), instead of ~40 small torch ops and one expand per group per hook.
# Debug: HORIZON_GDN_G3_CHECK=1 syncs in the non-fused GDN path and asserts no ring is pending (slow).
import os
import sys

import torch

_CTX: dict = {}          # device -> _Ctx
_RIDX: dict = {}         # device -> int32 request-state index per batch row (static, -1 = none)
_STATE = {"announced": False, "warm": set(), "disabled": None}
_RIDX_LEN = 4096
_MAX_REQS = int(os.environ.get("HORIZON_GDN_G3_MAX_REQS", "512"))
_CHECK = os.environ.get("HORIZON_GDN_G3_CHECK", "0") == "1"


def _g3():
    """the G3v kernels: the plugin's own copy (horizon_kernels.g3v_ops, fork branch horizon/g3v-0115) so TileLang's
    installed prefix stays untouched; else TileLang's contrib module once merged there"""
    try:
        from horizon_kernels.g3v_ops import gdn_g3v
    except ImportError:
        from tilelang.contrib.horizon_ops import gdn_g3v
    return gdn_g3v


def enabled() -> bool:
    return os.environ.get("HORIZON_GDN_G3", "0") == "1"


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _announce(msg, *args):
    try:
        sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    _log().info(msg, *args)


def _disable(why):
    if _STATE["disabled"] is None:
        _STATE["disabled"] = why
        _announce("horizon_kernels G3v: DISABLED, vLLM's GDN decode kernel is used (%s)", why)


def _ridx(device):
    b = _RIDX.get(device)
    if b is None:
        b = _RIDX[device] = torch.full((_RIDX_LEN,), -1, dtype=torch.int32, device=device)
    return b


class _Ctx:
    """per device: every Qwen GDN layer with a bound KV cache, its ring view, the meta table, group launch tables"""

    def __init__(self, layers, device):
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention  # noqa
        self.device = device
        self.layers = layers                       # [(prefix, layer)] in a fixed order
        self.idx = {p: i for i, (p, _) in enumerate(layers)}
        l0 = layers[0][1]
        s0 = l0.kv_cache[1]
        self.HV, self.H = s0.shape[1], l0.num_k_heads // l0.tp_size
        self.SS = s0.stride(0)
        self.ptr0 = s0.data_ptr()
        self.ok, self.why = True, ""
        self.rings = []
        for p, l in layers:
            st, conv = l.kv_cache[1], l.kv_cache[0]
            if st.dtype != torch.float32 or st.shape[2:] != (128, 128) or st.stride()[1:] != (16384, 128, 1):
                self.ok, self.why = False, f"{p}: state {st.dtype} {tuple(st.shape)} {st.stride()} not fp32 K=V=128"
                break
            if st.stride(0) != self.SS or st.shape[1] != self.HV:
                self.ok, self.why = False, f"{p}: state layout differs between layers"
                break
            if conv.stride(0) * conv.element_size() != st.stride(0) * 4 or not conv[0].is_contiguous():
                self.ok, self.why = False, f"{p}: conv region not a contiguous per-block region on the state's page"
                break
            off = conv.data_ptr() - st.untyped_storage().data_ptr()
            nbytes = conv[0].numel() * conv.element_size()
            if off % 4 or conv.untyped_storage().data_ptr() != st.untyped_storage().data_ptr():
                self.ok, self.why = False, f"{p}: conv region not 4-byte aligned on the state's storage"
                break
            base = torch.empty(0, dtype=torch.float32, device=device).set_(st.untyped_storage())
            self.rings.append(base.as_strided((st.shape[0], nbytes // 4), (self.SS, 1), off // 4))
        if self.ok:
            self.RF = self.rings[0].shape[1]
            g3 = _g3()
            self.PPS, self.HALF = g3.ring_geometry(self.HV, self.H, self.RF, 4)
            self.meta = torch.zeros(len(layers), _MAX_REQS, self.HV, 4, dtype=torch.int32, device=device)
            self.saddr = [l.kv_cache[1].data_ptr() for _, l in layers]
            self.raddr = [r.data_ptr() for r in self.rings]
        self.groups = None                          # built by the model-state hooks (kv_cache_config needed)
        self.rows = torch.zeros(_RIDX_LEN + 1, dtype=torch.int32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)

    def build_groups(self, kv_cache_config, SW):
        g3 = _g3()
        groups = []
        for gid, g in enumerate(kv_cache_config.kv_cache_groups):
            ls = [self.idx[n] for n in g.layer_names if n in self.idx]
            if ls:
                groups.append((gid, torch.tensor([self.saddr[i] for i in ls], dtype=torch.int64, device=self.device),
                               torch.tensor([self.raddr[i] for i in ls], dtype=torch.int64, device=self.device),
                               torch.tensor(ls, dtype=torch.int32, device=self.device)))
        if sum(int(t[3].numel()) for t in groups) != len(self.layers):
            self.ok, self.why = False, "GDN layers missing from the KV-cache groups"
        self.PPS, self.HALF = g3.ring_geometry(self.HV, self.H, self.RF, SW)
        if self.HALF < SW:
            self.ok, self.why = False, f"ring half {self.HALF} < window {SW} (conv region {self.RF} floats)"
        self.SW = SW
        self.groups = groups
        # G3v-sel: one launch per hook over every layer of every group (row selection on the GPU)
        self.sel = os.environ.get("HORIZON_GDN_G3_SEL", "1") == "1" and 1 <= len(groups) <= 4
        if self.sel:
            self.gids = [gid for gid, _, _, _ in groups]
            order = [i for _, _, _, ls in groups for i in ls.tolist()]
            self.all_saddr = torch.tensor([self.saddr[i] for i in order], dtype=torch.int64, device=self.device)
            self.all_raddr = torch.tensor([self.raddr[i] for i in order], dtype=torch.int64, device=self.device)
            self.all_lmeta = torch.tensor(order, dtype=torch.int32, device=self.device)
            self.all_lgrp = torch.tensor([k for k, (_, _, _, ls) in enumerate(groups) for _ in range(ls.numel())],
                                         dtype=torch.int32, device=self.device)


def _ensure_ctx(fc):
    """the context, (re)built from the forward context's GDN layers once their KV caches are bound; None if not
    (yet) usable. fc = compilation_config.static_forward_context"""
    if _STATE["disabled"] is not None:
        return None
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    layers = [(p, l) for p, l in fc.items() if isinstance(l, QwenGatedDeltaNetAttention)
              and isinstance(getattr(l, "kv_cache", None), (list, tuple)) and len(l.kv_cache) >= 2
              and l.kv_cache[1].numel() > 0]
    if not layers:
        return None
    layers.sort(key=lambda x: x[0])
    dev = layers[0][1].kv_cache[1].device
    c = _CTX.get(dev)
    if c is None or c.ptr0 != layers[0][1].kv_cache[1].data_ptr():     # new, or the KV cache was re-allocated
        c = _CTX[dev] = _Ctx(layers, dev)
        if not c.ok:
            _disable(c.why)
            return None
    return c if c.ok else None


def _ctx_for_layer(layer):
    """the fully integrated context (model-state hooks active: groups built), else None -> vLLM's kernel"""
    c = _CTX.get(layer.kv_cache[1].device)
    if _STATE["disabled"] is not None or c is None or not c.ok or c.groups is None:
        return None
    if c.ptr0 != c.layers[0][1].kv_cache[1].data_ptr():
        return None
    return c


# ------------------------------------------------------------------------------------------------ layer hook (L)
def _warm(ctx, layer, mixed_qkv, a, b, out_gate, si):
    import torch as t
    g3 = _g3()
    key = (mixed_qkv.stride(0), a.stride(0), b.stride(0), out_gate.stride(0), si.shape[1], layer.dt_bias.dtype,
           layer.norm.weight.dtype, layer.norm.activation, ctx.device)
    if key in _STATE["warm"]:
        return True
    if t.cuda.is_current_stream_capturing():
        return False
    HV, dev = ctx.HV, ctx.device
    sidx = t.zeros(1, si.shape[1], dtype=t.int32, device=dev)
    cu = t.tensor([0, 1], dtype=t.int32, device=dev)
    nacc = t.zeros(1, dtype=t.int32, device=dev)        # zero path: no state, ring or meta access
    out = t.empty(1, HV, 128, dtype=t.bfloat16, device=dev)
    g3.step(mixed_qkv[:1], a[:1], b[:1], t.zeros(HV, device=dev), layer.dt_bias, sidx, cu, nacc,
            _ridx(dev)[:1], layer.kv_cache[1], ctx.rings[0], ctx.meta[0], out_gate[:1], layer.norm.weight, out,
            0.1, 1e-6, layer.norm.activation)
    rows = t.zeros(1, dtype=t.int32, device=dev)
    cnt = t.zeros(1, dtype=t.int32, device=dev)
    g3.expand(t.tensor(ctx.saddr[:1], dtype=t.int64, device=dev), t.tensor(ctx.raddr[:1], dtype=t.int64, device=dev),
              t.zeros(1, dtype=t.int32, device=dev), rows, cnt, sidx, rows, ctx.meta, HV, ctx.H, ctx.SS, ctx.RF)
    t.cuda.synchronize()
    _STATE["warm"].add(key)
    if not _STATE["announced"]:
        _STATE["announced"] = True
        _announce("horizon_kernels G3v: GDN decode on horizon_ops.gdn_g3v (layers %d, HV %d, ring half %d, "
                  "positions per slot %d, conv region %d floats, window %d)", len(ctx.layers), HV, ctx.HALF,
                  ctx.PPS, ctx.RF, si.shape[1])
    return True


def _install_layer():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
    Cls = q.QwenGatedDeltaNetAttention
    if getattr(Cls._forward_core_decode_spec_post_conv_fused_norm, "_horizon_g3", False):
        return
    orig = Cls._forward_core_decode_spec_post_conv_fused_norm

    def fused(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata):
        ctx = _ctx_for_layer(self)
        n = attn_metadata.num_spec_decodes
        si = attn_metadata.spec_state_indices_tensor
        if (ctx is None or si is None or self.prefix not in ctx.idx or ctx.HALF < si.shape[1]
                or self.kv_cache[1].dtype != torch.float32 or not _warm(ctx, self, mixed_qkv, a, b, output_gate, si)):
            return orig(self, mixed_qkv, b, a, output_gate, core_attn_out, attn_metadata)
        g3 = _g3()
        li = ctx.idx[self.prefix]
        g3.step(mixed_qkv, a, b, self.A_log, self.dt_bias, si[:n], attn_metadata.spec_query_start_loc[: n + 1],
                attn_metadata.num_accepted_tokens[:n], _ridx(ctx.device)[:n], self.kv_cache[1], ctx.rings[li],
                ctx.meta[li], output_gate, self.norm.weight, core_attn_out, self.head_k_dim ** -0.5,
                self.layer_norm_epsilon, self.norm.activation)

    fused._horizon_g3 = True
    Cls._forward_core_decode_spec_post_conv_fused_norm = fused

    if os.environ.get("HORIZON_GDN_G3_CHECK", "0") == "1":
        orig_core = Cls._forward_core

        def core(self, *args, **kw):
            c = _CTX.get(self.kv_cache[1].device)
            if c is not None and c.ok and self.prefix in c.idx:
                pend = int(((c.meta[c.idx[self.prefix], :, :, 0] // 256) % 256 > 0).sum())
                assert pend == 0, f"G3v: {pend} ring(s) pending in a non-fused GDN forward ({self.prefix})"
                _check_ckpt_slots(self)
            return orig_core(self, *args, **kw)

        Cls._forward_core = core
        _install_check_copies()



# ------------------------------------------------------------------------------------------------ debug guards (CHECK)
def _check_ckpt_slots(layer):
    """HORIZON_GDN_G3_CHECK: the GDN checkpoint's in-step exports (prefill rows) never hit a block of a spec-decode
    row's window in the same step (they would clobber G3v's ring, or vLLM's own per-token states)"""
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    md = md.get(layer.prefix) if isinstance(md, dict) else None
    ck = getattr(md, "horizon_ckpt", None) if md is not None else None
    si = getattr(md, "spec_state_indices_tensor", None) if md is not None else None
    n = getattr(md, "num_spec_decodes", 0) if md is not None else 0
    if ck is None or si is None or not n:
        return
    win = set(int(x) for x in si[:n].flatten().tolist() if int(x) > 0)
    hit = win & set(int(x) for x in ck.slots.tolist())
    assert not hit, f"G3v: GDN checkpoint export into a spec-decode window block {sorted(hit)[:8]} ({layer.prefix})"


def _live_ring_blocks(ms, idx_mapping, n):
    """block ids holding a live ring (mode B) after this step: window slots 1..SW-1 of rows with pending records"""
    c = getattr(ms, "_hz_g3_ctx", None) or next(iter(_CTX.values()), None)
    bt = getattr(ms, "_hz_block_tables", None)
    if c is None or not c.ok or c.groups is None or bt is None or not n:
        return set()
    im = idx_mapping[:n].tolist()
    cols = ms._mamba_state_idx_gpu.tolist()
    pend = ((c.meta[:, :, :, 0] // 256) % 256 > 0).any(dim=2).any(dim=0).tolist()   # per request-state index
    live = set()
    for gid, _, _, _ in c.groups:
        t = bt[gid]
        for r, rq in enumerate(im):
            if rq < 0 or rq >= len(pend) or not pend[rq] or cols[rq] < 0:
                continue
            for k in range(1, c.SW):
                if cols[rq] + k < t.shape[1]:
                    b = int(t[r, cols[rq] + k])
                    if b > 0:
                        live.add(b)
    return live


def _install_check_copies():
    """HORIZON_GDN_G3_CHECK: the scheduler's copy-on-write block copies (fine-grained partial tails) and the zeroing of
    freshly allocated blocks, both run before the next step's hooks, never touch a block holding a live ring"""
    from vllm.v1.worker.gpu import model_runner as mr
    from vllm.v1.worker import utils as wu
    if getattr(mr.copy_kv_cache_blocks_inplace, "_horizon_g3", False):
        return
    orig_copy, orig_zero = mr.copy_kv_cache_blocks_inplace, wu.KVBlockZeroer.zero_block_ids

    def copy_blocks(kv_caches, num_blocks, copies):
        live = _STATE.get("live") or set()
        bad = [cp for cp in copies if int(cp[0]) in live or int(cp[1]) in live]
        assert not bad, f"G3v: copy-on-write block copy touches a live ring block: {bad[:4]}"
        return orig_copy(kv_caches, num_blocks, copies)

    def zero_block_ids(self, block_ids):
        live = _STATE.get("live") or set()
        bad = [b for b in block_ids if int(b) in live]
        assert not bad, f"G3v: zeroing a freshly allocated block that holds a live ring: {bad[:8]}"
        return orig_zero(self, block_ids)

    copy_blocks._horizon_g3 = True
    mr.copy_kv_cache_blocks_inplace = copy_blocks
    wu.KVBlockZeroer.zero_block_ids = zero_block_ids

# ------------------------------------------------------------------------------------------------ model-state hooks
def _pure_spec(input_batch) -> bool:
    import numpy as np
    n = input_batch.num_reqs
    d = input_batch.num_draft_tokens_per_req
    if d is None or n == 0:
        return False
    d = np.asarray(d[:n])
    return bool((d > 0).all() and (np.asarray(input_batch.num_scheduled_tokens[:n]) == d + 1).all()
                and not np.asarray(input_batch.is_prefilling_np[:n]).any())


def _expand_rows(ctx, mask, cols, idx_mapping, block_tables, n):
    """expand the batch rows with mask (device bool (n,)); window slot k of row r = block_table[r, cols[r] + k]"""
    g3 = _g3()
    m = mask.to(torch.int32)
    pos = torch.cumsum(m, 0, dtype=torch.int32) - 1
    tgt = torch.where(m > 0, pos, torch.full_like(pos, _RIDX_LEN))
    rows = ctx.rows
    rows.zero_()
    rows.scatter_(0, tgt.long(), torch.arange(n, dtype=torch.int32, device=ctx.device))
    torch.sum(m, 0, keepdim=True, dtype=torch.int32, out=ctx.cnt)
    im = idx_mapping[:n]
    ridx = torch.where(im >= 0, im, torch.full_like(im, -1)).to(torch.int32)
    ar = torch.arange(ctx.SW, device=ctx.device, dtype=torch.int64)
    for gid, saddr, raddr, lmeta in ctx.groups:
        bt = block_tables[gid][:n]
        c = cols.clamp(min=0).to(torch.int64)[:, None] + ar[None, :]
        ok = (c < bt.shape[1]) & (cols[:, None] >= 0)
        sidx = torch.where(ok, torch.gather(bt, 1, c.clamp(max=bt.shape[1] - 1)), torch.zeros_like(c, dtype=bt.dtype))
        g3.expand(saddr, raddr, lmeta, rows[:n], ctx.cnt, sidx.to(torch.int32).contiguous(), ridx, ctx.meta,
                  ctx.HV, ctx.H, ctx.SS, ctx.RF)


SEL_PRE_CROSS, SEL_PRE_ALL, SEL_POST = 0, 1, 2


def _expand_sel(ctx, mode, im, sti, nc, aux, block_tables, write_ridx):
    """G3v-sel: the row selection of H1 / H2 and the expand of every group in ONE launch (no torch ops, no host
    sync). False (nothing launched) when the inputs don't fit the kernel -> the caller runs _expand_rows."""
    if not getattr(ctx, "sel", False) or im.shape[0] > 512:
        return False
    bts = [block_tables[g] for g in ctx.gids]
    b0 = bts[0]
    if b0.dtype != torch.int32 or b0.dim() != 2 or b0.stride(1) != 1 or b0.shape[0] < im.shape[0]:
        return False
    for b in bts[1:]:
        if b.dtype != torch.int32 or b.shape != b0.shape or b.stride() != b0.stride():
            return False
    _g3().expand_sel(mode, ctx.B, im, sti, nc, aux, bts, ctx.all_saddr, ctx.all_raddr, ctx.all_lmeta, ctx.all_lgrp,
                     ctx.meta, _ridx(ctx.device), ctx.HV, ctx.H, ctx.SS, ctx.RF, ctx.SW, write_ridx)
    return True


def _warm_sel(ctx):
    """compile the three selecting-expand specialisations up front (outside any capture); off on failure"""
    if not getattr(ctx, "sel", False) or torch.cuda.is_current_stream_capturing():
        return
    try:
        g3 = _g3()
        for mode, wr in ((SEL_PRE_CROSS, 1), (SEL_PRE_ALL, 1), (SEL_POST, 0)):
            g3.get_expand_sel_kernel(ctx.HV, ctx.H, ctx.SS, ctx.SW, ctx.RF, mode, ctx.B, wr, "int64", "int32",
                                     "int32")
        ctx.sel_warm = True
        _announce("horizon_kernels G3v: one-launch expand (row selection on the GPU, %d groups, block size %d)",
                  len(ctx.gids), ctx.B)
    except Exception as ex:  # noqa: BLE001
        ctx.sel = False
        _announce("horizon_kernels G3v: one-launch expand unavailable (%s), per-group expand used", str(ex)[:200])


def _model_ctx(ms, kv_cache_config):
    c = getattr(ms, "_hz_g3_ctx", None)          # fast path: checked once, context unchanged
    if (c is not None and _STATE["disabled"] is None and c.ok and c.groups is not None
            and _CTX.get(c.device) is c and c.ptr0 == c.layers[0][1].kv_cache[1].data_ptr()):
        return c
    if _STATE["disabled"] is not None:
        return None
    if not getattr(ms, "_align_mode", False):
        _disable("mamba cache mode is not 'align' (prefix caching off): only align mode is integrated")
        return None
    cc = ms.vllm_config.cache_config
    fine = bool(getattr(cc, "enable_mamba_fine_grained_prefix_cache", False))
    pmu = getattr(cc, "prefix_match_unit", None)
    ckpt = os.environ.get("HORIZON_GDN_CKPT", "0") not in ("", "0")
    pi = os.environ.get("HORIZON_PI", "0") not in ("", "0")
    if os.environ.get("HORIZON_GDN_G3_COMPAT_STRICT", "0") == "1" and (fine or pmu or ckpt or pi):
        _disable("HORIZON_GDN_G3_COMPAT_STRICT=1 with the caching features (fine-grained / prefix_match_unit / "
                 "HORIZON_GDN_CKPT / HORIZON_PI) on")
        return None
    stop = os.environ.get("HORIZON_GDN_CKPT_STOP", "lcp") if ckpt else None
    if ckpt and stop not in ("lcp", "cont", "fill"):
        _disable(f"HORIZON_GDN_CKPT_STOP={stop}: not an analysed stop mode (lcp / cont / fill)")
        return None
    if (fine or pmu or ckpt or pi) and not _STATE.get("compat_announced"):
        # Compatible by construction (docs/gdn-state-design.md section 13): every one of these writes or copies GDN
        # state only for PREFILL rows (their own reserved / running blocks, partial-tail copy-on-write queued before
        # the next forward, in-step checkpoint and fill exports), or changes prefill kernels / chunking; G3v defers
        # only DECODE writes, and the H1 expand puts every row in vLLM's layout before any step with prefill.
        _STATE["compat_announced"] = True
        _announce("horizon_kernels G3v: caching features on (fine-grained=%s, prefix_match_unit=%s, PI=%s, "
                  "GDN checkpoint=%s): decode-only protocol, compatible", fine, pmu, pi, stop)
    c = _ensure_ctx(ms.vllm_config.compilation_config.static_forward_context)
    if c is None:
        return None
    if c.groups is None:
        _, spec = ms._get_mamba_group_info(kv_cache_config)
        c.build_groups(kv_cache_config, 1 + spec.num_speculative_blocks)
        c.B = spec.block_size
        if ms.max_num_reqs > _MAX_REQS:
            c.ok, c.why = False, f"max_num_reqs {ms.max_num_reqs} > HORIZON_GDN_G3_MAX_REQS {_MAX_REQS}"
        if not c.ok:
            _disable(c.why)
            return None
    if getattr(c, "sel", False) and not getattr(c, "sel_warm", False):
        _warm_sel(c)
    if getattr(c, "sel", False) and not getattr(c, "sel_warm", False):
        return c                                   # not warmed yet (capturing): check again next call
    ms._hz_g3_ctx = c
    return c


def _expand_rows_post(ms, c, idx_mapping, num_sampled, num_computed_tokens, bt, n):
    """H2 on the per-group path: rows the align post-copy saves (postprocess_mamba_fused_kernel's rule)"""
    imc = idx_mapping.clamp(min=0).long()
    acc = num_sampled[:n].clamp(min=1)
    new_c = num_computed_tokens[imc]
    running = new_c - acc + 1
    aligned = (new_c // c.B) * c.B
    cols = ms._mamba_state_idx_gpu[imc]
    mask = (aligned >= running) & (idx_mapping >= 0) & (cols >= 0)
    _expand_rows(c, mask, cols, idx_mapping, bt, n)


def _install_model_state():
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState as M
    if getattr(M.preprocess_state, "_horizon_g3", False):
        return
    o_add, o_pre, o_prep, o_post = M.add_request, M.preprocess_state, M.prepare_attn, M.postprocess_state

    def add_request(self, req_index, new_req_data):
        o_add(self, req_index, new_req_data)
        c = next(iter(_CTX.values()), None)
        if c is not None and c.ok and 0 <= req_index < _MAX_REQS:
            c.meta[:, req_index].zero_()                                   # H3

    def preprocess_state(self, input_batch, block_tables, kv_cache_config, num_computed_tokens):
        self._hz_block_tables = block_tables
        c = _model_ctx(self, kv_cache_config) if input_batch.num_reqs else None
        self._hz_ridx_batch = None
        if c is not None:                                                  # H1
            n = input_batch.num_reqs
            im = input_batch.idx_mapping[:n]
            pure = _pure_spec(input_batch)
            if _expand_sel(c, SEL_PRE_CROSS if pure else SEL_PRE_ALL, im, self._mamba_state_idx_gpu,
                           num_computed_tokens, input_batch.query_start_loc[: n + 1], block_tables, True):
                self._hz_ridx_batch = input_batch                          # H4 done by the same launch
                return o_pre(self, input_batch, block_tables, kv_cache_config, num_computed_tokens)
            imc = im.clamp(min=0).long()
            old = self._mamba_state_idx_gpu[imc]
            if pure:
                qsl = input_batch.query_start_loc[: n + 1]
                after = num_computed_tokens[imc] + (qsl[1:] - qsl[:-1])
                new = (after + c.B - 1) // c.B - 1
                mask = (old >= 0) & (new != old) & (im >= 0)
            else:
                mask = (old >= 0) & (im >= 0)
            _expand_rows(c, mask, old, im, block_tables, n)
        return o_pre(self, input_batch, block_tables, kv_cache_config, num_computed_tokens)

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False, ubatch_idx=0):
        _model_ctx(self, kv_cache_config)          # context + groups before the first (captured) forward
        if not for_capture and getattr(self, "_hz_ridx_batch", None) is input_batch:
            self._hz_ridx_batch = None                 # H4 already written by H1's one-launch expand
            return o_prep(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups,
                          kv_cache_config, for_capture=for_capture, ubatch_idx=ubatch_idx)
        rb = _ridx(input_batch.query_start_loc.device)                     # H4
        rb.fill_(-1)
        if not for_capture and input_batch.num_reqs:
            rb[: input_batch.num_reqs].copy_(input_batch.idx_mapping[: input_batch.num_reqs])
        return o_prep(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                      for_capture=for_capture, ubatch_idx=ubatch_idx)

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        c = next(iter(_CTX.values()), None)
        bt = getattr(self, "_hz_block_tables", None)
        n = idx_mapping.shape[0]
        if (c is not None and c.ok and c.groups is not None and bt is not None and n
                and not isinstance(num_sampled, int) and num_computed_tokens is not None
                and getattr(self, "_align_mode", False)):                  # H2
            if not _expand_sel(c, SEL_POST, idx_mapping, self._mamba_state_idx_gpu, num_computed_tokens,
                               num_sampled[:n], bt, False):
                _expand_rows_post(self, c, idx_mapping, num_sampled, num_computed_tokens, bt, n)
        r = o_post(self, idx_mapping, num_sampled, num_computed_tokens)
        if _CHECK:
            _STATE["live"] = _live_ring_blocks(self, idx_mapping, n)
        return r

    for f in (add_request, preprocess_state, prepare_attn, postprocess_state):
        f._horizon_g3 = True
    M.add_request, M.preprocess_state, M.prepare_attn, M.postprocess_state = (add_request, preprocess_state,
                                                                              prepare_attn, postprocess_state)


def install():
    if not enabled():
        return
    for k in ("HORIZON_GDN_RECOVER", "HORIZON_GDN", "HORIZON_GDN2"):
        if os.environ.get(k, "0") == "1":
            raise ValueError(f"HORIZON_GDN_G3=1 with {k}=1: pick one GDN decode path")
    if os.environ.get("HORIZON_GDN_STATE", "") not in ("", "0", "fp32", "float32", "auto"):
        raise ValueError("HORIZON_GDN_G3=1 needs the fp32 GDN state (HORIZON_GDN_STATE unset)")
    _install_layer()
    _install_model_state()
    _announce("horizon_kernels G3v: hooks installed (layer decode + model-state expand/reset; marker at first warm-up)")
