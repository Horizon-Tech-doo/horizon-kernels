# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels GDN in-step prefill checkpoint (30-09-2026, P1 item 2a). Default
# OFF: HORIZON_GDN_CKPT=1 (mamba cache mode "align").
#
# PORT TO vLLM 0.32.0 (24-10-2026; static port, not yet boot-tested). vLLM 0.32 has the checkpoint machinery
# for KDA/Mamba2 (MambaPrefillCheckpointBuilder in the GDN metadata builder) but NOT for Qwen GDN: the Qwen layer's spec
# has num_prefill_checkpoint_blocks = 0 and the layer never exports, so this module is still required. Changes vs 0.30:
#   * GDN metadata is shared between KV-cache groups with the same spec through GDNAttentionMetadataBuilder.
#     update_block_table (dataclasses.replace drops our .horizon_ckpt). _install_update_block_table re-attaches it with
#     the slots re-gathered from that group's block table. Without it the 2nd and 3rd GDN group would reserve a
#     checkpoint slot (spec) and never fill it: silent cache poisoning.
#   * the checkpoint position uses cache_config.cache_hit_alignment_tokens (what the scheduler / MambaManager use) and
#     the builder cross-checks its offsets against vLLM's own native checkpoint metadata (md.checkpoint, built from the
#     same spec): any disagreement raises instead of leaving a reserved slot unfilled.
#   * ChunkGatedDeltaRule.forward_native grew an aiter_prefill_metadata argument; the (no-op) wrapper is removed.
#   * the spec patch is installed LAST and only if every other piece installed (vLLM swallows plugin exceptions, so a
#     half-installed checkpoint would reserve slots nobody fills).
#   * lcp_clamp mirrors 0.32's last_cache_position = get_mamba_prefill_checkpoint_position(num_tokens, block, eagle).
#   * MambaManager (0.32) relocates the scratch blocks below the checkpoint column even in a checkpoint step, so the
#     "fill" exports below find (almost) no non-null columns; they stay as a harmless guard.
#
# Why: with --prefix-match-unit 64 --enable-mamba-fine-grained-prefix-cache the scheduler cuts every cold prompt at its
# last 64-token hash boundary (a 1k prompt runs as 960 + 64 tokens), only so the GDN state at that boundary gets
# cached for a later hit. That extra model step costs ~22% of a cold 1k prompt's TTFT (29-09 screen) and more under PI.
# vLLM 0.30 already knows the fix for KDA layers ("prefill checkpoint"): the step runs to the prompt end in one piece
# and the layer exports the state at the checkpoint position from inside the step (Scheduler._mamba_block_aligned_split
# use_internal_checkpoint; MambaManager reserves the checkpoint block). This module ports it to Qwen GDN:
#
#   SPEC     QwenGatedDeltaNetAttention.get_kv_cache_spec -> num_prefill_checkpoint_blocks=1,
#            prefill_checkpoint_alignment=64 (FLA's chunk). The scheduler and manager code is generic (MambaSpec).
#   META     GDNAttentionMetadataBuilder.build attaches .horizon_ckpt: for every prefill sequence (in the order of
#            prefill_query_start_loc) the checkpoint offset inside this step's query (0 = none) and the cache slot
#            (block table [row, cdiv(seq_len, block) - 2], as KDA), computed with vLLM's own
#            get_mamba_prefill_checkpoint_position / is_mamba_prefill_checkpoint_valid, so it agrees with the manager.
#   FORWARD  the chunked GDN rule runs as two FLA calls: the first on every prefill sequence cut at its checkpoint
#            offset, the second on the remainders from the first call's final states. The op probe showed a split at
#            a 64-aligned offset is bit-identical to one pass (outputs and final state), so the step's numerics do not
#            change. The first call's final states of the checkpointed sequences are the checkpoint recurrent states;
#            the checkpoint conv state is the last (conv width - 1) pre-conv input rows before the offset (slot positions 0..width-2, as the prefill conv kernel writes them). Both are
#            written into the checkpoint slot.
# Anything unexpected (sequence count mismatch, unsupported layout) raises: a reserved but unfilled checkpoint slot would
# silently poison later cache hits.
#
# LCP stop (always with the checkpoint, 30-09-2026 20:40): vLLM's scheduler sets last_cache_position = 0 when it uses an
# internal checkpoint, so a step like [8000, 14336) no longer stops at the last cacheable block boundary (11,200). In
# align mode only chunk-end states exist, but the manager's full-block pass still offered that block to a later request,
# which restored a state nobody wrote (determinism check: 15 of 16 identical, cache hit at 11200, first token wrong;
# the control without checkpoint 16/16). The wrapper re-inserts that stop: the prompt runs up to its last cacheable block
# boundary as usual, and only the final chunk carries the in-step checkpoint.
#
# HORIZON_GDN_CKPT_STOP (30-09-2026 22:10) picks how the checkpoint avoids that stale hit:
#   lcp  (default): the stop above on every prefill chunk. Cost: a fresh 4k prompt
#        runs as [0, 1600) + [1600, 4096) (cold 4k TTFT x0.909 vs x0.957 without the stop).
#   cont the stop only on a continuation chunk (the request already ran a chunk of its own). Mechanism, from vLLM 0.30's
#        MambaManager.allocate_new_blocks: with an internal checkpoint it skips the relocation of the request's
#        speculative scratch blocks (num_speculative_blocks = 3 with MTP), so on a continuation chunk the old scratch
#        blocks stay at columns below the checkpoint column, non-null, and the full-block pass hashes them as block
#        states (k11: [0, 8000) + [8000, 14336); columns 5 and 6 = 9,600 and 11,200 hashed, never written; the resend
#        hit 11,200). A fresh request (first allocation) null-fills those columns, so it needs no stop.
#   fill no stop at all: on a checkpointed step, every non-null block column c below the checkpoint column whose
#        boundary (c + 1) * block lies inside the step gets the state at that boundary written in-step (the same
#        split-and-export as the checkpoint), so the hash the manager registers is true (slot invariant "slot p holds
#        the state after (p + 1) * block tokens"). 64-aligned splits are bit-exact, so the step's numerics do not change.
#
# HORIZON_TAIL_MIN_BLOCK=1 (separate switch, default off): a prompt of at most one hybrid block (1,600 tokens) gets NO
# sub-block tail stop. vLLM's checkpoint slot is block column cdiv(seq_len, block) - 2, so such a prompt cannot use the
# in-step checkpoint, and the tail stop would cost a whole extra model step (cold 1k TTFT x0.80, screen
# 30-09-2026-1608-pi-pmu64-33). Without it the prompt runs in one step and caches no GDN state below one block, exactly
# like production's 1,600-token block caching. Under HORIZON_PI the numerics do not depend on where chunks stop, so
# determinism is unaffected; only sub-block reuse of prompts under one block is given up.
import copy
import dataclasses
import os

import torch

_DONE = False
_CUR = {}          # set around QwenGatedDeltaNetAttention._forward_core: layer, attn metadata, pre-conv conv rows
_ALIGN = 64
_MODE = "lcp"      # HORIZON_GDN_CKPT_STOP, read at install()


def enabled() -> bool:
    return os.environ.get("HORIZON_GDN_CKPT", "0") == "1"


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


# ---------------------------------------------------------------------------------------------------------------- META
def prefill_rows(num_reqs, num_decodes, spec_mask_cpu):
    """Batch rows of the prefill sequences, in the order of GDNAttentionMetadata.prefill_query_start_loc."""
    if spec_mask_cpu is None:
        return list(range(num_decodes, num_reqs))
    return [r for r in range(num_reqs) if not bool(spec_mask_cpu[r])]


def plan_checkpoints(rows, qsl_cpu, seq_lens, block_size, hash_block_size, drop_eagle_block, alignment=_ALIGN,
                     alignment_tokens=None):
    """-> (offsets per prefill sequence (0 = no checkpoint), checkpoint block-table columns (-1 = none),
    absolute token row of each checkpoint (qsl[row] + offset, -1 = none)). Pure host logic (unit-tested).
    alignment_tokens: vLLM 0.32's cache_hit_alignment_tokens (positions use it, validity uses hash_block_size, as
    vllm.model_executor.layers.mamba.checkpoint.compute_mamba_prefill_checkpoints); None = hash_block_size (0.30)."""
    from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position, is_mamba_prefill_checkpoint_valid
    offsets, cols, abs_rows = [], [], []
    for r in rows:
        qlen = int(qsl_cpu[r + 1]) - int(qsl_cpu[r])
        seq_len = int(seq_lens[r])
        qstart = seq_len - qlen
        if qlen <= 0:
            offsets.append(0), cols.append(-1), abs_rows.append(-1)
            continue
        pos = get_mamba_prefill_checkpoint_position(seq_len, alignment_tokens or hash_block_size,
                                                    drop_eagle_block=drop_eagle_block)
        valid = is_mamba_prefill_checkpoint_valid(query_start=qstart, query_end=seq_len, checkpoint_position=pos,
                                                  hash_block_size=hash_block_size, mamba_block_size=block_size,
                                                  checkpoint_alignment=alignment)
        off = pos - qstart if valid else 0
        offsets.append(off)
        cols.append(-(-seq_len // block_size) - 2 if valid else -1)
        abs_rows.append(int(qsl_cpu[r]) + off if valid else -1)
    return offsets, cols, abs_rows


def stop_mode() -> str:
    m = os.environ.get("HORIZON_GDN_CKPT_STOP", "lcp")
    if m not in ("lcp", "cont", "fill"):
        raise ValueError(f"HORIZON_GDN_CKPT_STOP={m!r}: expected lcp, cont or fill")
    return m


def plan_fills(block_row, qstart, qend, ck_col, block_size, null_id=0):
    """fill mode: block columns c < ck_col whose boundary (c + 1) * block_size lies strictly inside (qstart, qend) and
    whose block is real (not the null block) and owned by this column alone. -> [(offset in the query, column)]."""
    out = []
    for c in range(max(qstart // block_size, 0), ck_col):
        p = (c + 1) * block_size
        if not qstart < p < qend:
            continue
        bid = int(block_row[c])
        if bid == null_id or sum(int(b) == bid for b in block_row) != 1:
            continue
        out.append((p - qstart, c))
    return out


@dataclasses.dataclass
class CkptMeta:
    cuts: list               # per prefill sequence: sorted in-query offsets where a state is exported ([] = none)
    exports: list            # (prefill sequence, index into its cuts) per exported state, in slot order
    slots: torch.Tensor      # (len(exports),) cache slot per exported state (device, int)
    abs_rows: list           # per export: absolute token row (in the step's full token order) of the offset
    prefill_qsl_cpu: list    # prefill cu_seqlens (host)
    rows_cols: object = None  # packed (batch rows + block columns) of the exports, to re-gather slots per KV-cache group


def _crosscheck(md, rows, offsets, num_reqs, spec_mask):
    """vLLM 0.32 builds its own checkpoint metadata (md.checkpoint) from the same spec, scheduler rules and block
    table. Our plan must agree with it exactly: a checkpoint the manager reserved but we did not export would poison
    the prefix cache; one we export that vLLM did not plan writes into a slot nobody reserved. HORIZON_GDN_CKPT_CROSSCHECK=0
    switches the check off."""
    nat = getattr(md, "checkpoint", None)
    if not hasattr(md, "checkpoint") or os.environ.get("HORIZON_GDN_CKPT_CROSSCHECK", "1") != "1":
        return
    if nat is None:
        if any(offsets):
            raise RuntimeError(f"horizon GDN checkpoint: planned offsets {offsets} but vLLM's native plan has none")
        return
    native_rows = list(range(num_reqs)) if spec_mask is None else [r for r in range(num_reqs) if not spec_mask[r]]
    by_row = dict(zip(native_rows, nat.offsets or []))
    mine = {r: o for r, o in zip(rows, offsets)}
    theirs = {r: by_row.get(r, 0) for r in rows}
    extra = {r: o for r, o in by_row.items() if o and r not in mine}
    if mine != theirs or extra:
        raise RuntimeError(f"horizon GDN checkpoint: offsets {mine} disagree with vLLM's native checkpoint plan "
                           f"{theirs} (+ rows outside the prefill set: {extra})")


def _install_update_block_table():
    """GDNAttentionMetadataBuilder.update_block_table (vLLM 0.32, MRV2 / gpu_model_runner) re-uses the first GDN
    group's metadata for the other groups with the same spec through dataclasses.replace(), which drops our ad-hoc
    .horizon_ckpt attribute. Carry it over with the slots re-gathered from the group's own block table."""
    from vllm.v1.attention.backends import gdn_attn as G
    cls = G.GDNAttentionMetadataBuilder
    orig = getattr(cls, "update_block_table", None)
    if orig is None:
        _log().info("horizon_kernels GDN checkpoint: this vLLM has no GDNAttentionMetadataBuilder.update_block_table")
        return
    if getattr(orig, "_horizon_ckpt", False):
        return

    def update_block_table(self, metadata, blk_table, slot_mapping):
        new = orig(self, metadata, blk_table, slot_mapping)
        ck = getattr(metadata, "horizon_ckpt", None)
        if ck is None:
            new.horizon_ckpt = None
            return new
        from .gdn_checkpoint_metadata import regather_slots
        ck2 = copy.copy(ck)             # keeps horizon_abs_rows / horizon_export_map (batch-level, group independent)
        ck2.slots = regather_slots(blk_table, ck.rows_cols)
        new.horizon_ckpt = ck2
        return new

    update_block_table._horizon_ckpt = True
    cls.update_block_table = update_block_table
    _log().info("horizon_kernels GDN checkpoint: update_block_table carries the checkpoint to every GDN KV-cache group")


def _install_builder():
    from vllm.v1.attention.backends import gdn_attn as G
    orig = G.GDNAttentionMetadataBuilder.build
    if getattr(orig, "_horizon_ckpt", False):
        return

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None, num_decode_draft_tokens_cpu=None,
              fast_build=False):
        md = orig(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=num_accepted_tokens,
                  num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu, fast_build=fast_build)
        md.horizon_ckpt = None
        spec = self.kv_cache_spec
        if (md.num_prefills == 0 or getattr(spec, "num_prefill_checkpoint_blocks", 0) <= 0
                or self.vllm_config.cache_config.mamba_cache_mode != "align"):
            return md
        m = common_attn_metadata
        qsl = m.query_start_loc_cpu.tolist()
        spec_mask = None
        if md.spec_sequence_masks is not None:
            spec_mask = (num_decode_draft_tokens_cpu >= 0).tolist()
        rows = prefill_rows(m.num_reqs, md.num_decodes, spec_mask)
        n_seq = md.prefill_query_start_loc.shape[0] - 1
        # vLLM 0.32 leaves trailing padding rows (query length 0) out of num_prefills / prefill_query_start_loc
        while len(rows) > n_seq and qsl[rows[-1] + 1] == qsl[rows[-1]]:
            rows.pop()
        if len(rows) != n_seq:
            raise RuntimeError(f"horizon GDN checkpoint: {len(rows)} prefill rows vs {n_seq} prefill sequences")
        seq_lens = m.seq_lens_cpu_upper_bound
        if seq_lens is None:
            raise RuntimeError("horizon GDN checkpoint: seq_lens_cpu_upper_bound missing")
        cc = self.vllm_config.cache_config
        sc = self.vllm_config.speculative_config
        hash_size = getattr(cc, "hash_block_size", None) or cc.prefix_match_unit
        if getattr(cc, "hash_block_size", 1) is None:
            return md          # unresolved only for the minimal KV cache built for profiling (vLLM skips it too)
        hash_size = hash_size or spec.block_size
        align_tokens = getattr(cc, "cache_hit_alignment_tokens", None) or hash_size
        offsets, cols, abs_rows = plan_checkpoints(
            rows, qsl, seq_lens.tolist(), spec.block_size, hash_size,
            sc is not None and sc.use_eagle_block_drop(), spec.prefill_checkpoint_alignment or _ALIGN,
            alignment_tokens=align_tokens)
        _crosscheck(md, rows, offsets, m.num_reqs, spec_mask)
        seq_ids = [i for i, o in enumerate(offsets) if o]
        if not seq_ids:
            return md
        dev = m.query_start_loc.device
        cuts = [[] for _ in rows]
        exp_rc = {}                                     # (seq, offset) -> (batch row, block column)
        for i in seq_ids:
            cuts[i].append(offsets[i])
            exp_rc[(i, offsets[i])] = (rows[i], cols[i])
        if _MODE == "fill":
            cand = [i for i in seq_ids if cols[i] > 0]
            if cand:
                bt = m.block_table_tensor[torch.tensor([rows[i] for i in cand], device=dev)].cpu().tolist()
                for i, brow in zip(cand, bt):
                    qlen = qsl[rows[i] + 1] - qsl[rows[i]]
                    qstart = int(seq_lens[rows[i]]) - qlen
                    for off, c in plan_fills(brow, qstart, qstart + qlen, cols[i], spec.block_size):
                        if off % _ALIGN or (qstart % _ALIGN):
                            raise RuntimeError(f"horizon GDN checkpoint: fill at {qstart}+{off} not {_ALIGN}-aligned")
                        cuts[i].append(off)
                        exp_rc[(i, off)] = (rows[i], c)
                    cuts[i].sort()
        exports, rr, cc_l, abs_l = [], [], [], []
        for i, cs in enumerate(cuts):
            for j, off in enumerate(cs):
                r, c = exp_rc[(i, off)]
                exports.append((i, j))
                rr.append(r)
                cc_l.append(c)
                abs_l.append(qsl[r] + off)
        pq = [0]
        for r in rows:
            pq.append(pq[-1] + qsl[r + 1] - qsl[r])
        from .gdn_prefill_export import export_plan
        from .gdn_checkpoint_metadata import pack_metadata_ex
        slots, history_rows, export_map, rows_cols = pack_metadata_ex(
            m.block_table_tensor, rr, cc_l, abs_l, export_plan(pq, cuts, exports))
        md.horizon_ckpt = CkptMeta(cuts, exports, slots, abs_l, pq, rows_cols)
        md.horizon_ckpt.horizon_abs_rows = history_rows
        md.horizon_ckpt.horizon_export_map = export_map
        return md

    build._horizon_ckpt = True
    G.GDNAttentionMetadataBuilder.build = build
    _log().info("horizon_kernels checkpoint metadata: packed slot/history/export transfer")


# ---------------------------------------------------------------------------------------------------------------- SPEC
def _install_spec():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention as Q
    orig = Q.get_kv_cache_spec
    if getattr(orig, "_horizon_ckpt", False):
        return

    def get_kv_cache_spec(self, vllm_config):
        spec = orig(self, vllm_config)
        if vllm_config.cache_config.mamba_cache_mode != "align":
            return spec
        return dataclasses.replace(spec, num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=_ALIGN)

    get_kv_cache_spec._horizon_ckpt = True
    Q.get_kv_cache_spec = get_kv_cache_spec


# ------------------------------------------------------------------------------------------------------------- FORWARD
def split_rounds(prefill_qsl, cuts):
    """Token ranges of the chained FLA calls. Round r runs segment r of every sequence that has one: round 0 is every
    sequence from its start to its first cut (or its end), round r >= 1 the sequences with at least r cuts, from cut
    r - 1 to cut r (or the end). -> [(idx, cu, seqs)]: row indices into the prefill token order, cu_seqlens, and the
    prefill sequence of each segment. The state after round r of sequence i (r < len(cuts[i])) is its state at cut r."""
    rounds = []
    for r in range(1 + max((len(c) for c in cuts), default=0)):
        idx, cu, seqs = [], [0], []
        for i, cs in enumerate(cuts):
            if r > len(cs):
                continue
            s, e = prefill_qsl[i], prefill_qsl[i + 1]
            a = s + (cs[r - 1] if r else 0)
            b = s + cs[r] if r < len(cs) else e
            if not a < b and r:
                raise RuntimeError(f"horizon GDN checkpoint: empty segment {r} of sequence {i} ({cs}, {s}..{e})")
            idx.extend(range(a, b))
            cu.append(len(idx))
            seqs.append(i)
        rounds.append((idx, cu, seqs))
    return rounds


def _install_forward():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as QM
    Q = QM.QwenGatedDeltaNetAttention
    if getattr(Q._forward_core, "_horizon_ckpt", False):
        return
    orig_core = Q._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        md = md.get(self.prefix) if isinstance(md, dict) else None
        ck = getattr(md, "horizon_ckpt", None) if md is not None else None
        if ck is None:
            if (md is not None and getattr(md, "checkpoint", None) is not None
                    and os.environ.get("HORIZON_GDN_CKPT_CROSSCHECK", "1") == "1"):
                # vLLM reserved a checkpoint slot for this step but the metadata reaching this layer carries no export
                # plan (e.g. a metadata path that bypassed build / update_block_table): fail instead of poisoning.
                raise RuntimeError("horizon GDN checkpoint: vLLM planned a prefill checkpoint for this step but the "
                                   f"GDN metadata of layer {self.prefix} has no horizon_ckpt")
            return orig_core(self, mixed_qkv, b, a, core_attn_out)
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        conv_cache = self.kv_cache[0]
        cs = conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)   # (slots, dim, state_len)
        # The slot is conv width - 1 + num_speculative_tokens wide (extra history for spec-decode updates), but the
        # prefill kernel (causal_conv1d_fn, state_len = width - 1) writes and reads only positions [0, width - 1): the
        # last width - 1 pre-conv inputs, oldest first. The checkpoint writes exactly those (offset >= 64 rows).
        width = self.conv1d.weight.size(-1) - 1
        from .gdn_checkpoint_conv import export_conv
        if not hasattr(ck, 'horizon_abs_rows'):
            ck.horizon_abs_rows = torch.tensor(ck.abs_rows,device=mixed_qkv.device,dtype=torch.int64)
        export_conv(mixed_qkv,cs,ck.slots,ck.horizon_abs_rows,width)
        _CUR.update(layer=self, ck=ck, states=None)
        try:
            out = orig_core(self, mixed_qkv, b, a, core_attn_out)
            states = _CUR.get("states")
            if states is not True:
                raise RuntimeError("horizon GDN checkpoint: direct state export did not run")
            return out
        finally:
            _CUR.clear()

    _forward_core._horizon_ckpt = True
    Q._forward_core = _forward_core


def _install_tail_min_block():
    from vllm.v1.core.sched.scheduler import Scheduler
    orig = Scheduler._mamba_block_aligned_split
    if getattr(orig, "_horizon_tail", False):
        return

    def _mamba_block_aligned_split(self, request, num_new_tokens, *args, **kwargs):
        if self.mamba_partial_cache_hit and request.num_prompt_tokens <= self.cache_config.block_size:
            self.mamba_partial_cache_hit = False
            try:
                return orig(self, request, num_new_tokens, *args, **kwargs)
            finally:
                self.mamba_partial_cache_hit = True
        return orig(self, request, num_new_tokens, *args, **kwargs)

    _mamba_block_aligned_split._horizon_tail = True
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
    _log().info("horizon_kernels: no sub-block tail stop for prompts of at most one hybrid block")


def install_tail():
    if os.environ.get("HORIZON_TAIL_MIN_BLOCK", "0") == "1":
        _install_tail_min_block()


def lcp_clamp(start, n, num_tokens, block_size, eagle_drop):
    """Chunk length after re-inserting the stop at the last cacheable block boundary (vLLM's last_cache_position)."""
    # vLLM 0.32: last_cache_position = get_mamba_prefill_checkpoint_position(num_tokens, block_size, eagle_drop)
    proof_limit = num_tokens if eagle_drop else num_tokens - 1
    lcp = proof_limit // block_size * block_size
    if eagle_drop:
        lcp -= block_size
    lcp = max(lcp, 0)
    return lcp - start if start < lcp < start + n else n


def _install_lcp_stop():
    from vllm.v1.core.sched.scheduler import Scheduler
    orig = Scheduler._mamba_block_aligned_split
    if getattr(orig, "_horizon_lcp", False):
        return

    def _mamba_block_aligned_split(self, request, num_new_tokens, num_new_local_computed_tokens=0,
                                   num_external_computed_tokens=0):
        n = orig(self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
        start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if n <= 0 or start >= prefill_end or (_MODE == "cont" and request.num_computed_tokens == 0):
            return n
        return lcp_clamp(start, n, request.num_tokens, self.cache_config.block_size, self.use_eagle_block_drop)

    _mamba_block_aligned_split._horizon_lcp = True
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split


def install():
    global _DONE
    global _MODE
    if _DONE or not enabled():
        return
    _MODE = stop_mode()
    # vLLM swallows exceptions raised by general plugins, so a half-installed checkpoint (spec patched, builder or layer
    # not) would reserve checkpoint slots nobody fills and poison the prefix cache. The spec patch therefore goes LAST
    # and only after everything else installed; any failure leaves the stock (slower, correct) behaviour, loudly.
    try:
        if _MODE != "fill":
            _install_lcp_stop()
        _install_update_block_table()
        _install_builder()
        _install_forward()
        _install_spec()
    except Exception:  # noqa: BLE001
        _log().exception("horizon_kernels GDN checkpoint: INSTALL FAILED, the in-step checkpoint stays OFF "
                         "(stock vLLM behaviour: correct, but a cold prefill takes an extra step)")
        return
    _DONE = True
    _log().info("horizon_kernels GDN checkpoint: direct-cache single-pass prefill checkpoint for Qwen GDN (alignment %d)", _ALIGN)
    _log().info("horizon_kernels GDN checkpoint: stale-block guard %s", _MODE)
