# HORIZON_PMU_MTP_FIX=1 (27-09-2026, default 0 = stock vLLM): makes `--prefix-match-unit` work with EAGLE/MTP
# speculative decoding on hybrid (attention + Mamba/GDN "align" mode) models in vLLM 0.29.
#
# The bug (vLLM 0.29.0, scheduler / KV-cache manager only; the model and the drafter are not involved):
#   With a prefix_match_unit u finer than the block, align mode stores the GDN state at the prompt's last hash
#   boundary T = floor(N/u)*u ("partial tail": scheduler.py:442-446 adds a chunk stop there,
#   single_type_kv_cache_manager.py:1867-1872 registers it). But with EAGLE/MTP every attention hit drops one hash
#   unit (single_type_kv_cache_manager.py:778-779: the draft layer's KV at the last matched position was written
#   with the NEXT token), so attention can offer at most T - u. The state at T is therefore never usable:
#     - the Mamba lookup (max_length T - u) falls back to the last full-block state (no finer reuse);
#     - attention (T - u) and Mamba (a block boundary) now disagree, so the coordinator reports an "uncached
#       common prefix" (kv_cache_coordinator.py:899) that the manager pins as request.shared_prefix_boundary
#       (kv_cache_manager.py:289-293);
#     - the scheduler's Marconi junction stop (scheduler.py:458-462) then cuts the cached request's prefill at an
#       extra block boundary the uncached request never had (N=3601, u=64, block 1600: uncached chunks
#       1600-3584-3601, cached 1600-3200-3584-3601).
#   A different chunk split runs the target model with different GEMM / attention shapes, so the prompt's logits
#   differ slightly (mean |dlogp| 0.009 in experiment 4) and greedy answers part at near-ties.
#
# The fix (only when EAGLE/MTP drops a unit, backoff = u; without EAGLE everything stays stock):
#   HORIZON_PMU_MTP_FIX=1 (minimal): store the GDN partial-tail state at T - u instead of T, in BOTH places that
#     must agree (the scheduler's chunk stop and the Mamba manager's registration). An exact repeat then hits T - u
#     in every group, no junction is reported, and the cached request recomputes exactly the uncached request's last
#     chunk [T - u, N). Follow-up turns (previous prompt + answer + new text) match attention's entry at T, drop u,
#     and hit the same state. Corner: when u divides N, a repeat can only match up to N - 1 < T, so it falls back
#     to a block boundary (still the same chunk split, except when the block size also divides N, which stock vLLM
#     gets wrong even without prefix_match_unit).
#   HORIZON_PMU_MTP_FIX=2: also move the attention partial tail to R = floor((N-1)/u)*u and the GDN state to R - u
#     (R == T unless u divides N). Exact repeats of every length then hit R - u; the price is u tokens on follow-up
#     turns whose previous prompt length was a multiple of u.
#
# Installed in every vLLM process by vllm_plugin.register(); only the EngineCore (scheduler) process uses it.
# Guarded by source hashes of the replaced vLLM functions: on any other vLLM version it stays off, loudly.
import hashlib
import inspect
import os

_EXPECTED_SRC = {   # sha256(inspect.getsource(fn))[:16] of vLLM v0.29.0 (98dff2a8)
    "Scheduler._mamba_block_aligned_split": "fe3e2f4166e41974",
    "MambaManager._cache_partial_tail_block": "28c1b3f9cdbe65cb",
    "FullAttentionManager._cache_partial_tail_block": "d547cd92571462e3",
}
_installed = False


def _log():
    from vllm.logger import init_logger
    return init_logger("horizon_kernels.pmu_mtp_fix")


def attn_tail_boundary(num_prompt_tokens: int, hash_block_size: int, backoff: int, repeat: bool = False) -> int:
    """Where the attention groups register the prompt's partial-tail entry. Stock (and level 1): T = floor(N/u)*u.
    Level 2 with EAGLE/MTP: R = floor((N-1)/u)*u, the last boundary a repeat of the same prompt can match (hits are
    capped at N-1)."""
    if backoff and repeat:
        return (num_prompt_tokens - 1) // hash_block_size * hash_block_size
    return num_prompt_tokens // hash_block_size * hash_block_size


def mamba_tail_boundary(num_prompt_tokens: int, hash_block_size: int, backoff: int, repeat: bool = False) -> int:
    """Where align mode stores the prompt's partial-tail Mamba state (and the scheduler stops a chunk for it).
    Stock: floor(N/u)*u. With EAGLE/MTP (backoff = u): the attention entry minus the unit every EAGLE hit drops."""
    return max(attn_tail_boundary(num_prompt_tokens, hash_block_size, backoff, repeat) - backoff, 0)


def tail_backoff_for(coordinator) -> int:
    """The hash-unit drop an EAGLE/MTP attention hit applies (0 = none), computed from a built coordinator."""
    if not getattr(coordinator, "enable_partial_hash_hits", False):
        return 0
    if not getattr(coordinator, "eagle_group_ids", None):
        return 0
    from vllm.v1.kv_cache_interface import MambaSpec
    for g in getattr(coordinator, "attention_groups", ()):
        if g.use_eagle and not isinstance(g.spec, MambaSpec):
            # FullAttentionManager.find_longest_cache_hit: hit -= min(alignment_tokens, block_size); with partial
            # hits on, alignment_tokens == hash_block_size <= block_size.
            return int(coordinator.hash_block_size)
    return 0


def _scheduler_tail(sched) -> tuple[int, bool]:
    coord = getattr(getattr(sched, "kv_cache_manager", None), "coordinator", None)
    b = getattr(coord, "_horizon_tail_backoff", None)
    if b is None:   # e.g. a unit-test stand-in without a coordinator
        b = sched.hash_block_size if getattr(sched, "use_eagle", False) else 0
    return b, getattr(coord, "_horizon_tail_repeat", _level() >= 2)


def _level() -> int:
    v = os.environ.get("HORIZON_PMU_MTP_FIX", "0").strip()
    return int(v) if v.isdigit() else 0


def _mamba_block_aligned_split(
    self,
    request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    """vLLM 0.29.0 Scheduler._mamba_block_aligned_split, verbatim except the tail boundary (marked HORIZON)."""
    start = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
    if start >= prefill_end:
        return num_new_tokens

    block_size = self.cache_config.block_size
    last_cache_position = request.num_tokens - request.num_tokens % block_size
    if self.use_eagle:
        last_cache_position = max(last_cache_position - block_size, 0)

    end = start + num_new_tokens
    use_internal_checkpoint = (
        self.mamba_has_prefill_checkpoint_blocks and start % block_size == 0
    )
    if use_internal_checkpoint:
        last_cache_position = 0
    if end < prefill_end and not use_internal_checkpoint:
        max_prefill_tokens = self.max_num_scheduled_tokens
        long_prefill_threshold = self.scheduler_config.long_prefill_token_threshold
        if long_prefill_threshold > 0:
            max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
        aligned_end = end // block_size * block_size
        if aligned_end > start or block_size <= max_prefill_tokens:
            end = aligned_end

    next_block_boundary = (start // block_size + 1) * block_size
    # HORIZON: stock uses floor(N/u)*u, which an EAGLE/MTP lookup can never reach (it drops one hash unit).
    tail_boundary = (
        mamba_tail_boundary(request.num_prompt_tokens, self.hash_block_size, *_scheduler_tail(self))
        if self.mamba_partial_cache_hit
        else 0
    )
    stops = (
        next_block_boundary if start % block_size != 0 else 0,
        last_cache_position,
        tail_boundary
        if last_cache_position < tail_boundary < request.num_prompt_tokens
        else 0,
        start + (request.shared_prefix_boundary - start) // block_size * block_size
        if start < request.shared_prefix_boundary < end
        else 0,
    )
    end = min((s for s in stops if start < s < end), default=end)
    return max(end - start, 0)


def _mamba_cache_partial_tail_block(self, request, num_tokens: int):
    """vLLM 0.29.0 MambaManager._cache_partial_tail_block, verbatim except the boundary (marked HORIZON)."""
    hash_block_size = self.block_pool.hash_block_size
    if self.block_size == hash_block_size:
        return None
    if num_tokens % self.block_size == 0:
        return None
    if num_tokens % hash_block_size != 0:
        return None
    # HORIZON: register the state where an EAGLE/MTP lookup lands, not at floor(N/u)*u.
    latest_prompt_hash_boundary = mamba_tail_boundary(
        request.num_prompt_tokens, hash_block_size, getattr(self, "_horizon_tail_backoff", 0),
        getattr(self, "_horizon_tail_repeat", False),
    )
    if num_tokens != latest_prompt_hash_boundary:
        return None

    block_idx = num_tokens // self.block_size
    blocks = self.req_to_blocks[request.request_id]
    if block_idx >= len(blocks):
        return None
    source_block = blocks[block_idx]
    if source_block.is_null:
        return None

    partial_hash = self.block_pool.cache_partial_block(
        request=request,
        block=source_block,
        num_tokens=num_tokens,
        kv_cache_group_id=self.kv_cache_group_id,
        block_size=self.block_size,
    )
    if partial_hash is not None:
        self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
        self.num_cached_block[request.request_id] = block_idx
        self._producer_partial_tail_reqs[request.request_id] = num_tokens
    return partial_hash


def _attn_cache_partial_tail_block(self, request, num_tokens: int) -> None:
    """vLLM 0.29.0 FullAttentionManager._cache_partial_tail_block, verbatim except the boundary (marked HORIZON)."""
    hash_block_size = self.block_pool.hash_block_size
    # HORIZON (level 2 only): with EAGLE/MTP use floor((N-1)/u)*u, reachable by a repeat of the same prompt.
    boundary_tokens = attn_tail_boundary(
        request.num_prompt_tokens, hash_block_size, getattr(self, "_horizon_tail_backoff", 0),
        getattr(self, "_horizon_tail_repeat", False),
    )
    if boundary_tokens == 0 or boundary_tokens > num_tokens:
        return
    if boundary_tokens % self.block_size == 0:
        return

    blocks = self.req_to_blocks[request.request_id]
    block_idx = boundary_tokens // self.block_size
    if block_idx >= len(blocks):
        return
    self.block_pool.cache_partial_block(
        request=request,
        block=blocks[block_idx],
        num_tokens=boundary_tokens,
        kv_cache_group_id=self.kv_cache_group_id,
        block_size=self.block_size,
    )


def _src_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def install(level: int | None = None, force: bool = False) -> bool:
    """Patch vLLM's scheduler, Mamba / full-attention managers and hybrid coordinator. True = fix active.
    level: 1 = minimal (GDN tail at T - u), 2 = also exact repeats of every length (see the header); None = env."""
    global _installed
    if _installed:
        return True
    level = _level() if level is None else level
    level = 1 if level < 1 else min(level, 2)
    from vllm.v1.core import kv_cache_coordinator as kcc
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager

    found = {
        "Scheduler._mamba_block_aligned_split": _src_hash(Scheduler._mamba_block_aligned_split),
        "MambaManager._cache_partial_tail_block": _src_hash(MambaManager._cache_partial_tail_block),
        "FullAttentionManager._cache_partial_tail_block": _src_hash(FullAttentionManager._cache_partial_tail_block),
    }
    bad = {k: v for k, v in found.items() if v != _EXPECTED_SRC[k]}
    if bad and not force:
        _log().warning("horizon_kernels PMU-MTP fix: HORIZON_PMU_MTP_FIX=1 but this vLLM's %s differ from v0.29.0 "
                       "(source hash %s): fix NOT installed, vLLM stays stock", ", ".join(bad), bad)
        return False

    orig_init = kcc.HybridKVCacheCoordinator.__init__
    if getattr(orig_init, "_horizon_pmu_mtp_fix", False):   # already wrapped (re-install): wrap the original
        orig_init = orig_init.__wrapped__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        b = tail_backoff_for(self)
        self._horizon_tail_backoff = b
        self._horizon_tail_repeat = level >= 2
        for m in self.single_type_managers:
            if isinstance(m, (MambaManager, FullAttentionManager)):
                m._horizon_tail_backoff = b
                m._horizon_tail_repeat = level >= 2
        if b:
            _log().info("horizon_kernels PMU-MTP fix (level %d): EAGLE/MTP + prefix_match_unit=%d: GDN partial tail "
                        "at %s - %d, reachable after the EAGLE drop", level, self.hash_block_size,
                        "floor((N-1)/u)*u" if level >= 2 else "floor(N/u)*u", b)

    __init__.__wrapped__ = orig_init
    __init__._horizon_pmu_mtp_fix = True
    kcc.HybridKVCacheCoordinator.__init__ = __init__
    Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
    MambaManager._cache_partial_tail_block = _mamba_cache_partial_tail_block
    FullAttentionManager._cache_partial_tail_block = _attn_cache_partial_tail_block
    _installed = True
    if os.environ.get("HORIZON_PMU_MTP_FIX_QUIET", "0") != "1":
        _log().info("horizon_kernels PMU-MTP fix installed, level %d (scheduler tail stop + partial-tail "
                    "registration)", level)
    return True
