#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for horizon_kernels/pmu_mtp_fix.py (HORIZON_PMU_MTP_FIX), 27-09-2026.

They drive vLLM 0.29.0's REAL KVCacheManager (hybrid coordinator, FullAttention + Mamba "align" managers, block pool)
and the real Scheduler._mamba_block_aligned_split, with a request loop that mirrors the scheduler: prefix-cache
lookup -> chunk split -> allocate_slots -> CoW copies -> decode steps -> free. No GPU, no model.

What they show:
  - the bug (stock vLLM, EAGLE/MTP, prefix_match_unit 64, block 1600): an exact repeat of a 3,601-token prompt
    hits only 1,600 tokens, gets shared_prefix_boundary 3,520, and its prefill is cut 1600-3200-3584-3601 while
    the uncached run was cut 1600-3584-3601 (a different chunk split = different kernel shapes = drift);
  - the fix: the cached run hits 3,520 and recomputes exactly the uncached run's last chunk, for every det16
    prompt length and edge lengths, at units 32 and 64;
  - the fix is inert without EAGLE (bit-for-bit the stock chunking and hits), and follow-up turns reuse up to
    the previous prompt's last hash boundary minus one unit.

Run (Mac or server; needs vLLM 0.29.0 importable, e.g. a v0.29.0 source checkout on PYTHONPATH + torch):
    PYTHONPATH=<vllm-src>:<repo>/server python -m pytest -q -p no:cacheprovider --noconftest test_pmu_mtp_fix.py
"""
import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # server/

from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
from vllm.v1.request import Request  # noqa: E402

from horizon_kernels import pmu_mtp_fix as fix  # noqa: E402

STOCK_SPLIT = Scheduler._mamba_block_aligned_split
STOCK_TAIL = MambaManager._cache_partial_tail_block
STOCK_ATTN_TAIL = FullAttentionManager._cache_partial_tail_block
DET16 = [3601, 4313, 5217, 6388, 7107, 8216, 8897, 9812, 10599, 11895, 13002, 14336, 15617, 17228, 19903, 23426]
EDGES = [3200, 3210, 3264, 3300, 4800, 4864, 6400 + 64, 8000, 8192, 9600, 16001]
BLOCK = 1600          # production with MTP (without MTP vLLM pads to 1568)
BUDGET = 8192         # max_num_batched_tokens


def _restore_stock():
    from vllm.v1.core import kv_cache_coordinator as kcc
    Scheduler._mamba_block_aligned_split = STOCK_SPLIT
    MambaManager._cache_partial_tail_block = STOCK_TAIL
    FullAttentionManager._cache_partial_tail_block = STOCK_ATTN_TAIL
    init = kcc.HybridKVCacheCoordinator.__init__
    if getattr(init, "_horizon_pmu_mtp_fix", False):
        kcc.HybridKVCacheCoordinator.__init__ = init.__wrapped__
    fix._installed = False


@pytest.fixture(autouse=True)
def _hash_and_stock():
    init_none_hash(sha256)
    yield
    _restore_stock()   # never leak a patched class into other tests


class Sim:
    """One vLLM instance: a hybrid KV-cache manager (1 attention + 1 GDN group, like Qwen3.8's 16 + 48 layers after
    grouping) and a scheduler stand-in with the attributes _mamba_block_aligned_split reads."""

    def __init__(self, unit, eagle, patched, block=BLOCK, num_blocks=600, level=1):
        _restore_stock()
        if patched:
            assert fix.install(level=level)
        self.eagle, self.unit, self.block = eagle, unit, block
        self.lookahead = 3 if eagle else 0
        cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
            KVCacheGroupSpec(["attn"], FullAttentionSpec(block_size=block, num_kv_heads=1, head_size=1,
                                                         dtype=torch.float32)),
            KVCacheGroupSpec(["gdn"], MambaSpec(block_size=block, shapes=(1, 1), dtypes=(torch.float32,),
                                                mamba_cache_mode="align",
                                                num_speculative_blocks=3 if eagle else 0)),
        ])
        self.m = KVCacheManager(cfg, max_model_len=262144, scheduler_block_size=block, hash_block_size=unit,
                                enable_caching=True, use_eagle=eagle, num_prefill_lookahead=1 if eagle else 0)
        self.s = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=block), max_num_scheduled_tokens=BUDGET,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0), use_eagle=eagle,
            hash_block_size=unit, mamba_partial_cache_hit=self.m.coordinator.enable_partial_hash_hits,
            mamba_has_prefill_checkpoint_blocks=False, kv_cache_manager=self.m)
        self.n = 0

    def split(self, *a):
        return Scheduler._mamba_block_aligned_split(self.s, *a)

    def _step_done(self):
        _, retained = self.m.take_kv_cache_block_copies()
        self.m.block_pool.free_blocks(retained)
        self.m.new_step_starts()

    def run(self, tokens, n_out=4, salt="s"):
        """Prefill `tokens` (chunked like the scheduler), decode n_out tokens, free. Returns a record."""
        self.n += 1
        sp = SamplingParams(max_tokens=64)
        sp.update_from_generation_config({}, eos_token_id=10 ** 9)
        req = Request(request_id=f"r{self.n}", prompt_token_ids=list(tokens), sampling_params=sp, pooling_params=None,
                      cache_salt=salt, block_hasher=get_request_block_hasher(self.unit, sha256))
        blocks, hit, spb = self.m.get_computed_blocks(req)
        req.shared_prefix_boundary = spb
        N, pos, chunks = len(tokens), hit, []
        while pos < N:
            want = min(BUDGET, N - pos)
            if not chunks:
                k = self.split(req, want, hit, 0)
                assert k > 0
                assert self.m.allocate_slots(req, k, hit, blocks, num_lookahead_tokens=self.lookahead) is not None
                req.num_computed_tokens = hit + k
            else:
                k = self.split(req, want)
                assert k > 0
                assert self.m.allocate_slots(req, k, num_lookahead_tokens=self.lookahead) is not None
                req.num_computed_tokens += k
            chunks.append((pos, pos + k))
            pos += k
            self._step_done()
        out = []
        for i in range(n_out):   # decode: the sampled token enters the next step
            t = 900000 + i
            req.append_output_token_ids([t])
            out.append(t)
            assert self.m.allocate_slots(req, 1, num_lookahead_tokens=self.lookahead) is not None
            req.num_computed_tokens += 1
            self._step_done()
        self.m.free(req)
        self._step_done()
        return SimpleNamespace(hit=hit, spb=spb, chunks=chunks, out=out, N=N)


def prompt(n, seed=0):
    rng = random.Random(seed)
    return [rng.randrange(1, 150000) for _ in range(n)]


def tail_of(chunks, start):
    """The uncached run's chunk ends from `start` on (what a cached run resuming at `start` must reproduce)."""
    return [(max(a, start), b) for a, b in chunks if b > start]


def exact_repeat(unit, eagle, patched, N, level=1):
    sim = Sim(unit, eagle, patched, level=level)
    toks = prompt(N, seed=N)
    u1 = sim.run(toks, salt="X")
    c = sim.run(toks, salt="X")
    return u1, c


# ---------------------------------------------------------------------------------------------------------------
def test_pure_boundary_math():
    assert fix.mamba_tail_boundary(3601, 64, 0) == fix.attn_tail_boundary(3601, 64, 0) == 3584   # stock
    assert fix.mamba_tail_boundary(3601, 64, 0, True) == 3584                                   # no EAGLE: stock
    assert fix.attn_tail_boundary(3601, 64, 64) == fix.attn_tail_boundary(3601, 64, 64, True) == 3584
    assert fix.mamba_tail_boundary(3601, 64, 64) == fix.mamba_tail_boundary(3601, 64, 64, True) == 3520
    assert fix.attn_tail_boundary(14336, 64, 64) == 14336            # level 1 keeps stock's attention entry
    assert fix.mamba_tail_boundary(14336, 64, 64) == 14272
    assert fix.attn_tail_boundary(14336, 64, 64, True) == 14272      # level 2: u divides N
    assert fix.mamba_tail_boundary(14336, 64, 64, True) == 14208
    assert fix.mamba_tail_boundary(63, 64, 64) == 0


def test_bug_reproduced_on_stock_vllm():
    """Experiment 4's first prompt (3,601 tokens), stock vLLM 0.29.0 with MTP and prefix_match_unit 64."""
    u1, c = exact_repeat(64, eagle=True, patched=False, N=3601)
    assert u1.hit == 0 and u1.chunks == [(0, 1600), (1600, 3584), (3584, 3601)]
    assert c.hit == 1600                 # the same 1,600-token hit as without prefix_match_unit (det16: cached 1600)
    assert c.spb == 3520                 # attention matched 3,584, dropped one unit (EAGLE); Mamba had no state there
    assert c.chunks == [(1600, 3200), (3200, 3584), (3584, 3601)]   # extra Marconi junction stop at 3,200
    assert c.chunks != tail_of(u1.chunks, c.hit)                    # -> different kernel shapes -> drift


def test_fix_exact_repeat_3601():
    u1, c = exact_repeat(64, eagle=True, patched=True, N=3601)
    assert u1.chunks == [(0, 1600), (1600, 3520), (3520, 3601)]
    assert c.hit == 3520 and c.spb == 0
    assert c.chunks == [(3520, 3601)] == tail_of(u1.chunks, c.hit)


def _stock_matches(unit, N):
    u1, c = exact_repeat(unit, True, False, N)
    return c.chunks == tail_of(u1.chunks, c.hit)


def test_stock_mismatch_pattern_matches_experiment4():
    """Stock + MTP + unit 64 over the det16 prompt lengths: the chunk split differs for every prompt except
    k=11 (14,336 = 224 x 64), which is exactly the one prompt with zero log-prob drift in BOTH MTP arms of
    experiment 4 (D3 and D1). Hits are the 1,600-token multiples det16 reported."""
    det16_cached = [1600, 1600, 3200, 3200, 4800, 6400, 6400, 8000, 8000, 9600, 11200, 11200, 12800, 14400, 17600,
                    20800]
    same = []
    for k, N in enumerate(DET16):
        u1, c = exact_repeat(64, True, False, N)
        assert c.hit == det16_cached[k], (k, N, c.hit)
        if c.chunks == tail_of(u1.chunks, c.hit):
            same.append(k)
    assert same == [11]


@pytest.mark.parametrize("unit", [32, 64])
@pytest.mark.parametrize("N", DET16 + EDGES)
def test_fix_level1_exact_repeat(unit, N):
    """Level 1 (default): the repeat recomputes exactly the uncached run's chunks for every length except a multiple
    of the block size, which stock vLLM gets wrong even without prefix_match_unit (see the next test); when u does
    not divide N it reuses everything up to T - u."""
    if N % BLOCK == 0:
        pytest.skip("pre-existing stock corner (also without prefix_match_unit); fixed by level 2")
    u1, c = exact_repeat(unit, eagle=True, patched=True, N=N, level=1)
    assert c.chunks == tail_of(u1.chunks, c.hit), (u1.chunks, c.hit, c.spb, c.chunks)
    T = N // unit * unit
    if N % unit:
        assert c.hit == T - unit and c.spb == 0
        assert N - c.hit < 2 * unit
    else:   # a repeat matches at most N - 1 < T: block-boundary hit as in stock, same chunk split
        assert c.hit % BLOCK == 0


@pytest.mark.parametrize("N", [4800, 8000, 9600, 16000])
def test_stock_block_multiple_corner_exists_without_pmu(N):
    """Documented, not ours: N a multiple of the block, stock vLLM WITHOUT prefix_match_unit (unit == block) and
    MTP: the repeat gets a junction split the uncached run did not have."""
    u1, c = exact_repeat(BLOCK, eagle=True, patched=False, N=N)
    assert c.spb > 0 and c.chunks != tail_of(u1.chunks, c.hit)


@pytest.mark.parametrize("unit", [32, 64])
@pytest.mark.parametrize("N", DET16 + EDGES)
def test_fix_level2_exact_repeat(unit, N):
    """Level 2: every length, including multiples of the unit or the block: hit R - u, no junction, the uncached
    run's own last chunk(s)."""
    u1, c = exact_repeat(unit, eagle=True, patched=True, N=N, level=2)
    R = (N - 1) // unit * unit
    assert c.hit == R - unit and c.spb == 0, (c.hit, R - unit, c.spb)
    assert c.chunks == tail_of(u1.chunks, c.hit), (u1.chunks, c.hit, c.spb, c.chunks)
    assert N - c.hit <= 2 * unit


@pytest.mark.parametrize("unit", [32, 64])
@pytest.mark.parametrize("N", DET16 + EDGES)
@pytest.mark.parametrize("level", [1, 2])
def test_fix_inert_without_eagle(unit, N, level):
    """No EAGLE/MTP: the patched functions give exactly stock vLLM's chunking and hits (backoff 0)."""
    a = exact_repeat(unit, eagle=False, patched=False, N=N)
    b = exact_repeat(unit, eagle=False, patched=True, N=N, level=level)
    assert [(r.hit, r.spb, r.chunks) for r in a] == [(r.hit, r.spb, r.chunks) for r in b]


def test_split_differential_backoff_zero():
    """Randomized: the copied split function equals vLLM's whenever the backoff is 0."""
    rng = random.Random(1)
    for _ in range(4000):
        N = rng.randrange(1, 40000)
        bs = rng.choice([512, 1568, 1600])
        unit = rng.choice([u for u in (16, 32, 64, 320) if bs % u == 0])
        req = SimpleNamespace(num_prompt_tokens=N, num_tokens=N + rng.choice([0, 0, 1, 5]),
                              num_computed_tokens=rng.randrange(0, N), shared_prefix_boundary=rng.choice(
                                  [0, 0, rng.randrange(0, N + 1)]))
        s = SimpleNamespace(cache_config=SimpleNamespace(block_size=bs),
                            max_num_scheduled_tokens=rng.choice([2048, 8192]),
                            scheduler_config=SimpleNamespace(long_prefill_token_threshold=rng.choice([0, 0, 384])),
                            use_eagle=False, hash_block_size=unit, mamba_partial_cache_hit=rng.random() < 0.7,
                            mamba_has_prefill_checkpoint_blocks=rng.random() < 0.2)
        want = rng.randrange(1, 9000)
        extra = rng.choice([0, 0, rng.randrange(0, 64)])
        assert fix._mamba_block_aligned_split(s, req, want, extra, 0) == STOCK_SPLIT(s, req, want, extra, 0)


@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("N1", [9812, 14336])
def test_followup_turn_reuses_previous_prompt(level, N1):
    """Agent turn 2 = turn 1's prompt + its answer + new tool output. With the fix and MTP, turn 2 reuses turn 1's
    prompt up to its last reachable hash boundary; stock vLLM falls back to a block boundary."""
    t1 = prompt(N1, 7)
    hits = {}
    for patched in (False, True):
        sim = Sim(64, True, patched, level=level)
        r1 = sim.run(t1, n_out=40)
        t2 = t1 + r1.out + prompt(1500, 8)
        hits[patched] = sim.run(t2, n_out=4).hit
    assert hits[False] == {9812: 8000, 14336: 11200}[N1]
    T = N1 // 64 * 64
    if N1 % 64 == 0 and level == 2:
        assert hits[True] == T - 128      # the level-2 price: one unit
    else:
        assert hits[True] == T - 64
    # a cold run of the same turn-2 prompt (other salt) and the cached run end in the same final chunk
    sim = Sim(64, True, True, level=level)
    r1 = sim.run(t1, n_out=40, salt="A")
    t2 = t1 + r1.out + prompt(1500, 8)
    cold = sim.run(t2, salt="B")
    warm = sim.run(t2, salt="A")
    assert warm.chunks[-1] == cold.chunks[-1]


def test_backoff_derived_from_coordinator():
    assert Sim(64, True, True).m.coordinator._horizon_tail_backoff == 64
    assert Sim(64, False, True).m.coordinator._horizon_tail_backoff == 0
    # unit == block: no partial hits, no backoff
    assert Sim(1600, True, True).m.coordinator._horizon_tail_backoff == 0


def test_install_refuses_other_vllm_sources(monkeypatch):
    fix._installed = False
    monkeypatch.setitem(fix._EXPECTED_SRC, "Scheduler._mamba_block_aligned_split", "0" * 16)
    assert fix.install() is False
    assert Scheduler._mamba_block_aligned_split is STOCK_SPLIT
