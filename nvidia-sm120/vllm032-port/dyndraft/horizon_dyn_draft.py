# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Draft count by load (27-09-2026): makes vLLM 0.29's batch-size draft schedule usable with our MTP setup.
# 29-09-2026 port to vLLM 0.30.0: both gaps are still there, unchanged (CudaGraphManager._init_candidates expands the
# schedule exactly as in 0.29, and the V2 runner's execute_model still never reads num_spec_tokens_to_schedule; only
# the V1 gpu_model_runner does). Same hooks; the install line now names the vLLM version. 0.30's adaptive verification
# (enable_adaptive_verification, off by default) is refused by vLLM together with use_local_argmax_reduction, which
# production uses, so the two cannot overlap.
#
# vLLM 0.29 has `num_speculative_tokens_per_batch_size` in the speculative config: a list of inclusive ranges
# [[first_batch, last_batch, drafts], ...]. Each scheduler step looks up the number of scheduled requests and
# asks for that many drafts in the NEXT step (async scheduling: the placeholders it leaves per request decide how
# many draft tokens the next step verifies). On the V2 model runner with MTP two things are missing:
#
# 1. Startup crash (ZeroDivisionError, 26-09-2026). CudaGraphManager._init_candidates expands the decode query
#    length per schedule entry as `drafts + (decode_query_len - num_speculative_tokens)`. That is right for the
#    target model (query length drafts + 1) and the draft prefill, but the draft *decode* manager always runs one
#    token per request (decode_query_len = 1), so an entry below the maximum gives a query length <= 0 and
#    round_up(x, 0) divides by zero. Fix: a manager whose query length does not depend on the draft count (1 token
#    per request) keeps its single graph set.
# 2. The drafter ignores the schedule. The V1 runner passes the scheduled count to its proposer; the V2 runner never
#    reads scheduler_output.num_spec_tokens_to_schedule, so MTP always drafts the maximum and only verification
#    shrinks. Fix: remember the scheduled count in execute_model and let the speculator's propose() run only that
#    many draft steps (the per-step decode graph is the same graph, only replayed fewer times). Unused draft
#    columns stay stale and are never read: the next step verifies exactly the scheduled count.
#
# Output quality cannot change: the main model verifies every token, drafts only decide how many are accepted.
# Active only when VLLM_SPEC carries num_speculative_tokens_per_batch_size (every hook is a no-op otherwise);
# HORIZON_DYN_DRAFTS=0 turns the hooks off entirely, HORIZON_DYN_DRAFT_TRUNC=0 keeps fix 1 only (vLLM's behaviour:
# full drafting, shorter verification). HORIZON_DYN_DRAFTS_LOG=N logs the draft-count histogram every N steps.
import collections
import logging
import os

_log = logging.getLogger("horizon_kernels.dyn_draft")
TESTED = ("0.29.0", "0.30.0")   # versions whose hooked internals were checked
_done = False


def _install_graph_fix():
    from vllm.v1.worker.gpu import cudagraph_utils as cgu

    orig = cgu.CudaGraphManager._init_candidates
    if getattr(orig, "_horizon_dyn", False):
        return

    def _init_candidates(self):
        sc = self.vllm_config.speculative_config
        if (sc is not None and sc.uses_dynamic_speculative_decoding()
                and self.decode_query_len <= self.vllm_config.num_speculative_tokens):
            # one token per request per draft step, whatever the draft count: no per-count graphs needed
            saved = sc.num_speculative_tokens_per_batch_size
            sc.num_speculative_tokens_per_batch_size = None
            try:
                return orig(self)
            finally:
                sc.num_speculative_tokens_per_batch_size = saved
        return orig(self)

    _init_candidates._horizon_dyn = True
    cgu.CudaGraphManager._init_candidates = _init_candidates


def _install_draft_trunc():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

    if getattr(GPUModelRunner.execute_model, "_horizon_dyn", False):
        return
    every = int(os.environ.get("HORIZON_DYN_DRAFTS_LOG", "0") or 0)
    orig_exec = GPUModelRunner.execute_model
    orig_propose = AutoRegressiveSpeculator.propose

    def execute_model(self, scheduler_output, *args, **kwargs):
        sp = getattr(self, "speculator", None)
        if isinstance(sp, AutoRegressiveSpeculator):
            k = 0 if kwargs.get("dummy_run", False) else int(getattr(scheduler_output, "num_spec_tokens_to_schedule", 0) or 0)
            sp._horizon_k = k if 0 < k < sp.num_speculative_steps else None
        return orig_exec(self, scheduler_output, *args, **kwargs)

    def propose(self, *args, **kwargs):
        k = getattr(self, "_horizon_k", None)
        self._horizon_k = None
        full = self.num_speculative_steps
        if every:
            h = self.__dict__.setdefault("_horizon_hist", collections.Counter())
            h[k or full] += 1
            if sum(h.values()) % every == 0:
                _log.warning("horizon dyn_draft: draft steps per propose (count): %s", dict(sorted(h.items())))
        if k is None or kwargs.get("dummy_run", False) or kwargs.get("is_profile", False):
            return orig_propose(self, *args, **kwargs)
        self.num_speculative_steps = k
        try:
            return orig_propose(self, *args, **kwargs)
        finally:
            self.num_speculative_steps = full

    execute_model._horizon_dyn = True
    GPUModelRunner.execute_model = execute_model
    AutoRegressiveSpeculator.propose = propose


def _vllm_version():
    try:
        from importlib.metadata import version as _v
        return _v("vllm").split("+")[0]
    except Exception:  # noqa: BLE001
        return "?"


def _native_upstream(ver):
    """vLLM >= 0.32 (checked in the 0.32.0 source, 24-10-2026) does both jobs itself: CudaGraphManager._init_candidates
    only expands the schedule when decode_query_len > num_speculative_tokens (the ZeroDivisionError guard), the V2 runner
    passes scheduler_output.num_spec_tokens_to_schedule into speculator.propose(num_speculative_tokens=...) (draft steps
    truncated, unused draft columns set to -1) and the draft-decode graphs are specialised per draft count
    (SpeculatorCudaGraphManager.specialize_spec_tokens). The 0.30 hooks (AutoRegressiveSpeculator, which 0.32 no longer
    has: it became target_dependent_ar.TargetDependentARSpeculator) are not needed and must not run."""
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
    except Exception:  # noqa: BLE001
        return False
    return (major, minor) >= (0, 32)


def register():
    global _done
    if _done or os.environ.get("HORIZON_DYN_DRAFTS", "1") == "0":
        return
    ver = _vllm_version()
    if _native_upstream(ver):
        _done = True
        _log.warning("horizon dyn_draft: not installed, native in vllm %s (per-batch draft count + decode-graph guard are upstream)", ver)
        return
    spec = os.environ.get("VLLM_SPEC", "")
    # The env file is read by the service manager; vLLM gets the spec as a CLI flag. Hooks are no-ops without a schedule, so
    # install whenever the variable is unknown or mentions the schedule.
    if spec and "num_speculative_tokens_per_batch_size" not in spec:
        return
    _done = True
    try:
        _install_graph_fix()
        if os.environ.get("HORIZON_DYN_DRAFT_TRUNC", "1") != "0":
            _install_draft_trunc()
        try:
            from importlib.metadata import version as _v
            ver = _v("vllm")
        except Exception:  # noqa: BLE001
            ver = "?"
        _log.warning("horizon dyn_draft: installed (trunc=%s) vllm=%s%s", os.environ.get("HORIZON_DYN_DRAFT_TRUNC", "1"),
                     ver, "" if ver.split("+")[0] in TESTED else " (untested version)")
    except Exception as e:  # noqa: BLE001  (vLLM layout changed: stay stock rather than break startup)
        _log.warning("horizon dyn_draft: not installed (%s)", e)
