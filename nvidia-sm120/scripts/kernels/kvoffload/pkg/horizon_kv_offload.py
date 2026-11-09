# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kv_offload (Horizon Tech, 30-09-2026): guard rails around vLLM 0.30's NATIVE KV offloading
# (OffloadingConnector + CPUOffloadingSpec) for the hybrid Qwen3.8 27B on the test server. vLLM general plugin.
#
# Off unless HORIZON_KV_OFFLOAD=1. With it off nothing is patched and nothing is logged, so the package can sit on
# PYTHONPATH in any env. The offload itself is vLLM's own code, switched on by `--kv-offloading-size <GiB>`
# (backend "native"); this plugin only adds what the box needs around it:
#
#   1. RAM guard (API-server process, before the engine starts): the CPU tier is pinned memory, committed at start
#      (MADV_POPULATE_WRITE + cudaHostRegister); an out-of-memory kill would take the whole server down.
#      It needs MemAvailable >= size + HORIZON_KV_OFFLOAD_VLLM_GB (vLLM's own resident memory, default 9)
#      + HORIZON_KV_OFFLOAD_MIN_FREE_GB (headroom for everything else, default 16). If that fails the size is
#      shrunk to fit; below HORIZON_KV_OFFLOAD_MIN_GB (default 2) offloading is switched OFF (no connector at all,
#      the configuration tested). The decision is taken once and written into the connector's extra config, so
#      the scheduler and the worker see the same size.
#   2. Region: vLLM puts the tier in /dev/shm, a size-limited tmpfs. HORIZON_KV_OFFLOAD_REGION=auto (default)
#      keeps /dev/shm when the size fits in its free space minus 256 MiB and otherwise uses vLLM's private pinned
#      tensor path (TP=1: nothing needs the shared file); shm / private force one.
#   3. KV layout: the OffloadingConnector asks for LBHNC; the tested configuration resolves LBNHC. Unless VLLM_KV_CACHE_LAYOUT
#      is set, the plugin pins it to LBNHC so enabling the tier does not also change the attention KV layout.
#   4. Policy knobs into kv_connector_extra_config (which a whitespace-split VLLM_EXTRA_ARGS cannot carry as
#      JSON): HORIZON_KV_OFFLOAD_POLICY (lru | arc), HORIZON_KV_OFFLOAD_STORE_THRESHOLD (>= 2: a chunk is stored
#      only after being offered that often; NOT useful for growing conversations, whose GDN boundary states are
#      offered once each).
#   5. HORIZON_KV_OFFLOAD_MIN_PROMPT=N (default 0 = off): requests with fewer than N prompt tokens never STORE into
#      the CPU tier (they still look it up). One offloaded chunk is a whole 1,600-token hybrid block of all 17
#      attention layers or of 16 GDN layers (53 MiB), and with sub-block caching every prompt's tail boundary is
#      offered as 4 such chunks (212 MiB), so short one-off chats would otherwise flush the small tier that the long
#      agent conversations need; a short prompt is also cheap to recompute (4k tokens ~0.7 s).
#   6. HORIZON_KV_OFFLOAD_TEST_RESET=<prefix> (measure-only, for the hz determinism extra; default off): when the file
#      <prefix>.req exists at the start of a scheduler step, the GPU prefix cache is reset (not the CPU tier, no
#      running request touched) before that step schedules anything, and <prefix>.done records the result. A
#      client touches .req and then sends the request: its hit must now come from the CPU tier. This replaces
#      VLLM_SERVER_DEV_MODE, which would expose /reset_prefix_cache (and other dev endpoints) to every client.
#   7. HORIZON_KV_OFFLOAD_CONNECTOR=simple selects vLLM's other native connector, SimpleCPUOffloadConnector (sets
#      VLLM_USE_SIMPLE_KV_OFFLOAD=1): a CPU mirror of the GPU's hybrid prefix cache (its own HybridKVCacheCoordinator),
#      private pinned tensors (no /dev/shm limit), and with HORIZON_KV_OFFLOAD_LAZY=1 its victim-cache mode, which
#      stores GPU blocks only as they near eviction (GPU + CPU capacity instead of the smaller of the two).
#      Items 2 and 5 apply to the default OffloadingConnector only.
#   8. HORIZON_KV_OFFLOAD_KEEP_ON_SLEEP=1 (default off): vLLM 0.30's sleep
#      (level >= 1) pauses the scheduler with clear_cache=True, and EngineCore._reset_caches then also resets the
#      connector (reset_connector=True), which empties the CPU tier's index although its pinned host memory survives
#      the sleep untouched. With this switch _reset_caches keeps the connector: the GPU prefix cache is still reset
#      (its KV memory is discarded by the sleep), the CPU tier's chunks stay valid and serve hits after the wake.
#      The pause already synchronized the device, so no transfer is in flight when the GPU KV is unmapped. Runtime
#      override for A/B: while the file <HORIZON_KV_OFFLOAD_TEST_RESET>.nokeep exists, the stock reset applies.
#
# Every patched function is guarded by its vLLM 0.30.0 source hash; on any other vLLM nothing is patched and a
# warning says so (the stock connector would then run without the RAM guard, so the hz spec also checks the marker).
import hashlib
import inspect
import os

_done = False
MARK = "horizon kv_offload"

_EXPECTED_SRC = {  # sha256(inspect.getsource(fn))[:16], vLLM 0.30.0
    "VllmConfig._post_init_kv_transfer_config": "2cf51ff9e5d6b102",
    "CPUOffloadingSpec._uses_shared_region": "8cf954c1c0829827",
    "OffloadingConnectorScheduler.on_new_request": "cdd26d480cb06df6",
    "CPUOffloadingManager.prepare_store": "3e209769d0a2ed0a",
    "Scheduler.schedule": "a04a9c05b0193d47",
    "EngineCore._reset_caches": "f788041c1160bc77",
    "Worker.wake_up": "92503589f792ad6a",
    "OffloadingConnectorScheduler._lookup": "73d3e0be0ad4940a",
    "OffloadingConnectorScheduler._build_partial_tail_store_jobs": "db0f5789b064bfdb",
    "OffloadingConnectorScheduler._lookup_complete_chunks": "a674cd822157b5c0",
}
_SHORT_MAX = 100_000        # bounded memory of short request ids (FIFO)
GIB = 1 << 30


# ---------------------------------------------------------------- pure helpers (CPU-tested)
def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def read_meminfo(path: str = "/proc/meminfo") -> dict:
    """kB values of /proc/meminfo as bytes."""
    out = {}
    with open(path) as f:
        for line in f:
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[k.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def shm_free_bytes(path: str = "/dev/shm") -> int | None:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize
    except OSError:
        return None


def ram_guard(requested: int, mem_available: int, vllm_bytes: int, min_free: int, min_size: int) -> tuple[int, str]:
    """-> (bytes to use, reason). 0 means: switch offloading off."""
    room = mem_available - vllm_bytes - min_free
    if requested <= room:
        return requested, "fits"
    if room >= min_size:
        return int(room), f"shrunk: MemAvailable {mem_available / GIB:.1f} GiB - vLLM {vllm_bytes / GIB:.1f} - " \
                          f"headroom {min_free / GIB:.1f} leaves {room / GIB:.2f} GiB"
    return 0, f"off: MemAvailable {mem_available / GIB:.1f} GiB leaves {max(room, 0) / GIB:.2f} GiB after vLLM " \
              f"{vllm_bytes / GIB:.1f} + headroom {min_free / GIB:.1f} (< minimum {min_size / GIB:.1f})"


def choose_region(mode: str, requested: int, shm_free: int | None, margin: int = 256 << 20) -> str:
    if mode in ("shm", "private"):
        return mode
    if shm_free is not None and requested + margin <= shm_free:
        return "shm"
    return "private"


def extra_from_env() -> dict:
    extra = {}
    pol = os.environ.get("HORIZON_KV_OFFLOAD_POLICY")
    if pol:
        extra["eviction_policy"] = pol
    thr = os.environ.get("HORIZON_KV_OFFLOAD_STORE_THRESHOLD")
    if thr:
        extra["store_threshold"] = int(thr)
    if os.environ.get("HORIZON_KV_OFFLOAD_LAZY", "0") == "1":
        extra["lazy_offload"] = True       # SimpleCPUOffloadConnector only
    return extra


def decide(requested: int, meminfo: dict, shm_free: int | None) -> dict:
    """The whole config-time decision, from the environment and the measured memory."""
    use, why = ram_guard(requested, meminfo.get("MemAvailable", 0),
                         int(env_float("HORIZON_KV_OFFLOAD_VLLM_GB", 9) * GIB),
                         int(env_float("HORIZON_KV_OFFLOAD_MIN_FREE_GB", 16) * GIB),
                         int(env_float("HORIZON_KV_OFFLOAD_MIN_GB", 2) * GIB))
    region = choose_region(os.environ.get("HORIZON_KV_OFFLOAD_REGION", "auto"), use, shm_free) if use else None
    return {"bytes": use, "reason": why, "region": region, "extra": extra_from_env() if use else {}}


def _log():
    try:
        from vllm.logger import init_logger
    except ImportError:          # CPU tests without vLLM
        import logging
        return logging.getLogger("horizon_kernels.kv_offload")
    return init_logger("vllm.horizon_kernels.kv_offload")


def _src_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- vLLM hooks
def _post_init_hook(orig):
    def wrapped(self):
        cc = self.cache_config
        size = cc.kv_offloading_size
        if size is not None and cc.kv_offloading_backend == "native":
            ex = (self.kv_transfer_config.kv_connector_extra_config
                  if self.kv_transfer_config is not None else {})
            if "horizon_region" not in ex:   # decide once (a config built twice must not re-measure)
                d = decide(int(size * GIB), read_meminfo(os.environ.get("HORIZON_KV_OFFLOAD_MEMINFO", "/proc/meminfo")),
                           shm_free_bytes())
                if not d["bytes"]:
                    _log().warning("%s: RAM guard switched offloading OFF (%s); serving exactly as without "
                                 "--kv-offloading-size", MARK, d["reason"])
                    cc.kv_offloading_size = None
                    return orig(self)
                cc.kv_offloading_size = d["bytes"] / GIB
                orig(self)
                ex = self.kv_transfer_config.kv_connector_extra_config
                ex.update(d["extra"])
                if self.kv_transfer_config.kv_connector == "SimpleCPUOffloadConnector":
                    d["region"] = "private (simple connector)"
                ex["horizon_region"] = d["region"]
                _log().warning("%s: on (native %s, %.2f GiB CPU tier, region %s, RAM guard %s, extra %s, KV layout %s)",
                             MARK, self.kv_transfer_config.kv_connector, d["bytes"] / GIB, d["region"], d["reason"],
                             d["extra"] or "{}", os.environ.get("VLLM_KV_CACHE_LAYOUT"))
                return None
        return orig(self)

    wrapped._horizon_kv_offload = True
    return wrapped


def _uses_shared_region_hook(orig):
    def wrapped(self):
        if self.extra_config.get("horizon_region") == "private":
            return False
        return orig(self)

    wrapped._horizon_kv_offload = True
    return wrapped


def _short_filter_hooks(on_new_orig, prepare_orig, min_prompt: int):
    """Requests below min_prompt prompt tokens store nothing: remembered at on_new_request (scheduler side, same
    process as the CPU manager), filtered in prepare_store by the request id every store call carries."""
    from collections import OrderedDict
    short: OrderedDict = OrderedDict()

    def on_new_request(self, request):
        if request.num_prompt_tokens < min_prompt:
            short[request.request_id] = None
            if len(short) > _SHORT_MAX:
                short.popitem(last=False)
        return on_new_orig(self, request)

    def prepare_store(self, keys, req_context):
        if req_context.req_id in short:
            keys = []          # the manager's own empty-store answer (keys_to_store == [])
        return prepare_orig(self, keys, req_context)

    on_new_request._horizon_kv_offload = prepare_store._horizon_kv_offload = True
    on_new_request._horizon_short = short
    return on_new_request, prepare_store


def _test_reset_hook(orig, prefix: str):
    def schedule(self, *args, **kwargs):
        req = prefix + ".req"
        if os.path.exists(req):
            try:
                os.unlink(req)
                ok = bool(self.kv_cache_manager.reset_prefix_cache())   # GPU prefix cache only
                with open(prefix + ".done", "w") as f:
                    f.write(f"{int(ok)} waiting={len(self.waiting)} running={len(self.running)}\n")
            except Exception as e:  # noqa: BLE001  measure-only
                with open(prefix + ".done", "w") as f:
                    f.write(f"0 error={e!r}\n")
        return orig(self, *args, **kwargs)

    schedule._horizon_kv_offload = True
    return schedule


def _keep_on_sleep_hook(orig, nokeep_file):
    def _reset_caches(self, reset_running_requests: bool = True, reset_connector: bool = True):
        if reset_connector and not (nokeep_file and os.path.exists(nokeep_file)):
            _log().info("%s: cache reset (sleep/pause): GPU prefix cache reset, CPU tier KEPT", MARK)
            return orig(self, reset_running_requests=reset_running_requests, reset_connector=False)
        return orig(self, reset_running_requests=reset_running_requests, reset_connector=reset_connector)

    _reset_caches._horizon_kv_offload = True
    return _reset_caches


def install_keep_on_sleep() -> bool:
    """Item 8, engine-core side. Separate from install(): needs only EngineCore."""
    from vllm.v1.engine.core import EngineCore
    if getattr(EngineCore._reset_caches, "_horizon_kv_offload", False):
        return True
    h = _src_hash(EngineCore._reset_caches)
    if h != _EXPECTED_SRC["EngineCore._reset_caches"]:
        _log().warning("%s: KEEP_ON_SLEEP=1 but EngineCore._reset_caches differs from v0.30.0 (hash %s): CPU tier is "
                       "reset on sleep (stock)", MARK, h)
        return False
    reset = os.environ.get("HORIZON_KV_OFFLOAD_TEST_RESET")
    EngineCore._reset_caches = _keep_on_sleep_hook(EngineCore._reset_caches, reset + ".nokeep" if reset else None)
    _log().warning("%s: CPU tier kept across sleep/pause cache resets (KEEP_ON_SLEEP)", MARK)
    return True


# ---------------------------------------------------------------- items 9-11: partial tails on the PI caching base
_DIAG = {"f": None}


def _diag(ev: dict) -> None:
    f = _DIAG["f"]
    if f is None:
        return
    import json
    import time
    try:
        f.write(json.dumps(dict(ev, t=round(time.time(), 3))) + "\n")
        f.flush()
    except Exception:  # noqa: BLE001  measure-only
        pass


def split_partial_handoffs(handoffs: dict, block_size: int, cow_groups) -> tuple[dict, list, list]:
    """Item 9, pure. handoffs = {req: [(group, block_id, boundary), ...]}. -> (first, extra, skipped):
    `first` = the handoffs with every aligned entry and, per request, only the entries of its smallest complete sub-block
    boundary; `extra` = [{req: entries of one further boundary}, ...] in ascending boundary order; `skipped` =
    [(req, boundary, missing groups)] for boundaries whose hand-off does not cover every GDN (cow-source) group."""
    cow = set(cow_groups)
    first, extra, skipped = {}, [], []
    for req, entries in handoffs.items():
        aligned = [e for e in entries if block_size == 0 or e[2] % block_size == 0]
        by_b = {}
        for e in entries:
            if block_size and e[2] % block_size != 0:
                by_b.setdefault(e[2], []).append(e)
        complete = []
        for b in sorted(by_b):
            missing = sorted(cow - {g for g, _, _ in by_b[b]})
            if missing:
                skipped.append((req, b, missing))
            else:
                complete.append(b)
        first[req] = aligned + (by_b[complete[0]] if complete else [])
        for i, b in enumerate(complete[1:]):
            if len(extra) <= i:
                extra.append({})
            extra[i][req] = by_b[b]
    return first, extra, skipped


def drop_past_window(handoffs: dict, block_size: int, max_by_req: dict) -> tuple[dict, list]:
    """Item 9b, pure. Drops the SUB-BLOCK entries (boundary % block_size != 0) whose boundary lies past the request's
    save window (max_by_req[req] = min(num_prompt_tokens, max_offload_tokens); None / missing = request unknown to the
    connector) -> (kept, dropped [(req, boundary, max)]). Aligned entries are kept: vLLM's aligned path filters them
    against the same window itself (`boundary > max_boundary: continue`). Cause (kvoff-0610 screen, 06-10-2026 19:54
    UTC, `assert boundary <= max_boundary`): a request preempted under KV pressure re-prefills prompt + generated
    tokens, and the in-step GDN checkpoint of that resumed prefill (gdn_ckpt) lands past the prompt; vLLM's partial-tail
    store asserts instead of filtering (stock Qwen GDN has no checkpoint, its producer tail is bounded by the prompt)."""
    kept, dropped = {}, []
    for req, entries in handoffs.items():
        mx = max_by_req.get(req)
        keep = []
        for e in entries:
            if block_size and e[2] % block_size != 0 and (mx is None or e[2] > mx):
                dropped.append((req, e[2], mx))
            else:
                keep.append(e)
        if keep:
            kept[req] = keep
    return kept, dropped


def _multi_tail_hook(orig):
    import copy
    import dataclasses

    def _build_partial_tail_store_jobs(self, scheduler_output):
        bs = scheduler_output.kv_connector_block_state
        handoffs = bs.boundary_state_offloads if bs else None
        if handoffs and self.config.supports_partial_tail and _DIAG["f"] is not None:
            for req, entries in handoffs.items():
                rs = self._req_status.get(req)
                _diag({"ev": "handoff", "req": req, "entries": [list(e) for e in entries],
                       "prompt": rs.req.num_prompt_tokens if rs else None,
                       "computed": rs.req.num_computed_tokens if rs else None})
        if not handoffs or not self.config.supports_partial_tail or not _MULTI_TAIL:
            try:
                return orig(self, scheduler_output)
            except AssertionError:
                _diag({"ev": "assert", "handoffs": {k: [list(e) for e in v] for k, v in (handoffs or {}).items()}})
                raise
        max_by_req = {}
        for req in handoffs:
            rs = self._req_status.get(req)
            if rs is not None:
                n = rs.req.num_prompt_tokens
                max_by_req[req] = min(n, rs.max_offload_tokens or n)
        handoffs, past = drop_past_window(handoffs, self._partial_tail_block_size, max_by_req)
        if past:
            seen = {}
            for req, b, mx in past:
                seen.setdefault((req, mx), set()).add(b)
            for (req, mx), bs_ in seen.items():
                _diag({"ev": "skip_past_window", "req": req, "boundaries": sorted(bs_), "max": mx})
        if not handoffs:
            return {}
        first, extra, skipped = split_partial_handoffs(handoffs, self._partial_tail_block_size, self._cow_source_groups)
        for req, b, missing in skipped:
            _diag({"ev": "skip_incomplete_tail", "req": req, "boundary": b, "missing_groups": missing})

        def with_handoffs(h):
            so = copy.copy(scheduler_output)
            so.kv_connector_block_state = dataclasses.replace(bs, boundary_state_offloads=h)
            return so
        jobs = orig(self, with_handoffs(first))
        for h in extra:
            _diag({"ev": "extra_tail", "reqs": {k: sorted({e[2] for e in v}) for k, v in h.items()}})
            jobs.update(orig(self, with_handoffs(h)))
        return jobs

    _build_partial_tail_store_jobs._horizon_kv_offload = True
    return _build_partial_tail_store_jobs


def _tail_lookup_hook(orig):
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import LookupResult

    def _lookup(self, req_status, max_num_new_tokens=None):
        r = orig(self, req_status, max_num_new_tokens)
        rec = {"ev": "lookup", "req": req_status.req.request_id, "prompt": req_status.req.num_prompt_tokens,
               "local": req_status.num_locally_computed_tokens, "orig": r, "orig_tail": req_status.partial_tail_boundary}
        if (r is None or not _TAIL_LOOKUP or not self.config.supports_partial_tail
                or req_status.partial_tail_boundary is not None):
            _diag(rec)
            return r
        local = req_status.num_locally_computed_tokens
        saved = (self._lookup_groups, self._sliding_window_groups)
        self._lookup_groups = tuple(g for g in saved[0] if g not in self._cow_source_groups)
        self._sliding_window_groups = tuple(g for g in saved[1] if g not in self._cow_source_groups)
        try:
            attn = self._lookup_complete_chunks(req_status, max_num_new_tokens) if self._lookup_groups else 0
        finally:
            self._lookup_groups, self._sliding_window_groups = saved
        rec["attn"] = attn
        if not attn:
            _diag(rec)
            return r
        req = req_status.req
        tph, blk = self.config.tokens_per_hash, self._partial_tail_block_size
        lo = local + r
        hi = min(req.num_prompt_tokens - 1, local + attn + blk - 1)
        if max_num_new_tokens is not None:
            hi = min(hi, local + max_num_new_tokens)
        hi -= hi % tph
        for boundary in range(hi, lo, -tph):
            if boundary % blk == 0:
                continue
            keys = [self._make_boundary_key(req, g.group_idx, boundary) for g in self.config.kv_group_configs]
            res = [self.manager.lookup(k, req_status.req_context) for k in keys]
            if any(x is LookupResult.MISS for x in res):
                continue
            if any(x in (LookupResult.HIT_PENDING, LookupResult.RETRY) for x in res):
                rec["pending"] = boundary
                continue
            for g, k in zip(self.config.kv_group_configs, keys):
                self._events_tracker.record_partial_lookup(req, g, boundary, k)
            req_status.partial_tail_boundary = boundary
            rec.update(tail=boundary, final=boundary - local)
            _diag(rec)
            return boundary - local
        _diag(rec)
        return r

    _lookup._horizon_kv_offload = True
    return _lookup


def _load_diag_hook(orig):
    """Diag only: what a tier hit loads, per group (attention chunks and GDN boundary states)."""
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        before = set(self._current_batch_load_jobs)
        tail = self._req_status[request.request_id].partial_tail_boundary if request.request_id in self._req_status else None
        orig(self, request, blocks, num_external_tokens)
        for jid in set(self._current_batch_load_jobs) - before:
            job = self._current_batch_load_jobs[jid]
            dst = job.dst_spec
            _diag({"ev": "load", "req": request.request_id, "external_tokens": num_external_tokens, "tail": tail,
                   "group_blocks": list(getattr(dst, "group_sizes", []) or []),
                   "group_types": ["gdn_state" if g.requires_cow_source else "attention"
                                   for g in self.config.kv_group_configs],
                   "keys": len(self._jobs[jid].keys) if jid in self._jobs else None})

    update_state_after_alloc._horizon_kv_offload = True
    return update_state_after_alloc


_MULTI_TAIL = False
_TAIL_LOOKUP = False


def install_tails() -> bool:
    """Items 9-11 (scheduler side)."""
    global _MULTI_TAIL, _TAIL_LOOKUP
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler as S
    diag = os.environ.get("HORIZON_KV_OFFLOAD_DIAG")
    if diag and _DIAG["f"] is None:
        _DIAG["f"] = open(f"{diag}.{os.getpid()}.jsonl", "a", buffering=1)
    _MULTI_TAIL = os.environ.get("HORIZON_KV_OFFLOAD_MULTI_TAIL", "0") == "1"
    _TAIL_LOOKUP = os.environ.get("HORIZON_KV_OFFLOAD_TAIL_LOOKUP", "0") == "1"
    if not (_MULTI_TAIL or _TAIL_LOOKUP or diag):
        return True
    if getattr(S._lookup, "_horizon_kv_offload", False):
        return True
    for name in ("_lookup", "_build_partial_tail_store_jobs", "_lookup_complete_chunks"):
        h = _src_hash(getattr(S, name))
        if h != _EXPECTED_SRC[f"OffloadingConnectorScheduler.{name}"]:
            _log().warning("%s: OffloadingConnectorScheduler.%s differs from v0.30.0 (hash %s): tail fixes NOT "
                           "installed", MARK, name, h)
            return False
    S._build_partial_tail_store_jobs = _multi_tail_hook(S._build_partial_tail_store_jobs)
    S._lookup = _tail_lookup_hook(S._lookup)
    if diag:
        S.update_state_after_alloc = _load_diag_hook(S.update_state_after_alloc)
    _log().warning("%s: partial tails: one store per handed-off boundary (sub-block tails past the save window dropped) %s, tail lookup past the attention-only "
                   "boundary %s, diag %s", MARK, "ON" if _MULTI_TAIL else "off", "ON" if _TAIL_LOOKUP else "off",
                   "on" if diag else "off")
    return True


def install() -> bool:
    from vllm.config.vllm import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    targets = {"VllmConfig._post_init_kv_transfer_config": VllmConfig._post_init_kv_transfer_config,
               "CPUOffloadingSpec._uses_shared_region": CPUOffloadingSpec._uses_shared_region,
               "OffloadingConnectorScheduler.on_new_request": OffloadingConnectorScheduler.on_new_request,
               "CPUOffloadingManager.prepare_store": CPUOffloadingManager.prepare_store,
               "Scheduler.schedule": Scheduler.schedule}
    if getattr(VllmConfig._post_init_kv_transfer_config, "_horizon_kv_offload", False):
        return True
    # Scheduler.schedule may already be wrapped by the measure-only chunk log (it keeps the original's behaviour);
    # hash the innermost function then.
    sched_fn = getattr(Scheduler.schedule, "__wrapped__", Scheduler.schedule)
    hashes = {k: _src_hash(sched_fn if k == "Scheduler.schedule" else f) for k, f in targets.items()}
    bad = {k: h for k, h in hashes.items() if h != _EXPECTED_SRC[k]
           and not (k == "Scheduler.schedule" and getattr(Scheduler.schedule, "_horizon_chunk_log", False))}
    if bad:
        _log().warning("%s: HORIZON_KV_OFFLOAD=1 but this vLLM's %s differ from v0.30.0 (hash %s): guard NOT installed",
                     MARK, ", ".join(bad), bad)
        return False
    VllmConfig._post_init_kv_transfer_config = _post_init_hook(VllmConfig._post_init_kv_transfer_config)
    CPUOffloadingSpec._uses_shared_region = _uses_shared_region_hook(CPUOffloadingSpec._uses_shared_region)
    min_prompt = int(env_float("HORIZON_KV_OFFLOAD_MIN_PROMPT", 0))
    if min_prompt > 0:
        OffloadingConnectorScheduler.on_new_request, CPUOffloadingManager.prepare_store = _short_filter_hooks(
            OffloadingConnectorScheduler.on_new_request, CPUOffloadingManager.prepare_store, min_prompt)
        _log().info("%s: requests under %d prompt tokens do not store into the CPU tier", MARK, min_prompt)
    reset = os.environ.get("HORIZON_KV_OFFLOAD_TEST_RESET")
    if reset:
        Scheduler.schedule = _test_reset_hook(Scheduler.schedule, reset)
        _log().warning("%s: TEST hook: GPU prefix-cache reset on %s.req (measure-only)", MARK, reset)
    if not os.environ.get("VLLM_KV_CACHE_LAYOUT"):
        os.environ["VLLM_KV_CACHE_LAYOUT"] = "LBNHC"   # inherited by the engine-core process
    if os.environ.get("HORIZON_KV_OFFLOAD_CONNECTOR", "native") == "simple":
        os.environ.setdefault("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
    if os.environ.get("HORIZON_KV_OFFLOAD_KEEP_ON_SLEEP", "0") == "1":
        install_keep_on_sleep()
    install_tails()
    return True


def _zero_tensors(obj, seen=None) -> int:
    """Zeroes every distinct underlying storage once (a fresh start allocates the KV cache with torch.zeros). vLLM's
    hybrid KV views (attention K/V and GDN state per layer) share storages, so zeroing each view would write the same
    memory several times. Returns the bytes."""
    import torch
    if seen is None:
        seen = set()
    n = 0
    if isinstance(obj, torch.Tensor):
        st = obj.untyped_storage()
        key = (st.data_ptr(), st.nbytes())
        if key in seen:
            return 0
        seen.add(key)
        torch.empty(0, dtype=torch.uint8, device=obj.device).set_(st, 0, (st.nbytes(),)).zero_()
        return st.nbytes()
    if isinstance(obj, (list, tuple)):
        for x in obj:
            n += _zero_tensors(x, seen)
    return n


def _zero_kv_wake_hook(orig):
    def wake_up(self, tags=None):
        orig(self, tags)
        if tags is None or "kv_cache" in tags:
            import time
            import torch
            t0 = time.monotonic()
            n = _zero_tensors(getattr(self.model_runner, "kv_caches", []))
            torch.cuda.synchronize()
            _log().info("%s: KV cache zeroed after wake (%.2f GiB in %.1f ms)", MARK, n / GIB,
                        (time.monotonic() - t0) * 1000)

    wake_up._horizon_kv_offload = True
    return wake_up


def install_zero_kv_on_wake() -> bool:
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker.wake_up, "_horizon_kv_offload", False):
        return True
    h = _src_hash(Worker.wake_up)
    if h != _EXPECTED_SRC["Worker.wake_up"]:
        _log().warning("%s: HORIZON_SLEEP_ZERO_KV=1 but Worker.wake_up differs from v0.30.0 (hash %s): not applied", MARK, h)
        return False
    Worker.wake_up = _zero_kv_wake_hook(Worker.wake_up)
    _log().warning("%s: KV cache is zeroed after every wake (HORIZON_SLEEP_ZERO_KV)", MARK)
    return True


def install_sleep_pageable() -> bool:
    import vllm.device_allocator.cumem as cm
    if getattr(cm, "_horizon_pageable", False):
        return True
    if not hasattr(cm, "PIN_MEMORY"):
        _log().warning("%s: HORIZON_SLEEP_PAGEABLE=1 but cumem has no PIN_MEMORY: not applied", MARK)
        return False
    cm.PIN_MEMORY = False
    cm._horizon_pageable = True
    _log().warning("%s: sleep-mode weight backups in pageable host memory (HORIZON_SLEEP_PAGEABLE)", MARK)
    return True


def register():
    """vLLM general-plugin entry point (runs in every vLLM process)."""
    global _done
    if os.environ.get("HORIZON_SLEEP_PAGEABLE", "0") == "1":
        try:
            install_sleep_pageable()
        except Exception as e:  # noqa: BLE001
            _log().warning("%s: pageable sleep backups not installed: %r", MARK, e)
    if os.environ.get("HORIZON_SLEEP_ZERO_KV", "0") == "1":
        try:
            install_zero_kv_on_wake()
        except Exception as e:  # noqa: BLE001
            _log().warning("%s: KV zeroing on wake not installed: %r", MARK, e)
    if _done or os.environ.get("HORIZON_KV_OFFLOAD", "0") != "1":
        return
    _done = True
    try:
        if install():
            _log().info("%s: guard installed (pid %d)", MARK, os.getpid())
    except Exception as e:  # noqa: BLE001  never break a start over the guard itself
        _log().warning("%s: guard not installed: %r", MARK, e)
