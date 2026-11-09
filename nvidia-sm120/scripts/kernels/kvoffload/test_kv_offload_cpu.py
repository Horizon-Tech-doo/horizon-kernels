#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for horizon_kv_offload (Horizon Tech, 30-09-2026). No GPU, no CUDA context.

  python test_kv_offload_cpu.py            pure tests everywhere; the vLLM tests run where vLLM 0.30 imports
                                           (server: $HK_VENV/bin/python), else they are skipped."""
import os
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pkg"))
import horizon_kv_offload as H  # noqa: E402

GIB = 1 << 30


def meminfo_file(avail_gib):
    f = tempfile.NamedTemporaryFile("w", suffix=".meminfo", delete=False)
    f.write(f"MemTotal:       50331648 kB\nMemFree:        1000 kB\nMemAvailable:   {int(avail_gib * 1024 * 1024)} kB\n")
    f.close()
    return f.name


class Pure(unittest.TestCase):
    def test_guard_fits(self):
        self.assertEqual(H.ram_guard(7 * GIB, 48 * GIB, 9 * GIB, 16 * GIB, 2 * GIB), (7 * GIB, "fits"))

    def test_guard_shrinks(self):
        use, why = H.ram_guard(16 * GIB, 36 * GIB, 9 * GIB, 16 * GIB, 2 * GIB)
        self.assertEqual(use, 11 * GIB)
        self.assertTrue(why.startswith("shrunk"))

    def test_guard_off(self):
        use, why = H.ram_guard(7 * GIB, 26 * GIB, 9 * GIB, 16 * GIB, 2 * GIB)
        self.assertEqual(use, 0)
        self.assertTrue(why.startswith("off"))

    def test_region(self):
        self.assertEqual(H.choose_region("auto", 7 * GIB, 8 * GIB), "shm")
        self.assertEqual(H.choose_region("auto", 8 * GIB, 8 * GIB), "private")   # margin
        self.assertEqual(H.choose_region("auto", 7 * GIB, None), "private")
        self.assertEqual(H.choose_region("shm", 12 * GIB, 8 * GIB), "shm")
        self.assertEqual(H.choose_region("private", 1 * GIB, 8 * GIB), "private")

    def test_meminfo_and_decide(self):
        p = meminfo_file(40)
        try:
            mi = H.read_meminfo(p)
            self.assertEqual(mi["MemAvailable"], 40 * GIB)
            old = dict(os.environ)
            os.environ.update(HORIZON_KV_OFFLOAD_POLICY="arc", HORIZON_KV_OFFLOAD_STORE_THRESHOLD="2")
            try:
                d = H.decide(7 * GIB, mi, 8 * GIB)
            finally:
                os.environ.clear()
                os.environ.update(old)
            self.assertEqual(d["bytes"], 7 * GIB)
            self.assertEqual(d["region"], "shm")
            self.assertEqual(d["extra"], {"eviction_policy": "arc", "store_threshold": 2})
            d = H.decide(20 * GIB, mi, 8 * GIB)   # 40 - 9 - 16 = 15 GiB room -> shrunk, too big for shm
            self.assertEqual((d["bytes"], d["region"]), (15 * GIB, "private"))
            self.assertEqual(H.decide(7 * GIB, {"MemAvailable": 20 * GIB}, 8 * GIB)["bytes"], 0)
        finally:
            os.unlink(p)

    def test_short_prompt_filter(self):
        calls = []

        class Sched:
            def on_new_request(self, request):
                calls.append(("new", request.request_id))

        class Mgr:
            def prepare_store(self, keys, req_context):
                calls.append(("store", req_context.req_id, list(keys)))
                return "out"

        on_new, prep = H._short_filter_hooks(Sched.on_new_request, Mgr.prepare_store, 4096)
        R = type("R", (), {})
        short, long_ = R(), R()
        short.request_id, short.num_prompt_tokens = "s", 900
        long_.request_id, long_.num_prompt_tokens = "l", 12000
        on_new(Sched(), short)
        on_new(Sched(), long_)
        ctx = type("C", (), {})
        cs, cl = ctx(), ctx()
        cs.req_id, cl.req_id = "s", "l"
        self.assertEqual(prep(Mgr(), ["k1", "k2"], cs), "out")
        self.assertEqual(prep(Mgr(), ["k3"], cl), "out")
        self.assertEqual(calls, [("new", "s"), ("new", "l"), ("store", "s", []), ("store", "l", ["k3"])])

    def test_reset_hook(self):
        d = tempfile.mkdtemp()
        prefix = os.path.join(d, "reset")
        seen = []

        class KVM:
            def reset_prefix_cache(self):
                seen.append("reset")
                return True

        class Sched:
            kv_cache_manager = KVM()
            waiting, running = [1], []

            def schedule(self):
                seen.append("schedule")
                return "so"

        hooked = H._test_reset_hook(Sched.schedule, prefix)
        self.assertEqual(hooked(Sched()), "so")
        self.assertEqual(seen, ["schedule"])                  # no request file: untouched
        open(prefix + ".req", "w").close()
        self.assertEqual(hooked(Sched()), "so")
        self.assertEqual(seen, ["schedule", "reset", "schedule"])
        self.assertFalse(os.path.exists(prefix + ".req"))
        with open(prefix + ".done") as f:
            self.assertTrue(f.read().startswith("1 waiting=1 running=0"))

    def test_keep_on_sleep_hook(self):
        d = tempfile.mkdtemp()
        nokeep = os.path.join(d, "reset.nokeep")
        calls = []

        class Core:
            def _reset_caches(self, reset_running_requests=True, reset_connector=True):
                calls.append((reset_running_requests, reset_connector))

        hooked = H._keep_on_sleep_hook(Core._reset_caches, nokeep)
        hooked(Core())                                        # sleep path: default arguments
        self.assertEqual(calls[-1], (True, False))            # GPU reset, connector kept
        hooked(Core(), reset_connector=False)
        self.assertEqual(calls[-1], (True, False))
        open(nokeep, "w").close()                             # runtime override: stock behaviour
        hooked(Core())
        self.assertEqual(calls[-1], (True, True))

    def test_split_partial_handoffs(self):
        # block 832, GDN groups 0-2 (cow source), attention group 3
        h = {"a": [(0, 11, 1664), (1, 12, 1664), (2, 13, 1664),                  # aligned: kept as is
                   (0, 21, 4224), (1, 22, 4224), (2, 23, 4224),                  # checkpoint tail
                   (0, 31, 4032), (1, 32, 4032), (2, 33, 4032)],                 # older CoW tail, same step
             "b": [(0, 41, 900), (1, 42, 900)],                                  # incomplete: group 2 missing
             "c": [(0, 51, 832)]}
        first, extra, skipped = H.split_partial_handoffs(h, 832, {0, 1, 2})
        self.assertEqual(sorted(e[2] for e in first["a"]), [1664] * 3 + [4032] * 3)
        self.assertEqual(extra, [{"a": [(0, 21, 4224), (1, 22, 4224), (2, 23, 4224)]}])
        self.assertEqual(first["b"], [])
        self.assertEqual(skipped, [("b", 900, [2])])
        self.assertEqual(first["c"], [(0, 51, 832)])
        f2, e2, s2 = H.split_partial_handoffs({"x": [(0, 1, 100), (1, 2, 100), (2, 3, 100)]}, 832, {0, 1, 2})
        self.assertEqual((len(f2["x"]), e2, s2), (3, [], []))

    def test_drop_past_window(self):
        # the 06-10 19:54 crash: a preempted request (prompt 30,000) re-prefills prompt + output; its in-step checkpoint
        # at 30,464 lies past the save window -> dropped; aligned 29,952 kept (vLLM filters aligned ones itself)
        h = {"p": [(0, 1, 29952), (1, 2, 29952), (2, 3, 29952), (0, 4, 30464), (1, 5, 30464), (2, 6, 30464)],
             "q": [(0, 7, 4224), (1, 8, 4224), (2, 9, 4224)],
             "u": [(0, 10, 900)]}                                                 # unknown to the connector
        kept, dropped = H.drop_past_window(h, 832, {"p": 30000, "q": 4313})
        self.assertEqual(kept["p"], [(0, 1, 29952), (1, 2, 29952), (2, 3, 29952)])
        self.assertEqual(kept["q"], h["q"])
        self.assertNotIn("u", kept)
        self.assertEqual(sorted(dropped), [("p", 30464, 30000)] * 3 + [("u", 900, None)])

    def test_multi_tail_hook_drops_past_window(self):
        import dataclasses

        @dataclasses.dataclass
        class BS:
            boundary_state_offloads: dict

        class SO:
            def __init__(self, h): self.kv_connector_block_state = BS(h)

        calls = []

        def orig(self, so):                    # vLLM's partial path: asserts boundary <= min(prompt, max_offload)
            h = so.kv_connector_block_state.boundary_state_offloads
            for req, entries in h.items():
                for _, _, b in entries:
                    if b % 832:
                        assert b <= self._req_status[req].req.num_prompt_tokens
            calls.append({k: sorted({e[2] for e in v}) for k, v in h.items()})
            return {len(calls): True}

        class Req:
            num_prompt_tokens, num_computed_tokens = 30000, 0

        class RS:
            req, max_offload_tokens = Req(), None

        class Cfg:
            supports_partial_tail = True

        class Fake:
            config, _req_status = Cfg(), {"p": RS()}
            _partial_tail_block_size, _cow_source_groups = 832, frozenset({0, 1, 2})
        H._MULTI_TAIL = True
        try:
            hook = H._multi_tail_hook(orig)
            h = {"p": [(0, 1, 29952), (1, 2, 29952), (2, 3, 29952), (0, 4, 30464), (1, 5, 30464), (2, 6, 30464)]}
            with self.assertRaises(AssertionError):
                orig(Fake(), SO(h))
            calls.clear()
            self.assertEqual(hook(Fake(), SO(h)), {1: True})
            self.assertEqual(calls, [{"p": [29952]}])
            calls.clear()
            self.assertEqual(hook(Fake(), SO({"p": [(0, 4, 30464), (1, 5, 30464), (2, 6, 30464)]})), {})
            self.assertEqual(calls, [])
        finally:
            H._MULTI_TAIL = False

    def test_zero_kv_on_wake_hook(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch missing")
        calls = []

        class R:
            kv_caches = [torch.ones(4), [torch.ones(2, 2), torch.ones(3)]]

        class Wk:
            model_runner = R()
            def wake_up(self, tags=None): calls.append(tags)
        torch.cuda.synchronize = lambda: None
        w = Wk()
        hook = H._zero_kv_wake_hook(Wk.wake_up)
        hook(w, ["weights"])                         # weights only: KV untouched
        self.assertEqual(float(R.kv_caches[0].sum()), 4.0)
        hook(w, None)
        self.assertEqual(float(R.kv_caches[0].sum()) + float(R.kv_caches[1][0].sum()) + float(R.kv_caches[1][1].sum()), 0.0)
        self.assertEqual(calls, [["weights"], None])
        # shared storage: two views of one buffer (+ an offset view) are zeroed once, the whole storage is cleared
        base = torch.ones(16)
        views = [base[:8].view(2, 4), base[8:], base[4:12]]
        self.assertEqual(H._zero_tensors([views, base]), 16 * 4)
        self.assertEqual(float(base.sum()), 0.0)

    def test_register_is_inert_by_default(self):
        os.environ.pop("HORIZON_KV_OFFLOAD", None)
        H._done = False
        H.register()     # must not import vLLM or patch anything
        self.assertFalse(H._done)


def _have_vllm():
    try:
        import vllm  # noqa: F401
        return vllm.__version__.startswith("0.30")
    except Exception:  # noqa: BLE001
        return False


MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4-hmtp"


@unittest.skipUnless(_have_vllm() and os.path.isdir(MODEL), "needs vLLM 0.30 and the model config (server)")
class WithVLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("VLLM_KV_CACHE_LAYOUT", None)
        os.environ.update(HORIZON_KV_OFFLOAD_MIN_PROMPT="4096", HORIZON_KV_OFFLOAD_TEST_RESET="/tmp/hz-kvoff-test")
        assert H.install(), "hash guard refused vLLM 0.30"

    def test_all_hooks_installed(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
        for f in (OffloadingConnectorScheduler.on_new_request, CPUOffloadingManager.prepare_store, Scheduler.schedule):
            self.assertTrue(getattr(f, "_horizon_kv_offload", False), f)

    def cfg(self, gib, avail_gib, **env):
        from vllm.engine.arg_utils import EngineArgs
        p = meminfo_file(avail_gib)
        old = dict(os.environ)
        os.environ.update(HORIZON_KV_OFFLOAD_MEMINFO=p, **env)
        try:
            return EngineArgs(model=MODEL, max_model_len=262144, kv_cache_dtype="fp8", max_num_seqs=64,
                              kv_cache_memory_bytes=46 * GIB, long_prefill_token_threshold=1600,
                              speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 3,
                                                  "use_local_argmax_reduction": True},
                              kv_offloading_size=gib, kv_offloading_backend="native").create_engine_config()
        finally:
            os.environ.clear()
            os.environ.update(old)
            os.unlink(p)

    def test_tail_hooks_installed_on_real_scheduler(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler as S
        os.environ.update(HORIZON_KV_OFFLOAD_MULTI_TAIL="1", HORIZON_KV_OFFLOAD_TAIL_LOOKUP="1")
        try:
            self.assertTrue(H.install_tails())
        finally:
            os.environ.pop("HORIZON_KV_OFFLOAD_MULTI_TAIL"); os.environ.pop("HORIZON_KV_OFFLOAD_TAIL_LOOKUP")
        self.assertTrue(getattr(S._lookup, "_horizon_kv_offload", False))
        self.assertTrue(getattr(S._build_partial_tail_store_jobs, "_horizon_kv_offload", False))

    def test_tail_lookup_finds_checkpoint_past_attention_boundary(self):
        from vllm.v1.kv_offload.base import LookupResult
        orig_calls = []

        def orig(self, rs, mx=None):
            orig_calls.append(mx)
            rs.partial_tail_boundary = None
            return 0                                   # no aligned GDN state stored: stock lookup finds nothing

        class G:
            def __init__(self, i): self.group_idx = i
        stored = {("g", gi, 4224) for gi in range(4)}

        class M:
            def lookup(self, key, ctx): return LookupResult.HIT if key in stored else LookupResult.MISS

        class Ev:
            def record_partial_lookup(self, *a): pass

        class Req:
            request_id, num_prompt_tokens = "r", 4313

        class RS:
            req, num_locally_computed_tokens, partial_tail_boundary, req_context = Req(), 0, None, None

        class Cfg:
            supports_partial_tail, tokens_per_hash = True, 64
            kv_group_configs = [G(0), G(1), G(2), G(3)]

        class Fake:
            config, manager, _events_tracker = Cfg(), M(), Ev()
            _partial_tail_block_size, _cow_source_groups = 832, frozenset({0, 1, 2})
            _lookup_groups, _sliding_window_groups = (3,), (0, 1, 2)
            def _lookup_complete_chunks(self, rs, mx=None):
                assert self._lookup_groups == (3,) and self._sliding_window_groups == ()
                return 4160                            # attention chunks stored up to 5 blocks
            def _make_boundary_key(self, req, gi, b): return ("g", gi, b)
        H._TAIL_LOOKUP = True
        try:
            hook = H._tail_lookup_hook(orig)
            f, rs = Fake(), RS()
            self.assertEqual(hook(f, rs), 4224)
            self.assertEqual(rs.partial_tail_boundary, 4224)
            self.assertEqual((f._lookup_groups, f._sliding_window_groups), ((3,), (0, 1, 2)))   # restored
            stored.discard(("g", 1, 4224))             # one GDN group missing -> no tail, stock answer
            rs = RS()
            self.assertEqual(hook(f, rs), 0)
            self.assertIsNone(rs.partial_tail_boundary)
        finally:
            H._TAIL_LOOKUP = False

    def test_zero_kv_installed_on_real_worker(self):
        from vllm.v1.worker.gpu_worker import Worker
        self.assertTrue(H.install_zero_kv_on_wake())
        self.assertTrue(getattr(Worker.wake_up, "_horizon_kv_offload", False))

    def test_keep_on_sleep_installed(self):
        from vllm.v1.engine.core import EngineCore
        self.assertTrue(H.install_keep_on_sleep())
        self.assertTrue(getattr(EngineCore._reset_caches, "_horizon_kv_offload", False))
        self.assertTrue(H.install_keep_on_sleep())            # idempotent

    def test_installed_and_layout_pinned(self):
        from vllm.config.vllm import VllmConfig
        self.assertTrue(getattr(VllmConfig._post_init_kv_transfer_config, "_horizon_kv_offload", False))
        self.assertEqual(os.environ.get("VLLM_KV_CACHE_LAYOUT"), "LBNHC")

    def test_fits_shm(self):
        c = self.cfg(7, 48, HORIZON_KV_OFFLOAD_POLICY="arc")
        ex = c.kv_transfer_config.kv_connector_extra_config
        self.assertEqual(c.kv_transfer_config.kv_connector, "OffloadingConnector")
        self.assertEqual(ex["cpu_bytes_to_use"], 7 * GIB)
        self.assertEqual(ex["eviction_policy"], "arc")
        self.assertIn(ex["horizon_region"], ("shm", "private"))
        self.assertFalse(c.scheduler_config.disable_hybrid_kv_cache_manager)

    def test_guard_switches_off(self):
        c = self.cfg(7, 20)
        self.assertIsNone(c.kv_transfer_config)
        self.assertIsNone(c.cache_config.kv_offloading_size)
        self.assertFalse(c.scheduler_config.disable_hybrid_kv_cache_manager)

    def test_big_tier_goes_private(self):
        c = self.cfg(12, 48)
        ex = c.kv_transfer_config.kv_connector_extra_config
        self.assertEqual(ex["cpu_bytes_to_use"], 12 * GIB)
        self.assertEqual(ex["horizon_region"], "private")
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
        fake = type("S", (), {"extra_config": ex})()
        self.assertFalse(CPUOffloadingSpec._uses_shared_region(fake))
        fake.extra_config = dict(ex, horizon_region="shm")
        self.assertTrue(CPUOffloadingSpec._uses_shared_region(fake))

    def test_simple_connector_lazy(self):
        c = self.cfg(7, 48, VLLM_USE_SIMPLE_KV_OFFLOAD="1", HORIZON_KV_OFFLOAD_LAZY="1")
        ex = c.kv_transfer_config.kv_connector_extra_config
        self.assertEqual(c.kv_transfer_config.kv_connector, "SimpleCPUOffloadConnector")
        self.assertIs(ex["lazy_offload"], True)
        self.assertEqual(ex["horizon_region"], "private (simple connector)")
        self.assertFalse(c.scheduler_config.disable_hybrid_kv_cache_manager)

    def test_shrunk_tier(self):
        c = self.cfg(16, 36)   # 36 - 9 - 16 = 11 GiB
        self.assertEqual(c.kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"], 11 * GIB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
