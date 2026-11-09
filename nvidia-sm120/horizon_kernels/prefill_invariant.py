# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels P1 "chunk-invariant prefill" (30-09-2026, kernel plan §5.2 step 9). Default OFF.
#
# Goal: a prompt token's hidden state must not depend on how the scheduler cut the prompt into prefill chunks (chunk
# start, chunk length, what else is in the step), so a prefix-cache hit continues exactly as an uncached prefill would.
# Evidence for the three culprits: an op-level probe (GDN state restore and long-
# chunk attention were already exact).
#
# HORIZON_PI=1 turns on all three parts; each can be switched off on its own (HORIZON_PI_<PART>=0):
#   STEP   (the classifier, 30-09-2026 v2) a step "has prefill" when vLLM's V2 runner says so: BatchReqState.has_prefill
#          (some request's num_computed_prefill_tokens < prefill_len; spec-decode verify steps are decode). The flag is
#          recorded per step from GPUModelRunner.gather_batch_req_state, and every step with prefill is dispatched EAGER
#          (need_eager=True in dispatch_cg_and_sync_dp), so CUDA graphs (FULL or PIECEWISE) only ever replay decode-only
#          steps and are captured with the production kernel table. v1 keyed on the graph mode (not FULL = prefill) and
#          collapsed decode speed (x0.46 at 1 user): on this stack the MTP verify steps
#          replay PIECEWISE graphs that prefill steps share, so they ran fp4x2 and the unsplit attention.
#          Cost: prefill steps of <= 512 tokens (the largest capture size) lose their PIECEWISE graph.
#   GRAPHS (v3, HORIZON_PI_GRAPHS=1, default 1): instead of running small prefill steps eager, give them their own
#          PIECEWISE graph set. Steps with prefill keep vLLM's normal dispatch; their forward context gets a batch
#          descriptor with a PI key (num_active_loras + 1<<20; no LoRA is configured), so vLLM's CUDAGraphWrapper
#          captures a separate graph per padded size on first use, with the PI kernels baked in, while decode-only
#          steps keep the startup graphs on the production table. HORIZON_PI_GRAPHS=0 is v2 (prefill steps eager).
#          v3.2: the PI twin of every PIECEWISE size is captured at STARTUP, inside vLLM's own capture pass (its capture
#          stream, pool and warm-up): ModelCudaGraphManager.capture's forward_fn, after the normal capture, sets the
#          prefill flag, warms the PI kernels eagerly and captures again under the PI-keyed descriptor. Nothing is
#          captured while serving; a PI-keyed piece without a twin runs eagerly. (v3 captured on first use: "CUDA graphs
#          must be captured on a non-default stream"; v3.1 captured on a side stream and replayed and produced garbage.)
#   NVFP4  every NVFP4 linear in a step with prefill runs fp4x2 (the FP4 prefill variant: one kernel config for every M,
#          per-row activation scale; probe: bit-identical rows for every M on all 5 shapes), whatever the row count.
#          Today the calibration table picks regfrag (W4A16) for small row counts and fp4x2 for large ones, so the same
#          row changed numerics with the chunk size. Decode rows that share a step with prefill also run fp4x2 (as they
#          already do today whenever the step has more than ~512 rows).
#   BA     GDN in_proj_a/b (BF16, N 96): K5's skinny kernel at every row count in steps with prefill (it is
#          batch-invariant by construction: every output element is summed in the same fixed order for any M and any
#          tile config); today cuBLAS takes over above 48 rows and changes its algorithm with M. Needs HORIZON_K5=1.
#   ATTN   in steps with prefill, FlashInfer FA2 paged prefill is planned with disable_split_kv=True: the planner otherwise
#          splits the KV of a short query chunk over long context across CTAs and merges, which rounds differently from
#          the same rows in a long chunk (which is never split). Decode-only steps keep the split (probe: without it a
#          16-row query over 16k tokens is x10 slower).
#
# HORIZON_PI_PROBE=<file.json> (measure-only, independent of HORIZON_PI): runs the op-level invariance/speed probe on
# the real prepared layers at the first eager NVFP4 call (vLLM's profile run, before the KV cache takes the memory) and
# writes <file.json>. Used when the engine already holds the GPU memory and a standalone launch has no room.
import dataclasses
import json
import os
import time

import torch

_ATTN_DONE = False
_RUNNER_DONE = False
_PROBE_DONE = False
_STEP = {"prefill": False}   # set per step by the runner hook (host side, before dispatch and forward)
_ORIG_PLAN = None           # FlashInfer's unpatched BatchPrefillWithPagedKVCacheWrapper.plan (the probe times both)


def _on(part: str) -> bool:
    return os.environ.get("HORIZON_PI", "0") == "1" and os.environ.get(f"HORIZON_PI_{part}", "1") == "1"


def nvfp4_on() -> bool:
    return _on("NVFP4")


def ba_on() -> bool:
    return _on("BA")


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


PI_GRAPH_KEY = 1 << 20      # added to BatchDescriptor.num_active_loras for PIECEWISE steps with prefill


def graphs_on() -> bool:
    return _on("GRAPHS")


def prefill_step() -> bool:
    """True in a forward of a step that contains prefill tokens: eager, or (GRAPHS) the runtime capture of that step's
    own PIECEWISE graph. vLLM's startup capture runs dummy batches (flag False), so decode graphs keep the table."""
    if graphs_on():
        return _STEP["prefill"]
    return _STEP["prefill"] and not torch.cuda.is_current_stream_capturing()


def install_runner():
    """STEP: record BatchReqState.has_prefill per step and dispatch steps with prefill eagerly (vLLM 0.30 V2 runner)."""
    global _RUNNER_DONE
    if _RUNNER_DONE:
        return
    import vllm.v1.worker.gpu.model_runner as mr
    if getattr(mr.GPUModelRunner.gather_batch_req_state, "_horizon_pi", False):
        _RUNNER_DONE = True
        return
    orig_gather = mr.GPUModelRunner.gather_batch_req_state
    orig_dispatch = mr.dispatch_cg_and_sync_dp
    orig_sfc = mr.set_forward_context
    graphs = graphs_on()

    def gather_batch_req_state(self, scheduler_output, dummy_run):
        out = orig_gather(self, scheduler_output, dummy_run)
        bs = out[0]
        _STEP["prefill"] = bool(bs is not None and bs.has_prefill)
        return out

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        if _STEP["prefill"]:
            kwargs["need_eager"] = True
        return orig_dispatch(*args, **kwargs)

    def set_forward_context(*args, **kwargs):
        bd = kwargs.get("batch_descriptor")
        if _STEP["prefill"] and bd is not None and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.PIECEWISE:
            kwargs["batch_descriptor"] = dataclasses.replace(bd, num_active_loras=bd.num_active_loras + PI_GRAPH_KEY)
        return orig_sfc(*args, **kwargs)

    from vllm.config.compilation import CUDAGraphMode
    from vllm.compilation import cuda_graph as cg
    from vllm.v1.worker.gpu import cudagraph_utils as cgu
    orig_call = cg.CUDAGraphWrapper.__call__
    orig_cgu_sfc = cgu.set_forward_context
    orig_capture = cgu.CudaGraphManager.capture       # the base method takes create_forward_fn; the model manager's
                                                       # capture(model, ...) builds that function and calls this one
    warned = {}

    def wrapper_call(self, *args, **kwargs):
        if _STEP["prefill"] and not _STEP.get("startup_capture"):
            from vllm.forward_context import get_forward_context, is_forward_context_available
            fc = get_forward_context() if is_forward_context_available() else None
            bd = fc.batch_descriptor if fc is not None else None
            if (bd is not None and bd.num_active_loras >= PI_GRAPH_KEY
                    and fc.cudagraph_runtime_mode == self.runtime_mode):
                e = self.concrete_cudagraph_entries.get(bd)
                if e is None or e.cudagraph is None:              # never capture while serving: run the piece eager
                    if not warned:
                        warned[1] = True
                        _log().warning("horizon_kernels PI: no startup PI graph for %s, running that piece eager", bd)
                    return self.runnable(*args, **kwargs)
        return orig_call(self, *args, **kwargs)

    def cgu_set_forward_context(*args, **kwargs):
        bd = kwargs.get("batch_descriptor")
        if _STEP["prefill"] and bd is not None and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.PIECEWISE:
            kwargs["batch_descriptor"] = dataclasses.replace(bd, num_active_loras=bd.num_active_loras + PI_GRAPH_KEY)
        return orig_cgu_sfc(*args, **kwargs)

    def capture(self, create_forward_fn, *cargs, **ckw):
        if not isinstance(self, cgu.ModelCudaGraphManager):   # the MTP drafter's managers: unchanged
            return orig_capture(self, create_forward_fn, *cargs, **ckw)

        def cff(desc, *a, **kw):
            fwd = create_forward_fn(desc, *a, **kw)
            if desc.cg_mode != CUDAGraphMode.PIECEWISE:
                return fwd

            def forward_fn(cg_mode):
                out = fwd(cg_mode)
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    _STEP["prefill"] = True
                    _STEP["startup_capture"] = True     # lets the wrapper capture (it never does while serving)
                    try:
                        fwd(CUDAGraphMode.NONE)          # warm the PI kernels at this size (eager)
                        fwd(CUDAGraphMode.PIECEWISE)     # capture the PI twin under the PI-keyed descriptor
                    finally:
                        _STEP["prefill"] = False
                        _STEP["startup_capture"] = False
                return out
            return forward_fn
        return orig_capture(self, cff, *cargs, **ckw)

    gather_batch_req_state._horizon_pi = True
    mr.GPUModelRunner.gather_batch_req_state = gather_batch_req_state
    if graphs:
        set_forward_context._horizon_pi = True
        mr.set_forward_context = set_forward_context
        wrapper_call._horizon_pi = True
        cg.CUDAGraphWrapper.__call__ = wrapper_call
        cgu_set_forward_context._horizon_pi = True
        cgu.set_forward_context = cgu_set_forward_context
        capture._horizon_pi = True
        cgu.CudaGraphManager.capture = capture
        _log().info("horizon_kernels PI: steps with prefill replay their own PIECEWISE graphs (PI kernels, captured at "
                    "startup); decode-only steps keep the startup graphs")
    else:
        dispatch_cg_and_sync_dp._horizon_pi = True
        mr.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
        _log().info("horizon_kernels PI: steps with prefill run eager; CUDA graphs replay decode-only steps")
    _RUNNER_DONE = True


def install_attn():
    """ATTN: FA2 paged prefill plans without the KV split (patched on the flashinfer class, used by vLLM's backend)."""
    global _ATTN_DONE, _ORIG_PLAN
    if _ATTN_DONE or not _on("ATTN"):
        return
    import flashinfer
    cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
    orig = cls.plan
    if getattr(orig, "_horizon_pi", False):
        _ATTN_DONE = True
        return

    def plan(self, *args, **kwargs):
        if _STEP["prefill"]:
            kwargs["disable_split_kv"] = True
            kwargs.pop("fixed_split_size", None)
        return orig(self, *args, **kwargs)

    plan._horizon_pi = True
    _ORIG_PLAN = orig
    cls.plan = plan
    _ATTN_DONE = True
    _log().info("horizon_kernels PI: FlashInfer paged prefill planned with disable_split_kv in steps with prefill")


def install():
    """Called from register() when HORIZON_PI=1 (the NVFP4 and BA parts are hooks inside vllm_plugin / k5)."""
    install_runner()
    install_attn()
    _log().info("horizon_kernels PI: chunk-invariant prefill on (nvfp4=%s, ba=%s, attn=%s, graphs=%s)",
                nvfp4_on(), ba_on(), _on("ATTN"), graphs_on())


# ---------------------------------------------------------------------------------------------------------------
# HORIZON_PI_PROBE: in-process op probe
def _tms(fn, reps=10, rounds=5):
    import statistics
    best = None
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000)
        m = statistics.median(ts)
        best = m if best is None else min(best, m)
    return round(best, 2)


def _rows(y, ref):
    d = (y.float() - ref[:y.shape[0]].float()).abs()
    return bool(torch.equal(y, ref[:y.shape[0]])), float(d.max())


def maybe_probe(layers, op):
    """layers: vllm_plugin._LAYERS (weakrefs); op: the AdaptiveOp. Runs once, only with HORIZON_PI_PROBE set, only
    outside CUDA-graph capture."""
    global _PROBE_DONE
    path = os.environ.get("HORIZON_PI_PROBE", "")
    if _PROBE_DONE or not path or torch.cuda.is_current_stream_capturing():
        return
    _PROBE_DONE = True
    t0 = time.time()
    R = {"time": time.strftime("%d-%m-%Y %H:%M"), "device": torch.cuda.get_device_name(0), "N": {}, "S": {}, "A": {}}
    dev = torch.device("cuda")
    try:
        _probe_nvfp4(layers, op, R, dev)
    except Exception as e:  # noqa: BLE001
        R["N"]["error"] = repr(e)[:500]
    try:
        _probe_skinny(R, dev)
    except Exception as e:  # noqa: BLE001
        R["S"]["error"] = repr(e)[:500]
    try:
        _probe_attn(R, dev)
    except Exception as e:  # noqa: BLE001
        R["A"]["error"] = repr(e)[:500]
    R["secs"] = round(time.time() - t0, 1)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=1)
    os.replace(tmp, path)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    _log().warning("horizon_kernels PI probe written to %s in %.0f s", path, R["secs"])


MS_N = [1, 2, 3, 4, 5, 8, 16, 17, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 256, 511, 512, 513, 1000, 1024, 1025,
        1600, 2048, 3200, 4096, 8192]


def _probe_nvfp4(layers, op, R, dev):
    seen = {}
    for ref in layers:
        L = ref()
        if L is None or L.horizon_nvfp4 is None or getattr(L, "horizon_exact_only", False):
            continue
        key = (L.output_size_per_partition, L.input_size_per_partition)
        seen.setdefault(key, L)
    for (n, k), L in sorted(seen.items()):
        g = torch.Generator(device=dev).manual_seed(n + k)
        mmax = 4096 if n > 20000 else 8192
        x = torch.randn(mmax, k, device=dev, generator=g).to(torch.bfloat16)
        res = {}
        for name in ("fp4x2", "regfrag"):
            var = op.variants.get(name)
            if var is None or (name == "regfrag" and L.horizon_regfrag is None):
                continue
            ref_y = var.fn(x, L)
            bad, mx = [], 0.0
            for m in [m for m in MS_N if m <= mmax]:
                ok, d = _rows(var.fn(x[:m].contiguous(), L), ref_y)
                mx = max(mx, d)
                if not ok:
                    bad.append(m)
            res[name] = {"row_invariant": not bad, "differs_at_M": bad, "max_abs": mx}
            del ref_y
        a = op.variants["fp4x2"].fn(x[:1024].contiguous(), L).float()
        b = op.variants["regfrag"].fn(x[:1024].contiguous(), L).float()
        res["fp4x2_vs_regfrag_rel_err"] = float((a - b).norm() / b.norm())
        res["table_choice"] = {m: op.select((n, k), m).name for m in (1, 4, 16, 64, 128, 256, 512, 1024, 2048, 8192)}
        res["us"] = {m: {v: _tms(lambda v=v, m=m: op.variants[v].fn(x[:m], L)) for v in ("fp4x2", "regfrag")}
                     for m in (1, 4, 16, 32, 64, 128, 256, 512, 1024)}
        R["N"][f"{n}x{k}"] = res
        del x, a, b
        torch.cuda.empty_cache()


def _probe_skinny(R, dev):
    from horizon_kernels import k5
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    if not k5._SK_LAYERS:
        R["S"]["error"] = "no K5 skinny layer registered (HORIZON_K5=1 needed)"
        return
    w = k5._SK_LAYERS[0].weight
    n, k = w.shape
    g = torch.Generator(device=dev).manual_seed(3)
    x = torch.randn(8192, k, device=dev, generator=g).to(torch.bfloat16)
    ref = sk.linear(x, w, cfg=sk.config(n, k, 8192))
    ms = sorted(set(list(range(1, 65)) + [96, 127, 128, 129, 256, 512, 533, 960, 1024, 1025, 1600, 1920, 1984, 2048,
                                          2304, 2368, 3200, 4096, 4800, 6400, 8000, 8192]))
    bad = [m for m in ms if not torch.equal(sk.linear(x[:m].contiguous(), w), ref[:m])]
    cfgs = {}
    for cfg in [(8, 4, 128), (8, 8, 128), (16, 4, 128), (16, 2, 128), (4, 8, 128)]:
        try:
            y = sk.linear(x, w, cfg=cfg)
            cfgs[str(cfg)] = {"bit_identical_to_dispatch_config": bool(torch.equal(y, ref)),
                              "us": {m: _tms(lambda cfg=cfg, m=m: sk.linear(x[:m], w, cfg=cfg))
                                     for m in (48, 128, 256, 512, 1600, 4096, 8192)}}
        except Exception as e:  # noqa: BLE001
            cfgs[str(cfg)] = {"error": repr(e)[:300]}
    cub = torch.nn.functional.linear(x, w)
    R["S"] = {"shape": [n, k], "row_invariant_dispatch_configs": not bad, "differs_at_M": bad,
              "vs_cublas_max_abs": float((ref.float() - cub.float()).abs().max()),
              "vs_cublas_rel_err": float((ref.float() - cub.float()).norm() / cub.float().norm()),
              "cublas_us": {m: _tms(lambda m=m: torch.nn.functional.linear(x[:m], w))
                            for m in (48, 128, 256, 512, 1600, 4096, 8192)},
              "configs": cfgs}
    del x, ref, cub
    torch.cuda.empty_cache()


def _probe_attn(R, dev):
    import flashinfer
    QH, KVH, D, PAGE = 24, 4, 256, 16
    LMAX = 65536
    g = torch.Generator(device=dev).manual_seed(5)
    kv = (torch.randn(LMAX // PAGE, 2, KVH, PAGE, D, device=dev, generator=g) * 1.5).to(torch.float8_e4m3fn)
    q_all = torch.randn(2048, QH, D, device=dev, generator=g).to(torch.bfloat16)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    plan_fn = _ORIG_PLAN or flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan

    def plan_run(a, b, dsk):
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        npg = (b + PAGE - 1) // PAGE
        i32 = dict(dtype=torch.int32)
        args = (torch.tensor([0, b - a], **i32), torch.tensor([0, npg], **i32), torch.arange(npg, device=dev, **i32),
                torch.tensor([b - PAGE * (npg - 1)], **i32), QH, KVH, D, PAGE)
        kw = dict(causal=True, sm_scale=D ** -0.5, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
                  o_data_type=torch.bfloat16, disable_split_kv=dsk)
        plan_fn(wr, *args, **kw)
        return wr

    out = {"planner": "FlashInfer's unpatched plan with disable_split_kv False / True"}
    for L in (4096, 16384, 65536):
        for qn in (16, 64, 256, 512, 1024, 1600):
            row = {}
            outs = {}
            for dsk in (False, True):
                wr = plan_run(L - qn, L, dsk)
                q = q_all[:qn]
                row["nosplit_us" if dsk else "split_us"] = _tms(lambda: wr.run(q, kv, k_scale=1.0, v_scale=1.0))
                outs[dsk] = wr.run(q, kv, k_scale=1.0, v_scale=1.0)
            row["split_equals_nosplit"] = bool(torch.equal(outs[False], outs[True]))
            row["cost_ratio"] = round(row["nosplit_us"] / max(row["split_us"], 1e-9), 3)
            out[f"kv{L}_q{qn}"] = row
    big = plan_run(LMAX - 1600, LMAX, True).run(q_all[:1600], kv, k_scale=1.0, v_scale=1.0)
    small = plan_run(LMAX - 64, LMAX, True).run(q_all[1536:1600].contiguous(), kv, k_scale=1.0, v_scale=1.0)
    out["nosplit_tail64_vs_chunk1600_at_64k"] = bool(torch.equal(small, big[1536:]))
    R["A"] = out
    del kv, q_all, ws
    torch.cuda.empty_cache()
