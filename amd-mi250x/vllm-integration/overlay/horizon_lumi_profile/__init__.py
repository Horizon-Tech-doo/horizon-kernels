"""LUMI research only (track PROFILE). vLLM general plugin: where a serving step's wall time goes. Changes no result; does nothing unless HPF_ENABLE=1.
One JSON line per model-runner step into HPF_DIR/hpf-<pid>.jsonl (one file per tensor-parallel worker).
HPF_MODE:
  step  two clock reads around GPUModelRunner.execute_model and sample_tokens, no synchronisation. GPU work is asynchronous, so only the start-to-start period of consecutive steps is a real time.
  sync  the same plus torch.cuda.synchronize() at the start of execute_model and at the end of sample_tokens: the step's own wall time, at the cost of one wait per step.
  ops   sync plus a synchronised timer around every operator that vLLM runs as an opaque Python call at run time, also inside its torch.compile'd model:
          every dense linear layer (ROCm: the custom op rocm_unquantized_gemm; replaced by an identical-result op of ours that times the stock implementation and records M, N, K, dtype and
          which path the stock dispatch took: wvSplitK / LLMM1 / wvSplitKrc (vLLM's own small-M kernels), aiter, or torch F.linear),
          attention forward and KV-cache write (RocmAttentionImpl / TritonAttentionImpl), the Gated-DeltaNet core (QwenGatedDeltaNetAttention._forward_core*; its conv and delta-rule calls as nested sub-timers),
          tensor-parallel all-reduce / all-gather (GroupCoordinator).
        The time between two timed operators (norms, activations, residual adds, RoPE, reshapes: the Inductor-compiled code, plus Python) is booked as glue under "<previous op class>><next op class>".
        Operators plus glue partition the forward exactly. Use with --compilation-config '{"cudagraph_mode":"NONE"}': a replayed HIP graph runs no Python, so nothing below the step would be seen.
        Synchronising after every operator removes overlap between CPU launch and GPU work, so the sum can exceed the unprofiled step; compare with the sync and step arms."""
import json, os, time

MODE = os.environ.get("HPF_MODE", "step")
_S = None            # the open step record
_mark = 0.0          # end of the last timed operator (ops mode)
_prev = "start"
_depth = 0
_path = "linear"
_last_start = None
_f = None
_meta_done = False
_n = 0


def _log(**k):
    print("HPF " + json.dumps({"pid": os.getpid(), "t": round(time.time(), 2), **k}), flush=True)


def _out():
    global _f
    if _f is None:
        d = os.environ.get("HPF_DIR", ".")
        _f = open(os.path.join(d, f"hpf-{os.getpid()}.jsonl"), "a", buffering=1)
    return _f


def _op(label, cls, f):
    """Run f() as one timed operator of the open step. label: str or callable evaluated after the call. Nested calls are booked under sub and do not move the partition."""
    global _mark, _prev, _depth
    S = _S
    if S is None or MODE != "ops":
        return f()
    import torch
    if torch.cuda.is_current_stream_capturing():
        return f()
    now = time.perf_counter
    if _depth:
        torch.cuda.synchronize(); t0 = now()
        try:
            return f()
        finally:
            torch.cuda.synchronize()
            e = S["sub"].setdefault(label() if callable(label) else label, [0, 0.0]); e[0] += 1; e[1] += now() - t0
    torch.cuda.synchronize(); t0 = now()
    k = _prev + ">" + cls
    S["glue"][k] = S["glue"].get(k, 0.0) + t0 - _mark
    _depth += 1
    try:
        return f()
    finally:
        _depth -= 1
        torch.cuda.synchronize(); t1 = now()
        e = S["ops"].setdefault(label() if callable(label) else label, [0, 0.0]); e[0] += 1; e[1] += t1 - t0
        _mark = t1; _prev = cls


def _close():
    global _S
    S = _S
    if S is None:
        return
    _S = None
    S.pop("_t0", None)
    S["ops"] = {k: [v[0], round(v[1] * 1e3, 4)] for k, v in S["ops"].items()}
    S["sub"] = {k: [v[0], round(v[1] * 1e3, 4)] for k, v in S["sub"].items()}
    S["glue"] = {k: round(v * 1e3, 4) for k, v in S["glue"].items()}
    _out().write(json.dumps(S) + "\n")


def _meta(runner):
    """Once: every weight of two or more dimensions by shape and dtype, so the GEMM shapes can be matched to layers."""
    global _meta_done
    _meta_done = True
    try:
        import torch
        w = {}
        for n, p in runner.model.named_parameters():
            if p.dim() >= 2:
                k = f"{n.split('.')[-2] if n.count('.') else n}|{'x'.join(map(str, p.shape))}|{str(p.dtype)[6:]}"
                e = w.setdefault(k, [0, 0]); e[0] += 1; e[1] += p.numel() * p.element_size()
        rank = None
        try:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            pass
        _out().write(json.dumps({"kind": "meta", "pid": os.getpid(), "mode": MODE, "rank": rank, "device": torch.cuda.get_device_name(0), "weights": w,
                                 "param_bytes": sum(p.numel() * p.element_size() for p in runner.model.parameters())}) + "\n")
    except Exception as e:
        _log(ev="meta_error", err=repr(e)[:300])


def _wrap_runner(R):
    o_exec, o_samp, o_fwd = R.execute_model, R.sample_tokens, getattr(R, "_model_forward", None)

    def execute_model(self, scheduler_output, *a, **k):
        global _S, _mark, _prev, _last_start, _n
        import torch
        if not _meta_done:
            _meta(self)
        _close()
        if MODE != "step":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ns = scheduler_output.num_scheduled_tokens
        v = list(ns.values()) if hasattr(ns, "values") else []
        _n += 1
        S = {"kind": "step", "i": _n, "t": round(time.time(), 4), "period_ms": None if _last_start is None else round((t0 - _last_start) * 1e3, 4), "ntok": int(scheduler_output.total_num_scheduled_tokens),
             "nreq": len(v), "maxq": max(v, default=0), "n1": sum(1 for x in v if x == 1), "ops": {}, "glue": {}, "sub": {}, "_t0": t0}
        _last_start = t0
        _S = S; _mark = t0; _prev = "start"
        try:
            return o_exec(self, scheduler_output, *a, **k)
        finally:
            if MODE != "step":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if MODE == "ops":
                kk = _prev + ">exec_end"; S["glue"][kk] = S["glue"].get(kk, 0.0) + t1 - _mark
                _mark = t1; _prev = "exec_end"
            S["exec_ms"] = round((t1 - t0) * 1e3, 4)

    def _model_forward(self, *a, **k):
        import torch
        S = _S
        if S is None:
            return o_fwd(self, *a, **k)
        global _mark, _prev
        if MODE != "step":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if MODE == "ops":
            S["glue"]["start>fwd"] = S["glue"].get("start>fwd", 0.0) + t0 - _mark
            _mark = t0; _prev = "fwd"
        S["pre_ms"] = round((t0 - S["_t0"]) * 1e3, 4)
        try:
            return o_fwd(self, *a, **k)
        finally:
            if MODE != "step":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if MODE == "ops":
                kk = _prev + ">fwd_end"; S["glue"][kk] = S["glue"].get(kk, 0.0) + t1 - _mark
                _mark = t1; _prev = "fwd_end"
            S["fwd_ms"] = round((t1 - t0) * 1e3, 4)

    def sample_tokens(self, *a, **k):
        import torch
        S = _S
        if S is None or "exec_ms" not in S:
            return o_samp(self, *a, **k)
        t0 = time.perf_counter()
        S["between_ms"] = round((t0 - S["_t0"]) * 1e3 - S["exec_ms"], 4)
        try:
            return o_samp(self, *a, **k)
        finally:
            if MODE != "step":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            S["sample_ms"] = round((t1 - t0) * 1e3, 4); S["step_ms"] = round((t1 - S["_t0"]) * 1e3, 4)
            _close()

    # execute_model stores _t0 only at its end; the forward wrapper needs it earlier
    def execute_model_outer(self, scheduler_output, *a, **k):
        r = None
        try:
            r = execute_model(self, scheduler_output, *a, **k)
            return r
        finally:
            if r is not None and _S is not None:      # no sample_tokens call follows a step that returns its output directly
                _close()

    R.execute_model, R.sample_tokens = execute_model_outer, sample_tokens
    if o_fwd is not None:
        R._model_forward = _model_forward


def _wrap_method(cls, name, label, klass):
    orig = getattr(cls, name, None)
    if orig is None:
        return False

    def w(self, *a, **k):
        return _op(label, klass, lambda: orig(self, *a, **k))
    w.__wrapped__ = orig
    setattr(cls, name, w)
    return True


def _wrap_global(mod, name, label):
    orig = getattr(mod, name, None)
    if orig is None or not callable(orig):
        return False

    def w(*a, **k):
        return _op(label, label, lambda: orig(*a, **k))
    w.__wrapped__ = orig
    setattr(mod, name, w)
    return True


def _install_ops():
    import torch
    done = []
    # dense linear layers
    from vllm.model_executor.layers import utils as U
    from vllm.utils.torch_utils import direct_register_custom_op
    stock = U.rocm_unquantized_gemm_impl
    import vllm._custom_ops as C

    def mark(name):
        o = getattr(C, name, None)
        if o is None:
            return

        def w(*a, **k):
            global _path
            _path = name
            return o(*a, **k)
        setattr(C, name, w)
    for nm in ("wvSplitK", "wvSplitKrc", "LLMM1"):
        mark(nm)

    def hpf_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        global _path
        if _S is None:
            return stock(x, weight, bias)
        _path = "linear"
        n = x.numel() // x.size(-1)
        N, K = weight.shape[0], weight.shape[1]
        return _op(lambda: f"gemm|{_path}|{n}|{N}|{K}|{str(x.dtype)[6:]}|{str(weight.dtype)[6:]}|b{int(bias is not None)}", f"g{N}x{K}", lambda: stock(x, weight, bias))

    direct_register_custom_op(op_name="hpf_gemm", op_func=hpf_gemm, fake_impl=U.rocm_unquantized_gemm_fake)

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        return torch.ops.vllm.hpf_gemm(x, weight, bias)
    U.rocm_unquantized_gemm = rocm_unquantized_gemm
    done.append("gemm")
    # attention
    from vllm.v1.attention.backends import rocm_attn, triton_attn
    for cls in (rocm_attn.RocmAttentionImpl, triton_attn.TritonAttentionImpl):
        if _wrap_method(cls, "forward", "attn", "attn"): done.append(cls.__name__ + ".forward")
        if _wrap_method(cls, "do_kv_cache_update", "kv_write", "kvw"): done.append(cls.__name__ + ".do_kv_cache_update")
    # Gated-DeltaNet core
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as G
    for nm in sorted(n for n in vars(G.QwenGatedDeltaNetAttention) if n.startswith("_forward_core")):
        if _wrap_method(G.QwenGatedDeltaNetAttention, nm, "gdn:" + nm, "gdn"): done.append(nm)
    for nm in ("fla_chunk_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "fused_recurrent_gated_delta_rule_packed_decode", "fused_gdn_gating",
               "gdn_aiter_fused_reshape_causal_conv1d_update_single_token", "gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule"):
        if _wrap_global(G, nm, "gdnsub:" + nm): done.append(nm)
    # tensor-parallel communication
    from vllm.distributed import parallel_state as PS
    for nm, lb in (("_all_reduce_out_place", "all_reduce"), ("_all_gather_out_place", "all_gather"), ("_reduce_scatter_out_place", "reduce_scatter")):
        if _wrap_method(PS.GroupCoordinator, nm, lb, "comm"): done.append(nm)
    return done


_done = False


def register():
    global _done
    if _done or os.environ.get("HPF_ENABLE") != "1":
        return
    _done = True
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    _wrap_runner(GPUModelRunner)
    done = _install_ops() if MODE == "ops" else []
    try:                                              # the V2 runner (vllm/v1/worker/gpu/model_runner.py) is the one this server uses; it has no _model_forward
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as R2
        _wrap_runner(R2)
        done.append("runner_v2")
        if MODE == "ops":
            for nm in ("add_requests", "update_requests", "finish_requests", "free_states", "prepare_inputs", "prepare_attn", "sample", "postprocess_sampled"):
                if _wrap_method(R2, nm, "run:" + nm, "run"): done.append("v2." + nm)
    except Exception as e:
        _log(ev="runner_v2_error", err=repr(e)[:300])
    _log(ev="registered", mode=MODE, wrapped=done)
