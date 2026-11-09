#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""test_pi_cpu.py (Horizon Tech, 30-09-2026): CPU tests of the HORIZON_PI hooks (no CUDA context is created).
Run with the vLLM 0.30 venv python and the candidate plugin package first on PYTHONPATH:
  PYTHONPATH=$HK_DATA/horizon-pkgs-pi-30-09-2026:<tl-pkgs> python test_pi_cpu.py
Each test runs in a fresh subprocess (the switches are read at import time)."""
import os
import subprocess
import sys

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


PRE = r'''
import os, sys, types, torch
torch.cuda.is_current_stream_capturing = lambda: CAPT["c"]   # no CUDA context in a CPU test (gpushare blocks cuInit)
CAPT = {"c": False}
from horizon_kernels import prefill_invariant as pi
def step(prefill, capturing=False):
    pi._STEP["prefill"] = prefill; CAPT["c"] = capturing
'''


@test
def step_classifier():
    return PRE + r'''
os.environ["HORIZON_PI"] = "1"
step(False); assert pi.prefill_step() is False
step(True); assert pi.prefill_step() is True
step(True, capturing=True); assert pi.prefill_step() is True        # v3: a prefill step captures its own PI graph
step(False, capturing=True); assert pi.prefill_step() is False      # startup capture (dummy batches): decode graphs
os.environ["HORIZON_PI_GRAPHS"] = "0"
step(True, capturing=True); assert pi.prefill_step() is False       # v2 (eager prefill): captures are decode only
'''


RUNNER = r'''
import dataclasses
from vllm.config.compilation import CUDAGraphMode
mr = types.ModuleType("vllm.v1.worker.gpu.model_runner")
seen = {}
@dataclasses.dataclass(frozen=True)
class BD:
    num_tokens: int
    num_active_loras: int = 0
class GPUModelRunner:
    def gather_batch_req_state(self, so, dummy_run):
        return (None if dummy_run else types.SimpleNamespace(has_prefill=so)), 4
def dispatch_cg_and_sync_dp(*a, need_eager=False, **kw):
    seen["need_eager"] = need_eager; return "desc"
def set_forward_context(*a, **kw):
    seen["bd"] = kw.get("batch_descriptor"); return "ctx"
mr.GPUModelRunner, mr.dispatch_cg_and_sync_dp, mr.set_forward_context = GPUModelRunner, dispatch_cg_and_sync_dp, set_forward_context
sys.modules["vllm.v1.worker.gpu.model_runner"] = mr
import vllm.v1.worker.gpu as g; g.model_runner = mr
pi._log = lambda: types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
'''


@test
def runner_hook_graphs():
    return "os.environ['HORIZON_PI']='1'\n" + PRE + RUNNER + r'''
pi.install_runner()
r = GPUModelRunner()
r.gather_batch_req_state(True, False); mr.dispatch_cg_and_sync_dp(1, 2, need_eager=False)
assert pi._STEP["prefill"] is True and seen["need_eager"] is False, seen       # graphs mode: normal dispatch
mr.set_forward_context(batch_descriptor=BD(64), cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
assert seen["bd"] == BD(64, pi.PI_GRAPH_KEY), seen                               # own PI graph key
mr.set_forward_context(batch_descriptor=BD(64), cudagraph_runtime_mode=CUDAGraphMode.NONE)
assert seen["bd"] == BD(64), seen                                                 # eager: untouched
r.gather_batch_req_state(False, False)
mr.set_forward_context(batch_descriptor=BD(64), cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
assert seen["bd"] == BD(64), seen                                                 # decode step: startup graphs
'''


@test
def runner_hook_startup_capture():
    return "os.environ['HORIZON_PI']='1'\n" + PRE + RUNNER + r'''
import vllm.compilation.cuda_graph as cg
import vllm.v1.worker.gpu.cudagraph_utils as cgu
import vllm.forward_context as fc
log = []
CTX = {"bd": None, "mode": None}
class FakeWrapper:
    runtime_mode = CUDAGraphMode.PIECEWISE
    def __init__(self): self.concrete_cudagraph_entries = {}
    def runnable(self, x): log.append(("eager", CTX["bd"])); return "eager-out"
    def __call__(self, x):
        bd = CTX["bd"]
        if bd not in self.concrete_cudagraph_entries:
            self.concrete_cudagraph_entries[bd] = types.SimpleNamespace(cudagraph=object()); log.append(("capture", bd))
            return "cap"
        log.append(("replay", bd)); return "rep"
cg.CUDAGraphWrapper = FakeWrapper
W = FakeWrapper()
def sfc(*a, **kw):
    CTX["bd"], CTX["mode"] = kw.get("batch_descriptor"), kw.get("cudagraph_runtime_mode"); return "ctx"
cgu.set_forward_context = sfc
class FakeBase:                                   # vLLM: CudaGraphManager.capture(create_forward_fn, desc)
    def capture(self, create_forward_fn, progress_bar_desc="x"):
        for n in (8, 16):
            f = create_forward_fn(types.SimpleNamespace(cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=n), warmup=True)
            f(CUDAGraphMode.NONE); f(CUDAGraphMode.PIECEWISE)
        g = create_forward_fn(types.SimpleNamespace(cg_mode=CUDAGraphMode.FULL, num_tokens=4), warmup=True)
        g(CUDAGraphMode.FULL)
class FakeModelManager(FakeBase):                 # vLLM: ModelCudaGraphManager.capture(model, ...) -> super().capture
    def capture(self, model, create):
        super().capture(create, "Capturing CUDA graphs")
cgu.CudaGraphManager, cgu.ModelCudaGraphManager = FakeBase, FakeModelManager
fc.is_forward_context_available = lambda: True
fc.get_forward_context = lambda: types.SimpleNamespace(batch_descriptor=CTX["bd"], cudagraph_runtime_mode=CTX["mode"])
pi.install_runner()
modes = []
def create_forward_fn(desc, warmup=True):
    def fwd(mode):
        cgu.set_forward_context(batch_descriptor=BD(desc.num_tokens) if mode == CUDAGraphMode.PIECEWISE else None,
                                cudagraph_runtime_mode=mode)
        modes.append((desc.num_tokens, mode.name, pi._STEP["prefill"]))
        if mode == CUDAGraphMode.PIECEWISE:
            W(0)
    return fwd
FakeBase().capture(create_forward_fn)          # a drafter manager: no PI twins
assert [e for e in log if e[0] == "capture"] == [("capture", BD(8)), ("capture", BD(16))], log
log.clear(); W.concrete_cudagraph_entries.clear(); modes.clear()
FakeModelManager().capture(object(), create_forward_fn)
assert pi._STEP["prefill"] is False, "flag must be reset after capture"
caps = [e for e in log if e[0] == "capture"]
assert caps == [("capture", BD(8)), ("capture", BD(8, pi.PI_GRAPH_KEY)),
                ("capture", BD(16)), ("capture", BD(16, pi.PI_GRAPH_KEY))], caps
assert (8, "NONE", True) in modes and (8, "PIECEWISE", True) in modes, modes      # PI warm-up + PI capture
assert (4, "FULL", False) in modes and not any(m[0] == 4 and m[2] for m in modes), modes   # FULL graphs untouched
log.clear()
r = GPUModelRunner(); r.gather_batch_req_state(True, False)
torch.cuda.is_current_stream_capturing = lambda: False
CTX["bd"], CTX["mode"] = BD(8, pi.PI_GRAPH_KEY), CTX["mode"]
CTX["mode"] = CUDAGraphMode.PIECEWISE
assert W(0) == "rep" and log == [("replay", BD(8, pi.PI_GRAPH_KEY))], log          # serving: replays the startup twin
log.clear(); CTX["bd"] = BD(24, pi.PI_GRAPH_KEY)
assert W(0) == "eager-out" and log == [("eager", BD(24, pi.PI_GRAPH_KEY))], log   # no twin: eager, never capture
'''


@test
def runner_hook():
    return "os.environ['HORIZON_PI']='1'; os.environ['HORIZON_PI_GRAPHS']='0'\n" + PRE + RUNNER + r'''
pi.install_runner()
r = GPUModelRunner()
r.gather_batch_req_state(True, False); mr.dispatch_cg_and_sync_dp(1, 2, need_eager=False)
assert pi._STEP["prefill"] is True and seen["need_eager"] is True, seen
assert mr.set_forward_context is set_forward_context                             # v2: context untouched
r.gather_batch_req_state(False, False); mr.dispatch_cg_and_sync_dp(1, 2, need_eager=False)
assert pi._STEP["prefill"] is False and seen["need_eager"] is False, seen
r.gather_batch_req_state(True, True)          # dummy run: no batch state -> not prefill
assert pi._STEP["prefill"] is False
r.gather_batch_req_state(False, False); mr.dispatch_cg_and_sync_dp(1, 2, need_eager=True)
assert seen["need_eager"] is True             # the profile run's own eager request is kept
'''


NV = PRE + r'''
from horizon_kernels import vllm_plugin as vp
calls = []
class V:
    def __init__(s, name): s.name, s.tags = name, ({"exact"} if name == "regfrag" else set())
    def fn(s, x, layer):
        calls.append((s.name, x.shape[0])); return torch.zeros(x.shape[0], layer.output_size_per_partition)
class Op:
    variants = {"regfrag": V("regfrag"), "fp4x2": V("fp4x2")}
    default = "regfrag"
    def select(s, key, m): return s.variants["regfrag" if m <= 512 else "fp4x2"]
vp._OP = Op()
layer = types.SimpleNamespace(output_size_per_partition=8, input_size_per_partition=4, horizon_nvfp4=object(),
                              horizon_regfrag=object(), horizon_exact_only=False, horizon_has_marlin=False)
vp._LAYERS[:] = [lambda: layer]
def run(m, prefill, capturing=False):
    step(prefill, capturing); calls.clear(); vp._nvfp4_linear_impl(torch.zeros(m, 4, dtype=torch.bfloat16), 0)
    return calls[-1]
'''


@test
def nvfp4_on():
    return "os.environ['HORIZON_PI']='1'\n" + NV + r'''
assert run(17, True) == ("fp4x2", 17)                        # prefill step: fp4x2 at any row count
assert run(4, False) == ("regfrag", 4)                       # decode-only step keeps the table
assert run(17, True, capturing=True) == ("fp4x2", 17)        # v3: runtime capture of a prefill step's own graph
assert run(17, False, capturing=True) == ("regfrag", 17)     # startup capture (dummy batches): decode graphs
assert run(4096, True) == ("fp4x2", 4096)
layer.horizon_exact_only = True                              # LM heads stay on the exact class
assert run(17, True) == ("regfrag", 17)
'''


@test
def nvfp4_off():
    return "os.environ.pop('HORIZON_PI', None)\n" + NV + r'''
assert run(17, True) == ("regfrag", 17)                      # switch off: unchanged table dispatch
assert run(4096, True) == ("fp4x2", 4096)
'''


@test
def nvfp4_sub_switch_off():
    return "os.environ['HORIZON_PI']='1'; os.environ['HORIZON_PI_NVFP4']='0'\n" + NV + r'''
assert run(17, True) == ("regfrag", 17)
'''


SK = PRE + r'''
import horizon_kernels.k5 as k5
calls = []
mod = types.ModuleType("tilelang.contrib.horizon_fused.skinny_bf16")
mod.SUPPORTS_M = 48
mod.linear = lambda x, w, cfg=None, out=None: (calls.append(("skinny", x.shape[0])), torch.zeros(x.shape[0], w.shape[0]))[1]
# stub the tilelang package chain: importing the real tilelang runs torch.cuda.is_available() (cuInit), which gpushare
# turns into a hang when no device is visible
for name in ("tilelang", "tilelang.contrib", "tilelang.contrib.horizon_fused"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["tilelang.contrib"].horizon_fused = sys.modules["tilelang.contrib.horizon_fused"]
sys.modules["tilelang"].contrib = sys.modules["tilelang.contrib"]
sys.modules["tilelang.contrib.horizon_fused.skinny_bf16"] = mod
sys.modules["tilelang.contrib.horizon_fused"].skinny_bf16 = mod
import torch.nn.functional as F
F_lin = F.linear
k5.F.linear = lambda x, w: (calls.append(("cublas", x.shape[0])), F_lin(x.float(), w.float()).to(x.dtype))[1]
k5._SK_LAYERS[:] = [types.SimpleNamespace(weight=torch.zeros(96, 64, dtype=torch.bfloat16))]
def run(m, prefill):
    step(prefill); calls.clear(); k5._skinny_impl(torch.zeros(m, 64, dtype=torch.bfloat16), 0); return calls[-1]
'''


@test
def ba_on():
    return "os.environ['HORIZON_PI']='1'\n" + SK + r'''
assert run(16, False) == ("skinny", 16)
assert run(256, False) == ("cublas", 256)                    # decode above 48 rows keeps cuBLAS
assert run(256, True) == ("skinny", 256)                     # prefill step: skinny at every row count
assert run(8192, True) == ("skinny", 8192)
'''


@test
def ba_off():
    return "os.environ.pop('HORIZON_PI', None)\n" + SK + r'''
assert run(256, True) == ("cublas", 256)
assert run(16, True) == ("skinny", 16)
'''


FI = r'''
fi = types.ModuleType("flashinfer")          # stub: importing the real flashinfer without a GPU blocks under gpushare
class _P:
    def plan(self, *a, **kw): pass
class _D:
    def plan(self, *a, **kw): pass
fi.BatchPrefillWithPagedKVCacheWrapper, fi.BatchDecodeWithPagedKVCacheWrapper = _P, _D
sys.modules["flashinfer"] = fi
'''


@test
def attn_plan_patch():
    return "os.environ['HORIZON_PI']='1'\n" + PRE + FI + r'''
import flashinfer
seen = {}
def fake_plan(self, *a, **kw): seen.update(kw)
flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan = fake_plan
pi.install_attn()
step(False)
flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan(object(), 1, 2, disable_split_kv=False, fixed_split_size=-1)
assert seen["disable_split_kv"] is False and seen["fixed_split_size"] == -1, seen   # decode-only step: stock plan
seen.clear(); step(True)
flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan(object(), 1, 2, disable_split_kv=False, fixed_split_size=-1)
assert seen["disable_split_kv"] is True and "fixed_split_size" not in seen, seen
assert pi._ORIG_PLAN is fake_plan
assert not getattr(getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper").plan, "_horizon_pi", False)
'''


@test
def attn_off_untouched():
    return "os.environ['HORIZON_PI']='1'; os.environ['HORIZON_PI_ATTN']='0'\n" + PRE + FI + r'''
import flashinfer
orig = flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan
pi.install_attn()
assert flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan is orig
'''


if __name__ == "__main__":
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    bad = 0
    only = set(sys.argv[1:])
    for name, fn in TESTS.items():
        if only and name not in only:
            continue
        r = subprocess.run([sys.executable, "-c", "import os\n" + fn()], env=env, capture_output=True, text=True,
                           timeout=600)
        ok = r.returncode == 0
        bad += not ok
        print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            print(r.stdout[-2000:], r.stderr[-3000:])
    n = len(only) if only else len(TESTS)
    print(f"{n - bad}/{n} passed")
    sys.exit(1 if bad else 0)
