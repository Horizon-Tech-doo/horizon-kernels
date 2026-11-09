# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""G3v + caching package compatibility, real vLLM 0.30 classes, CPU only (01-10-2026).

usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<dir with the caching package's horizon_kernels + the candidate gdn_g3.py>
       $HK_VENV/bin/python test_g3v_compat_cpu.py          (exit 0 = all pass)
Installs the caching package's GDN checkpoint (HORIZON_GDN_CKPT=1, stop fill) and, when importable on CPU, the PI
runner hooks, then G3v (HORIZON_GDN_G3=1, HORIZON_GDN_G3_CHECK=1), in the plugin's order. Checks:
  1. the wrapper chain: QwenGatedDeltaNetAttention._forward_core = G3v's CHECK guard -> the checkpoint's wrapper ->
     vLLM's; the pure-decode path inside vLLM's _forward_core reaches the (G3v-hooked) fused decode method through
     self., so a checkpointed build does not bypass G3v; G3v's model-state hooks and the checkpoint's builder / spec
     hooks and PI's runner hooks are all in place at the same time
  2. _model_ctx with fine-grained prefix caching + prefix_match_unit 64 + HORIZON_GDN_CKPT + HORIZON_PI: G3v stays on
     and logs the compatibility marker; HORIZON_GDN_G3_COMPAT_STRICT=1 disables it; an unknown checkpoint stop mode
     disables it
  3. the CHECK guards: a checkpoint export into a spec-decode window block raises, a disjoint one passes; a
     copy-on-write block copy or a block zeroing that touches a live ring block raises, others pass; the live ring
     blocks are exactly window slots 1..3 of the rows whose request has pending records, in every GDN group
No kernel runs.
"""
import inspect
import os
import sys
import types

import torch

os.environ.update({"HORIZON_GDN_G3": "1", "HORIZON_GDN_G3_CHECK": "1", "HORIZON_GDN_CKPT": "1",
                   "HORIZON_GDN_CKPT_STOP": "fill", "HORIZON_PI": "1"})
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def raises(fn, exc=AssertionError):
    try:
        fn()
    except exc:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as QM  # noqa: E402
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState as M  # noqa: E402

Q = QM.QwenGatedDeltaNetAttention
VLLM_CORE = Q._forward_core
src = inspect.getsource(Q)
check("vLLM 0.30: the pure spec-decode path calls self._forward_core_decode_spec_post_conv_fused_norm (dynamic)",
      "self._forward_core_decode_spec_post_conv_fused_norm(" in src)

# fake G3v kernel module (no kernels run here)
g3 = types.ModuleType("horizon_kernels.g3v_ops.gdn_g3v")
g3.ring_geometry = lambda HV, H, RF, SW: (3, 4)
g3.expand = g3.step = g3.expand_sel = lambda *a, **k: None
g3.get_expand_sel_kernel = lambda *a, **k: None
import horizon_kernels  # noqa: E402

ops_pkg = types.ModuleType("horizon_kernels.g3v_ops")
ops_pkg.gdn_g3v = g3
sys.modules["horizon_kernels.g3v_ops"] = ops_pkg
sys.modules["horizon_kernels.g3v_ops.gdn_g3v"] = g3

from horizon_kernels import gdn_ckpt  # noqa: E402

gdn_ckpt.install()
CKPT_CORE = Q._forward_core
pi_ok = True
try:
    from horizon_kernels import prefill_invariant as PI
    PI.install_runner()
except Exception as e:  # noqa: BLE001
    pi_ok = False
    print(f"note: PI runner hooks not installable on CPU here ({type(e).__name__}: {str(e)[:120]})")
from horizon_kernels import gdn_g3 as P  # noqa: E402

P.install()
check("checkpoint wrapper installed over vLLM's _forward_core", getattr(CKPT_CORE, "_horizon_ckpt", False)
      and CKPT_CORE is not VLLM_CORE)
check("G3v CHECK guard wraps the checkpoint's wrapper (plugin order: checkpoint, then G3v)",
      Q._forward_core is not CKPT_CORE and Q._forward_core.__code__.co_name == "core")
check("G3v layer hook in place", getattr(Q._forward_core_decode_spec_post_conv_fused_norm, "_horizon_g3", False))
check("G3v model-state hooks in place", all(getattr(getattr(M, n), "_horizon_g3", False)
                                            for n in ("add_request", "preprocess_state", "prepare_attn",
                                                      "postprocess_state")))
from vllm.v1.attention.backends import gdn_attn as GB  # noqa: E402

check("checkpoint builder / spec hooks coexist (the checkpoint's, G3v patches neither)",
      GB.GDNAttentionMetadataBuilder.build.__code__.co_filename.endswith("gdn_ckpt.py")
      and Q.get_kv_cache_spec.__code__.co_filename.endswith("gdn_ckpt.py"))
if pi_ok:
    import vllm.v1.worker.gpu.model_runner as mr
    check("PI runner hooks coexist", getattr(mr.GPUModelRunner.gather_batch_req_state, "_horizon_pi", False))
from vllm.v1.worker.gpu import model_runner as mr  # noqa: E402
from vllm.v1.worker import utils as wu  # noqa: E402

check("CHECK copy / zero guards installed", getattr(mr.copy_kv_cache_blocks_inplace, "_horizon_g3", False)
      and wu.KVBlockZeroer.zero_block_ids.__name__ == "zero_block_ids")

# 2. _model_ctx compatibility decisions
LOGS = []
P._announce = lambda msg, *a: LOGS.append(msg % a)


def fresh_state():
    P._STATE.update({"disabled": None, "compat_announced": False})
    LOGS.clear()


def ms_with(fine, pmu):
    ms = object.__new__(M)
    ms._align_mode = True
    cc = types.SimpleNamespace(enable_mamba_fine_grained_prefix_cache=fine, prefix_match_unit=pmu)
    ms.vllm_config = types.SimpleNamespace(cache_config=cc,
                                           compilation_config=types.SimpleNamespace(static_forward_context={}))
    return ms


fresh_state()
P._model_ctx(ms_with(True, 64), None)
check("fine-grained + pmu 64 + checkpoint (fill) + PI: G3v stays on and logs the compatibility marker",
      P._STATE["disabled"] is None and any("caching features on (fine-grained=True, prefix_match_unit=64, PI=True, "
                                           "GDN checkpoint=fill)" in m for m in LOGS), str(LOGS))
fresh_state()
os.environ["HORIZON_GDN_G3_COMPAT_STRICT"] = "1"
P._model_ctx(ms_with(True, 64), None)
check("HORIZON_GDN_G3_COMPAT_STRICT=1 restores the refusal", P._STATE["disabled"] is not None
      and "COMPAT_STRICT" in P._STATE["disabled"])
del os.environ["HORIZON_GDN_G3_COMPAT_STRICT"]
fresh_state()
os.environ["HORIZON_GDN_CKPT_STOP"] = "something"
P._model_ctx(ms_with(False, None), None)
check("an unknown checkpoint stop mode disables G3v", P._STATE["disabled"] is not None
      and "HORIZON_GDN_CKPT_STOP=something" in P._STATE["disabled"])
os.environ["HORIZON_GDN_CKPT_STOP"] = "fill"
for mode in ("lcp", "cont"):
    fresh_state()
    os.environ["HORIZON_GDN_CKPT_STOP"] = mode
    P._model_ctx(ms_with(True, 64), None)
    check(f"checkpoint stop {mode}: G3v stays on", P._STATE["disabled"] is None)
os.environ["HORIZON_GDN_CKPT_STOP"] = "fill"
fresh_state()

# 3. CHECK guards
import vllm.forward_context as FC  # noqa: E402

lay = types.SimpleNamespace(prefix="model.layers.0.linear_attn")


def with_md(md, fn):
    orig = FC.get_forward_context
    FC.get_forward_context = lambda: types.SimpleNamespace(attn_metadata={lay.prefix: md})
    try:
        return fn()
    finally:
        FC.get_forward_context = orig


si = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [0, 0, 0, 0]], dtype=torch.int32)
md_hit = types.SimpleNamespace(horizon_ckpt=types.SimpleNamespace(slots=torch.tensor([5, 23])),
                               spec_state_indices_tensor=si, num_spec_decodes=2)
md_ok = types.SimpleNamespace(horizon_ckpt=types.SimpleNamespace(slots=torch.tensor([5, 31])),
                              spec_state_indices_tensor=si, num_spec_decodes=2)
check("CHECK: checkpoint export into a spec-decode window block raises",
      raises(lambda: with_md(md_hit, lambda: P._check_ckpt_slots(lay))))
check("CHECK: a disjoint checkpoint export passes",
      not raises(lambda: with_md(md_ok, lambda: P._check_ckpt_slots(lay)), Exception))
P._STATE["live"] = {12, 13, 14}
check("CHECK: a copy-on-write copy from a live ring block raises",
      raises(lambda: mr.copy_kv_cache_blocks_inplace([], 100, [(13, 40)])))
check("CHECK: a copy-on-write copy into a live ring block raises",
      raises(lambda: mr.copy_kv_cache_blocks_inplace([], 100, [(40, 14)])))
check("CHECK: an unrelated copy passes", not raises(lambda: mr.copy_kv_cache_blocks_inplace([], 100, [(40, 41)])))
check("CHECK: zeroing a live ring block raises",
      raises(lambda: wu.KVBlockZeroer.zero_block_ids(types.SimpleNamespace(), [3, 12])))
check("CHECK: zeroing other blocks reaches vLLM's zeroer",
      not raises(lambda: wu.KVBlockZeroer.zero_block_ids(types.SimpleNamespace(), [3, 4])))
# live ring blocks: rows whose request has pending records, window slots 1..3 per group
NRQ, HV = 8, 48
meta = torch.zeros(3, NRQ, HV, 4, dtype=torch.int32)
meta[1, 5, 7, 0] = 2 * 256 + 1            # request-state index 5 pending (one layer, one head suffices)
ctx = types.SimpleNamespace(ok=True, groups=[(1, None, None, None), (2, None, None, None)], SW=4, meta=meta)
ms = object.__new__(M)
ms._hz_g3_ctx = ctx
bt1 = torch.arange(100, 100 + 3 * 10, dtype=torch.int32).view(3, 10)
bt2 = torch.arange(500, 500 + 3 * 10, dtype=torch.int32).view(3, 10)
ms._hz_block_tables = (None, bt1, bt2)
ms._mamba_state_idx_gpu = torch.tensor([0, 0, 0, 0, 0, 3, 2, 0], dtype=torch.int32)
im = torch.tensor([6, 5, -1], dtype=torch.int64)       # batch row 1 = request 5 (col 3); row 0 = request 6 (none)
live = P._live_ring_blocks(ms, im, 3)
want = {int(bt1[1, 4]), int(bt1[1, 5]), int(bt1[1, 6]), int(bt2[1, 4]), int(bt2[1, 5]), int(bt2[1, 6])}
check("CHECK: live ring blocks = window slots 1..3 of pending requests, every group", live == want,
      f"{sorted(live)} vs {sorted(want)}")
print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
