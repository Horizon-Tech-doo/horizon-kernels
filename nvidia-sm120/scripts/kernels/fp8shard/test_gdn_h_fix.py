# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU tests for horizon_kernels/gdn_h_fix.py (fp8fix track, 27-09-2026). No GPU: runs vLLM 0.29's REAL vendored
# chunk_delta_h.py (plus its real index.py) under the Triton interpreter (TRITON_INTERPRET=1) inside a minimal
# stand-in "vllm" package. Needs torch (CPU) + triton (Linux; on a laptop use the tlinterp Docker image).
#   TRITON_INTERPRET=1 python test_gdn_h_fix.py <vllm029 site-packages copy containing vllm/> <path to gdn_h_fix.py>
# Checks:
#   1. the post-import hook restricts the fwd-h kernel's autotuner to num_warps=2 (FLA #953 space), and drops a
#      previously chosen config when the module was imported first;
#   2. the source guard refuses another vLLM version or a changed chunk_delta_h.py (vLLM left unchanged);
#   3. (numerics: GPU only, see the note at the end)
import importlib, os, shutil, sys, tempfile

import torch

SRC, FIX = sys.argv[1], sys.argv[2]
OPS = "vllm/third_party/flash_linear_attention/ops"
root = tempfile.mkdtemp()
os.makedirs(f"{root}/{OPS}")
os.makedirs(f"{root}/vllm/utils")
for d in ("vllm/third_party", "vllm/third_party/flash_linear_attention", OPS, "vllm/utils"):
    open(f"{root}/{d}/__init__.py", "a").close()
open(f"{root}/vllm/__init__.py", "w").write('__version__ = "0.29.0"\n')
open(f"{root}/vllm/triton_utils.py", "w").write("import triton\nimport triton.language as tl\ntldevice = None\n")
open(f"{root}/vllm/logger.py", "w").write("import logging\ndef init_logger(n):\n    return logging.getLogger(n)\n")
open(f"{root}/vllm/utils/gpu_sync_debug.py", "w").write(
    "import contextlib\n@contextlib.contextmanager\ndef gpu_sync_allowed():\n    yield\n")
open(f"{root}/{OPS}/utils.py", "w").write(
    "import triton.language as tl\nFLA_CHUNK_SIZE = 64\nuse_cuda_graph = False\n"
    "is_gather_supported = hasattr(tl, 'gather')\ndef tensor_cache(fn):\n    return fn\n")
# op.py aliases tl.exp at import time, which the interpreter cannot patch: a jit-wrapped stand-in (same math)
open(f"{root}/{OPS}/op.py", "w").write(
    "import triton\nimport triton.language as tl\n@triton.jit\ndef exp(x):\n    return tl.exp(x)\n"
    "@triton.jit\ndef exp2(x):\n    return tl.exp2(x)\n")
for f in ("chunk_delta_h.py", "index.py"):
    shutil.copy(f"{SRC}/{OPS}/{f}", f"{root}/{OPS}/{f}")
sys.path.insert(0, root)
spec = importlib.util.spec_from_file_location("gdn_h_fix", FIX)
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
os.environ["HORIZON_GDN_H_FIX"] = "1"
fails = 0


def check(name, ok, info=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""), flush=True)


# --- 2. guard
check("guard: source hash matches the vendored file", fix.source_ok()[0], fix.source_ok()[1])
# 29-09-2026: vLLM 0.30.0 vendors the byte-identical chunk_delta_h.py, so the guard accepts it (hash still checked)
check("guard: accepts vLLM 0.30.0", fix.source_ok(version="0.30.0")[0], fix.source_ok(version="0.30.0")[1])
check("guard: refuses vLLM 0.31.0", not fix.source_ok(version="0.31.0")[0], fix.source_ok(version="0.31.0")[1])
bad = f"{root}/changed.py"
open(bad, "w").write(open(f"{root}/{OPS}/chunk_delta_h.py").read() + "\n# changed\n")
check("guard: refuses a changed chunk_delta_h.py", not fix.source_ok(path=bad)[0], fix.source_ok(path=bad)[1])

# --- 1. hook
import triton  # noqa: E402

check("hook: installs", fix.install())
# Triton-interpreter compatibility (after the guard/install, before the import): the interpreter runs range() loops
# with Python ints (no .to(); unbounded, so the int64 widening is a no-op there).
cp = f"{root}/{OPS}/chunk_delta_h.py"
src = open(cp).read()
src = src.replace("i_t.to(tl.int64)", "i_t")
# Python's min() on an interpreter tensor (T comes from cu_seqlens) does not compare element values: tl.minimum
src = src.replace("last_idx = min((i_t + 1) * BT, T) - 1", "last_idx = tl.minimum((i_t + 1) * BT, T) - 1")
open(cp, "w").write(src)
m = importlib.import_module(fix.TARGET)
tuner = fix.unwrap_autotuner(getattr(m, fix.KERNEL))
cfg = sorted((c.kwargs["BV"], c.num_warps, c.num_stages) for c in tuner.configs)
check("hook: only num_warps=2 configs", {c[1] for c in cfg} == {2} and len(cfg) == 4, str(cfg))
tuner.cache = {("fake",): tuner.configs[0]}
fix.patch_module(m)
check("patch in place: previously chosen config dropped", tuner.cache == {})
# the interpreter has no GPU driver to benchmark with: every patched config "times" the same (the first wins)
tuner._bench = lambda *a, **kw: [1.0, 1.0, 1.0]

# 3. Numerics are NOT checked here: under the Triton 3.7 interpreter this kernel's state update is not faithful even
#    without the patch (the per-chunk state explodes after the first 64-token chunk while chunk 1 matches exactly),
#    so the interpreter cannot stand in for the GPU. The fix only changes launch configs (warps/stages/BV), not the
#    kernel's math; its numerical effect is measured on the GPU (run_chunkcheck2.sh arms Nf / Pf / Mf).
print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
