# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels GDN chunk-state fix (fp8fix track, 27-09-2026): HORIZON_GDN_H_FIX=1 (default 0).
# RESULT (27-09-2026): NO EFFECT on this box. NVIDIA 27B with the fix is bit-identical to stock, and an
# unchunked run is just as bad, so this is NOT the cause of the long-prose loss (the stock autotuner already picks a
# 2-warp config here). Kept, default off, only as a guard for a future Triton / autotuner change.
#
# Symptom: every Qwen3.8 checkpoint served with its TRUE in_proj_qkv / in_proj_z magnitudes (NVIDIA ModelOpt FP8 GDN,
# official Qwen FP8, our NVFP4 with exact shard scales) matches production up to prompt token 8000 and breaks from
# token 8000 on (prose PPL x3 at 16k, x12-28 at 32k). 8000 is vLLM's first chunked-prefill boundary (8192-token budget
# aligned down to the 1600-token mamba block). The first chunk is bitwise identical between a 16k and a 32k request;
# only chunks that START FROM A CARRIED-OVER GDN STATE (h0) go wrong. our run notes, 27-09-2026 19:15.
#
# Suspect (upstream, not fixed in vLLM 0.29.0 / 0.30.0 / main): Triton <= 3.7 miscompiles the recurrence of
# flash-linear-attention's chunk_gated_delta_rule_fwd_kernel_h_blockdim64 on Blackwell (sm_100/103/120) with
# num_warps=4 (triton#9871, triton#10590; FLA #790, #945). FLA fixed it by restricting the kernel to num_warps=2 on
# Blackwell (FLA PR #953, commit 3eeef4e5, 20-06-2026). vLLM's vendored copy
# (vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py) still autotunes over num_warps [2, 4] x num_stages
# [2, 3, 4] x BV [32, 64]; vLLM 0.29 ships Triton 3.7.1.
#
# Fix: the same restriction as FLA #953 for SM120 (FLA's space on a < 166 KB shared-memory part):
#   num_warps 2, num_stages HORIZON_GDN_H_FIX_STAGES (default "2,1"), BV 32 and 64.
# Applied through a post-import hook on the vendored module, so nothing here imports vLLM's FLA package (whose utils
# query the CUDA device at import) in the API-server process, and no CUDA is touched at registration.
# Installs only on vLLM 0.29.0 with the exact chunk_delta_h.py source this was checked against (sha256 below);
# anything else logs a warning and leaves vLLM unchanged.
#
# Registration: horizon_kernels.vllm_plugin.register() calls install() when HORIZON_GDN_H_FIX=1, or (arm overlays)
# this file is deployed as the top-level module horizon_gdn_h_fix with its own vllm.general_plugins entry point.
import hashlib
import importlib.abc
import importlib.util
import os
import sys

TARGET = "vllm.third_party.flash_linear_attention.ops.chunk_delta_h"
KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
VLLM_VERSION = "0.29.0"
# 29-09-2026: vLLM 0.30.0 vendors the byte-identical chunk_delta_h.py (same sha256), so the fix applies unchanged;
# the source hash below stays the real guard.
VLLM_VERSIONS = ("0.29.0", "0.30.0")
SOURCE_SHA256 = "1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26"
STATE = {"installed": False, "patched": False, "reason": "", "configs": None}


def enabled() -> bool:
    return os.environ.get("HORIZON_GDN_H_FIX", "0") == "1"


def _log():
    try:
        from vllm.logger import init_logger
        return init_logger("vllm.horizon_kernels")
    except Exception:  # noqa: BLE001
        import logging
        return logging.getLogger("horizon_kernels")


def stages() -> list:
    return [int(s) for s in os.environ.get("HORIZON_GDN_H_FIX_STAGES", "2,1").split(",") if s.strip()]


def target_path() -> str:
    import vllm
    return os.path.join(os.path.dirname(vllm.__file__), "third_party", "flash_linear_attention", "ops",
                        "chunk_delta_h.py")


def source_ok(path: str | None = None, version: str | None = None) -> tuple[bool, str]:
    """(ok, reason): vLLM version and the vendored kernel source match what the fix was checked against."""
    try:
        if version is None:
            import vllm
            version = getattr(vllm, "__version__", "?")
        if version not in VLLM_VERSIONS:
            return False, f"vLLM {version} not in {VLLM_VERSIONS}"
        path = path or target_path()
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        if sha != SOURCE_SHA256:
            return False, f"{os.path.basename(path)} sha256 {sha[:12]} != {SOURCE_SHA256[:12]} (source changed)"
        return True, "ok"
    except Exception as ex:  # noqa: BLE001
        return False, f"check failed: {ex}"


def unwrap_autotuner(fn):
    """heuristics -> autotune -> jit: return the object that owns .configs (the Autotuner)."""
    seen = 0
    while not hasattr(fn, "configs"):
        fn = fn.fn
        seen += 1
        if seen > 8:
            raise TypeError("no autotuner found")
    return fn


def fixed_configs(triton_mod):
    return [triton_mod.Config({"BV": bv}, num_warps=2, num_stages=s) for s in stages() for bv in (32, 64)]


def patch_module(mod, triton_mod=None) -> list:
    """Restrict the module's fwd-h kernel autotuner to 2 warps; drop any config it already picked. Returns configs."""
    if triton_mod is None:
        import triton as triton_mod
    tuner = unwrap_autotuner(getattr(mod, KERNEL))
    tuner.configs = fixed_configs(triton_mod)
    if hasattr(tuner, "cache"):
        tuner.cache = {}                     # forget a config chosen before the patch (none in practice)
    if hasattr(tuner, "configs_timings"):
        tuner.configs_timings = None
    STATE["patched"] = True
    STATE["configs"] = [(c.kwargs.get("BV"), c.num_warps, c.num_stages) for c in tuner.configs]
    return tuner.configs


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            cfg = patch_module(module)
            _log().info("horizon_kernels: GDN fwd-h kernel restricted to num_warps=2 (FLA #953): %s configs %s",
                        len(cfg), STATE["configs"])
        except Exception as ex:  # noqa: BLE001
            STATE["reason"] = f"patch failed: {ex}"
            _log().warning("horizon_kernels: GDN fwd-h fix NOT applied (%s)", ex)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def install(check_source: bool = True) -> bool:
    if STATE["installed"]:
        return True
    if check_source:
        ok, why = source_ok()
        if not ok:
            STATE["reason"] = why
            _log().warning("horizon_kernels: HORIZON_GDN_H_FIX=1 but not installed: %s", why)
            return False
    if TARGET in sys.modules:                  # already imported in this process: patch in place
        patch_module(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
    STATE["installed"] = True
    STATE["reason"] = "installed"
    return True


def register():
    """Stand-alone vllm.general_plugins entry point (arm overlays). No CUDA here."""
    if enabled():
        install()
