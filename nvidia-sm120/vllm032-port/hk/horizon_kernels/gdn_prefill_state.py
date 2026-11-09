# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Canonical stored-state rounding for the FLA gated-delta recurrence (default off).

The arithmetic state must cross every reusable FLA chunk boundary in the same
precision as the cached state. Otherwise an in-call fp32 carry differs from a
cached fp16 resume. Shared operation geometry stays in FLA; no model-name dispatch.
This experimental adapter applies to the pinned FLA GDN operation sources listed in SOURCES only (vLLM 0.29/0.30 and
vLLM 0.32.0; the 0.32 kernel switched from block pointers to masked pointer arithmetic but keeps the same recurrence
loop, so the same transformation applies; ported 24-10-2026).
"""
import functools
import hashlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

TARGET = "vllm.third_party.flash_linear_attention.ops.chunk"
PACKAGE = "vllm.third_party.flash_linear_attention.ops"
SOURCE_SHA256 = "1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26"   # vLLM 0.29/0.30 (kept for tests)
# sha256 of vLLM's chunk_delta_h.py -> the private (transformed) copy shipped in gdn_round_ops/
SOURCES = {
    SOURCE_SHA256: "chunk_delta_h.py",
    "240fbe430200c4168799c05243edffc0b8dce7b41651b95233266d632b70071b": "chunk_delta_h_v032.py",   # vLLM 0.32.0
}
STATE = {"installed": False, "patched": False}


def transform_source(source):
    """Produce a private kernel module; never modify the installed vendor source."""
    if hashlib.sha256(source.encode()).hexdigest() not in SOURCES:
        raise ValueError("FLA state kernel source changed; review the recurrence adapter")
    replacements = {
        "    USE_EXP2: tl.constexpr,\n": "    USE_EXP2: tl.constexpr,\n    STATE_ROUND_DTYPE: tl.constexpr,\n",
        "    use_exp2: bool = False,\n": "    use_exp2: bool = False,\n    state_round_dtype: str | None = None,\n",
        "        USE_EXP2=use_exp2,\n": (
            "        USE_EXP2=use_exp2,\n"
            "        STATE_ROUND_DTYPE=(tl.float16 if state_round_dtype == 'float16' else\n"
            "                           tl.bfloat16 if state_round_dtype == 'bfloat16' else None),\n"),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("ambiguous FLA adapter anchor")
        source = source.replace(old, new)
    # Insert inside the recurrence loop after all key slices have been updated.
    # The widened result is used by the NEXT chunk and exported as the checkpoint.
    round_lines = ["        if STATE_ROUND_DTYPE is not None:"]
    for i in range(1, 5):
        indent = "            "
        if i > 1:
            round_lines.append(f"            if K > {(i - 1) * 64}:")
            indent += "    "
        round_lines.extend([
            indent + "if STATE_ROUND_DTYPE == tl.float16:",
            indent + f"    b_h{i} = tl.minimum(tl.maximum(b_h{i}, -65504.0), 65504.0)",
            indent + f"b_h{i} = b_h{i}.to(STATE_ROUND_DTYPE).to(tl.float32)",
        ])
    anchor = "    # epilogue\n"
    if source.count(anchor) != 1:
        raise ValueError("missing unique recurrence epilogue")
    return source.replace(anchor, "\n".join(round_lines) + "\n" + anchor)


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def patch_module(module, private_file=None, storage_dtype=None):
    if getattr(module.chunk_gated_delta_rule_fwd_h, "_horizon_state_round", False):
        return
    storage_dtype = storage_dtype or os.environ["HORIZON_GDN_PREFILL_STATE_ROUND"]
    if storage_dtype not in ("float16", "bfloat16"):
        raise ValueError("prefill state round requires float16 or bfloat16")
    original_file = Path(module.__file__).with_name("chunk_delta_h.py")
    original_text = original_file.read_text()
    sha = hashlib.sha256(original_text.encode()).hexdigest()
    if sha not in SOURCES:
        raise ValueError(f"FLA state kernel source changed (sha256 {sha}); review the recurrence adapter")
    private_file = Path(private_file or Path(__file__).with_name("gdn_round_ops") / SOURCES[sha])
    from .gdn_prefill_export import transform_source as export_transform
    expected = export_transform(transform_source(original_text))
    if private_file.read_text() != expected:
        raise ValueError("private prefill-state kernel does not match the pinned transformation")
    name = PACKAGE + ".horizon_state_round_h"
    spec = importlib.util.spec_from_file_location(name, private_file)
    private = importlib.util.module_from_spec(spec)
    sys.modules[name] = private
    spec.loader.exec_module(private)

    @functools.wraps(module.chunk_gated_delta_rule_fwd_h)
    def forward_h(*args, **kwargs):
        chunk_size = kwargs.get("chunk_size", args[7] if len(args) > 7 else 64)
        if chunk_size != 64:
            raise ValueError("canonical GDN cache rounding currently requires 64-token FLA chunks")
        from .gdn_prefill_export import forward_h
        return forward_h(private, storage_dtype, args, kwargs)

    forward_h._horizon_state_round = True
    module.chunk_gated_delta_rule_fwd_h = forward_h
    STATE["patched"] = True
    _log().info("horizon_kernels GDN prefill state: %s rounding at every FLA chunk boundary; geometry from tensors",
                storage_dtype)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        patch_module(module)  # fail closed: an unpatched 16-bit cache must not masquerade as fixed


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


def register():
    """Independent general-plugin entry point; registration imports no CUDA code."""
    storage_dtype = os.environ.get("HORIZON_GDN_PREFILL_STATE_ROUND", "")
    if not storage_dtype or STATE["installed"]:
        return
    aliases = {"fp16": "float16", "float16": "float16", "bf16": "bfloat16", "bfloat16": "bfloat16"}
    if storage_dtype not in ("float16", "bfloat16") or aliases.get(os.environ.get("HORIZON_GDN_STATE")) != storage_dtype:
        raise ValueError("prefill rounding precision must match HORIZON_GDN_STATE")
    if TARGET in sys.modules:
        patch_module(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
    STATE["installed"] = True
