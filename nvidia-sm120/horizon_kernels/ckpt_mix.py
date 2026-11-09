# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels checkpoint mixing (formats track, 27-09-2026): HORIZON_MIX_MLP_FROM=<ModelOpt NVFP4 checkpoint dir>
# (default unset = off).
#
# Serves OUR compressed-tensors NVFP4 checkpoint with the MLP layers (gate/up/down of every decoder layer) taken from
# another NVFP4 quantisation of the same BF16 model in NVIDIA ModelOpt format (nvidia/Qwen3.8-27B-NVFP4). Both store
# the same thing per layer: e2m1 codes uint8 (N, K/2) (low nibble = even k) and e4m3 block scales (N, K/16); they differ
# in the global scale's direction (compressed-tensors weight_global_scale is a divisor, ModelOpt weight_scale_2 the
# multiplier) and the input scale (ignored by our kernels: W4A16 decode, dynamic FP4 prefill). The conversion is
# therefore lossless: weight_packed <- weight, weight_scale <- weight_scale, weight_global_scale <- 1 / weight_scale_2,
# input_global_scale <- 1 / input_scale. Shapes, formats and every kernel stay exactly production's (same speed); only
# the MLP weights' values change. Question it answers: does NVIDIA's quality gain (27-09-2026 experiment 3: MBPP 82.8
# vs 76.6, MATH-500 96.8 vs 94.6) come from its MLP quantisation, which we can then take without its slower FP8
# attention / GDN layers?
# Hook: vLLM's safetensors weight iterators (checkpoint tensor names, before vLLM's own name mapping). Nothing on disk
# is written.
import json
import os
import re

import torch

_PAT = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.(?:gate|up|down)_proj)\."
                  r"(weight_packed|weight_scale|weight_global_scale|input_global_scale)$")
_STATS = {"replaced": 0}


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


class _Source:
    def __init__(self, root):
        self.root = root
        self.wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open
        fn = self.wm[name]
        h = self.handles.get(fn)
        if h is None:
            h = self.handles[fn] = safe_open(os.path.join(self.root, fn), "pt", device="cpu")
        return h.get_tensor(name)


def convert(src, base, kind, like):
    if kind == "weight_packed":
        t = src.get(base + ".weight")
    elif kind == "weight_scale":
        t = src.get(base + ".weight_scale")
    elif kind == "weight_global_scale":
        t = (1.0 / src.get(base + ".weight_scale_2").float()).reshape(like.shape)
    else:
        t = (1.0 / src.get(base + ".input_scale").float()).reshape(like.shape)
    assert tuple(t.shape) == tuple(like.shape) and t.dtype == like.dtype, (base, kind, t.shape, t.dtype, like.shape,
                                                                          like.dtype)
    return t


def _wrap(it_fn, src):
    def wrapped(*a, **k):
        for name, t in it_fn(*a, **k):
            m = _PAT.match(name)
            if m:
                t = convert(src, m.group(1), m.group(2), t)
                _STATS["replaced"] += 1
            yield name, t
        _log().info("horizon_kernels: checkpoint mix: %d MLP tensors taken from %s", _STATS["replaced"], src.root)
    return wrapped


def install():
    install_nvfp4_mix()
    root = os.environ.get("HORIZON_MIX_MLP_FROM", "")
    if not root:
        return
    from vllm.model_executor.model_loader import default_loader as dl
    if getattr(dl, "_horizon_mix", False):
        return
    src = _Source(root)
    for n in ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator"):
        if hasattr(dl, n):
            setattr(dl, n, _wrap(getattr(dl, n), src))
    dl._horizon_mix = True
    _log().info("horizon_kernels: checkpoint mix active: MLP layers from %s", root)


# ---------------------------------------------------------------------------------------------------------------
# Mode 2 (27-09-2026): HORIZON_MIX_NVFP4_FROM_CT=<our compressed-tensors NVFP4 checkpoint> + HORIZON_MIX_NVFP4_MODULES=
# gdn|attn: serve a ModelOpt checkpoint (NVIDIA's, FP8 attention / GDN) with the chosen FP8 projection layers replaced
# by OUR NVFP4 ones, converted to ModelOpt NVFP4 at load (weight <- weight_packed, weight_scale <- weight_scale,
# weight_scale_2 <- 1 / gs, input_scale <- 1 / input_global_scale). The model directory must be a shadow whose
# config.json / hf_quant_config.json declare those layers NVFP4 (scripts/kernels/formats/make_mix_shadow.py).
# Fused shards (in_proj_qkv + in_proj_z, q/k/v) get ONE weight_scale_2 = 1 / max(gs over the shards): vLLM collapses a
# ModelOpt layer to max(weight_scale_2), and this makes that equal to production's compressed-tensors collapse
# (alpha = 1 / max(gs)), so the swapped layers compute exactly what they compute in production.
_GROUPS = {"gdn": [("linear_attn", ["in_proj_qkv", "in_proj_z"]), ("linear_attn", ["out_proj"])],
           "attn": [("self_attn", ["q_proj", "k_proj", "v_proj"]), ("self_attn", ["o_proj"])]}


class _CTSource(_Source):
    def __init__(self, root):
        idx = os.path.join(root, "model.safetensors.index.json")
        self.root = root
        self.handles = {}
        if os.path.exists(idx):
            self.wm = json.load(open(idx))["weight_map"]
        else:
            from safetensors import safe_open
            with safe_open(os.path.join(root, "model.safetensors"), "pt") as f:
                self.wm = {k: "model.safetensors" for k in f.keys()}


def _nvfp4_targets(kinds):
    """{checkpoint module name regex -> (block, shards)} for the chosen module kinds."""
    out = []
    for kind in kinds:
        for block, shards in _GROUPS[kind]:
            out.append((re.compile(r"^(model\.language_model\.layers\.(\d+))\.%s\.(%s)\.(weight|weight_scale|input_scale)$"
                                   % (block, "|".join(shards))), block, shards))
    return out


def _wrap_nvfp4(it_fn, src, targets):
    def conv(layer_prefix, block, shards, shard, kind):
        base = f"{layer_prefix}.{block}.{shard}"
        if kind == "weight":
            return src.get(base + ".weight_packed")
        if kind == "weight_scale":
            return src.get(base + ".weight_scale")
        if kind == "input_scale":
            return (1.0 / src.get(base + ".input_global_scale").float()).reshape(())
        gmax = max(float(src.get(f"{layer_prefix}.{block}.{s}.weight_global_scale").float().max()) for s in shards)
        return torch.tensor(1.0 / gmax, dtype=torch.float32)          # weight_scale_2

    def wrapped(*a, **k):
        n = 0
        for name, t in it_fn(*a, **k):
            for pat, block, shards in targets:
                m = pat.match(name)
                if m:
                    layer_prefix, shard, kind = m.group(1), m.group(3), m.group(4)
                    t = conv(layer_prefix, block, shards, shard, kind)
                    n += 1
                    if kind == "weight":
                        yield name[:-len("weight")] + "weight_scale_2", conv(layer_prefix, block, shards, shard, "ws2")
                    break
            yield name, t
        if n:
            _log().info("horizon_kernels: checkpoint mix: %d FP8 tensors replaced by NVFP4 ones from %s", n, src.root)
    return wrapped


def install_nvfp4_mix():
    root = os.environ.get("HORIZON_MIX_NVFP4_FROM_CT", "")
    if not root:
        return
    kinds = [k for k in os.environ.get("HORIZON_MIX_NVFP4_MODULES", "gdn").split(",") if k]
    from vllm.model_executor.model_loader import default_loader as dl
    if getattr(dl, "_horizon_mix_nvfp4", False):
        return
    src = _CTSource(root)
    targets = _nvfp4_targets(kinds)
    for n in ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator"):
        if hasattr(dl, n):
            setattr(dl, n, _wrap_nvfp4(getattr(dl, n), src, targets))
    dl._horizon_mix_nvfp4 = True
    _log().info("horizon_kernels: checkpoint mix active: %s projections as NVFP4 from %s", kinds, root)
