# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_shard_factor (fp8fix investigation, 27-09-2026; arm overlay plugin, measurement only).
# HORIZON_SHARD_FACTORS=<json> scales the fused GDN in_proj shards of a Qwen3.5-family checkpoint while vLLM loads it,
# to transplant production's scale collapse onto another checkpoint (e.g. NVIDIA's ModelOpt FP8):
#   HORIZON_SHARD_FACTOR_MODE=both  in_proj_qkv.weight_scale *= f_qkv[layer], in_proj_z.weight_scale *= f_z[layer]
#   HORIZON_SHARD_FACTOR_MODE=z     only in_proj_z (the output-gate projection)
# Factors: prod_collapse_factors.json = gs_shard / max(gs) of our NVFP4 checkpoint (what vLLM's max-collapse does to
# production). Works on per-tensor multiplier scales (ModelOpt FP8 weight_scale). Wraps Qwen3_5Model.load_weights via
# a post-import hook; no CUDA at registration.
import importlib.abc, json, os, re, sys

TARGET = "vllm.model_executor.models.qwen3_5"
PAT = re.compile(r"layers\.(\d+)\.linear_attn\.in_proj_(qkv|z)\.weight_scale$")
STATE = {"n": 0}


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def _wrap(mod):
    cls = mod.Qwen3_5Model
    if getattr(cls.load_weights, "_horizon_sf", False):
        return
    fac = json.load(open(os.environ["HORIZON_SHARD_FACTORS"]))["layers"]
    mode = os.environ.get("HORIZON_SHARD_FACTOR_MODE", "both")
    orig = cls.load_weights

    def gen(weights):
        for name, t in weights:
            m = PAT.search(name)
            if m and (mode == "both" or m.group(2) == "z"):
                f = fac[m.group(1)][m.group(2)]
                if f != 1.0:
                    t = t * f
                    STATE["n"] += 1
            yield name, t

    def load_weights(self, weights):
        r = orig(self, gen(weights))
        _log().info("horizon_shard_factor: mode=%s, %d GDN in_proj scales multiplied (%s)", mode, STATE["n"],
                    os.environ["HORIZON_SHARD_FACTORS"])
        return r
    load_weights._horizon_sf = True
    cls.load_weights = load_weights


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _wrap(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def register():
    if not os.environ.get("HORIZON_SHARD_FACTORS"):
        return
    if TARGET in sys.modules:
        _wrap(sys.modules[TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
