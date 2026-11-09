# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""In-worker hooks on vLLM's MoE runners (vLLM 0.30 API, read from the installed source on 03-10-2026):

  MoERunner._forward_impl -> (gate) -> _apply_quant_method(hidden_states, router_logits, shared_experts_input,
                                                           input_ids=None, shared_experts_overlapping=False)
      -> self.router.select_experts(hidden_states=, router_logits=, topk_indices_dtype=, input_ids=)
      -> self.routed_experts.forward_modular(x=, topk_weights=, topk_ids=, shared_experts=, shared_experts_input=)

Both features wrap `_apply_quant_method`, which runs inside the opaque `vllm::moe_forward` custom op, so the Python
wrapper executes on every eager forward. Under CUDA graphs it executes only at capture time: calibration therefore
REQUIRES enforce_eager (checked), while the mask works with graphs (its ops are captured; its tensors are persistent).

Calibration (REAP, Lasby et al. 2025, arXiv:2510.13999; reference code github.com/CerebrasResearch/reap
@1970473c, src/reap/pruning_metrics.py: reap[e] = mean over tokens routed to e of ||f_e(x)|| * g_e(x), with the
top-k gate values renormalized): for every MoE call we take vLLM's own routing (select_experts: softmax over the 512
routed experts, top-10, renormalized for Qwen4Exp since norm_topk_prob=True) and evaluate each routed expert ALONE
with vLLM's own expert kernel (forward_modular with one slot, weight 1.0), so ||f_e(x)|| comes from exactly the
numerics that will serve (Marlin W4A16 on our SM120 box). Accumulators live on the GPU; only a small token sample
is copied out for the offline cross-check (reap_calibrate.py verify).

Runtime mask: router logits of masked experts set to -inf before routing; used to screen a pruning (incl. a
non-uniform one that the checkpoint format cannot express) for QUALITY without rewriting weights. It saves no memory.
"""
from __future__ import annotations

import json
import os
import re

import torch

_STATE: dict = {"calib": None, "mask": None, "installed": {}}


# ---------------------------------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------------------------------

def is_runner(m) -> bool:
    return (hasattr(m, "router") and hasattr(m, "routed_experts") and hasattr(m, "_apply_quant_method")
            and isinstance(getattr(m, "layer_name", None), str))


def worker_roots(worker):
    """The target model and, when speculative decoding is on, the drafter model (MTP layer)."""
    roots = []
    mr = getattr(worker, "model_runner", None)
    for obj in (mr, getattr(mr, "drafter", None), getattr(mr, "proposer", None)):
        if obj is None:
            continue
        for attr in ("model",):
            m = getattr(obj, attr, None)
            if isinstance(m, torch.nn.Module):
                roots.append(m)
        if hasattr(obj, "get_model"):
            try:
                m = obj.get_model()
                if isinstance(m, torch.nn.Module):
                    roots.append(m)
            except Exception:  # noqa: BLE001
                pass
    return roots


def find_runners(roots):
    seen, out = set(), []
    for root in roots:
        for _, m in root.named_modules():
            if id(m) not in seen and is_runner(m):
                seen.add(id(m))
                out.append(m)
    return out


def num_hidden_layers(worker) -> int:
    cfg = worker.vllm_config.model_config.hf_text_config
    return int(cfg.num_hidden_layers)


def ckpt_prefix(layer_name: str, nhl: int) -> str:
    m = re.search(r"layers\.(\d+)\.", layer_name + ".")
    if not m:
        return layer_name
    L = int(m.group(1))
    if L >= nhl or re.search(r"(^|\.)mtp(\.|$)", layer_name):
        return f"mtp.layers.{L - nhl if L >= nhl else L}.mlp"
    return f"model.language_model.layers.{L}.mlp"


# ---------------------------------------------------------------------------------------------------------------------
# the wrapper (one per runner instance, or the class-level one from the plugin)
# ---------------------------------------------------------------------------------------------------------------------

def _split(args, kwargs):
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else (args[0] if args else None)
    rl = kwargs["router_logits"] if "router_logits" in kwargs else (args[1] if len(args) > 1 else None)
    return hs, rl


def _replace_logits(args, kwargs, rl):
    if "router_logits" in kwargs:
        kwargs = dict(kwargs)
        kwargs["router_logits"] = rl
        return args, kwargs
    return (args[0], rl) + tuple(args[2:]), kwargs


def _apply_mask(runner, rl):
    mask = _STATE["mask"]
    if mask is None or rl is None:
        return rl
    key = getattr(runner, "_hz_fn_prefix", None) or runner.layer_name
    keep = mask["dev"].get(key)
    if keep is None:
        host = mask["host"].get(key)
        if host is None:
            return rl
        keep = torch.as_tensor(host, dtype=torch.bool, device=rl.device)
        mask["dev"][key] = keep
    E = keep.shape[0]
    if rl.shape[-1] < E:
        raise RuntimeError(f"expert mask for {key} has {E} entries, router logits have {rl.shape[-1]}")
    out = rl.clone()
    out[..., :E] = out[..., :E].masked_fill(~keep, float("-inf"))
    return out


def wrapped_apply(runner, orig, args, kwargs):
    hs, rl = _split(args, kwargs)
    rl2 = _apply_mask(runner, rl)
    if rl2 is not rl:
        args, kwargs = _replace_logits(args, kwargs, rl2)
    cal = _STATE["calib"]
    if cal is not None and cal.active and hs is not None and rl2 is not None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            cal.errors.append(f"{runner.layer_name}: called during CUDA graph capture (calibration needs enforce_eager)")
        else:
            try:
                cal.observe(runner, hs, rl2, kwargs.get("input_ids"))
            except Exception as e:  # noqa: BLE001
                cal.errors.append(f"{runner.layer_name}: {type(e).__name__}: {e}")
                cal.active = False
    return orig(*args, **kwargs)


def install_instance_wrappers(runners, nhl):
    n = 0
    for r in runners:
        if getattr(r, "_hz_fn_wrapped", False):
            continue
        orig = r._apply_quant_method
        r._hz_fn_prefix = ckpt_prefix(r.layer_name, nhl)

        def w(*args, _r=r, _o=orig, **kwargs):
            return wrapped_apply(_r, _o, args, kwargs)
        r._apply_quant_method = w
        r._hz_fn_wrapped = True
        n += 1
    return n


def install_class_wrapper():
    """Plugin path: wrap MoERunner._apply_quant_method for every runner ever built (mask only)."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner  # noqa: PLC0415
    if getattr(MoERunner, "_hz_fn_class_wrapped", False):
        return
    orig = MoERunner._apply_quant_method

    def w(self, *args, **kwargs):
        if getattr(self, "_hz_fn_wrapped", False):          # an instance wrapper already handles it
            return orig(self, *args, **kwargs)
        if not hasattr(self, "_hz_fn_prefix"):
            try:
                from vllm.config import get_current_vllm_config  # noqa: PLC0415
                nhl = int(get_current_vllm_config().model_config.hf_text_config.num_hidden_layers)
            except Exception:  # noqa: BLE001
                nhl = int(_STATE["mask"]["meta"].get("num_hidden_layers", 48)) if _STATE["mask"] else 48
            self._hz_fn_prefix = ckpt_prefix(self.layer_name, nhl)
        return wrapped_apply(self, lambda *a, **k: orig(self, *a, **k), args, kwargs)
    MoERunner._apply_quant_method = w
    MoERunner._hz_fn_class_wrapped = True


# ---------------------------------------------------------------------------------------------------------------------
# runtime mask
# ---------------------------------------------------------------------------------------------------------------------

def load_mask(path: str, variant: str = "uniform"):
    import numpy as np  # noqa: PLC0415
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    host = {}
    for k in z.files:
        if k.endswith("|" + variant):
            host[k.rsplit("|", 1)[0]] = z[k].astype(bool)
    if not host:
        raise ValueError(f"{path}: no '|{variant}' masks")
    _STATE["mask"] = {"host": host, "dev": {}, "meta": meta, "path": path, "variant": variant}
    return len(host)


# ---------------------------------------------------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------------------------------------------------

class Calibration:
    def __init__(self, num_experts, score_mode="kernel", sample_tokens=256, sample_dir=None):
        self.E = int(num_experts)
        self.score_mode = score_mode
        self.sample_tokens = int(sample_tokens)
        self.sample_dir = sample_dir
        self.active = False
        self.errors: list = []
        self.acc: dict = {}
        self.samples: dict = {}
        self.calls = 0

    def _acc(self, key, device):
        a = self.acc.get(key)
        if a is None:
            z = lambda: torch.zeros(self.E, dtype=torch.float64, device=device)  # noqa: E731
            a = {"count": z(), "gsum": z(), "reap_sum": z(), "ean_sum": z(), "tokens": 0, "calls": 0}
            self.acc[key] = a
        return a

    @torch.no_grad()
    def observe(self, runner, hs, rl, input_ids):
        key = runner._hz_fn_prefix
        qm = runner.routed_experts.quant_method
        topk_w, topk_ids = runner.router.select_experts(hidden_states=hs, router_logits=rl,
                                                        topk_indices_dtype=qm.topk_indices_dtype, input_ids=input_ids)
        ids = topk_ids.long()
        w = topk_w.float()
        T, K = ids.shape
        valid = ids < self.E                                   # a fused shared expert would sit at id E
        norms = torch.zeros((T, K), dtype=torch.float32, device=hs.device)
        if self.score_mode == "kernel":
            ones = torch.ones((T, 1), dtype=topk_w.dtype, device=hs.device)
            for s in range(K):
                col = valid[:, s]
                if not bool(col.any()):
                    continue
                ids_s = torch.where(col, topk_ids[:, s], torch.zeros_like(topk_ids[:, s])).unsqueeze(1).contiguous()
                out = runner.routed_experts.forward_modular(x=hs.clone(), topk_weights=ones, topk_ids=ids_s,
                                                            shared_experts=None, shared_experts_input=None)
                if isinstance(out, tuple):
                    out = out[-1]
                if not isinstance(out, torch.Tensor) or out.shape[0] != T:
                    raise RuntimeError(f"forward_modular returned {type(out).__name__}; use --score-mode routing")
                norms[:, s] = out.float().norm(dim=-1)
        a = self._acc(key, hs.device)
        iv, wv, nv = ids[valid], w[valid].double(), norms[valid].double()
        a["count"].index_add_(0, iv, torch.ones_like(wv))
        a["gsum"].index_add_(0, iv, wv)
        a["reap_sum"].index_add_(0, iv, wv * nv)
        a["ean_sum"].index_add_(0, iv, nv)
        a["tokens"] += T
        a["calls"] += 1
        self.calls += 1
        if self.sample_dir and self.sample_tokens > 0:
            have = sum(x["hidden"].shape[0] for x in self.samples.get(key, []))
            take = min(T, self.sample_tokens - have)
            if take > 0:
                self.samples.setdefault(key, []).append({
                    "hidden": hs[:take].detach().to(torch.bfloat16).view(torch.int16).cpu(),
                    "ids": ids[:take].int().cpu(), "weights": w[:take].cpu(), "norms": norms[:take].cpu(),
                    "logits": rl[:take, : self.E].float().cpu()})

    def results(self):
        import numpy as np  # noqa: PLC0415
        out = {}
        for k, a in self.acc.items():
            out[k] = {s: a[s].cpu().numpy() for s in ("count", "gsum", "reap_sum", "ean_sum")}
            out[k]["tokens"] = a["tokens"]
            out[k]["calls"] = a["calls"]
        if self.sample_dir:
            os.makedirs(self.sample_dir, exist_ok=True)
            for k, parts in self.samples.items():
                cat = {f: torch.cat([p[f] for p in parts]).numpy() for f in parts[0]}
                np.savez(os.path.join(self.sample_dir, k.replace("/", "_") + ".npz"), **cat)
        return out


# ---- functions called through LLM.collective_rpc (first argument: the worker) -------------------------------------

def rpc_install_calibration(worker, score_mode="kernel", sample_tokens=256, sample_dir=None):
    cfg = worker.vllm_config
    if not cfg.model_config.enforce_eager:
        raise RuntimeError("calibration hooks need enforce_eager=True (CUDA graphs would skip the Python wrapper)")
    nhl = num_hidden_layers(worker)
    E = int(cfg.model_config.hf_text_config.num_experts)
    runners = find_runners(worker_roots(worker))
    if not runners:
        raise RuntimeError("no MoE runners found (vLLM API changed?)")
    install_instance_wrappers(runners, nhl)
    _STATE["calib"] = Calibration(E, score_mode, sample_tokens, sample_dir)
    return sorted({r._hz_fn_prefix: r.layer_name for r in runners}.items())


def rpc_set_active(worker, active=True):
    _STATE["calib"].active = bool(active)
    return True


def rpc_status(worker):
    c = _STATE["calib"]
    return {"active": c.active, "calls": c.calls, "errors": c.errors[:20], "layers": len(c.acc)}


def rpc_results(worker):
    return _STATE["calib"].results()


def rpc_install_mask(worker, path, variant="uniform"):
    n = load_mask(path, variant)
    runners = find_runners(worker_roots(worker))
    install_instance_wrappers(runners, num_hidden_layers(worker))
    return {"masks": n, "runners": len(runners)}
