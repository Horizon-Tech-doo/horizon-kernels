# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# horizon_kernels quantized LM head (formats track, 27-09-2026). Both switches default off.
#
# HORIZON_LMHEAD_QUANT=nvfp4|fp8 + HORIZON_LMHEAD_FILE=<safetensors>: the LM head (whatever format the checkpoint stores
#   it in: ours BF16, 2.54 GB read per verify step; NVIDIA's ModelOpt NVFP4) is served from a quantized copy:
#     nvfp4  weight uint8 (V, H/2) e2m1 (low nibble = even k), weight_scale e4m3 (V, H/16), weight_scale_2 fp32 (a
#            multiplier, scalar or (V,)): ModelOpt naming, e.g. NVIDIA's own calibrated lm_head of the same BF16 model.
#            Runs on decode_regfrag_xl (exact class for the stored weights; 0.72 GB per read).
#     fp8    weight e4m3 (V, H), weight_scale fp32 (V,) per row. Runs on tilelang.contrib.sm120_fp8.decode_w8a16.
#   The checkpoint is never modified. HORIZON_LMHEAD_FREE_BF16=1 also frees the BF16 weight (default 0: kept, so the
#   K1a draft-head build and any stock path keep working; memory is then +0.7 / +1.3 GB instead of -1.8 / -1.2 GB).
#   The quantized head changes the logits: it must pass the quality gate (kernel-master-plan-27-09-2026 (not in this release) S3).
# HORIZON_DUMP_LMHEAD_H=<prefix>: record the LM head's input rows (the model's final hidden states) to
#   <prefix>_<n>.pt (bf16, HORIZON_DUMP_LMHEAD_EVERY-th row, default every 4th, up to HORIZON_DUMP_LMHEAD_MAX rows,
#   default 65536) for the offline logit gate (scripts/kernels/formats/lmhead_logit_gate.py). Output unchanged.
import os

import torch

_HEADS: list = []
_DUMP = {"rows": [], "n": 0, "saved": 0, "files": 0}
XL_MAX_M = 1024


def _log():
    from vllm.logger import init_logger
    return init_logger("vllm.horizon_kernels")


def enabled() -> bool:
    return bool(os.environ.get("HORIZON_LMHEAD_QUANT") or os.environ.get("HORIZON_DUMP_LMHEAD_H"))


def load_quant_file(path, mode, device):
    """-> ("nvfp4", decode_regfrag_xl prepared) or ("fp8", decode_w8a16 prepared)."""
    from safetensors.torch import load_file
    t = load_file(path, device=str(device))
    if mode == "nvfp4":
        from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
        wp, ws = t["weight"], t["weight_scale"]
        a = t["weight_scale_2"].float().reshape(-1)
        V = wp.shape[0]
        alpha = (a.expand(V) if a.numel() == 1 else a).contiguous()
        return "nvfp4", xl.prepare(wp.view(torch.int8), ws, alpha)
    if mode == "fp8":
        from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
        return "fp8", w8.prepare(t["weight"].contiguous(), t["weight_scale"].float().reshape(-1))
    raise ValueError(f"HORIZON_LMHEAD_QUANT={mode!r}: expected nvfp4 or fp8")


def quant_logits(kind, prep, x2):
    if kind == "nvfp4":
        from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
        if x2.shape[0] <= XL_MAX_M:
            return xl.linear(x2, prep)
        return torch.cat([xl.linear(x2[i:i + XL_MAX_M], prep) for i in range(0, x2.shape[0], XL_MAX_M)])
    from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
    return w8.linear(x2, prep)


def _configuration_boundary_rows(configuration_key, max_rows):
    """Cover every dynamic-row kernel at its smallest/largest selected row count.

    Only valid for kernels whose row count is dynamic and absent from the
    compilation key. Keep a full warm-up for paths without that contract.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    bounds = {}
    for rows in range(1, max_rows + 1):
        key = configuration_key(rows)
        if key not in bounds:
            bounds[key] = [rows, rows]
        bounds[key][1] = rows
    return sorted({rows for pair in bounds.values() for rows in pair})


def _head_warmup_rows(kind, prep):
    if kind != "fp8":
        return range(1, XL_MAX_M + 1)
    from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8

    def key(rows):
        # Match decode_w8a16.linear's cache key exactly (N/K are fixed here).
        cfg = w8.config(prep.N, prep.K, rows)
        return tuple(cfg[:7]) + (cfg[7] if len(cfg) > 7 else 1,)

    return _configuration_boundary_rows(key, XL_MAX_M)


def _dump(x2):
    every = int(os.environ.get("HORIZON_DUMP_LMHEAD_EVERY", "4"))
    cap = int(os.environ.get("HORIZON_DUMP_LMHEAD_MAX", "65536"))
    if _DUMP["saved"] >= cap or torch.cuda.is_current_stream_capturing():
        return
    n0 = _DUMP["n"]
    idx = [i for i in range(x2.shape[0]) if (n0 + i) % every == 0]
    _DUMP["n"] += x2.shape[0]
    if idx:
        _DUMP["rows"].append(x2[idx].detach().to("cpu", torch.bfloat16))
        _DUMP["saved"] += len(idx)
    if sum(r.shape[0] for r in _DUMP["rows"]) >= 4096 or _DUMP["saved"] >= cap:
        p = f"{os.environ['HORIZON_DUMP_LMHEAD_H']}_{_DUMP['files']:03d}.pt"
        torch.save(torch.cat(_DUMP["rows"]), p)
        _DUMP["rows"], _DUMP["files"] = [], _DUMP["files"] + 1


def _attach(lm_head, device):
    """Load the quantized copy for this LM head (any stored format: BF16, ModelOpt NVFP4, ...) and warm its kernels."""
    mode = os.environ.get("HORIZON_LMHEAD_QUANT", "")
    path = os.environ.get("HORIZON_LMHEAD_FILE", "")
    if not path:
        raise RuntimeError("horizon_kernels: HORIZON_LMHEAD_QUANT needs HORIZON_LMHEAD_FILE")
    kind, prep = load_quant_file(path, mode, device)
    V = getattr(lm_head, "num_embeddings_per_partition", None) or lm_head.weight.shape[0]
    H = getattr(lm_head, "embedding_dim", None) or prep.K
    if prep.N != V or prep.K != H:
        raise RuntimeError(f"horizon_kernels: quantized lm_head {prep.N}x{prep.K} != model head {V}x{H}")
    warmup_rows = _head_warmup_rows(kind, prep)
    for m in warmup_rows:
        quant_logits(kind, prep, torch.zeros(m, prep.K, device=device, dtype=torch.bfloat16))
    torch.cuda.synchronize()
    _log().info("horizon_kernels: output projection warm-up covered %d row counts (%s)",
                len(warmup_rows), kind)
    w = getattr(lm_head, "weight", None)
    freed = False
    if (os.environ.get("HORIZON_LMHEAD_FREE_BF16", "0") == "1" and w is not None and w.is_floating_point()
            and w.dtype == torch.bfloat16):
        lm_head.weight.data = torch.empty(0, dtype=w.dtype, device=w.device)
        freed = True
    lm_head.horizon_lmq = (kind, prep)
    _HEADS.append(lm_head)
    _log().info("horizon_kernels: quantized LM head active (%s, %s, %dx%d, head stored as %s, BF16 weight %s)", kind,
                path, prep.N, prep.K, type(getattr(lm_head, "quant_method", None)).__name__,
                "freed" if freed else "kept")


def install():
    """Hook LogitsProcessor._apply_head (every LM head format goes through it; logits are computed eagerly, outside
    CUDA graphs). The quantized copy is attached on the first call (vLLM's warm-up forward, before serving)."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    if getattr(LogitsProcessor._apply_head, "_horizon_lmhead", False):
        return
    orig = LogitsProcessor._apply_head
    mode = os.environ.get("HORIZON_LMHEAD_QUANT", "")

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        if os.environ.get("HORIZON_DUMP_LMHEAD_H"):
            _dump(hidden_states.reshape(-1, hidden_states.shape[-1]))
        if mode and embedding_bias is None and getattr(lm_head, "tp_size", 1) == 1 \
                and hidden_states.dtype == torch.bfloat16:
            q = getattr(lm_head, "horizon_lmq", None)
            if q is None and not getattr(lm_head, "_horizon_lmq_tried", False) \
                    and not torch.cuda.is_current_stream_capturing():
                lm_head._horizon_lmq_tried = True
                _attach(lm_head, hidden_states.device)
                q = lm_head.horizon_lmq
            if q is not None:
                x2 = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
                return quant_logits(q[0], q[1], x2).reshape(*hidden_states.shape[:-1], q[1].N)
        return orig(self, lm_head, hidden_states, embedding_bias)

    _apply_head._horizon_lmhead = True
    LogitsProcessor._apply_head = _apply_head
