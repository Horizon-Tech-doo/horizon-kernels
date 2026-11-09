# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Draft-path option (a) for nvidia/Qwen3.8-27B-NVFP4 (27-09-2026): a K1a draft head built from NVIDIA's own NVFP4
# lm_head, i.e. the dequantised rows of the 48k draft vocabulary, as a BF16 (V, 5120) safetensors file {"weight": ...}
# for HORIZON_DRAFT_HEAD_WEIGHTS. CPU only (no CUDA context), ~3 GB RAM.
#
#   python make_nvidia_draft_head.py [OUT]     (default $HK_WORK/nvours/draft_head_nvidia_lmhead.safetensors)
#
# Checks printed (sanity of the dequantisation, incl. the nibble order): per-row cosine similarity of the dequantised
# NVIDIA rows vs our checkpoint's BF16 lm_head rows (the original Qwen weights: both come from the same BF16 model)
# with the correct (low nibble = even k) and the swapped nibble order, and vs our trained 48k draft head.
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from safetensors import safe_open
from safetensors.torch import save_file

NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
OURS = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
VOCAB = f"{OURS}/horizon_mtp/draft_vocab.pt"
TRAINED = f"{OURS}/horizon_mtp/draft_head.safetensors"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("$HK_WORK/nvours/draft_head_nvidia_lmhead.safetensors")
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def tensor(folder, name):
    idx = json.load(open(f"{folder}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{folder}/{idx[name]}", "pt", device="cpu") as f:
        return f.get_tensor(name)


def dequant(packed_u8, scale_fp8, ws2, swap=False):
    lo, hi = (packed_u8 & 0xF).long(), (packed_u8 >> 4).long()
    if swap:
        lo, hi = hi, lo
    w = torch.empty(packed_u8.shape[0], packed_u8.shape[1] * 2, dtype=torch.float32)
    w[:, 0::2] = LUT[lo]
    w[:, 1::2] = LUT[hi]
    return w * scale_fp8.float().repeat_interleave(16, 1) * ws2


def cos_rows(a, b):
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)


vocab = torch.load(VOCAB, map_location="cpu").to(torch.long)
wp = tensor(NV, "lm_head.weight")[vocab]                       # uint8 (V, 2560)
ws = tensor(NV, "lm_head.weight_scale")[vocab]                 # fp8 e4m3 (V, 320)
ws2 = tensor(NV, "lm_head.weight_scale_2").float().item()      # multiplier (ModelOpt)
assert wp.dtype == torch.uint8 and ws.dtype == torch.float8_e4m3fn, (wp.dtype, ws.dtype)
W = dequant(wp, ws, ws2)                                        # fp32 (V, 5120)
head = W.to(torch.bfloat16).contiguous()

stats = {"vocab": int(vocab.numel()), "shape": list(head.shape), "weight_scale_2": ws2}
try:
    ref_name = "lm_head.weight"
    try:
        ref = tensor(OURS, ref_name)
    except (KeyError, FileNotFoundError):
        with safe_open(f"{OURS}/model.safetensors", "pt", device="cpu") as f:
            ref = f.get_tensor(ref_name)
    ref = ref[vocab].float()
    c = cos_rows(W, ref)
    cs = cos_rows(dequant(wp, ws, ws2, swap=True), ref)
    rel = ((W - ref).norm() / ref.norm()).item()
    stats.update({"cos_vs_ours_bf16_lm_head_mean": round(c.mean().item(), 5), "cos_min": round(c.min().item(), 5),
                  "rel_err_vs_ours_bf16_lm_head": round(rel, 5),
                  "cos_swapped_nibbles_mean": round(cs.mean().item(), 5)})
    del ref
except Exception as ex:  # noqa: BLE001
    stats["ref_check"] = f"skipped: {type(ex).__name__}: {ex}"
try:
    with safe_open(TRAINED, "pt", device="cpu") as f:
        tr = f.get_tensor("weight").float()
    c = cos_rows(W, tr)
    stats.update({"cos_vs_our_trained_head_mean": round(c.mean().item(), 5), "cos_vs_trained_min": round(c.min().item(), 5)})
except Exception as ex:  # noqa: BLE001
    stats["trained_check"] = f"skipped: {type(ex).__name__}: {ex}"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({"weight": head}, OUT, metadata={"format": "pt", "source": f"{NV} lm_head (NVFP4) dequantised rows of {VOCAB}"})
stats["out"] = OUT
print(json.dumps(stats))
