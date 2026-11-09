# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# MoE step 6 (27-09-2026): a K1a draft head (48k draft vocabulary, BF16 (V, 2048) {"weight": ...} safetensors for
# HORIZON_DRAFT_HEAD_WEIGHTS) for nvidia/Qwen3.6-35B-A3B-NVFP4, built from the model's own NVFP4 lm_head: the exactly
# dequantised rows (e2m1 low nibble = even k, x e4m3 block scale, x weight_scale_2) of production's draft vocabulary.
# CPU only (no CUDA context). Checks printed: the Qwen3.6 and Qwen3.8 vocabularies agree on the draft ids (the draft
# vocabulary was built from Qwen3.8's own outputs), and each dequantised row is finite and non-zero.
#   make_moe_draft_head.py [OUT]   (default $HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4-horizon/draft_head.safetensors)
import json, os, sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
from safetensors import safe_open
from safetensors.torch import save_file

MOE = "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4"
OURS = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
VOCAB = f"{OURS}/horizon_mtp/draft_vocab.pt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "$HK_DATA/models/nvidia-Qwen3.6-35B-A3B-NVFP4-horizon/draft_head.safetensors"
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
idx = json.load(open(f"{MOE}/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    with safe_open(f"{MOE}/{idx[name]}", "pt", device="cpu") as f:
        return f.get_tensor(name)


keys = [k for k in idx if k.startswith("lm_head.") or k.endswith(".lm_head.weight")]
print("lm_head tensors:", keys)
w = tensor("lm_head.weight").view(torch.uint8)
s = tensor("lm_head.weight_scale")
s2 = tensor("lm_head.weight_scale_2").float().reshape(())
print("lm_head", tuple(w.shape), w.dtype, tuple(s.shape), s.dtype, "weight_scale_2", float(s2))
vocab = torch.load(VOCAB, map_location="cpu").long()
print("draft vocab", vocab.numel(), "max id", int(vocab.max()))
from transformers import AutoTokenizer
tq = AutoTokenizer.from_pretrained(OURS); tm = AutoTokenizer.from_pretrained(MOE)
vq, vm = tq.get_vocab(), tm.get_vocab()
inv_q = {i: t for t, i in vq.items()}; inv_m = {i: t for t, i in vm.items()}
same = sum(1 for i in vocab.tolist() if inv_q.get(i) is not None and inv_q.get(i) == inv_m.get(i))
print(f"draft ids with the same token string in both vocabularies: {same}/{vocab.numel()} "
      f"(full vocab sizes {len(vq)} / {len(vm)}, identical maps: {vq == vm})")
rows = w[vocab]
lo, hi = (rows & 0xF).long(), (rows >> 4).long()
deq = torch.empty(rows.shape[0], rows.shape[1] * 2, dtype=torch.float32)
deq[:, 0::2] = LUT[lo]; deq[:, 1::2] = LUT[hi]
deq = deq * s[vocab].float().repeat_interleave(16, 1) * s2
print("finite", bool(torch.isfinite(deq).all()), "zero rows", int((deq.abs().sum(1) == 0).sum()),
      "row norm mean", float(deq.norm(dim=1).mean()))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file({"weight": deq.to(torch.bfloat16).contiguous()}, OUT)
print("wrote", OUT, tuple(deq.shape))
