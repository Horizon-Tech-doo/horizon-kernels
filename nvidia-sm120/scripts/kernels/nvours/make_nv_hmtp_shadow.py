# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Draft-path option (b) for nvidia/Qwen3.8-27B-NVFP4 (27-09-2026): a SHADOW model folder = NVIDIA's checkpoint with
# its (original, BF16) MTP layer replaced by our retrained one (Qwen3.8-27B-NVFP4/horizon_mtp/model_mtp.safetensors,
# the one production serves). The original folders are never modified. CPU only.
#
#   python make_nv_hmtp_shadow.py [OUT_DIR]     (default $HK_WORK/nvours/nv-hmtp)
#
# - every NVIDIA file is symlinked, except the shard holding the mtp.* tensors (rewritten WITHOUT them, ~1.1 GB, because
#   vLLM streams every tensor of every listed shard: a duplicate mtp.* would load twice) and the index (mtp.* ->
#   model_mtp.safetensors);
# - model_mtp.safetensors -> our retrained MTP layer (same 15 tensor names / shapes / BF16 as NVIDIA's, checked).
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from safetensors import safe_open
from safetensors.torch import save_file

NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
MTP = "$HK_DATA/models/Qwen3.8-27B-NVFP4/horizon_mtp/model_mtp.safetensors"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("$HK_WORK/nvours/nv-hmtp")

idx = json.load(open(f"{NV}/model.safetensors.index.json"))
wm = idx["weight_map"]
mtp_nv = sorted(k for k in wm if k.startswith("mtp."))
shards = sorted({wm[k] for k in mtp_nv})
assert len(shards) == 1, shards
shard = shards[0]

with safe_open(MTP, "pt", device="cpu") as f:
    ours = {k: (f.get_slice(k).get_dtype(), tuple(f.get_slice(k).get_shape())) for k in f.keys()}
with safe_open(f"{NV}/{shard}", "pt", device="cpu") as f:
    theirs = {k: (f.get_slice(k).get_dtype(), tuple(f.get_slice(k).get_shape())) for k in mtp_nv}
    assert set(ours) == set(theirs), (sorted(set(ours) ^ set(theirs)))
    bad = {k: (ours[k], theirs[k]) for k in ours if ours[k] != theirs[k]}
    assert not bad, bad
    keep = {k: f.get_tensor(k) for k in f.keys() if not k.startswith("mtp.")}
    meta = f.metadata() or {}

os.makedirs(OUT, exist_ok=True)
for name in os.listdir(NV):
    if name.startswith(".") or name in (shard, "model.safetensors.index.json"):
        continue
    dst = os.path.join(OUT, name)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.join(NV, name), dst)
tmp = os.path.join(OUT, shard + ".tmp")
save_file(keep, tmp, metadata={**meta, "format": "pt"})
os.replace(tmp, os.path.join(OUT, shard))
dst = os.path.join(OUT, "model_mtp.safetensors")
if os.path.lexists(dst):
    os.remove(dst)
os.symlink(MTP, dst)
for k in mtp_nv:
    wm[k] = "model_mtp.safetensors"
json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
print(json.dumps({"out": OUT, "rewritten_shard": shard, "kept_tensors": len(keep), "mtp_tensors": len(mtp_nv),
                  "mtp_source": MTP}))
