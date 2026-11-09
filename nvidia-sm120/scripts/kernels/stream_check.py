# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Does the TileLang kernel run on torch's current stream and replay correctly inside a CUDA graph?
import torch, importlib.util
from safetensors import safe_open
from horizon_kernels.draft_head import DraftHeadFP8
print("torch_c_dlpack_ext installed:", importlib.util.find_spec("torch_c_dlpack_ext") is not None)
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    W = f.get_tensor("lm_head.weight")
for act in ("fp8", "bf16"):
    head = DraftHeadFP8(W, act=act)
    h = torch.zeros(4, W.shape[1], device="cuda", dtype=torch.bfloat16)
    ids1 = torch.tensor([11, 222, 3333, 44444], device="cuda"); ids2 = torch.tensor([55555, 6666, 777, 88], device="cuda")
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        h.copy_(W[ids1] * 4); head(h); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            out = head(h)
    for ids in (ids1, ids2, ids1):
        h.copy_(W[ids] * 4); g.replay(); torch.cuda.synchronize()
        ref = (h @ W.T).argmax(-1)
        print(act, "replay expects", ref.tolist(), "got", out.tolist(), "OK" if torch.equal(ref, out) else "STALE/WRONG")
