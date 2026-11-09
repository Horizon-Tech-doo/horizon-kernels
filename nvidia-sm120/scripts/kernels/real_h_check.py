# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K1a v1 vs v2 on REAL draft hidden states captured from vLLM (HORIZON_DUMP_DRAFT_H).
import torch, json
from safetensors import safe_open
from horizon_kernels.draft_head import DraftHeadFP8
H = torch.load("$HK_WORK/draft_h.pt").cuda()
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    W = f.get_tensor("lm_head.weight")
hf = H.float()
rms = hf.pow(2).mean(-1).sqrt(); peak = hf.abs().amax(-1)
chmax = hf.abs().amax(0); top = chmax.topk(8)
print(json.dumps({"rows": H.shape[0], "median_peak_over_rms": round((peak / rms).median().item(), 1),
                  "top8_channel_absmax": [round(v, 1) for v in top.values.tolist()], "median_channel_absmax": round(chmax.median().item(), 2)}))
Wf = W.float()
logits = torch.cat([H[i:i + 256].float() @ Wf.T for i in range(0, H.shape[0], 256)])  # fp32 reference (review #29)
logits_bf16 = torch.cat([(H[i:i + 256] @ W.T).float() for i in range(0, H.shape[0], 256)])
t2 = logits.topk(2, -1); ref = t2.indices[:, 0]; gap = t2.values[:, 0] - t2.values[:, 1]
bf_ref = logits_bf16.argmax(-1)
print(json.dumps({"fp32_vs_bf16_reference_argmax_agree_pct": round(100 * (bf_ref == logits.argmax(-1)).float().mean().item(), 2)}))
print(json.dumps({"top2_gap_median": round(gap.median().item(), 3), "pct_gap_below_0.25": round(100 * (gap < 0.25).float().mean().item(), 1)}))
for act in ("fp8", "bf16"):
    head = DraftHeadFP8(W, act=act)
    got = torch.cat([head(H[i:i + 16]) for i in range(0, H.shape[0], 16)])
    dis = got != ref
    print(json.dumps({"act": act, "agree_pct": round(100 * (~dis).float().mean().item(), 2),
                      "disagreements": int(dis.sum()), "median_gap_where_disagree": round(gap[dis].median().item(), 4) if dis.any() else None, "max_gap_where_disagree": round(gap[dis].max().item(), 4) if dis.any() else None}))
