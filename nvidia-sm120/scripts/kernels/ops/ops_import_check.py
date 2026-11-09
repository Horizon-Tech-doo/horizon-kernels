# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("HORIZON_GDN", "HORIZON_ATTN", "HORIZON_PREFILL_RS", "HORIZON_GDN_CONV", "HORIZON_QKNORM"):
    os.environ[k] = "1"
import torch
torch.cuda.is_available = lambda: False
sys.path.insert(0, "$HK_WORK/ops/hpkgs3")
from horizon_kernels import ops
ops.install()
import torch
print("custom op:", torch.ops.horizon.qk_norm_rope_gate)
from vllm.model_executor.models import qwen3_next as qn
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
from vllm import _custom_ops as co
from vllm.v1.attention.backends import flashinfer as fi
from tilelang.contrib.sm120_nvfp4 import prefill_rs
print([getattr(f, "_horizon", False) for f in (qn.fused_qk_rmsnorm_rope_gate, q.causal_conv1d_update,
       co.fused_gdn_decode_post_conv_mtp, fi.flashinfer_xqa_batch_decode_with_kv_cache, prefill_rs.quantize_rs)])
