# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HORIZON_GDN_RECOVER"] = "1"
import torch
torch.cuda.is_available = lambda: False
sys.path.insert(0, "$HK_WORK/ops/hpkgs2")
from horizon_kernels import gdn_recover as g
g.install()
from vllm.v1.attention.backends import gdn_attn
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata
B = gdn_attn.GDNAttentionBackend.get_builder_cls()
print("builder:", B.__name__, issubclass(B, gdn_attn.GDNAttentionMetadataBuilder))
M = g._metadata_cls()
from dataclasses import fields
print("meta fields:", len(fields(M)), issubclass(M, RecoverSSMMetadata))
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as q
print("fused hook:", getattr(q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm, "_horizon", False),
      q.fused_sigmoid_gating_delta_rule_update.__name__)
from vllm.models.kimi_k3.nvidia.ops.recoverssm import _compact_conv_state_kernel, _prepare_commit_plan_kernel
print("kimi kernels importable")
