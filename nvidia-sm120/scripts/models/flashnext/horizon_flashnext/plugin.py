# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin (entry point `vllm.general_plugins: horizon_flashnext = horizon_flashnext.plugin:register`).

Does nothing unless HORIZON_FN_EXPERT_MASK names a mask .npz written by prune_checkpoint.py --emit-mask. Then every
MoE runner sets the router logits of the masked experts to -inf before routing (uniform or non-uniform keep sets,
HORIZON_FN_MASK_VARIANT=uniform|nonuniform). For quality screens of a pruning on the FULL checkpoint only: weights stay
loaded, so it saves no memory. Works with CUDA graphs (the masking ops are captured with the MoE call).
"""
import os


def register():
    path = os.environ.get("HORIZON_FN_EXPERT_MASK", "").strip()
    if not path:
        return
    from . import hooks  # noqa: PLC0415
    n = hooks.load_mask(path, os.environ.get("HORIZON_FN_MASK_VARIANT", "uniform"))
    hooks.install_class_wrapper()
    print(f"horizon_flashnext: expert mask {path} ({n} MoE blocks, {hooks._STATE['mask']['variant']}) installed",
          flush=True)
