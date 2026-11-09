# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-only packed-byte permutation; retain original weights for fallback."""
from types import SimpleNamespace
from tilelang.contrib.horizon_k4.mlp_schedule import paired_block_order


def prepare_paired(gate_up):
    import torch
    if getattr(gate_up, '_k4_layout', None) == 'gate128-up128':
        return gate_up
    n = gate_up.Wg.shape[0] * 32
    if n % 256 or gate_up.Sg.shape[0] * 32 != n or gate_up.alpha.numel() != n:
        raise ValueError('packed gate/up metadata is incompatible')
    order = torch.tensor(paired_block_order(n//2), device=gate_up.Wg.device, dtype=torch.long)
    return SimpleNamespace(Wg=gate_up.Wg.index_select(0,order).contiguous(),
                           Sg=gate_up.Sg.index_select(0,order).contiguous(),
                           alpha=gate_up.alpha.view(n//32,32).index_select(0,order).reshape(-1).contiguous(),
                           _k4_layout='gate128-up128')
