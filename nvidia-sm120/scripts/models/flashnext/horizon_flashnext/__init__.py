# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""horizon_flashnext: vLLM-side helpers for the Qwen3.8-Flash-Next expert-pruning work (Horizon Tech, 03-10-2026).

hooks.py   in-worker REAP calibration hooks and the runtime expert mask (MoERunner._apply_quant_method wrapper)
plugin.py  vLLM general plugin: installs the runtime expert mask when HORIZON_FN_EXPERT_MASK names a mask .npz
"""
