# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""horizon_ops: Horizon Tech op bodies for the Qwen3.8 decode step (GDN core, attention) on SM120.

Every op is a TileLang T.macro body (task-friendly: task indices + tid + caller-provided shared buffers, no grid
assumptions, ends with `sync()`) plus a thin kernel wrapper that the vLLM plugin (horizon_kernels) launches today and
that the K4 persistent kernel will run as tasks later. See each module's docstring for its contract and DEPS table.
"""
