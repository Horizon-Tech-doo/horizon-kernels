# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Pack checkpoint indices and export metadata into one device transfer."""


def pack_metadata_ex(block_table, rows, columns, history_rows, export_map):
    """-> (slots, history_rows, export_map, rows_cols). rows_cols is the packed (rows + columns) int64 device tensor, kept
    so another KV-cache group's block table can re-derive its own slots (regather_slots; vLLM 0.32's
    update_block_table reuses one group's metadata for the groups with the same spec)."""
    import torch
    n = len(rows)
    if not n or len(columns) != n or len(history_rows) != n:
        raise ValueError('checkpoint metadata lengths differ or are empty')
    if any(index < -1 or index >= n for index in export_map):
        raise ValueError('checkpoint export map contains an invalid slot')
    packed = torch.tensor(rows + columns + history_rows + export_map,
                          device=block_table.device, dtype=torch.int64)
    slots = block_table[packed[:n], packed[n:2*n]]
    return slots, packed[2*n:3*n], packed[3*n:], packed[:2*n]


def pack_metadata(block_table, rows, columns, history_rows, export_map):
    slots, history, export, _ = pack_metadata_ex(block_table, rows, columns, history_rows, export_map)
    return slots, history, export


def regather_slots(block_table, rows_cols):
    """Cache slot of every export in another group's block table (same rows and columns as pack_metadata)."""
    n = rows_cols.numel() // 2
    return block_table[rows_cols[:n], rows_cols[n:]]
