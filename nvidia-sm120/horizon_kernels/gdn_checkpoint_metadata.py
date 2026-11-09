# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Pack checkpoint indices and export metadata into one device transfer."""


def pack_metadata(block_table, rows, columns, history_rows, export_map):
    import torch
    n = len(rows)
    if not n or len(columns) != n or len(history_rows) != n:
        raise ValueError('checkpoint metadata lengths differ or are empty')
    if any(index < -1 or index >= n for index in export_map):
        raise ValueError('checkpoint export map contains an invalid slot')
    packed = torch.tensor(rows + columns + history_rows + export_map,
                          device=block_table.device, dtype=torch.int64)
    slots = block_table[packed[:n], packed[n:2*n]]
    return slots, packed[2*n:3*n], packed[3*n:]
