#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Check packed checkpoint values/layout without opening a CUDA context."""
import torch
from gdn_checkpoint_metadata import pack_metadata

for strided in (False, True):
    table = torch.arange(60, dtype=torch.int32).view(6, 10)
    if strided:
        table = table[:, ::2]
    for n in (1, 3, 6):
        rr = list(range(n))
        cc = [i % table.shape[1] for i in rr]
        history = [64*(i+1) for i in rr]
        mapping = [-1] + rr + [-1]
        slots, rows, exports = pack_metadata(table, rr, cc, history, mapping)
        assert torch.equal(slots, table[rr, cc])
        assert rows.tolist() == history and exports.tolist() == mapping
        assert rows.dtype == exports.dtype == torch.int64
        assert rows.untyped_storage().data_ptr() == exports.untyped_storage().data_ptr()
        assert rows.is_contiguous() and exports.is_contiguous()
for args in (([], [], [], []), ([0], [], [64], [0]), ([0], [0], [64], [1])):
    try:
        pack_metadata(torch.zeros((1, 1), dtype=torch.int32), *args)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid checkpoint metadata accepted')
print('6 contiguous/strided metadata cases and 3 invalid-input controls pass')
