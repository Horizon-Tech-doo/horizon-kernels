# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Check opaque packed rows and alpha columns map to the kernel's logical columns."""
import importlib.util
from pathlib import Path
import sys
spec=importlib.util.spec_from_file_location('mlp_schedule',Path(sys.argv[1])/'mlp_schedule.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
columns=0
for intermediate in (128,256,14336,17408):
    order=module.paired_block_order(intermediate)
    assert sorted(order)==list(range(2*intermediate//32))
    opaque=[(block,0xffffffff-block) for block in range(len(order))]
    packed=[opaque[i] for i in order]
    restored=[None]*len(order)
    for dst,original in enumerate(order):restored[original]=packed[dst]
    assert restored==opaque
    for tile in range(intermediate//128):
        for warp in range(8):
            physical=tile*8+warp
            for lane_column in range(32):
                logical=tile*128+(warp%4)*32+(warp//4)*intermediate+lane_column
                assert order[physical]*32+lane_column==logical
                columns+=1
for invalid in (0,127,129,-128):
    try:module.paired_block_order(invalid)
    except ValueError:pass
    else:raise AssertionError('invalid paired width accepted')
print(f'MLP_PACKED_LAYOUT_CPU_PASS widths=4 columns={columns} inverse_exact=True invalid_rejected=4')
