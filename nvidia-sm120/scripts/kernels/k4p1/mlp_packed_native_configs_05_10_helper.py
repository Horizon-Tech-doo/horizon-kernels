# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Frozen configuration families; GU uses the production bucket's stage depth."""
def configs(rows):
    result=[dict(name=f'control-gu128-down128-s4-grid{g}-packed',stages=4,grid=g,
                 block_n=128,block_k=64,stages_down=4,block_n_down=128) for g in (272,376)]
    for nd in (128,256):
        for sd in (3,4):
            result.append(dict(name=f'native-gu256-down{nd}-sd{sd}-grid188-packed',
                stages=6 if rows<=16 else 4,grid=188,block_n=256,block_k=64,
                stages_down=sd,block_n_down=nd))
    return result
