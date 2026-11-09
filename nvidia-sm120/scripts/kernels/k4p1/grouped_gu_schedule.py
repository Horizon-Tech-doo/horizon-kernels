# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Experimental task placement: group GU tiles by consumer slice, keep op phases.

No device code changes. This factory changes only the input order of GU tasks;
all waits, signal counts, split IDs and other operation phases are preserved.
"""
def schedule_type(base, gu_op, intermediate, block_n, down_splits):
    if intermediate % (block_n * down_splits):
        raise ValueError('gate/up consumer slices must contain whole output tiles')
    half_tiles = intermediate // block_n
    tiles_per_slice = half_tiles // down_splits

    class GroupedGateUpSchedule(base):
        def add(self, op, tasks, cta_of=None, signals_per_task=1.0):
            if op == gu_op:
                # Pair the gate and up halves within each consumer slice.
                tasks = sorted(tasks, key=lambda d: (
                    (d[0] % half_tiles) // tiles_per_slice, d[0], d[1]))
            return super().add(op, tasks, cta_of=cta_of,
                               signals_per_task=signals_per_task)
    return GroupedGateUpSchedule
