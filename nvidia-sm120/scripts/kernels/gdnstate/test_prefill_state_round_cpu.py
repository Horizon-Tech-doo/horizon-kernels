# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU reproduction: cache storage alone changes the recurrence; canonical rounding fixes it.

Small grouped gated-delta recurrences, multiple head/key/value geometries, a
fixed 64-token reusable boundary and different aligned prefill segment plans.
Stdlib only. This tests the boundary contract, not GPU FLA arithmetic/quality.
"""
import math
import random
import struct
import unittest


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def half(x):
    return struct.unpack("e", struct.pack("e", max(-65504.0, min(65504.0, x))))[0]


def inputs(n, heads, dk, dv, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        k = [rng.uniform(-1, 1) for _ in range(dk)]
        norm = math.sqrt(sum(x*x for x in k))
        k = [f32(x / norm) for x in k]
        rows.append((k, [[f32(rng.uniform(-.8, .8)) for _ in range(dv)] for _ in range(heads)],
                     [f32(rng.uniform(.98, 1)) for _ in range(heads)], f32(rng.uniform(.1, .9))))
    return rows


def process(rows, state, start, end, canonical=False, store=True, round_period=64):
    state = [[row[:] for row in head] for head in state]
    outputs = []
    for t in range(start, end):
        k, v, g, beta = rows[t]
        for h in range(len(state)):
            for j in range(len(state[h])):
                sh = [f32(x * g[h]) for x in state[h][j]]
                dot = 0.
                for x, ki in zip(sh, k):
                    dot = f32(dot + f32(x * ki))
                delta = f32(f32(v[h][j] - dot) * beta)
                state[h][j] = [f32(x + f32(ki * delta)) for x, ki in zip(sh, k)]
        outputs.append(tuple(x for head in state for row in head for x in row))
        if canonical and (t + 1) % round_period == 0:
            state = [[[half(x) for x in row] for row in head] for head in state]
    if store:
        state = [[[half(x) for x in row] for row in head] for head in state]
    return outputs, state


class BoundaryTests(unittest.TestCase):
    def test_speculative_decode_window_partition(self):
        rows = inputs(13, 3, 5, 4, 73)
        zero = [[[0.] * 5 for _ in range(4)] for _ in range(3)]
        full, _ = process(rows, zero, 0, 13)
        state, joined, start = zero, [], 0
        for end in (1, 4, 5, 9, 13):
            part, state = process(rows, state, start, end)
            joined.extend(part)
            start = end
        self.assertNotEqual(full, joined, "stored-only rounding depends on the verify-window partition")
        full, final = process(rows, zero, 0, 13, canonical=True, round_period=1)
        state, joined, start = zero, [], 0
        for end in (1, 4, 5, 9, 13):
            part, state = process(rows, state, start, end, canonical=True, round_period=1)
            joined.extend(part)
            start = end
        self.assertEqual(full, joined)
        self.assertEqual(final, state)

    def test_cache_and_scheduler_splits(self):
        for heads, dk, dv in ((1, 2, 3), (3, 5, 2), (4, 7, 6)):
            with self.subTest(geometry=(heads, dk, dv)):
                rows = inputs(389, heads, dk, dv, heads + dk + dv)
                zero = [[[0.] * dk for _ in range(dv)] for _ in range(heads)]
                plain, _ = process(rows, zero, 0, len(rows))
                _, checkpoint = process(rows, zero, 0, 128)
                resumed, _ = process(rows, checkpoint, 128, len(rows))
                self.assertNotEqual(plain[128:], resumed, "negative control must expose fp16 cache rounding")
                full, final = process(rows, zero, 0, len(rows), canonical=True)
                for cut in (64, 128, 256, 384):
                    _, checkpoint = process(rows, zero, 0, cut, canonical=True)
                    tail, tail_final = process(rows, checkpoint, cut, len(rows), canonical=True)
                    self.assertEqual(full[cut:], tail)
                    self.assertEqual(final, tail_final)
                # Different scheduler/checkpoint segmentation, same reusable boundaries.
                state, joined, start = zero, [], 0
                for end in (64, 192, 256, len(rows)):
                    part, state = process(rows, state, start, end, canonical=True)
                    joined.extend(part)
                    start = end
                self.assertEqual(full, joined)
                self.assertEqual(final, state)

    def test_partial_unaligned_resume_is_not_claimed(self):
        rows = inputs(139, 1, 3, 2, 19)
        zero = [[[0.] * 3 for _ in range(2)]]
        full, _ = process(rows, zero, 0, 139, canonical=True)
        _, checkpoint = process(rows, zero, 0, 67, canonical=True)
        tail, _ = process(rows, checkpoint, 67, 139, canonical=True)
        self.assertNotEqual(full[67:], tail)


if __name__ == "__main__":
    unittest.main()
