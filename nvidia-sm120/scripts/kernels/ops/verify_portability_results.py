# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate complete GPU evidence for planned prefill and RMSNorm candidates."""
import itertools
import json
import math
import sys


def verify_prefill(rows):
    assert rows[-1] == {'test': 'complete', 'ok': True}
    shapes = {((17, 63), (100, 129)), ((1024,), (4096,)), ((8192,), (16384,))}
    gates = [r for r in rows if r['test'] == 'planned_gate']
    expected = {(v, q, k) for v in ('v1', 'v2', 'v3') for q, k in shapes}
    assert len(gates) == len(expected)
    assert {(r['variant'], tuple(r['qlens']), tuple(r['kvlens'])) for r in gates} == expected
    assert all(r['ok'] is True and math.isfinite(r['rel_l2']) and 0 <= r['rel_l2'] <= .0025 for r in gates)
    timings = [r for r in rows if r['test'] == 'planned_timing']
    assert len(timings) == len(shapes)
    assert {(tuple(r['qlens']), tuple(r['kvlens'])) for r in timings} == shapes
    for r in timings:
        assert r['page_size'] == 64 and isinstance(r['flashinfer_backend'], str) and r['flashinfer_backend']
        assert set(r['ms']) == {'flashinfer', 'v1', 'v2', 'v3'}
        assert all(len(v) == 3 and all(math.isfinite(x) and x > 0 for x in v) for v in r['ms'].values())


def verify_norm(rows):
    assert rows[-1] == {'complete': True, 'cases': 48, 'zero_weight_semantics': True}
    cases = rows[:-1]
    expected = set(itertools.product((256, 768, 4096, 5120), (1, 17, 32), (False, True), (False, True)))
    assert len(cases) == len(expected)
    assert {(r['H'], r['M'], r['zero_centered_gamma'], r['residual']) for r in cases} == expected
    assert all(r['deterministic'] is True and r['graph_replays'] == 2 and
               math.isfinite(r['relative_error']) and 0 <= r['relative_error'] <= .0025 for r in cases)


if __name__ == '__main__':
    with open(sys.argv[2]) as f:
        rows = [json.loads(line) for line in f]
    {'prefill': verify_prefill, 'norm': verify_norm}[sys.argv[1]](rows)
    print('VERIFIED', sys.argv[1])
