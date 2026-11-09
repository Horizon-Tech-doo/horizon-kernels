# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression for reusable prefill metadata; no CUDA/TileLang imports."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(sys.argv.pop(1))
names = {'_prefill_tile_pairs', 'prefill', 'prefill_v2', 'prefill_v3'}
tree = ast.parse(source.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
ns = {'LOG2E': 1.4426950408889634}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), ns)


class PlanTests(unittest.TestCase):
    def test_ragged_and_empty_requests(self):
        self.assertEqual(ns['_prefill_tile_pairs']([0, 0, 17, 25], 8),
                         [(1, 0), (1, 8), (1, 16), (2, 0)])
        self.assertEqual(ns['_prefill_tile_pairs']([0], 16), [])
        self.assertEqual(ns['_prefill_tile_pairs']([0, 16, 32], 16), [(0, 0), (1, 0)])

    def test_invalid_offsets(self):
        for offsets, tq in [([], 8), ([1, 3], 8), ([0, 3, 2], 8), ([0, 3], 0)]:
            with self.assertRaises(ValueError):
                ns['_prefill_tile_pairs'](offsets, tq)
        with self.assertRaises(TypeError):
            ns['_prefill_tile_pairs']([0, 3.5], 8)

    def test_planned_wrappers_skip_host_planning(self):
        class Cache:
            shape = (3, 64, 4, 256)
            def stride(self):
                return (65536, 1024, 256, 1)
        class Offsets:
            def cpu(self):
                raise AssertionError('unexpected GPU-to-CPU read')
        query = SimpleNamespace(shape=(17, 24, 256), device='sentinel-device')
        for name in ('prefill', 'prefill_v2', 'prefill_v3'):
            calls = []
            tiles, output = object(), object()
            def factory(*args):
                return lambda *a: calls.append(a)
            def bad_plan(*args):
                raise AssertionError('replanned during execution')
            ns.update(_KERNELS={}, plan_prefill_tiles=bad_plan,
                      attn_prefill_kernel=factory, attn_prefill_kernel_v2=factory,
                      attn_prefill_kernel_v3=factory)
            with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
                result = ns[name](query, Cache(), Cache(), Offsets(), object(), object(),
                                  .125, .05, out=output, tiles=tiles)
            self.assertIs(result, output)
            self.assertIs(calls[0][6], tiles)
            self.assertIs(calls[0][7], output)
            if name == 'prefill_v3':
                with patch.dict(sys.modules, {'torch': SimpleNamespace()}):
                    ns[name](query, Cache(), Cache(), Offsets(), object(), object(),
                             .125, .05, out=output, tiles=tiles, num_warps=8,
                             policy_qk=3, policy_pv=3, pad_rows=True)
                self.assertEqual(len(ns['_KERNELS']), 2)
                self.assertEqual({key[-4:] for key in ns['_KERNELS']},
                                 {(0, 0, 0, False), (8, 3, 3, True)})



if __name__ == '__main__':
    unittest.main()
