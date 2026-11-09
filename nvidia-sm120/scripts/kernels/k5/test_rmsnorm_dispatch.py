# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of dispatch/cache functions from the candidate kernel source."""
import ast
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

source = Path(sys.argv.pop(1))
tree = ast.parse(source.read_text())
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name in {'supported', '_threads', 'get_kernel'}], type_ignores=[])


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'VEC': 8, '_KERNELS': {}, 'add_rmsnorm_kernel': Mock(side_effect=lambda *a: object())}
        exec(compile(functions, str(source), 'exec'), self.ns)

    def test_shape_coverage_and_legacy_threads(self):
        for h, threads in ((256,32),(512,64),(768,32),(1024,128),(4096,128),
                           (5120,640),(7168,128),(10240,640)):
            with self.subTest(H=h):
                self.assertEqual(self.ns['_threads'](h), threads)
                self.assertEqual(h % (8 * threads), 0)

    def test_invalid_shapes(self):
        for h in (0, -256, 128, 257):
            self.assertFalse(self.ns['supported'](h))
            with self.assertRaises(ValueError):
                self.ns['_threads'](h)

    def test_weight_conventions_have_separate_binaries(self):
        get = self.ns['get_kernel']
        legacy = get(5120, 1e-6, True)
        self.assertIs(legacy, get(5120, 1e-6, True, zero_centered_gamma=True))
        standard = get(5120, 1e-6, True, zero_centered_gamma=False)
        self.assertIsNot(legacy, standard)
        self.assertIs(standard, get(5120, 1e-6, True, zero_centered_gamma=False))
        self.assertEqual(self.ns['add_rmsnorm_kernel'].call_count, 2)


if __name__ == '__main__':
    unittest.main()
