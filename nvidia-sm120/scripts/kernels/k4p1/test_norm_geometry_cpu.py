# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Verify virtual scheduling preserves standalone per-thread element order."""
import ast
from pathlib import Path
import sys
import unittest

source = Path(sys.argv.pop(1)).read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'norm_geometry')
ns = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<norm_geometry>', 'exec'), ns)
geometry = ns['norm_geometry']

class GeometryTests(unittest.TestCase):
    def test_expected_dispatch(self):
        for h, expected in ((256,(32,1)),(512,(64,1)),(768,(32,3)),(4096,(128,4)),(5120,(640,1)),(10240,(640,2))):
            self.assertEqual(geometry(h), expected)

    def test_element_order_and_coverage(self):
        for h in range(256, 16385, 256):
            vt, per = geometry(h)
            seen = []
            warp_indices = []
            for q in range((vt + 127)//128):
                for tid in range(128):
                    virtual_tid = q*128 + tid
                    if virtual_tid >= vt:
                        continue
                    actual = [(i*vt + virtual_tid)*8+v for i in range(per) for v in range(8)]
                    expected = [(i*vt + virtual_tid)*8+v for i in range(h//(vt*8)) for v in range(8)]
                    self.assertEqual(actual, expected)
                    seen.extend(actual)
                    if tid % 32 == 0:
                        warp_indices.append(q*4+tid//32)
            self.assertEqual(sorted(seen), list(range(h)))
            self.assertEqual(warp_indices, list(range(vt//32)))

    def test_invalid(self):
        for h, vt in ((0,None),(-256,None),(257,None),(4096,640),(256,0),(768,96),(32768,2048)):
            with self.assertRaises(ValueError):
                geometry(h, vt)

if __name__ == '__main__':
    unittest.main()
