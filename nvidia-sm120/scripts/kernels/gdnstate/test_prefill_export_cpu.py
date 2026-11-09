# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('export', ROOT/'horizon_kernels/gdn_prefill_export.py')
adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)


class ExportPlan(unittest.TestCase):
    def test_packed_sequences_multiple_cuts_and_partial_tails(self):
        plan = adapter.export_plan([0, 1000, 1313, 8416], [[832, 896], [], [832, 7040]],
                                   [(0, 0), (0, 1), (2, 0), (2, 1)])
        self.assertEqual(len(plan), 16+5+111)
        self.assertEqual([(i, x) for i, x in enumerate(plan) if x >= 0],
                         [(13, 0), (14, 1), (34, 2), (131, 3)])

    def test_rejects_invalid_or_duplicate_exports(self):
        for cuts, exports in [([[65]], [(0, 0)]), ([[128]], [(0, 0)]),
                              ([[64]], [(0, 0), (0, 0)])]:
            with self.assertRaises(ValueError):
                adapter.export_plan([0, 128], cuts, exports)

    def test_kernel_exports_before_low_precision_scratch_write(self):
        # Use the pinned canonical source retained locally by the previous probe.
        source = Path('/tmp/wp13_chunk_delta_h.py')
        if not source.exists():
            self.skipTest('pinned vendor source unavailable')
        s = importlib.util.spec_from_file_location('rounding', ROOT/'horizon_kernels/gdn_prefill_state.py')
        rounding = importlib.util.module_from_spec(s); s.loader.exec_module(rounding)
        result = adapter.transform_source(rounding.transform_source(source.read_text()))
        ast.parse(result)
        self.assertLess(result.index('tl.store(p_export, b_h1'), result.index('tl.store(p_h1, b_h1'))
        self.assertEqual(result.count('tl.store(p_export,'), 4)


if __name__ == '__main__':
    unittest.main()
