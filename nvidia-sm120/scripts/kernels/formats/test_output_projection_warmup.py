# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage tests for FP8 output-projection warm-up selection."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = Path(os.environ.get("HORIZON_HEAD_TEST_SOURCE",
                             Path(__file__).resolve().parents[3] / "horizon_kernels/lmhead.py"))
spec = importlib.util.spec_from_file_location("output_projection_under_test", SOURCE)
head = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    spec.loader.exec_module(head)


class WarmupCoverage(unittest.TestCase):
    def test_dynamic_bucket_edges(self):
        key = lambda m: 0 if m <= 16 else 1 if m <= 32 else 2 if m <= 64 else 3
        rows = head._configuration_boundary_rows(key, 1024)
        self.assertEqual(rows, [1, 16, 17, 32, 33, 64, 65, 1024])
        self.assertEqual({key(m) for m in rows}, {key(m) for m in range(1, 1025)})

    def test_noncontiguous_and_singleton_configurations(self):
        configs = ["a", "b", "a", "c", "b", "a"]
        self.assertEqual(head._configuration_boundary_rows(lambda m: configs[m-1], 6),
                         [1, 2, 4, 5, 6])

    def test_each_row_can_have_its_own_configuration(self):
        self.assertEqual(head._configuration_boundary_rows(lambda m: m, 32),
                         list(range(1, 33)))

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            head._configuration_boundary_rows(lambda m: m, 0)

    def test_nvfp4_keeps_full_warmup(self):
        self.assertEqual(list(head._head_warmup_rows("nvfp4", None)),
                         list(range(1, 1025)))

    def test_fp8_normalizes_implicit_min_blocks_and_keeps_distinct_variants(self):
        base = (1, 16, 4, 1, 128, 1, 4)
        module = types.ModuleType("tilelang.contrib.sm120_fp8")
        module.decode_w8a16 = types.SimpleNamespace(
            config=lambda n, k, m: base if m <= 16 else base+(1,) if m <= 32 else base+(2,))
        with patch.dict(sys.modules, {"tilelang.contrib.sm120_fp8": module}):
            rows = head._head_warmup_rows("fp8", types.SimpleNamespace(N=248320, K=5120))
        self.assertEqual(rows, [1, 32, 33, 1024])


if __name__ == "__main__":
    unittest.main()
