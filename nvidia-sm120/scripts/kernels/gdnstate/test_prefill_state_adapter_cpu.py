# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only registration guards and pinned upstream transformation checks."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[3] / "horizon_kernels/gdn_prefill_state.py"


def adapter():
    spec = importlib.util.spec_from_file_location("round_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Registration(unittest.TestCase):
    def test_default_off_and_precision_guard_import_no_cuda_code(self):
        m = adapter()
        before = set(sys.modules)
        with patch.dict(os.environ, {}, clear=True):
            m.register()
            self.assertFalse(m.STATE["installed"])
        with patch.dict(os.environ, {"HORIZON_GDN_STATE": "fp32",
                                    "HORIZON_GDN_PREFILL_STATE_ROUND": "float16"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must match"):
                m.register()
        self.assertFalse(any(name.startswith(("torch", "vllm", "triton"))
                             for name in set(sys.modules) - before))

    def test_changed_vendor_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "source changed"):
            adapter().transform_source("changed upstream kernel")

    @unittest.skipUnless(os.environ.get("GDN_VENDOR_SOURCE"), "set GDN_VENDOR_SOURCE to pinned vendor file")
    def test_pinned_vendor_rounds_inside_recurrence_loop(self):
        m = adapter()
        source = Path(os.environ["GDN_VENDOR_SOURCE"]).read_text()
        tree = ast.parse(m.transform_source(source))
        kernel = next(f for f in tree.body if isinstance(f, ast.FunctionDef)
                      and f.name.endswith("blockdim64"))
        loop = next(s for s in kernel.body if isinstance(s, ast.For))
        self.assertIsInstance(loop.body[-1], ast.If)
        self.assertEqual(ast.unparse(loop.body[-1].test), "STATE_ROUND_DTYPE is not None")
        self.assertEqual(sum(isinstance(s, ast.Assign) for s in ast.walk(loop.body[-1])), 8)


if __name__ == "__main__":
    unittest.main()
