# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Run the GDN regression using only the isolated candidate segment module."""
import importlib.util
from pathlib import Path
import runpy
import sys

candidate = Path(__file__).resolve().parent
import tilelang.contrib.horizon_k4 as package
name = "tilelang.contrib.horizon_k4.segment"
spec = importlib.util.spec_from_file_location(name, candidate / "segment.py")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
package.segment = module
sys.argv[0] = str(candidate / "gdn_layer_test.py")
runpy.run_path(sys.argv[0], run_name="__main__")
