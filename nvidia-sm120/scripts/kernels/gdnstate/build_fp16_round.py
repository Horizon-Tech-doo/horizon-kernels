#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Create a fresh isolated rounding candidate from the immutable fp16 PI overlay."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def build(base, out, adapter, vendor):
    base, out, adapter, vendor = map(Path, (base, out, adapter, vendor))
    if out.exists() or out.is_relative_to(base):
        raise ValueError("use a new candidate directory outside the source")
    manifest = json.loads((base / "candidate.json").read_text())
    for name, sha in manifest["sha256"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"base candidate changed: {name}")
    spec = importlib.util.spec_from_file_location("round_adapter", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.transform_source(vendor.read_text())
    ast.parse(source)
    shutil.copytree(base, out, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        hk = out / "hk/horizon_kernels"
        shutil.copy2(adapter, hk / "gdn_prefill_state.py")
        private = hk / "gdn_round_ops/chunk_delta_h.py"
        private.parent.mkdir()
        private.write_text(source)
        dist = out / "hk/horizon_gdn_state_round-0.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: horizon-gdn-state-round\nVersion: 0.1\n")
        (dist / "entry_points.txt").write_text(
            "[vllm.general_plugins]\nhorizon_gdn_state_round = horizon_kernels.gdn_prefill_state:register\n")
        manifest["set"]["PYTHONPATH"] = manifest["set"]["PYTHONPATH"].replace(str(base), str(out))
        manifest["set"]["HORIZON_GDN_PREFILL_STATE_ROUND"] = "float16"
        manifest["variant"] += "; canonical fp16 carry every FLA chunk"
        for name in ("hk/horizon_kernels/gdn_prefill_state.py",
                     "hk/horizon_kernels/gdn_round_ops/chunk_delta_h.py",
                     "hk/horizon_gdn_state_round-0.1.dist-info/METADATA",
                     "hk/horizon_gdn_state_round-0.1.dist-info/entry_points.txt"):
            manifest["sha256"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
        (out / "candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception:
        shutil.rmtree(out)
        raise
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "out", "adapter", "vendor"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    m = build(a.base, a.out, a.adapter, a.vendor)
    print(json.dumps({"candidate": str(a.out), "variant": m["variant"]}))
