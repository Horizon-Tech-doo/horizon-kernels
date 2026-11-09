#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Build an immutable WP1.3 fp16 candidate from the current caching package.

CPU only: no imports of torch/vLLM/TileLang, no service or production writes.
The candidate uses G1S fp16; production's fp32 G3v remains the comparison arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil


def env_values(path):
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.startswith("#"):
            words = shlex.split(value)
            values[key] = " ".join(words)
    return values


def build(env, out, kernels, plugin):
    env, out, kernels, plugin = map(Path, (env, out, kernels, plugin))
    values = env_values(env)
    required = {"HORIZON_PI": "1", "HORIZON_GDN_CKPT": "1",
                "HORIZON_GDN_CKPT_STOP": "fill", "HORIZON_TAIL_MIN_BLOCK": "1"}
    for key, val in required.items():
        if values.get(key) != val:
            raise ValueError(f"base must have {key}={val}")
    args = shlex.split(values.get("VLLM_EXTRA_ARGS", ""))
    if "--prefix-match-unit" not in args or args[args.index("--prefix-match-unit") + 1] != "64":
        raise ValueError("base must have --prefix-match-unit 64")
    if "--enable-mamba-fine-grained-prefix-cache" not in args:
        raise ValueError("base must have fine-grained mamba prefix caching")
    entries = values.get("PYTHONPATH", "").split(":")
    tl = next((Path(p) for p in entries if p and (Path(p) / "tilelang/__init__.py").is_file()), None)
    hk = next((Path(p) for p in entries if p and (Path(p) / "horizon_kernels/vllm_plugin.py").is_file()), None)
    if tl is None or hk is None:
        raise ValueError("base PYTHONPATH lacks TileLang or the plugin")
    for source in (kernels / "gdn_state.py", kernels / "gdn_decode.py", plugin):
        if not source.is_file():
            raise ValueError(f"missing source: {source}")
    if out.exists() or out.is_symlink():
        raise ValueError("output already exists; use a fresh candidate name")
    # Prevent creating the overlay inside its own source tree, including via symlinks.
    for source in (tl, hk, kernels):
        if out.resolve().is_relative_to(source.resolve()):
            raise ValueError("output must be outside all source packages")
    hook_path = hk / "horizon_kernels/vllm_plugin.py"
    register = hook_path.read_text()
    if "from horizon_kernels import gdn_state" not in register:
        anchor = "        gdn_recover.install()\n"
        if register.count(anchor) != 1:
            raise ValueError("missing unique plugin registration anchor")
        hook = ('    if (os.environ.get("HORIZON_GDN_STATE", "") not in ("", "0", "fp32", "float32", "auto")\n'
                '            or os.environ.get("HORIZON_GDN2", "0") == "1"):\n'
                '        from horizon_kernels import gdn_state\n'
                '        gdn_state.install()\n')
        register = register.replace(anchor, anchor + hook)
    out.mkdir(parents=True)
    try:
        # Link unchanged TileLang files, but NEVER link a file we will overwrite.
        replaced = {"gdn_state.py", "gdn_decode.py"}
        def overlay(src, dst, branch):
            dst.mkdir()
            for item in src.iterdir():
                if item.name == "__pycache__":
                    continue
                if branch and item.name == branch[0]:
                    overlay(item, dst / item.name, branch[1:])
                elif not branch and item.name in replaced:
                    shutil.copy2(kernels / item.name, dst / item.name)
                else:
                    (dst / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
            if not branch:
                for name in replaced:
                    if not (dst / name).exists():
                        shutil.copy2(kernels / name, dst / name)
        overlay(tl, out / "tl", ["tilelang", "contrib", "horizon_ops"])
        # Dereference plugin links: all patched files are private ordinary files.
        shutil.copytree(hk, out / "hk", symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(plugin, out / "hk/horizon_kernels/gdn_state.py")
        (out / "hk/horizon_kernels/vllm_plugin.py").write_text(register)
        pp = ":".join(str(out / "tl") if Path(p) == tl else str(out / "hk") if Path(p) == hk else p
                      for p in entries)
        overrides = {"PYTHONPATH": pp, "HORIZON_GDN_G3": "0", "HORIZON_GDN_G3_CHECK": "0",
                     "HORIZON_GDN": "0", "HORIZON_GDN_RECOVER": "0",
                     "HORIZON_GDN_STATE": "fp16", "HORIZON_GDN2": "1"}
        manifest = {"variant": "G1S fp16 on PI; G3v disabled in candidate only", "set": overrides,
                    "sha256": {}}
        for name in ("tl/tilelang/contrib/horizon_ops/gdn_state.py",
                     "tl/tilelang/contrib/horizon_ops/gdn_decode.py",
                     "hk/horizon_kernels/gdn_state.py", "hk/horizon_kernels/vllm_plugin.py"):
            manifest["sha256"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
        (out / "candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except Exception:
        shutil.rmtree(out)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    a = parser.parse_args()
    result = build(a.env, a.out, a.kernels, a.plugin)
    print(json.dumps({"candidate": str(a.out / "candidate.json"), "variant": result["variant"]}))
