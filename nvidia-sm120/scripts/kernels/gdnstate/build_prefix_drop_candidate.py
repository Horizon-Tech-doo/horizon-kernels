#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Add the default-off drop-aware cache lookup to a fresh private candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def build(base, out, adapter):
    base, out, adapter = map(Path, (base, out, adapter))
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a fresh candidate outside the base')
    manifest = json.loads((base/'candidate.json').read_text())
    for name, sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'base candidate changed: {name}')
    shutil.copytree(base, out, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'offline', 'offline-output'))
    try:
        target = out/'hk/horizon_kernels/prefix_drop_lookup.py'
        shutil.copy2(adapter, target)
        dist = out/'hk/horizon_prefix_drop_lookup-0.1.dist-info'
        dist.mkdir()
        (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: horizon-prefix-drop-lookup\nVersion: 0.1\n')
        (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nhorizon_prefix_drop_lookup = horizon_kernels.prefix_drop_lookup:register\n')
        manifest['set']['PYTHONPATH'] = manifest['set']['PYTHONPATH'].replace(str(base), str(out))
        manifest['set']['HORIZON_PREFIX_DROP_LOOKUP'] = '1'
        manifest['variant'] += '; drop-aware fine-prefix lookup cap'
        for name in ('hk/horizon_kernels/prefix_drop_lookup.py',
                     'hk/horizon_prefix_drop_lookup-0.1.dist-info/METADATA',
                     'hk/horizon_prefix_drop_lookup-0.1.dist-info/entry_points.txt'):
            manifest['sha256'][name] = hashlib.sha256((out/name).read_bytes()).hexdigest()
        (out/'candidate.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest
    except Exception:
        shutil.rmtree(out)
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'out', 'adapter'):
        p.add_argument(name, type=Path)
    a=p.parse_args()
    m=build(a.base, a.out, a.adapter)
    print(json.dumps({'candidate': str(a.out), 'variant': m['variant']}))
