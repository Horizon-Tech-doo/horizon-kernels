#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Private G1S fp16 candidate with canonical carry at every verify token."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

SOURCE_SHA256 = '1a4a28834379ba766b0a699d72ea485a7f84426028135638e4215d7d0dde4369'
KERNEL = 'tl/tilelang/contrib/horizon_ops/gdn_state.py'


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('G1S source changed; review speculative carry adapter')
    anchor = '                        dst = sidx[req, t]\n'
    if source.count(anchor) != 1:
        raise ValueError('missing unique G1S token store anchor')
    rounding = '''                        if sdt == "float16":
                            # Next verify token uses exactly the value a resumed window loads.
                            for r in T.unroll(RPW):
                                for i in T.unroll(4):
                                    h[r * 4 + i] = _ld(sdt, _st(sdt, h[r * 4 + i]))
'''
    result = source.replace(anchor, rounding + anchor)
    ast.parse(result)
    return result


def build(base, out):
    base, out = Path(base), Path(out)
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a new candidate directory outside the base')
    manifest = json.loads((base/'candidate.json').read_text())
    if manifest['set'].get('HORIZON_GDN_PREFILL_STATE_ROUND') != 'float16':
        raise ValueError('base must include canonical fp16 prefill carry')
    for name, sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'base candidate changed: {name}')
    source = transform((base/KERNEL).read_text())
    shutil.copytree(base, out, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    try:
        if (out/KERNEL).is_symlink():
            raise ValueError('patched kernel must be a private ordinary file')
        (out/KERNEL).write_text(source)
        manifest['sha256'][KERNEL] = hashlib.sha256((out/KERNEL).read_bytes()).hexdigest()
        manifest['set']['PYTHONPATH'] = manifest['set']['PYTHONPATH'].replace(str(base), str(out))
        manifest['variant'] += '; canonical fp16 carry every G1S verify token'
        (out/'candidate.json').write_text(json.dumps(manifest, indent=2)+'\n')
        return manifest
    except Exception:
        shutil.rmtree(out)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = build(args.base, args.out)
    print(json.dumps({'candidate': str(args.out), 'variant': result['variant']}))
