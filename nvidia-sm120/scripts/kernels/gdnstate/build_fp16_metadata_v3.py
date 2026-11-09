#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Build a fresh candidate with packed checkpoint metadata; preserve arithmetic."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def build(base, out, helper):
    base, out, helper = map(Path, (base, out, helper))
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a fresh candidate directory')
    manifest = json.loads((base/'candidate.json').read_text())
    for name, sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'base source changed: {name}')
    name = 'hk/horizon_kernels/gdn_ckpt.py'
    source = (base/name).read_text()
    old = '''        slots = m.block_table_tensor[torch.tensor(rr, dtype=torch.int64, device=dev),
                                     torch.tensor(cc_l, dtype=torch.int64, device=dev)]
        pq = [0]
        for r in rows:
            pq.append(pq[-1] + qsl[r + 1] - qsl[r])
        md.horizon_ckpt = CkptMeta(cuts, exports, slots, abs_l, pq)
        return md'''
    new = '''        pq = [0]
        for r in rows:
            pq.append(pq[-1] + qsl[r + 1] - qsl[r])
        from .gdn_prefill_export import export_plan
        from .gdn_checkpoint_metadata import pack_metadata
        slots, history_rows, export_map = pack_metadata(
            m.block_table_tensor, rr, cc_l, abs_l, export_plan(pq, cuts, exports))
        md.horizon_ckpt = CkptMeta(cuts, exports, slots, abs_l, pq)
        md.horizon_ckpt.horizon_abs_rows = history_rows
        md.horizon_ckpt.horizon_export_map = export_map
        return md'''
    if source.count(old) != 1:
        raise ValueError('checkpoint metadata source anchor changed')
    source = source.replace(old, new)
    marker = '    G.GDNAttentionMetadataBuilder.build = build\n'
    if source.count(marker) != 1:
        raise ValueError('checkpoint installation anchor changed')
    source = source.replace(marker, marker + '    _log().info("horizon_kernels checkpoint metadata: packed slot/history/export transfer")\n')
    updates = {name: source, 'hk/horizon_kernels/gdn_checkpoint_metadata.py': helper.read_text()}
    for value in updates.values():
        ast.parse(value)
    shutil.copytree(base, out, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'selftest.json'))
    for name, value in updates.items():
        path = out/name
        if path.is_symlink():
            raise ValueError('modified source must be private')
        path.write_text(value)
        manifest['sha256'][name] = hashlib.sha256(value.encode()).hexdigest()
    manifest['set'] = {key: value.replace(str(base), str(out)) if isinstance(value, str) else value
                       for key, value in manifest['set'].items()}
    manifest['variant'] += '; packed checkpoint metadata, unchanged recurrence and export kernels'
    (out/'candidate.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'out', 'helper'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    build(a.base, a.out, a.helper)
    print(json.dumps({'candidate': str(a.out)}))
