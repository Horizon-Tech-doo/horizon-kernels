#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Build an immutable single-pass checkpoint candidate from canonical fp16."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

CKPT_SHA = '42ef8f232f761a8f0b5a63543bb1e9afd46342a25a9d9e0415b2afbeb1f0672c'


def build(base, out, adapter):
    base, out, adapter = map(Path, (base, out, adapter))
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a fresh candidate directory')
    manifest = json.loads((base/'candidate.json').read_text())
    for name, sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'base changed: {name}')
    ckpt_path = 'hk/horizon_kernels/gdn_ckpt.py'
    ckpt = (base/ckpt_path).read_text()
    if hashlib.sha256(ckpt.encode()).hexdigest() != CKPT_SHA:
        raise ValueError('checkpoint architecture adapter source changed')
    spec = importlib.util.spec_from_file_location('export_adapter', adapter)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    kernel_path = 'hk/horizon_kernels/gdn_round_ops/chunk_delta_h.py'
    kernel = module.transform_source((base/kernel_path).read_text())
    round_path = 'hk/horizon_kernels/gdn_prefill_state.py'
    rounding = (base/round_path).read_text()
    rounding = rounding.replace('    expected = transform_source(original_file.read_text())',
        '    from .gdn_prefill_export import transform_source as export_transform\n'
        '    expected = export_transform(transform_source(original_file.read_text()))')
    old = '        return private.chunk_gated_delta_rule_fwd_h(*args, **kwargs, state_round_dtype=storage_dtype)'
    assert rounding.count(old) == 1
    rounding = rounding.replace(old, '        from .gdn_prefill_export import forward_h\n'
        '        return forward_h(private, storage_dtype, args, kwargs)')
    # Keep the adapter's outer core/conv exports and metadata; bypass only split rounds.
    start = ckpt.index('        dev = q.device\n', ckpt.index('    def forward_native('))
    end = ckpt.index('\n    _forward_core._horizon_ckpt', start)
    ckpt = ckpt[:start] + '''        return orig_native(self, q, k, v, g, beta, initial_state, output_final_state,
                           cu_seqlens, chunk_indices, chunk_offsets,
                           use_qk_l2norm_in_kernel, core_attn_out)
''' + ckpt[end:]
    ckpt = ckpt.replace('in-step prefill checkpoint for Qwen GDN (alignment %d)',
                         'single-pass prefill checkpoint for Qwen GDN (alignment %d)')
    updated = {ckpt_path: ckpt, kernel_path: kernel, round_path: rounding,
               'hk/horizon_kernels/gdn_prefill_export.py': adapter.read_text()}
    for source in updated.values():
        ast.parse(source)
    shutil.copytree(base, out, symlinks=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'offline', 'offline-output'))
    for name, source in updated.items():
        target = out/name
        if target.is_symlink():
            raise ValueError('modified source must be private')
        target.write_text(source)
        manifest['sha256'][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest['set']['PYTHONPATH'] = manifest['set']['PYTHONPATH'].replace(str(base), str(out))
    manifest['variant'] += '; single-pass exact checkpoint export'
    (out/'candidate.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'out', 'adapter'):
        p.add_argument(name, type=Path)
    a = p.parse_args(); build(a.base, a.out, a.adapter)
    print(json.dumps({'candidate': str(a.out)}))
