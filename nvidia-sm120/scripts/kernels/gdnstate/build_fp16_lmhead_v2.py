#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Private optional exact-head screen on the validated fp16 candidate; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def build(base,out,head_package):
    base,out,head_package=map(Path,(base,out,head_package))
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a fresh candidate')
    manifest=json.loads((base/'candidate.json').read_text())
    for name,sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'base changed: {name}')
    sources=[head_package/'lmhead_screen.py',head_package/'lmscreen_ops/__init__.py',
             head_package/'lmscreen_ops/lmhead_screen.py']
    if not all(s.is_file() for s in sources): raise ValueError('exact-head sources missing')
    shutil.copytree(base,out,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for source in sources:
        target=out/'hk/horizon_kernels'/source.relative_to(head_package)
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        manifest['sha256'][str(target.relative_to(out))]=hashlib.sha256(target.read_bytes()).hexdigest()
    registration = {
        'hk/horizon_kernels/lmhead_screen_entry.py': 'import os\n\ndef register():\n    if os.environ.get("HORIZON_LMHEAD_SCREEN", "0") == "1":\n        from . import lmhead_screen\n        lmhead_screen.install()\n',
        'hk/horizon_exact_head-0.1.dist-info/METADATA': 'Metadata-Version: 2.1\nName: horizon-exact-head\nVersion: 0.1\n',
        'hk/horizon_exact_head-0.1.dist-info/entry_points.txt': '[vllm.general_plugins]\nhorizon_exact_head = horizon_kernels.lmhead_screen_entry:register\n',
    }
    for name, source in registration.items():
        target=out/name; target.parent.mkdir(exist_ok=True); target.write_text(source)
        manifest['sha256'][name]=hashlib.sha256(source.encode()).hexdigest()
    manifest['set']['PYTHONPATH']=manifest['set']['PYTHONPATH'].replace(str(base),str(out))
    manifest['set'].update(HORIZON_LMHEAD_SCREEN='1',HORIZON_LMHEAD_SCREEN_TOPK='0',
        HORIZON_LMHEAD_SCREEN_SELFTEST=str(out/'selftest.json'),
        HORIZON_LMHEAD_SCREEN_SELFTEST_H='$HK_WORK/formats/s1/mainh_000.pt',
        HORIZON_LMHEAD_SCREEN_LOG_EVERY='200')
    manifest['variant']+='; optional exact lm_head screen (memory gate required)'
    (out/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','out','head_package'): p.add_argument(name,type=Path)
    a=p.parse_args(); build(a.base,a.out,a.head_package)
    print(json.dumps(dict(candidate=str(a.out))))
