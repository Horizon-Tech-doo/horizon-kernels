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
    hook=(base/'hk/horizon_kernels/vllm_plugin.py').read_text()
    if 'lmhead_screen.install()' not in hook:
        raise ValueError('base plugin lacks the default-off exact-head hook')
    sources=[head_package/'lmhead_screen.py',head_package/'lmscreen_ops/__init__.py',
             head_package/'lmscreen_ops/lmhead_screen.py']
    if not all(s.is_file() for s in sources): raise ValueError('exact-head sources missing')
    shutil.copytree(base,out,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for source in sources:
        target=out/'hk/horizon_kernels'/source.relative_to(head_package)
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        manifest['sha256'][str(target.relative_to(out))]=hashlib.sha256(target.read_bytes()).hexdigest()
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
