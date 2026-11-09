#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Build direct cache exports without modifying the running one-pass candidate."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def build(base,out,conv):
    base,out,conv=map(Path,(base,out,conv))
    if out.exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError('use a fresh candidate')
    manifest=json.loads((base/'candidate.json').read_text())
    for name,sha in manifest['sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'base changed: {name}')
    hk='hk/horizon_kernels/'
    kernel=(base/(hk+'gdn_round_ops/chunk_delta_h.py')).read_text()
    def replace(source,old,new):
        if source.count(old)!=1: raise ValueError(f'direct-export source anchor changed: {old[:70]}')
        return source.replace(old,new)
    kernel=replace(kernel,'    checkpoint_states,\n    EXPORT_CHECKPOINTS:',
        '    checkpoint_states,\n    checkpoint_slots,\n    CHECKPOINT_STRIDE: tl.constexpr,\n    EXPORT_CHECKPOINTS:')
    kernel=replace(kernel,'    checkpoint_states=None,\n',
        '    checkpoint_states=None,\n    checkpoint_slots=None,\n')
    kernel=replace(kernel,'        checkpoint_states=checkpoint_states,\n',
        '        checkpoint_states=checkpoint_states,\n        checkpoint_slots=checkpoint_slots,\n'
        '        CHECKPOINT_STRIDE=checkpoint_states.stride(0) if checkpoint_states is not None else 0,\n')
    kernel=replace(kernel,'                state = checkpoint_states + (export * H + i_h) * V * K',
        '                if checkpoint_slots is not None:\n'
        '                    export = tl.load(checkpoint_slots + export).to(tl.int64)\n'
        '                state = checkpoint_states + export * CHECKPOINT_STRIDE + i_h * V * K')
    helper=(base/(hk+'gdn_prefill_export.py')).read_text()
    # Matching transform for the pinned vendor source check.
    old='    return source.replace(anchor, anchor + \'\\n\'.join(lines) + \'\\n\')'
    new='''    source = source.replace(anchor, anchor + '\\n'.join(lines) + '\\n')
    source = source.replace('    checkpoint_states,\\n    EXPORT_CHECKPOINTS:',
        '    checkpoint_states,\\n    checkpoint_slots,\\n    CHECKPOINT_STRIDE: tl.constexpr,\\n    EXPORT_CHECKPOINTS:')
    source = source.replace('    checkpoint_states=None,\\n',
        '    checkpoint_states=None,\\n    checkpoint_slots=None,\\n')
    source = source.replace('        checkpoint_states=checkpoint_states,\\n',
        '        checkpoint_states=checkpoint_states,\\n        checkpoint_slots=checkpoint_slots,\\n'
        '        CHECKPOINT_STRIDE=checkpoint_states.stride(0) if checkpoint_states is not None else 0,\\n')
    return source.replace('                state = checkpoint_states + (export * H + i_h) * V * K',
        '                if checkpoint_slots is not None:\\n'
        '                    export = tl.load(checkpoint_slots + export).to(tl.int64)\\n'
        '                state = checkpoint_states + export * CHECKPOINT_STRIDE + i_h * V * K')'''
    helper=replace(helper,old,new)
    start=helper.index('    states = torch.empty(')
    helper=helper[:start]+'''    states = gdn_ckpt._CUR['layer'].kv_cache[1]
    H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
    if states.shape[1:] != (H,V,K) or states.stride()[1:] != (V*K,K,1):
        raise ValueError('checkpoint state cache inner dimensions must be contiguous')
    result = private.chunk_gated_delta_rule_fwd_h(*args, **kwargs,
        state_round_dtype=storage_dtype, checkpoint_map=ck.horizon_export_map,
        checkpoint_states=states, checkpoint_slots=ck.slots)
    gdn_ckpt._CUR['states'] = True
    return result
'''
    ckpt=(base/(hk+'gdn_ckpt.py')).read_text()
    start=ckpt.index('        rows = torch.tensor([r - width')
    end=ckpt.index('        _CUR.update(',start)
    ckpt=ckpt[:start]+'''        from .gdn_checkpoint_conv import export_conv
        if not hasattr(ck, 'horizon_abs_rows'):
            ck.horizon_abs_rows = torch.tensor(ck.abs_rows,device=mixed_qkv.device,dtype=torch.int64)
        export_conv(mixed_qkv,cs,ck.slots,ck.horizon_abs_rows,width)
'''+ckpt[end:]
    start=ckpt.index('            if states is None:')
    end=ckpt.index('            return out',start)
    ckpt=ckpt[:start]+'''            if states is not True:
                raise RuntimeError("horizon GDN checkpoint: direct state export did not run")
'''+ckpt[end:]
    ckpt=ckpt.replace('single-pass prefill checkpoint for Qwen GDN','direct-cache single-pass prefill checkpoint for Qwen GDN')
    # Triton block-pointer stores require an explicit value dtype conversion.
    for i in range(1,5):
        old=f'tl.store(p_export, b_h{i}, boundary_check=(0, 1))'
        new=f'tl.store(p_export, b_h{i}.to(p_export.dtype.element_ty), boundary_check=(0, 1))'
        kernel=replace(kernel,old,new)
    helper=replace(helper,'    return source.replace(\'                state = checkpoint_states',
                         '    source = source.replace(\'                state = checkpoint_states')
    end=helper.index('\n\n\ndef forward_h')
    casts="\n".join(f"    source = source.replace('tl.store(p_export, b_h{i},', 'tl.store(p_export, b_h{i}.to(p_export.dtype.element_ty),')" for i in range(1,5))
    helper=helper[:end]+"\n"+casts+"\n    return source"+helper[end:]
    updates={hk+'gdn_round_ops/chunk_delta_h.py':kernel,hk+'gdn_prefill_export.py':helper,
             hk+'gdn_ckpt.py':ckpt,hk+'gdn_checkpoint_conv.py':conv.read_text()}
    for source in updates.values(): ast.parse(source)
    shutil.copytree(base,out,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name,source in updates.items():
        if (out/name).is_symlink(): raise ValueError('modified file must be private')
        (out/name).write_text(source)
        manifest['sha256'][name]=hashlib.sha256(source.encode()).hexdigest()
    manifest['set']['PYTHONPATH']=manifest['set']['PYTHONPATH'].replace(str(base),str(out))
    manifest['variant']+='; fused conv history and direct recurrent-state cache exports'
    (out/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','out','conv'): p.add_argument(name,type=Path)
    a=p.parse_args(); build(a.base,a.out,a.conv)
    print(json.dumps(dict(candidate=str(a.out))))
