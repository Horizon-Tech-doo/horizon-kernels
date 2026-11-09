# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Single-pass checkpoint export for the canonical FLA GDN state operation.

Exports the fp32 carry at selected chunk starts, before the lower-precision
scratch h tensor is written. Selection follows checkpoint metadata, not a model
name. The architecture adapter still supplies conv-state exports and cache slots.
Only private candidates enable this module; ordinary prefill is unchanged.
"""


def export_plan(qsl, cuts, exports, chunk_size=64):
    if len(qsl) != len(cuts) + 1 or qsl[0] != 0:
        raise ValueError('checkpoint sequence layout mismatch')
    offsets = [0]
    for a, b in zip(qsl, qsl[1:]):
        if b <= a:
            raise ValueError('checkpoint sequences must be nonempty')
        offsets.append(offsets[-1] + (b-a+chunk_size-1)//chunk_size)
    mapping = [-1] * offsets[-1]
    for slot, (seq, cut) in enumerate(exports):
        pos = cuts[seq][cut]
        if not 0 < pos < qsl[seq+1]-qsl[seq] or pos % chunk_size:
            raise ValueError('checkpoint must be an interior aligned chunk start')
        index = offsets[seq] + pos//chunk_size
        if mapping[index] != -1:
            raise ValueError('duplicate checkpoint export')
        mapping[index] = slot
    return mapping


def transform_source(source):
    """Extend the already source-guarded canonical kernel, preserving scratch h."""
    replacements = {
        '    chunk_offsets,\n    T,': '    chunk_offsets,\n    checkpoint_map,\n    checkpoint_states,\n    EXPORT_CHECKPOINTS: tl.constexpr,\n    T,',
        '    state_round_dtype: str | None = None,\n':
            '    state_round_dtype: str | None = None,\n    checkpoint_map=None,\n    checkpoint_states=None,\n',
        '        T=T,\n': '        checkpoint_map=checkpoint_map,\n        checkpoint_states=checkpoint_states,\n        EXPORT_CHECKPOINTS=checkpoint_map is not None,\n        T=T,\n',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError('checkpoint kernel source anchor changed')
        source = source.replace(old, new)
    lines = ['        if EXPORT_CHECKPOINTS:',
             '            export = tl.load(checkpoint_map + boh + i_t).to(tl.int64)',
             '            if export >= 0:',
             '                state = checkpoint_states + (export * H + i_h) * V * K']
    for i in range(1, 5):
        indent = '                '
        if i > 1:
            lines.append(f'                if K > {(i-1)*64}:')
            indent += '    '
        lines.extend([
            indent + f'p_export = tl.make_block_ptr(state, (V, K), (K, 1), (i_v * BV, {(i-1)*64}), (BV, 64), (1, 0))',
            indent + f'tl.store(p_export, b_h{i}, boundary_check=(0, 1))',
        ])
    anchor = '    for i_t in range(NT):\n'
    if source.count(anchor) != 1:
        raise ValueError('checkpoint recurrence source changed')
    source = source.replace(anchor, anchor + '\n'.join(lines) + '\n')
    source = source.replace('    checkpoint_states,\n    EXPORT_CHECKPOINTS:',
        '    checkpoint_states,\n    checkpoint_slots,\n    CHECKPOINT_STRIDE: tl.constexpr,\n    EXPORT_CHECKPOINTS:')
    source = source.replace('    checkpoint_states=None,\n',
        '    checkpoint_states=None,\n    checkpoint_slots=None,\n')
    source = source.replace('        checkpoint_states=checkpoint_states,\n',
        '        checkpoint_states=checkpoint_states,\n        checkpoint_slots=checkpoint_slots,\n'
        '        CHECKPOINT_STRIDE=checkpoint_states.stride(0) if checkpoint_states is not None else 0,\n')
    source = source.replace('                state = checkpoint_states + (export * H + i_h) * V * K',
        '                if checkpoint_slots is not None:\n'
        '                    export = tl.load(checkpoint_slots + export).to(tl.int64)\n'
        '                state = checkpoint_states + export * CHECKPOINT_STRIDE + i_h * V * K')
    source = source.replace('tl.store(p_export, b_h1,', 'tl.store(p_export, b_h1.to(p_export.dtype.element_ty),')
    source = source.replace('tl.store(p_export, b_h2,', 'tl.store(p_export, b_h2.to(p_export.dtype.element_ty),')
    source = source.replace('tl.store(p_export, b_h3,', 'tl.store(p_export, b_h3.to(p_export.dtype.element_ty),')
    source = source.replace('tl.store(p_export, b_h4,', 'tl.store(p_export, b_h4.to(p_export.dtype.element_ty),')
    return source


def forward_h(private, storage_dtype, args, kwargs):
    from horizon_kernels import gdn_ckpt
    ck = gdn_ckpt._CUR.get('ck')
    if ck is None:
        return private.chunk_gated_delta_rule_fwd_h(*args, **kwargs, state_round_dtype=storage_dtype)
    import torch
    k = kwargs.get('k', args[0] if args else None)
    u = kwargs.get('u', args[2] if len(args) > 2 else None)
    if k.shape[0] != 1 or k.shape[1] != ck.prefill_qsl_cpu[-1]:
        raise ValueError('checkpoint token layout does not match the FLA operation')
    if not hasattr(ck, 'horizon_export_map'):
        ck.horizon_export_map = torch.tensor(export_plan(ck.prefill_qsl_cpu, ck.cuts, ck.exports),
                                            device=k.device, dtype=torch.int32)
    states = gdn_ckpt._CUR['layer'].kv_cache[1]
    H, V, K = u.shape[-2], u.shape[-1], k.shape[-1]
    if states.shape[1:] != (H,V,K) or states.stride()[1:] != (V*K,K,1):
        raise ValueError('checkpoint state cache inner dimensions must be contiguous')
    result = private.chunk_gated_delta_rule_fwd_h(*args, **kwargs,
        state_round_dtype=storage_dtype, checkpoint_map=ck.horizon_export_map,
        checkpoint_states=states, checkpoint_slots=ck.slots)
    gdn_ckpt._CUR['states'] = True
    return result
