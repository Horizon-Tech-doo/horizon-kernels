# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Require complete candidate K4 evidence, including timeout recovery, before in-vLLM testing."""
import argparse
import json
import math
from pathlib import Path


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evaluate(rows, variant='ra', sizes=(4, 8, 16, 32), min_iters=150):
    errors = []
    if not rows or rows[-1].get('part') != 'END':
        errors.append('missing final END record (test did not finish)')
    if sum(r.get('part') == 'END' for r in rows) != 1:
        errors.append('expected exactly one complete run')
    for size in sizes:
        for part in ('A', 'A2', 'B', 'C'):
            hits = [r for r in rows if r.get('part') == part and r.get('M') == size
                    and (part == 'B' or r.get('variant', r.get('ncta')) == variant)]
            if len(hits) != 1:
                errors.append(f'{part} M={size}: expected one candidate record, found {len(hits)}')
                continue
            r = hits[0]
            if part == 'A':
                if not (r.get('ok') is True and r.get('bit_identical') is True
                        and r.get('err') == 0 and r.get('not_reset') == 0):
                    errors.append(f'A M={size}: correctness/counter reset failed')
            elif part == 'A2':
                if not (r.get('bit_identical') is True and r.get('err') == 0):
                    errors.append(f'A2 M={size}: attention-next correctness failed')
            elif part == 'B':
                medians = r.get('us_per_layer_median', {})
                if (type(r.get('iters')) is not int or r['iters'] < min_iters
                        or not all(positive(medians.get(k)) for k in ('p1_pdl0', 'p1_policy', 'k4_seg_' + variant))):
                    errors.append(f'B M={size}: incomplete/invalid paired timing')
            elif not positive(r.get('span_us')) or not positive(r.get('ncta')):
                errors.append(f'C M={size}: incomplete/invalid HKT profile')
    guards = [r for r in rows if r.get('part') == 'G']
    if len(guards) != 1:
        errors.append('expected exactly one timeout-recovery test')
    else:
        guard = guards[0]
        checks = ('ok', 'host_flag_mapped', 'failed', 'step_check_raised', 'recovered_bit_identical')
        if not all(guard.get(k) is True for k in checks):
            errors.append('timeout guard or recovery failed')
        seen, returned = guard.get('host_saw_flag_after_s'), guard.get('kernel_returned_after_s')
        if not positive(seen) or not positive(returned) or returned < seen:
            errors.append('missing/invalid timeout observation times')
    return {'variant': variant, 'sizes': list(sizes), 'pass': not errors, 'errors': errors,
            'all_variants_ok': rows[-1].get('ok_all') if rows and rows[-1].get('part') == 'END' else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path', type=Path)
    p.add_argument('variant', nargs='?', default='ra')
    p.add_argument('--sizes', default='4,8,16,32')
    p.add_argument('--min-iters', type=int, default=150)
    a = p.parse_args()
    try:
        rows = [json.loads(line) for line in a.path.read_text().splitlines() if line.strip().startswith('{')]
        if not all(isinstance(r, dict) for r in rows):
            raise ValueError('non-object result record')
        verdict = evaluate(rows, a.variant, tuple(map(int, a.sizes.split(','))), a.min_iters)
    except (OSError, ValueError, TypeError) as e:
        verdict = {'pass': False, 'errors': [str(e)]}
    print(f"seg gate {a.variant}: {'PASS' if verdict['pass'] else 'FAIL'}")
    print(json.dumps(verdict, allow_nan=False))
    return 0 if verdict['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
