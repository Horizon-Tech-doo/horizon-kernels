#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Require the combined package to preserve previously measured resident bounds."""
import argparse
import json
from pathlib import Path


def check(path, context, minimum):
    try:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        final=rows[-1]
        evidence=[r for r in rows[:-1] if r.get('test')=='probe' and r.get('context')==context
                  and type(r.get('level')) is int and r['level']>=minimum and r.get('clean') is True
                  and r.get('peak_decode_overlap')==r['level'] and r.get('requests_ok')==r['level']
                  and r.get('finished_delta')==r['level'] and r.get('preemptions')==0
                  and r.get('out_tokens')==1024 and r.get('errors')==0 and r.get('wrong_token_counts')==0]
        ok=(final.get('test')=='probe_result' and final.get('context')==context
            and type(final.get('max_clean_concurrency')) is int
            and final['max_clean_concurrency']>=minimum and bool(evidence))
        observed=dict(summary=final,verified_levels=[r['level'] for r in evidence])
    except (OSError,ValueError,IndexError,TypeError):
        ok=False;observed='missing or invalid capacity evidence'
    return dict(result='pass' if ok else 'fail',params=dict(minimum=minimum),observed=observed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--arms',default='X');p.add_argument('--minimum-4k',type=int,default=88)
    p.add_argument('--minimum-32k',type=int,default=24);a=p.parse_args()
    gates={f'resident_{context}_{arm}':check(a.data/f'capacity{context}-{arm}.jsonl',context,minimum)
           for arm in a.arms.split(',') for context,minimum in ((4096,a.minimum_4k),(32768,a.minimum_32k))}
    a.output.write_text(json.dumps(dict(gates=gates,summary_md='Preserve measured resident capacity bounds'),indent=2)+'\n')
