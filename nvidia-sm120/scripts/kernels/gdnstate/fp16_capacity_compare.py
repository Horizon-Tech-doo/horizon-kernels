#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Report measured capacity lower bounds, preserving the capacity goal."""
import argparse
import json
from pathlib import Path


def compare(data,output,base):
    results,gates={},{}
    for context in (4096,32768):
        arms={}
        for arm in ('X',base):
            try:
                rows=[json.loads(l) for l in (data/f'capacity{context}-{arm}.jsonl').read_text().splitlines() if l.strip()]
                summary=rows[-1]
                if summary.get('test')!='probe_result': raise ValueError('incomplete capacity probe')
                arms[arm]=dict(summary=summary,levels=[r for r in rows if r.get('test')=='probe'])
            except (OSError,ValueError,IndexError):
                arms[arm]={}
        x=arms['X'].get('summary',{}).get('max_clean_concurrency')
        p=arms[base].get('summary',{}).get('max_clean_concurrency')
        valid=type(x) is int and type(p) is int and x>0 and p>0
        results[str(context)]=dict(candidate=x,baseline=p,ratio=x/p if valid else None,details=arms)
        gates[f'capacity_{context}_not_worse']=dict(result='pass' if valid and x>=p else 'fail',
            params=dict(min_ratio=1),observed=dict(candidate=x,baseline=p))
    output.write_text(json.dumps(dict(gates=gates,numbers=results,
        summary_md='Measured full-context resident concurrency lower bounds: '+
        ', '.join(f'{c}: {r["baseline"]} → {r["candidate"]}' for c,r in results.items())),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('data',type=Path)
    p.add_argument('output',type=Path); p.add_argument('--base',default='C')
    a=p.parse_args(); compare(a.data,a.output,a.base)
