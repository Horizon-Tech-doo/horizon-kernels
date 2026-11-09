#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exact-head checks for explicit hz labels, without assuming control is P."""
import argparse
import json
from pathlib import Path


def load(data, name):
    try:
        return json.loads((data/name).read_text())
    except (OSError, ValueError):
        return {}


def compare(data, candidates, base=None, memory_prefix='mem'):
    gates = {}
    def gate(name, ok, observed):
        gates[name] = dict(result='pass' if ok else 'fail', observed=observed)
    for arm in candidates:
        st=load(data,f'selftest-{arm}.json'); s=st.get('summary',{}); rows=st.get('per_M',{})
        complete=s.get('M_tested')==256 and set(rows)=={str(i) for i in range(1,257)}
        exact=complete and s.get('M_ok')==256 and s.get('M_not_ok')==[] and all(
            all(row.get(key) is True for key in ('values_bitwise','argmax_equal','top20_inside','top20_values_bitwise'))
            for row in rows.values())
        gate(f'selftest_{arm}', exact, dict(M_tested=s.get('M_tested'),M_ok=s.get('M_ok'),complete=complete))
        c=load(data,f'counters-{arm}.json'); cn=c.get('counters') or {}
        used=sum(cn.get(k,0) for k in ('mode:greedy','mode:topk') if type(cn.get(k,0)) is int)
        gate(f'screen_used_{arm}', c.get('hooks_line_found') is True and used>0, dict(screened_steps=used))
        mem=load(data,f'{memory_prefix}-{arm}.json'); free=mem.get('real_free_mib')
        gate(f'memory_{arm}', type(free) is int and free>=1536 and mem.get('smokes') is True, mem)
        if base:
            x,p=load(data,f'greedy-{arm}.json').get('seq',[]),load(data,f'greedy-{base}.json').get('seq',[])
            exact_seq=len(x)==len(p)==10 and all(a.get('toks')==b.get('toks') and a.get('toks') for a,b in zip(x,p))
            gate(f'greedy_{arm}', bool(exact_seq), dict(candidate=len(x),baseline=len(p),base=base))
            x,p=load(data,f'acc-{arm}.json'),load(data,f'acc-{base}.json')
            delta={k:x[k]['mean_accept_len']-p[k]['mean_accept_len'] for k in ('prose','maths','code')
                   if isinstance(x.get(k),dict) and isinstance(p.get(k),dict)
                   and isinstance(x[k].get('mean_accept_len'),(int,float)) and isinstance(p[k].get('mean_accept_len'),(int,float))}
            gate(f'acceptance_{arm}', len(delta)==3 and all(v>=-.02 for v in delta.values()), delta)
    return dict(gates=gates,summary_md='Exact output-layer evidence for '+','.join(candidates))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--candidates',default='X');p.add_argument('--base');p.add_argument('--memory-prefix',default='mem')
    a=p.parse_args();a.output.write_text(json.dumps(compare(a.data,a.candidates.split(','),a.base,a.memory_prefix),indent=2)+'\n')
