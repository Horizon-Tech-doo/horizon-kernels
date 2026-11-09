# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate existing prompt-logprob artifacts; emit hashes and exact comparisons.

Args: artifact directory, baseline tag, candidate tags. This validates recorded
prose/code arrays, not corpus provenance (the legacy collector lacks input hashes).
"""
import hashlib
import json
import math
from pathlib import Path
import sys

root=Path(sys.argv[1]);base=sys.argv[2];tags=sys.argv[3:]
expected={f'{kind}_{n}':n-1 for kind in ('prose','code') for n in (1024,4096,16384,32768)}

def read(tag):
    raw=(root/(tag+'.json')).read_bytes();doc=json.loads(raw)
    assert set(doc)==set(expected),(tag,'missing/unexpected cases')
    for key,n in expected.items():
        assert set(doc[key])=={'lp','top1'}
        assert len(doc[key]['lp'])==len(doc[key]['top1'])==n
        assert all(type(x) in (float,int) and math.isfinite(x) for x in doc[key]['lp'])
        assert all(isinstance(x,str) and x.isdigit() for x in doc[key]['top1'])
    return doc,hashlib.sha256(raw).hexdigest()

a,sha=read(base)
report={'baseline':base,'baseline_sha256':sha,'cases':sorted(expected),'tokens':sum(expected.values()),'comparisons':[]}
for tag in tags:
    b,sha=read(tag);lp=top=0;maxdiff=0
    for key in expected:
        lp+=sum(x!=y for x,y in zip(a[key]['lp'],b[key]['lp']))
        top+=sum(x!=y for x,y in zip(a[key]['top1'],b[key]['top1']))
        maxdiff=max(maxdiff,max(abs(x-y) for x,y in zip(a[key]['lp'],b[key]['lp'])))
    report['comparisons'].append(dict(tag=tag,sha256=sha,lp_differences=lp,top1_differences=top,max_abs_lp_difference=maxdiff))
print(json.dumps(report,indent=2))
