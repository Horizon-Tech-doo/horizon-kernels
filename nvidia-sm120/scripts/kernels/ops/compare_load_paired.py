# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate fixed-token load artifacts and report paired ratios, never promote."""
import argparse
import hashlib
import json
import math
import statistics
from bench_load_paired import request_body


def positive_int(value):
    return type(value) is int and value>0


def near(a,b):
    return type(a) in (int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-9)


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate(records):
    if not records or records[-1].get('test')!='complete':raise ValueError('Missing completion marker')
    meta=records[-1];levels=meta['levels'];rounds=meta['rounds'];tokens=meta['requested_tokens']
    if not levels or not all(positive_int(n) for n in levels) or len(set(levels))!=len(levels):raise ValueError('Invalid concurrency levels')
    if not positive_int(rounds) or not positive_int(tokens):raise ValueError('Invalid rounds/token limit')
    if type(meta['seed']) is not int or not isinstance(meta['model'],str) or not meta['model']:raise ValueError('Invalid request configuration')
    expected={(n,r) for n in levels for r in range(rounds)}
    rows=records[:-1]
    if len(rows)!=len(expected):raise ValueError('Incomplete or duplicate measurements')
    indexed={}
    for row in rows:
        if row.get('test')!='measurement':raise ValueError('Unexpected record')
        if any(row[k]!=meta[k] for k in ('tag','model','seed','requested_tokens')):raise ValueError('Mixed arm configuration')
        n=row['concurrency'];identity=(n,row['round'])
        if identity not in expected or identity in indexed:raise ValueError('Missing/duplicate trial identity')
        reqs=row['requests']
        if len(reqs)!=n or {r['index'] for r in reqs}!=set(range(n)) or row['completed']!=n:raise ValueError('Missing/duplicate requests')
        for r in reqs:
            payload=json.dumps(request_body(meta['model'],r['index'],tokens,meta['seed']),sort_keys=True,separators=(',',':')).encode()
            if r['request_sha256']!=hashlib.sha256(payload).hexdigest() or not digest(r['response_sha256']):raise ValueError('Request/response fingerprint invalid')
            if r['tokens']!=tokens or not positive_int(r['prompt_tokens']):raise ValueError('Wrong token count')
            cached=r.get('cached_tokens')
            if cached is not None and (type(cached) is not int or not 0<=cached<=r['prompt_tokens']):raise ValueError('Invalid cache count')
            if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) for k in ('start_s','end_s','ttft_s','total_s')):raise ValueError('Invalid timing')
            if not 0<r['ttft_s']<=r['total_s'] or not near(r['total_s'],r['end_s']-r['start_s']):raise ValueError('Inconsistent request timing')
        span=max(r['end_s'] for r in reqs)-min(r['start_s'] for r in reqs)
        if row['tokens']!=n*tokens or not near(row['wall_s'],span) or not near(row['total_tok_s'],n*tokens/span):raise ValueError('Inconsistent throughput')
        if not near(row['ttft_median_s'],statistics.median(r['ttft_s'] for r in reqs)):raise ValueError('Inconsistent TTFT')
        peak=row.get('peak_running');errors=row['metric_errors']
        guard=(type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=n and errors==[])
        if row['traffic_guard_passed'] is not guard:raise ValueError('Inconsistent traffic guard')
        indexed[identity]=row
    return meta,indexed


def compare(a,b):
    ma,aa=validate(a);mb,bb=validate(b)
    if any(ma[k]!=mb[k] for k in ('model','seed','levels','rounds','requested_tokens')):raise ValueError('Arm configurations differ')
    if ma['tag']==mb['tag']:raise ValueError('Arm tags must differ')
    result=[]
    for n in ma['levels']:
        speed=[];ttft=[];same_output=True;matched_cache=True;guard=True
        for rnd in range(ma['rounds']):
            x,y=aa[n,rnd],bb[n,rnd]
            speed.append(y['total_tok_s']/x['total_tok_s'])
            ttft.append(x['ttft_median_s']/y['ttft_median_s'])
            guard &= x['traffic_guard_passed'] and y['traffic_guard_passed']
            xx={r['index']:r for r in x['requests']};yy={r['index']:r for r in y['requests']}
            for index in xx:
                u,v=xx[index],yy[index]
                if u['request_sha256']!=v['request_sha256'] or u['prompt_tokens']!=v['prompt_tokens']:raise ValueError('Workload inputs differ')
                same_output &= u['response_sha256']==v['response_sha256']
                matched_cache &= u.get('cached_tokens') is not None and u.get('cached_tokens')==v.get('cached_tokens')
        result.append(dict(concurrency=n,paired_throughput_ratios=speed,
                           median_throughput_ratio=statistics.median(speed),median_ttft_speed_ratio=statistics.median(ttft),
                           exact_output_hashes=same_output,matched_reported_cache=matched_cache,traffic_guard_passed=guard,
                           eligible_for_paired_workload_comparison=bool(same_output and matched_cache and guard)))
    return dict(baseline=ma['tag'],candidate=mb['tag'],results=result,
                limitation='Sampled traffic guard needs an exclusive test window; interleaved arm repeats, runtime identity and model quality gates are still required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline');parser.add_argument('candidate');args=parser.parse_args()
    def read(path):return [json.loads(s) for s in open(path)]
    print(json.dumps(compare(read(args.baseline),read(args.candidate)),indent=2))
