# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Controlled fixed-token load benchmark; run only in an exclusive test window.

Same request hashes/seed across arms. Warm-up uses identical prompts at 32 output
tokens. Missing traffic metrics or extra running requests fail the sampled traffic guard.
Polling cannot prove the absence of other traffic; an exclusive test window is required.
No deployment/configuration changes. Output lengths are forced, not natural chat.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from bench_prefill_paired import parse_event

PROMPTS=(
    'Explain the difference between a VPN and a proxy for a small business.',
    'Write a story about a lighthouse keeper who finds a message in a bottle.',
    'Explain step by step how to implement a thread-safe bounded queue.',
    'Write a Python ISO-8601 parser with type hints and tests.',
    'Discuss the benefits and costs of remote work for twenty engineers.',
    'Explain TCP congestion control with concrete examples.',
    'Describe how a database index changes read and write performance.',
    'Write a TypeScript HTTP route that validates a JSON request.',
)


def request_body(model,index,tokens,seed):
    return dict(model=model,messages=[dict(role='user',content=PROMPTS[index%len(PROMPTS)])],
                max_tokens=tokens,ignore_eos=True,temperature=0,seed=seed+index,
                stream=True,stream_options=dict(include_usage=True),
                chat_template_kwargs=dict(enable_thinking=False))


def measure(url,body,barrier):
    payload=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
    request=urllib.request.Request(url,payload,{'Content-Type':'application/json'})
    barrier.wait(timeout=60)
    start=time.perf_counter();first=None;usage=None;finish=None;content=[]
    with urllib.request.urlopen(request,timeout=900) as response:
        for line in response:
            event=parse_event(line)
            if event is None:continue
            if event.get('usage'):usage=event['usage']
            for choice in event.get('choices',[]):
                text=choice.get('delta',{}).get('content')
                if isinstance(text,str) and text:
                    if first is None:first=time.perf_counter()
                    content.append(text)
                if choice.get('finish_reason') is not None:finish=choice['finish_reason']
    end=time.perf_counter()
    if first is None or finish!='length' or not usage:
        raise RuntimeError('Incomplete/non-length stream')
    if usage.get('completion_tokens')!=body['max_tokens'] or usage.get('prompt_tokens',0)<=0:
        raise RuntimeError('Token count mismatch or missing prompt usage')
    return dict(request_sha256=hashlib.sha256(payload).hexdigest(),
                response_sha256=hashlib.sha256(''.join(content).encode()).hexdigest(),
                start_s=start,end_s=end,ttft_s=first-start,total_s=end-start,
                tokens=usage['completion_tokens'],prompt_tokens=usage['prompt_tokens'],
                cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'))


def metrics(url):
    with urllib.request.urlopen(url,timeout=5) as response:
        text=response.read().decode()
    values=[float(line.split()[-1]) for line in text.splitlines()
            if line.startswith('vllm:num_requests_running{') or line.startswith('vllm:num_requests_running ')]
    if not values or not all(math.isfinite(v) and v>=0 for v in values):
        raise RuntimeError('Missing/invalid running-request metric')
    return sum(values)


def summarize(rows,concurrency,tokens,observations,metric_errors):
    if len(rows)!=concurrency or {r['index'] for r in rows}!=set(range(concurrency)):
        raise ValueError('Missing or duplicate requests')
    for r in rows:
        if r['tokens']!=tokens or any(not math.isfinite(r[k]) for k in ('start_s','end_s','ttft_s','total_s')):
            raise ValueError('Invalid measurement')
        if not 0<r['ttft_s']<=r['total_s'] or r['end_s']<=r['start_s']:
            raise ValueError('Invalid timing')
    span=max(r['end_s'] for r in rows)-min(r['start_s'] for r in rows)
    return dict(concurrency=concurrency,completed=len(rows),tokens=sum(r['tokens'] for r in rows),
                wall_s=span,total_tok_s=sum(r['tokens'] for r in rows)/span,
                ttft_median_s=statistics.median(r['ttft_s'] for r in rows),
                traffic_guard_passed=bool(observations) and not metric_errors and max(observations)<=concurrency,
                peak_running=max(observations) if observations else None,metric_errors=metric_errors,
                requests=sorted(rows,key=lambda r:r['index']))


def run_level(url,metrics_url,model,concurrency,tokens,seed):
    observations=[];errors=[];stop=threading.Event()
    def poll():
        while True:
            try:observations.append(metrics(metrics_url))
            except Exception as exc:errors.append(type(exc).__name__)
            if stop.wait(.25):return
    watcher=threading.Thread(target=poll,daemon=True);watcher.start()
    barrier=threading.Barrier(concurrency)
    def one(index):
        row=measure(url,request_body(model,index,tokens,seed),barrier)
        row['index']=index
        return row
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows=list(pool.map(one,range(concurrency)))  # propagates every request failure
    finally:
        stop.set();watcher.join(timeout=6)
    if watcher.is_alive():errors.append('MetricsWatcherTimeout')
    return summarize(rows,concurrency,tokens,observations,errors)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag');p.add_argument('--output',required=True)
    p.add_argument('--url',default='http://localhost:8000/v1/chat/completions')
    p.add_argument('--metrics-url',default='http://localhost:8000/metrics')
    p.add_argument('--model',default='qwen3.8');p.add_argument('--levels',default='1,8,32')
    p.add_argument('--tokens',type=int,default=600);p.add_argument('--seed',type=int,default=29)
    p.add_argument('--rounds',type=int,default=3);a=p.parse_args()
    levels=[int(x) for x in a.levels.split(',')]
    if min(levels+[a.tokens,a.rounds])<1 or len(set(levels))!=len(levels):p.error('positive distinct levels and positive tokens/rounds required')
    with open(a.output,'w') as output:
        for n in levels:
            warm=run_level(a.url,a.metrics_url,a.model,n,32,a.seed)
            if not warm['traffic_guard_passed']:raise RuntimeError('Warm-up traffic guard failed')
            for round_index in range(a.rounds):
                row=run_level(a.url,a.metrics_url,a.model,n,a.tokens,a.seed)
                row.update(test='measurement',tag=a.tag,model=a.model,round=round_index,seed=a.seed,requested_tokens=a.tokens)
                output.write(json.dumps(row)+'\n');output.flush()
                print(json.dumps({k:v for k,v in row.items() if k!='requests'}),flush=True)
                if not row['traffic_guard_passed']:raise RuntimeError('Traffic guard failed')
        output.write(json.dumps(dict(test='complete',tag=a.tag,model=a.model,seed=a.seed,levels=levels,rounds=a.rounds,requested_tokens=a.tokens))+'\n')

if __name__=='__main__':main()
