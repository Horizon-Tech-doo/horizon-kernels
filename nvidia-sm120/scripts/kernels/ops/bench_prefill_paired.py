# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Matched-prompt TTFT measurement. Run each arm on a fresh server/cache.

Same seed produces the same prompts across arms. Distinct trial prefixes avoid
within-arm reuse. Records reported cache usage; missing usage is not proof of a
cold cache. No environment changes, cache resets, or deployment are performed.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request

WORDS = ('azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue model '
         'tensor memory bandwidth latency throughput scheduler token prompt decode cache layer').split()


def prompt(words, trial, seed):
    r = random.Random(f'{seed}:{words}:{trial}')
    return f'[{r.getrandbits(128):032x}] Summarize in one sentence: ' + ' '.join(r.choice(WORDS) for _ in range(words))


def parse_event(line):
    line = line.decode().strip()
    if not line.startswith('data:') or line[5:].strip() == '[DONE]':
        return None
    event = json.loads(line[5:])
    if 'error' in event:
        raise RuntimeError(f'Server stream error: {event["error"]}')
    return event


def measure(url, model, text):
    body = dict(model=model, messages=[dict(role='user', content=text)], max_tokens=8,
                temperature=0, stream=True, stream_options=dict(include_usage=True),
                chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type':'application/json'})
    start = time.perf_counter(); first = None; usage = None; finished = False
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            event = parse_event(line)
            if event is None:
                continue
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                content = choice.get('delta', {}).get('content')
                if first is None and isinstance(content, str) and content:
                    first = time.perf_counter()-start
                if choice.get('finish_reason') is not None:
                    finished = True
    if first is None or not finished or usage is None or usage.get('prompt_tokens',0) <= 0:
        raise RuntimeError('Incomplete stream: need nonempty content, finish reason and prompt usage')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    return dict(prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), prompt_tokens=usage['prompt_tokens'],
                ttft_s=first, prefill_tok_s=usage['prompt_tokens']/first, cached_tokens=cached,
                cache_evidence='unknown' if cached is None else ('cold' if cached == 0 else 'cache_hit'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag');p.add_argument('--output',required=True)
    p.add_argument('--url',default='http://localhost:8000/v1/chat/completions')
    p.add_argument('--model',default='qwen3.8');p.add_argument('--seed',type=int,default=29)
    p.add_argument('--lengths',default='1000,4000,16000,64000');p.add_argument('--trials',type=int,default=3)
    a=p.parse_args();lengths=[int(n) for n in a.lengths.split(',')]
    if a.trials < 1 or any(n < 1 for n in lengths):p.error('lengths and trials must be positive')
    measure(a.url,a.model,prompt(200,-1,a.seed))
    with open(a.output,'w') as out:
        for words in lengths:
            runs=[]
            for trial in range(a.trials):
                r=measure(a.url,a.model,prompt(words,trial,a.seed))
                r.update(tag=a.tag,words=words,trial=trial,seed=a.seed)
                out.write(json.dumps(r)+'\n');out.flush();runs.append(r)
            print(json.dumps(dict(tag=a.tag,words=words,
                 median_ttft_s=statistics.median(r['ttft_s'] for r in runs),
                 median_prefill_tok_s=statistics.median(r['prefill_tok_s'] for r in runs),
                 cache_evidence=[r['cache_evidence'] for r in runs])),flush=True)

if __name__=='__main__':main()
