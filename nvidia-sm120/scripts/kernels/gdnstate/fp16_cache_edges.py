#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic exact-token cache checks around hash, page and prefill boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid


def post(url,path,body):
    req=urllib.request.Request(url.rstrip('/')+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response: return json.load(response)


def signature(response):
    choice=response['choices'][0]
    lp=choice.get('logprobs')
    if not lp or not lp.get('token_logprobs') or any(x is None for x in lp['token_logprobs']):
        raise ValueError('cache check requires generated-token log probabilities')
    return choice['text'],lp


def run(url,output):
    ids=post(url,'/tokenize',dict(model='qwen3.8',prompt=' '.join(
        ['Consider these integers and write a short explanation of their sum: 17 23 11 29.']*1700)))['tokens']
    cases=(512,768,832,896,960,1024,1536,1600,1664,4096,4160,8192,8320,14336,14400)
    if len(ids)<max(cases)+137: raise ValueError('synthetic token buffer too short')
    rows=[]
    for n in cases:
        salt='wp13-edges-'+uuid.uuid4().hex
        def request(prompt,cache_salt):
            return post(url,'/v1/completions',dict(model='qwen3.8',prompt=prompt,cache_salt=cache_salt,
                max_tokens=64,ignore_eos=True,temperature=0,logprobs=5))
        cold=request(ids[:n],salt)
        warm=request(ids[:n],salt)
        extension=ids[:n]+ids[-137:]
        extended_cold=request(extension,salt+'-cold-extension')
        extended_warm=request(extension,salt)
        row=dict(prompt_tokens=n,resend_exact=signature(cold)==signature(warm),
            extend_exact=signature(extended_cold)==signature(extended_warm),
            resend_hit=warm['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),
            extend_hit=extended_warm['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),
            cold_hit=cold['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),
            cold_extension_hit=extended_cold['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0),
            resend_digest=hashlib.sha256(json.dumps(signature(warm),sort_keys=True).encode()).hexdigest())
        rows.append(row); print(json.dumps(row),flush=True)
        output.write_text(json.dumps(rows,indent=2)+'\n')
    if not all(r['resend_exact'] and r['extend_exact'] and r['cold_hit']==0
               and r['cold_extension_hit']==0 for r in rows):
        raise RuntimeError('synthetic cache-boundary equivalence failed')
    # Longer prompts must exercise cache reuse, not a silent no-hit fallback.
    if not all(r['resend_hit']>0 and r['extend_hit']>0 for r in rows if r['prompt_tokens']>=1664):
        raise RuntimeError('cache check missed a required reusable long-prompt boundary')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    p.add_argument('--url',default='http://127.0.0.1:8001'); a=p.parse_args()
    run(a.url,a.output)
