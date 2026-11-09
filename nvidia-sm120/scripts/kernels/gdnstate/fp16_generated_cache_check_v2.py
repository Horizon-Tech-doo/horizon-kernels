#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Cached vs recomputed chat history after a long generated assistant answer."""
import argparse
import json
from pathlib import Path
import urllib.request
import uuid


def ask(url,messages,salt,max_tokens,logprobs=False):
    body=dict(model='qwen3.8',messages=messages,cache_salt=salt,max_tokens=max_tokens,
              temperature=0,ignore_eos=True,chat_template_kwargs=dict(enable_thinking=False))
    if logprobs: body.update(logprobs=True,top_logprobs=5)
    request=urllib.request.Request(url.rstrip('/')+'/v1/chat/completions',
        json.dumps(body).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response: return json.load(response)


def run(url,output):
    rows=[]
    for repeats in (80,100,200,350):
        source='\n'.join(f'def operation_{i}(items):\n    return sum(x*x for x in items) + {i}\n'
                          for i in range(repeats))
        prompt=('Write a comprehensive tutorial of at least 2500 words about this Python module. '
                'Explain types, edge cases, examples, alternatives and tests in detail.\n\n'+source)
        messages=[dict(role='user',content=prompt)]
        salt='wp13-generated-'+uuid.uuid4().hex
        primer=ask(url,messages,salt,1200)
        text=primer['choices'][0]['message'].get('content') or ''
        if not text: raise ValueError('generated conversation history is empty')
        followup=messages+[dict(role='assistant',content=text),
            dict(role='user',content='Summarize your previous answer in five specific recommendations.')]
        cold=ask(url,followup,salt+'-cold',64,True)
        warm=ask(url,followup,salt,64,True)
        cc,wc=cold['choices'][0],warm['choices'][0]
        cl,wl=cc.get('logprobs'),wc.get('logprobs')
        if not cl or not wl or not cl.get('content') or not wl.get('content'):
            raise ValueError('generated history check requires token log probabilities')
        pu=primer['usage']; cu=cold['usage']; wu=warm['usage']
        cached=(wu.get('prompt_tokens_details') or {}).get('cached_tokens',0)
        row=dict(repeats=repeats,primer_prompt_tokens=pu['prompt_tokens'],
            primer_output_tokens=pu['completion_tokens'],followup_prompt_tokens=wu['prompt_tokens'],
            cached_tokens=cached,reused_generated_tokens=max(0,cached-pu['prompt_tokens']),
            cold_cached_tokens=(cu.get('prompt_tokens_details') or {}).get('cached_tokens',0),
            text_exact=cc['message'].get('content')==wc['message'].get('content'),
            logprobs_exact=cl==wl)
        rows.append(row); print(json.dumps(row),flush=True)
        output.write_text(json.dumps(rows,indent=2)+'\n')
    if not all(r['text_exact'] and r['logprobs_exact'] and r['cold_cached_tokens']==0
               and r['cached_tokens']>0 for r in rows):
        raise RuntimeError('generated-history check requires four exact comparisons with warm hits; inspect row coverage')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    p.add_argument('--url',default='http://127.0.0.1:8001'); a=p.parse_args()
    run(a.url,a.output)
