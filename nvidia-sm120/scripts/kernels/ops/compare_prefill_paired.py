# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare complete matched-prompt arms; report speed without auto-promotion."""
import argparse
import hashlib
import json
import math
import statistics
from bench_prefill_paired import prompt


def compare(a, b, lengths=(1000,4000,16000,64000), trials=3, seed=29):
    expected={(words,trial) for words in lengths for trial in range(trials)}
    if not expected or trials < 1:raise ValueError('Need positive trial count and lengths')
    arms=[]
    for records in (a,b):
        if len(records)!=len(expected):raise ValueError('Incomplete or duplicate measurements')
        indexed={(r['words'],r['trial']):r for r in records}
        if set(indexed)!=expected:raise ValueError('Missing or duplicate trial identity')
        if len({r['tag'] for r in records})!=1:raise ValueError('Mixed arm tags')
        for (words,trial),r in indexed.items():
            digest=hashlib.sha256(prompt(words,trial,seed).encode()).hexdigest()
            if r['seed']!=seed or r['prompt_sha256']!=digest:raise ValueError('Prompt fixture mismatch')
            if not isinstance(r['prompt_tokens'],int) or r['prompt_tokens'] <= 0:raise ValueError('Invalid token count')
            if not math.isfinite(r['ttft_s']) or r['ttft_s'] <= 0:raise ValueError('Invalid TTFT')
            if r.get('cached_tokens') is not None and (not isinstance(r['cached_tokens'],int) or r['cached_tokens']<0):
                raise ValueError('Invalid cached token count')
        arms.append(indexed)
    results=[]
    for words in lengths:
        pairs=[(arms[0][words,t],arms[1][words,t]) for t in range(trials)]
        if any(x['prompt_tokens']!=y['prompt_tokens'] for x,y in pairs):
            raise ValueError('Tokenized prompt lengths differ between arms')
        cold=all(r.get('cached_tokens')==0 for pair in pairs for r in pair)
        results.append(dict(words=words,trials=trials,
            baseline_median_ttft_s=statistics.median(x['ttft_s'] for x,y in pairs),
            candidate_median_ttft_s=statistics.median(y['ttft_s'] for x,y in pairs),
            median_paired_speed_ratio=statistics.median(x['ttft_s']/y['ttft_s'] for x,y in pairs),
            cold_cache_verified=cold, eligible_for_cold_cache_comparison=cold))
    return dict(complete_matched_inputs=True, results=results,
                limitation='Sequential arm timing; interleaved repeats and quality gates still required.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline');p.add_argument('candidate')
    p.add_argument('--lengths',default='1000,4000,16000,64000')
    p.add_argument('--trials',type=int,default=3);p.add_argument('--seed',type=int,default=29)
    args=p.parse_args()
    def load(path):
        with open(path) as f:return [json.loads(line) for line in f]
    print(json.dumps(compare(load(args.baseline),load(args.candidate),
          tuple(map(int,args.lengths.split(','))),args.trials,args.seed),indent=2))

if __name__=='__main__':main()
