# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import unittest
from bench_prefill_paired import prompt
from compare_prefill_paired import compare


def fixture(tag='a'):
    return [dict(tag=tag,words=1000,trial=t,seed=29,
                 prompt_sha256=hashlib.sha256(prompt(1000,t,29).encode()).hexdigest(),
                 prompt_tokens=1036,ttft_s=2.,cached_tokens=0) for t in range(3)]

class Tests(unittest.TestCase):
    def test_paired_ratio(self):
        a,b=fixture(),fixture('b')
        for r in b:r['ttft_s']=1.
        r=compare(a,b,(1000,))['results'][0]
        self.assertEqual(r['median_paired_speed_ratio'],2.)
        self.assertTrue(r['cold_cache_verified'])

    def test_unknown_or_hit_cache_not_eligible(self):
        for cached in (None,32):
            b=fixture('b');b[0]['cached_tokens']=cached
            self.assertFalse(compare(fixture(),b,(1000,))['results'][0]['eligible_for_cold_cache_comparison'])

    def test_incomplete_mismatched_invalid(self):
        mutations=[lambda b:b.pop(),lambda b:b.__setitem__(1,copy.deepcopy(b[0])),
                   lambda b:b[0].update(prompt_sha256='wrong'),lambda b:b[0].update(prompt_tokens=1037),
                   lambda b:b[0].update(ttft_s=float('nan')),lambda b:b[0].update(tag='mixed')]
        for mutate in mutations:
            b=fixture('b');mutate(b)
            with self.assertRaises(ValueError):compare(fixture(),b,(1000,))

if __name__=='__main__':unittest.main()
