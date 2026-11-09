# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import unittest
from bench_load_paired import request_body, summarize
from compare_load_paired import compare


def arm(tag,seconds=1):
    result=[]
    for rnd in range(3):
        req=[]
        for index in range(2):
            payload=json.dumps(request_body('model',index,10,29),sort_keys=True,separators=(',',':')).encode()
            req.append(dict(index=index,tokens=10,prompt_tokens=20,cached_tokens=20,
                            request_sha256=hashlib.sha256(payload).hexdigest(),response_sha256='a'*64,
                            start_s=1,end_s=1+seconds,total_s=seconds,ttft_s=seconds/10))
        r=summarize(req,2,10,[0,2],[])
        r.update(test='measurement',tag=tag,model='model',seed=29,round=rnd,requested_tokens=10)
        result.append(r)
    result.append(dict(test='complete',tag=tag,model='model',seed=29,levels=[2],rounds=3,requested_tokens=10))
    return result

class ComparisonTests(unittest.TestCase):
    def test_paired_ratios(self):
        r=compare(arm('a'),arm('b',.5))['results'][0]
        self.assertEqual(r['paired_throughput_ratios'],[2,2,2])
        self.assertTrue(r['eligible_for_paired_workload_comparison'])
    def test_reject_incomplete_or_forged_evidence(self):
        mutations=[lambda b:b.pop(),lambda b:b.pop(0),lambda b:b.__setitem__(1,copy.deepcopy(b[0])),
                   lambda b:b[0].__setitem__('total_tok_s',999),lambda b:b[0].__setitem__('traffic_guard_passed',False),
                   lambda b:b[0]['requests'][0].__setitem__('request_sha256','0'*64),
                   lambda b:b[0]['requests'][0].__setitem__('tokens',9),
                   lambda b:b[0]['requests'][0].__setitem__('end_s',float('nan')),
                   lambda b:b[0]['requests'][0].__setitem__('cached_tokens',21),
                   lambda b:b[0]['requests'][0].__setitem__('prompt_tokens',21)]
        for change in mutations:
            b=arm('b');change(b)
            with self.assertRaises(ValueError):compare(arm('a'),b)
    def test_changed_outputs_or_cache_are_diagnostic_only(self):
        for key,value in (('response_sha256','b'*64),('cached_tokens',None),('cached_tokens',0)):
            b=arm('b');b[0]['requests'][0][key]=value
            self.assertFalse(compare(arm('a'),b)['results'][0]['eligible_for_paired_workload_comparison'])
    def test_failed_traffic_guard_is_not_eligible(self):
        b=arm('b');b[0].update(peak_running=3,traffic_guard_passed=False)
        self.assertFalse(compare(arm('a'),b)['results'][0]['eligible_for_paired_workload_comparison'])

if __name__=='__main__':unittest.main()
