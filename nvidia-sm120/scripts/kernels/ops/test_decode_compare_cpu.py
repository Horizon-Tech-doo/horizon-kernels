# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from decode_check import compare_records


def fixture():
    return {part: [dict(tokens=['a','b'],lp=[-.1,-.2],prompt_tokens=42) for _ in range(n)]
            for part,n in (('seq',10),('long',3),('conc',8))}


class Tests(unittest.TestCase):
    def test_full_and_divergence_identity(self):
        a=fixture();b=copy.deepcopy(a)
        b['conc'][5]['tokens'][1]='c'
        r=compare_records(a,b)
        self.assertEqual(r['seq']['identical'],'10/10')
        self.assertEqual(r['conc']['divergence_details'],
                         [dict(prompt_index=5,token_index=1,tokens_a=2,tokens_b=2)])

    def test_rejects_incomplete_or_invalid_evidence(self):
        for mutation in (lambda b:b['conc'].pop(), lambda b:b['seq'][0]['lp'].pop(),
                         lambda b:b['seq'][0].update(tokens=[],lp=[]),
                         lambda b:b['long'][0].update(prompt_tokens=43),
                         lambda b:b['seq'][0]['lp'].__setitem__(0,float('nan'))):
            a=fixture();b=copy.deepcopy(a);mutation(b)
            with self.assertRaises(ValueError):compare_records(a,b)

    def test_missing_logprobs_not_counted_equal(self):
        a=fixture();a['seq'][0]['lp'][0]=None
        r=compare_records(a,copy.deepcopy(a))
        self.assertEqual(r['seq']['logprob_bit_equal'],'19/19')
        self.assertEqual(r['seq']['missing_logprob_pairs'],1)

if __name__=='__main__':unittest.main()
