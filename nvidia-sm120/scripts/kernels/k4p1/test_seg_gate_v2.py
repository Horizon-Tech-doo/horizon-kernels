# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from seg_gate_v2 import evaluate


def complete():
    rows=[]
    for m in (4,8,16,32):
        rows += [dict(part='A',M=m,ncta='ra',ok=True,bit_identical=True,err=0,not_reset=0),
                 dict(part='A2',M=m,variant='ra',bit_identical=True,err=0),
                 dict(part='B',M=m,iters=150,us_per_layer_median={'p1_pdl0':130.,'p1_policy':128.,'k4_seg_ra':125.}),
                 dict(part='C',M=m,variant='ra',ncta=376,span_us=120.)]
    rows += [dict(part='G',ok=True,host_flag_mapped=True,failed=True,step_check_raised=True,
                  recovered_bit_identical=True,host_saw_flag_after_s=.1,kernel_returned_after_s=.11),
             dict(part='END',ok_all=True)]
    return rows


class Gate(unittest.TestCase):
    def test_complete_candidate(self):
        self.assertTrue(evaluate(complete())['pass'])

    def test_original_crash_partial_output_is_rejected(self):
        self.assertFalse(evaluate(complete()[:3])['pass'])

    def test_missing_shape_or_stage(self):
        for i in range(16):
            with self.subTest(i=i):
                rows=complete(); rows.pop(i)
                self.assertFalse(evaluate(rows)['pass'])

    def test_timeout_guard_required(self):
        for key in ('ok','host_flag_mapped','failed','step_check_raised','recovered_bit_identical'):
            with self.subTest(key=key):
                rows=complete(); rows[-2][key]=False
                self.assertFalse(evaluate(rows)['pass'])

    def test_failures_not_hidden_by_ok_flag(self):
        for key,value in [('err',1),('not_reset',1),('bit_identical',False)]:
            rows=complete(); rows[0][key]=value
            self.assertFalse(evaluate(rows)['pass'])

    def test_bad_timings_and_duplicates(self):
        rows=complete(); rows[2]['us_per_layer_median']['k4_seg_ra']=float('nan')
        self.assertFalse(evaluate(rows)['pass'])
        rows=complete(); rows[2]['iters']=1
        self.assertFalse(evaluate(rows)['pass'])
        rows=complete(); rows.insert(0,copy.deepcopy(rows[0]))
        self.assertFalse(evaluate(rows)['pass'])

    def test_other_variant_failure_does_not_disqualify_candidate(self):
        rows=complete(); rows.insert(0,dict(part='A',M=4,ncta='experimental',ok=False))
        rows[-1]['ok_all']=False
        self.assertTrue(evaluate(rows)['pass'])

if __name__=='__main__':unittest.main()
