# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from fp16_lmhead_compare import compare

class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        self.write('selftest-X.json',dict(summary=dict(M_tested=256,M_ok=256,M_not_ok=[]),per_M={str(i):dict(values_bitwise=True,argmax_equal=True,top20_inside=True,top20_values_bitwise=True) for i in range(1,257)}))
        self.write('counters-X.json',dict(hooks_line_found=True,counters={'mode:greedy':2}))
        self.write('mem-X.json',dict(real_free_mib=2000,smokes=True))
        for arm in ('X','C'):
            self.write(f'greedy-{arm}.json',dict(seq=[dict(toks=[i,1]) for i in range(10)]))
            self.write(f'acc-{arm}.json',{k:dict(mean_accept_len=2.) for k in ('prose','maths','code')})
    def write(self,name,obj): (self.p/name).write_text(json.dumps(obj))
    def test_control_C_not_P(self):
        self.assertTrue(all(g['result']=='pass' for g in compare(self.p,['X'],'C')['gates'].values()))
        self.assertEqual(compare(self.p,['X'],'P')['gates']['greedy_X']['result'],'fail')
    def test_missing_M_or_wrong_values_fail(self):
        s=json.loads((self.p/'selftest-X.json').read_text());s['per_M']['256']['values_bitwise']=False;self.write('selftest-X.json',s)
        self.assertEqual(compare(self.p,['X'])['gates']['selftest_X']['result'],'fail')
        s['per_M'].pop('256');self.write('selftest-X.json',s)
        self.assertEqual(compare(self.p,['X'])['gates']['selftest_X']['result'],'fail')
    def test_missing_arm_and_low_memory_fail(self):
        self.write('mem-X.json',dict(real_free_mib=1535,smokes=True))
        g=compare(self.p,['X','X2'])['gates'];self.assertEqual(g['memory_X']['result'],'fail')
        self.assertTrue(all(v['result']=='fail' for k,v in g.items() if k.endswith('_X2')))

if __name__=='__main__': unittest.main()
