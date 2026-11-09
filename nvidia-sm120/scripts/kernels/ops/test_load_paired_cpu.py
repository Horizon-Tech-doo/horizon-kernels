# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import copy
import io
import json
import unittest
from unittest.mock import patch, Mock
import bench_load_paired as b

class LoadTests(unittest.TestCase):
    def rows(self):
        return [dict(index=i,tokens=10,prompt_tokens=20,start_s=1+i*.01,end_s=2+i*.01,ttft_s=.1,total_s=1) for i in range(2)]
    def test_no_partial_or_invalid_success(self):
        rows=self.rows()
        for bad in (rows[:1],rows+[rows[0]],[rows[0],rows[0]]):
            with self.assertRaises(ValueError):b.summarize(bad,2,10,[2],[])
        for key,value in (('tokens',9),('ttft_s',float('nan')),('ttft_s',0),('total_s',.01)):
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):b.summarize(bad,2,10,[2],[])
    def test_isolation_requires_valid_metrics(self):
        for observations,errors in (([],[]),([2],['TimeoutError']),([3],[])):
            self.assertFalse(b.summarize(self.rows(),2,10,observations,errors)['traffic_guard_passed'])
        self.assertTrue(b.summarize(self.rows(),2,10,[0,2,1],[])['traffic_guard_passed'])
    def test_fixed_requests(self):
        self.assertEqual(b.request_body('m',3,600,29),b.request_body('m',3,600,29))
        self.assertTrue(b.request_body('m',3,600,29)['ignore_eos'])
        self.assertNotEqual(b.request_body('m',3,600,29),b.request_body('m',4,600,29))
    def test_stream_requires_length_and_exact_usage(self):
        def events(finish='length',tokens=10):
            values=[{'choices':[{'delta':{'role':'assistant'},'finish_reason':None}]},
                    {'choices':[{'delta':{'content':'answer'},'finish_reason':None}]},
                    {'choices':[{'delta':{},'finish_reason':finish}]},
                    {'choices':[],'usage':{'completion_tokens':tokens,'prompt_tokens':12}}]
            return io.BytesIO((''.join('data: '+json.dumps(v)+'\n' for v in values)+'data: [DONE]\n').encode())
        for finish,tokens,valid in (('length',10,True),('stop',10,False),('length',9,False)):
            with patch.object(b.urllib.request,'urlopen',return_value=events(finish,tokens)),patch.object(b.time,'perf_counter',side_effect=[1,1.2,2]):
                if valid:
                    row=b.measure('http://localhost/test',b.request_body('m',0,10,29),Mock())
                    self.assertAlmostEqual(row['ttft_s'],.2)
                    self.assertEqual(row['tokens'],10)
                    self.assertEqual(len(row['request_sha256']),64)
                else:
                    with self.assertRaises(RuntimeError):b.measure('http://localhost/test',b.request_body('m',0,10,29),Mock())

    def test_thread_failure_propagates(self):
        with patch.object(b,'metrics',return_value=0),patch.object(b,'measure',side_effect=RuntimeError('request failed')):
            with self.assertRaisesRegex(RuntimeError,'request failed'):
                b.run_level('url','metrics','model',2,10,29)

if __name__=='__main__':unittest.main()
