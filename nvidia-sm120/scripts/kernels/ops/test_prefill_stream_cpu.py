# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from unittest.mock import patch
from bench_prefill_paired import measure, prompt

class Stream:
    def __init__(self, events):self.events=events
    def __enter__(self):return iter(self.events)
    def __exit__(self,*a):pass

def event(x):return ('data: '+json.dumps(x)+'\n').encode()

class Tests(unittest.TestCase):
    def test_matched_prompts(self):
        self.assertEqual(prompt(100,0,29),prompt(100,0,29))
        self.assertNotEqual(prompt(100,0,29),prompt(100,1,29))

    def test_empty_delta_is_not_first_token(self):
        events=[event({'choices':[{'delta':{'content':''}}]}),
                event({'choices':[{'delta':{'content':'answer'}}]}),
                event({'choices':[{'delta':{},'finish_reason':'length'}]}),
                event({'usage':{'prompt_tokens':100,'prompt_tokens_details':{'cached_tokens':0}}}),b'data: [DONE]\n']
        with patch('urllib.request.urlopen',return_value=Stream(events)),patch('time.perf_counter',side_effect=[10.,12.]):
            r=measure('http://example.invalid','model','text')
        self.assertEqual(r['ttft_s'],2.)
        self.assertEqual(r['cache_evidence'],'cold')

    def test_incomplete_stream_rejected(self):
        with patch('urllib.request.urlopen',return_value=Stream([event({'choices':[{'delta':{'content':'x'}}]})])):
            with self.assertRaises(RuntimeError):measure('http://example.invalid','model','text')

if __name__=='__main__':unittest.main()
