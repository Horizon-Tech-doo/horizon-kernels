# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from fp16_capacity_floor import check

class Floor(unittest.TestCase):
    def test_complete_resident_evidence_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'probe.jsonl'
            row=dict(test='probe',context=4096,level=88,clean=True,peak_decode_overlap=88,
                     out_tokens=1024,requests_ok=88,finished_delta=88,preemptions=0,errors=0,wrong_token_counts=0)
            final=dict(test='probe_result',context=4096,max_clean_concurrency=88)
            def save(items):p.write_text('\n'.join(json.dumps(r) for r in items))
            save([row,final]);self.assertEqual(check(p,4096,88)['result'],'pass')
            for field,value in [('peak_decode_overlap',87),('preemptions',1),('finished_delta',89),('wrong_token_counts',1)]:
                save([dict(row,**{field:value}),final]);self.assertEqual(check(p,4096,88)['result'],'fail')
            save([final]);self.assertEqual(check(p,4096,88)['result'],'fail')
            save([row]);self.assertEqual(check(p,4096,88)['result'],'fail')
            self.assertEqual(check(Path(tmp)/'absent',4096,88)['result'],'fail')

if __name__=='__main__':unittest.main()
