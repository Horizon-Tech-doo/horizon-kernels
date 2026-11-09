# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU schedule contracts and legacy descriptor regression (no GPU imports).

Arguments: candidate segment.py, previous segment.py, runtime.py.
"""
import ast
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

candidate, previous, runtime = map(Path, sys.argv[1:4])
del sys.argv[1:4]

def extract(path, names, ns):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

rns = extract(runtime, {'Schedule'}, {'NCTA_DEFAULT':188})
base = dict(H=5120,I=17408,KO=6144,NQ=16384,NBA=96,DEFAULT_GEOMETRY=(5120,17408,6144),
            DEFAULT_GDN_HEADS=(16,48),GH_K=16,GH_V=48,GD=128,GC=10240,CONV_CH=256,GDN_VR=32,
            tk=SimpleNamespace(BLOCK_N=128,BLOCK_K=64),rt=SimpleNamespace(Schedule=rns['Schedule']))
base.update(zip(('OP_OUT','OP_N2','OP_GU','OP_SILU','OP_DOWN','OP_N1','OP_QKVZ','OP_BA','OP_CONV','OP_CORE','OP_GNORM'),range(1,12)))
new = extract(candidate, {'gdn_geometry','segment_geometry','validate_splits','build_schedule'}, dict(base))
old = extract(previous, {'build_schedule'}, dict(base))

class SegmentGeometryTests(unittest.TestCase):
    def test_legacy_identical(self):
        for m in (1,4,16,32):
            for nq in (16384,14336):
                for requests in (0,1,3):
                    args = (m,376,(4,1,8,4),(4,1,128),nq,requests)
                    a,b = new['build_schedule'](*args),old['build_schedule'](*args)
                    self.assertEqual(a.lists,b.lists)
                    self.assertEqual(a.targets,b.targets)
                    a.check()

    def test_generic_split_coverage_and_progress(self):
        for nba in (0,80):
            sc = new['build_schedule'](4,32,(4,2,8,4),(4,1,128),12288,
                                       geometry=(4096,14336,4096),nba=nba)
            sc.check()
            flat = [d for queue in sc.lists for d in queue]
            gu = [d for d in flat if d[0] == base['OP_GU']]
            self.assertEqual({(d[1],d[2]) for d in gu},{(b,s) for b in range(224) for s in range(2)})
            self.assertEqual(len(gu),448)
            self.assertEqual(sum(d[0]==base['OP_BA'] for d in flat),nba)
            # Simulate one task per CTA per sweep; split outputs signal only on
            # final completion of that tile, regardless of CTA completion order.
            cursors=[0]*sc.ncta
            counters=Counter(); partial=defaultdict(set)
            splits={base['OP_OUT']:4,base['OP_GU']:2,base['OP_DOWN']:8,base['OP_QKVZ']:4}
            completed=0
            while completed < len(flat):
                before=completed
                for c,queue in enumerate(sc.lists):
                    if cursors[c] == len(queue): continue
                    d=queue[cursors[c]]
                    if any(w>=0 and counters[w]<t for w,t in ((d[4],d[5]),(d[6],d[7]))): continue
                    op,bx,bk=d[:3]
                    if op in splits:
                        partial[op,bx].add(bk)
                        signal=len(partial[op,bx])==splits[op]
                    else: signal=True
                    if signal and d[8]>=0: counters[d[8]]+=1
                    cursors[c]+=1;completed+=1
                self.assertGreater(completed,before,'schedule deadlock')
            for counter,target in sc.targets.items():
                self.assertAlmostEqual(counters[counter],target)

    def test_slice_order_task_conservation_and_progress(self):
        for geometry,nq,nba,splits in (((5120,17408,6144),16384,-1,(4,1,8,4)),
                                      ((4096,14336,4096),12288,80,(4,2,8,4))):
            for m in (4,32):
                for ncta in (1,32,188,376):
                    kwargs=dict(geometry=geometry,nba=nba)
                    a=new['build_schedule'](m,ncta,splits,(4,1,128),nq,**kwargs)
                    b=new['build_schedule'](m,ncta,splits,(4,1,128),nq,slice_order=True,**kwargs)
                    self.assertEqual(Counter(d for q in a.lists for d in q),Counter(d for q in b.lists for d in q))
                    b.check()
                    for order in (range(ncta),range(ncta-1,-1,-1)):
                        cursor=[0]*ncta;counts=Counter();partial=defaultdict(set)
                        sp=dict(zip((base['OP_OUT'],base['OP_GU'],base['OP_DOWN'],base['OP_QKVZ']),splits))
                        target=sum(map(len,b.lists));done=0
                        while done<target:
                            before=done
                            for c in order:
                                if cursor[c]==len(b.lists[c]):continue
                                d=b.lists[c][cursor[c]]
                                if any(w>=0 and counts[w]<t for w,t in ((d[4],d[5]),(d[6],d[7]))):continue
                                op,bx,bk=d[:3]
                                if op in sp:
                                    partial[op,bx].add(bk)
                                    signal=len(partial[op,bx])==sp[op]
                                else:signal=True
                                if signal and d[8]>=0:counts[d[8]]+=1
                                cursor[c]+=1;done+=1
                            self.assertGreater(done,before,'sliced schedule deadlock')
                        for c,n in b.targets.items():self.assertAlmostEqual(counts[c],n)

    def test_gdn_heads_drive_all_schedule_tasks(self):
        self.assertEqual(new['gdn_geometry']((8,32),4096,4),(8,32,6144,10240))
        sc=new['build_schedule'](8,376,(4,2,8,4),(4,1,128),12288,R=2,
                                 geometry=(4096,14336,4096),nba=0,gdn_heads=(8,32))
        sc.check()
        flat=[d for q in sc.lists for d in q]
        counts=Counter(d[0] for d in flat)
        self.assertEqual(counts[base['OP_CONV']],2*24)
        self.assertEqual(counts[base['OP_CORE']],2*32*4)
        self.assertEqual(counts[base['OP_GNORM']],2*32)
        out=[d for d in flat if d[0]==base['OP_OUT']]
        self.assertTrue(all(d[5]==16 for d in out))  # two requests * eight heads/split
        self.assertEqual({d[2] for d in out},{0,1,2,3})
        for heads,ko,split in (((0,32),4096,4),((3,32),4096,4),((8,32),6144,4),
                               ((8,32),4096,3),((1,1),128,1)):
            with self.assertRaises(ValueError):new['gdn_geometry'](heads,ko,split)

    def test_explicit_auxiliary(self):
        self.assertFalse(new['segment_geometry']((4096,14336,4096),16384,0)[4])
        self.assertTrue(new['segment_geometry']((4096,14336,4096),12288,80)[4])

    def test_reject_invalid_geometry_and_splits(self):
        for shape in ((0,128,64),(4097,14336,4096),(4096,14337,4096),(4096,14336,4097)):
            with self.assertRaises(ValueError):new['segment_geometry'](shape)
        for splits in ((4,0,8,4),(3,2,8,4),(4,2,3,4)):
            with self.assertRaises(ValueError):new['validate_splits']((4096,14336,4096),splits)
        with self.assertRaises(ValueError):
            new['build_schedule'](4,32,(4,2,8,4),(4,1,128),12288,R=1,geometry=(4096,14336,4096))

if __name__=='__main__':unittest.main()
