# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Run on CPU with serving Python; imports only quantizer definitions via AST.
Args: candidate mtp_quant.py, original mtp_quant.py.
"""
import ast
import os
from pathlib import Path
import sys
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK']='1'
import torch
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0


def load(path):
    tree=ast.parse(Path(path).read_text())
    selected=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='quantize_nvfp4_rtn')
              or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_E2M1_POS' for t in n.targets))]
    ns={'torch':torch};exec(compile(ast.Module(body=selected,type_ignores=[]),path,'exec'),ns)
    return ns['quantize_nvfp4_rtn']
new,old=map(load,sys.argv[1:3]);del sys.argv[1:3]

class QuantizerTests(unittest.TestCase):
    def test_zero_blocks_with_denormal_flushing(self):
        try:
            for flush in (False, True):
                if not torch.set_flush_denormal(flush):
                    self.skipTest('CPU does not support denormal flushing')
                for magnitude in (0., 1e-35, .03):
                    w=torch.full((2,256),magnitude,dtype=torch.bfloat16)
                    w[0,:128]=0
                    packed,scales,alpha=new(w)
                    self.assertEqual(torch.count_nonzero(packed[0,:64]).item(),0)
                    sf=scales.float()*alpha[:,None]
                    self.assertTrue(torch.isfinite(sf).all())
                    self.assertTrue((sf>=torch.finfo(torch.float32).tiny).all())
        finally:
            torch.set_flush_denormal(False)
    def test_zero_weights_are_finite_exact_zero(self):
        packed,scales,alpha=new(torch.zeros(4,256,dtype=torch.bfloat16))
        self.assertTrue(torch.isfinite(scales.float()).all())
        self.assertTrue(torch.isfinite(alpha).all())
        self.assertTrue((alpha>0).all())
        self.assertEqual(torch.count_nonzero(packed).item(),0)
    def test_typical_weights_unchanged(self):
        generator=torch.Generator().manual_seed(2909)
        for magnitude in (1e-6,.03,1,100):
            w=(torch.randn(4,256,generator=generator)*magnitude).to(torch.bfloat16)
            for a,b in zip(new(w),old(w)):
                self.assertTrue(torch.equal(a.view(torch.uint8),b.view(torch.uint8)))
    def test_zero_blocks_and_extremely_small_weights(self):
        for magnitude in (0.,1e-40,1e-35):
            w=torch.full((2,256),magnitude,dtype=torch.bfloat16);w[0,:128]=0
            packed,scales,alpha=new(w)
            self.assertTrue(torch.isfinite(scales.float()).all())
            self.assertTrue(torch.isfinite(alpha).all())
            self.assertTrue((alpha>0).all())
            self.assertEqual(torch.count_nonzero(packed[0,:64]).item(),0)
    def test_original_zero_case_reproduces_bug(self):
        _,scales,_=old(torch.zeros(2,256,dtype=torch.bfloat16))
        self.assertFalse(torch.isfinite(scales.float()).all())

if __name__=='__main__':unittest.main()
