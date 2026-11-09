# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only capability failures must prevent K4 execution, not weaken its guard."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

path=Path(__file__).resolve().parents[3]/'horizon_kernels'/'k4seg.py'
tree=ast.parse(path.read_text())
names={'_install_step_check','_watch','install','_gdn_layer','warm','_seg_impl'}


def namespace():
    ns={'_STATE':{'disabled':False,'watchdog':None,'flags':{}},'_log':lambda:Mock(),
        '_wrap_get_output':Mock(),
        'torch':types.SimpleNamespace(Tensor=object,cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return ns

class GuardCapabilityTests(unittest.TestCase):
    def test_missing_output_hooks_disables(self):
        ns=namespace()
        with patch('importlib.import_module',side_effect=ImportError('missing')):
            self.assertFalse(ns['_install_step_check']())
        self.assertTrue(ns['_STATE']['disabled'])
        ns['_wrap_get_output'].assert_not_called()
    def test_existing_output_hook_installed(self):
        ns=namespace();cls=type('AsyncOutput',(),{'get_output':lambda self:'tokens'})
        with patch('importlib.import_module',return_value=types.SimpleNamespace(AsyncOutput=cls,AsyncGPUModelRunnerOutput=cls)):
            self.assertTrue(ns['_install_step_check']())
        self.assertFalse(ns['_STATE']['disabled'])
        self.assertEqual(ns['_wrap_get_output'].call_count,2)
    def test_device_only_flag_rejected_without_gpu_read(self):
        ns=namespace();flag=types.SimpleNamespace(mapped=False,get=Mock(side_effect=AssertionError('GPU sync')))
        self.assertFalse(ns['_watch'](types.SimpleNamespace(errh=flag)))
        self.assertTrue(ns['_STATE']['disabled'])
        self.assertEqual(ns['_STATE']['flags'],{})
        flag.get.assert_not_called()
    def test_mapped_flag_registered(self):
        ns=namespace();watchdog=Mock();ns['_STATE']['watchdog']=watchdog
        flag=types.SimpleNamespace(mapped=True,dptr=123,get=lambda:0)
        self.assertTrue(ns['_watch'](types.SimpleNamespace(errh=flag)))
        self.assertIs(ns['_STATE']['flags']['123'],flag)
        watchdog.add.assert_called_once_with('123',flag)
    def test_failed_warmup_uses_per_op_path_before_segment_launch(self):
        ns=namespace()
        def warm(*args):
            ns['_STATE']['disabled']=True
            return False
        fallback=Mock(return_value=('safe projection','safe auxiliary','safe residual'))
        segment=Mock(side_effect=AssertionError('unprotected segment launch'))
        ns.update(_WARMED=set(),_layer=lambda *args:object(),warm=warm,
                  _k4_ready=lambda *args:not ns['_STATE']['disabled'],_fallback=fallback,_segment=segment)
        tensor=types.SimpleNamespace(shape=(4,5120),device='cuda')
        self.assertEqual(ns['_seg_impl'](tensor,tensor,0,0,1),fallback.return_value)
        segment.assert_not_called();fallback.assert_called_once()

    def test_install_returns_before_registration_without_hook(self):
        ns=namespace();model=type('Model',(),{'forward':lambda:None})
        ns.update(enabled=lambda:True,_marker=lambda:'unused',os=types.SimpleNamespace(path=types.SimpleNamespace(exists=lambda _:False)),_install_step_check=lambda:False)
        module=types.ModuleType('vllm.model_executor.models.qwen3_5');module.Qwen3_5Model=model
        # Fake torch has no ops: any custom-op registration fails the test.
        with patch.dict('sys.modules',{'vllm.model_executor.models.qwen3_5':module}):ns['install']()

if __name__=='__main__':unittest.main()
