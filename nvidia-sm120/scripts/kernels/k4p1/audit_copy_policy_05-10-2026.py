# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Inspect deployed CUDA copy defaults without a CUDA context or kernel launch."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
root = Path(TILELANG_TEMPLATE_PATH)
output = Path(sys.argv[1])
output.mkdir(exist_ok=False)
source = '#include "tl_templates/cuda/copy.h"\n'
with tempfile.TemporaryDirectory() as tmp:
    cu = Path(tmp)/'probe.cu'
    cu.write_text(source)
    command = [get_nvcc_compiler(), '-E', '-std=c++20', '-arch=sm_120a', '-I'+str(root), '-I'+CUTLASS_INCLUDE_DIR, str(cu)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stderr
    instructions = sorted(set(re.findall(r'cp\.async\.(?:cg|ca)\.shared\.global[^"\n]*', process.stdout)))
    assert instructions and all('L2::' not in x for x in instructions), instructions
package = root.parent
factories = {}
for name in ('decode_regfrag.py', 'decode_regfrag_xl.py'):
    path = package/'contrib/sm120_nvfp4'/name
    tree = ast.parse(path.read_text())
    decorators = [ast.unparse(d) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for d in n.decorator_list if 'tilelang.jit' in ast.unparse(d)]
    assert decorators and all('COMPILE_FLAGS' not in d and 'PREFETCH' not in d for d in decorators)
    factories[name] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'jit_decorators':decorators}
backend = package/'cuda/backend.py'
backend_ast = ast.parse(backend.read_text())
options = [ast.unparse(n.value) for n in ast.walk(backend_ast) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='options' for t in n.targets)]
assert options and all('PREFETCH' not in x for x in options)
report = {'decision':'NO_DEFAULT_L2_PREFETCH_DIFFERENCE', 'gpu_context_created':False,
 'scope':'Deployed template preprocessed with backend default flags; production RF/XL decorators contain no compile-flag override. This is static evidence, not a performance measurement.',
 'template':str(root/'tl_templates/cuda/copy.h'), 'template_sha256':hashlib.sha256((root/'tl_templates/cuda/copy.h').read_bytes()).hexdigest(),
 'default_compile_options':options, 'preprocessed_copy_instructions':instructions, 'production_factories':factories,
 'next_action':'Do not attribute the K4 gap to a missing production L2 hint. Any added hint would be a new tuning experiment requiring a fresh GPU window.'}
(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
