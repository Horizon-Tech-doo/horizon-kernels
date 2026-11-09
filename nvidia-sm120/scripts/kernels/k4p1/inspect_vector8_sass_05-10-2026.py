# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Disassemble frozen cubins on CPU; no CUDA modules or kernels are loaded."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root,output=map(Path,sys.argv[1:3]);output.mkdir(exist_ok=False)
tool=Path('$HK_VENV/lib/python3.12/site-packages/nvidia/cu13/bin/nvdisasm')
version=subprocess.check_output([str(tool),'--version'],text=True)
compiled=json.loads((root/'offline/result.json').read_text())['records']
assert len(compiled)==20
instruction=re.compile(r'^\s*/\*[0-9a-fA-F]+\*/\s*(?:@!?[A-Z]+\d*\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9_]+)*)\s')
assert instruction.match(' /*0010*/ @!UP2 LDG.E.ENL2.256.STRONG.GPU R4, R8;').group(1)=='LDG.E.ENL2.256.STRONG.GPU'
assert instruction.match(' /*0020*/ STG.E.128 [R0], R4;').group(1)=='STG.E.128'
assert instruction.match(' /*0020*/ .byte 0xff') is None
records=[]
for r in compiled:
    cubin=root/'offline'/(r['name']+'.cubin')
    process=subprocess.run([str(tool),'-ndf',str(cubin)],capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    target=output/(r['name']+'.sass');target.write_text(process.stdout)
    opcodes=collections.Counter(m.group(1) for line in process.stdout.splitlines() if (m:=instruction.match(line)))
    assert opcodes and '.target\tsm_120a' in process.stdout
    widths={w:sum(v for k,v in opcodes.items() if k.startswith('LDG.') and f'.{w}.' in k+'.') for w in (64,128,256)}
    store128=sum(v for k,v in opcodes.items() if k.startswith('STG.') and '.128' in k)
    assert widths[256 if r['vector']==8 else 128]==r['packed_ptx_loads']
    records.append(dict(name=r['name'],rows=r['rows'],grid=r['grid'],vector=r['vector'],
                        cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                        sass_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                        registers=r['registers'],global_load_width_histogram=widths,
                        global_store128_instructions=store128,static_sass_opcodes=dict(sorted(opcodes.items()))))
report={'decision':'STATIC_SASS_WIDER_LOAD_CONFIRMED','GPU_context_created':False,
        'tool_version':version,'tool_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),
        'documentation':'https://docs.nvidia.com/cuda/cuda-binary-utilities/',
        'scope':'Static hardware instructions only; no executed instruction counts, latency, bandwidth, race or quality evidence.',
        'records':records}
(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'decision':report['decision'],'kernels':len(records),'GPU_context_created':False}))
