# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile the actual CUDA helper's pure conversion body on CPU and test both lanes."""
import ast
from pathlib import Path
import re
import subprocess
import sys
import tempfile

source=Path(sys.argv[1])/'tasks.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cache_src')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
helper=re.search(r'__device__ __forceinline__ float hz_k4_bf16_pair_cvt\([^}]+\}',ns['cache_src']()).group()
helper=helper.replace('__device__ __forceinline__','inline')
program='''#include <cassert>
#include <cstring>
#include <cstdio>
float __uint_as_float(unsigned b) { float f; std::memcpy(&f,&b,4); return f; }
unsigned bits_of(float f) { unsigned b; std::memcpy(&b,&f,4); return b; }
'''+helper+'''
int main() {
  for (unsigned b=0;b<65536;++b) {
    unsigned high=b^0xa55au, pair=b|(high<<16);
    assert(bits_of(hz_k4_bf16_pair_cvt(pair,0))==(b<<16));
    assert(bits_of(hz_k4_bf16_pair_cvt(pair,1))==(high<<16));
  }
  puts("PAIR_CONVERSION_CPU_PASS 131072 lane conversions, including nonfinite bit patterns");
}
'''
with tempfile.TemporaryDirectory() as directory:
    p=Path(directory);(p/'test.cpp').write_text(program)
    subprocess.run(['c++','-std=c++17','-O2',str(p/'test.cpp'),'-o',str(p/'test')],check=True)
    subprocess.run([str(p/'test')],check=True)
