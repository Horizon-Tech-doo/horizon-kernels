#!/bin/bash
# Why does (256,3) help 128x128x64 and (256,2) not? Same source, JIT flags, asm per second argument. CPU only.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p s8
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null); HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
FL="-O3 -S -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
for a in "" ", 2" ", 3" ", 4"; do n=$(echo "$a" | tr -dc 0-9); n=${n:-0}; sed -E "s/__launch_bounds__\(256[^)]*\)/__launch_bounds__(256$a)/" s3src/wheel_b12864_8k.cu > s8/a$n.cc; $HC $FL -o s8/a$n.s s8/a$n.cc & done; wait
for n in 0 2 3 4; do f=s8/a$n.s; echo "== a$n lines=$(grep -cE '^\s+[a-z]' $f) mfma=$(grep -c v_mfma $f) ds_read=$(grep -c ds_read $f) ds_write=$(grep -c ds_write $f) gl=$(grep -c global_load $f) bar=$(grep -c s_barrier $f) waitcnt=$(grep -c s_waitcnt $f) cbr=$(grep -c s_cbranch $f) v_add=$(grep -cE 'v_add|v_lshl' $f) nop=$(grep -c s_nop $f) mov=$(grep -cE 'v_mov|v_accvgpr' $f)"; grep -E "^; (NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy):" $f | tr '\n' ' '; echo; done
python3 - <<'P'
import re
# K loop = the innermost backward-branch region holding the barrier and MFMAs. Print instruction kinds in order, run-length coded.
def seq(fn):
    L=[l.strip() for l in open(fn)]
    lab={};
    for i,l in enumerate(L):
        m=re.match(r'^(\.LBB\S+):',l)
        if m: lab[m.group(1)]=i
    best=None
    for i,l in enumerate(L):
        m=re.match(r's_cbranch\S*\s+(\.LBB\S+)',l)
        if m and m.group(1) in lab and lab[m.group(1)]<i:
            body=L[lab[m.group(1)]:i+1]
            if any('v_mfma' in b for b in body):
                if best is None or len(body)>len(best): best=body
    return best
def k(l):
    op=l.split()[0] if l else ''
    if op.startswith('v_mfma'): return 'MFMA'
    if op.startswith('global_load'): return 'GLOAD'
    if op.startswith('global_store'): return 'GSTORE'
    if op.startswith('ds_read'): return 'DSR'
    if op.startswith('ds_write'): return 'DSW'
    if op.startswith('s_barrier'): return 'BARRIER'
    if op.startswith('s_waitcnt'): return 'WAIT('+l.split()[1][:12]+')' if 'vmcnt' in l or 'lgkmcnt' in l else 'WAIT'
    if op.startswith('s_cbranch') or op.startswith('s_branch'): return 'BR'
    if op.startswith('v_accvgpr') or op.startswith('v_mov'): return 'MOV'
    if op.startswith('v_'): return 'V'
    if op.startswith('s_'): return 'S'
    return ''
for n in (0,2,3,4):
    b=seq(f's8/a{n}.s'); out=[];
    for l in b:
        t=k(l)
        if not t: continue
        if out and out[-1][0]==t: out[-1][1]+=1
        else: out.append([t,1])
    print(f'a{n} loop lines',len(b),' '.join(f"{t}{c if c>1 else ''}" for t,c in out))
P
