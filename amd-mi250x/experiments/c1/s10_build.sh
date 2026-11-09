#!/bin/bash
# Can the gap between loads come from device source rather than assembly surgery? Insert an asm nop after each global->LDS copy line. JIT flags.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p s10
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null); HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
FL="-O3 -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
: > s10_tab.txt
for sn in 4k 8k 32k 17k 16k; do
python3 - "$sn" <<'P'
import re,sys
sn=sys.argv[1]
s=open(f's3src/wheel_b12864_{sn}.cu').read()
L=s.split('\n');out=[];n=0
for l in L:
    out.append(l)
    # global source: A or B argument in the right-hand side, LDS destination on the left
    if re.match(r'\s*\*\(uint4\*\)\(\(\(half_t\*\)[AB]_s\) \+.*= \*\(uint4\*\)\([AB] \+',l) or re.match(r'\s*\*\(uint4\*\)\(\(\(half_t\*\)[AB]_s\).*=.*\*\(uint4\*\)\(\(*[AB]\b',l):
        out.append('      __asm__ volatile("s_nop 0");'); n+=1
print(sn,'inserted',n)
open(f's10/srcnop_{sn}.cc','w').write('\n'.join(out))
open(f's10/srcnopa3_{sn}.cc','w').write('\n'.join(out).replace('__launch_bounds__(256)','__launch_bounds__(256, 3)'))
P
for v in srcnop srcnopa3; do ( $HC $FL -c -o s10/${v}_$sn.hsaco s10/${v}_$sn.cc && $HC $FL -S -o s10/${v}_$sn.s s10/${v}_$sn.cc ) & done; wait
done
python3 - <<'P'
import re
for v in ('srcnop','srcnopa3'):
    f=f's10/{v}_8k.s'; print(f, {k:len(re.findall(k,open(f).read())) for k in ('s_nop','global_load','v_mfma')}, re.findall(r'; (Occupancy: \d+|TotalNumVgprs: \d+|ScratchSize: \d+)',open(f).read()))
    t=open(f).read().split('s_barrier')[1].split('s_barrier')[0]
    print('\n'.join(l for l in t.split('\n') if re.match(r'\s*(global_load|s_nop|ds_write|s_waitcnt)',l))[:900])
P
