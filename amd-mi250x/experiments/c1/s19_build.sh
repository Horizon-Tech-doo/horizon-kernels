#!/bin/bash
# C1 s19 build, in the container, on LUMI. Harness with allocation modes; xnack- objects of rr25632w2 and b25632a2 from the JIT C of s9; arm tables.
R=/scratch/project_XXXXXXXXX/tilelang; cd $R/exp/C1; mkdir -p s19
TPL=$(CUDA_VISIBLE_DEVICES= python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null); HC=$(CUDA_VISIBLE_DEVICES= python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
echo HC $HC
$HC -O2 -std=c++17 hbm4.cpp -o hbm4 -lhipblaslt 2>&1 | grep -E "error" | head
FL="-O3 -gline-tables-only --offload-arch=gfx90a:xnack- --genco -std=c++17 -I$TPL"
for sn in 8k 17k 16k; do for k in rr25632w2 b25632a2; do
  $HC $FL -c -o s19/$k.xn_$sn.hsaco s9/${k}_$sn.cc 2>/dev/null || echo FAIL $k $sn
  $HC $FL -S -o s19/$k.xn_$sn.s s9/${k}_$sn.cc 2>/dev/null || echo FAILS $k $sn
done; done
python3 - <<'P'
import os,re
A=os.path.abspath; N4=A('../N4/h11')
SH={'8k':(8192,8192,8192,20),'17k':(4096,17408,5120,20),'16k':(16384,16384,16384,5)}
def base(sn,M,N,K,it,arms):
    T=[f"hipblaslt_direct - - {M} {N} {K} 0 0 0 0 {it}"]
    for a in arms:
        if a in ('rr25632w2.asis','b25632a2.asis'):
            k=a.split('.')[0]; f=f's12/{a}_{sn}'
        elif a.endswith('.xn'):
            k=a.split('.')[0]; f=f's19/{a}_{sn}'
        else:
            continue
        kn=re.search(r'^([A-Za-z_0-9]+):\s+; @',open(f+'.s').read(),re.M).group(1)
        T.append(f"{a}_{sn} {A(f+'.hsaco')} {kn} {M} {N} {K} 256 128 32 24576 {it}")
    return T
M8,N8,K8,i8=SH['8k']; M7,N7,K7,i7=SH['17k']; M6,N6,K6,i6=SH['16k']
main=base('8k',M8,N8,K8,i8,['rr25632w2.asis','b25632a2.asis','rr25632w2.xn','b25632a2.xn'])+[f"h11_c8_8k {N4}/h11_c8_8k.co h11_c8_8k {M8} {N8} {K8} 256 128 64 61440 {i8}"]
main+=base('17k',M7,N7,K7,i7,['rr25632w2.asis','b25632a2.asis','rr25632w2.xn','b25632a2.xn'])+[f"h11_c8_17k {N4}/h11_c8_17k.co h11_c8_17k {M7} {N7} {K7} 256 128 64 61440 {i7}"]
big=base('16k',M6,N6,K6,i6,['b25632a2.asis','b25632a2.xn'])+[f"h11_o128_16k {N4}/h11_o128_16k.co h11_o128_16k {M6} {N6} {K6} 128 256 64 61440 {i6}"]
open('s19_tab.txt','w').write('\n'.join(main)+'\n'); open('s19_big.txt','w').write('\n'.join(big)+'\n')
smoke=[l for l in main if ' 8192 8192 8192 ' in l]; open('s19_smoke.txt','w').write('\n'.join(smoke)+'\n')
P
wc -l s19_tab.txt s19_big.txt; ls -la hbm4 s19 | head -30
