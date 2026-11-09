#!/bin/bash
# clang 22 (vLLM image, ROCm 7.2.3): the same device sources and the same flags as clang 18, compile, with hipcc from the image.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p s16
TPLI=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages/tilelang/src
FL="-O3 -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPLI"
/opt/rocm/bin/hipcc --version | head -2
/opt/rocm/bin/hipcc -O2 -std=c++17 hbm3.cpp -o hbm3_22 -lhipblaslt 2>&1 | grep -E "error" | head
for sn in 4k 8k 32k 17k; do
 mk(){ # src clause name
  F=""; [ "$2" != 0 ] && F="-mllvm -amdgpu-max-memory-clause=$2"
  HIPCC_COMPILE_FLAGS_APPEND="$F" /opt/rocm/bin/hipcc $FL -c -o s16/$3.hsaco $1 2>&1 | grep -E " error" | head -3
  HIPCC_COMPILE_FLAGS_APPEND="$F" /opt/rocm/bin/hipcc $FL -S -o s16/$3.s $1 2>/dev/null; }
 mk s9/b12864a0_$sn.cc 0 b12864.asis_$sn & mk s9/b12864a0_$sn.cc 1 b12864.cl1_$sn & mk s9/b12864a0_$sn.cc 2 b12864.cl2_$sn &
 mk s9/rr25632w2_$sn.cc 0 rr25632w2.asis_$sn & mk s9/rr25632w2_$sn.cc 1 rr25632w2.cl1_$sn & mk s9/rr12864w3_$sn.cc 0 rr12864w3.asis_$sn & mk s9/b25632a2_$sn.cc 0 b25632a2.asis_$sn & wait
done
python3 -I - <<'P'
import glob,os,re
SH={'4k':(4096,4096,4096,40),'8k':(8192,8192,8192,20),'32k':(32768,1024,2048,40),'17k':(4096,17408,5120,20)}
T=[]
for sn,(M,N,K,it) in SH.items():
    T.append(f"hipblaslt_direct - - {M} {N} {K} 0 0 0 0 {it}")
    for f in sorted(glob.glob(f's16/*_{sn}.hsaco')):
        n=os.path.basename(f)[:-6]; k=n.split('.')[0]; bM,bK,dyn=(128,64,32768) if '12864' in k else (256,32,24576)
        kn=re.search(r'^([A-Za-z_0-9]+):\s+; @',open(f[:-6]+'.s').read(),re.M).group(1)
        T.append(f"{n} {os.path.abspath(f)} {kn} {M} {N} {K} {bM} 128 {bK} {dyn} {it}")
open('s16_tab.txt','w').write('\n'.join(T)+'\n'); print(len(T),'arms')
real=lambda l:l.strip() and not re.match(r'\s*(\.loc|;|\.file|\.Ltmp|\.cfi)',l)
out=open('s16_asm_table.txt','w'); print('arm loads between_nops other_nops mfma V A Occ scratch',file=out)
for f in sorted(glob.glob('s16/*.s')):
    R=[l for l in open(f) if real(l)]
    b=[1 for i in range(1,len(R)-1) if re.match(r'\s*s_nop',R[i]) and re.match(r'\s*global_load',R[i-1]) and re.match(r'\s*global_load',R[i+1])]
    t=open(f).read(); F=lambda k:(re.search(r'^; %s: (\d+)'%k,t,re.M) or [None,'?'])[1]
    print(os.path.basename(f)[:-2],sum(1 for l in R if re.match(r'\s*global_load',l)),len(b),sum(1 for l in R if re.match(r'\s*s_nop',l))-len(b),sum(1 for l in R if re.match(r'\s*v_mfma',l)),F('NumVgprs'),F('NumAgprs'),F('Occupancy'),F('ScratchSize'),file=out)
P
