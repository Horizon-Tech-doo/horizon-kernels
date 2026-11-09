#!/bin/bash
# Experiment A variants, compiled inside the container. Base asm comes from s9/ (JIT flags). Clause arms use the JIT's own compile form, with HIPCC_COMPILE_FLAGS_APPEND.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p s12
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null); HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
FL="-O3 -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
BIN=/opt/rocm-6.2.4/llvm/bin
as1(){ b=${1%.s}; $BIN/clang -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx90a -c $1 -o $b.o && $BIN/ld.lld -shared $b.o -o $b.elf && $BIN/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input=$b.elf -output=$b.hsaco || echo ASMFAIL $1; rm -f $b.o; }
for sn in 4k 8k 32k 17k 16k; do
  for k in rr25632w2 rr12864w3; do cp s9/${k}_$sn.s s12/$k.asis_$sn.s
    for m in rep3 rep7 ctrl; do python3 s9_asm.py s9/${k}_$sn.s $m s12/$k.${m}_$sn.s; done; done
  cp s9/b12864a0_$sn.s s12/b12864.asis_$sn.s
  for m in rep3 rep7 fill0 fill3 fill7; do python3 s9_asm.py s9/b12864a0_$sn.s $m s12/b12864.${m}_$sn.s; done
  python3 s9_asm.py s12/b12864.fill0_$sn.s ctrl s12/b12864.ctrl_$sn.s
  cp s9/b25632a2_$sn.s s12/b25632a2.asis_$sn.s
  python3 s9_asm.py s9/b25632a2_$sn.s fill3 s12/b25632a2.fill3_$sn.s
  for f in s12/*_$sn.s; do as1 $f & done; wait
  # compiler flag arms, -c as the JIT does and -S for inspection of the same flags
  cl(){ # srccc n name
    HIPCC_COMPILE_FLAGS_APPEND="-mllvm -amdgpu-max-memory-clause=$2" $HC $FL -c -o s12/$3.hsaco $1 2>/dev/null || echo FAIL $3
    HIPCC_COMPILE_FLAGS_APPEND="-mllvm -amdgpu-max-memory-clause=$2" $HC $FL -S -o s12/$3.s $1 2>/dev/null; }
  cl s9/b12864a0_$sn.cc 1 b12864.cl1_$sn & cl s9/b12864a0_$sn.cc 2 b12864.cl2_$sn & cl s9/rr25632w2_$sn.cc 1 rr25632w2.cl1_$sn & wait
done
python3 - <<'P'
import glob,os,re
A=os.path.abspath
SH={'4k':(4096,4096,4096,40),'8k':(8192,8192,8192,20),'32k':(32768,1024,2048,40),'17k':(4096,17408,5120,20),'16k':(16384,16384,16384,5)}
tabs={'a':[],'d':[]}
for sn,(M,N,K,it) in SH.items():
    T=tabs['d'] if sn=='16k' else tabs['a']
    T.append(f"hipblaslt_direct - - {M} {N} {K} 0 0 0 0 {it}")
    for f in sorted(glob.glob(f's12/*_{sn}.hsaco')):
        n=os.path.basename(f)[:-6]; k=n.split('.')[0]
        bM,bK,dyn=(128,64,32768) if '12864' in k else (256,32,24576)
        kn=re.search(r'^([A-Za-z_0-9]+):\s+; @',open(f[:-6]+'.s').read(),re.M).group(1)
        if sn=='16k' and n not in (f'b25632a2.asis_{sn}',f'b12864.asis_{sn}'): continue
        T.append(f"{n} {A(f)} {kn} {M} {N} {K} {bM} 128 {bK} {dyn} {it}")
    if sn!='16k':
        T.append(f"rr25632w2.hand_{sn} {A('../E1/solo/rr25632w2_'+sn+'.co')} rr25632w2_{sn} {M} {N} {K} 256 128 32 24576 {it}")
for t,T in tabs.items():
    open(f's12_tab{t}.txt','w').write('\n'.join(T)+'\n'); print(t,len(T),'arms')
P
# what is in the K loop of each arm: nop counts between loads, registers and load addresses from the linked object
python3 - > s12_asm_table.txt <<'P'
import glob,re,os
real=lambda l:l.strip() and not re.match(r'\s*(\.loc|;|\.file|\.Ltmp|\.cfi)',l)
print('arm loads between_nops nop_types other_nops V A Occ scratch')
for f in sorted(glob.glob('s12/*.s')):
    L=[l for l in open(f)]; R=[l for l in L if real(l)]
    b=[R[i].split()[1] for i in range(1,len(R)-1) if re.match(r'\s*s_nop',R[i]) and re.match(r'\s*global_load',R[i-1]) and re.match(r'\s*global_load',R[i+1])]
    o=sum(1 for l in R if re.match(r'\s*s_nop',l))-len(b); g=sum(1 for l in R if re.match(r'\s*global_load',l))
    t=open(f).read(); F=lambda k:(re.search(r'^; %s: (\d+)'%k,t,re.M) or [None,'?'])[1]
    print(os.path.basename(f)[:-2],g,len(b),','.join(sorted(set(b))) or '-',o,F('NumVgprs'),F('NumAgprs'),F('Occupancy'),F('ScratchSize'))
P
for f in s12/rr25632w2.asis_8k s12/rr25632w2.rep3_8k s12/rr25632w2.ctrl_8k s12/b12864.fill0_8k s12/b12864.ctrl_8k s12/b12864.asis_8k; do echo "== $f"; $BIN/llvm-objdump -d --mcpu=gfx90a $f.elf 2>/dev/null | grep -E "global_load|s_nop" | awk '{print $NF, $0}' | cut -c1-90 | tail -18; done > s12_addr.txt
# xnack-: record, compile only
HIPCC_X="$HC -O3 -gline-tables-only --offload-arch=gfx90a:xnack- --genco -std=c++17 -I$TPL"
for sn in 8k; do for s in s9/b12864a3_$sn.cc s9/rr25632w2_$sn.cc s9/rr12864w3_$sn.cc; do n=$(basename $s .cc); $HIPCC_X -S -o s12/xn_$n.s $s 2>/dev/null; echo "$n xnack-: s_nop 0 count $(grep -c 's_nop 0' s12/xn_$n.s), xnack-any: $(grep -c 's_nop 0' s9/$n.s)"; done; done > s12_xnack.txt
