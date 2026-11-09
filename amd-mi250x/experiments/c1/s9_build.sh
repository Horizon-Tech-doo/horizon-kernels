#!/bin/bash
# Build all asm variants as code objects. Uses JIT flags for -S, then the container's assembler, lld and bundler.
cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p s9; E1=../E1/solo
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null); HC=$(python -c "from tilelang.contrib.rocm import find_hipcc; print(find_hipcc())" 2>/dev/null)
FL="-O3 -S -gline-tables-only --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
BIN=/opt/rocm-6.2.4/llvm/bin
mk(){ # src bounds name
  sed -E "s/__launch_bounds__\(256[^)]*\)/$2/" $1 > s9/$3.cc; $HC $FL -o s9/$3.s s9/$3.cc 2>/dev/null || echo FAILS $3; }
: > s9_tab.txt
for sn in 4k 8k 32k 17k 16k; do
 mk s3src/wheel_b12864_$sn.cu "__launch_bounds__(256)" b12864a0_$sn &
 mk s3src/wheel_b12864_$sn.cu "__launch_bounds__(256, 2)" b12864a2_$sn &
 mk s3src/wheel_b12864_$sn.cu "__launch_bounds__(256, 3)" b12864a3_$sn &
 mk s3src/fork_b25632_ann2_$sn.cu "__launch_bounds__(256, 2)" b25632a2_$sn &
 mk $E1/rr25632w2_$sn.cu "__launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2)))" rr25632w2_$sn &
 mk $E1/rr12864w3_$sn.cu "__launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(3)))" rr12864w3_$sn &
 wait
done
for f in s9/*_*.s; do b=${f%.s}; n=$(basename $b); base=${n%_*}; sn=${n##*_}
  [[ $n == *nop* ]] && continue
  python3 s9_asm.py $f add s9/${base}+nop_$sn.s
  python3 s9_asm.py $f del s9/${base}-nop_$sn.s
done
for f in s9/*.s; do b=${f%.s}; $BIN/clang -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx90a -c $f -o $b.o && $BIN/ld.lld -shared $b.o -o $b.elf && $BIN/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input=$b.elf -output=$b.hsaco || echo ASMFAIL $f; done
python3 - <<'P'
import glob,os,re
SH={'4k':(4096,4096,4096,40),'8k':(8192,8192,8192,40),'32k':(32768,1024,2048,40),'17k':(4096,17408,5120,40),'16k':(16384,16384,16384,15)}
T=[]
for f in sorted(glob.glob('s9/*.hsaco')):
    n=os.path.basename(f)[:-6]; base,sn=n.rsplit('_',1); M,N,K,it=SH[sn]
    bM,bK,dyn=(128,64,32768) if '12864' in base else (256,32,24576)
    kn=re.search(r'^([A-Za-z_0-9]+):\s+; @',open(f[:-6]+'.s').read(),re.M).group(1)
    if base.endswith('-nop') and not base.startswith('b12864a3'): continue
    T.append(f"{base}_{sn} {os.path.abspath(f)} {kn} {M} {N} {K} {bM} 128 {bK} {dyn} 0 {it}")
open('s9_tab.txt','w').write('\n'.join(T)+'\n'); print(len(T),'arms')
P
grep -c "" s9_tab.txt; grep -E "s9/b12864a[023]_8k|s9/b25632a2_8k|rr25632w2_8k|rr12864w3_8k" -h /dev/null; for f in s9/*8k.s; do echo "$f nop=$(grep -c 's_nop 0' $f) vg=$(grep -m1 -E '^; TotalNumVgprs' $f)"; done
