# Placement of the gap: control, every gap, every 2nd gap, every 4th gap, longer nop. b12864 with no second bounds argument. Also the source-level arm from s10.
import re, sys, subprocess
SH={'4k':(4096,4096,4096,40),'8k':(8192,8192,8192,40),'32k':(32768,1024,2048,40),'17k':(4096,17408,5120,40),'16k':(16384,16384,16384,15)}
BIN='/opt/rocm-6.2.4/llvm/bin/'
T=[]
def asm(name,txt):
    b=f's11/{name}'; open(b+'.s','w').write(txt)
    subprocess.check_call(f'{BIN}clang -x assembler --target=amdgcn-amd-amdhsa -mcpu=gfx90a -c {b}.s -o {b}.o && {BIN}ld.lld -shared {b}.o -o {b}.elf && {BIN}clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input={b}.elf -output={b}.hsaco',shell=True)
def place(L,every,nop='s_nop 0'):
    # L without any existing nop between loads; gap after load number n when n%every==0, inside a run of 8 loads
    out=[];n=0;idx=[i for i,l in enumerate(L) if re.match(r'\s*global_load',l)]
    kloop=idx[:8]; 
    for i,l in enumerate(L):
        out.append(l)
        if i in kloop:
            n+=1
            if n<8 and n%every==0: out.append(f'\t{nop}\n')
    return out
for sn,(M,N,K,it) in SH.items():
    src=open(f's9/b12864a0-nop_{sn}.s').readlines()
    kn='base_kernel'
    arms={'g0':src,'g1':place(src,1),'g2':place(src,2),'g4':place(src,4),'g1nop3':place(src,1,'s_nop 3'),'g1vnop':place(src,1,'v_nop')}
    for a,txt in arms.items():
        asm(f'{a}_{sn}',''.join(txt)); T.append(f"{a}_{sn} {sys.argv[1]}/s11/{a}_{sn}.hsaco {kn} {M} {N} {K} 128 128 64 32768 0 {it}")
    T.append(f"srcnop_{sn} {sys.argv[1]}/s10/srcnop_{sn}.hsaco {kn} {M} {N} {K} 128 128 64 32768 0 {it}")
    T.append(f"srcnopa3_{sn} {sys.argv[1]}/s10/srcnopa3_{sn}.hsaco {kn} {M} {N} {K} 128 128 64 32768 0 {it}")
    T.append(f"jit_a2b25632_{sn} {sys.argv[1]}/s2co/{sn}.hsaco {kn} {M} {N} {K} 256 128 32 24576 0 {it}")
open('s11_tab.txt','w').write('\n'.join(T)+'\n'); print(len(T))
