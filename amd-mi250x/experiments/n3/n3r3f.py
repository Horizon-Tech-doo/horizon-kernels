"""N3 round 3 forms: compile each layout/stage form at 32x8k and report load groups, barriers, VALU, registers from the generated source. usage: n3r3f.py"""
import sys, os, re, subprocess, importlib.util, json, torch
E="/scratch/project_XXXXXXXXX/tilelang/exp/N3"
spec = importlib.util.spec_from_file_location("gq", E+"/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(gq)
TPL="/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages/tilelang/src"
B,L=32,8192; Hkv,G,D=4,6,256; PAGE=16; MAXP=L//16; NBLK=B*MAXP
def rep(name,k):
    src=k.get_kernel_source(); fn=f"gen/r3_{name}.cu"; open(fn,"w").write(src)
    r=subprocess.run(f"/scratch/project_XXXXXXXXX/tilelang/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I{TPL} -S -x hip {fn} -o gen/r3_{name}.s",shell=True,capture_output=True,text=True)
    t=open(f"gen/r3_{name}.s").read(); lines=t.split("\n"); grp=[];c=0;x4=0;other=0
    for l in lines:
        if "global_load" in l:
            c+=1
            if "dwordx4" in l: x4+=1
            else: other+=1
        elif "vmcnt" in l and "s_waitcnt" in l:
            if c: grp.append(c)
            c=0
    print(json.dumps({"form":name,"groups":grp,"x4":x4,"other_loads":other,"barriers":t.count("s_barrier"),"sync_src":src.count("__syncthreads"),"valu":sum(1 for l in lines if l.strip().startswith("v_")),"vgpr":re.findall(r"; NumVgprs: (\d+)",t)[:1],"agpr":re.findall(r"; NumAgprs: (\d+)",t)[:1],"occ":re.findall(r"; Occupancy: (\d+)",t)[:1],"scratch":re.findall(r"; ScratchSize: (\d+)",t)[:1],"dyn":re.findall(r"buf_dyn_shmem \+ (\d+)",src)}),flush=True)
S1,S2=9,6
for lay in (True,"lbnhc","lhbnc",False):
    for st in ("joint","k_v"):
        if lay=="lhbnc" and st=="k_v": continue
        n=f"{lay if isinstance(lay,str) else ('lbhnc' if lay else 'nhd')}_{st}"
        try: rep(n,gq.decode_partial_paged_v3(B,NBLK,MAXP,Hkv,G,D,S1,S2,128,PAGE,32,86,"kvg",lay,st))
        except Exception as e: print(json.dumps({"form":n,"err":(type(e).__name__+": "+str(e))[-600:]}),flush=True)
# hybrid pages
N=784; NB=-(-L//N)*B
for lay,st,ko in [("rocm_hybrid","joint","xj"),("rocm_hybrid","v_k","xj"),("lbnhc","joint","jx"),("lbnhc","k_v","jx"),(True,"k_v","jx")]:
    n=f"h_{lay}_{st}_{ko}"
    try: rep(n,gq.decode_partial_paged_v3(B,NB,MAXP,Hkv,G,D,S1,S2,128,PAGE,32,86,"kvg",lay,st,ko,N))
    except Exception as e: print(json.dumps({"form":n,"err":(type(e).__name__+": "+str(e))[-600:]}),flush=True)
