import sys, glob, os, re, subprocess, importlib.util, json, time, torch
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); sys.argv=["x"]; spec.loader.exec_module(gq)
TPL="/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages/tilelang/src"
B,L,S=32,8192,13; Hkv,G,D=4,6,256; MAXP=L//16; NBLK=B*MAXP
def rep(name,k):
    src=k.get_kernel_source(); fn=f"gen/form_{name}.cu"; open(fn,"w").write(src)
    r=subprocess.run(f"/scratch/project_XXXXXXXXX/tilelang/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I{TPL} -S -x hip {fn} -o gen/form_{name}.s",shell=True,capture_output=True,text=True)
    t=open(f"gen/form_{name}.s").read(); lines=t.split("\n"); grp=[];c=0
    for l in lines:
        if "global_load" in l: c+=1
        elif "vmcnt" in l and "s_waitcnt" in l:
            if c: grp.append(c)
            c=0
    print(json.dumps({"form":name,"groups":grp,"barriers":t.count("s_barrier"),"sync_src":src.count("__syncthreads"),"valu":sum(1 for l in lines if l.strip().startswith("v_")),"vgpr":re.findall(r"; NumVgprs: (\d+)",t)[:1],"occ":re.findall(r"; Occupancy: (\d+)",t)[:1],"dyn":re.findall(r"buf_dyn_shmem \+ (\d+)",src)}),flush=True)
V=lambda **k: gq.decode_partial_paged_v2(B,NBLK,MAXP,Hkv,G,D,**k)
rep("a13",V(S=13)); rep("a13rc",V(S=13,rc=True)); rep("a20rc",V(S=20,rc=True)); rep("a26rc",V(S=26,rc=True))
rep("s13rc",V(S=13,rc=True,swap=True))
rep("tp12_4",V(S=16,rc=True,swap=True,taper=(12,40,8)))
rep("tp10_6",V(S=16,rc=True,swap=True,taper=(10,44,12)))
rep("tp11_8",V(S=19,rc=True,swap=True,taper=(11,40,9)))
rep("tp8_16",V(S=24,rc=True,swap=True,taper=(8,48,8)))
rep("tpn12_4",V(S=16,rc=True,swap=False,taper=(12,40,8)))
