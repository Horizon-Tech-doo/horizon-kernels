# TileLang-generated K2 sources, private cache. rr() and rename() are the same text transforms as E1/gen_e1.py.
import json,os,re,sys,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch, tilelang
sys.path.insert(0,'/scratch/project_XXXXXXXXX/tilelang/exp/E1')
from gemm_prefetch import gemm
def hdr(s): return s[:s.index('extern "C"')]
def body(s): return s[s.index('extern "C"'):]
def rename(s,name,attr=""): return re.sub(r'__launch_bounds__\((\d+)\)\s+\w+\(', lambda m:'__launch_bounds__(%s) '%m.group(1)+(attr+' ' if attr else '')+name+'(', s)
def rr(s):
    m=re.search(r'half_t a_r\[(\d+)\];\s*\n\s*half_t b_r\[(\d+)\];',s)
    na,nb=int(m.group(1))//8,int(m.group(2))//8
    decl=" ".join("uint4 a_r_%d;"%i for i in range(na))+" "+" ".join("uint4 b_r_%d;"%i for i in range(nb))
    s=s.replace(m.group(0),decl)
    pat=re.compile(r'for \(int (v(?:_\d+)?) = 0; \1 < (\d+); \+\+\1\) \{\n(.*?)\n\s*\}',re.S)
    def exp(mm):
        v,n,b=mm.group(1),int(mm.group(2)),mm.group(3)
        out=[]
        for i in range(n):
            t=re.sub(r'\*\(uint4\*\)\(([ab])_r \+ \(%s \* 8\)\)'%v,lambda q:"%s_r_%d"%(q.group(1),i),b)
            t=re.sub(r'\b%s\b'%v,"(%d)"%i,t)
            out.append(t)
        return "\n".join(out)
    s=pat.sub(exp,s)
    s=re.sub(r'\*\(uint4\*\)\(([ab])_r( \+ 0)?\)',lambda q:"%s_r_0"%q.group(1),s)
    assert "a_r +" not in s and "b_r +" not in s, s[:3000]
    return s
W=lambda n:"__attribute__((amdgpu_waves_per_eu(%d)))"%n
SH={'4k':(4096,4096,4096),'8k':(8192,8192,8192),'32k':(32768,1024,2048),'17k':(4096,17408,5120),'16k':(16384,16384,16384)}
os.makedirs('tlsrc',exist_ok=True); os.makedirs('solo',exist_ok=True)
tile={'b25664':(256,128,64,256,'base',49152),'r25664':(256,128,64,256,'regstage',49152),'b25632t512':(256,128,32,512,'base',24576),'r25632t512':(256,128,32,512,'regstage',24576),'b12825664':(128,256,64,256,'base',49152),'r12825664':(128,256,64,256,'regstage',49152),'r12825632':(128,256,32,256,'regstage',24576)}
tbl={}
for sn,(M,N,K) in SH.items():
    for vn,(bM,bN,bK,th,v,dyn) in tile.items():
        t0=time.time(); k=gemm(M,N,K,bM,bN,bK,th,2,v); s=k.get_kernel_source(); open(f'tlsrc/{vn}_{sn}.cu','w').write(s)
        print(vn,sn,len(s),round(time.time()-t0,1),flush=True)
        forms=[(vn,0,None)]
        if v=='regstage': forms=[('rr'+vn[1:],1,None),('rr'+vn[1:]+'w2',1,2),('rr'+vn[1:]+'w3',1,3)]
        else: forms=[(vn,0,None),(vn+'w2',0,2)]
        for nm,isrr,w in forms:
            x=body(s)
            try:
                if isrr: x=rr(x)
            except AssertionError as e:
                print('rr failed',nm,sn,flush=True); open(f'tlsrc/{nm}_{sn}.fail','w').write(str(e)); continue
            name=f'{nm}_{sn}'
            open(f'solo/{name}.cu','w').write(hdr(s)+rename(x,name,W(w) if w else ""))
            tbl[name]=dict(bM=bM,bN=bN,bK=bK,th=th,dyn=dyn,M=M,N=N,K=K)
json.dump(tbl,open('tl_table.json','w'))
