"""CPU. Unbundles hsaco, disassembles gfx90a; compares instruction text of bytes vs bytes2, writes .s for the kept arms. Hash excludes addresses and debug comments."""
import subprocess,hashlib,struct,re,glob,os,sys,collections,json
OD="/opt/rocm/llvm/bin/llvm-objdump"
def co(p):
    data=open(p,"rb").read(); i=data.find(b"__CLANG_OFFLOAD_BUNDLE__")
    assert i>=0,p
    c=struct.unpack_from("<Q",data,i+24)[0]; q=i+32
    for _ in range(c):
        off,sz,tl=struct.unpack_from("<QQQ",data,q); q+=24
        tr=data[q:q+tl].decode(errors="replace"); q+=tl
        if "gfx90a" in tr and sz>0: return data[i+off:i+off+sz]
def dis(p,keep=None):
    t="/tmp/d128_"+str(os.getpid())+".co"; open(t,"wb").write(co(p))
    o=subprocess.run([OD,"-d","--no-show-raw-insn",t],capture_output=True,text=True).stdout; os.remove(t)
    ins=[re.sub(r"\s*//.*","",l).strip() for l in o.splitlines() if l.startswith((" ","\t"))]
    ins=[l for l in ins if l]
    if keep: open(keep,"w").write(o)
    return hashlib.md5("\n".join(ins).encode()).hexdigest(),ins
for p in sorted(glob.glob("bytes2/*.hsaco")):
    q=p.replace("bytes2","bytes"); a,ia=dis(p); b,ib=dis(q)
    print(json.dumps(dict(name=os.path.basename(p)[:-6],same_insn=a==b,n=[len(ia),len(ib)],insn_md5=b[:12])))
m={}
for p in sorted(glob.glob("bytes/*.hsaco")):
    h,i=dis(p,p[:-6]+".s"); ops=collections.Counter(x.split()[0] for x in i)
    m[os.path.basename(p)[:-6]]=dict(insn_md5=h[:12],n=len(i),mfma=sum(v for k,v in ops.items() if "mfma" in k or "mfmacc" in k or "mfma" in k),ds=sum(v for k,v in ops.items() if k.startswith("ds_")),gl=sum(v for k,v in ops.items() if k.startswith(("flat_","global_","buffer_"))),barrier=ops.get("s_barrier",0),nops=sum(v for k,v in ops.items() if "nop" in k),waitcnt=sum(v for k,v in ops.items() if "waitcnt" in k))
    print(os.path.basename(p)[:-6],m[os.path.basename(p)[:-6]])
json.dump(m,open("pre/asm.json","w"),indent=0)
