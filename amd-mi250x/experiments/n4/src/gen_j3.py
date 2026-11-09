# Block swizzle and K-start stagger as text patches on the E1 solo sources. sw: r<P> rasterization2DRow, c<P> rasterization2DColumn. st: <n> starts, 128 K apart.
import re,json,os
E='/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo/'
SH={'4k':(4096,4096,4096),'8k':(8192,8192,8192),'32k':(32768,1024,2048),'17k':(4096,17408,5120),'16k':(16384,16384,16384)}
B={'rr25632w2':(256,128,32,24576,1),'b25632w2':(256,128,32,24576,0),'rr12864w3':(128,128,64,32768,1)}
os.makedirs('sw',exist_ok=True); tbl={}
def patch(s,sw,st,bK,nk,isrr,old,new):
    s=s.replace(old+'(',new+'(',1)
    decl=[]
    if sw:
        decl.append('const dim3 blockIdx = tl::rasterization2D%s<%d>();'%('Row' if sw[0]=='r' else 'Column',int(sw[1:])))
    if st:
        decl.append('const int s0_ = (((int)blockIdx.x + (int)blockIdx.y) & %d) * %d;'%(st-1,128//bK))
    m=re.search(r'extern __shared__ __align__\(\d+\) uchar buf_dyn_shmem\[\];',s)
    s=s[:m.end()]+'\n  '+' '.join(decl)+s[m.end():]
    if st:
        i=s.index('for (int k = 0;')
        head,loop=s[:i],s[i:]
        if isrr:
            head=re.sub(r'\*\(uint4\*\)\(([AB]) \+ \(',lambda q:'*(uint4*)(%s + (s0_*%d + '%(q.group(1),bK),head)
            lb=loop.index('{')+1
            loop=loop[:lb]+'\n    const int kw_ = ((k + 1 + s0_) %% %d) - 1;'%nk+loop[lb:]
            n=loop.count('(k * %d)'%bK); assert n>0
            loop=loop.replace('(k * %d)'%bK,'(kw_ * %d)'%bK)
        else:
            lb=loop.index('{')+1
            loop=loop[:lb]+'\n    const int kw_ = (k + s0_) %% %d;'%nk+loop[lb:]
            assert re.search(r'\(k \* %d\)'%bK,loop)
            loop=loop.replace('(k * %d)'%bK,'(kw_ * %d)'%bK)
        s=head+loop
    return s
for sn,(M,N,K) in SH.items():
    for bn,(bM,bN,bK,dyn,isrr) in B.items():
        src=open(f'{E}{bn}_{sn}.cu').read(); old=f'{bn}_{sn}'; nk=K//bK
        for sw in [None,'r4','r8','r16','c8','c16']:
            for st in [0,4]:
                if not sw and not st: continue
                tag=(sw or '')+('s%d'%st if st else '')
                nm=f'{bn}{tag}_{sn}'
                open(f'sw/{nm}.cu','w').write(patch(src,sw,st,bK,nk,isrr,old,nm))
                tbl[nm]=dict(bM=bM,bN=bN,bK=bK,dyn=dyn)
json.dump(tbl,open('sw_table.json','w')); print(len(tbl))
