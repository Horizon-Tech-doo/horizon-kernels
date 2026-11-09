# One row per compile: T, requested second argument, compiler-reported resources (from the kernel-resource-usage remark)
import re,glob,os,sys
rows=[]
for f in glob.glob('s1/*.log'):
    n=os.path.basename(f)[:-4]; t,a=n.split('_'); t=int(t[1:])
    d={}
    for l in open(f):
        m=re.search(r'remark:\s+(VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|LDS Size \[bytes/block\]|VGPRs Spill|TotalSGPRs|SGPRs): (\d+)',l)
        if m: d[m.group(1)]=int(m.group(2))
    k=(t,0 if a=='none' else (99 if a.startswith('attr') else int(a[1:])))
    wpb=(t+63)//64
    implied=(wpb+3)//4
    rows.append((k,a,wpb,implied,d))
print('T a waves/block implied_min_w/eu V A scratch Occ')
for k,a,wpb,imp,d in sorted(rows,key=lambda r:r[0]):
    print(k[0],a,wpb,imp,d.get('VGPRs'),d.get('AGPRs'),d.get('ScratchSize [bytes/lane]'),d.get('Occupancy [waves/SIMD]'))
