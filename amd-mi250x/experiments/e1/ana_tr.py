import glob,json,re,sys
import numpy as np
dt=np.dtype([('re','u8'),('r0','u8'),('r1','u8'),('me','u8'),('m0','u8'),('m1','u8'),('hw','u4'),('bk','u4')])
rate=25e6
rows=[]
for fn in sorted(glob.glob('traces/*.bin')):
    m=re.search(r'(mfma\d*_tr)_P(\d)_d(\d+)_B(\d+)_g(\d+)x(\d+)_nk(\d+)_r(\d)',fn)
    k,P,dyn,B,gx,gy,nk,r=m.groups();P,dyn,B,nk,r=int(P),int(dyn),int(B),int(nk),int(r)
    d=np.fromfile(fn,dt)
    span=(int(d['r1'].max())-int(d['re'].min()))/rate*1e6
    dr=(d['r1']-d['r0']).astype(float); dm=(d['m1']-d['m0']).astype(float)
    mhz=dm/dr*rate/1e6
    nm=(64 if k=='mfma_tr' else 32)*nk
    cyc=dm/nm
    cu=(d['hw']>>8)&0xff; simd=(d['hw']>>4)&3
    # block table
    bid=d['bk']; ub=np.unique(bid)
    bs={}
    for b in ub:
        i=bid==b
        bs[b]=(int(d['re'][i].min()),int(d['r1'][i].max()),int(cu[i][0]),len(set(cu[i].tolist())),len(set(simd[i].tolist())))
    blocks=np.array([bs[b] for b in ub])
    pc,cnt=np.unique(blocks[:,2],return_counts=True)
    # peak concurrency per CU
    peak=[];busy=0
    t0=blocks[:,0].min()
    for c in pc:
        x=blocks[blocks[:,2]==c]
        ev=sorted([(a,1) for a in x[:,0]]+[(b,-1) for b in x[:,1]])
        cur=mx=0;last=None;bz=0
        for t,e in ev:
            if cur>0: bz+=t-last
            last=t;cur+=e;mx=max(mx,cur)
        peak.append(mx);busy+=bz
    dur=(blocks[:,1]-blocks[:,0])/rate*1e6
    starts=np.sort((blocks[:,0]-t0)/rate*1e6)
    # clusters of starts separated by > 25% of median duration
    gaps=np.diff(starts);med=np.median(dur)
    cl=[];c=1
    for g in gaps:
        if g>0.25*med: cl.append(c);c=1
        else: c+=1
    cl.append(c)
    per_cu_end=[(blocks[blocks[:,2]==c][:,1].max()-t0)/rate*1e6 for c in pc]
    rows.append(dict(k=k,P=P,dyn=dyn,B=B,nk=nk,r=r,span_us=round(span,1),TF_span=round(B*nk*2097152/span*1e-6,2),mhz_med=round(float(np.median(mhz)),1),mhz_p5=round(float(np.percentile(mhz,5)),1),mhz_p95=round(float(np.percentile(mhz,95)),1),
      cyc_per_mfma_med=round(float(np.median(cyc)),2),cyc_p5=round(float(np.percentile(cyc,5)),2),cyc_p95=round(float(np.percentile(cyc,95)),2),ncu=len(pc),blocks_per_cu_min=int(cnt.min()),blocks_per_cu_max=int(cnt.max()),peak_conc_max=int(max(peak)),peak_conc_med=float(np.median(peak)),
      cu_busy_frac=round(busy/rate*1e6/(span*len(pc)),3),dur_us_med=round(float(med),1),dur_us_min=round(float(dur.min()),1),dur_us_max=round(float(dur.max()),1),start_clusters=cl[:20],n_clusters=len(cl),last_cu_end_p10_us=round(float(np.percentile(per_cu_end,10)),1),multi_cu_blocks=int((blocks[:,3]>1).sum()),simds_per_block=float(np.median(blocks[:,4]))))
for r in rows: print(json.dumps(r))
