import glob,json,re,sys
import numpy as np
dt=np.dtype([('re','u8'),('r0','u8'),('r1','u8'),('me','u8'),('m0','u8'),('m1','u8'),('hw','u4'),('bk','u4')])
rate=25e6
for fn in sorted(glob.glob('traces2/*.bin')):
    d=np.fromfile(fn,dt)
    tot_r=(d['r1']-d['re']).astype(float); tot_m=(d['m1']-d['me']).astype(float)
    mhz=tot_m/tot_r*rate/1e6
    span=(int(d['r1'].max())-int(d['re'].min()))/rate*1e6
    cu=(d['hw']>>8)&0xff
    bid=d['bk']; ub=np.unique(bid); rows=[]
    for b in ub:
        i=bid==b; rows.append((int(d['re'][i].min()),int(d['r1'][i].max()),int(cu[i][0])))
    bl=np.array(rows); pc,cnt=np.unique(bl[:,2],return_counts=True)
    busy=0;peak=[]
    for c in pc:
        x=bl[bl[:,2]==c]; ev=sorted([(a,1) for a in x[:,0]]+[(b,-1) for b in x[:,1]]); cur=mx=0;last=None;bz=0
        for t,e in ev:
            if cur>0: bz+=t-last
            last=t;cur+=e;mx=max(mx,cur)
        peak.append(mx);busy+=bz
    print(json.dumps(dict(f=fn.split('/')[-1],span_us=round(span,1),mhz_p5=round(float(np.percentile(mhz,5)),1),mhz_med=round(float(np.median(mhz)),1),mhz_p95=round(float(np.percentile(mhz,95)),1),ncu=len(pc),bpc_min=int(cnt.min()),bpc_max=int(cnt.max()),peak_max=int(max(peak)),peak_med=float(np.median(peak)),busy=round(busy/rate*1e6/(span*len(pc)),3),blocks=len(ub))))
