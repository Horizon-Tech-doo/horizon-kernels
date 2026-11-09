import json,os
t=json.load(open('table.json'))
W=['b12864','b25632w2','rr12864w3','rr25632w2']
it=lambda K,M,N:(40 if M*N*K<9e10 else 20)
f=open('tab_std.txt','w')
for e in t:
    n=e['name']; v=e['variant']; I=it(e['K'],e['M'],e['N'])
    f.write(f"{n}|solo solo/{n}.co {n} {e['M']} {e['N']} {e['K']} {e['bM']} 128 {e['bK']} {e['dyn']} 0 {I}\n")
    if v in W: f.write(f"{n}|comb combined.co {n} {e['M']} {e['N']} {e['K']} {e['bM']} 128 {e['bK']} {e['dyn']} 0 {I}\n")
SH={'4k':(4096,4096,4096),'8k':(8192,8192,8192),'32k':(32768,1024,2048),'17k':(4096,17408,5120)}
# MB combined TU, kernel names as in analysis/mb
for lab,kn,(M,N,K),bM,bK,dyn in [('b12864_4k','b12864',SH['4k'],128,64,32768),('b25632w2_4k','b25632w2',SH['4k'],256,32,24576),('rr12864w3_4k','rr12864w3',SH['4k'],128,64,32768),('rr25632w2_4k','rr25632w2',SH['4k'],256,32,24576),
  ('b12864_8k','b12864_8k',SH['8k'],128,64,32768),('b25632w2_8k','b25632w2_8k',SH['8k'],256,32,24576),('rr12864w3_8k','rr12864_8kw3',SH['8k'],128,64,32768),('rr25632w2_8k','rr25632_8kw2',SH['8k'],256,32,24576),
  ('b12864_32k','b12864_32k',SH['32k'],128,64,32768),('b25632w2_32k','b25632w2_32k',SH['32k'],256,32,24576)]:
    f.write(f"{lab}|mbtu mb_orig/kernels.co {kn} {M} {N} {K} {bM} 128 {bK} {dyn} 0 {it(K,M,N)}\n")
for vn in ['b12864m32','rr12864m32w3','rr12864m32']:
  for sn,(M,N,K) in SH.items():
    f.write(f"{vn}_{sn}|solo solo/{vn}_{sn}.co {vn}_{sn} {M} {N} {K} 128 128 64 32768 0 {it(K,M,N)}\n")
f.close()
f=open('tab_ck.txt','w')
for vn,bM,bK,dyn in [('b12864',128,64,32768),('b25632w2',256,32,24576),('rr12864w3',128,64,32768),('rr25632w2',256,32,24576)]:
  for sn,(M,N,K) in SH.items():
    I=8
    f.write(f"{vn}_{sn}_ck ck/{vn}_{sn}_ck.co {vn}_{sn}_ck {M} {N} {K} {bM} 128 {bK} {dyn} 1 {I}\n")
  for (M,N) in [(5632,3200),(5888,3072)] if bM==128 else [(5632,3200),(5888,3072)]:
    f.write(f"{vn}_n{N}_ck ck/{vn}_n{N}_ck.co {vn}_n{N}_ck {M} {N} 4096 {bM} 128 {bK} {dyn} 1 {I}\n")
f.close()
f=open('tab_geo.txt','w')
# pairs: (bM, M, N): B and footprint M*N, kernel only depends on N
P128=[(5632,3200),(5888,3072),(3200,5632),(3072,5888),(3840,4224),(4096,4096)]
P256=[(5632,3200),(5888,3072),(6400,2816),(6144,2944)]
for vn,bM,bK,dyn,pl in [('b12864',128,64,32768,P128),('rr12864w3',128,64,32768,P128),('b25632w2',256,32,24576,P256),('rr25632w2',256,32,24576,P256)]:
  for M,N in pl:
    src=f"solo/{vn}_n{N}.co" if N!=4096 else f"solo/{vn}_4k.co"
    kn=f"{vn}_n{N}" if N!=4096 else f"{vn}_4k"
    f.write(f"{vn}_M{M}_N{N} {src} {kn} {M} {N} 4096 {bM} 128 {bK} {dyn} 0 30\n")
f.close()
