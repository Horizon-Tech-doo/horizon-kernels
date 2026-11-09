# Poll gpu_metrics and hwmon of every card visible in sysfs. Writes one JSON line per sample. Stop with SIGTERM.
import glob,json,os,struct,sys,time,signal
out=open(sys.argv[1],'w'); dt=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
paths=sorted(glob.glob('/sys/class/drm/card[0-9]*/device/gpu_metrics'))
stop=[False]; signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True))
while not stop[0]:
    t=time.time(); rec={'t':round(t,3)}
    for p in paths:
        try: b=open(p,'rb').read()
        except OSError as e: continue
        c=p.split('/')[4]
        size,fr,cr=struct.unpack_from('<HBB',b,0)
        if fr==1 and cr>=3 and len(b)>=120:
            vals=struct.unpack_from('<10H',b,4)
            agfx,cgfx=struct.unpack_from('<H',b,40)[0],struct.unpack_from('<H',b,54)[0]
            th=struct.unpack_from('<I',b,68)[0]; it=struct.unpack_from('<Q',b,112)[0]; en=struct.unpack_from('<Q',b,24)[0]
            vg=struct.unpack_from('<H',b,106)[0]
            rec[c]=dict(te=vals[0],th=vals[1],tm=vals[2],act=vals[6],umc=vals[7],pw=vals[9],en=en,agfx=agfx,cgfx=cgfx,thr=th,ithr=it,vgfx=vg)
        else: rec[c]=dict(size=size,fr=fr,cr=cr,hex=b[:128].hex())
    out.write(json.dumps(rec)+'\n'); out.flush()
    d=dt-(time.time()-t)
    if d>0: time.sleep(d)
