# Median TF per arm and per file; ratio to 'hb' in the same file. Usage: ana.py <dir>...
import json,glob,os,sys,statistics as st
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d+'/*.jsonl')):
        b=os.path.basename(f)
        if b.startswith('metrics') or b.startswith('poll'): continue
        arms={}
        checks={}
        for l in open(f,errors='replace'):
            try:r=json.loads(l)
            except: continue
            if 'impl' in r:
                x=arms.setdefault(r['impl'],dict(tf=[],mhz=[]));x['tf'].append(r['TF'])
                if 'mhz_med' in r:x['mhz'].append(r['mhz_med'])
            if 'check' in r: checks[r['check']]=(r['rel'],r['elements_differ'])
        if not arms: print(b,'EMPTY');continue
        hb=st.median(arms['hb']['tf']) if 'hb' in arms else None
        s=[]
        for k,x in sorted(arms.items(),key=lambda kv:-st.median(kv[1]['tf'])):
            m=st.median(x['tf']);e=f"{k} {m:.2f}[{min(x['tf']):.2f},{max(x['tf']):.2f}]"
            if hb and k!='hb': e+=f" r={m/hb:.3f}"
            if x['mhz']: e+=f" {st.median(x['mhz']):.0f}MHz"
            if k in checks and checks[k][0]>2e-3: e+=" BADCHK"
            s.append(e)
        print(b,' | '.join(s))
