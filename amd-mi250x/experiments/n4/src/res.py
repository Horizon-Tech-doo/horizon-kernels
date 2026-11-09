import re,glob,sys
def parse(fn):
    out={};name=None
    for l in open(fn):
        m=re.match(r'^([A-Za-z_0-9]+):\s+; @',l)
        if m: name=m.group(1);out[name]={}
        m=re.match(r'^; (NumVgprs|NumAgprs|TotalNumVgprs|ScratchSize|Occupancy|NumSgprs): (\d+)',l)
        if m and name: out[name][m.group(1)]=int(m.group(2))
    return out
for fn in sys.argv[1:]:
    for k,v in sorted(parse(fn).items()):
        if v: print(fn.split('/')[-1],k,v.get('NumVgprs'),v.get('NumAgprs'),v.get('TotalNumVgprs'),v.get('ScratchSize'),v.get('Occupancy'))
