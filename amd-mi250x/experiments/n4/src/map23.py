# Map ROCR device index to DRM card and OAM from sysfs. Prints no host identifiers, only indices. Last line: "<target rocr index> <twin rocr index>".
import glob,os,re
nodes=[]
for p in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*'),key=lambda x:int(x.rsplit('/',1)[1])):
    try: gid=int(open(p+'/gpu_id').read())
    except OSError: continue
    if gid==0: continue
    pr=dict(l.split() for l in open(p+'/properties') if len(l.split())==2)
    nodes.append((int(p.rsplit('/',1)[1]),int(pr.get('drm_render_minor',-1)),int(pr.get('location_id',-1))))
cards={}
for c in glob.glob('/sys/class/drm/card[0-9]*'):
    if not re.fullmatch(r'card\d+',os.path.basename(c)): continue
    rm=glob.glob(c+'/device/drm/renderD*')
    if rm: cards[int(rm[0].rsplit('renderD',1)[1])]=(os.path.basename(c),int(open(c+'/device/numa_node').read()))
rows=[]
for i,(n,minor,loc) in enumerate(nodes):
    cd,numa=cards.get(minor,('?',-1)); rows.append((i,n,cd,numa)); print('rocr',i,'kfd_node',n,'card',cd,'numa',numa)
t=rows[0]; tw=[r for r in rows if r[3]==t[3] and r[0]!=t[0]]
print('target',t[2],'twin',tw[0][2] if tw else None)
print(t[0],tw[0][0] if tw else -1)
