# Regenerate the patch series from the edit scripts on a clean v0.1.15 tree. Uses the git index as the moving baseline. Nothing is committed.
import os, subprocess as sp, shutil
R='/scratch/project_XXXXXXXXX/tilelang'; C=R+'/exp/C1'; P=R+'/fork_p'; O=C+'/patches'
shutil.rmtree(P, ignore_errors=True); shutil.rmtree(O, ignore_errors=True); os.makedirs(O)
sh=lambda c,**k: sp.run(c,shell=True,check=True,universal_newlines=True,stdout=sp.PIPE,stderr=sp.PIPE,**k).stdout
sh(f'git clone -q --shared {R}/src-v0.1.15 {P}'); sh('git checkout -q a35f8ddf45eba16c21211ec8822d56ce5363036f',cwd=P)
env=dict(os.environ,FORK=P+'/')
series=[('0001-rocm-emit-occupancy-from-annotate_min_blocks_per_sm','h1.txt',['e1_c1_edit.py','e4_c1_test_fix.py']),
        ('0002-rocm-fix-8-bit-integer-broadcast','h2.txt',['e2_c9a_edit.py']),
        ('0003-rocm-explain-gemm-limits-on-small-tiles','h3.txt',['e3_c9b_edit.py']),
        ('0004-rocm-add-occupancy-report','h4.txt',['e5_c7_edit.py'])]
for name,h,scripts in series:
    for s in scripts: sp.run(['python3',f'{C}/{s}'],check=True,env=env)
    sh('git add -N testing || true',cwd=P)
    d=sh('git diff -- src tilelang testing',cwd=P)
    open(f'{O}/{name}.patch','w').write(open(f'{C}/{h}').read()+'\n'+d)
    print(name, d.count('\n+')-d.count('\n+++'), '+', d.count('\n-')-d.count('\n---'), '-')
    sh('git add -A -- src tilelang testing',cwd=P)
# Check: the series applies in order on a clean tree and matches the final tree
sh('git diff --cached > /tmp/c1_final.diff; git reset -q --hard a35f8ddf45eba16c21211ec8822d56ce5363036f; git clean -fdq',cwd=P)
for name,_,_ in series: sh(f'git apply --index {O}/{name}.patch',cwd=P)
print('series applies; identical to direct edits:', sh('git diff --cached',cwd=P)==open('/tmp/c1_final.diff').read())
os.remove('/tmp/c1_final.diff'); shutil.rmtree(P)
