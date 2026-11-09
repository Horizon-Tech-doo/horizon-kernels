import re
src=open('gen_e1.py').read(); exec(src[:src.index("W=lambda n")])
W=lambda n:"__attribute__((amdgpu_waves_per_eu(%d)))"%n
h=hdr(open('src/b12864_4k.cu').read())
def mk(srcn,name,w,isrr=0):
    s=body(open(f'src/{srcn}.cu').read())
    if isrr: s=rr(s)
    open(f'solo/{name}.cu','w').write(h+rename(s,name,W(w) if w else ""))
mk('b12864_16k','b12864_16k',None); mk('b25632_16k','b25632w2_16k',2); mk('r25632_16k','rr25632w2_16k',2,1); mk('r12864_16k','rr12864w3_16k',3,1)
mk('b25632sw8_8k','b25632w2sw8_8k',2); mk('b25632sw8_16k','b25632w2sw8_16k',2)
print(open('src/b25632sw8_8k.cu').read().count('swizzle'))
