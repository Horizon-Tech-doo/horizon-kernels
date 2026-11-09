#!/bin/bash
# inside container, cwd exp/E1
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
python gen_e1.py || exit 1
for f in solo/*.cu ck/*.cu; do b=${f%.cu}; ( $HC -c -x hip $f -o $b.co 2>&1 | grep -E "error" ; $HC -S -x hip $f -o $b.s 2>&1 | grep -E "error" ) & 
  while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 0.3; done; done; wait
$HC -c -x hip combined.cu -o combined.co 2>&1 | grep -E "error"; $HC -S -x hip combined.cu -o combined.s 2>&1 | grep -E "error"
# the MB combined TU, unchanged source, for the provenance of the 192/Occ2 claim
$HC -S -x hip mb_orig/kernels.cu -o mb_orig/kernels_re.s 2>&1 | grep -E "error"
$HC -c -x hip mb_orig/kernels.cu -o mb_orig/kernels.co 2>&1 | grep -E "error"
cat > res.py <<'P'
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
P
echo "file kernel V A Total Scratch Occupancy"
python res.py solo/*.s ck/*.s combined.s
grep -E "^(b12864|b25632|rr12864|rr25632|b25632w2|rr12864w3|rr25632w2|b12864_8k|b25632w2_8k|b12864_32k|b25632w2_32k|rr12864_8kw3|rr25632_8kw2) " /dev/null
python res.py mb_orig/kernels_re.s | grep -E " (b12864|b25632w2|rr12864w3|rr25632w2|b12864_8k|b25632w2_8k|b12864_32k|b25632w2_32k|rr12864_8kw3|rr25632_8kw2) "
python res.py mb_orig/kernels.s | grep -E " (b12864|b25632w2|rr12864w3|rr25632w2|b12864_8k|b25632w2_8k|b12864_32k|b25632w2_32k|rr12864_8kw3|rr25632_8kw2) "
