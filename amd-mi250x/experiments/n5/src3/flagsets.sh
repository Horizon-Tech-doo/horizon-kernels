declare -A FS
FS[ilp]="-mllvm -misched=gcn-max-ilp"
FS[igl0]="-mllvm -amdgpu-igrouplp=0"
FS[pad]="-mllvm -amdgpu-mfma-padding-ratio=50"
FS[itilp]="-mllvm -misched=gcn-iterative-ilp"
ORDER=(base ilp igl0 pad itilp)
