declare -A FS
FS[ilp]="-mllvm -misched=gcn-max-ilp"
FS[clause]="-mllvm -misched=gcn-max-memory-clause"
FS[pad]="-mllvm -amdgpu-mfma-padding-ratio=50"
