declare -A FS
FS[vg0]="-mllvm -amdgpu-mfma-vgpr-form=0"
FS[vg1]="-mllvm -amdgpu-mfma-vgpr-form=1"
FS[clause]="-mllvm -misched=gcn-max-memory-clause"
FS[bias0]="-mllvm -amdgpu-schedule-metric-bias=0"
ORDER=(base vg0 vg1 clause bias0)
