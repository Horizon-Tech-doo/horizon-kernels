#!/bin/bash
#SBATCH --job-name=gm-probe
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
# PROFILE / GEMM gate on one GCD: (1) what a job may read and set of the card's power and clock controls (nothing is written), (2) which BLAS library torch calls for the server's linear shapes,
# (3) gm_probe.py: the server's GEMM path against gemm_prefetch.py at the server's shapes.
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/PROFILE; VI=$E/vi; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export TILELANG_CACHE_DIR=$E/cache-tl; mkdir -p $TILELANG_CACHE_DIR
md5sum $VI/gm-src/gemm_prefetch.py $VI/gm_probe.py $VI/gm_probe.sbatch | sed "s# .*/# #" | tee $O/header.txt
echo "## power and clock controls $(date +%T)"
{ id -un >/dev/null; rocm-smi --showpower --showmaxpower --showperflevel --showclocks --showoverdrive --showpowerplay 2>&1 | grep -v "^=\|^$" | head -60
  for b in $(rocm-smi --showbus --json 2>/dev/null | python3 -c "import json,sys; j=json.load(sys.stdin); print(' '.join(j[k]['PCI Bus'].lower() for k in sorted(j) if k.startswith('card')))"); do
    d=/sys/bus/pci/devices/$b
    for f in $d/hwmon/hwmon*/power1_cap $d/hwmon/hwmon*/power1_cap_max $d/hwmon/hwmon*/power1_cap_min $d/hwmon/hwmon*/power1_cap_default $d/power_dpm_force_performance_level $d/pp_dpm_sclk $d/pp_od_clk_voltage $d/pp_power_profile_mode $d/pp_sclk_od; do
      [ -e $f ] && echo "$(basename $f) mode=$(stat -c %A $f) user=$(stat -c %U $f) writable_by_me=$([ -w $f ] && echo yes || echo no) value=$(head -c 200 $f 2>/dev/null | tr '\n' ' ')"
    done
  done
  ls -l /sys/cray/pm_counters/ 2>/dev/null | grep -i cap; for f in /sys/cray/pm_counters/*cap*; do [ -e $f ] && echo "$f writable_by_me=$([ -w $f ] && echo yes || echo no) value=$(cat $f 2>/dev/null)"; done
  which amd-smi rocm-smi 2>&1; amd-smi set --help 2>&1 | head -30
  true; } > $O/power-controls.txt 2>&1
grep -c . $O/power-controls.txt; grep "writable_by_me" $O/power-controls.txt | cut -c1-200
X="singularity exec -B /pfs,/scratch,/flash $SIF bash -c"
echo "## liblog $(date +%T)"
$X "export PYTHONNOUSERSITE=1 HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_LEVEL=5 ROCBLAS_LAYER=2 GM_LIBLOG=1; python3 $VI/gm_probe.py /dev/null" > $O/liblog.txt 2>&1; echo "rc=$? lines $(wc -l < $O/liblog.txt)"
grep -c "hipblaslt" $O/liblog.txt; grep -c "rocblas" $O/liblog.txt; grep -B1 -A3 "^## liblog" $O/liblog.txt | cut -c1-260 | head -80
echo "## probe $(date +%T)"
timeout 40m $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/gm-src:$SITE; python3 $VI/gm_probe.py $O/gm.jsonl ${GM_ARGS:-}" > $O/gm.log 2>&1; echo "## probe rc=$? $(date +%T)"; grep -c '"kind": "cell"' $O/gm.jsonl; tail -3 $O/gm.log | cut -c1-400
echo "## job end $(date +%T)"
