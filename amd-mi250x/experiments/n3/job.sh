#!/bin/bash
#SBATCH --job-name=n3-dec
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=00:12:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
# usage: sbatch job.sh <tag> B L rounds iters
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E
tag=$1; B=$2; L=$3; RO=$4; IT=$5
echo "== job $SLURM_JOB_ID tag $tag B=$B L=$L $(date +%T)"; rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
srun --ntasks=1 /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd $E; for seq in 0 1; do echo '## seq='\$seq; ./n3 gen/$tag.co gen/$tag.tsv $B $L $RO $IT \$seq; done" 2>&1 | grep -v "Warning" | sed "s/nid[0-9]*/NODE/g" | tee raw/$tag-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"; rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
