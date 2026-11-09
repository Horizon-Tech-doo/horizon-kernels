#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -J N1-prep
# CPU only. Synthetic data, then the Qwen3.6-35B-A3B checkpoint into this track's own folder (shared models folder is left alone).
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
TS=1,2,4,8,16,17,32,64,128,256,1024,2049,2200,2400,2560,2800,3000,3072,3200,3400,3584,3800,4000,4095,4096,4097,6144,7168,8192,8193,12288,16384
$TL "cd $E; python -I $D/n1_mk.py $E/data.pt; python -I $D/n1_mk3.py $E/data.pt $E/data_r6.pt $TS; rm -f $E/data.pt" 2>&1 | grep -v Warning
ls -la $E/data_r6.pt; date
CK=$E/ckpt36; mkdir -p $CK
$TL "python -I -c \"
from huggingface_hub import snapshot_download
import huggingface_hub
p = snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir='$CK', max_workers=8, token=False)
from huggingface_hub import HfApi
print('rev', HfApi().model_info('Qwen/Qwen3.6-35B-A3B', token=False).sha)
print('DONE', p)
\"" 2>&1 | tail -4 | cut -c1-200
du -sh $CK; date
