#!/bin/bash

#SBATCH --nodes=1
#SBATCJ --ntaks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32g 
#SBATCH --gres=gpumem:16g
#SBATCH --time=10:00:00
#SBATCH -o "slurm-output/slurm-%j.out"

export RUN_NAME="Detic_convnet_base_laion_400m_run_1"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "counted $gpu_count GPUS"

nvidia-smi

nvidia-smi -L

## Load the needed modules
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy cuda/11.7.0 eth_proxy
## Activate the virtualenv created earlier
# source /cluster/project/zhang/umarka/clip_detector/dev/Detic/jupyter_kernel/bin/activate
source /cluster/project/zhang/umarka/clip_detector/dev/segment-anything/notebooks/.tf_torch/bin/activate
python -u train_cl.py

