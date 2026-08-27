#!/usr/bin/env bash
#

#SBATCH --job-name=TA_BEHRT365
#SBATCH --output=output/TW-gpu-out.txt
#SBATCH --error=output/TW-gpu-err.txt
#

#SBATCH --time=04:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=1

# load the module
module load PyTorch/Python3.10

# move to work directory
cd ~/mycodelab/mycodelab/DN_Project

# do the submission
echo "===== GPU ALLOCATION CHECK ====="
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "==============================="

python3.10 -u train_timeaware_behrt_365d.py
# train_behrt_365d_control.py
