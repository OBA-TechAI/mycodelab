#!/usr/bin/env bash
#

#SBATCH --job-name=TWMEDB_gpu
#SBATCH --output=output/TWMED-gpu-out.txt
#SBATCH --error=output/TWMED-gpu-err.txt
#

#SBATCH --time=06:00:00
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

python3.10 -u train_timeaware_medbert_365d.py
