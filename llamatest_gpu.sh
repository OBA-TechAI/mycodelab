#!/usr/bin/env bash
#

#SBATCH --job-name=LLAMAtest_gpu
#SBATCH --output=output/Llamatst-gpu-out.txt
#SBATCH --error=output/Llamatst-gpu-err.txt
#

#SBATCH --time=00:30:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4

# load the module
module load PyTorch/Python3.11

# move to work directory
cd ~/mycodelab/mycodelab/DN_Project

# do the submission
echo "===== GPU ALLOCATION CHECK ====="
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "==============================="

python3.11 test_llama_3b_gpu .py
