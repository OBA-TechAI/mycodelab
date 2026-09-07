#!/usr/bin/env bash
#

#SBATCH --job-name=TA-LSTMGRU_gpu
#SBATCH --output=output/TA-LSTMGRU-gpu-out.txt
#SBATCH --error=output/TA-LSTMGRU-gpu-err.txt
#

#SBATCH --time=06:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4

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

python3.10 -u train_timeaware_lstm_gru_365d.py
