#!/bin/bash
#SBATCH --job-name=CB_surv
#SBATCH --output=output/clinicalbert-survival-out.txt
#SBATCH --error=output/clinicalbert-survival-err.txt
#SBATCH --time=01:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=2

set -e

module purge
module load PyTorch/Python3.10

cd ~/mycodelab/mycodelab/DN_Project

mkdir -p output

echo "===== CLINICALBERT SURVIVAL ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export TOKENIZERS_PARALLELISM=false

python3 -u train_survival_clinicalbert_365d.py

echo "===== SCRIPT COMPLETE ====="
