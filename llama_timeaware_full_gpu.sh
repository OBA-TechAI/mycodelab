#!/usr/bin/env bash
#SBATCH --job-name=llama32_ta
#SBATCH --output=output/llama32_ta_%j.out
#SBATCH --error=output/llama32_ta_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4


cd ~/mycodelab/mycodelab/DN_Project || exit 1

source llama_env/bin/activate

echo "===== LLAMA 3.2 3B TIME-AWARE FULL TRAINING ====="
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "================================================="

python -u train_llama32_3b_lora_timeaware_365d.py
