#!/usr/bin/env bash
#SBATCH --job-name=llama_smalltest
#SBATCH --output=output/llama_smalltest_%j.out
#SBATCH --error=output/llama_smalltest_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4

cd ~/mycodelab/mycodelab/DN_Project || exit 1

source llama_env/bin/activate

echo "===== LLAMA SMALL TEST ====="
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "============================"

python -u train_llama32_3b_lora_365d_control.py
