#!/usr/bin/env bash
#SBATCH --job-name=Llamatrain_gpu
#SBATCH --output=output/Llamatrain-gpu-out.txt
#SBATCH --error=output/Llamatrain-gpu-err.txt
#SBATCH --time=00:30:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4

# Move to project directory
cd ~/mycodelab/mycodelab/DN_Project || exit 1

# Activate the verified Llama environment
source llama_env/bin/activate

echo "===== GPU ALLOCATION CHECK ====="
hostname
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "==============================="

# Run the small LoRA training-step test
python -u test_llama_lora_trainstep.py
