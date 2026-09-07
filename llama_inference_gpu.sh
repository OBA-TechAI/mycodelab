#!/bin/bash
#SBATCH --job-name=llama_inference
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#SBATCH --time=00:20:00
#SBATCH --output=llama_inference_%j.out
#SBATCH --error=llama_inference_%j.err

cd /bigdata/users/22053430/mycodelab/mycodelab/DN_Project || exit 1

source llama_env/bin/activate

echo "Python: $(which python)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -u test_llama_load.py
