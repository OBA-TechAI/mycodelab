#!/bin/bash
#SBATCH --job-name=RQ6_LLAMA_CONTROL
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=rq6_llama_control_%j.out
#SBATCH --error=rq6_llama_control_%j.err

cd ~/mycodelab/mycodelab/DN_Project || exit 1

module load PyTorch/Python3.10

./llama_env/bin/python -u rq6_llama_control_importance.py

