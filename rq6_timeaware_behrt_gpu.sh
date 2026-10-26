#!/bin/bash
#SBATCH --job-name=RQ6_TIMEBEHRT
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=rq6_timeaware_behrt_%j.out
#SBATCH --error=rq6_timeaware_behrt_%j.err

cd ~/mycodelab/mycodelab/DN_Project
module load PyTorch/Python3.10

nvidia-smi
python3 -u rq6_timeaware_behrt_importance.py
