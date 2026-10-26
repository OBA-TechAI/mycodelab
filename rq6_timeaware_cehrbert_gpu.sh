#!/bin/bash
#SBATCH --job-name=RQ6_TA_CEHR
#SBATCH --partition=ampere24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=rq6_timeaware_cehrbert_%j.out
#SBATCH --error=rq6_timeaware_cehrbert_%j.err

cd ~/mycodelab/mycodelab/DN_Project || exit 1

module load PyTorch/Python3.10

python3 -u rq6_timeaware_cehrbert_importance.py

