#!/bin/bash
#SBATCH --job-name=ETHOS_BOOT
#SBATCH --partition=cpu32
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=output/ethos_boot_%j.out
#SBATCH --error=output/ethos_boot_%j.err

set -e

module purge
module load PyTorch/Python3.10

cd ~/mycodelab/mycodelab/DN_Project

python -u paired_bootstrap_ethos_vs_all_5seed.py
