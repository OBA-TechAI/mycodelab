O#! /usr/bin/env bash
#
#SBATCH --job-name=MIMIC_GPU
#SBATCH --output=output/S-gpu-out.txt
#SBATCH --error=output/S-gpu-err.txt
#
#SBATCH --time=00:05:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=1

# load the module
module load PyTorch/Python3.10

# move to work directory
cd ~/mycodelab/mycodelab/DN_Project/

# do the submission
python3 -u mimic_dn_four_models_longitudinal-Copy1.py
sleep 60

