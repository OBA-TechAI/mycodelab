#! /usr/bin/env bash
#
#SBATCH --job-name=TEST
#SBATCH --output=S-res.txt
#SBATCH --error=S-err.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --partition=cpu

# load the module
module load Python/Python3.10

# move to work diOrectory
cd ~/mycodelab/mycodelab/DN_Project/

# do the submission
python3 -u mimic_dn_four_models_longitudinal-Copy2.py
sleep 60
