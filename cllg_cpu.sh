#! /usr/bin/env bash
#
#SBATCH --job-name=CLILNG
#SBATCH --output=S-res1.txt
#SBATCH --error=S-err1.txt
#
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --partition=cpu

# load the module
module load Python/Python3.10

# move to work diOrectory
cd ~/mycodelab/mycodelab/DN_Project/

# do the submission
python3 -u train_clinical_longformer_365d_control.py
