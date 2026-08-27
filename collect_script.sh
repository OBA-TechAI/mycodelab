#! /usr/bin/env bash
#
#SBATCH --job-name=collect
#SBATCH --output=output/S-collect-out.txt
#SBATCH --error=output/S-collect_err.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=cpu

# load the module
module load Python/Python3.10

# move to work directory
cd ~/mycodelab/mycodelab/DN_Project/

# do the submission
python3 gather_results.py
