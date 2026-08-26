#! /usr/bin/env bash
#
#SBATCH --job-name=MIMIC-CPU
#SBATCH --output=output/S-%a-res.txt
#SBATCH --error=output/S-%a-err.txt
#
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --partition=cpu
#SBATCH --array=1-10   # Array job with 10 tasks

# load the module
module load Python/Python3.10

# move to work directory
cd ~/mycodelab/mycodelab/DN_Project/

data_file='random_state.txt'
# read the i-th line from the file and store it as "n"
n=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $data_file)

echo "Running task ${SLURM_ARRAY_TASK_ID} with random state ${n}"

# do the submission
python3 -u mimic_dn_four_models_longitudinal-Copy2.py --random_state $n
sleep 60
