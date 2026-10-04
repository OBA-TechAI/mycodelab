#!/bin/bash
#SBATCH --job-name=behrt_ctrl_surv5
#SBATCH --output=output/behrt_ctrl_surv5_%j.out
#SBATCH --error=output/behrt_ctrl_surv5_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ampere24
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -e

module purge
module load PyTorch/Python3.10

cd ~/mycodelab/mycodelab/DN_Project

echo "===== BEHRT CONTROL SURVIVAL 5-SEED ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

for SEED in 42 43 44 45 46
do
    echo "========================================"
    echo "RUNNING SEED $SEED"
    echo "========================================"

    export SURVIVAL_SEED=$SEED
    python3 -u train_survival_behrt_control_365d.py
done

echo "===== ALL 5 SEEDS COMPLETE ====="
