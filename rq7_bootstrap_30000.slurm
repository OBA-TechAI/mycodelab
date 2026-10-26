#!/bin/bash
#SBATCH --job-name=RQ7_bootstrap
#SBATCH --partition=ampere24
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=output/rq7_bootstrap_%j.out
#SBATCH --error=output/rq7_bootstrap_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "===== RQ7 BOOTSTRAP ANALYSIS ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Start: $(date)"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

unset PYTHONPATH
./llama_env/bin/python -u rq7_bootstrap_30000_all26.py

echo "Finish: $(date)"
echo "===== COMPLETED ====="
