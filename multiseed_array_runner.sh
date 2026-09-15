#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"

SCRIPTS=(
    train_behrt_365d_control.py
    train_timeaware_behrt_365d.py

    train_medbert_365d_control.py
    train_timeaware_medbert_365d.py

    train_cehrbert_365d_control.py
    train_timeaware_cehrbert_365d.py

    train_fttransformer_365d_control.py
    train_timeaware_fttransformer_365d.py

    train_transtab_365d_control.py
    train_timeaware_transtab_365d.py

    train_biobert_365d_control.py
    train_timeaware_biobert_365d.py

    train_clinicalbert_365d_control.py
    train_timeaware_clinicalbert_365d.py

    train_clinical_longformer_365d_control.py
    train_timeaware_clinical_longformer_365d.py

    train_lstm_gru_365d_baselines.py
    train_timeaware_lstm_gru_365d.py

    train_structured_baselines_365d.py
)

SEEDS=(43 44 45 46)

NSEEDS=4

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

SCRIPT_INDEX=$(( TASK_ID / NSEEDS ))
SEED_INDEX=$(( TASK_ID % NSEEDS ))

SCRIPT="${SCRIPTS[$SCRIPT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

echo "Array task: ${TASK_ID}"
echo "Training script: ${SCRIPT}"
echo "Seed: ${SEED}"

if [ "${MAP_ONLY:-0}" = "1" ]; then
    exit 0
fi

exec "${ROOT}/run_isolated_seed.sh" "${SCRIPT}" "${SEED}"
