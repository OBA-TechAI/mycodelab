#!/bin/bash
set -euo pipefail

SCRIPT="$1"
SEED="$2"

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
STEM="${SCRIPT%.py}"
WORK="${ROOT}/multiseed_runs/${STEM}/seed_${SEED}"

echo "=============================================="
echo "Script: ${SCRIPT}"
echo "Seed:   ${SEED}"
echo "Work:   ${WORK}"
echo "Host:   $(hostname)"
echo "=============================================="

if [ -f "${WORK}/_SUCCESS" ]; then
    echo "This run is already complete. Nothing to do."
    exit 0
fi

if [ -d "${WORK}" ]; then
    echo "ERROR: Run directory already exists but is incomplete:"
    echo "${WORK}"
    echo "Refusing to overwrite it."
    exit 2
fi

mkdir -p "${WORK}/output"

# ------------------------------------------------------------
# Link input datasets into the isolated working directory
# ------------------------------------------------------------
find "${ROOT}" -maxdepth 1 -type f \
    \( -name "*.parquet" -o -name "*.csv.gz" \) \
    -print0 |
while IFS= read -r -d '' f
do
    ln -s "$f" "${WORK}/$(basename "$f")"
done

# ------------------------------------------------------------
# Link helper Python files if any script imports local code
# ------------------------------------------------------------
find "${ROOT}" -maxdepth 1 -type f -name "*.py" -print0 |
while IFS= read -r -d '' f
do
    name="$(basename "$f")"

    if [ "$name" != "$SCRIPT" ]; then
        ln -s "$f" "${WORK}/${name}"
    fi
done

# ------------------------------------------------------------
# Link local pretrained-model directories when available
# ------------------------------------------------------------
for d in \
    BioBERT \
    Bio_ClinicalBERT \
    Clinical-Longformer \
    TransTab \
    transtab
do
    if [ -e "${ROOT}/${d}" ]; then
        ln -s "${ROOT}/${d}" "${WORK}/${d}"
    fi
done

# ------------------------------------------------------------
# Copy only this training script and replace Seed 42
# ------------------------------------------------------------
cp "${ROOT}/${SCRIPT}" "${WORK}/${SCRIPT}"

python3.10 - "${WORK}/${SCRIPT}" "${SEED}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
seed = int(sys.argv[2])

text = path.read_text()

text, n = re.subn(
    r"(?m)^SEED\s*=\s*42\s*$",
    f"SEED = {seed}",
    text,
    count=1
)

if n != 1:
    raise RuntimeError(
        f"Expected exactly one 'SEED = 42' in {path}; found {n}"
    )

compile(text, str(path), "exec")
path.write_text(text)

print(f"Prepared {path.name} with SEED = {seed}")
PY

# ------------------------------------------------------------
# Record hardware information
# ------------------------------------------------------------
{
    echo "Script=${SCRIPT}"
    echo "Seed=${SEED}"
    echo "Host=$(hostname)"
    echo "Date=$(date)"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader 2>/dev/null || true
} > "${WORK}/run_metadata.txt"

cd "${WORK}"

echo
echo "Starting training..."
echo

python3.10 -u "${SCRIPT}"

touch "${WORK}/_SUCCESS"

echo
echo "SUCCESS: ${SCRIPT}, seed ${SEED}"
