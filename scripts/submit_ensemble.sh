#!/usr/bin/env bash
# Submit N training jobs with different seeds, each writing to its own
# checkpoint directory, so their outputs can be ensembled afterwards via
# project-oncp/scripts/ensemble_predict.py.
#
# Usage:
#   ./scripts/submit_ensemble.sh [NUM_SEEDS] [CONFIG_PATH]
#
# Defaults: NUM_SEEDS=3, CONFIG_PATH=configs/default.yaml
#
# After all jobs finish, run (from project-oncp/):
#   python scripts/ensemble_predict.py \
#       --config configs/default.yaml \
#       --run-dirs runs/seed1 runs/seed2 runs/seed3 \
#       --out runs/ensemble_seed123/test_report.json

set -euo pipefail

NUM_SEEDS="${1:-3}"
CONFIG_PATH="${2:-configs/default.yaml}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PBS_SCRIPT="${REPO_DIR}/longleaf_train.pbs"

if [[ ! -f "${PBS_SCRIPT}" ]]; then
  echo "PBS script not found at ${PBS_SCRIPT}" >&2
  exit 1
fi

for (( seed=1; seed<=NUM_SEEDS; seed++ )); do
  ckpt_dir="runs/seed${seed}"
  job_name="COMP560-oncp-seed${seed}"
  train_args="--config ${CONFIG_PATH} --seed ${seed} --ckpt-dir ${ckpt_dir}"
  echo "Submitting seed ${seed}: TRAIN_ARGS=\"${train_args}\""
  sbatch \
    --job-name="${job_name}" \
    --export="ALL,TRAIN_ARGS=${train_args}" \
    "${PBS_SCRIPT}"
done
