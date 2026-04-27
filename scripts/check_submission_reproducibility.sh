#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_submission_reproducibility.sh [options]

Tests the submission package in an isolated temporary directory.

Options:
  --source DIR          Submission directory to test (default: submission)
  --python PYTHON      Python executable for creating the venv (default: python3)
  --tolerance FLOAT    Metric comparison tolerance (default: 1e-6)
  --skip-install       Do not run pip install -r requirements.txt
  --use-current-python Run checks with the current Python instead of a venv
  --full-sweep         Also regenerate and compare threshold_sweep_ema.json
  --keep               Keep the temporary directory after the run
  -h, --help           Show this help

Default behavior creates a fresh venv, installs requirements, and runs only:
  1. scripts/smoke_test.py
  2. StudentModel import/load/predict determinism check

No training is run. The optional --full-sweep flag adds:
  3. scripts/threshold_sweep.py on runs/default/best_ema.pt
  4. JSON comparison against runs/default/threshold_sweep_ema.json
EOF
}

SOURCE_DIR="submission"
PYTHON_BIN="python3"
TOLERANCE="1e-6"
SKIP_INSTALL=0
USE_CURRENT_PYTHON=0
FULL_SWEEP=0
KEEP_TMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DIR="${2:?--source requires a directory}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires an executable}"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="${2:?--tolerance requires a float}"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --use-current-python)
      USE_CURRENT_PYTHON=1
      shift
      ;;
    --full-sweep)
      FULL_SWEEP=1
      shift
      ;;
    --keep)
      KEEP_TMP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Submission directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_DIR/requirements.txt" ]]; then
  echo "Missing requirements.txt under $SOURCE_DIR" >&2
  exit 1
fi

if [[ "$FULL_SWEEP" -eq 1 && ! -f "$SOURCE_DIR/runs/default/threshold_sweep_ema.json" ]]; then
  echo "Missing expected sweep: $SOURCE_DIR/runs/default/threshold_sweep_ema.json" >&2
  exit 1
fi

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/comp560-submission-repro.XXXXXX")"
cleanup() {
  if [[ "$KEEP_TMP" -eq 1 ]]; then
    echo "Kept temporary directory: $TMP_ROOT"
  else
    rm -rf "$TMP_ROOT"
  fi
}
trap cleanup EXIT

echo "Copying $SOURCE_DIR to isolated workspace: $TMP_ROOT/package"
mkdir -p "$TMP_ROOT/package"
tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  -C "$SOURCE_DIR" \
  -cf - . | tar -C "$TMP_ROOT/package" -xf -

cd "$TMP_ROOT/package"

if [[ "$USE_CURRENT_PYTHON" -eq 1 ]]; then
  RUN_PYTHON="$PYTHON_BIN"
else
  echo "Creating fresh virtualenv"
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  RUN_PYTHON="python"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  echo "Installing requirements"
  "$RUN_PYTHON" -m pip install --upgrade pip
  "$RUN_PYTHON" -m pip install -r requirements.txt
fi

echo "Python executable: $("$RUN_PYTHON" -c 'import sys; print(sys.executable)')"
echo "Running smoke test"
"$RUN_PYTHON" scripts/smoke_test.py

echo "Checking StudentModel load and deterministic prediction"
"$RUN_PYTHON" - <<'PY'
import numpy as np
import torch

from model import StudentModel

np.random.seed(42)
torch.manual_seed(42)

model = StudentModel(
    checkpoint_path="runs/default/best_ema.pt",
    config_path="configs/default.yaml",
    device="cpu",
)

x = np.random.default_rng(42).normal(size=(64, 6)).astype("float32")
pred_a = model.predict(x)
pred_b = model.predict(x)

for key in ("pred", "obj_prob", "newness", "class_logits", "best_query", "is_foreground", "is_unknown"):
    if key not in pred_a:
        raise AssertionError(f"StudentModel output is missing key: {key}")
    a = pred_a[key]
    b = pred_b[key]
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise AssertionError(f"Non-deterministic tensor output for {key}")
    elif a != b:
        raise AssertionError(f"Non-deterministic output for {key}")

print("StudentModel ok. pred =", pred_a["pred"].tolist())
PY

if [[ "$FULL_SWEEP" -eq 0 ]]; then
  echo "Minimal reproducibility check passed."
  echo "All checks passed."
  exit 0
fi

echo "Regenerating EMA threshold sweep"
"$RUN_PYTHON" scripts/threshold_sweep.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --out runs/default/repro_threshold_sweep_ema.json

echo "Comparing regenerated sweep with saved threshold_sweep_ema.json"
"$RUN_PYTHON" - "$TOLERANCE" <<'PY'
import json
import math
import sys
from pathlib import Path

tol = float(sys.argv[1])
expected_path = Path("runs/default/threshold_sweep_ema.json")
actual_path = Path("runs/default/repro_threshold_sweep_ema.json")

expected = json.loads(expected_path.read_text())
actual = json.loads(actual_path.read_text())

failures: list[str] = []

def compare(exp, got, path="root"):
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            failures.append(f"{path}: expected dict, got {type(got).__name__}")
            return
        if set(exp) != set(got):
            failures.append(f"{path}: expected keys {sorted(exp)}, got {sorted(got)}")
            return
        for key in sorted(exp):
            compare(exp[key], got[key], f"{path}.{key}")
        return

    if isinstance(exp, list):
        if not isinstance(got, list):
            failures.append(f"{path}: expected list, got {type(got).__name__}")
            return
        if len(exp) != len(got):
            failures.append(f"{path}: expected length {len(exp)}, got {len(got)}")
            return
        for idx, (exp_item, got_item) in enumerate(zip(exp, got)):
            compare(exp_item, got_item, f"{path}[{idx}]")
        return

    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        if not math.isclose(float(exp), float(got), rel_tol=tol, abs_tol=tol):
            failures.append(f"{path}: expected {exp}, got {got}")
        return

    if exp != got:
        failures.append(f"{path}: expected {exp!r}, got {got!r}")

compare(expected, actual)

if failures:
    print("Reproducibility check failed:")
    for failure in failures:
        print("  -", failure)
    raise SystemExit(1)

print("Reproducibility check passed.")
print(f"  checkpoint: {actual['checkpoint']}")
print(f"  epoch: {actual['epoch']}")
print(f"  sweep rows: {len(actual['rows'])}")
PY

echo "All checks passed."
