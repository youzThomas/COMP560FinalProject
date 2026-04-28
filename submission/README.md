# COMP 560 Final Submission Package

This directory is a **self-contained** snapshot of our Object Newness Perception
experiments on the Mill tool-wear (signal) dataset. Unzip or copy it so that
`model.py` sits at the **root of what you open in a terminal** (do not add an
extra nesting level, or Python will not find `src` and the default checkpoint
paths will break).

**Tested environment:** Python 3.10–3.12 and PyTorch 2.1+ (including 2.6+). On
PyTorch 2.6 and later, `model.py` loads checkpoints with
`torch.load(..., weights_only=False)` so that full training checkpoints
(including any pickled metadata) load correctly.

---

## Directory layout

| Path | Purpose |
|------|---------|
| `model.py` | **Course entry point:** `StudentModel` for loading weights and running inference. |
| `requirements.txt` | Pinned-style dependency list (`pip install -r requirements.txt`). |
| `configs/default.yaml` | Full hyperparameters, data paths, and class split for the submitted run. |
| `configs/quick_preview.yaml` | Shorter run for quick pipeline checks (optional). |
| `src/` | Library code: `models/` (transformer + newness head), `losses/`, `data/`, `training/`, `evaluation/`, `utils/`. |
| `scripts/train.py` | End-to-end training; writes to `training.ckpt_dir` in the config. |
| `scripts/evaluate.py` | Metrics on val/test; optional threshold sweep. |
| `scripts/smoke_test.py` | Fast random-tensor test of model + loss + metrics (no dataset read). |
| `scripts/threshold_sweep.py` | Grid over objectness/newness thresholds on a checkpoint. |
| `scripts/ensemble_predict.py` | Optional ensemble helper over multiple checkpoints. |
| `runs/default/` | Checkpoints, training history, and JSON reports for the default config. |
| `data/raw/` | Source `dataset_a.mat`. |
| `data/processed/` | HDF5 windows, labels CSV, and train/val/test splits used by the code. |
| `data_loader_eval.py` | Standalone loader reference from the project baseline. |
| `longleaf_train.pbs` / `LONGLEAF_RUNBOOK.md` | HPC (UNC Longleaf) job and notes—optional; not required for local reproduction. |

### `runs/default/` artifacts (default submit)

- **`best_ema.pt`** — EMA-smoothed weights; **recommended** for `StudentModel` and
  the primary reported numbers (see `test_report.json`).
- **`best.pt`**, **`last.pt`** — Best non-EMA and last epoch, for ablation or resume.
- **`history.json`** — Per-epoch train/val losses and scores.
- **`test_report.json`**, **`eval_val_test.json`** — Open-world metrics on held-out splits.
- **`threshold_sweep.json`**, **`threshold_sweep_ema.json`** — Threshold grids used
  for operating-point selection.

---

## Environment setup

From **this directory** (the folder that contains `model.py`):

**venv (Linux / macOS)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**venv (Windows, PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Conda (any OS)**

```bash
conda create -n comp560-submission python=3.11 -y
conda activate comp560-submission
pip install -r requirements.txt
```

If `torch` fails to install, follow the [PyTorch install selector](https://pytorch.org/get-started/locally/)
for your OS and CUDA/CPU, then run `pip install -r requirements.txt` again
(`pip` will keep other packages in sync).

---

## Quick verification (no training)

Sanity-check imports and the full graph without reading HDF5 data:

```bash
python scripts/smoke_test.py
```

---

## Training (optional reproduction)

Re-runs use the same seed and paths as the submitted config:

```bash
python scripts/train.py --config configs/default.yaml
```

Checkpoints and logs are written to `runs/default/` (see `training.ckpt_dir` in
`configs/default.yaml`). Training is long; use the **pre-bundled** `runs/default/*.pt`
if you only need to **evaluate** or run **`StudentModel`**.

---

## Evaluation (metrics + optional threshold sweep)

Example: test split, using the EMA checkpoint, with a threshold sweep:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --split test \
  --sweep
```

Use `--split val` for validation. Omit `--sweep` for a single run at the
`objectness_threshold` / `newness_threshold` values in the config.

---

## `model.py` and `StudentModel` (grader / local API)

`StudentModel` builds `NewnessTransformer` from `configs/default.yaml` (unless
you pass another YAML path), loads the checkpoint, and runs inference on
**float32** windows of shape:

- one sample: `[T, C]` with `T = window_size` and `C = in_channels` (in the
  default submit: **T = 64**, **C = 6**),
- or a batch: `[B, T, C]`.

Default checkpoint and config are **relative to the directory that contains
`model.py`:**

```text
checkpoint_path=runs/default/best_ema.pt
config_path=configs/default.yaml
```

### Example

```python
import numpy as np
from model import StudentModel

model = StudentModel(
    checkpoint_path="runs/default/best_ema.pt",
    config_path="configs/default.yaml",
    device="auto",  # or "cpu" / "cuda"; config's device used if "auto"
)

x = np.random.randn(64, 6).astype("float32")  # [T, C]
pred = model.predict(x)
```

`predict(...)` returns a `dict` (tensor values on **CPU** for easy inspection).
Typical keys include:

| Key | Role |
|-----|------|
| `pred` | Integer class id per window in batch (see label convention below). |
| `batch_size` | `B` for batched input. |
| `obj_prob`, `newness`, `class_logits` | Intermediate scores (shape depends on batch and queries). |
| `is_unknown`, `is_foreground` | Boolean masks from the dual-head + thresholds. |
| `best_query` | Which object query was selected for the prediction. |

Override thresholds for a one-off run without editing the config:

```python
pred = model.predict(x, objectness_threshold=0.5, newness_threshold=0.0)
```

`forward(x)` returns raw model outputs (dict of tensors) before
`NewnessTransformer.predict` post-processing.

### Label convention

- **`0`, `1`**: known tool states (see `data.known_classes` in the YAML).
- **`-1`**: not assigned to a known class (treated as unknown / open-world
  background in the project evaluation code).

This matches the open-world setup in `configs/default.yaml`: known classes
`[0, 1]`, unknown/held-out `[2]`.

---

## Reproducibility and experiment identity

- **Global seed:** `seed: 42` in `configs/default.yaml` (data splits and
  `src.utils.seed` usage depend on this).
- **Data:** Windows are built from `data/raw/dataset_a.mat` and
  `data/processed/labels_with_tool_class.csv` with the splits and
  `known_classes` / `unknown_classes` in the same file.
- **Checkpoints in `runs/default/`** were produced with this config; swapping
  config or data paths without retraining will not match the published JSON
  reports.

For questions about HPC job submission, see `LONGLEAF_RUNBOOK.md` (not required
for a standard laptop reproduction).
