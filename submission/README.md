# COMP 560 Final Submission Package

This folder is a self-contained package for reproducing our Object Newness
Perception results on the Mill tool-wear dataset.

## Included Contents

- `src/`: training, model, loss, data, evaluation, and utility code
- `scripts/`: train/evaluate/smoke-test/threshold-sweep scripts
- `configs/default.yaml`: main experiment configuration
- `model.py`: grader-facing entrypoint with `StudentModel` class
- `runs/default/`: trained checkpoints and reports
  - `best.pt`, `best_ema.pt`, `last.pt`
  - `history.json`, `test_report.json`, evaluation/sweep JSON files
- `data/`: dataset files used for our run
- `requirements.txt`: Python dependencies
- `data_loader_eval.py`: original loader baseline file

## Environment Setup

```bash
cd submission
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Conda alternative:

```bash
conda create -n comp560-submission python=3.11 -y
conda activate comp560-submission
pip install -r requirements.txt
```

## Run Training

```bash
python scripts/train.py --config configs/default.yaml
```

Artifacts are written to `runs/default/` (as configured by `training.ckpt_dir`).

## Run Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --split test --sweep
```

## `model.py` / `StudentModel` Usage

`model.py` exposes the required `StudentModel` class for local model loading and inference.

```python
import numpy as np
from model import StudentModel

model = StudentModel(
    checkpoint_path="runs/default/best_ema.pt",
    config_path="configs/default.yaml",
    device="auto",
)

# one sample with shape [T, C] = [64, 6]
x = np.random.randn(64, 6).astype("float32")
pred = model.predict(x)
print(pred["pred"])
```

Predicted label convention:
- `0, 1`: known classes
- `-1`: unknown/background prediction

## Reproducibility Notes

- Seed control is configured in `configs/default.yaml` (`seed: 42`).
- Default open-world split in config:
  - known classes: `[0, 1]`
  - unknown class: `[2]`
- The checkpoints in `runs/default/` correspond to this setup.
