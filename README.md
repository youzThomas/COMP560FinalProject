# Open-World Tool-Wear Newness Detection

PyTorch implementation of an open-world newness detector for multi-channel
Mill tool-wear sensor windows. The project holds out the failed tool class
during training, then evaluates whether the model can keep high recall on known
tool states while flagging the held-out class as unknown.

This was built for COMP 560 and adapts the Newness Transformer idea from image
perception to 1-D sensor data.

## Highlights

- Transformer encoder/decoder over 1-D multi-channel sensor windows.
- Prototype-Attention Memory for class-centered feature comparison.
- Energy, prototype-distance, and objectness scores for open-world prediction.
- Validation-tuned threshold sweep for known/unknown operating points.
- Reproducible training, evaluation, smoke-test, and Longleaf/HPC run scripts.

## Results

Final test metrics from the saved EMA checkpoint:

| Metric | Value |
| --- | ---: |
| Known-class recall | 0.803 |
| Unknown precision | 0.436 |
| Unknown recall | 0.231 |
| Binary known/unknown accuracy | 0.892 |
| Newness AUROC | 0.706 |
| Energy AUROC | 0.687 |
| Prototype-distance AUROC | 0.567 |

The selected operating point used `objectness_threshold = 0.5` and
`newness_threshold = -1.497`.

## Repository Layout

```text
configs/                  Experiment configs
data/                     Local dataset location; large files are ignored
docs/                     Architecture, results, and Longleaf run notes
scripts/                  Train, evaluate, sweep, smoke-test, and ensemble helpers
src/                      Data, model, loss, training, evaluation, and utilities
model.py                  Simple StudentModel inference wrapper
requirements.txt          Python dependencies
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the Mill dataset files under `data/` as described in
[`data/README.md`](data/README.md).

Run a synthetic forward/backward check:

```bash
python scripts/smoke_test.py
```

Train the default model:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --split test --sweep
```

## Inference API

`model.py` exposes a small `StudentModel` wrapper around the trained model:

```python
import numpy as np
from model import StudentModel

model = StudentModel(
    checkpoint_path="runs/default/best_ema.pt",
    config_path="configs/default.yaml",
    device="auto",
)

x = np.random.randn(64, 6).astype("float32")
prediction = model.predict(x)
print(prediction["pred"])
```

Predictions use `0` and `1` for known classes and `-1` for unknown/background.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) maps the model design to code.
- [`docs/results.md`](docs/results.md) records the saved test metrics.
- [`docs/LONGLEAF_RUNBOOK.md`](docs/LONGLEAF_RUNBOOK.md) covers HPC usage.

## Notes

Large datasets, checkpoints, generated runs, virtual environments, and packaged
submission artifacts are intentionally ignored. Keep those locally or publish
them separately as release assets if needed.
