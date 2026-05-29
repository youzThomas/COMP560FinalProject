# Open-World Tool-Wear Newness Detection

**University of North Carolina at Chapel Hill**

**COMP 560 Artificial Intelligence - Final Project**  
**Mentored by:** [Prof. Tianlong Chen](https://tianlong-chen.github.io)  
**Author:** Thomas You, Yuyang Deng, Junyi Zhang  
**Task:** Open-world recognition for multi-channel mill tool-wear sensor data  
**Report:** [`docs/report/final-report.pdf`](docs/report/final-report.pdf)

## Abstract

This repository studies open-world tool-wear recognition on the UC Berkeley Mill
dataset. The central problem is to classify known tool states while detecting a
held-out failure state as novel at evaluation time. We adapt a Newness
Transformer-style architecture from visual open-world perception to 1-D
multi-channel sensor windows. The model combines a transformer encoder/decoder,
learnable object queries, Prototype-Attention Memory, free-energy scoring, and a
validation-tuned newness threshold.

In the default split, classes `0` and `1` are treated as known during training,
while class `2` is held out as the unknown/failure class for validation and test.
The final saved EMA checkpoint achieves `0.803` known-class recall and `0.436`
unknown precision on the test split.

## Problem Setting

Open-world recognition assumes that a deployed model can encounter classes that
were absent during training. For tool-wear monitoring, this matters because
failure conditions may be rare, expensive to collect, or missing from the
training distribution. This project evaluates whether a model trained on known
tool states can preserve known-class recall while assigning high newness scores
to held-out failure examples.

## Method

The model operates on fixed-length sensor windows with shape `[T, C]`, where the
default configuration uses `T = 64` and `C = 6`.

1. **Windowing and split construction:** `src/data/dataset.py` builds sensor
   windows, applies the known/unknown split, and preserves original labels for
   evaluation.
2. **Transformer feature extraction:** `src/models/transformer.py` embeds 1-D
   patches and applies a transformer encoder/decoder with learnable queries.
3. **Prototype comparison:** `src/models/pam.py` stores class prototypes and
   computes prototype-distance signals for each query.
4. **Newness scoring:** `src/models/newness_model.py` fuses free energy,
   prototype distance, and max-softmax probability into a per-query newness
   score.
5. **Open-world decision rule:** evaluation selects an operating point using
   validation thresholds, then reports known recall and unknown detection
   metrics on held-out data.

## Experimental Setup

| Component | Default |
| --- | --- |
| Dataset | UC Berkeley Mill tool-wear dataset |
| Input representation | 1-D multi-channel sensor windows |
| Known training classes | `[0, 1]` |
| Held-out unknown class | `[2]` |
| Window size | `64` |
| Model family | Transformer encoder/decoder + Prototype-Attention Memory |
| Main config | [`configs/default.yaml`](configs/default.yaml) |
| Quick config | [`configs/quick_preview.yaml`](configs/quick_preview.yaml) |

Large data files and checkpoints are not tracked in Git. See
[`data/README.md`](data/README.md) for expected local dataset paths.

## Results

Final test metrics from the saved EMA checkpoint:

| Metric | Value |
| --- | ---: |
| Known-class recall | 0.803 |
| Class 0 recall | 0.831 |
| Class 1 recall | 0.785 |
| Unknown precision | 0.436 |
| Unknown recall | 0.231 |
| Binary known/unknown accuracy | 0.892 |
| Newness AUROC | 0.706 |
| Newness AUPR | 0.371 |
| Energy AUROC | 0.687 |
| Prototype-distance AUROC | 0.567 |

The selected test operating point used:

```text
objectness_threshold = 0.5
newness_threshold = -1.497143646276222
threshold_selection_mode = val_tuned:meets_known_and_precision
```

Additional result details are recorded in [`docs/results.md`](docs/results.md).

## Repository Structure

```text
configs/                  Experiment configurations
data/                     Local dataset placeholders and data instructions
docs/                     Final report, architecture notes, results, runbook
scripts/                  Training, evaluation, threshold sweep, smoke test
src/                      Data, model, loss, training, evaluation, utilities
model.py                  StudentModel inference wrapper
requirements.txt          Python dependencies
```

## Reproducibility

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a synthetic forward/backward smoke test:

```bash
python scripts/smoke_test.py
```

Train with the default configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Evaluate a saved checkpoint:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --split test --sweep
```

For HPC execution, see [`docs/LONGLEAF_RUNBOOK.md`](docs/LONGLEAF_RUNBOOK.md).

## Inference Interface

`model.py` exposes a compact `StudentModel` API for checkpoint loading and
single-window or batched prediction.

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

Predicted labels use `0` and `1` for known classes and `-1` for
unknown/background predictions.

## Documentation

- Final report: [`docs/report/final-report.pdf`](docs/report/final-report.pdf)
- Architecture notes: [`docs/architecture.md`](docs/architecture.md)
- Results: [`docs/results.md`](docs/results.md)
- Longleaf runbook: [`docs/LONGLEAF_RUNBOOK.md`](docs/LONGLEAF_RUNBOOK.md)
