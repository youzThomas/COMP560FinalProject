# Object Newness Perception on the Mill Tool-Wear Dataset

Implementation of the Newness Transformer described in
*"Midterm Report: Object Newness Perception in the Open World"* (COMP 560, Spring
2026, Team Members: Thomas You, Yuyang Deng, Junyi Zhang) applied to the UC Berkeley
Mill tool-wear dataset.

The report frames open-world perception for images; here we adapt the same
architecture to the 1-D multi-channel sensor windows produced by the Mill
dataset. Class 2 (Failed) is held out during training and used as the "unknown"
category at evaluation time, giving a clean open-world benchmark without
collecting additional data.

## Repository layout

```
project-oncp/
├── configs/default.yaml          # all hyper-parameters
├── data/                         # pre-supplied (see below)
│   ├── raw/dataset_a.mat
│   └── processed/labels_with_tool_class.csv
├── data_loader_eval.py           # original starter loader (kept untouched)
├── requirements.txt
├── scripts/
│   ├── smoke_test.py             # forward/backward sanity check on random tensors
│   ├── train.py                  # main training entry point
│   └── evaluate.py               # evaluation + threshold sweep from a checkpoint
└── src/
    ├── data/dataset.py           # Mill windowing + known/unknown split
    ├── models/
    │   ├── transformer.py        # 1-D patch embed + encoder + DETR-style decoder
    │   ├── pam.py                # Prototype-Attention Memory (section 4.2 A)
    │   └── newness_model.py      # dual-head model with energy + distance fusion
    ├── losses/losses.py          # CE + objectness BCE + energy margin + prototype NCE
    ├── training/trainer.py       # training loop with prototype EMA updates
    ├── evaluation/metrics.py     # open-world report + AUROC/AUPR + threshold sweep
    └── utils/{config,logging,seed}.py
```

## Mapping the report to the code

| Report element                              | File / symbol                                                       |
| ------------------------------------------- | ------------------------------------------------------------------- |
| Encoder backbone (ViT/Swin-style)           | `src/models/transformer.py::PatchEmbed1D`, `TransformerEncoderDecoder` |
| DETR-style object queries + decoder         | `TransformerEncoderDecoder.query_embed` + decoder block             |
| Prototype-Attention Memory (PAM)            | `src/models/pam.py::PrototypeAttentionMemory`                       |
| Distance score `D(q_j)` (eq. 2)             | `PrototypeAttentionMemory.forward` → `dist`                         |
| Energy score `E(q_j)` (eq. 1)               | `src/models/newness_model.py::energy_score`                         |
| Fused newness `S_new` (eq. 3)               | `NewnessTransformer.forward` (`fusion_alpha`, z-normalised)         |
| Objectness + newness gate (eq. 4, 5)        | `NewnessTransformer.predict`                                        |
| Bipartite matching (training)               | `src/losses/losses.py::HungarianMatcher`                            |
| Energy-based OOD margin loss                | `NewnessLoss` (`m_in` / `m_out` from the config)                    |
| Open-world metrics (recall / precision / AUROC) | `src/evaluation/metrics.py`                                       |

## Installation

```bash
pip install -r requirements.txt
```

The code is tested on Python 3.11+/3.13 with PyTorch 2.1+ on CPU and CUDA.

## Data

Place the Mill files at:

```
project-oncp/data/raw/dataset_a.mat
project-oncp/data/processed/labels_with_tool_class.csv
```

They are already present in this repository. `data_loader_eval.py` is left
unchanged as a reference baseline.

## Training

```bash
cd project-oncp
python scripts/train.py --config configs/default.yaml
```

The trainer logs per-epoch classification, objectness, prototype and energy
losses alongside validation known-recall, unknown-precision and AUROC on
`newness`. Best and last checkpoints + a JSON history land in
`runs/default/` (configurable via `training.ckpt_dir`).

A forward/backward smoke test on synthetic tensors (no data required):

```bash
python scripts/smoke_test.py
```

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint runs/default/best.pt \
    --split test --sweep
```

The script emits:

* `known_recall`, `per_class_recall` on the known set (target **> 80%**);
* `unknown_precision`, `unknown_recall` on the held-out class (target
  **≥ 35% unknown precision**);
* `auroc_newness`, `aupr_newness`, `auroc_energy`, `auroc_dist` for novelty
  stability analysis;
* if `--sweep` is set, a grid of thresholds to inspect sensitivity to
  `objectness_threshold` and `newness_threshold`.

## Open-world split

Training batches contain only samples whose tool class is in
`data.known_classes` (default `[0, 1]`). The held-out classes in
`data.unknown_classes` (default `[2]`) are split between the validation and
test sets so every evaluation mixes knowns and unknowns. Unknown samples carry
the remapped label `-1` internally; original tool classes are preserved in
`y_orig` for reporting.

## Configuration highlights

All knobs live in `configs/default.yaml`. A few worth calling out:

* `model.num_queries` — number of DETR-style learnable queries. Even though
  each Mill window has a single label, multiple queries give the bipartite
  matcher room to specialise (one "foreground" winner, the rest supervised as
  background / surrogate unknowns).
* `model.fusion_alpha` — weight on energy vs. distance in `S_new` (eq. 3).
* `training.energy_m_in` / `energy_m_out` — energy-margin targets from Liu et
  al. 2020 (reference [2] in the report). Tune these if training and held-out
  energies overlap.
* `training.proto_ema` — EMA momentum for PAM's class-centre updates.

## Reproduction notes

* `seed` is threaded through NumPy, Python, and PyTorch.
* `min-max` statistics are fit on **training-known** samples only; validation
  and test are scaled with the same statistics to avoid leaking information
  about unknown classes.
* Bad cuts `17` and `94` from the CSV are dropped, matching
  `data_loader_eval.py`.

## Target metrics (from the report)

| Metric                | Target      | Where it is computed                      |
| --------------------- | ----------- | ----------------------------------------- |
| Known-class recall    | > 80%       | `openworld_report["known_recall"]`        |
| Unknown precision     | ≥ 35%       | `openworld_report["unknown_precision"]`   |
| Novelty stability     | consistent  | `sweep_thresholds(...)` in `evaluate.py`  |
