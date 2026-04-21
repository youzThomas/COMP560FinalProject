# COMP 560 Final Report: Object Newness Perception in the Open World

**Team:** Thomas You, Yuyang Deng, Junyi Zhang  
**Final result source:** `result/COMP560-oncp-44468619.out`

## Abstract
We study open-world perception, where a model must classify known classes while flagging unseen samples as unknown. Following our midterm proposal, we implemented a Newness Transformer with (1) query-level objectness, (2) known-class classification, and (3) fused newness scoring from energy and prototype distance. On the UC Berkeley Mill tool-wear dataset, we train on classes 0/1 and hold out class 2 as unknown. In the final run (`44468619`), the model achieves **known recall 0.8033** and **unknown precision 0.4360**, meeting the project targets (**known recall > 0.80**, **unknown precision >= 0.35**). We also provide runtime efficiency and error analysis.

## Introduction
### General overview of task and method
Closed-world classifiers assume all test categories were seen during training. In our project setting, that assumption fails because unknown samples are often misclassified with high confidence or absorbed into background. Our goal is to preserve known-class performance while explicitly detecting novelty. Consistent with our midterm report, we decouple prediction into objectness, known-class identity, and newness scoring. Objectness estimates whether a query corresponds to meaningful foreground behavior, known-class identity predicts among seen classes only, and newness scoring rejects unreliable known predictions and emits unknown. Although the midterm framing used image-based language, this repository applies the same logic to 1-D multi-channel sensor windows from the Mill dataset.

### Past approaches and how ours differs
Prior open-set/open-world work (e.g., OpenMax, OWOD, energy-based OOD detection) often emphasizes either post-hoc novelty scoring or classifier uncertainty calibration alone. Our approach combines both uncertainty and feature geometry in one pipeline by using energy-based uncertainty from class logits together with distance-to-prototype geometry from Prototype-Attention Memory (PAM). We then choose operating thresholds on validation data to satisfy acceptance criteria directly, instead of optimizing AUROC alone.

## Methodology
The implementation in `project-oncp/src/` maps directly to our midterm design. The transformer backbone in `src/models/transformer.py` performs 1-D patch embedding and query-based decoding. `src/models/pam.py` maintains class prototypes and computes query-to-prototype distances. `src/models/newness_model.py` produces class logits and objectness logits, computes energy, and fuses normalized energy and distance into a newness score. `src/losses/losses.py` defines the composite training objective with classification cross-entropy, objectness binary cross-entropy, energy margin/rank terms, and prototype contrastive learning. `src/evaluation/metrics.py` implements open-world reporting and criteria-aware threshold selection.

The data protocol follows the project setup and current config. We use known classes `{0,1}` and hold out class `{2}` as unknown. In the final run, split sizes are train `14994`, validation `4053`, and test `4053`. During evaluation, unknown labels are remapped to `-1` to form the open-world decision task.

### Experimental journey from git history
We organized the implementation into seven rounds, each targeting one major bottleneck.

| Round | Commits | Main change | Practical effect |
|---|---|---|---|
| R1 | `6f66a49` | Built initial backbone, PAM path, and train/eval pipeline | Established the end-to-end baseline |
| R2 | `09a6e7d`, `45dddb9` | Revised loss path and added stronger classification/objectness controls | Improved optimization stability |
| R3 | `e38729d`, `078625c` | Refactored loss weights; added class balancing and energy-penalty controls | Reduced coupling between known retention and unknown separation |
| R4 | `2f83621`, `899c94f` | Tuned default energy/loss weights and added quick-preview config | Faster ablation cycles, fewer wasted long runs |
| R5 | `35826a5` | Aligned checkpoint scoring and threshold selection with acceptance criteria | Model selection became target-driven (`known recall`, `unknown precision`) |
| R6 | `e9cc0d9`, `8acb312` | Optimized open-world behavior; added EMA evaluation and rank-margin energy training | Stronger deployment-oriented robustness and checkpoint reliability |
| R7 | `e0c99a0`, `21d1092`, `76a2637` | Added ensemble/selection utilities, Longleaf submission script, and final run artifact | Completed reproducible final workflow and report traceability to run `44468619` |

## Experiments
### Training details
All reported numbers come from `result/COMP560-oncp-44468619.out`, launched by `longleaf_train.pbs` using `configs/default.yaml`. The run used one V100 GPU on `volta-gpu`, with 8 CPU cores and 32 GB RAM. Software was PyTorch `2.5.1+cu121` in conda env `comp560`. The model has 1.23M parameters and was trained with AdamW, cosine learning-rate schedule, and warmup at batch size 128. Early stopping triggered at epoch 37 with patience 10. Final evaluation chose the better checkpoint between `best.pt` and `best_ema.pt` by validation score, then tuned the newness threshold on validation and froze it for test. No teacher-model distillation or Tinker-specific pipeline was used in this final run. CPU inference is supported because `device: auto` falls back to CPU when CUDA is unavailable, though runtime is slower than GPU.

### Results + Analysis
Final test metrics:

| Metric | Value |
|---|---:|
| Known recall | **0.8033** |
| Unknown precision | **0.4360** |
| Unknown recall | 0.2310 |
| Binary accuracy | 0.8919 |
| AUROC (newness) | 0.7059 |
| AUPR (newness) | 0.3709 |
| AUROC (energy) | 0.6869 |
| AUROC (distance) | 0.5666 |
| Threshold mode | `val_tuned:meets_known_and_precision` |

Acceptance check:
Known recall > 0.80 is a pass, and unknown precision >= 0.35 is also a pass.

Error pattern: unknown precision is strong, but unknown recall is lower, meaning the model is conservative and still routes many true unknowns into known classes.

### Efficiency analysis
From final log timestamps, training start to early stop is 267 seconds (4.45 minutes), and training start to final test report is 274 seconds. Across 37 epochs, average epoch time is about 7.1 seconds. This corresponds to roughly 0.074 GPU-hours per full run and approximate training throughput of `(14994 * 37) / 267 ≈ 2078 samples/s`. The model therefore meets acceptance metrics with relatively low compute cost for a 1.23M-parameter architecture. Runtime efficiency is helped by early stopping, EMA-based checkpoint comparison, and acceptance-aware thresholding that reduces manual retuning cycles.

## Conclusion and Discussion
The final implementation remains consistent with our midterm direction and meets both project acceptance targets on run `44468619`. The approach effectively combines objectness, uncertainty, and prototype geometry for open-world behavior on the Mill split. The main limitation is unknown recall (0.231), which indicates conservative unknown prediction. Performance also depends on threshold calibration, and evaluation currently uses a single held-out unknown class, limiting unknown diversity. Future work will focus on improving unknown recall at fixed precision, strengthening outlier synthesis/calibration, and testing robustness with additional held-out unknown classes and domain shifts.

## References
K. J. Joseph et al., "Towards Open World Object Detection," CVPR, 2021.  
W. Liu et al., "Energy-based Out-of-distribution Detection," NeurIPS, 2020.  
A. Bendale and T. Boult, "Towards Open Set Deep Networks," CVPR, 2016.  
