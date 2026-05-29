# Results

See the full final report at [`docs/report/final-report.pdf`](report/final-report.pdf).

Final metrics from the saved `best_ema.pt` checkpoint.

| Metric | Value |
| --- | ---: |
| Known-class recall | 0.803299 |
| Class 0 recall | 0.830601 |
| Class 1 recall | 0.785197 |
| Unknown precision | 0.435955 |
| Unknown recall | 0.230952 |
| Binary known/unknown accuracy | 0.891932 |
| Newness AUROC | 0.705859 |
| Newness AUPR | 0.370853 |
| Energy AUROC | 0.686896 |
| Prototype-distance AUROC | 0.566620 |

Selected operating point:

```text
objectness_threshold = 0.5
newness_threshold = -1.497143646276222
threshold_selection_mode = val_tuned:meets_known_and_precision
```

Confusion matrix labels are ordered as `[-1, 0, 1]`:

```text
[[194,  51,  595],
 [  6, 1064, 211],
 [245, 170, 1517]]
```

Reproduce evaluation after placing the checkpoint under `runs/default/`:

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best_ema.pt \
  --split test --sweep
```
