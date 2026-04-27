# Longleaf Runbook (Ready-to-Run)

This runbook is for launching training/evaluation on Longleaf using the current repository state.

## 1) Preflight on Login Node

```bash
cd /work/users/d/y/dyy12/COMP-560/project-oncp

# Optional: inspect active defaults used by this runbook
cat configs/default.yaml
```

## 2) Submit Training Job

If your PBS/Slurm script is already staged (e.g. `longleaf_train.pbs`), submit it directly:

```bash
sbatch longleaf_train.pbs
```

If the script reads `TRAIN_ARGS`, keep:

```bash
export TRAIN_ARGS="--config configs/default.yaml"
sbatch longleaf_train.pbs
```

## 3) Monitor Logs

```bash
squeue -u "$USER"
tail -f /work/users/d/y/dyy12/COMP-560/logs/COMP560-oncp-<JOB_ID>.out
```

Expected per-epoch logging now includes:

- `known_rec`
- `unk_prec`
- `sel` (threshold selection mode)
- `score` (criteria-aware checkpoint score)

## 4) Evaluate Best Checkpoint

```bash
cd /work/users/d/y/dyy12/COMP-560/project-oncp
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint runs/default/best.pt \
  --split both --sweep
```

## 5) Acceptance Criteria (Project Targets)

From report requirements:

- `known_recall > 0.80`
- `unknown_precision >= 0.35`

Check in:

- `runs/default/test_report.json`
- `runs/default/eval_val_test.json` (or equivalent output name from `evaluate.py`)

## 6) Quick Local Gate (Before Burning GPU Hours)

```bash
cd project-oncp
python scripts/train.py --config configs/quick_preview.yaml
python scripts/evaluate.py \
  --config configs/quick_preview.yaml \
  --checkpoint runs/quick_preview/best.pt \
  --split val --sweep
```

Use this only for directionality. Final decisions should come from full `default.yaml` runs on Longleaf.

