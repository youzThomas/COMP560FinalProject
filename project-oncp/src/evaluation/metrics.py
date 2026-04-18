"""Evaluation utilities for the open-world setting.

Metrics returned mirror the "Expected Results" table in the midterm report:

* Known-class recall / accuracy (target > 80%).
* Unknown precision at a chosen threshold (target >= 35%).
* Full threshold sweep with AUROC and AUPR so we can report novelty stability.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


@torch.no_grad()
def collect_predictions(model, loader, device: str | torch.device) -> dict[str, np.ndarray]:
    """Run the model once over ``loader`` and stash the raw scores we need."""
    model.eval()
    obj_probs, newness, class_logits, energy, dist, y_orig, y = [], [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        out = model(x)
        best_q = out["obj_prob"].argmax(dim=-1, keepdim=True)
        obj_probs.append(out["obj_prob"].gather(1, best_q).squeeze(-1).cpu().numpy())
        newness.append(out["newness"].gather(1, best_q).squeeze(-1).cpu().numpy())
        energy.append(out["energy"].gather(1, best_q).squeeze(-1).cpu().numpy())
        dist.append(out["dist"].gather(1, best_q).squeeze(-1).cpu().numpy())
        class_logits.append(
            out["class_logits"].gather(
                1, best_q.unsqueeze(-1).expand(-1, -1, out["class_logits"].size(-1))
            ).squeeze(1).cpu().numpy()
        )
        y_orig.append(batch["y_orig"].numpy())
        y.append(batch["y"].numpy())
    return {
        "obj_prob": np.concatenate(obj_probs),
        "newness": np.concatenate(newness),
        "energy": np.concatenate(energy),
        "dist": np.concatenate(dist),
        "class_logits": np.concatenate(class_logits),
        "y_orig": np.concatenate(y_orig),
        "y_known": np.concatenate(y),
    }


def unknown_detection_curve(preds: dict[str, np.ndarray]) -> dict[str, float]:
    """AUROC / AUPR of ``newness`` as an unknown-vs-known detector."""
    is_unknown = (preds["y_known"] < 0).astype(np.int32)
    scores = preds["newness"]
    out: dict[str, float] = {}
    if is_unknown.sum() > 0 and is_unknown.sum() < len(is_unknown):
        out["auroc_newness"] = float(roc_auc_score(is_unknown, scores))
        out["aupr_newness"] = float(average_precision_score(is_unknown, scores))
        out["auroc_energy"] = float(roc_auc_score(is_unknown, preds["energy"]))
        out["auroc_dist"] = float(roc_auc_score(is_unknown, preds["dist"]))
    return out


def _apply_decision(
    preds: dict[str, np.ndarray],
    obj_threshold: float,
    newness_threshold: float,
    known_classes: list[int],
) -> np.ndarray:
    """Return predicted "original" class ids, with ``-1`` meaning unknown."""
    is_fg = preds["obj_prob"] > obj_threshold
    is_unknown = is_fg & (preds["newness"] > newness_threshold)
    cls_idx = preds["class_logits"].argmax(axis=-1)
    known_arr = np.asarray(known_classes, dtype=np.int64)
    pred_known = known_arr[cls_idx]
    pred = np.where(is_unknown | ~is_fg, -1, pred_known)
    return pred


def openworld_report(
    preds: dict[str, np.ndarray],
    known_classes: list[int],
    obj_threshold: float,
    newness_threshold: float,
) -> dict:
    """Human-readable + machine-readable evaluation at a given operating point."""
    y_true = preds["y_orig"].astype(np.int64).copy()
    # Merge all held-out classes into the single pseudo-class ``-1``.
    y_true_ow = np.where(np.isin(y_true, known_classes), y_true, -1)
    y_pred = _apply_decision(preds, obj_threshold, newness_threshold, known_classes)

    # Known recall / accuracy (evaluated on the known subset only).
    known_mask = y_true_ow != -1
    if known_mask.any():
        known_recall = float(
            (y_pred[known_mask] == y_true_ow[known_mask]).mean()
        )
        # Per-class recall.
        per_class = {}
        for c in known_classes:
            m = y_true_ow == c
            if m.any():
                per_class[c] = float((y_pred[m] == c).mean())
            else:
                per_class[c] = float("nan")
    else:
        known_recall = float("nan")
        per_class = {}

    # Unknown precision / recall.
    unk_mask = y_true_ow == -1
    pred_unk = y_pred == -1
    if pred_unk.any():
        unk_precision = float((pred_unk & unk_mask).sum() / pred_unk.sum())
    else:
        unk_precision = float("nan")
    if unk_mask.any():
        unk_recall = float((pred_unk & unk_mask).sum() / unk_mask.sum())
    else:
        unk_recall = float("nan")

    # Binary healthy-vs-worn view (to align with the existing eval script).
    bin_true = (y_true > 0).astype(np.int64)
    bin_pred_score = preds["newness"] + preds["obj_prob"] * 0.0  # newness as proxy risk
    # Score for "worn": higher newness OR any non-healthy prediction.
    bin_pred = (y_pred != 0).astype(np.int64)

    cm = confusion_matrix(
        y_true_ow, y_pred, labels=sorted(set(known_classes) | {-1})
    )

    curve = unknown_detection_curve(preds)

    return {
        "known_recall": known_recall,
        "per_class_recall": per_class,
        "unknown_precision": unk_precision,
        "unknown_recall": unk_recall,
        "binary_accuracy": float((bin_pred == bin_true).mean()),
        "confusion_matrix": cm.tolist(),
        "labels_order": sorted(set(known_classes) | {-1}),
        "operating_point": {
            "objectness_threshold": obj_threshold,
            "newness_threshold": newness_threshold,
        },
        **curve,
    }


def sweep_thresholds(
    preds: dict[str, np.ndarray],
    known_classes: list[int],
    obj_thresholds: Iterable[float] = (0.3, 0.4, 0.5, 0.6),
    newness_thresholds: Iterable[float] | None = None,
) -> list[dict]:
    """Evaluate at a grid of thresholds for sensitivity analysis."""
    if newness_thresholds is None:
        quantiles = np.quantile(preds["newness"], np.linspace(0.5, 0.99, 15))
        newness_thresholds = [float(q) for q in quantiles]
    results = []
    for o in obj_thresholds:
        for n in newness_thresholds:
            rep = openworld_report(preds, known_classes, o, n)
            rep.pop("confusion_matrix", None)
            rep.pop("labels_order", None)
            results.append(rep)
    return results


def choose_newness_threshold(
    preds: dict[str, np.ndarray],
    target_unknown_precision: float = 0.35,
    known_classes: list[int] | None = None,
    obj_threshold: float = 0.5,
) -> float:
    """Pick the lowest newness threshold that meets ``target_unknown_precision``.

    Falls back to the median newness if the target is unreachable.
    """
    candidates = np.quantile(preds["newness"], np.linspace(0.05, 0.99, 40))
    best = float(np.median(preds["newness"]))
    for t in candidates:
        rep = openworld_report(preds, known_classes or [], obj_threshold, float(t))
        if not np.isnan(rep["unknown_precision"]) and rep["unknown_precision"] >= target_unknown_precision:
            return float(t)
    return best
