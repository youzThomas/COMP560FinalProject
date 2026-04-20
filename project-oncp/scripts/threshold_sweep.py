"""Sweep newness thresholds on val and report both val & test metrics.

Useful for answering questions like "if I lower target_known_recall from 0.815
to 0.80, what test unknown_recall can I get at test unknown_precision >= 0.35
on the existing best.pt (no retrain)?".

Example::

    python scripts/threshold_sweep.py \
        --config configs/default.yaml \
        --checkpoint runs/default/best.pt \
        --out runs/default/threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import build_dataloaders  # noqa: E402
from src.evaluation import collect_predictions, openworld_report  # noqa: E402
from src.models import NewnessTransformer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=80,
        help="Number of newness-threshold candidates to evaluate.",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    logger = get_logger("oncp.sweep")
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    loaders = build_dataloaders(cfg)
    cfg.model.in_channels = loaders["n_channels"]
    device = _resolve_device(cfg.get("device", "auto"))
    logger.info("Device: %s", device)

    model = NewnessTransformer(
        in_channels=int(cfg.model.in_channels),
        window_size=int(loaders["window_size"]),
        num_known_classes=len(loaders["known_classes"]),
        patch_size=int(cfg.model.patch_size),
        d_model=int(cfg.model.d_model),
        n_heads=int(cfg.model.n_heads),
        n_encoder_layers=int(cfg.model.n_encoder_layers),
        n_decoder_layers=int(cfg.model.n_decoder_layers),
        num_queries=int(cfg.model.num_queries),
        dropout=float(cfg.model.dropout),
        pam_prototypes_per_class=int(cfg.model.pam_prototypes_per_class),
        pam_proto_dim=int(cfg.model.pam_proto_dim),
        energy_temperature=float(cfg.model.energy_temperature),
        fusion_alpha=float(cfg.model.fusion_alpha),
        fusion_msp=float(cfg.model.get("fusion_msp", 0.0)),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    logger.info(
        "Loaded %s (epoch %d, is_ema=%s)",
        args.checkpoint,
        int(state.get("epoch", -1)),
        bool(state.get("is_ema", False)),
    )

    val_preds = collect_predictions(model, loaders["val_loader"], device)
    test_preds = collect_predictions(model, loaders["test_loader"], device)

    # Same candidate grid as choose_operating_point.
    candidates = np.unique(
        np.quantile(
            val_preds["newness"], np.linspace(0.005, 0.999, args.n_thresholds)
        )
    )

    rows: list[dict] = []
    obj_thr = float(cfg.model.objectness_threshold)
    for t in candidates:
        t = float(t)
        v = openworld_report(val_preds, loaders["known_classes"], obj_thr, t)
        te = openworld_report(test_preds, loaders["known_classes"], obj_thr, t)
        rows.append(
            {
                "newness_threshold": t,
                "val_known_recall": v["known_recall"],
                "val_unknown_precision": v["unknown_precision"],
                "val_unknown_recall": v["unknown_recall"],
                "test_known_recall": te["known_recall"],
                "test_unknown_precision": te["unknown_precision"],
                "test_unknown_recall": te["unknown_recall"],
            }
        )

    def _safe(x, default=0.0):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if x != x:  # nan
            return default
        return x

    def _pick(rows, min_known: float, min_prec: float) -> dict | None:
        cands = [
            r for r in rows
            if _safe(r["val_known_recall"]) >= min_known
            and _safe(r["val_unknown_precision"]) >= min_prec
        ]
        if not cands:
            return None
        return max(cands, key=lambda r: _safe(r["val_unknown_recall"]))

    summaries: list[dict] = []
    for min_known in [0.80, 0.81, 0.815]:
        for min_prec in [0.35, 0.40, 0.45]:
            pick = _pick(rows, min_known, min_prec)
            summary = {
                "min_val_known_recall": min_known,
                "min_val_unknown_precision": min_prec,
                "picked": pick,
            }
            summaries.append(summary)
            if pick is not None:
                logger.info(
                    "min_known=%.3f min_prec=%.2f -> thr=%.4f "
                    "VAL (k=%.3f p=%.3f r=%.3f)  TEST (k=%.3f p=%.3f r=%.3f)",
                    min_known,
                    min_prec,
                    pick["newness_threshold"],
                    pick["val_known_recall"],
                    pick["val_unknown_precision"],
                    pick["val_unknown_recall"],
                    pick["test_known_recall"],
                    pick["test_unknown_precision"],
                    pick["test_unknown_recall"],
                )
            else:
                logger.info(
                    "min_known=%.3f min_prec=%.2f -> no feasible threshold",
                    min_known,
                    min_prec,
                )

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(state.get("epoch", -1)),
        "rows": rows,
        "pick_by_targets": summaries,
    }
    dest = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        "threshold_sweep.json"
    )
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info("Wrote %s", dest)


if __name__ == "__main__":
    main()
