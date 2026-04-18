"""Evaluate a trained Newness Transformer checkpoint on val / test splits.

Also produces a threshold sweep (objectness x newness) for the sensitivity
analysis called out in section 3 of the midterm report.
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
from src.evaluation import (  # noqa: E402
    collect_predictions,
    openworld_report,
    sweep_thresholds,
)
from src.evaluation.metrics import choose_newness_threshold  # noqa: E402
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
    parser.add_argument("--split", choices=["val", "test", "both"], default="test")
    parser.add_argument("--sweep", action="store_true", help="Emit a threshold sweep JSON.")
    args = parser.parse_args()

    logger = get_logger("oncp.eval")
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))

    loaders = build_dataloaders(cfg)
    cfg.model.in_channels = loaders["n_channels"]
    device = _resolve_device(cfg.get("device", "auto"))

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
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    logger.info("Loaded checkpoint from %s (epoch %d)", args.checkpoint, state.get("epoch", -1))

    splits = [args.split] if args.split != "both" else ["val", "test"]
    out: dict = {}
    for s in splits:
        loader = loaders[f"{s}_loader"]
        preds = collect_predictions(model, loader, device)
        newness_thr = choose_newness_threshold(
            preds,
            target_unknown_precision=0.35,
            known_classes=loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
        )
        rep = openworld_report(
            preds, loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
            newness_threshold=newness_thr,
        )
        rep["tuned_newness_threshold"] = newness_thr
        out[s] = rep
        logger.info("[%s] %s", s, json.dumps(rep, indent=2, default=float))

        if args.sweep:
            sweep = sweep_thresholds(preds, loaders["known_classes"])
            out[f"{s}_sweep"] = sweep

    dest = Path(args.checkpoint).with_name(f"eval_{'_'.join(splits)}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    logger.info("Wrote %s", dest)


if __name__ == "__main__":
    main()
