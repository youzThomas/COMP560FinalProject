"""Entry point for training the Newness Transformer on the Mill dataset.

Example (from the ``project-oncp`` directory)::

    python scripts/train.py --config configs/default.yaml

Checkpoints and a JSON training history land in ``training.ckpt_dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data import build_dataloaders  # noqa: E402
from src.models import NewnessTransformer  # noqa: E402
from src.training import Trainer  # noqa: E402
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
    parser.add_argument("--override-epochs", type=int, default=None)
    args = parser.parse_args()

    logger = get_logger("oncp.train")
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    if args.override_epochs is not None:
        cfg.training.epochs = args.override_epochs

    logger.info("Loading Mill dataset ...")
    loaders = build_dataloaders(cfg)
    logger.info(
        "Train/Val/Test sizes: %d / %d / %d (channels=%d, window=%d)",
        len(loaders["train_loader"].dataset),
        len(loaders["val_loader"].dataset),
        len(loaders["test_loader"].dataset),
        loaders["n_channels"],
        loaders["window_size"],
    )
    logger.info("Class counts: %s", json.dumps(loaders["class_counts"], default=int))

    # Patch config with runtime-derived sizes before constructing the model.
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
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", n_params / 1e6)

    trainer = Trainer(cfg, model, loaders, device)
    trainer.fit()

    logger.info("Training complete. Running final evaluation on the test split ...")
    # evaluate_best loads runs/.../best.pt and uses a val-tuned threshold so the
    # final test numbers reflect the best checkpoint, not the last epoch, and do
    # not tune the threshold on the test split itself.
    final = trainer.evaluate_best(loaders["test_loader"], tag="test")
    logger.info("Final test report:\n%s", json.dumps(final, indent=2, default=float))

    with open(Path(cfg.training.ckpt_dir) / "test_report.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=float)


if __name__ == "__main__":
    main()
