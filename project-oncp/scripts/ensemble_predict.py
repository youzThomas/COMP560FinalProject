"""Ensemble inference for the Newness Transformer.

Given N run directories (each produced by ``scripts/train.py`` with a distinct
``--seed`` and ``--ckpt-dir``), load each best checkpoint, collect per-sample
``newness`` / ``energy`` / ``dist`` / ``obj_prob`` / ``class_logits`` on val and
test, average them across models, tune the newness threshold on the averaged val
predictions, freeze it, and produce a final test report.

This is the cheapest AUROC booster available -- averaging rarely hurts on OOD
detection because each model's calibration noise cancels while the shared
signal survives.

Example (from the ``project-oncp`` directory)::

    python scripts/ensemble_predict.py \
        --config configs/default.yaml \
        --run-dirs runs/seed1 runs/seed2 runs/seed3 \
        --use ema_or_best \
        --out runs/ensemble_seed123/test_report.json

``--use`` choices:

* ``best``         -- always load ``best.pt`` from each run.
* ``best_ema``     -- always load ``best_ema.pt`` from each run.
* ``ema_or_best``  -- pick whichever of {best, best_ema} scored higher on val
                      during the run (mirrors ``Trainer.evaluate_best``).

The threshold is tuned on the averaged val predictions and frozen before test,
so the ensemble respects the same no-test-leakage discipline as single-model
evaluation.
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
    choose_operating_point,
    collect_predictions,
    openworld_report,
)
from src.models import NewnessTransformer  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402


def _resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def _build_model(cfg, loaders, device) -> NewnessTransformer:
    cfg.model.in_channels = loaders["n_channels"]
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
    return model


def _score_report(rep: dict, t_known: float, t_prec: float, t_rec: float) -> float:
    """Same checkpoint score the trainer uses, re-implemented to avoid a circular import."""

    import math

    def _safe(x) -> float:
        if x is None:
            return 0.0
        x = float(x)
        return 0.0 if math.isnan(x) else x

    known = _safe(rep.get("known_recall"))
    unk_prec = _safe(rep.get("unknown_precision"))
    unk_rec = _safe(rep.get("unknown_recall"))
    auroc_raw = rep.get("auroc_newness")
    auroc = 0.5
    if auroc_raw is not None:
        try:
            av = float(auroc_raw)
            if not math.isnan(av):
                auroc = av
        except (TypeError, ValueError):
            auroc = 0.5
    s = (
        0.50 * known
        + 0.25 * unk_prec
        + 0.15 * unk_rec
        + 0.40 * max(0.0, auroc - 0.5)
    )
    s -= 3.0 * max(0.0, t_known - known)
    s -= 3.0 * max(0.0, t_prec - unk_prec)
    s -= 0.5 * max(0.0, t_rec - unk_rec)
    return s


def _pick_checkpoint(
    run_dir: Path,
    mode: str,
    cfg,
    loaders,
    device,
    logger,
) -> Path:
    """Return the path to the checkpoint we should load from ``run_dir``."""

    best = run_dir / "best.pt"
    best_ema = run_dir / "best_ema.pt"

    if mode == "best":
        if not best.exists():
            raise FileNotFoundError(f"{best} not found")
        return best
    if mode == "best_ema":
        if not best_ema.exists():
            raise FileNotFoundError(f"{best_ema} not found")
        return best_ema

    candidates: list[Path] = [p for p in (best, best_ema) if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"Neither best.pt nor best_ema.pt found under {run_dir}"
        )
    if len(candidates) == 1:
        return candidates[0]

    t_known = float(cfg.training.get("target_known_recall", 0.80))
    t_prec = float(cfg.training.get("target_unknown_precision", 0.35))
    t_rec = float(cfg.training.get("target_unknown_recall", 0.35))
    model = _build_model(cfg, loaders, device)

    best_path = candidates[0]
    best_score = -float("inf")
    for path in candidates:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state"])
        val_preds = collect_predictions(model, loaders["val_loader"], device)
        sel = choose_operating_point(
            val_preds,
            target_unknown_precision=t_prec,
            target_known_recall=t_known,
            target_unknown_recall=t_rec,
            known_classes=loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
        )
        thr = float(sel["newness_threshold"])
        rep = openworld_report(
            val_preds,
            loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
            newness_threshold=thr,
        )
        s = _score_report(rep, t_known, t_prec, t_rec)
        logger.info(
            "  %-12s epoch %d  val_score=%.3f  known=%.3f  unk_prec=%.3f  unk_rec=%.3f  auroc=%.3f",
            path.name,
            int(state.get("epoch", -1)),
            s,
            rep.get("known_recall", float("nan")),
            rep.get("unknown_precision", float("nan")),
            rep.get("unknown_recall", float("nan")),
            rep.get("auroc_newness", float("nan")),
        )
        if s > best_score:
            best_score = s
            best_path = path
    return best_path


def _average_preds(pred_list: list[dict]) -> dict:
    """Average scalar scores across models; keep class_logits as mean logits.

    Assumes all members were produced on the *same* loader (and therefore the
    same y_orig / y_known ordering), which is true as long as the DataLoader
    is built once per ensemble run with a fixed seed.
    """
    out: dict = {}
    per_sample_keys = ["newness", "energy", "dist", "obj_prob"]
    for k in per_sample_keys:
        out[k] = np.mean(np.stack([p[k] for p in pred_list], axis=0), axis=0)
    out["class_logits"] = np.mean(
        np.stack([p["class_logits"] for p in pred_list], axis=0), axis=0
    )
    # Labels are identical across members by construction; sanity-check one.
    base = pred_list[0]
    for p in pred_list[1:]:
        if not np.array_equal(p["y_orig"], base["y_orig"]):
            raise RuntimeError(
                "Ensemble members produced mismatched y_orig arrays; "
                "make sure --config yields the same DataLoader ordering."
            )
    out["y_orig"] = base["y_orig"]
    out["y_known"] = base["y_known"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories to ensemble (each must contain best.pt or best_ema.pt).",
    )
    parser.add_argument(
        "--use",
        choices=["best", "best_ema", "ema_or_best"],
        default="ema_or_best",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Defaults to runs/ensemble/test_report.json.",
    )
    args = parser.parse_args()

    logger = get_logger("oncp.ensemble")
    cfg = load_config(args.config)
    # Loaders are built from the config, not from the run-dirs, so every member
    # sees the same Train/Val/Test split and DataLoader ordering. This is
    # required for per-sample averaging to be well-defined.
    loaders = build_dataloaders(cfg)
    logger.info(
        "Train/Val/Test sizes: %d / %d / %d",
        len(loaders["train_loader"].dataset),
        len(loaders["val_loader"].dataset),
        len(loaders["test_loader"].dataset),
    )

    device = _resolve_device(cfg.get("device", "auto"))
    logger.info("Device: %s", device)

    model = _build_model(cfg, loaders, device)

    val_members: list[dict] = []
    test_members: list[dict] = []
    used_checkpoints: list[str] = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        logger.info("--- Ensemble member: %s", run_dir)
        ckpt_path = _pick_checkpoint(run_dir, args.use, cfg, loaders, device, logger)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        logger.info(
            "Loaded %s (epoch %d, is_ema=%s)",
            ckpt_path,
            int(state.get("epoch", -1)),
            bool(state.get("is_ema", False)),
        )
        used_checkpoints.append(str(ckpt_path))
        val_members.append(collect_predictions(model, loaders["val_loader"], device))
        test_members.append(collect_predictions(model, loaders["test_loader"], device))

    if not val_members:
        raise SystemExit("No ensemble members loaded.")

    val_avg = _average_preds(val_members)
    test_avg = _average_preds(test_members)

    t_known = float(cfg.training.get("target_known_recall", 0.80))
    t_prec = float(cfg.training.get("target_unknown_precision", 0.35))
    t_rec = float(cfg.training.get("target_unknown_recall", 0.35))

    sel = choose_operating_point(
        val_avg,
        target_unknown_precision=t_prec,
        target_known_recall=t_known,
        target_unknown_recall=t_rec,
        known_classes=loaders["known_classes"],
        obj_threshold=float(cfg.model.objectness_threshold),
    )
    val_thr = float(sel["newness_threshold"])
    logger.info(
        "Ensemble val-tuned newness threshold=%.4f (mode=%s)",
        val_thr,
        str(sel["selection_mode"]),
    )

    # Per-member, single-model test reports (for comparison).
    per_member_reports: list[dict] = []
    for idx, (preds_val, preds_test, ckpt) in enumerate(
        zip(val_members, test_members, used_checkpoints), start=1
    ):
        sel_i = choose_operating_point(
            preds_val,
            target_unknown_precision=t_prec,
            target_known_recall=t_known,
            target_unknown_recall=t_rec,
            known_classes=loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
        )
        rep_i = openworld_report(
            preds_test,
            loaders["known_classes"],
            obj_threshold=float(cfg.model.objectness_threshold),
            newness_threshold=float(sel_i["newness_threshold"]),
        )
        rep_i["member_index"] = idx
        rep_i["checkpoint"] = ckpt
        rep_i["tuned_newness_threshold"] = float(sel_i["newness_threshold"])
        rep_i["threshold_selection_mode"] = str(sel_i["selection_mode"])
        per_member_reports.append(rep_i)
        logger.info(
            "Member %d (%s): known=%.3f unk_prec=%.3f unk_rec=%.3f auroc=%.3f",
            idx,
            ckpt,
            rep_i.get("known_recall", float("nan")),
            rep_i.get("unknown_precision", float("nan")),
            rep_i.get("unknown_recall", float("nan")),
            rep_i.get("auroc_newness", float("nan")),
        )

    ensemble_report = openworld_report(
        test_avg,
        loaders["known_classes"],
        obj_threshold=float(cfg.model.objectness_threshold),
        newness_threshold=val_thr,
    )
    ensemble_report["tag"] = "test"
    ensemble_report["tuned_newness_threshold"] = val_thr
    ensemble_report["threshold_selection_mode"] = (
        f"val_tuned:{str(sel['selection_mode'])}"
    )
    ensemble_report["ensemble_members"] = used_checkpoints
    ensemble_report["per_member_test_reports"] = per_member_reports

    logger.info(
        "Ensemble final test report:\n%s",
        json.dumps(ensemble_report, indent=2, default=float),
    )

    out_path = Path(args.out) if args.out else Path("runs/ensemble/test_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_report, f, indent=2, default=float)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
