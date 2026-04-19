"""Training loop for the Newness Transformer."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..evaluation.metrics import (
    choose_newness_threshold,
    collect_predictions,
    openworld_report,
)
from ..losses import NewnessLoss
from ..losses.losses import NewnessLossWeights
from ..models import NewnessTransformer
from ..utils.logging import get_logger


def _cosine_with_warmup(total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return lr_lambda


class Trainer:
    def __init__(
        self,
        cfg,
        model: NewnessTransformer,
        loaders: dict[str, Any],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.train_loader = loaders["train_loader"]
        self.val_loader = loaders["val_loader"]
        self.test_loader = loaders["test_loader"]
        self.known_classes = loaders["known_classes"]

        tcfg = cfg.training
        self.epochs = int(tcfg.epochs)
        self.grad_clip = float(tcfg.grad_clip)
        self.proto_ema = float(tcfg.proto_ema)
        self.log_every = int(tcfg.get("log_every", 20))

        self.loss_fn = NewnessLoss(
            weights=NewnessLossWeights(
                w_class=float(tcfg.w_class),
                w_objectness=float(tcfg.w_objectness),
                w_energy=float(tcfg.w_energy),
                w_proto=float(tcfg.w_proto),
                m_in=float(tcfg.energy_m_in),
                m_out=float(tcfg.energy_m_out),
            )
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(tcfg.lr),
            weight_decay=float(tcfg.weight_decay),
        )
        steps_per_epoch = max(1, len(self.train_loader))
        warmup_steps = int(tcfg.get("warmup_epochs", 2)) * steps_per_epoch
        total_steps = self.epochs * steps_per_epoch
        self.scheduler = LambdaLR(
            self.optimizer, _cosine_with_warmup(total_steps, warmup_steps)
        )

        self.ckpt_dir = Path(tcfg.get("ckpt_dir", "runs/default"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("oncp.trainer")
        self.best_val = -1.0

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        running = {"loss": 0.0, "cls": 0.0, "obj": 0.0, "energy_in": 0.0, "energy_out": 0.0, "proto": 0.0}
        count = 0
        pbar = tqdm(self.train_loader, desc=f"epoch {epoch:03d}", leave=False)
        for step, batch in enumerate(pbar):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            # Training batches should only contain known-class samples.
            mask = y >= 0
            if not mask.all():
                x, y = x[mask], y[mask]
            if x.numel() == 0:
                continue

            out = self.model(x)
            protos = self.model.pam.prototypes()
            class_ids = self.model.pam.class_ids
            loss, comps, matched = self.loss_fn(out, y, protos, class_ids)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            # Prototype EMA update using the matched queries.
            with torch.no_grad():
                B = x.size(0)
                batch_idx = torch.arange(B, device=x.device)
                matched_feats = out["proto_feats"][batch_idx, matched].detach()
                self.model.pam.ema_update(matched_feats, y.detach(), self.proto_ema)

            for k, v in comps.items():
                running[k] = running.get(k, 0.0) + float(v)
            count += 1

            if (step + 1) % self.log_every == 0:
                pbar.set_postfix(
                    loss=f"{running['loss'] / count:.3f}",
                    cls=f"{running['cls'] / count:.3f}",
                    obj=f"{running['obj'] / count:.3f}",
                )
        return {k: v / max(1, count) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, loader, tag: str = "val") -> dict[str, Any]:
        preds = collect_predictions(self.model, loader, self.device)
        newness_thr = choose_newness_threshold(
            preds, target_unknown_precision=0.35,
            known_classes=self.known_classes,
            obj_threshold=float(self.cfg.model.objectness_threshold),
        )
        rep = openworld_report(
            preds, self.known_classes,
            obj_threshold=float(self.cfg.model.objectness_threshold),
            newness_threshold=newness_thr,
        )
        rep["tag"] = tag
        rep["tuned_newness_threshold"] = newness_thr
        return rep

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        for epoch in range(1, self.epochs + 1):
            train_stats = self._train_one_epoch(epoch)
            val_report = self.evaluate(self.val_loader, tag="val")
            score = val_report["known_recall"] if not math.isnan(val_report["known_recall"]) else 0.0
            if not math.isnan(val_report.get("auroc_newness", float("nan"))):
                score = 0.5 * score + 0.5 * val_report["auroc_newness"]
            self.logger.info(
                "epoch %03d | loss=%.3f cls=%.3f obj=%.3f proto=%.3f | "
                "val known_rec=%.3f unk_prec=%.3f auroc=%.3f",
                epoch,
                train_stats.get("loss", 0.0),
                train_stats.get("cls", 0.0),
                train_stats.get("obj", 0.0),
                train_stats.get("proto", 0.0),
                val_report["known_recall"],
                val_report["unknown_precision"],
                val_report.get("auroc_newness", float("nan")),
            )
            history.append({"epoch": epoch, "train": train_stats, "val": val_report})

            if score > self.best_val:
                self.best_val = score
                self.save_checkpoint(self.ckpt_dir / "best.pt", epoch, val_report)

        self.save_checkpoint(self.ckpt_dir / "last.pt", self.epochs, history[-1]["val"])
        with open(self.ckpt_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=float)
        return {"history": history, "best_val": self.best_val}

    def save_checkpoint(self, path: Path, epoch: int, extra: dict) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "config": dict(self.cfg),
                "val_report": extra,
            },
            path,
        )
