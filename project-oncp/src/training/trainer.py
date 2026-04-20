"""Training loop for the Newness Transformer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..evaluation.metrics import (
    choose_operating_point,
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
        self.target_unknown_precision = float(tcfg.get("target_unknown_precision", 0.35))
        self.target_known_recall = float(tcfg.get("target_known_recall", 0.80))
        self.target_unknown_recall = float(tcfg.get("target_unknown_recall", 0.35))
        # VOS (tail-extrapolation outlier synthesis) weight; 0 disables. The
        # outlier margin and extrapolation strength each have their own knob
        # so they can be tuned independently of the main energy margin.
        self.w_vos = float(tcfg.get("w_vos", 0.0))
        self.vos_m_out = float(tcfg.get("vos_m_out", tcfg.energy_m_out))
        self.vos_alpha = float(tcfg.get("vos_alpha", 1.5))
        # Number of warmup epochs before VOS loss is activated. Gives the Hungarian
        # matcher and the classifier enough time to build coherent class centroids;
        # VOS applied before that destabilized training in run 44444382.
        self.vos_warmup_epochs = int(tcfg.get("vos_warmup_epochs", 0))
        # Training-only input noise std (fraction of each sample's own std).
        self.input_noise_std = float(tcfg.get("input_noise_std", 0.0))
        # Early stopping: stop if val score has not improved for this many epochs.
        # 0 (or negative) disables. The best-scoring ckpt is saved to disk regardless.
        self.early_stop_patience = int(tcfg.get("early_stop_patience", 0))
        # Parallel EMA snapshot of the model. Evaluated on val at every epoch and
        # saved as best_ema.pt when its score beats the live model's best. Disabled
        # when decay <= 0. Using 0.999 gives an effective averaging window of ~1000
        # optimizer steps which roughly matches one training epoch here.
        self.model_ema_decay = float(tcfg.get("model_ema_decay", 0.0))
        if self.model_ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model).to(device)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None
        self.best_val_ema = -1.0

        self.loss_fn = NewnessLoss(
            weights=NewnessLossWeights(
                w_class=float(tcfg.w_class),
                w_objectness=float(tcfg.w_objectness),
                w_energy=float(tcfg.w_energy),
                w_proto=float(tcfg.w_proto),
                m_in=float(tcfg.energy_m_in),
                m_out=float(tcfg.energy_m_out),
                # Optional knobs exposed via config; fall back to the dataclass
                # defaults when the corresponding key is absent.
                cls_label_smoothing=float(tcfg.get("cls_label_smoothing", 0.0)),
                energy_rank_margin=float(tcfg.get("energy_rank_margin", 0.0)),
                energy_log_compress=bool(tcfg.get("energy_log_compress", False)),
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
        running = {"loss": 0.0, "cls": 0.0, "obj": 0.0, "energy_in": 0.0, "energy_out": 0.0, "proto": 0.0, "vos": 0.0}
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

            # Per-sample Gaussian noise augmentation. Scale the noise by each
            # sample's own stddev (over time and channels) so the perturbation
            # magnitude is consistent across the heterogeneous Mill channels.
            if self.input_noise_std > 0.0:
                sample_std = x.detach().float().std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                x = x + torch.randn_like(x) * (self.input_noise_std * sample_std)

            out = self.model(x)
            protos = self.model.pam.prototypes()
            class_ids = self.model.pam.class_ids
            loss, comps, matched = self.loss_fn(out, y, protos, class_ids)

            if self.w_vos > 0.0 and epoch > self.vos_warmup_epochs:
                vos_loss = self.model.compute_vos_loss(
                    out["query_feats"],
                    matched,
                    y,
                    m_out=self.vos_m_out,
                    alpha=self.vos_alpha,
                )
                loss = loss + self.w_vos * vos_loss
                comps["vos"] = vos_loss.detach()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            # Model-weight EMA: update in lockstep with the optimizer so the
            # snapshot reflects the smoothed trajectory, not the noisy one.
            if self.ema_model is not None:
                d = self.model_ema_decay
                with torch.no_grad():
                    for p_ema, p in zip(
                        self.ema_model.parameters(), self.model.parameters()
                    ):
                        p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)
                    for b_ema, b in zip(
                        self.ema_model.buffers(), self.model.buffers()
                    ):
                        if b_ema.dtype.is_floating_point:
                            b_ema.mul_(d).add_(b.detach(), alpha=1.0 - d)
                        else:
                            b_ema.copy_(b)

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
    def evaluate(
        self,
        loader,
        tag: str = "val",
        newness_threshold: float | None = None,
        model: NewnessTransformer | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``model`` (defaults to ``self.model``) on ``loader``.

        When ``newness_threshold`` is None the threshold is tuned on this split
        (appropriate for the val split during training). For the test split
        the caller should pass a threshold tuned on val to avoid test-set
        tuning leakage.
        """
        eval_model = model if model is not None else self.model
        preds = collect_predictions(eval_model, loader, self.device)
        if newness_threshold is None:
            sel = choose_operating_point(
                preds,
                target_unknown_precision=self.target_unknown_precision,
                target_known_recall=self.target_known_recall,
                target_unknown_recall=self.target_unknown_recall,
                known_classes=self.known_classes,
                obj_threshold=float(self.cfg.model.objectness_threshold),
            )
            newness_thr = float(sel["newness_threshold"])
            sel_mode = str(sel["selection_mode"])
        else:
            newness_thr = float(newness_threshold)
            sel_mode = "provided"
        rep = openworld_report(
            preds, self.known_classes,
            obj_threshold=float(self.cfg.model.objectness_threshold),
            newness_threshold=newness_thr,
        )
        rep["tag"] = tag
        rep["tuned_newness_threshold"] = newness_thr
        rep["threshold_selection_mode"] = sel_mode
        return rep

    @torch.no_grad()
    def evaluate_best(self, loader, tag: str = "test") -> dict[str, Any]:
        """Pick the better of ``best.pt`` / ``best_ema.pt`` on val, then eval.

        We compare the two checkpoints on a fresh val pass (using the same
        threshold-selection rules the scorer uses), load the winner into
        ``self.model``, tune the threshold on val, and freeze it for ``loader``.
        """
        candidates: list[tuple[str, Path]] = []
        for name, fname in (("best", "best.pt"), ("best_ema", "best_ema.pt")):
            p = self.ckpt_dir / fname
            if p.exists():
                candidates.append((name, p))

        if not candidates:
            self.logger.warning(
                "No best*.pt found under %s; evaluating the current (last) model.",
                self.ckpt_dir,
            )
        else:
            best_name: str | None = None
            best_score = -float("inf")
            best_epoch = -1
            for name, path in candidates:
                state = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state["model_state"])
                rep = self.evaluate(self.val_loader, tag="val_select")
                s = self._score_report(
                    rep,
                    self.target_known_recall,
                    self.target_unknown_precision,
                    self.target_unknown_recall,
                )
                self.logger.info(
                    "Candidate %s (epoch %d): val score=%.3f known=%.3f "
                    "unk_prec=%.3f unk_rec=%.3f auroc=%.3f",
                    name,
                    int(state.get("epoch", -1)),
                    s,
                    rep.get("known_recall", float("nan")),
                    rep.get("unknown_precision", float("nan")),
                    rep.get("unknown_recall", float("nan")),
                    rep.get("auroc_newness", float("nan")),
                )
                if s > best_score:
                    best_score = s
                    best_name = name
                    best_epoch = int(state.get("epoch", -1))
            assert best_name is not None  # non-empty candidates
            final_path = self.ckpt_dir / (
                "best.pt" if best_name == "best" else "best_ema.pt"
            )
            self.model.load_state_dict(
                torch.load(final_path, map_location=self.device)["model_state"]
            )
            self.logger.info(
                "Selected %s (epoch %d, val score=%.3f) for final evaluation.",
                best_name,
                best_epoch,
                best_score,
            )
        # Tune threshold on val with the winning model, then freeze it for test.
        val_preds = collect_predictions(self.model, self.val_loader, self.device)
        sel = choose_operating_point(
            val_preds,
            target_unknown_precision=self.target_unknown_precision,
            target_known_recall=self.target_known_recall,
            target_unknown_recall=self.target_unknown_recall,
            known_classes=self.known_classes,
            obj_threshold=float(self.cfg.model.objectness_threshold),
        )
        val_thr = float(sel["newness_threshold"])
        self.logger.info(
            "Val-tuned newness threshold=%.4f (mode=%s)",
            val_thr,
            str(sel["selection_mode"]),
        )
        rep = self.evaluate(loader, tag=tag, newness_threshold=val_thr)
        rep["threshold_selection_mode"] = (
            f"val_tuned:{str(sel['selection_mode'])}"
        )
        return rep

    @staticmethod
    def _score_report(val_report: dict, t_known: float, t_prec: float, t_rec: float) -> float:
        """Checkpoint score used for both live model and EMA snapshot.

        Same formula as the previous inline scorer, extracted so the two
        branches (live / ema) cannot drift apart.
        """
        known = 0.0 if math.isnan(val_report["known_recall"]) else float(val_report["known_recall"])
        unk_prec = 0.0 if math.isnan(val_report["unknown_precision"]) else float(val_report["unknown_precision"])
        unk_rec = 0.0 if math.isnan(val_report["unknown_recall"]) else float(val_report["unknown_recall"])
        auroc_raw = val_report.get("auroc_newness", float("nan"))
        auroc = 0.5 if (auroc_raw is None or math.isnan(float(auroc_raw))) else float(auroc_raw)
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

    def fit(self) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        epochs_since_improve = 0
        for epoch in range(1, self.epochs + 1):
            train_stats = self._train_one_epoch(epoch)
            val_report = self.evaluate(self.val_loader, tag="val")
            known = 0.0 if math.isnan(val_report["known_recall"]) else float(val_report["known_recall"])
            unk_prec = 0.0 if math.isnan(val_report["unknown_precision"]) else float(val_report["unknown_precision"])
            unk_rec = 0.0 if math.isnan(val_report["unknown_recall"]) else float(val_report["unknown_recall"])
            auroc_raw = val_report.get("auroc_newness", float("nan"))
            auroc = 0.5 if (auroc_raw is None or math.isnan(float(auroc_raw))) else float(auroc_raw)
            score = self._score_report(
                val_report,
                self.target_known_recall,
                self.target_unknown_precision,
                self.target_unknown_recall,
            )

            # Evaluate the EMA snapshot on the same val split so we can save
            # whichever checkpoint (live or EMA) scores higher. The EMA eval
            # typically lags the live one by a few epochs but catches up and
            # overtakes near convergence.
            ema_score = None
            ema_auroc = float("nan")
            if self.ema_model is not None:
                ema_report = self.evaluate(
                    self.val_loader, tag="val_ema", model=self.ema_model
                )
                ema_score = self._score_report(
                    ema_report,
                    self.target_known_recall,
                    self.target_unknown_precision,
                    self.target_unknown_recall,
                )
                ema_auroc = float(ema_report.get("auroc_newness", float("nan")))

            self.logger.info(
                "epoch %03d | loss=%.3f cls=%.3f obj=%.3f proto=%.3f vos=%.3f | "
                "val known_rec=%.3f unk_prec=%.3f unk_rec=%.3f auroc=%.3f sel=%s score=%.3f"
                "%s",
                epoch,
                train_stats.get("loss", 0.0),
                train_stats.get("cls", 0.0),
                train_stats.get("obj", 0.0),
                train_stats.get("proto", 0.0),
                train_stats.get("vos", 0.0),
                val_report["known_recall"],
                val_report["unknown_precision"],
                val_report["unknown_recall"],
                val_report.get("auroc_newness", float("nan")),
                val_report.get("threshold_selection_mode", "n/a"),
                score,
                ""
                if ema_score is None
                else f" | ema auroc={ema_auroc:.3f} score={ema_score:.3f}",
            )
            history.append({"epoch": epoch, "train": train_stats, "val": val_report})

            improved = False
            if score > self.best_val:
                self.best_val = score
                self.save_checkpoint(self.ckpt_dir / "best.pt", epoch, val_report)
                improved = True
            if (
                ema_score is not None
                and ema_score > self.best_val_ema
            ):
                self.best_val_ema = ema_score
                self.save_checkpoint(
                    self.ckpt_dir / "best_ema.pt",
                    epoch,
                    ema_report,
                    use_ema=True,
                )
                improved = True

            if improved:
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if (
                self.early_stop_patience > 0
                and epochs_since_improve >= self.early_stop_patience
            ):
                self.logger.info(
                    "Early stopping after epoch %d: no val score improvement for %d epochs.",
                    epoch,
                    self.early_stop_patience,
                )
                break

        self.save_checkpoint(self.ckpt_dir / "last.pt", history[-1]["epoch"], history[-1]["val"])
        with open(self.ckpt_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=float)
        return {"history": history, "best_val": self.best_val}

    def save_checkpoint(
        self, path: Path, epoch: int, extra: dict, use_ema: bool = False
    ) -> None:
        state_source = (
            self.ema_model if (use_ema and self.ema_model is not None) else self.model
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state": state_source.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "config": dict(self.cfg),
                "val_report": extra,
                "is_ema": bool(use_ema),
            },
            path,
        )
