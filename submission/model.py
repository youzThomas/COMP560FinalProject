"""Grader-facing model entrypoint.

Provides ``StudentModel`` with a simple load + predict API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models import NewnessTransformer
from src.utils.config import load_config


def _resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


class StudentModel:
    """Inference wrapper around the trained Newness Transformer.

    Expected input shape for ``predict`` / ``forward``:
    - single sample: ``[T, C]``
    - batch: ``[B, T, C]``
    where ``T=64`` and ``C=6`` for the provided config/checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "runs/default/best_ema.pt",
        config_path: str | Path = "configs/default.yaml",
        device: str = "auto",
    ) -> None:
        root = Path(__file__).resolve().parent
        self.config_path = (root / config_path).resolve()
        self.checkpoint_path = (root / checkpoint_path).resolve()
        self.cfg = load_config(self.config_path)
        self.device = _resolve_device(device if device != "auto" else str(self.cfg.get("device", "auto")))

        self.model = NewnessTransformer(
            in_channels=int(self.cfg.model.in_channels),
            window_size=int(self.cfg.data.window_size),
            num_known_classes=len(self.cfg.data.known_classes),
            patch_size=int(self.cfg.model.patch_size),
            d_model=int(self.cfg.model.d_model),
            n_heads=int(self.cfg.model.n_heads),
            n_encoder_layers=int(self.cfg.model.n_encoder_layers),
            n_decoder_layers=int(self.cfg.model.n_decoder_layers),
            num_queries=int(self.cfg.model.num_queries),
            dropout=float(self.cfg.model.dropout),
            pam_prototypes_per_class=int(self.cfg.model.pam_prototypes_per_class),
            pam_proto_dim=int(self.cfg.model.pam_proto_dim),
            energy_temperature=float(self.cfg.model.energy_temperature),
            fusion_alpha=float(self.cfg.model.fusion_alpha),
            fusion_msp=float(self.cfg.model.get("fusion_msp", 0.0)),
        ).to(self.device)

        state = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        model_state = state.get("model_state", state)
        self.model.load_state_dict(model_state)
        self.model.eval()

        self.objectness_threshold = float(self.cfg.model.objectness_threshold)
        self.newness_threshold = float(self.cfg.model.newness_threshold)

    def _to_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [T,C] or [B,T,C], got {tuple(x.shape)}")
        return x.to(self.device)

    @torch.inference_mode()
    def forward(self, x: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
        x_t = self._to_tensor(x)
        return self.model(x_t)

    @torch.inference_mode()
    def predict(
        self,
        x: np.ndarray | torch.Tensor,
        objectness_threshold: float | None = None,
        newness_threshold: float | None = None,
    ) -> dict[str, Any]:
        x_t = self._to_tensor(x)
        outputs = self.model(x_t)
        preds = self.model.predict(
            outputs,
            objectness_threshold=self.objectness_threshold
            if objectness_threshold is None
            else float(objectness_threshold),
            newness_threshold=self.newness_threshold
            if newness_threshold is None
            else float(newness_threshold),
        )
        # Return CPU tensors so callers can serialize/inspect safely.
        out = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in preds.items()}
        out["batch_size"] = int(x_t.shape[0])
        return out
