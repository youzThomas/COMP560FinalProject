"""Smoke-test the forward + loss + metric pipeline with random tensors.

Useful as a fast sanity check (no data needed) before launching a full training
run on the Mill dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.metrics import openworld_report, unknown_detection_curve  # noqa: E402
from src.losses import NewnessLoss  # noqa: E402
from src.models import NewnessTransformer  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    B, T, C = 4, 64, 6
    num_known = 2

    model = NewnessTransformer(
        in_channels=C,
        window_size=T,
        num_known_classes=num_known,
        patch_size=8,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        num_queries=3,
        dropout=0.0,
        pam_prototypes_per_class=2,
        pam_proto_dim=64,
    )
    x = torch.randn(B, T, C)
    out = model(x)
    assert out["class_logits"].shape == (B, 3, num_known)
    assert out["obj_logits"].shape == (B, 3)

    y = torch.randint(0, num_known, (B,))
    loss_fn = NewnessLoss()
    protos = model.pam.prototypes()
    loss, comps, matched = loss_fn(out, y, protos, model.pam.class_ids)
    loss.backward()
    print("forward + backward ok. loss =", float(loss), "components =", {k: float(v) for k, v in comps.items()})

    pred = model.predict(out, objectness_threshold=0.3, newness_threshold=0.0)
    print("predict keys:", sorted(pred.keys()))

    fake_preds = {
        "obj_prob": out["obj_prob"].max(dim=-1).values.detach().numpy(),
        "newness": out["newness"].mean(dim=-1).detach().numpy(),
        "energy": out["energy"].mean(dim=-1).detach().numpy(),
        "dist": out["dist"].mean(dim=-1).detach().numpy(),
        "class_logits": out["class_logits"].mean(dim=1).detach().numpy(),
        "y_orig": np.array([0, 1, 2, 2]),
        "y_known": np.array([0, 1, -1, -1]),
    }
    report = openworld_report(
        fake_preds, known_classes=[0, 1],
        obj_threshold=0.3, newness_threshold=0.0,
    )
    print("openworld_report ok. keys:", sorted(report.keys()))
    print("auroc:", unknown_detection_curve(fake_preds))


if __name__ == "__main__":
    main()
