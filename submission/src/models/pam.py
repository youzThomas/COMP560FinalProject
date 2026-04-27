"""Prototype-Attention Memory (PAM).

Section 4.2(A) of the midterm report: we maintain a prototype memory bank for
known classes and integrate it with the decoder via cross-attention so prototype
matching is context-aware rather than relying on static class averages.

Design choices:
* Each known class owns ``prototypes_per_class`` prototype vectors of dimension
  ``proto_dim``. Prototypes are stored both as EMA buffers (updated from
  matched query features during training) and as learnable offsets that let the
  attention step refine them.
* An attention pooling step produces an attention-weighted prototype for every
  query, from which we derive the distance-based newness score ``D(q_j)``
  (equation 2 in the report).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAttentionMemory(nn.Module):
    def __init__(
        self,
        num_known_classes: int,
        proto_dim: int,
        prototypes_per_class: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_known_classes
        self.per_class = prototypes_per_class
        self.total = num_known_classes * prototypes_per_class
        self.proto_dim = proto_dim

        # EMA-tracked "class centres" used for the L2 distance score.
        self.register_buffer(
            "ema_protos", torch.randn(self.total, proto_dim) * 0.02
        )
        # Learnable offsets that shape the attention keys/values.
        self.offset = nn.Parameter(torch.zeros(self.total, proto_dim))

        # Bookkeeping: which class each prototype slot belongs to.
        class_ids = torch.arange(num_known_classes).repeat_interleave(prototypes_per_class)
        self.register_buffer("class_ids", class_ids, persistent=False)

        self.q_proj = nn.Linear(proto_dim, proto_dim, bias=False)
        self.k_proj = nn.Linear(proto_dim, proto_dim, bias=False)
        self.v_proj = nn.Linear(proto_dim, proto_dim, bias=False)
        self.out_proj = nn.Linear(proto_dim, proto_dim, bias=False)
        self.scale = proto_dim ** -0.5

    @torch.no_grad()
    def ema_update(self, feats: torch.Tensor, class_idx: torch.Tensor, momentum: float) -> None:
        """EMA update: each class' prototypes drift toward the mean of its matched
        query features. Only the first prototype per class is moved; the others
        act as residual slots refined by ``self.offset`` to capture sub-modes.
        """
        if feats.numel() == 0:
            return
        for c in class_idx.unique().tolist():
            mask = class_idx == c
            mean = feats[mask].mean(dim=0)
            slot = int(c) * self.per_class
            self.ema_protos[slot] = (
                momentum * self.ema_protos[slot] + (1.0 - momentum) * mean
            )

    def prototypes(self) -> torch.Tensor:
        """Current prototype table with learnable offsets applied."""
        return self.ema_protos + self.offset

    def forward(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute attention-weighted prototype and newness distance.

        q: [B, Q, D] query features.
        Returns dict with::

            pooled:    [B, Q, D] attention-weighted prototype feature
            dist:      [B, Q]   min-L2 distance to any prototype (eq. 2)
            class_dist:[B, Q, K] min-L2 distance per class (used for logits aux)
            attn:      [B, Q, P] prototype attention weights
        """
        protos = self.prototypes()                    # [P, D]
        qh = self.q_proj(q)                          # [B, Q, D]
        kh = self.k_proj(protos).unsqueeze(0)        # [1, P, D]
        vh = self.v_proj(protos).unsqueeze(0)        # [1, P, D]

        attn = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale  # [B, Q, P]
        weights = F.softmax(attn, dim=-1)
        pooled = self.out_proj(torch.matmul(weights, vh))            # [B, Q, D]

        # Euclidean distance table: [B, Q, P]
        # ||q - p||^2 = ||q||^2 + ||p||^2 - 2 q.p
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)                   # [B, Q, 1]
        p_sq = (protos ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, P]
        qp = torch.matmul(q, protos.t().unsqueeze(0))               # [B, Q, P]
        d2 = (q_sq + p_sq - 2.0 * qp).clamp_min(0.0)

        # Per-class min distance
        B, Q, _ = d2.shape
        d2_per_class = d2.view(B, Q, self.num_classes, self.per_class).min(dim=-1).values
        dist = d2_per_class.min(dim=-1).values.sqrt()
        class_dist = d2_per_class.sqrt()
        return {"pooled": pooled, "dist": dist, "class_dist": class_dist, "attn": weights}
