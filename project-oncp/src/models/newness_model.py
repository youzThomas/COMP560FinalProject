"""Full Newness Transformer: backbone + PAM + dual-head decoder.

Implements the "Discovery-Localization Pipeline" described in section 4.2(C):

    patch-embed -> encoder self-attn -> decoder cross-attn -> PAM -> newness gate

Each of the ``num_queries`` learnable object queries emits:

* ``objectness_logit``  — whether the query corresponds to a foreground object
  (equation 4 / 5, ``o(q_j)``);
* ``class_logits``      — distribution over the known-class set;
* ``energy``            — temperature-scaled free energy ``E(q_j)`` (eq. 1);
* ``dist``              — L2 distance to the nearest prototype (eq. 2);
* ``newness``           — fused score ``S_new = alpha * E + (1-alpha) * D``
  (eq. 3), rendered as a z-score using running statistics to keep
  ``fusion_alpha`` interpretable even though ``E`` and ``D`` live on very
  different scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pam import PrototypeAttentionMemory
from .transformer import PatchEmbed1D, TransformerEncoderDecoder


class _RunningStats(nn.Module):
    """Tiny EMA tracker so that energy and distance can be fused on a common scale."""

    def __init__(self, momentum: float = 0.99) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False).clamp_min(1e-8)
        if not bool(self.initialized):
            self.mean.copy_(m.detach().view(1))
            self.var.copy_(v.detach().view(1))
            self.initialized.fill_(True)
            return
        self.mean.mul_(self.momentum).add_(m.detach().view(1), alpha=1.0 - self.momentum)
        self.var.mul_(self.momentum).add_(v.detach().view(1), alpha=1.0 - self.momentum)

    def zscore(self, x: torch.Tensor) -> torch.Tensor:
        std = self.var.clamp_min(1e-8).sqrt()
        return (x - self.mean) / std


def energy_score(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Free energy from logits: E(q) = -T * logsumexp(l / T).

    A higher energy corresponds to lower likelihood under the known-class
    distribution and should therefore correlate with novelty.
    """
    T = float(temperature)
    return -T * torch.logsumexp(logits / T, dim=-1)


class NewnessTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: int,
        num_known_classes: int,
        patch_size: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        num_queries: int = 4,
        dropout: float = 0.1,
        pam_prototypes_per_class: int = 2,
        pam_proto_dim: int | None = None,
        energy_temperature: float = 1.0,
        fusion_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        if window_size % patch_size != 0:
            raise ValueError(
                f"window_size ({window_size}) must be divisible by patch_size ({patch_size})"
            )
        proto_dim = pam_proto_dim or d_model

        self.patch_embed = PatchEmbed1D(in_channels, patch_size, d_model)
        self.backbone = TransformerEncoderDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
        )

        # Project decoder features to the prototype space if dimensions differ.
        self.proto_proj = nn.Linear(d_model, proto_dim) if proto_dim != d_model else nn.Identity()

        self.pam = PrototypeAttentionMemory(
            num_known_classes=num_known_classes,
            proto_dim=proto_dim,
            prototypes_per_class=pam_prototypes_per_class,
        )

        # Dual-head decoder: (known classifier) + (newness head).
        # The newness head consumes the PAM-refined feature so it can reason
        # about both the raw query and its closest prototype context.
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_known_classes),
        )
        self.obj_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.num_known_classes = num_known_classes
        self.num_queries = num_queries
        self.energy_temperature = energy_temperature
        self.fusion_alpha = fusion_alpha

        self.energy_stats = _RunningStats()
        self.dist_stats = _RunningStats()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dict of per-query predictions.

        x: [B, T, C]
        """
        tokens = self.patch_embed(x)                      # [B, N, D]
        queries, memory = self.backbone(tokens)           # [B, Q, D]

        class_logits = self.class_head(queries)           # [B, Q, K]
        obj_logits = self.obj_head(queries).squeeze(-1)   # [B, Q]
        energy = energy_score(class_logits, self.energy_temperature)  # [B, Q]

        proto_feats = self.proto_proj(queries)            # [B, Q, P]
        pam_out = self.pam(proto_feats)
        dist = pam_out["dist"]                            # [B, Q]

        if self.training:
            self.energy_stats.update(energy.detach().flatten())
            self.dist_stats.update(dist.detach().flatten())
        energy_z = self.energy_stats.zscore(energy)
        dist_z = self.dist_stats.zscore(dist)
        newness = self.fusion_alpha * energy_z + (1.0 - self.fusion_alpha) * dist_z

        return {
            "class_logits": class_logits,
            "obj_logits": obj_logits,
            "obj_prob": torch.sigmoid(obj_logits),
            "energy": energy,
            "energy_z": energy_z,
            "dist": dist,
            "dist_z": dist_z,
            "newness": newness,
            "query_feats": queries,
            "proto_feats": proto_feats,
            "pam_class_dist": pam_out["class_dist"],
            "pam_attn": pam_out["attn"],
            "memory": memory,
        }

    def predict(
        self,
        outputs: dict[str, torch.Tensor],
        objectness_threshold: float,
        newness_threshold: float,
    ) -> dict[str, torch.Tensor]:
        """Aggregate per-query predictions into a single per-sample decision.

        We pick the query with the highest objectness. If that query's newness
        score crosses ``newness_threshold`` we emit the ``unknown`` label
        (integer ``-1``); otherwise we emit the argmax of its class logits.
        Queries whose objectness falls below ``objectness_threshold`` collapse
        to ``background`` (also encoded as ``-1`` here but separable via
        ``obj_prob``).
        """
        obj_prob = outputs["obj_prob"]                     # [B, Q]
        newness = outputs["newness"]
        class_logits = outputs["class_logits"]

        best_q = obj_prob.argmax(dim=-1)                   # [B]
        idx = best_q.unsqueeze(-1)

        chosen_obj = obj_prob.gather(1, idx).squeeze(-1)
        chosen_new = newness.gather(1, idx).squeeze(-1)
        chosen_cls = class_logits.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, class_logits.size(-1))
        ).squeeze(1)

        is_fg = chosen_obj > objectness_threshold
        is_unknown = is_fg & (chosen_new > newness_threshold)
        pred = chosen_cls.argmax(dim=-1)
        pred = torch.where(is_unknown | ~is_fg, torch.full_like(pred, -1), pred)
        return {
            "pred": pred,
            "best_query": best_q,
            "obj_prob": chosen_obj,
            "newness": chosen_new,
            "class_logits": chosen_cls,
            "is_unknown": is_unknown,
            "is_foreground": is_fg,
        }
