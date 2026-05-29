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


def msp_newness(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """``1 - max softmax prob`` under temperature scaling (high = more novel).

    MSP captures classifier *margin* rather than classifier *magnitude*, which
    makes it complementary to the free-energy score when the known-class
    feature distributions are tight (small margins indicate potential OOD
    samples even when the absolute logits are large).
    """
    T = float(temperature)
    probs = torch.softmax(logits / T, dim=-1)
    return 1.0 - probs.max(dim=-1).values


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
        fusion_msp: float = 0.0,
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
        # Fusion weights for newness = a*energy_z + b*msp_z + (1-a-b)*dist_z.
        # Clamp to keep dist_weight >= 0 so we never invert a component.
        a = float(fusion_alpha)
        b = float(fusion_msp)
        if a + b > 1.0:
            scale = 1.0 / (a + b)
            a *= scale
            b *= scale
        self.fusion_alpha = a
        self.fusion_msp = b
        self.fusion_dist = max(0.0, 1.0 - a - b)

        self.energy_stats = _RunningStats()
        self.dist_stats = _RunningStats()
        self.msp_stats = _RunningStats()

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
        msp = msp_newness(class_logits, self.energy_temperature)       # [B, Q]

        proto_feats = self.proto_proj(queries)            # [B, Q, P]
        pam_out = self.pam(proto_feats)
        dist = pam_out["dist"]                            # [B, Q]

        if self.training:
            self.energy_stats.update(energy.detach().flatten())
            self.dist_stats.update(dist.detach().flatten())
            self.msp_stats.update(msp.detach().flatten())
        energy_z = self.energy_stats.zscore(energy)
        dist_z = self.dist_stats.zscore(dist)
        msp_z = self.msp_stats.zscore(msp)
        newness = (
            self.fusion_alpha * energy_z
            + self.fusion_msp * msp_z
            + self.fusion_dist * dist_z
        )

        return {
            "class_logits": class_logits,
            "obj_logits": obj_logits,
            "obj_prob": torch.sigmoid(obj_logits),
            "energy": energy,
            "energy_z": energy_z,
            "dist": dist,
            "dist_z": dist_z,
            "msp": msp,
            "msp_z": msp_z,
            "newness": newness,
            "query_feats": queries,
            "proto_feats": proto_feats,
            "pam_class_dist": pam_out["class_dist"],
            "pam_attn": pam_out["attn"],
            "memory": memory,
        }

    def compute_vos_loss(
        self,
        query_feats: torch.Tensor,
        matched_idx: torch.Tensor,
        targets: torch.Tensor,
        m_out: float,
        alpha: float = 1.5,
    ) -> torch.Tensor:
        """Tail-extrapolation Virtual Outlier Synthesis.

        An earlier cross-class MixUp version of this loss inverted the
        energy signal on the Mill dataset because Failed samples are *not*
        halfway between Healthy and Degraded -- they are *beyond* Degraded
        on the wear continuum. Training "midpoint => OOD" made the classifier
        *more* confident on pure-class features, including Failed samples.

        This version instead builds per-class centroids from matched query
        features in the current batch (detached from graph so the target
        does not move) and extrapolates each sample away from its own class
        center::

            q_tail = q + alpha * (q - class_center(q))

        With ``alpha=1.5`` this lands the synthetic feature 2.5x further
        from the class centroid than the real sample, out in the tail of
        that class' distribution. Pushing ``E(q_tail) > m_out`` teaches the
        model that the tails of the known classes should read as OOD --
        which is exactly where Failed samples live relative to Degraded.
        """
        B = query_feats.size(0)
        device = query_feats.device
        if B < 2:
            return query_feats.new_zeros(())
        batch_idx = torch.arange(B, device=device)
        matched_q = query_feats[batch_idx, matched_idx]       # [B, D]

        # Per-class centroid from this batch (detached to decouple targets).
        centroids: list[torch.Tensor] = []
        for c in range(self.num_known_classes):
            mask = targets == c
            if mask.any():
                centroids.append(matched_q[mask].mean(dim=0, keepdim=True).detach())
            else:
                centroids.append(matched_q.mean(dim=0, keepdim=True).detach())
        centroid_table = torch.cat(centroids, dim=0)          # [K, D]
        sample_center = centroid_table[targets]                # [B, D]
        tail = matched_q + float(alpha) * (matched_q - sample_center)

        logits = self.class_head(tail)                         # [B, K]
        T = float(self.energy_temperature)
        energy = -T * torch.logsumexp(logits / T, dim=-1)      # [B]
        # Hinge-squared margin: tail-sample energy should sit above m_out.
        return F.relu(float(m_out) - energy).pow(2).mean()

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
