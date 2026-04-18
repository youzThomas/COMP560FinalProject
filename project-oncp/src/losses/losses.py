"""Training losses for the Newness Transformer.

We assemble four terms, mirroring the report:

* **Classification** (``w_class``): cross-entropy on the matched query.
* **Objectness** (``w_objectness``): binary cross-entropy distinguishing the
  matched "foreground" query from the remaining "background" queries.
* **Energy margin** (``w_energy``): per the energy-based OOD literature (Liu et
  al. 2020, cited in the report) we push the matched query's free energy below
  ``m_in`` and the unmatched queries' energy above ``m_out``. This gives an
  explicit signal that separates knowns from "would-be unknowns" even when we
  cannot yet see unknowns during training.
* **Prototype contrastive** (``w_proto``): pulls the matched query close to its
  class prototype and pushes it away from the other classes, tightening the
  geometry that the distance score ``D(q_j)`` relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _per_sample_match(
    class_logits: torch.Tensor,
    obj_logits: torch.Tensor,
    targets: torch.Tensor,
    cost_class: float = 1.0,
    cost_obj: float = 1.0,
) -> torch.Tensor:
    """Return the index of the best-matching query for every sample.

    Each sample has exactly one ground-truth known-class label. We pick the
    query that minimises ``cost_class * -log p(y|q) + cost_obj * -log sigmoid(o)``
    — a single-GT specialisation of DETR's Hungarian matcher.
    """
    log_probs = F.log_softmax(class_logits, dim=-1)           # [B, Q, K]
    gather_idx = targets.view(-1, 1, 1).expand(-1, log_probs.size(1), 1)
    cls_cost = -log_probs.gather(-1, gather_idx).squeeze(-1)  # [B, Q]
    obj_cost = -F.logsigmoid(obj_logits)                      # [B, Q]
    total = cost_class * cls_cost + cost_obj * obj_cost
    return total.argmin(dim=-1)                               # [B]


class HungarianMatcher(nn.Module):
    """Tiny wrapper so the trainer can swap in alternative matchers later."""

    def __init__(self, cost_class: float = 1.0, cost_obj: float = 1.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_obj = cost_obj

    @torch.no_grad()
    def forward(
        self,
        class_logits: torch.Tensor,
        obj_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return _per_sample_match(
            class_logits, obj_logits, targets, self.cost_class, self.cost_obj
        )


@dataclass
class NewnessLossWeights:
    w_class: float = 1.0
    w_objectness: float = 1.0
    w_energy: float = 0.1
    w_proto: float = 0.5
    m_in: float = -7.0
    m_out: float = -3.0


class NewnessLoss(nn.Module):
    def __init__(
        self,
        weights: NewnessLossWeights | None = None,
        matcher: HungarianMatcher | None = None,
    ) -> None:
        super().__init__()
        self.w = weights or NewnessLossWeights()
        self.matcher = matcher or HungarianMatcher()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        prototypes: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Compute the composite training loss.

        outputs: forward() dict from the NewnessTransformer.
        targets: [B] known-class index (``>= 0``).
        prototypes: [P, D] current prototype table (from ``pam.prototypes()``).
        class_ids: [P] class id for each prototype row.
        Returns ``(loss, components_dict, matched_query_index)``.
        """
        class_logits = outputs["class_logits"]     # [B, Q, K]
        obj_logits = outputs["obj_logits"]         # [B, Q]
        energy = outputs["energy"]                 # [B, Q]
        proto_feats = outputs["proto_feats"]       # [B, Q, P]

        B, Q, K = class_logits.shape
        device = class_logits.device

        # --- Matching -----------------------------------------------------
        matched = self.matcher(class_logits, obj_logits, targets)  # [B]
        batch_idx = torch.arange(B, device=device)

        # --- Classification loss on the matched query --------------------
        matched_logits = class_logits[batch_idx, matched]          # [B, K]
        cls_loss = F.cross_entropy(matched_logits, targets)

        # --- Objectness loss (matched=1, rest=0) -------------------------
        obj_target = torch.zeros_like(obj_logits)
        obj_target[batch_idx, matched] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

        # --- Energy margin loss ------------------------------------------
        matched_energy = energy[batch_idx, matched]
        mask = torch.ones_like(energy, dtype=torch.bool)
        mask[batch_idx, matched] = False
        unmatched_energy = energy[mask].view(B, Q - 1) if Q > 1 else energy.new_zeros(B, 0)

        energy_loss_in = F.relu(matched_energy - self.w.m_in).pow(2).mean()
        if unmatched_energy.numel() > 0:
            energy_loss_out = F.relu(self.w.m_out - unmatched_energy).pow(2).mean()
        else:
            energy_loss_out = energy.new_zeros(())
        energy_loss = energy_loss_in + energy_loss_out

        # --- Prototype contrastive loss ----------------------------------
        # Pull matched query toward its class's prototypes; push away from others.
        # Uses the PAM projection space.
        q_feat = proto_feats[batch_idx, matched]                   # [B, P_dim]
        q_norm = F.normalize(q_feat, dim=-1)
        p_norm = F.normalize(prototypes, dim=-1)                   # [P, D]
        sim = q_norm @ p_norm.t()                                  # [B, P]

        # For each sample, positives = prototypes whose class == target.
        pos_mask = (class_ids.unsqueeze(0) == targets.unsqueeze(1))  # [B, P]
        # Softmax over prototypes, average negative log-likelihood across positives.
        logits = sim / 0.1  # temperature
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        pos_log_prob = (log_prob * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp_min(1)
        proto_loss = -pos_log_prob.mean()

        total = (
            self.w.w_class * cls_loss
            + self.w.w_objectness * obj_loss
            + self.w.w_energy * energy_loss
            + self.w.w_proto * proto_loss
        )

        components = {
            "loss": total.detach(),
            "cls": cls_loss.detach(),
            "obj": obj_loss.detach(),
            "energy_in": energy_loss_in.detach(),
            "energy_out": energy_loss_out.detach(),
            "proto": proto_loss.detach(),
        }
        return total, components, matched
