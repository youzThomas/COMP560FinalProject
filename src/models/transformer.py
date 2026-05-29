"""Transformer backbone adapted for 1-D multi-channel time-series.

The original report describes a Swin/ViT encoder operating on image patches; for
the Mill sensor windows we substitute a ``PatchEmbed1D`` that slices the input
along the temporal axis and projects each patch to ``d_model``. The encoder
applies global self-attention so each query sees the full window, matching the
"full-image context" requirement in the report.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed1D(nn.Module):
    """Split a [B, T, C] tensor into non-overlapping temporal patches and project."""

    def __init__(self, in_channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T] -> [B, d_model, T/patch]
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)  # [B, N_tokens, d_model]
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderDecoder(nn.Module):
    """DETR-style encoder + decoder.

    Encoder produces context tokens via global self-attention; decoder cross-
    attends learnable object queries over those tokens to yield object-centric
    representations that downstream heads operate on.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        num_queries: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)

        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.num_queries = num_queries

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (query_features, memory_tokens).

        tokens: [B, N, d_model]
        returns queries: [B, Q, d_model], memory: [B, N, d_model]
        """
        B = tokens.size(0)
        memory = self.encoder(self.pos_enc(tokens))
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return self.norm(out), memory
