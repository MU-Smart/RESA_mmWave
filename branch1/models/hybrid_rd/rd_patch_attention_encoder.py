"""Attention-based alternative to RDPatchEncoder for RD/RA patch fusion.

Treats each cell of a point's local range-Doppler (or range-azimuth) patch as a
token and lets cells attend to each other before pooling to one embedding,
instead of collapsing the patch through a CNN + average-pool. Matches
RDPatchEncoder's forward() contract exactly so it can be selected via
RDPatchTemporalSegmenter's patch_fusion_mode flag with no other code changes.
"""

from __future__ import annotations

import torch
from torch import nn


class RDPatchAttentionEncoder(nn.Module):
    """Encode a per-point RD/RA patch via self-attention over its grid cells."""

    def __init__(
        self,
        patch_channels: int = 1,
        embed_dim: int = 32,
        doppler_bins: int = 17,
        range_bins: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_channels = int(patch_channels)
        self.embed_dim = int(embed_dim)
        self.doppler_bins = int(doppler_bins)
        self.range_bins = int(range_bins)

        self.cell_proj = nn.Linear(self.patch_channels, self.embed_dim)
        self.pos_row = nn.Parameter(torch.zeros(self.doppler_bins, self.embed_dim))
        self.pos_col = nn.Parameter(torch.zeros(self.range_bins, self.embed_dim))
        nn.init.normal_(self.pos_row, std=0.02)
        nn.init.normal_(self.pos_col, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _positions(self) -> torch.Tensor:
        pos = self.pos_row[:, None, :] + self.pos_col[None, :, :]
        return pos.reshape(self.doppler_bins * self.range_bins, self.embed_dim)

    def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """flat: (M, C, D, R) -> (M, E)"""
        m, c, d, r = flat.shape
        tokens = flat.reshape(m, c, d * r).permute(0, 2, 1)  # (M, D*R, C)
        tokens = self.cell_proj(tokens) + self._positions().unsqueeze(0)
        tokens = self.transformer(tokens)  # (M, D*R, E)
        return tokens.mean(dim=1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode patches.

        Accepts either:

        - ``(B, N, C, D, R)`` and returns ``(B, E, N)``
        - ``(B, T, N, C, D, R)`` and returns ``(B, T, E, N)``
        """
        patches = torch.nan_to_num(patches.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if patches.ndim == 5:
            bsz, n_pts, channels, d_bins, r_bins = patches.shape
            flat = patches.reshape(bsz * n_pts, channels, d_bins, r_bins)
            emb = self._encode_flat(flat).reshape(bsz, n_pts, self.embed_dim)
            return emb.permute(0, 2, 1).contiguous()
        if patches.ndim == 6:
            bsz, timesteps, n_pts, channels, d_bins, r_bins = patches.shape
            flat = patches.reshape(bsz * timesteps * n_pts, channels, d_bins, r_bins)
            emb = self._encode_flat(flat).reshape(bsz, timesteps, n_pts, self.embed_dim)
            return emb.permute(0, 1, 3, 2).contiguous()
        raise ValueError("Patch tensors must have shape (B,N,C,D,R) or (B,T,N,C,D,R).")
