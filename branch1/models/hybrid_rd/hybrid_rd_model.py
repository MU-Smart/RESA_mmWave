"""Model wrapper for hybrid point-cloud + RD/RA patch inference.

Supports an optional accumulatability head (Step 8 of the directness-aware
ghost suppression plan).  When has_acc_head=True, forward() returns a dict
with keys "sem_logits" and "acc_logits".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from branch1.models.hybrid_rd.rd_patch_attention_encoder import RDPatchAttentionEncoder


@dataclass(frozen=True)
class RDPatchModelConfig:
    patch_channels: int = 1
    patch_doppler_bins: int = 17
    patch_range_bins: int = 7
    embed_dim: int = 32


class RDPatchEncoder(nn.Module):
    """Compact per-point encoder for local radar texture patches."""

    def __init__(
        self,
        patch_channels: int = 1,
        embed_dim: int = 32,
        hidden_channels: Sequence[int] = (8, 16),
    ) -> None:
        super().__init__()
        hidden = tuple(int(x) for x in hidden_channels)
        if not hidden:
            raise ValueError("hidden_channels must not be empty.")
        layers: list[nn.Module] = []
        in_ch = int(patch_channels)
        for out_ch in hidden:
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(1, out_ch),
                    nn.SiLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, int(embed_dim)),
            nn.SiLU(inplace=True),
        )
        self.embed_dim = int(embed_dim)
        self.patch_channels = int(patch_channels)

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
            emb = self.proj(self.pool(self.conv(flat))).reshape(bsz, n_pts, self.embed_dim)
            return emb.permute(0, 2, 1).contiguous()
        if patches.ndim == 6:
            bsz, timesteps, n_pts, channels, d_bins, r_bins = patches.shape
            flat = patches.reshape(bsz * timesteps * n_pts, channels, d_bins, r_bins)
            emb = self.proj(self.pool(self.conv(flat))).reshape(
                bsz, timesteps, n_pts, self.embed_dim
            )
            return emb.permute(0, 1, 3, 2).contiguous()
        raise ValueError("Patch tensors must have shape (B,N,C,D,R) or (B,T,N,C,D,R).")


class AccumulatabilityHead(nn.Module):
    """Small MLP head that predicts per-point accumulatability (ghost vs real).

    Takes the center-frame augmented features (original scalar features + patch
    embedding) and outputs a single logit per point.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        augmented_features: torch.Tensor | list[torch.Tensor],
        window_size: int,
    ) -> torch.Tensor:
        """Extract center-frame features and produce ``(B, 1, N)`` acc logits."""
        center_idx = window_size // 2
        if isinstance(augmented_features, torch.Tensor):
            # (B, T, F, N) -> (B, F, N)
            center_feats = augmented_features[:, center_idx, :, :]
        else:
            # list of (B, F, N) -> (B, F, N)
            center_feats = augmented_features[center_idx]
        # (B, F, N) -> (B, N, F)
        center_feats = center_feats.permute(0, 2, 1)
        logits = self.net(center_feats).squeeze(-1)  # (B, N)
        return logits.unsqueeze(1)  # (B, 1, N)


class RDPatchTemporalSegmenter(nn.Module):
    """Wrap an existing temporal point segmenter with RD/RA patch feature fusion.

    Optionally includes a second accumulatability head for ghost detection.

    Parameters
    ----------
    base_model : nn.Module
        Temporal point segmenter (e.g. ``KPConvTemporalSegmenter``).  Must have
        a ``window_size`` attribute and, when ``has_acc_head=True``, a positive
        ``n_features`` attribute.
    patch_channels : int
        Number of input channels in each patch stream (default 1).
    patch_embed_dim : int
        Output embedding dimension of the patch encoders (default 32).
    patch_hidden_channels : Sequence[int]
        Hidden channel sizes for the patch encoder CNN (default ``(8, 16)``).
    use_ra_patches : bool
        Whether to fuse a second RA patch encoder alongside RD (default ``False``).
    has_acc_head : bool
        Whether to add the accumulatability head (default ``False``).
    acc_head_hidden_dim : int
        Hidden dimension of the accumulatability MLP (default 64).
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        patch_channels: int = 1,
        patch_embed_dim: int = 32,
        patch_hidden_channels: Sequence[int] = (8, 16),
        use_ra_patches: bool = False,
        has_acc_head: bool = False,
        acc_head_hidden_dim: int = 64,
        patch_fusion_mode: str = "cnn",
        ra_patch_doppler_bins: int = 9,
        ra_patch_range_bins: int = 7,
        rd_patch_doppler_bins: int = 17,
        rd_patch_range_bins: int = 7,
    ) -> None:
        super().__init__()
        if patch_fusion_mode not in ("cnn", "attention"):
            raise ValueError(f"patch_fusion_mode must be 'cnn' or 'attention', got {patch_fusion_mode!r}.")
        self.base_model = base_model
        self.patch_fusion_mode = str(patch_fusion_mode)

        def _make_patch_encoder(doppler_bins: int, range_bins: int) -> nn.Module:
            if self.patch_fusion_mode == "attention":
                return RDPatchAttentionEncoder(
                    patch_channels=patch_channels,
                    embed_dim=patch_embed_dim,
                    doppler_bins=doppler_bins,
                    range_bins=range_bins,
                )
            return RDPatchEncoder(
                patch_channels=patch_channels,
                embed_dim=patch_embed_dim,
                hidden_channels=patch_hidden_channels,
            )

        self.rd_patch_encoder = _make_patch_encoder(rd_patch_doppler_bins, rd_patch_range_bins)
        self.use_ra_patches = bool(use_ra_patches)
        self.ra_patch_encoder = (
            _make_patch_encoder(ra_patch_doppler_bins, ra_patch_range_bins)
            if self.use_ra_patches
            else None
        )
        self.patch_embed_dim = int(patch_embed_dim)
        self.patch_channels = int(patch_channels)
        self.has_acc_head = bool(has_acc_head)

        if self.has_acc_head:
            # The augmented features passed to the base model have dimension
            # base_model.n_features (= original scalar features + enabled patch embeddings).
            acc_input_dim = int(getattr(base_model, "n_features", 0))
            if acc_input_dim <= 0:
                raise ValueError(
                    "has_acc_head=True requires base_model to have a positive "
                    "n_features attribute (the number of features after patch "
                    "embedding concatenation)."
                )
            self.acc_head = AccumulatabilityHead(
                input_dim=acc_input_dim,
                hidden_dim=int(acc_head_hidden_dim),
            )

    @staticmethod
    def _coerce_patch_list(
        window_patches: torch.Tensor | list[torch.Tensor],
        *,
        window_size: int,
        patch_kind: str,
    ) -> list[torch.Tensor]:
        if isinstance(window_patches, torch.Tensor):
            if window_patches.ndim != 6:
                raise ValueError(f"Tensor {patch_kind} patches must have shape (B,T,N,C,D,R).")
            if int(window_patches.size(1)) != int(window_size):
                raise ValueError(f"{patch_kind.upper()} patch temporal dimension does not match window_size.")
            return [window_patches[:, t, ...] for t in range(window_patches.size(1))]
        if not isinstance(window_patches, list):
            raise ValueError(f"{patch_kind.upper()} patches must be a tensor or list of tensors.")
        if len(window_patches) != int(window_size):
            raise ValueError(f"{patch_kind.upper()} patch list length does not match window_size.")
        return window_patches

    def _append_embeddings(
        self,
        window_features: torch.Tensor | list[torch.Tensor],
        window_rd_patches: torch.Tensor | list[torch.Tensor],
        window_ra_patches: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        window_size = int(getattr(self.base_model, "window_size", 1))

        if isinstance(window_features, torch.Tensor):
            if window_features.ndim != 4:
                raise ValueError("Tensor features must have shape (B,T,F,N).")
            rd_patch_embeddings = self.rd_patch_encoder(window_rd_patches)
            if rd_patch_embeddings.ndim != 4:
                raise ValueError("Encoded tensor patches must have shape (B,T,E,N).")
            if self.use_ra_patches:
                if window_ra_patches is None:
                    raise ValueError("RA patches are required but were not provided.")
                ra_patch_embeddings = self.ra_patch_encoder(window_ra_patches)
                if ra_patch_embeddings.ndim != 4:
                    raise ValueError("Encoded tensor RA patches must have shape (B,T,E,N).")
                return torch.cat([window_features, rd_patch_embeddings, ra_patch_embeddings], dim=2)
            return torch.cat([window_features, rd_patch_embeddings], dim=2)

        rd_patch_list = self._coerce_patch_list(window_rd_patches, window_size=window_size, patch_kind="rd")
        ra_patch_list = None
        if self.use_ra_patches:
            if window_ra_patches is None:
                raise ValueError("RA patches are required but were not provided.")
            ra_patch_list = self._coerce_patch_list(window_ra_patches, window_size=window_size, patch_kind="ra")
        out: list[torch.Tensor] = []
        if self.use_ra_patches and ra_patch_list is not None:
            for feats_t, rd_patches_t, ra_patches_t in zip(window_features, rd_patch_list, ra_patch_list):
                rd_emb_t = self.rd_patch_encoder(rd_patches_t)
                ra_emb_t = self.ra_patch_encoder(ra_patches_t)
                out.append(torch.cat([feats_t, rd_emb_t, ra_emb_t], dim=1))
        else:
            for feats_t, patches_t in zip(window_features, rd_patch_list):
                emb_t = self.rd_patch_encoder(patches_t)
                out.append(torch.cat([feats_t, emb_t], dim=1))
        return out

    def forward(
        self,
        window_features: torch.Tensor | list[torch.Tensor],
        window_xyz: torch.Tensor | list[torch.Tensor],
        frame_meta: torch.Tensor | None,
        window_rd_patches: torch.Tensor | list[torch.Tensor],
        window_ra_patches: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        torch.Tensor
            Semantic logits ``(B, C, N)`` when ``has_acc_head=False``.
        dict[str, torch.Tensor]
            Dictionary with keys ``sem_logits`` and ``acc_logits`` when
            ``has_acc_head=True``.
        """
        augmented_features = self._append_embeddings(window_features, window_rd_patches, window_ra_patches)
        sem_logits = self.base_model(augmented_features, window_xyz, frame_meta)

        if not self.has_acc_head:
            return sem_logits

        window_size = int(getattr(self.base_model, "window_size", 1))
        acc_logits = self.acc_head(augmented_features, window_size)
        return {
            "sem_logits": sem_logits,
            "acc_logits": acc_logits,
        }
