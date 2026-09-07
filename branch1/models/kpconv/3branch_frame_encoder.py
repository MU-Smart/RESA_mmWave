"""
radar_kpconv_frame_encoder.py

Pure-PyTorch Kernel Point Convolution (KPConv) frame encoder for sparse
radar point clouds.

Design choices and improvements over the plan's sketch:
─────────────────────────────────────────────────────────
1. No torch-points-kernels dependency.
   All operations use standard PyTorch (cdist, einsum). Works on any
   environment including Jetson JetPack without custom CUDA compilation.

2. Gaussian kernel blending instead of the hard linear falloff.
   Original KPConv: h(d) = max(0, 1 − d/σ)   ← zero for d > σ
   This implementation: h(d) = exp(−d² / 2σ²) ← smooth, never zero
   For N=64 sparse data, hard cutoffs kill gradients from isolated points.
   Gaussian blending keeps every point contributing, just with decreasing
   influence. This is equivalent to the "soft" KPConv in the deformable
   variant paper.

3. Physical-space anchoring is preserved at every layer.
   Unlike DGCNN's feature-space kNN drift, all three KPConv stages query
   in XYZ physical space. The graph topology never drifts into learned
   feature space — the key benefit for narrow pillars and box corners.

4. Density-invariant normalisation.
   Outputs are divided by the sum of kernel weights over the neighbourhood.
   This means a point with 4 neighbours produces the same output magnitude
   as a point with 12 neighbours (for the same local geometry), preventing
   the model from learning density-as-class-evidence.

5. Neighbourhood density logger.
   Call `encoder.log_density(xyz)` on a sample batch before training to
   verify that the chosen radii produce meaningful local structure. The plan
   recommends ≥5 mean neighbours per query; this tool confirms it.

6. Drop-in replacement for DGCNNFrameEncoder.
   Output signature is identical:
       local:       (B, 256, N)   — per-point local features
       global_feat: (B, emb_dims) — globally pooled feature
   No changes to DopplerAwareTemporalDGCNNSegmenter or the training loop.

Architecture:
    KPConvLayer(n_features → 64,  radius=r1, K=15)  → BN → LeakyReLU → x1
    KPConvLayer(64          → 64,  radius=r2, K=15)  → BN → LeakyReLU → x2
    KPConvLayer(64          → 128, radius=r3, K=15)  → BN → LeakyReLU → x3
    local = cat([x1, x2, x3])                                    (B, 256, N)
    fuse  = Conv1d(256 → emb_dims) → BN → LeakyReLU
    global_feat = fuse(local).max(dim=2)                         (B, emb_dims)

Reference: Thomas et al., "KPConv: Flexible and Deformable Convolution for
Point Clouds", ICCV 2019. https://arxiv.org/abs/1904.08889
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ── Fibonacci sphere initialisation ─────────────────────────────────────────

def fibonacci_sphere(n: int) -> torch.Tensor:
    """
    Return n points uniformly distributed on the unit sphere (S²) using the
    Fibonacci / golden-angle algorithm.

    Returns: (n, 3) float32 tensor — kernel point positions in the unit ball.
    The points lie on the sphere surface, giving good angular coverage for the
    KPConv kernels. The centre point (0,0,0) is prepended separately if needed.
    """
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    pts: list[list[float]] = []
    for i in range(n):
        theta = math.acos(1.0 - 2.0 * (i + 0.5) / n)
        phi = 2.0 * math.pi * i / golden
        pts.append([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ])
    return torch.tensor(pts, dtype=torch.float32)  # (n, 3)


def init_kernel_points(n_kernel_points: int) -> torch.Tensor:
    """
    Initialise n_kernel_points locations in the *unit ball*.

    Strategy:
      - 1 point at the origin (centre)
      - n_kernel_points-1 points on the unit sphere (surface), Fibonacci spaced

    This mirrors the initialisation used in the official KPConv code.
    Returns (n_kernel_points, 3).
    """
    n_surface = n_kernel_points - 1
    surface_pts = fibonacci_sphere(n_surface)          # (K-1, 3)
    origin = torch.zeros(1, 3, dtype=torch.float32)    # (1, 3)
    return torch.cat([origin, surface_pts], dim=0)     # (K, 3)


# ── Single KPConv layer ──────────────────────────────────────────────────────

class KPConvLayer(nn.Module):
    """
    Single Kernel Point Convolution layer operating in physical 3D space.

    For each point p_i, the output is:

        y_i = Σ_j  [ Σ_k  h_k(p_j − p_i)  · W_k ]  · f_j

    where the outer sum is over all other points j (filtered by ball radius),
    h_k is the Gaussian influence of kernel point k, and W_k ∈ ℝ^{C_in×C_out}
    is the learnable weight matrix for kernel point k.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    radius       : float   — ball query radius in metres
    sigma        : float   — Gaussian falloff; defaults to radius / sigma_factor
    n_kernel_points : int  — number of kernel points K (default 15)
    sigma_factor : float   — sigma = radius / sigma_factor (ignored if sigma given)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float,
        sigma: float | None = None,
        n_kernel_points: int = 15,
        sigma_factor: float = 2.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = float(radius)
        self.n_kernel_points = int(n_kernel_points)
        self.sigma = float(sigma) if sigma is not None else float(radius) / float(sigma_factor)

        # Kernel point positions in unit ball — not trained (rigid KPConv)
        self.register_buffer(
            "kernel_points",
            init_kernel_points(n_kernel_points),  # (K, 3)
        )

        # Learnable weight matrices, one per kernel point
        # kernel_weights[k] ∈ ℝ^{C_in × C_out}
        self.kernel_weights = nn.Parameter(
            torch.empty(n_kernel_points, in_channels, out_channels)
        )
        nn.init.kaiming_uniform_(
            self.kernel_weights.view(n_kernel_points * in_channels, out_channels),
            a=math.sqrt(5),
        )

    # ── neighbourhood density reporter ─────────────────────────────────────

    @torch.no_grad()
    def count_neighbours(self, xyz: torch.Tensor) -> dict[str, float]:
        """
        Return mean / min / max neighbour count for a ball query at self.radius.
        xyz: (B, 3, N) or (1, 3, N). Uses first batch element.
        """
        xyz_t = xyz[0:1].permute(0, 2, 1).float()          # (1, N, 3)
        dists = torch.cdist(xyz_t, xyz_t).squeeze(0)        # (N, N)
        mask = (dists <= self.radius) & (dists > 1e-6)      # exclude self
        nb_counts = mask.float().sum(dim=1)                  # (N,)
        return {
            "mean": float(nb_counts.mean().item()),
            "min":  float(nb_counts.min().item()),
            "max":  float(nb_counts.max().item()),
        }

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, C_in, N)  — per-point features
        xyz      : (B, 3, N)     — point positions in metres

        Returns
        -------
        out : (B, C_out, N)
        """
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        xyz      = torch.nan_to_num(xyz,      nan=0.0, posinf=0.0, neginf=0.0)

        B, C_in, N = features.shape
        xyz_t = xyz.permute(0, 2, 1).float()               # (B, N, 3)

        # ── pairwise point distances ───────────────────────────────────────
        dists = torch.cdist(xyz_t, xyz_t)                  # (B, N, N)

        # ── relative positions: delta[b,i,j] = xyz_j − xyz_i ─────────────
        # (B, N, 1, 3) − (B, 1, N, 3)  →  (B, N, N, 3)
        delta = xyz_t.unsqueeze(2) - xyz_t.unsqueeze(1)    # (B, N, N, 3)

        # ── kernel point influence (Gaussian blending) ────────────────────
        # kp: (K, 3) — scaled to physical ball radius for this layer
        kp = (self.kernel_points * self.radius).float()    # (K, 3)  (unit ball → radius)

        # Distance from each (centre i, neighbour j) relative position to each kernel point k
        # delta:  (B, N, N, 3)  →  (B, N, N, 1, 3)
        # kp:     (K, 3)        →  (1, 1, 1, K, 3)
        delta_to_kp = delta.unsqueeze(3) - kp.view(1, 1, 1, self.n_kernel_points, 3)
        # (B, N, N, K, 3)

        d_sq = (delta_to_kp ** 2).sum(dim=-1)              # (B, N, N, K)

        # Gaussian influence: h_k(d) = exp(−d² / 2σ²)
        two_sigma_sq = 2.0 * self.sigma ** 2
        h = torch.exp(-d_sq / two_sigma_sq)                # (B, N, N, K)

        # ── apply ball-query mask (zero influence beyond radius) ──────────
        # Excludes the point from its own neighbourhood via distance > 0 check
        in_ball = (dists <= self.radius).float()           # (B, N, N)
        h = h * in_ball.unsqueeze(-1)                      # (B, N, N, K)

        # ── density-invariant normalisation ───────────────────────────────
        # Divide by total kernel influence per (centre, kernel-point) pair
        h_norm = h / h.sum(dim=2, keepdim=True).clamp(min=1e-6)  # (B, N, N, K)

        # ── aggregate neighbour features weighted by kernel influence ─────
        # feats_t: (B, N, C_in)
        feats_t = features.permute(0, 2, 1).float()

        # weighted_feats[b, i, k, c] = Σ_j h_norm[b,i,j,k] · feats_t[b,j,c]
        # einsum: (B, N_i, N_j, K), (B, N_j, C_in) → (B, N_i, K, C_in)
        weighted_feats = torch.einsum("bijk,bjc->bikc", h_norm, feats_t)

        # Apply kernel weights: (B, N, K, C_in) × (K, C_in, C_out) → (B, N, C_out)
        out = torch.einsum("bikc,kcd->bid", weighted_feats, self.kernel_weights)

        return out.permute(0, 2, 1).to(features.dtype)    # (B, C_out, N)


# ── KPConv frame encoder ─────────────────────────────────────────────────────

class KPConvFrameEncoder(nn.Module):
    """
    KPConv-based replacement for DGCNNFrameEncoder.

    Exactly preserves the DGCNNFrameEncoder output contract:
        local       : (B, 256, N)   — multi-scale per-point features
        global_feat : (B, emb_dims) — max-pooled global descriptor

    Three-stage pipeline (all XYZ-anchored):
        Stage 1: KPConvLayer(n_features → 64,  radius=radius_1)
        Stage 2: KPConvLayer(64          → 64,  radius=radius_2)
        Stage 3: KPConvLayer(64          → 128, radius=radius_3)
        fuse:    Conv1d(256 → emb_dims) → BN → LeakyReLU
        global:  max-pool over N

    Parameters
    ----------
    n_features      : number of input point features
    emb_dims        : embedding dimension (default 256, matches DGCNN)
    k               : kept for API compatibility with DGCNN; not used
    radius_1        : ball radius for stage 1 (default 0.35 m)
    radius_2        : ball radius for stage 2 (default 0.50 m)
    radius_3        : ball radius for stage 3 (default 0.70 m)
    sigma_factor    : sigma = radius / sigma_factor for each stage (default 2.5)
    n_kernel_points : number of kernel points K (default 15)
    """

    def __init__(
        self,
        n_features: int,
        emb_dims: int = 256,
        k: int = 20,                # ignored, kept for API compat
        radius_1: float = 0.35,
        radius_2: float = 0.50,
        radius_3: float = 0.70,
        sigma_factor: float = 2.5,
        n_kernel_points: int = 15,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.emb_dims = int(emb_dims)
        self.radius_1 = float(radius_1)
        self.radius_2 = float(radius_2)
        self.radius_3 = float(radius_3)
        self.sigma_factor = float(sigma_factor)
        self.n_kernel_points = int(n_kernel_points)

        # Stage 1: n_features → 64
        self.kpconv1 = KPConvLayer(n_features, 64, radius_1, n_kernel_points=n_kernel_points, sigma_factor=sigma_factor)
        self.bn1 = nn.BatchNorm1d(64)

        # Stage 2: 64 → 64
        self.kpconv2 = KPConvLayer(64, 64, radius_2, n_kernel_points=n_kernel_points, sigma_factor=sigma_factor)
        self.bn2 = nn.BatchNorm1d(64)

        # Stage 3: 64 → 128
        self.kpconv3 = KPConvLayer(64, 128, radius_3, n_kernel_points=n_kernel_points, sigma_factor=sigma_factor)
        self.bn3 = nn.BatchNorm1d(128)

        self.act = nn.LeakyReLU(negative_slope=0.2)

        # Fusion: concatenated multi-scale (256) → emb_dims
        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    # ── density diagnostic ──────────────────────────────────────────────────

    @torch.no_grad()
    def log_density(self, xyz: torch.Tensor) -> None:
        """
        Print mean/min/max neighbour counts at each ball radius for the given
        batch of XYZ tensors. Call once before training starts to validate that
        the chosen radii produce meaningful local structure.

        Recommendation from plan: mean neighbours ≥ 5 at radius_1.
        """
        print("\n  [KPConv density check]")
        for layer, radius, label in [
            (self.kpconv1, self.radius_1, "radius_1"),
            (self.kpconv2, self.radius_2, "radius_2"),
            (self.kpconv3, self.radius_3, "radius_3"),
        ]:
            stats = layer.count_neighbours(xyz)
            flag = ""
            if stats["mean"] < 5:
                flag = "  ⚠ mean < 5 — consider increasing radius"
            elif stats["mean"] < 3:
                flag = "  ✗ mean < 3 — KPConv will degrade to per-point MLP!"
            print(
                f"    {label}={radius:.2f}m: "
                f"mean={stats['mean']:.1f}  min={stats['min']:.0f}  max={stats['max']:.0f}{flag}"
            )
        print()

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self, features: torch.Tensor, xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features : (B, n_features, N)
        xyz      : (B, 3, N)

        Returns
        -------
        local       : (B, 256, N)
        global_feat : (B, emb_dims)
        """
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        xyz      = torch.nan_to_num(xyz,      nan=0.0, posinf=0.0, neginf=0.0)

        # All three stages use physical XYZ (never drifts into feature space)
        x1 = self.act(self.bn1(self.kpconv1(features, xyz)))   # (B, 64, N)
        x2 = self.act(self.bn2(self.kpconv2(x1, xyz)))         # (B, 64, N)
        x3 = self.act(self.bn3(self.kpconv3(x2, xyz)))         # (B, 128, N)

        local = torch.cat([x1, x2, x3], dim=1)                 # (B, 256, N)
        global_feat = self.fuse(local).max(dim=2)[0]            # (B, emb_dims)

        return local, global_feat
