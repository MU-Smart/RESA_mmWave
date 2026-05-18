"""
3branch_unet.py
===============
U-Net with spatial attention for Branch 3 dense free-space segmentation.

Architecture follows Hu et al. (arXiv:2411.00499) with one extension:
the input gains a +1 MapBuilder prior channel alongside the RD tensor.

Input tensor shape:
    (2C + 1, H, W) where:
        2C = 24  — real and imaginary parts of 12 virtual RD channels
                   (3 TX × 4 RX = 12 virtual, separated real/imag = 24)
        +1  = 1  — MapBuilder polar prior (accumulated structure occupancy)
        H   = 32 — Doppler bins (loops_per_frame)
        W   = 256— Range bins   (adc_samples)

Output:
    (H, W) float32 — sigmoid probability of unobstructed FoV per cell
    1.0 = free (navigable), 0.0 = blocked

Key design decisions:
    - Spatial attention (Hu et al.) outperforms CAM and Swin Transformer
      on IoU by ~3% while being faster — used in every DoubleConv block
    - Instance normalisation instead of BatchNorm — more stable at small
      batch sizes (batch=16 as per Hu et al.) and avoids batch statistics
      accumulating differently between training and Jetson inference
    - Average pooling (not max pooling) for downsampling — preserves
      background energy distribution which carries free-space signal
    - Prior channel is concatenated at input, not injected mid-network,
      so the encoder sees spatial correspondence from the first layer

Reference: Hu et al., "Cross-Modal Semantic Segmentation for Indoor
Environmental Perception Using Single-Chip Millimeter-Wave Radar Raw Data",
arXiv:2411.00499, December 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Spatial attention module (Hu et al. Fig 3c)
# =============================================================================

class SpatialAttention(nn.Module):
    """
    Spatial attention using max-pool and avg-pool along channel dimension.
    Concatenates the two single-channel maps → Conv2d → sigmoid → multiply.

    Per Hu et al.: this captures both peak values (max) and background
    noise suppression (avg), producing a spatial attention map that focuses
    on navigationally relevant regions of the RD tensor.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        max_pool = x.max(dim=1, keepdim=True)[0]   # (B, 1, H, W)
        avg_pool = x.mean(dim=1, keepdim=True)      # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([max_pool, avg_pool], dim=1)))
        return x * attn


# =============================================================================
# Double convolution block with spatial attention
# =============================================================================

class DoubleConv(nn.Module):
    """
    Two consecutive:
        Conv2d(3×3, pad=1) → InstanceNorm → SpatialAttention → ReLU

    SpatialAttention is applied before ReLU so the attention weights
    are computed on pre-activation features — more expressive than
    post-activation gating for small feature maps.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            SpatialAttention(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            SpatialAttention(),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =============================================================================
# Encoder downsampling block
# =============================================================================

class DownBlock(nn.Module):
    """AvgPool(2×2) → DoubleConv — halves spatial dims, doubles channels."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# =============================================================================
# Decoder upsampling block
# =============================================================================

class UpBlock(nn.Module):
    """TransposeConv(2×) + skip connection → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # in_ch = out_ch (skip) + out_ch (up)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from non-power-of-2 input dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# =============================================================================
# U-Net
# =============================================================================

class BEVUNet(nn.Module):
    """
    U-Net for polar BEV free-space segmentation from RD tensor input.

    Architecture change (v2, Apr 2026)
    ----------------------------------
    Previous version produced an output matching the RD tensor shape
    (H_in, W_in) = (32, 256) because the final head was only a Conv2d(1x1).
    Labels are generated in polar (range, azimuth) coordinates at (128, 128)
    by the historical offline Branch 3 BEV label-generation pipeline
    (not present in this checkout), following Hu et al. (arXiv:2411.00499).

    The old training code resized the (128, 128) label down to (32, 256)
    to match the output shape, which crushed range information (128 -> 32,
    4x compression) and stretched azimuth to match the Range axis
    (128 -> 256, 2x stretch) — both semantically wrong. The model could
    not learn a meaningful mapping and produced near-constant output.

    This version adds a final projection head that takes the decoder's
    (base_ch, 32, 256) feature map and produces a (1, 128, 128) output
    via bilinear resize + 1x1 conv + sigmoid. Labels are now trained at
    their native (128, 128) resolution with no information loss.

    This matches the paper's setup more closely: the paper's input tensor
    happens to be (128, 128) so no resize is needed in their case, but
    the semantic relationship (input axes -> (Doppler, Range), output
    axes -> (Range, Azimuth)) is identical to ours. The network learns
    the implicit DOA transform via the 24 real/imag virtual-channel inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 24 = real+imag of 12 virtual
        RD channels (3 TX x 4 RX). Set to 25 to enable a MapBuilder polar
        prior input channel (must be provided during both train and
        inference to avoid distribution mismatch).
    base_ch : int
        Base channel count for the encoder. Doubles at each stage.
        Default 32 matches Hu et al. (C32 -> C64 -> C128).
    out_h : int
        Output height (range bins). Default 128 matches label shape.
    out_w : int
        Output width (azimuth bins). Default 128 matches label shape.

    Forward input:  (B, in_channels, H_in=32, W_in=256)
    Forward output: (B, out_h=128, out_w=128) — sigmoid probabilities
        axis 0 = range (0 near, 127 far), axis 1 = azimuth (0 = -45 deg,
        127 = +45 deg) — matching the historical 128x128 polar BEV label
        convention used for Branch 3 training.
    """

    def __init__(self,
                 in_channels: int = 24,
                 base_ch: int = 32,
                 out_h: int = 128,
                 out_w: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_ch     = base_ch
        self.out_h       = out_h
        self.out_w       = out_w

        # Encoder
        self.enc1 = DoubleConv(in_channels,    base_ch)       # skip 1
        self.enc2 = DownBlock(base_ch,         base_ch * 2)   # skip 2
        self.enc3 = DownBlock(base_ch * 2,     base_ch * 4)   # skip 3

        # Bottleneck
        self.bottleneck = DownBlock(base_ch * 4, base_ch * 8)

        # Decoder — returns to input resolution (32, 256)
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch)

        # Polar reprojection head: (base_ch, 32, 256) -> (1, out_h, out_w)
        # The network's feature map lives in (Doppler, Range) coordinates
        # but the supervised label lives in (Range, Azimuth) polar space.
        # We resize first (bilinear) then project to 1 output channel.
        # The network is free to learn that the Doppler axis encodes
        # azimuth information via inter-channel phase patterns (the 24
        # real/imag virtual-channel inputs contain DOA phase data).
        self.polar_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, H_in, W_in) — RD tensor, typically (B, 24, 32, 256)
        returns: (B, out_h, out_w) sigmoid probabilities in polar (range, az) space
        """
        s1 = self.enc1(x)           # (B, 32,  H,    W)
        s2 = self.enc2(s1)          # (B, 64,  H/2,  W/2)
        s3 = self.enc3(s2)          # (B, 128, H/4,  W/4)
        b  = self.bottleneck(s3)    # (B, 256, H/8,  W/8)
        d3 = self.dec3(b,  s3)      # (B, 128, H/4,  W/4)
        d2 = self.dec2(d3, s2)      # (B, 64,  H/2,  W/2)
        d1 = self.dec1(d2, s1)      # (B, 32,  H,    W)

        # Resize (Doppler=32, Range=256) feature map to polar
        # (Range=out_h, Azimuth=out_w). The spatial transformation from
        # (Doppler, Range) to (Range, Azimuth) is learned by the conv
        # block inside polar_head, not just the resize.
        resized = F.interpolate(
            d1,
            size=(self.out_h, self.out_w),
            mode="bilinear",
            align_corners=False,
        )                           # (B, 32, out_h, out_w)
        logits = self.polar_head(resized)   # (B, 1, out_h, out_w)
        return torch.sigmoid(logits).squeeze(1)  # (B, out_h, out_w)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# RD tensor preprocessing
# =============================================================================

def rd_cube_to_input(
    rd_cube: "np.ndarray",
    prior: "np.ndarray | None" = None,
    use_prior: bool = False,
) -> "torch.Tensor":
    """
    Convert a raw rd_cube from _radar_tensors.npz into a model input tensor.

    Parameters
    ----------
    rd_cube : complex64 ndarray (num_tx, loops_per_frame, num_rx, adc_samples)
              Shape from IWR1843: (3, 32, 4, 256)
    prior   : float32 ndarray (32, 256) or None
              MapBuilder polar prior reprojected to RD grid shape. Only used
              when use_prior=True. Ignored otherwise.
    use_prior : bool
              Whether to append the MapBuilder prior as a 25th input channel.
              Default False (matches the v2 24-channel model). Set True only
              if the checkpoint was trained with in_channels=25.

    Returns
    -------
    torch.Tensor
        Shape (1, 24, 32, 256) when use_prior=False,
        or    (1, 25, 32, 256) when use_prior=True.
    """
    import numpy as np

    num_tx, n_dopp, num_rx, n_range = rd_cube.shape  # (3, 32, 4, 256)

    # Reshape to virtual channels: (12, 32, 256) complex
    virtual = rd_cube.reshape(num_tx * num_rx, n_dopp, n_range)

    # Split real and imaginary: (24, 32, 256)
    real_part = virtual.real.astype(np.float32)
    imag_part = virtual.imag.astype(np.float32)
    rd_tensor = np.concatenate([real_part, imag_part], axis=0)  # (24, 32, 256)

    # Normalise each channel to [-1, 1] independently
    for c in range(rd_tensor.shape[0]):
        ch = rd_tensor[c]
        mx = float(np.abs(ch).max())
        if mx > 1e-9:
            rd_tensor[c] = ch / mx

    if use_prior:
        # Concatenate prior channel
        if prior is None:
            prior_ch = np.zeros((1, n_dopp, n_range), dtype=np.float32)
        else:
            p = prior.astype(np.float32)
            mx = float(p.max())
            if mx > 1e-9:
                p = p / mx
            prior_ch = p[np.newaxis, :, :]  # (1, 32, 256)
        inp = np.concatenate([rd_tensor, prior_ch], axis=0)  # (25, 32, 256)
    else:
        inp = rd_tensor                                       # (24, 32, 256)

    return torch.from_numpy(inp).unsqueeze(0)


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_unet_checkpoint(
    path: "Path",
    epoch: int,
    model: BEVUNet,
    extra: dict | None = None,
) -> None:
    import torch
    payload = {
        "branch":      "3",
        "model":       "BEVUNet",
        "epoch":       epoch,
        "in_channels": model.in_channels,
        "base_ch":     model.base_ch,
        "out_h":       model.out_h,
        "out_w":       model.out_w,
        "model_state": model.state_dict(),
        **(extra or {}),
    }
    torch.save(payload, path)


def load_unet_checkpoint(
    path: "Path",
    map_location: str = "cpu",
) -> tuple[dict, BEVUNet]:
    """
    Load a BEVUNet checkpoint.

    Backward compatibility: v1 checkpoints (before Apr 2026) did not save
    out_h / out_w and produced output at RD tensor shape. Loading a v1
    checkpoint into the v2 model (which has a polar_head) will fail at
    load_state_dict because the v1 state_dict has `head.weight` /
    `head.bias` but the v2 state_dict expects `polar_head.*` keys.

    When a v1 checkpoint is detected we raise a clear error rather than
    producing a silently-misconfigured model. Retrain with the v2 script
    to produce a v2-compatible checkpoint.
    """
    import torch
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    state = ckpt["model_state"]
    is_v1 = any(k.startswith("head.") for k in state.keys()) and \
            not any(k.startswith("polar_head.") for k in state.keys())
    if is_v1:
        raise RuntimeError(
            "Detected a v1 BEVUNet checkpoint (no polar_head). The v2 model "
            "has an incompatible architecture — retrain with 3branch_train_unet.py "
            "to produce a v2 checkpoint. If you need to keep the old checkpoint, "
            "set UNET_PT='' in the nav loop to disable Branch 3 until retrained."
        )

    model = BEVUNet(
        in_channels=int(ckpt.get("in_channels", 24)),
        base_ch    =int(ckpt.get("base_ch", 32)),
        out_h      =int(ckpt.get("out_h", 128)),
        out_w      =int(ckpt.get("out_w", 128)),
    )
    model.load_state_dict(state)
    model.eval()
    return ckpt, model


def input_from_npz_payload(
    payload: "dict | np.lib.npyio.NpzFile",
    prior: "np.ndarray | None",
    in_channels: int = 24,
) -> "torch.Tensor | None":
    """
    Build a model input tensor from a branch3_hybrid_rd sidecar frame payload.

    The sidecar stores `rd_cube` (complex64, shape (3, 32, 4, 256)) and
    optionally `rd_power` (float32, shape (32, 256)).

    Returns (1, in_channels, 32, 256) float32 tensor, or None if rd_cube
    is missing or has an unexpected shape.
    """
    import numpy as np
    rd_cube = payload.get("rd_cube") if hasattr(payload, "get") else payload["rd_cube"]
    if rd_cube is None:
        return None
    rd_cube = np.asarray(rd_cube)
    if rd_cube.ndim != 4 or rd_cube.shape[0] != 3 or rd_cube.shape[2] != 4:
        return None
    use_prior = (in_channels == 25)
    return rd_cube_to_input(rd_cube, prior=prior, use_prior=use_prior)


def sector_probabilities(
    prob_map: "np.ndarray",
    axis: str = "columns",
    az_fov_deg: float = 90.0,
    sector_half_deg: float = 20.0,
) -> dict:
    """
    Summarise a BEVUNet free-space probability map into left/center/right
    sector mean probabilities.

    Parameters
    ----------
    prob_map    : (out_h, out_w) float32 — sigmoid output after squeeze(0),
                  axis-1 = azimuth from -az_fov_deg/2 to +az_fov_deg/2.
    axis        : "columns" uses the width axis as azimuth (default);
                  "rows" uses the height axis; "global" returns a single
                  scalar averaged over the full map.
    az_fov_deg  : total azimuth field of view in degrees (default 90).
    sector_half_deg : half-width of the center sector in degrees (default 20).
                  Points outside ±sector_half_deg go to left/right.
    """
    import numpy as np

    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.ndim == 3 and prob_map.shape[0] == 1:
        prob_map = prob_map[0]
    if prob_map.ndim != 2:
        return {"left": 0.0, "center": 0.0, "right": 0.0, "full": 0.0}

    h, w = prob_map.shape

    if axis == "global":
        v = float(prob_map.mean())
        return {"left": v, "center": v, "right": v, "full": v}

    # Azimuth spans -fov/2 to +fov/2 linearly across the width axis.
    if axis == "rows":
        prob_map = prob_map.T  # make width = azimuth
        _, w = prob_map.shape

    az = np.linspace(-az_fov_deg / 2, az_fov_deg / 2, w)
    left_mask   = az <  -sector_half_deg
    center_mask = (az >= -sector_half_deg) & (az <= sector_half_deg)
    right_mask  = az >   sector_half_deg

    def _mean(mask):
        cols = prob_map[:, mask]
        return float(cols.mean()) if cols.size > 0 else 0.0

    return {
        "available": True,
        "left":      _mean(left_mask),
        "center":    _mean(center_mask),
        "right":     _mean(right_mask),
        "full":      float(prob_map.mean()),
    }


# =============================================================================
# Quick sanity check
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    print("BEVUNet v2 sanity check")
    model = BEVUNet(in_channels=24, base_ch=32, out_h=128, out_w=128)
    print(f"  Parameters: {model.n_params():,}")

    # Simulate one batch from IWR1843 (3, 32, 4, 256) rd_cube
    rd_cube = (np.random.randn(3, 32, 4, 256) +
               1j * np.random.randn(3, 32, 4, 256)).astype(np.complex64)

    inp = rd_cube_to_input(rd_cube, use_prior=False)
    print(f"  Input shape : {tuple(inp.shape)}")  # (1, 24, 32, 256)

    with torch.no_grad():
        out = model(inp)
    print(f"  Output shape: {tuple(out.shape)}")   # (1, 128, 128)
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    print("  OK")
