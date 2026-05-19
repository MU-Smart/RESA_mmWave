"""
radar_3dgcnn_common_kpconv.py

KPConvTemporalSegmenter — a drop-in subclass of DopplerAwareTemporalDGCNNSegmenter
that replaces only the DGCNNFrameEncoder with KPConvFrameEncoder.

Everything else — temporal transformer, segmentation head, training loop,
evaluation framework, checkpoint format — is unchanged.

Design rationale:
    The only architectural change is self.frame_encoder = KPConvFrameEncoder(...).
    The temporal transformer operates on the max-pooled global descriptor from
    whatever frame encoder is used; it has no knowledge of encoder internals.
    The segmentation head receives the concatenated (local, global, temporal)
    features — the local features are (B, 256, N) in both cases.

    This means the same training loop, the same loss function, the same early
    stopping logic, and the same evaluation script all work unchanged.

Plan improvement — encoder_type in checkpoint:
    The checkpoint saves "encoder_type": "kpconv" so that any downstream
    script (evaluate_5class_temporal.py, inference_simulation.py) can
    reconstruct the correct model variant from the checkpoint alone, without
    needing to know at call time which encoder was used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from radar_3dgcnn_common_doorwall import (
    BUCKET_ORDER,
    FEATURE_COLS,
    FRAME_META_NAMES,
    N_CLASSES,
    N_FEATURES,
    N_FRAME_META,
    DopplerAwareTemporalDGCNNSegmenter,
    TemporalRadarTrainDataset,
    TemporalRadarEvalDataset,
    build_frame_records,
    build_temporal_windows,
    compute_feature_stats,
    ensure_feature_columns,
    filter_known_buckets,
    temporal_eval_collate_fn,
)
from radar_kpconv_frame_encoder import KPConvFrameEncoder


class KPConvTemporalSegmenter(DopplerAwareTemporalDGCNNSegmenter):
    """
    Temporal radar segmenter with KPConv-based frame encoder.

    Subclasses DopplerAwareTemporalDGCNNSegmenter and replaces only
    self.frame_encoder. All temporal and classification logic is inherited.

    Extra constructor parameters (beyond the base class):
        radius_1        : float — ball radius for KPConv stage 1 (default 0.35 m)
        radius_2        : float — ball radius for KPConv stage 2 (default 0.50 m)
        radius_3        : float — ball radius for KPConv stage 3 (default 0.70 m)
        sigma_factor    : float — Gaussian σ = radius / sigma_factor (default 2.5)
        n_kernel_points : int   — number of kernel points K (default 15)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_classes: int = N_CLASSES,
        window_size: int = 3,
        k: int = 20,
        emb_dims: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        dropout: float = 0.25,
        frame_meta_dim: int = N_FRAME_META,
        gate_hidden: int = 128,
        # KPConv-specific
        radius_1: float = 0.35,
        radius_2: float = 0.50,
        radius_3: float = 0.70,
        sigma_factor: float = 2.5,
        n_kernel_points: int = 15,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_classes=n_classes,
            window_size=window_size,
            k=k,
            emb_dims=emb_dims,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            dropout=dropout,
            frame_meta_dim=frame_meta_dim,
            gate_hidden=gate_hidden,
        )
        # KPConv radii (stored for checkpoint serialisation)
        self.radius_1 = float(radius_1)
        self.radius_2 = float(radius_2)
        self.radius_3 = float(radius_3)
        self.sigma_factor = float(sigma_factor)
        self.n_kernel_points = int(n_kernel_points)

        # Replace ONLY the frame encoder — everything else is unchanged
        self.frame_encoder = KPConvFrameEncoder(
            n_features=n_features,
            emb_dims=emb_dims,
            k=k,
            radius_1=radius_1,
            radius_2=radius_2,
            radius_3=radius_3,
            sigma_factor=sigma_factor,
            n_kernel_points=n_kernel_points,
        )

    def log_density(self, xyz: torch.Tensor) -> None:
        """Delegate to the underlying KPConvFrameEncoder density logger."""
        self.frame_encoder.log_density(xyz)


def load_kpconv_checkpoint(
    ckpt_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict, KPConvTemporalSegmenter]:
    """
    Load a checkpoint saved by train_3dgcnn_kpconv.py.
    Returns (ckpt_dict, model).
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if ckpt.get("encoder_type", "dgcnn") != "kpconv":
        raise ValueError(
            f"Checkpoint at {ckpt_path} has encoder_type="
            f"'{ckpt.get('encoder_type')}', expected 'kpconv'."
        )
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    bucket_order = ckpt.get("bucket_order", BUCKET_ORDER)
    model = KPConvTemporalSegmenter(
        n_features=len(feature_cols),
        n_classes=len(bucket_order),
        window_size=int(ckpt.get("window_size", 3)),
        k=int(ckpt.get("k", 20)),
        emb_dims=int(ckpt.get("emb_dims", 256)),
        temporal_layers=int(ckpt.get("temporal_layers", 2)),
        temporal_heads=int(ckpt.get("temporal_heads", 4)),
        dropout=float(ckpt.get("dropout", 0.0)),
        frame_meta_dim=int(ckpt.get("frame_meta_dim", N_FRAME_META)),
        gate_hidden=int(ckpt.get("gate_hidden", 128)),
        radius_1=float(ckpt.get("radius_1", 0.35)),
        radius_2=float(ckpt.get("radius_2", 0.50)),
        radius_3=float(ckpt.get("radius_3", 0.70)),
        sigma_factor=float(ckpt.get("sigma_factor", 2.5)),
        n_kernel_points=int(ckpt.get("n_kernel_points", 15)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt, model
