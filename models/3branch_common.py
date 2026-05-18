"""
3branch_common.py
=================
Single consolidated common module for the 3-branch radar navigation pipeline.

Replaces:
    radar_3dgcnn_common_doorwall.py   — label contract, dataset classes
    radar_3dgcnn_common_kpconv.py     — KPConv segmenter and checkpoint loader
    corridor_soft_structural.py       — RANSAC plane fitting, soft structural features

Branch 1 model: KPConvTemporalSegmenter (KPConv only — no DGCNN fallback).

Change 3 additions:
    RDPatchEncoder  — depthwise-separable CNN with SE attention for 7×17 RD patches
    RAPatchEncoder  — same architecture for 7×9 RA patches
    KPConvTemporalSegmenter.forward() now accepts optional rd_patches and ra_patches
    per temporal frame. Each encoder outputs 32-dim embeddings concatenated onto
    the 18 scalar features → 82 total features entering KPConv.
    Patch dimensions:
        RD: 7 range bins × 17 Doppler bins = 119 values per point
        RA: 7 range bins × 9 azimuth bins  = 63 values per point
    Embed dim: 32 each → total augmented feature width = 18 + 32 + 32 = 82

Branch 1 feature contract (18 features):
    Base (10):
        x, y, z, range_m, azimuth_deg, elevation_deg,
        doppler, snr, local_density, persist_score
    Geometric (1):
        z_norm_range
    Soft structural (7):
        z_above_floor, corridor_margin, wall_anomaly,
        floor_ang, wall_ang, has_floor, has_wall

Note: frame_doppler_abs and frame_doppler_std from the legacy 20-feature
extended mode are intentionally omitted. Their signal is superseded by
the per-point RD patch embeddings added in Branch 1 augmentation (change 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


# =============================================================================
# Label contract
# =============================================================================

RAW_BUCKET_ORDER = ["wall", "floor", "door", "pillar", "human", "box_like"]

LABEL_REMAP = {
    "door":     "structure",
    "wall":     "structure",
    "pillar":   "structure",
    "box_like": "structure",
}

BUCKET_ORDER = ["structure", "floor", "human"]
N_CLASSES    = 3
BUCKET_TO_IDX = {b: i for i, b in enumerate(BUCKET_ORDER)}
IDX_TO_BUCKET = {i: b for b, i in BUCKET_TO_IDX.items()}


# =============================================================================
# Feature contract — 18 features, locked for Branch 1
# =============================================================================

BASE_FEATURE_COLS = [
    "x", "y", "z",
    "range_m", "azimuth_deg", "elevation_deg",
    "doppler", "snr",
    "local_density", "persist_score",
]

GEOMETRIC_COLS = [
    "z_norm_range",
]

SOFT_STRUCTURAL_COLS = [
    "z_above_floor",
    "corridor_margin",
    "wall_anomaly",
    "floor_ang",
    "wall_ang",
    "has_floor",
    "has_wall",
]

FEATURE_COLS = BASE_FEATURE_COLS + GEOMETRIC_COLS + SOFT_STRUCTURAL_COLS
N_FEATURES   = len(FEATURE_COLS)  # 18

FRAME_META_NAMES = [
    "dt_video_norm",
    "dt_radar_norm",
    "mean_doppler_norm",
    "abs_mean_doppler_norm",
    "std_doppler_norm",
]
N_FRAME_META = len(FRAME_META_NAMES)  # 5

# =============================================================================
# Patch encoder constants — change 3
# =============================================================================

# RD patch: 7 range bins (±3) × 17 Doppler bins (±8)
RD_PATCH_RANGE  = 7
RD_PATCH_DOPPLER = 17
RD_PATCH_SIZE   = RD_PATCH_RANGE * RD_PATCH_DOPPLER  # 119

# RA patch: 7 range bins (±3) × 9 azimuth bins (±4)
RA_PATCH_RANGE   = 7
RA_PATCH_AZIMUTH = 9
RA_PATCH_SIZE    = RA_PATCH_RANGE * RA_PATCH_AZIMUTH  # 63

PATCH_EMBED_DIM  = 32   # output embedding dim for each encoder
# Total augmented features entering KPConv: 18 + 32 + 32 = 82
N_FEATURES_AUGMENTED = N_FEATURES + PATCH_EMBED_DIM * 2  # 82

SESSION_COL       = "session"
SPLIT_COL         = "split"
FRAME_COL         = "radar_frame_num"
VIDEO_FRAME_COL   = "video_frame_index"
LABEL_COL         = "bucket"
DOPPLER_COL       = "doppler"


# =============================================================================
# Label utilities
# =============================================================================

def apply_label_remap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].replace(LABEL_REMAP)
    return df


def filter_known_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_label_remap(df)
    return df[df[LABEL_COL].isin(BUCKET_ORDER)].reset_index(drop=True)


# =============================================================================
# Feature utilities
# =============================================================================

def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or FEATURE_COLS
    df = df.copy()

    fill_defaults: dict[str, Any] = {
        "doppler": 0.0,
        "snr": 0.0,
        "local_density": 0.0,
        "persist_score": 0.0,
    }
    if all(c in df.columns for c in ["x", "y", "z"]):
        fill_defaults["range_m"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
        fill_defaults["elevation_deg"] = np.degrees(
            np.arctan2(df["z"], np.sqrt(df["x"] ** 2 + df["y"] ** 2) + 1e-6)
        )
    if all(c in df.columns for c in ["x", "y"]):
        fill_defaults["azimuth_deg"] = np.degrees(np.arctan2(df["y"], df["x"] + 1e-6))

    for col in feature_cols:
        if col not in df.columns:
            default = fill_defaults.get(col, 0.0)
            df[col] = default if isinstance(default, (pd.Series, np.ndarray)) else float(default)
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    if VIDEO_FRAME_COL not in df.columns:
        df[VIDEO_FRAME_COL] = pd.to_numeric(df.get(FRAME_COL, 0), errors="coerce").fillna(0).astype(np.int64)
    else:
        df[VIDEO_FRAME_COL] = pd.to_numeric(df[VIDEO_FRAME_COL], errors="coerce").fillna(0).astype(np.int64)

    if FRAME_COL in df.columns:
        df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce").fillna(0).astype(np.int64)

    return df


def compute_feature_stats(df_train: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = feature_cols or FEATURE_COLS
    means = df_train[feature_cols].mean().to_numpy(dtype=np.float32)
    stds  = df_train[feature_cols].std().to_numpy(dtype=np.float32)
    stds  = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return means, stds


# =============================================================================
# Soft structural features (corridor_soft_structural merged in)
# =============================================================================

@dataclass
class Plane:
    normal: np.ndarray
    offset: float
    inlier_frac: float = 0.0
    n_inliers: int = 0

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.offset

    def abs_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_distance(pts))


@dataclass
class CorridorPlanes:
    floor: Optional[Plane] = None
    walls: list[Plane] = field(default_factory=list)
    centerline_y: float = 0.0
    corridor_width: float = 3.0
    floor_z: float = -0.5
    wall_fit_max_inlier_frac: float = 0.0
    floor_fit_inlier_frac: float = 0.0


def _ransac_plane(
    pts: np.ndarray,
    n_iter: int = 200,
    dist_thresh: float = 0.06,
    min_inlier_frac: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Plane]:
    if len(pts) < 3:
        return None
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(pts)
    best_n_inliers = 0
    best_normal = None
    best_offset = None
    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n /= norm
        d = -np.dot(n, p0)
        dists = np.abs(pts @ n + d)
        n_inliers = int(np.sum(dists < dist_thresh))
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_normal = n.copy()
            best_offset = d
    if best_normal is None or best_n_inliers / N < min_inlier_frac:
        return None
    inlier_mask = np.abs(pts @ best_normal + best_offset) < dist_thresh
    inlier_pts = pts[inlier_mask]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        cov = np.cov((inlier_pts - centroid).T)
        _, eigvecs = np.linalg.eigh(cov)
        best_normal = eigvecs[:, 0].copy()
        best_offset = -np.dot(best_normal, centroid)
    return Plane(normal=best_normal, offset=best_offset,
                 inlier_frac=best_n_inliers / N, n_inliers=best_n_inliers)


def fit_corridor_planes(
    pts: np.ndarray,
    floor_dist_thresh: float = 0.06,
    floor_min_inlier_frac: float = 0.08,
    wall_dist_thresh: float = 0.12,
    wall_min_inlier_frac: float = 0.02,
    floor_normal_tol_deg: float = 30.0,
    wall_normal_tol_deg: float = 45.0,
    ransac_iters: int = 300,
    max_walls: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> CorridorPlanes:
    if rng is None:
        rng = np.random.default_rng(42)
    result = CorridorPlanes()
    if len(pts) < 10:
        return result
    up = np.array([0.0, 0.0, 1.0])
    floor_tol_cos = np.cos(np.radians(floor_normal_tol_deg))
    wall_tol_sin  = np.sin(np.radians(wall_normal_tol_deg))

    z_vals = pts[:, 2]
    floor_candidates = pts[z_vals <= np.percentile(z_vals, 65)]
    floor_plane = None
    if len(floor_candidates) >= 3:
        floor_plane = _ransac_plane(floor_candidates, n_iter=ransac_iters,
                                    dist_thresh=floor_dist_thresh,
                                    min_inlier_frac=floor_min_inlier_frac, rng=rng)
    if floor_plane is None and len(pts) >= 3:
        floor_plane = _ransac_plane(pts, n_iter=ransac_iters // 2,
                                    dist_thresh=floor_dist_thresh,
                                    min_inlier_frac=floor_min_inlier_frac * 0.5, rng=rng)
    if floor_plane is not None and abs(np.dot(floor_plane.normal, up)) >= floor_tol_cos:
        if np.dot(floor_plane.normal, up) < 0:
            floor_plane.normal = -floor_plane.normal
            floor_plane.offset = -floor_plane.offset
        result.floor = floor_plane
        floor_inlier_z = pts[floor_plane.abs_distance(pts) < floor_dist_thresh, 2]
        if len(floor_inlier_z) > 0:
            result.floor_z = float(np.median(floor_inlier_z))

    remaining_mask = np.ones(len(pts), dtype=bool)
    if result.floor is not None:
        remaining_mask = result.floor.abs_distance(pts) >= floor_dist_thresh
    for _ in range(max_walls):
        wall_candidates = pts[remaining_mask]
        if len(wall_candidates) < 3:
            break
        wall_plane = _ransac_plane(wall_candidates, n_iter=ransac_iters,
                                   dist_thresh=wall_dist_thresh,
                                   min_inlier_frac=wall_min_inlier_frac, rng=rng)
        if wall_plane is None:
            break
        if abs(np.dot(wall_plane.normal, up)) > wall_tol_sin:
            continue
        result.walls.append(wall_plane)
        remaining_mask = remaining_mask & ~(wall_plane.abs_distance(pts) < wall_dist_thresh)

    if result.floor is not None:
        result.floor_fit_inlier_frac = result.floor.inlier_frac
    if result.walls:
        result.wall_fit_max_inlier_frac = max(w.inlier_frac for w in result.walls)
    if len(result.walls) == 2:
        d0 = result.walls[0].signed_distance(np.zeros((1, 3)))[0]
        d1 = result.walls[1].signed_distance(np.zeros((1, 3)))[0]
        result.corridor_width = abs(d0 - d1)
        result.centerline_y   = (d0 + d1) / 2.0
    elif len(result.walls) == 1:
        wall_dists = result.walls[0].signed_distance(pts)
        result.corridor_width = float(np.percentile(np.abs(wall_dists), 95)) * 2
        result.centerline_y   = 0.0
    return result


def estimate_local_normals(pts: np.ndarray, k: int = 8,
                            fallback_normal: Optional[np.ndarray] = None) -> np.ndarray:
    from scipy.spatial import cKDTree
    N = len(pts)
    if fallback_normal is None:
        fallback_normal = np.array([0.0, 0.0, 1.0])
    if N < 3:
        return np.tile(fallback_normal, (N, 1))
    k_actual = min(k, N)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_actual)
    if k_actual < 3:
        return np.tile(fallback_normal, (N, 1))
    normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        neighbors = pts[idx[i]]
        centroid  = neighbors.mean(axis=0)
        cov = (neighbors - centroid).T @ (neighbors - centroid) / len(neighbors)
        try:
            _, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        except np.linalg.LinAlgError:
            normals[i] = fallback_normal
    norms = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    return (normals / norms).astype(np.float32)


def compute_soft_structural_features(
    pts: np.ndarray,
    corridor: CorridorPlanes,
    compute_normals: bool = True,
    normal_k: int = 8,
) -> dict[str, np.ndarray]:
    """Compute the 7 soft structural features used in SOFT_STRUCTURAL_COLS."""
    N = len(pts)
    features: dict[str, np.ndarray] = {}

    # Floor features
    if corridor.floor is not None:
        d_floor = corridor.floor.signed_distance(pts)
        features["z_above_floor"] = (pts[:, 2] - corridor.floor_z).astype(np.float32)
    else:
        d_floor = np.full(N, 2.0, dtype=np.float32)
        features["z_above_floor"] = pts[:, 2].astype(np.float32)

    # Wall features
    if corridor.walls:
        wall_dists  = np.column_stack([w.abs_distance(pts) for w in corridor.walls])
        wall_signed = np.column_stack([w.signed_distance(pts) for w in corridor.walls])
        nearest_wall_idx = np.argmin(wall_dists, axis=1)
        d_wall_min    = wall_dists[np.arange(N), nearest_wall_idx]
        d_wall_signed = wall_signed[np.arange(N), nearest_wall_idx]
        half_width = max(corridor.corridor_width / 2.0, 0.1)
        if len(corridor.walls) >= 2:
            lateral = pts[:, 0] - corridor.centerline_y
            features["corridor_margin"] = np.clip(lateral / half_width, -1.0, 1.0).astype(np.float32)
        else:
            features["corridor_margin"] = np.clip(d_wall_signed / half_width, -1.0, 1.0).astype(np.float32)
        sigma_lateral    = 0.3
        lateral_closeness = np.exp(-0.5 * (d_wall_min / sigma_lateral) ** 2)
        protrusion_score  = np.clip((d_wall_min * np.exp(-d_wall_min / 0.3)) / 0.12, 0.0, 1.0)
        if corridor.floor is not None:
            height_factor = np.clip((pts[:, 2] - corridor.floor_z) / 0.5, 0.0, 1.0)
        else:
            height_factor = np.ones(N, dtype=np.float32)
        features["wall_anomaly"] = (lateral_closeness * protrusion_score * height_factor).astype(np.float32)
    else:
        d_wall_min    = np.full(N, 2.0, dtype=np.float32)
        d_wall_signed = np.zeros(N, dtype=np.float32)
        features["corridor_margin"] = np.zeros(N, dtype=np.float32)
        features["wall_anomaly"]    = np.zeros(N, dtype=np.float32)

    # Angular features
    if compute_normals and N >= 3:
        normals = estimate_local_normals(pts, k=normal_k)
        if corridor.floor is not None:
            cos_floor = np.clip(np.abs(np.sum(normals * corridor.floor.normal, axis=1)), 0.0, 1.0)
            features["floor_ang"] = np.arccos(cos_floor).astype(np.float32)
        else:
            features["floor_ang"] = np.full(N, np.pi / 2, dtype=np.float32)
        if corridor.walls:
            nearest_wall_normals = np.array([corridor.walls[i].normal for i in nearest_wall_idx])
            cos_wall = np.clip(np.abs(np.sum(normals * nearest_wall_normals, axis=1)), 0.0, 1.0)
            features["wall_ang"] = np.arccos(cos_wall).astype(np.float32)
        else:
            features["wall_ang"] = np.full(N, np.pi / 2, dtype=np.float32)
    else:
        features["floor_ang"] = np.full(N, np.pi / 2, dtype=np.float32)
        features["wall_ang"]  = np.full(N, np.pi / 2, dtype=np.float32)

    # Availability flags
    features["has_floor"] = np.full(N, 1.0 if corridor.floor is not None else 0.0, dtype=np.float32)
    features["has_wall"]  = np.full(N, 1.0 if corridor.walls else 0.0, dtype=np.float32)

    return features


def process_frame_group(
    anchor_pts: np.ndarray,
    aggregated_pts: np.ndarray,
    floor_dist_thresh: float = 0.06,
    wall_dist_thresh: float = 0.08,
    ransac_iters: int = 300,
    compute_normals: bool = True,
    normal_k: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[CorridorPlanes, dict[str, np.ndarray]]:
    corridor = fit_corridor_planes(
        aggregated_pts,
        floor_dist_thresh=floor_dist_thresh,
        wall_dist_thresh=wall_dist_thresh,
        ransac_iters=ransac_iters,
        rng=rng,
    )
    features = compute_soft_structural_features(
        anchor_pts, corridor,
        compute_normals=compute_normals,
        normal_k=normal_k,
    )
    return corridor, features


# =============================================================================
# Dataset internals
# =============================================================================

@dataclass(frozen=True)
class FrameRecord:
    session: str
    video_frame_index: int
    radar_frame_num: int
    point_indices: np.ndarray


@dataclass(frozen=True)
class TemporalWindowRecord:
    session: str
    center_index: int
    window_indices: tuple[int, ...]


def build_frame_records(df: pd.DataFrame) -> list[FrameRecord]:
    required = {SESSION_COL, FRAME_COL, LABEL_COL, "x", "y", "z", VIDEO_FRAME_COL}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    records: list[FrameRecord] = []
    for (session, video_frame_index, frame_num), idx in df.groupby(
        [SESSION_COL, VIDEO_FRAME_COL, FRAME_COL], sort=True
    ).indices.items():
        point_indices = np.asarray(idx, dtype=np.int64)
        if len(point_indices) == 0:
            continue
        records.append(FrameRecord(
            session=str(session),
            video_frame_index=int(video_frame_index),
            radar_frame_num=int(frame_num),
            point_indices=point_indices,
        ))
    return records


def build_temporal_windows(frame_records: list[FrameRecord], window_size: int) -> list[TemporalWindowRecord]:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd positive integer.")
    session_to_indices: dict[str, list[int]] = {}
    for idx, rec in enumerate(frame_records):
        session_to_indices.setdefault(rec.session, []).append(idx)
    windows: list[TemporalWindowRecord] = []
    half = window_size // 2
    for session, idxs in session_to_indices.items():
        idxs = sorted(idxs, key=lambda i: (
            int(frame_records[i].video_frame_index),
            int(frame_records[i].radar_frame_num),
        ))
        n = len(idxs)
        for pos, center_idx in enumerate(idxs):
            window_local = [idxs[min(max(pos + rel, 0), n - 1)] for rel in range(-half, half + 1)]
            windows.append(TemporalWindowRecord(
                session=session,
                center_index=center_idx,
                window_indices=tuple(window_local),
            ))
    return windows


# =============================================================================
# Dataset classes
# =============================================================================

class _BaseTemporalRadarDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_means: np.ndarray,
                 feature_stds: np.ndarray, window_size: int = 3,
                 feature_cols: list[str] | None = None):
        self.feature_cols  = feature_cols or FEATURE_COLS
        self.df            = ensure_feature_columns(df, self.feature_cols)
        self.feature_means = feature_means.astype(np.float32)
        self.feature_stds  = feature_stds.astype(np.float32)
        self.window_size   = int(window_size)
        self.frames        = build_frame_records(self.df)
        self.windows       = build_temporal_windows(self.frames, self.window_size)
        self.X             = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.xyz           = self.df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        self.y             = self.df[LABEL_COL].map(BUCKET_TO_IDX).to_numpy(dtype=np.int64)
        self.doppler_idx   = self.feature_cols.index(DOPPLER_COL)

        # Load patch arrays if present in dataset
        rd_cols = [f"rd_{i}" for i in range(RD_PATCH_SIZE)]
        ra_cols = [f"ra_{i}" for i in range(RA_PATCH_SIZE)]
        self.has_patches = all(c in self.df.columns for c in rd_cols[:1] + ra_cols[:1])
        if self.has_patches:
            self.RD = self.df[rd_cols].to_numpy(dtype=np.float32)
            self.RA = self.df[ra_cols].to_numpy(dtype=np.float32)
        else:
            self.RD = None
            self.RA = None

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        denom = float(np.max(np.abs(vec)))
        if denom < 1e-6:
            denom = 1.0
        return (vec / denom).astype(np.float32)

    def _build_frame_meta(self, frame_recs: list[FrameRecord],
                          doppler_values: list[np.ndarray]) -> np.ndarray:
        center   = frame_recs[self.window_size // 2]
        dt_video = np.array([rec.video_frame_index - center.video_frame_index for rec in frame_recs], dtype=np.float32)
        dt_radar = np.array([rec.radar_frame_num  - center.radar_frame_num   for rec in frame_recs], dtype=np.float32)
        mean_d     = np.array([float(np.mean(d))        if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        abs_mean_d = np.array([float(np.mean(np.abs(d))) if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        std_d      = np.array([float(np.std(d))         if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        meta = np.stack([
            self._normalize_vector(dt_video),
            self._normalize_vector(dt_radar),
            self._normalize_vector(mean_d),
            self._normalize_vector(abs_mean_d),
            self._normalize_vector(std_d),
        ], axis=1)
        return meta.astype(np.float32)


class TemporalRadarTrainDataset(_BaseTemporalRadarDataset):
    def __init__(self, df: pd.DataFrame, feature_means: np.ndarray,
                 feature_stds: np.ndarray, window_size: int = 3,
                 n_points: int = 64, feature_cols: list[str] | None = None, seed: int = 42):
        super().__init__(df, feature_means, feature_stds, window_size=window_size, feature_cols=feature_cols)
        self.n_points = int(n_points)
        self.rng      = np.random.default_rng(seed)

    def _sample_indices(self, point_indices: np.ndarray) -> np.ndarray:
        n = len(point_indices)
        if n >= self.n_points:
            return self.rng.choice(point_indices, size=self.n_points, replace=False)
        return self.rng.choice(point_indices, size=self.n_points, replace=True)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        win          = self.windows[idx]
        feat_frames  = []
        xyz_frames   = []
        doppler_vals = []
        frame_recs   = []
        center_labels = None
        center_slot   = self.window_size // 2
        # Preserve the exact sampled point ids per temporal slot so scalar
        # features, labels, RD patches, and RA patches stay aligned.
        chosen_per_slot = []
        for slot, frame_idx in enumerate(win.window_indices):
            rec    = self.frames[frame_idx]
            chosen = self._sample_indices(rec.point_indices)
            chosen_per_slot.append(chosen)
            raw    = self.X[chosen]
            feats  = np.nan_to_num((raw - self.feature_means) / (self.feature_stds + 1e-6),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(feats.T.copy())
            xyz_frames.append(np.nan_to_num(self.xyz[chosen], nan=0.0, posinf=0.0, neginf=0.0).T.copy())
            doppler_vals.append(raw[:, self.doppler_idx].copy())
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = self.y[chosen].copy()
        if center_labels is None:
            raise RuntimeError("Center labels were not populated.")
        frame_meta = self._build_frame_meta(frame_recs, doppler_vals)

        if self.has_patches:
            rd_frames = []
            ra_frames = []
            for chosen in chosen_per_slot:
                rd_frames.append(self.RD[chosen].copy())  # (N, RD_PATCH_SIZE)
                ra_frames.append(self.RA[chosen].copy())  # (N, RA_PATCH_SIZE)
            return (
                torch.from_numpy(np.stack(feat_frames, axis=0)),
                torch.from_numpy(np.stack(xyz_frames, axis=0)),
                torch.from_numpy(center_labels),
                torch.from_numpy(frame_meta),
                torch.from_numpy(np.stack(rd_frames, axis=0)),  # (T, N, RD_PATCH_SIZE)
                torch.from_numpy(np.stack(ra_frames, axis=0)),  # (T, N, RA_PATCH_SIZE)
            )

        return (
            torch.from_numpy(np.stack(feat_frames, axis=0)),
            torch.from_numpy(np.stack(xyz_frames, axis=0)),
            torch.from_numpy(center_labels),
            torch.from_numpy(frame_meta),
        )


class TemporalRadarEvalDataset(_BaseTemporalRadarDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        win          = self.windows[idx]
        center_slot  = self.window_size // 2
        feat_frames  = []
        xyz_frames   = []
        doppler_vals = []
        frame_recs   = []
        center_labels = None
        center_rec    = None
        for slot, frame_idx in enumerate(win.window_indices):
            rec      = self.frames[frame_idx]
            pidx     = rec.point_indices
            raw      = self.X[pidx]
            feats    = np.nan_to_num((raw - self.feature_means) / (self.feature_stds + 1e-6),
                                     nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(torch.from_numpy(feats.T.copy()))
            xyz_frames.append(torch.from_numpy(
                np.nan_to_num(self.xyz[pidx], nan=0.0, posinf=0.0, neginf=0.0).T.copy()))
            doppler_vals.append(raw[:, self.doppler_idx].copy())
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = torch.from_numpy(self.y[pidx].copy())
                center_rec    = rec
        if center_labels is None or center_rec is None:
            raise RuntimeError("Center frame metadata was not populated.")
        frame_meta = self._build_frame_meta(frame_recs, doppler_vals)

        result = {
            "window_features":   feat_frames,
            "window_xyz":        xyz_frames,
            "frame_meta":        torch.from_numpy(frame_meta),
            "center_labels":     center_labels,
            "session":           center_rec.session,
            "video_frame_index": int(center_rec.video_frame_index),
            "radar_frame_num":   int(center_rec.radar_frame_num),
            "n_points":          int(len(center_rec.point_indices)),
        }

        if self.has_patches:
            rd_frames = []
            ra_frames = []
            for frame_idx in win.window_indices:
                rec  = self.frames[frame_idx]
                pidx = rec.point_indices
                rd_frames.append(torch.from_numpy(self.RD[pidx].copy()))
                ra_frames.append(torch.from_numpy(self.RA[pidx].copy()))
            result["window_rd_patches"] = rd_frames  # list of (N_t, RD_PATCH_SIZE)
            result["window_ra_patches"] = ra_frames  # list of (N_t, RA_PATCH_SIZE)

        return result


def temporal_eval_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("TemporalRadarEvalDataset requires batch_size=1.")
    return batch[0]



# =============================================================================
# Patch encoders — change 3
# =============================================================================

class _SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention. Negligible FLOPs, meaningful gain."""
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = x.mean(dim=(2, 3))          # (B, C) global avg
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class _PatchEncoderBase(nn.Module):
    """
    Depthwise-separable CNN with SE attention for 2D radar patches.
    Jetson-efficient: ~8× fewer FLOPs than an equivalent standard CNN.

    Architecture per stage:
        depthwise Conv 3×3 → pointwise Conv 1×1 → BN → GELU
    Two stages, followed by SE block and adaptive avg pooling.

    Input:  (B*N, 1, H, W)   — single-channel 2D patch
    Output: (B,   N, embed_dim)
    """
    def __init__(self, patch_h: int, patch_w: int,
                 embed_dim: int = PATCH_EMBED_DIM) -> None:
        super().__init__()
        self.patch_h   = patch_h
        self.patch_w   = patch_w
        self.embed_dim = embed_dim

        # Stage 1: 1 → 16
        self.dw1 = nn.Conv2d(1,  1,  kernel_size=3, padding=1, groups=1,  bias=False)
        self.pw1 = nn.Conv2d(1,  16, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 2: 16 → 32
        self.dw2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.pw2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.act  = nn.GELU()
        self.se   = _SEBlock(32, reduction=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, N, patch_size)  — flattened patches
        returns: (B, N, embed_dim)
        """
        B, N, P = patches.shape
        # Reshape to image format for conv layers
        x = patches.reshape(B * N, 1, self.patch_h, self.patch_w)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Stage 1
        x = self.act(self.bn1(self.pw1(self.dw1(x))))
        # Stage 2
        x = self.act(self.bn2(self.pw2(self.dw2(x))))
        # SE + pool + project
        x = self.se(x)
        x = self.pool(x).view(B * N, 32)
        x = self.proj(x)                # (B*N, embed_dim)
        return x.view(B, N, self.embed_dim)


class RDPatchEncoder(_PatchEncoderBase):
    """
    Encoder for Range-Doppler patches.
    Input patch shape: (RD_PATCH_DOPPLER, RD_PATCH_RANGE) = (17, 7)
    The Doppler axis is the row axis — tall dimension — to preserve
    the micro-Doppler spread structure for conv spatial reasoning.
    """
    def __init__(self, embed_dim: int = PATCH_EMBED_DIM) -> None:
        super().__init__(patch_h=RD_PATCH_DOPPLER, patch_w=RD_PATCH_RANGE,
                         embed_dim=embed_dim)


class RAPatchEncoder(_PatchEncoderBase):
    """
    Encoder for Range-Azimuth patches.
    Input patch shape: (RA_PATCH_AZIMUTH, RA_PATCH_RANGE) = (9, 7)
    """
    def __init__(self, embed_dim: int = PATCH_EMBED_DIM) -> None:
        super().__init__(patch_h=RA_PATCH_AZIMUTH, patch_w=RA_PATCH_RANGE,
                         embed_dim=embed_dim)


def extract_rd_patches(
    rd_cube: "np.ndarray",
    doppler_bins: "np.ndarray",
    range_bins: "np.ndarray",
) -> "np.ndarray":
    """
    Extract and log-normalise RD patches from the full rd_cube.

    Parameters
    ----------
    rd_cube     : (num_tx, loops_per_frame, num_rx, adc_samples) complex64
    doppler_bins: (N,) int  — CFAR detection doppler bin indices
    range_bins  : (N,) int  — CFAR detection range bin indices

    Returns
    -------
    patches : (N, RD_PATCH_SIZE) float32
    """
    import numpy as np
    # Sum power across TX and RX to get (Doppler, Range) power map
    power = np.sum(np.abs(rd_cube) ** 2, axis=(0, 2)).astype(np.float32)
    return _extract_patches_from_map(
        power, doppler_bins, range_bins,
        half_h=RD_PATCH_DOPPLER // 2, half_w=RD_PATCH_RANGE // 2,
        out_h=RD_PATCH_DOPPLER, out_w=RD_PATCH_RANGE,
    )


def extract_ra_patches(
    power_map: "np.ndarray",
    doppler_bins: "np.ndarray",
    range_bins: "np.ndarray",
) -> "np.ndarray":
    """
    Extract and log-normalise RA patches from the summed power map.

    Parameters
    ----------
    power_map   : (loops_per_frame, adc_samples) float32  — summed power (RA proxy)
    doppler_bins: (N,) int  — used as azimuth-axis index in the power map
    range_bins  : (N,) int

    Returns
    -------
    patches : (N, RA_PATCH_SIZE) float32
    """
    return _extract_patches_from_map(
        power_map, doppler_bins, range_bins,
        half_h=RA_PATCH_AZIMUTH // 2, half_w=RA_PATCH_RANGE // 2,
        out_h=RA_PATCH_AZIMUTH, out_w=RA_PATCH_RANGE,
    )


def _extract_patches_from_map(
    power_map: "np.ndarray",
    row_bins: "np.ndarray",
    col_bins: "np.ndarray",
    half_h: int,
    half_w: int,
    out_h: int,
    out_w: int,
) -> "np.ndarray":
    """Shared patch extraction with zero-padding and log-normalisation."""
    import numpy as np
    N = len(row_bins)
    patches = np.zeros((N, out_h * out_w), dtype=np.float32)

    log_map = np.log1p(power_map.astype(np.float32))
    max_val = log_map.max()
    if max_val > 1e-6:
        log_map /= max_val

    n_rows, n_cols = log_map.shape

    for i in range(N):
        rb = int(row_bins[i])
        cb = int(col_bins[i])

        r0s = max(0, rb - half_h);  r1s = min(n_rows, rb + half_h + 1)
        c0s = max(0, cb - half_w);  c1s = min(n_cols, cb + half_w + 1)

        r0d = r0s - (rb - half_h)
        c0d = c0s - (cb - half_w)

        patch = np.zeros((out_h, out_w), dtype=np.float32)
        patch[r0d:r0d + (r1s - r0s), c0d:c0d + (c1s - c0s)] = (
            log_map[r0s:r1s, c0s:c1s]
        )
        patches[i] = patch.reshape(-1)

    return patches


# =============================================================================
# Temporal segmenter base
# =============================================================================

class TemporalSegmenterBase(nn.Module):
    """
    Shared temporal transformer + segmentation head.
    Subclass sets self.frame_encoder (KPConvFrameEncoder).
    """
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES,
                 window_size: int = 3, k: int = 20, emb_dims: int = 256,
                 temporal_layers: int = 2, temporal_heads: int = 4, dropout: float = 0.25,
                 frame_meta_dim: int = N_FRAME_META, gate_hidden: int = 128):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd positive integer.")
        self.window_size   = int(window_size)
        self.center_index  = self.window_size // 2
        self.n_features    = int(n_features)
        self.n_classes     = int(n_classes)
        self.k             = int(k)
        self.emb_dims      = int(emb_dims)
        self.frame_meta_dim = int(frame_meta_dim)

        # frame_encoder is set by the subclass
        self.frame_encoder: nn.Module

        self.pos_embedding = nn.Parameter(torch.zeros(1, self.window_size, emb_dims))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.meta_proj = nn.Sequential(
            nn.Linear(frame_meta_dim, emb_dims), nn.GELU(), nn.Linear(emb_dims, emb_dims),
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(emb_dims * 3 + frame_meta_dim * 2, gate_hidden),
            nn.GELU(), nn.Linear(gate_hidden, emb_dims), nn.Sigmoid(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dims, nhead=temporal_heads, dim_feedforward=emb_dims * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.temporal_attn = nn.Sequential(
            nn.Linear(emb_dims + frame_meta_dim, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1),
        )
        local_dim = 64 + 64 + 128
        head_in   = local_dim + emb_dims * 3
        self.head = nn.Sequential(
            nn.Conv1d(head_in, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False),     nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Conv1d(128, n_classes, 1, bias=True),
        )

    def _coerce_temporal_inputs(self, window_features, window_xyz, frame_meta):
        if frame_meta is None:
            raise ValueError("frame_meta is required.")
        if isinstance(window_features, torch.Tensor):
            feat_list = [window_features[:, t] for t in range(window_features.size(1))]
            xyz_list  = [window_xyz[:, t] for t in range(window_xyz.size(1))]
            return feat_list, xyz_list, frame_meta
        if frame_meta.ndim == 2:
            frame_meta = frame_meta.unsqueeze(0)
        return window_features, window_xyz, frame_meta

    def forward(self, window_features, window_xyz, frame_meta) -> torch.Tensor:
        feat_list, xyz_list, frame_meta = self._coerce_temporal_inputs(window_features, window_xyz, frame_meta)
        frame_meta = torch.nan_to_num(frame_meta, nan=0.0, posinf=0.0, neginf=0.0)

        local_list, global_list = [], []
        for feats_t, xyz_t in zip(feat_list, xyz_list):
            local_t, global_t = self.frame_encoder(feats_t, xyz_t)
            local_list.append(local_t)
            global_list.append(global_t)

        global_seq          = torch.stack(global_list, dim=1)
        center_token        = global_seq[:, self.center_index]
        center_meta         = frame_meta[:, self.center_index]
        center_token_expand = center_token.unsqueeze(1).expand(-1, global_seq.size(1), -1)
        center_meta_expand  = center_meta.unsqueeze(1).expand(-1, frame_meta.size(1), -1)

        gate_in = torch.cat([
            global_seq, center_token_expand,
            torch.abs(global_seq - center_token_expand),
            frame_meta, torch.abs(frame_meta - center_meta_expand),
        ], dim=-1)
        gate        = self.temporal_gate(gate_in)
        meta_embed  = self.meta_proj(frame_meta)
        gated_seq   = global_seq * gate + meta_embed + self.pos_embedding[:, :global_seq.size(1)]
        temporal_seq = self.temporal_encoder(gated_seq)

        attn_weights   = torch.softmax(
            self.temporal_attn(torch.cat([temporal_seq, frame_meta], dim=-1)).squeeze(-1), dim=1
        )
        pooled_context = torch.sum(temporal_seq * attn_weights.unsqueeze(-1), dim=1)
        temporal_center = temporal_seq[:, self.center_index]

        center_local  = local_list[self.center_index]
        N_pts = center_local.size(-1)
        logits = self.head(torch.cat([
            center_local,
            global_list[self.center_index].unsqueeze(-1).expand(-1, -1, N_pts),
            temporal_center.unsqueeze(-1).expand(-1, -1, N_pts),
            pooled_context.unsqueeze(-1).expand(-1, -1, N_pts),
        ], dim=1))
        return logits


# =============================================================================
# KPConv frame encoder (imported from radar_kpconv_frame_encoder)
# =============================================================================

def _get_kpconv_encoder(n_features: int, emb_dims: int, k: int,
                        radius_1: float, radius_2: float, radius_3: float,
                        sigma_factor: float, n_kernel_points: int):
    """Load KPConvFrameEncoder from 3branch_frame_encoder.py via importlib."""
    import importlib.util
    _HERE = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "branch_frame_encoder",
        _HERE / "3branch_frame_encoder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KPConvFrameEncoder(
        n_features=n_features, emb_dims=emb_dims, k=k,
        radius_1=radius_1, radius_2=radius_2, radius_3=radius_3,
        sigma_factor=sigma_factor, n_kernel_points=n_kernel_points,
    )


# =============================================================================
# KPConvTemporalSegmenter — Branch 1 primary model
# =============================================================================

class KPConvTemporalSegmenter(TemporalSegmenterBase):
    """
    Branch 1 primary model — KPConv frame encoder with temporal transformer.

    Change 3: accepts optional rd_patches and ra_patches per temporal frame.
    When provided, each patch is encoded by RDPatchEncoder / RAPatchEncoder
    (depthwise-separable CNN + SE attention → 32-dim embedding) and
    concatenated with the scalar features before KPConv processes them.

    Feature widths:
        Without patches : n_features (18) entering KPConv
        With patches    : n_features + 64 (82) entering KPConv
    The KPConvFrameEncoder is constructed with the augmented width automatically.
    """
    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES,
                 window_size: int = 3, k: int = 20, emb_dims: int = 256,
                 temporal_layers: int = 2, temporal_heads: int = 4, dropout: float = 0.25,
                 frame_meta_dim: int = N_FRAME_META, gate_hidden: int = 128,
                 radius_1: float = 0.35, radius_2: float = 0.50, radius_3: float = 0.70,
                 sigma_factor: float = 2.5, n_kernel_points: int = 15,
                 use_patches: bool = True):
        # KPConv encoder sees augmented feature width when patches enabled
        kpconv_n_features = (n_features + PATCH_EMBED_DIM * 2) if use_patches else n_features
        super().__init__(n_features=kpconv_n_features, n_classes=n_classes,
                         window_size=window_size, k=k, emb_dims=emb_dims,
                         temporal_layers=temporal_layers, temporal_heads=temporal_heads,
                         dropout=dropout, frame_meta_dim=frame_meta_dim,
                         gate_hidden=gate_hidden)
        self.n_scalar_features = int(n_features)
        self.use_patches       = bool(use_patches)
        self.radius_1          = float(radius_1)
        self.radius_2          = float(radius_2)
        self.radius_3          = float(radius_3)
        self.sigma_factor      = float(sigma_factor)
        self.n_kernel_points   = int(n_kernel_points)

        self.frame_encoder = _get_kpconv_encoder(
            n_features=kpconv_n_features, emb_dims=emb_dims, k=k,
            radius_1=radius_1, radius_2=radius_2, radius_3=radius_3,
            sigma_factor=sigma_factor, n_kernel_points=n_kernel_points,
        )

        if use_patches:
            self.rd_encoder = RDPatchEncoder(embed_dim=PATCH_EMBED_DIM)
            self.ra_encoder = RAPatchEncoder(embed_dim=PATCH_EMBED_DIM)

    def log_density(self, xyz: torch.Tensor) -> None:
        self.frame_encoder.log_density(xyz)

    def forward(
        self,
        window_features,
        window_xyz,
        frame_meta,
        window_rd_patches=None,
        window_ra_patches=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        window_features     : list of (B, n_scalar_features, N) tensors
        window_xyz          : list of (B, 3, N) tensors
        frame_meta          : (B, T, frame_meta_dim) tensor
        window_rd_patches   : optional list of (B, N, RD_PATCH_SIZE) tensors
        window_ra_patches   : optional list of (B, N, RA_PATCH_SIZE) tensors

        If patches are None or use_patches=False, falls back to scalar-only
        forward (backward compatible with pre-change-3 checkpoints).
        """
        if self.use_patches and window_rd_patches is not None:
            augmented = []
            for t, feats_t in enumerate(window_features):
                rd_emb = self.rd_encoder(window_rd_patches[t])  # (B, N, 32)
                ra_emb = self.ra_encoder(window_ra_patches[t])  # (B, N, 32)
                # feats_t: (B, n_scalar, N) — permute to (B, N, n_scalar) for cat
                feats_BNF = feats_t.permute(0, 2, 1)
                combined  = torch.cat([feats_BNF, rd_emb, ra_emb], dim=2)  # (B, N, 82)
                augmented.append(combined.permute(0, 2, 1))  # back to (B, 82, N)
            window_features = augmented

        # Call base forward (TemporalSegmenterBase) with (possibly augmented) features
        return super().forward(window_features, window_xyz, frame_meta)


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_kpconv_checkpoint(path: Path, epoch: int, model: KPConvTemporalSegmenter,
                           feature_cols: list[str], feature_means: np.ndarray,
                           feature_stds: np.ndarray, extra: dict | None = None) -> None:
    payload = {
        "encoder_type":    "kpconv",
        "label_space":     "3class",
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "feature_cols":    feature_cols,
        "feature_means":   feature_means.tolist(),
        "feature_stds":    feature_stds.tolist(),
        "bucket_order":    BUCKET_ORDER,
        "window_size":     model.window_size,
        "emb_dims":        model.emb_dims,
        "temporal_layers": model.temporal_layers if hasattr(model, "temporal_layers") else 2,
        "temporal_heads":  model.temporal_heads  if hasattr(model, "temporal_heads")  else 4,
        "dropout":         model.dropout         if hasattr(model, "dropout")         else 0.0,
        "frame_meta_dim":  model.frame_meta_dim,
        "gate_hidden":     model.gate_hidden     if hasattr(model, "gate_hidden")     else 128,
        "radius_1":        model.radius_1,
        "radius_2":        model.radius_2,
        "radius_3":        model.radius_3,
        "sigma_factor":    model.sigma_factor,
        "n_kernel_points": model.n_kernel_points,
        # Change 3 patch encoder params
        "use_patches":     model.use_patches,
        "n_scalar_features": model.n_scalar_features,
        "rd_patch_size":   RD_PATCH_SIZE,
        "ra_patch_size":   RA_PATCH_SIZE,
        "patch_embed_dim": PATCH_EMBED_DIM,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_kpconv_checkpoint(ckpt_path: str | Path,
                           map_location: str | torch.device = "cpu") -> tuple[dict, KPConvTemporalSegmenter]:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if ckpt.get("encoder_type", "dgcnn") != "kpconv":
        raise ValueError(f"Expected encoder_type='kpconv', got '{ckpt.get('encoder_type')}'.")
    feature_cols = ckpt.get("feature_cols", FEATURE_COLS)
    bucket_order = ckpt.get("bucket_order", BUCKET_ORDER)
    model = KPConvTemporalSegmenter(
        n_features=len(feature_cols), n_classes=len(bucket_order),
        window_size=int(ckpt.get("window_size", 3)),
        k=int(ckpt.get("k", 20)), emb_dims=int(ckpt.get("emb_dims", 256)),
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
