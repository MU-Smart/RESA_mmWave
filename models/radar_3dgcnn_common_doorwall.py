from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

if hasattr(torch.backends, "mkldnn"):
    torch.backends.mkldnn.enabled = False
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

RAW_BUCKET_ORDER = ["wall", "floor", "door", "pillar", "human", "box_like"]
LABEL_REMAP = {
    "door": "wall",
}
BUCKET_ORDER = ["wall", "floor", "pillar", "human", "box_like"]
BUCKET_TO_IDX = {b: i for i, b in enumerate(BUCKET_ORDER)}
IDX_TO_BUCKET = {i: b for b, i in BUCKET_TO_IDX.items()}

FEATURE_COLS = [
    "x", "y", "z",
    "range_m", "azimuth_deg", "elevation_deg",
    "doppler",
    "snr",
    "local_density",
    "persist_score",
]
FRAME_META_NAMES = [
    "dt_video_norm",
    "dt_radar_norm",
    "mean_doppler_norm",
    "abs_mean_doppler_norm",
    "std_doppler_norm",
]

N_FEATURES = len(FEATURE_COLS)
N_FRAME_META = len(FRAME_META_NAMES)
N_CLASSES = len(BUCKET_ORDER)
SESSION_COL = "session"
SPLIT_COL = "split"
FRAME_COL = "radar_frame_num"
VIDEO_FRAME_COL = "video_frame_index"
LABEL_COL = "bucket"
DOPPLER_COL = "doppler"


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


def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    feature_cols = feature_cols or FEATURE_COLS
    df = df.copy()

    fill_defaults = {
        "doppler": 0.0,
        "snr": 0.0,
        "local_density": 0.0,
        "persist_score": 0.0,
        "range_m": np.sqrt(df.get("x", 0.0) ** 2 + df.get("y", 0.0) ** 2 + df.get("z", 0.0) ** 2)
        if all(c in df.columns for c in ["x", "y", "z"]) else 0.0,
        "azimuth_deg": np.degrees(np.arctan2(df.get("y", 0.0), df.get("x", 1e-6)))
        if all(c in df.columns for c in ["x", "y"]) else 0.0,
        "elevation_deg": np.degrees(
            np.arctan2(
                df.get("z", 0.0),
                np.sqrt(df.get("x", 0.0) ** 2 + df.get("y", 0.0) ** 2) + 1e-6,
            )
        ) if all(c in df.columns for c in ["x", "y", "z"]) else 0.0,
    }

    for col in feature_cols:
        if col not in df.columns:
            default = fill_defaults.get(col, 0.0)
            if isinstance(default, (pd.Series, np.ndarray)):
                df[col] = default
            else:
                df[col] = float(default)
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    if VIDEO_FRAME_COL not in df.columns:
        df[VIDEO_FRAME_COL] = pd.to_numeric(df.get(FRAME_COL, 0), errors="coerce").fillna(0).astype(np.int64)
    else:
        df[VIDEO_FRAME_COL] = pd.to_numeric(df[VIDEO_FRAME_COL], errors="coerce").fillna(0).astype(np.int64)

    if FRAME_COL in df.columns:
        df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce").fillna(0).astype(np.int64)

    return df


def apply_label_remap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].replace(LABEL_REMAP)
    return df


def filter_known_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_label_remap(df)
    return df[df[LABEL_COL].isin(BUCKET_ORDER)].reset_index(drop=True)


def compute_feature_stats(df_train: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = feature_cols or FEATURE_COLS
    means = df_train[feature_cols].mean().to_numpy(dtype=np.float32)
    stds = df_train[feature_cols].std().to_numpy(dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return means, stds


def build_frame_records(df: pd.DataFrame) -> list[FrameRecord]:
    required = {SESSION_COL, FRAME_COL, LABEL_COL, "x", "y", "z", VIDEO_FRAME_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    records: list[FrameRecord] = []
    grouped = df.groupby([SESSION_COL, VIDEO_FRAME_COL, FRAME_COL], sort=True).indices
    for (session, video_frame_index, frame_num), idx in grouped.items():
        point_indices = np.asarray(idx, dtype=np.int64)
        if len(point_indices) == 0:
            continue
        records.append(
            FrameRecord(
                session=str(session),
                video_frame_index=int(video_frame_index),
                radar_frame_num=int(frame_num),
                point_indices=point_indices,
            )
        )
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
        idxs = sorted(
            idxs,
            key=lambda i: (
                int(frame_records[i].video_frame_index),
                int(frame_records[i].radar_frame_num),
            ),
        )
        n = len(idxs)
        for pos, center_idx in enumerate(idxs):
            window_local: list[int] = []
            for rel in range(-half, half + 1):
                clamped = min(max(pos + rel, 0), n - 1)
                window_local.append(idxs[clamped])
            windows.append(
                TemporalWindowRecord(
                    session=session,
                    center_index=center_idx,
                    window_indices=tuple(window_local),
                )
            )
    return windows


class _BaseTemporalRadarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        window_size: int = 3,
        feature_cols: list[str] | None = None,
    ):
        self.df = ensure_feature_columns(df, feature_cols)
        self.feature_cols = feature_cols or FEATURE_COLS
        self.feature_means = feature_means.astype(np.float32)
        self.feature_stds = feature_stds.astype(np.float32)
        self.window_size = int(window_size)

        self.frames = build_frame_records(self.df)
        self.windows = build_temporal_windows(self.frames, self.window_size)

        self.X = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.xyz = self.df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        self.y = self.df[LABEL_COL].map(BUCKET_TO_IDX).to_numpy(dtype=np.int64)
        self.doppler_idx = self.feature_cols.index(DOPPLER_COL)

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        denom = float(np.max(np.abs(vec)))
        if denom < 1e-6:
            denom = 1.0
        return (vec / denom).astype(np.float32)

    def _build_frame_meta(self, frame_recs: list[FrameRecord], doppler_values: list[np.ndarray]) -> np.ndarray:
        center = frame_recs[self.window_size // 2]
        dt_video = np.array([rec.video_frame_index - center.video_frame_index for rec in frame_recs], dtype=np.float32)
        dt_radar = np.array([rec.radar_frame_num - center.radar_frame_num for rec in frame_recs], dtype=np.float32)
        mean_d = np.array([float(np.mean(d)) if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        abs_mean_d = np.array([float(np.mean(np.abs(d))) if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        std_d = np.array([float(np.std(d)) if len(d) else 0.0 for d in doppler_values], dtype=np.float32)
        meta = np.stack(
            [
                self._normalize_vector(dt_video),
                self._normalize_vector(dt_radar),
                self._normalize_vector(mean_d),
                self._normalize_vector(abs_mean_d),
                self._normalize_vector(std_d),
            ],
            axis=1,
        )
        return meta.astype(np.float32)


class TemporalRadarTrainDataset(_BaseTemporalRadarDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        window_size: int = 3,
        n_points: int = 64,
        feature_cols: list[str] | None = None,
        seed: int = 42,
    ):
        super().__init__(df, feature_means, feature_stds, window_size=window_size, feature_cols=feature_cols)
        self.n_points = int(n_points)
        self.rng = np.random.default_rng(seed)

    def _sample_indices(self, point_indices: np.ndarray) -> np.ndarray:
        n = len(point_indices)
        if n >= self.n_points:
            return self.rng.choice(point_indices, size=self.n_points, replace=False)
        return self.rng.choice(point_indices, size=self.n_points, replace=True)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        win = self.windows[idx]
        feat_frames: list[np.ndarray] = []
        xyz_frames: list[np.ndarray] = []
        doppler_values: list[np.ndarray] = []
        frame_recs: list[FrameRecord] = []
        center_labels: np.ndarray | None = None

        center_slot = self.window_size // 2
        for slot, frame_idx in enumerate(win.window_indices):
            rec = self.frames[frame_idx]
            chosen = self._sample_indices(rec.point_indices)
            raw_feats = self.X[chosen]
            feats = (raw_feats - self.feature_means) / (self.feature_stds + 1e-6)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(feats.T.copy())
            xyz_frames.append(np.nan_to_num(self.xyz[chosen], nan=0.0, posinf=0.0, neginf=0.0).T.copy())
            doppler_values.append(raw_feats[:, self.doppler_idx].copy())
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = self.y[chosen].copy()

        if center_labels is None:
            raise RuntimeError("Center labels were not populated.")

        frame_meta = self._build_frame_meta(frame_recs, doppler_values)
        return (
            torch.from_numpy(np.stack(feat_frames, axis=0)),
            torch.from_numpy(np.stack(xyz_frames, axis=0)),
            torch.from_numpy(center_labels),
            torch.from_numpy(frame_meta),
        )


class TemporalRadarEvalDataset(_BaseTemporalRadarDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        window_size: int = 3,
        feature_cols: list[str] | None = None,
    ):
        super().__init__(df, feature_means, feature_stds, window_size=window_size, feature_cols=feature_cols)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        win = self.windows[idx]
        center_slot = self.window_size // 2

        feat_frames: list[torch.Tensor] = []
        xyz_frames: list[torch.Tensor] = []
        doppler_values: list[np.ndarray] = []
        frame_recs: list[FrameRecord] = []
        center_labels: torch.Tensor | None = None
        center_rec: FrameRecord | None = None

        for slot, frame_idx in enumerate(win.window_indices):
            rec = self.frames[frame_idx]
            pidx = rec.point_indices
            raw_feats = self.X[pidx]
            feats = (raw_feats - self.feature_means) / (self.feature_stds + 1e-6)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(torch.from_numpy(feats.T.copy()))
            xyz_frames.append(torch.from_numpy(np.nan_to_num(self.xyz[pidx], nan=0.0, posinf=0.0, neginf=0.0).T.copy()))
            doppler_values.append(raw_feats[:, self.doppler_idx].copy())
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = torch.from_numpy(self.y[pidx].copy())
                center_rec = rec

        if center_labels is None or center_rec is None:
            raise RuntimeError("Center frame metadata was not populated.")

        frame_meta = self._build_frame_meta(frame_recs, doppler_values)
        return {
            "window_features": feat_frames,
            "window_xyz": xyz_frames,
            "frame_meta": torch.from_numpy(frame_meta),
            "center_labels": center_labels,
            "session": center_rec.session,
            "video_frame_index": int(center_rec.video_frame_index),
            "radar_frame_num": int(center_rec.radar_frame_num),
            "n_points": int(len(center_rec.point_indices)),
        }


def temporal_eval_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("TemporalRadarEvalDataset is intended for batch_size=1.")
    return batch[0]


class DGCNNFrameEncoder(nn.Module):
    def __init__(self, n_features: int, k: int, emb_dims: int):
        super().__init__()
        self.k = int(k)
        self.n_features = int(n_features)
        self.emb_dims = int(emb_dims)

        self.edge1 = nn.Sequential(
            nn.Conv2d(2 * n_features, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge2 = nn.Sequential(
            nn.Conv2d(2 * 64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.edge3 = nn.Sequential(
            nn.Conv2d(2 * 64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def knn(x: torch.Tensor, k: int) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        xt = x.transpose(2, 1).contiguous()
        n = int(xt.size(1))
        k = max(1, min(int(k), n))
        dists = torch.cdist(xt.float(), xt.float(), p=2)
        idx = torch.topk(dists, k=k, dim=-1, largest=False)[1]
        return idx

    @staticmethod
    def get_graph_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        bsz, channels, n_points = x.shape
        k = idx.size(-1)
        device = x.device

        idx_base = torch.arange(bsz, device=device).view(-1, 1, 1) * n_points
        idx = (idx + idx_base).reshape(-1)

        xt = x.transpose(2, 1).contiguous()
        xt = torch.nan_to_num(xt, nan=0.0, posinf=0.0, neginf=0.0)
        neighbors = xt.reshape(bsz * n_points, channels)[idx, :].reshape(bsz, n_points, k, channels)
        centers = xt.unsqueeze(2).expand(-1, -1, k, -1)
        edge = torch.cat((neighbors - centers, centers), dim=3)
        return edge.permute(0, 3, 1, 2).contiguous()

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)

        idx1 = self.knn(xyz, self.k)
        x1 = self.edge1(self.get_graph_feature(features, idx1)).max(dim=-1)[0]

        idx2 = self.knn(x1, self.k)
        x2 = self.edge2(self.get_graph_feature(x1, idx2)).max(dim=-1)[0]

        idx3 = self.knn(x2, self.k)
        x3 = self.edge3(self.get_graph_feature(x2, idx3)).max(dim=-1)[0]

        local = torch.cat([x1, x2, x3], dim=1)
        global_feat = self.fuse(local).max(dim=2)[0]
        return local, global_feat


class DopplerAwareTemporalDGCNNSegmenter(nn.Module):
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
    ):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd positive integer.")

        self.window_size = int(window_size)
        self.center_index = self.window_size // 2
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.k = int(k)
        self.emb_dims = int(emb_dims)
        self.temporal_layers = int(temporal_layers)
        self.temporal_heads = int(temporal_heads)
        self.dropout = float(dropout)
        self.frame_meta_dim = int(frame_meta_dim)
        self.gate_hidden = int(gate_hidden)

        self.frame_encoder = DGCNNFrameEncoder(n_features=n_features, k=k, emb_dims=emb_dims)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.window_size, emb_dims))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.meta_proj = nn.Sequential(
            nn.Linear(frame_meta_dim, emb_dims),
            nn.GELU(),
            nn.Linear(emb_dims, emb_dims),
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(emb_dims * 3 + frame_meta_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, emb_dims),
            nn.Sigmoid(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dims,
            nhead=temporal_heads,
            dim_feedforward=emb_dims * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        self.temporal_attn = nn.Sequential(
            nn.Linear(emb_dims + frame_meta_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        local_dim = 64 + 64 + 128
        head_in = local_dim + emb_dims * 3
        self.head = nn.Sequential(
            nn.Conv1d(head_in, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, n_classes, kernel_size=1, bias=True),
        )

    def _coerce_temporal_inputs(
        self,
        window_features: torch.Tensor | list[torch.Tensor],
        window_xyz: torch.Tensor | list[torch.Tensor],
        frame_meta: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        if frame_meta is None:
            raise ValueError("frame_meta is required for Doppler-aware temporal aggregation.")

        if isinstance(window_features, torch.Tensor):
            if window_features.ndim != 4:
                raise ValueError("Tensor window_features must have shape (B, T, F, N)")
            if not isinstance(window_xyz, torch.Tensor) or window_xyz.ndim != 4:
                raise ValueError("Tensor window_xyz must have shape (B, T, 3, N)")
            if frame_meta.ndim != 3:
                raise ValueError("frame_meta tensor must have shape (B, T, M)")
            if window_features.size(1) != self.window_size or window_xyz.size(1) != self.window_size or frame_meta.size(1) != self.window_size:
                raise ValueError("Temporal tensor inputs do not match model window_size.")
            feat_list = [window_features[:, t, :, :] for t in range(window_features.size(1))]
            xyz_list = [window_xyz[:, t, :, :] for t in range(window_xyz.size(1))]
            return feat_list, xyz_list, frame_meta

        if not isinstance(window_features, list) or not isinstance(window_xyz, list):
            raise ValueError("Temporal inputs must both be tensors or both be lists.")
        if len(window_features) != self.window_size or len(window_xyz) != self.window_size:
            raise ValueError("Temporal list inputs do not match model window_size.")
        if frame_meta.ndim == 2:
            frame_meta = frame_meta.unsqueeze(0)
        if frame_meta.ndim != 3 or frame_meta.size(1) != self.window_size:
            raise ValueError("frame_meta tensor must have shape (B, T, M) for list inputs.")
        return window_features, window_xyz, frame_meta

    def forward(
        self,
        window_features: torch.Tensor | list[torch.Tensor],
        window_xyz: torch.Tensor | list[torch.Tensor],
        frame_meta: torch.Tensor | None,
    ) -> torch.Tensor:
        feat_list, xyz_list, frame_meta = self._coerce_temporal_inputs(window_features, window_xyz, frame_meta)
        frame_meta = torch.nan_to_num(frame_meta, nan=0.0, posinf=0.0, neginf=0.0)

        local_list: list[torch.Tensor] = []
        global_list: list[torch.Tensor] = []
        batch_size: int | None = None

        for feats_t, xyz_t in zip(feat_list, xyz_list):
            if feats_t.ndim != 3 or xyz_t.ndim != 3:
                raise ValueError("Every temporal slice must have shape (B, C, N)")
            if batch_size is None:
                batch_size = int(feats_t.size(0))
            elif int(feats_t.size(0)) != batch_size:
                raise ValueError("All temporal slices must share the same batch size.")
            local_t, global_t = self.frame_encoder(feats_t, xyz_t)
            local_list.append(local_t)
            global_list.append(global_t)

        global_seq = torch.stack(global_list, dim=1)
        center_token = global_seq[:, self.center_index, :]
        center_meta = frame_meta[:, self.center_index, :]
        center_token_expand = center_token.unsqueeze(1).expand(-1, global_seq.size(1), -1)
        center_meta_expand = center_meta.unsqueeze(1).expand(-1, frame_meta.size(1), -1)

        gate_in = torch.cat(
            [
                global_seq,
                center_token_expand,
                torch.abs(global_seq - center_token_expand),
                frame_meta,
                torch.abs(frame_meta - center_meta_expand),
            ],
            dim=-1,
        )
        gate = self.temporal_gate(gate_in)
        meta_embed = self.meta_proj(frame_meta)
        gated_seq = global_seq * gate + meta_embed + self.pos_embedding[:, : global_seq.size(1), :]
        temporal_seq = self.temporal_encoder(gated_seq)

        attn_logits = self.temporal_attn(torch.cat([temporal_seq, frame_meta], dim=-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled_context = torch.sum(temporal_seq * attn_weights.unsqueeze(-1), dim=1)
        temporal_center = temporal_seq[:, self.center_index, :]

        center_local = local_list[self.center_index]
        center_global = global_list[self.center_index].unsqueeze(-1).expand(-1, -1, center_local.size(-1))
        temporal_center_rep = temporal_center.unsqueeze(-1).expand(-1, -1, center_local.size(-1))
        pooled_context_rep = pooled_context.unsqueeze(-1).expand(-1, -1, center_local.size(-1))
        logits = self.head(torch.cat([center_local, center_global, temporal_center_rep, pooled_context_rep], dim=1))
        return logits


def load_checkpoint(ckpt_path: str | Path, map_location: str | torch.device = "cpu") -> tuple[dict[str, Any], DopplerAwareTemporalDGCNNSegmenter]:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model = DopplerAwareTemporalDGCNNSegmenter(
        n_features=len(ckpt.get("feature_cols", FEATURE_COLS)),
        n_classes=len(ckpt.get("bucket_order", BUCKET_ORDER)),
        window_size=int(ckpt.get("window_size", 3)),
        k=int(ckpt.get("k", 20)),
        emb_dims=int(ckpt.get("emb_dims", 256)),
        temporal_layers=int(ckpt.get("temporal_layers", 2)),
        temporal_heads=int(ckpt.get("temporal_heads", 4)),
        dropout=float(ckpt.get("dropout", 0.25)),
        frame_meta_dim=int(ckpt.get("frame_meta_dim", N_FRAME_META)),
        gate_hidden=int(ckpt.get("gate_hidden", 128)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt, model
