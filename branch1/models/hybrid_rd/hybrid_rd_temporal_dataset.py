"""Temporal point-cloud datasets that include per-point RD and optional RA patches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FRAME_COL = "radar_frame_num"
VIDEO_FRAME_COL = "video_frame_index"
SESSION_COL = "session"


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


class RDPatchShardStore:
    """Lazy loader for paired RD/RA patch shard files produced by the patch builder."""

    def __init__(self, dataset_root: str | Path, df: pd.DataFrame, *, use_ra_patches: bool = False) -> None:
        self.dataset_root = Path(dataset_root)
        if "rd_patch_shard" not in df.columns or "rd_patch_index" not in df.columns:
            raise ValueError("Dataset CSV must contain rd_patch_shard and rd_patch_index.")
        self.shards = df["rd_patch_shard"].astype(str).to_numpy()
        self.indices = pd.to_numeric(df["rd_patch_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        self.use_ra_patches = bool(use_ra_patches)
        if self.use_ra_patches and ("ra_patch_shard" not in df.columns or "ra_patch_index" not in df.columns):
            raise ValueError("Dataset CSV must contain ra_patch_shard and ra_patch_index when use_ra_patches=True.")
        self.ra_shards = (
            df["ra_patch_shard"].astype(str).to_numpy()
            if self.use_ra_patches else self.shards
        )
        self.ra_indices = (
            pd.to_numeric(df["ra_patch_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
            if self.use_ra_patches else self.indices
        )
        self.cache: dict[str, dict[str, np.ndarray]] = {}

    def _load_shard(self, rel_path: str) -> dict[str, np.ndarray]:
        if rel_path not in self.cache:
            path = self.dataset_root / rel_path
            if not path.exists():
                raise FileNotFoundError(f"Missing patch shard: {path}")
            with np.load(path) as payload:
                shard: dict[str, np.ndarray] = {
                    "rd_patches": np.asarray(payload["rd_patches"], dtype=np.float32),
                }
                if "ra_patches" in payload:
                    shard["ra_patches"] = np.asarray(payload["ra_patches"], dtype=np.float32)
                self.cache[rel_path] = shard
        return self.cache[rel_path]

    def get_many(self, row_indices: Sequence[int], *, patch_kind: str = "rd") -> np.ndarray:
        if patch_kind not in {"rd", "ra"}:
            raise ValueError("patch_kind must be 'rd' or 'ra'.")
        patches: list[np.ndarray] = []
        shards = self.ra_shards if patch_kind == "ra" else self.shards
        indices = self.ra_indices if patch_kind == "ra" else self.indices
        for row_idx in row_indices:
            rel = shards[int(row_idx)]
            patch_idx = int(indices[int(row_idx)])
            if not rel or patch_idx < 0:
                raise ValueError(f"Row {row_idx} does not have a valid patch index.")
            shard = self._load_shard(rel)
            key = f"{patch_kind}_patches"
            if key not in shard:
                if patch_kind == "ra" and not self.use_ra_patches:
                    raise ValueError("RA patches requested from a store that was not configured for RA.")
                raise ValueError(f"Patch shard {rel} is missing '{key}'.")
            patches.append(np.asarray(shard[key][patch_idx], dtype=np.float32))
        return np.stack(patches, axis=0).astype(np.float32)


def ensure_feature_columns(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
    return out


def filter_known_labels(df: pd.DataFrame, *, label_col: str, bucket_order: Sequence[str]) -> pd.DataFrame:
    return df[df[label_col].isin(list(bucket_order))].reset_index(drop=True)


def build_frame_records(df: pd.DataFrame, *, label_col: str) -> list[FrameRecord]:
    required = {SESSION_COL, FRAME_COL, label_col, "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if VIDEO_FRAME_COL not in df.columns:
        df = df.copy()
        df[VIDEO_FRAME_COL] = df[FRAME_COL]

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
            window_local = []
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


class _BaseHybridRDTemporalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        dataset_root: str | Path,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        feature_cols: Sequence[str],
        bucket_order: Sequence[str],
        label_col: str = "bucket",
        window_size: int = 3,
        sample_weight_col: str | None = None,
        acc_label_col: str | None = None,
        use_ra_patches: bool = False,
    ) -> None:
        df = df.copy()
        if "rd_patch_valid" in df.columns:
            df = df[df["rd_patch_valid"].astype(bool)].reset_index(drop=True)
        self.use_ra_patches = bool(use_ra_patches)
        if "ra_patch_valid" in df.columns:
            df = df[df["ra_patch_valid"].astype(bool)].reset_index(drop=True)
        self.df = ensure_feature_columns(df, feature_cols)
        self.feature_cols = list(feature_cols)
        self.feature_means = np.asarray(feature_means, dtype=np.float32)
        self.feature_stds = np.asarray(feature_stds, dtype=np.float32)
        self.feature_stds = np.where(self.feature_stds < 1e-6, 1.0, self.feature_stds).astype(np.float32)
        self.bucket_order = list(bucket_order)
        self.bucket_to_idx = {bucket: i for i, bucket in enumerate(self.bucket_order)}
        self.label_col = label_col
        self.window_size = int(window_size)
        self.acc_label_col = acc_label_col
        if self.use_ra_patches and ("ra_patch_shard" not in self.df.columns or "ra_patch_index" not in self.df.columns):
            raise ValueError("Dataset missing RA patch columns required for use_ra_patches=True.")

        if label_col not in self.df.columns:
            raise ValueError(f"Dataset missing label column '{label_col}'.")
        self.df = filter_known_labels(self.df, label_col=label_col, bucket_order=self.bucket_order)
        self.patch_store = RDPatchShardStore(dataset_root, self.df, use_ra_patches=self.use_ra_patches)
        self.frames = build_frame_records(self.df, label_col=label_col)
        self.windows = build_temporal_windows(self.frames, self.window_size)

        self.X = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        self.xyz = self.df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        self.y = self.df[self.label_col].map(self.bucket_to_idx).to_numpy(dtype=np.int64)
        self.doppler_idx = self.feature_cols.index("doppler") if "doppler" in self.feature_cols else None
        self.acc_targets: np.ndarray | None = None
        self.acc_mask: np.ndarray | None = None
        if acc_label_col is not None:
            if acc_label_col in self.df.columns:
                labels = self.df[acc_label_col].astype(str).str.strip().str.lower()
                self.acc_targets = labels.map({"yes": 1.0, "no": 0.0}).fillna(0.0).to_numpy(dtype=np.float32)
                self.acc_mask = labels.isin(["yes", "no"]).astype(np.float32).to_numpy()
            else:
                self.acc_targets = np.zeros(len(self.df), dtype=np.float32)
                self.acc_mask = np.zeros(len(self.df), dtype=np.float32)
        self.sample_weights: np.ndarray | None = None
        if sample_weight_col is not None:
            if sample_weight_col not in self.df.columns:
                raise ValueError(f"sample_weight_col '{sample_weight_col}' not found in dataset columns.")
            self.sample_weights = (
                pd.to_numeric(self.df[sample_weight_col], errors="coerce")
                .fillna(1.0)
                .clip(lower=0.0, upper=1.0)
                .astype(np.float32)
                .to_numpy()
            )

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        denom = float(np.max(np.abs(vec))) if vec.size else 0.0
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
        return np.stack(
            [
                self._normalize_vector(dt_video),
                self._normalize_vector(dt_radar),
                self._normalize_vector(mean_d),
                self._normalize_vector(abs_mean_d),
                self._normalize_vector(std_d),
            ],
            axis=1,
        ).astype(np.float32)

    def _doppler_values(self, raw_feats: np.ndarray) -> np.ndarray:
        if self.doppler_idx is None:
            return np.zeros(raw_feats.shape[0], dtype=np.float32)
        return raw_feats[:, self.doppler_idx].copy()


class HybridRDTemporalTrainDataset(_BaseHybridRDTemporalDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        dataset_root: str | Path,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        feature_cols: Sequence[str],
        bucket_order: Sequence[str],
        label_col: str = "bucket",
        window_size: int = 3,
        n_points: int = 64,
        seed: int = 42,
        sample_weight_col: str | None = None,
        acc_label_col: str | None = None,
        use_ra_patches: bool = False,
    ) -> None:
        super().__init__(
            df,
            dataset_root=dataset_root,
            feature_means=feature_means,
            feature_stds=feature_stds,
            feature_cols=feature_cols,
            bucket_order=bucket_order,
            label_col=label_col,
            window_size=window_size,
            sample_weight_col=sample_weight_col,
            acc_label_col=acc_label_col,
            use_ra_patches=use_ra_patches,
        )
        self.n_points = int(n_points)
        self.rng = np.random.default_rng(seed)

    def _sample_indices(self, point_indices: np.ndarray) -> np.ndarray:
        n = len(point_indices)
        if n >= self.n_points:
            return self.rng.choice(point_indices, size=self.n_points, replace=False)
        return self.rng.choice(point_indices, size=self.n_points, replace=True)

    def __getitem__(self, idx: int):
        win = self.windows[idx]
        feat_frames: list[np.ndarray] = []
        xyz_frames: list[np.ndarray] = []
        rd_patch_frames: list[np.ndarray] = []
        ra_patch_frames: list[np.ndarray] = []
        doppler_values: list[np.ndarray] = []
        frame_recs: list[FrameRecord] = []
        center_labels: np.ndarray | None = None
        center_weights: np.ndarray | None = None
        center_acc_targets: np.ndarray | None = None
        center_acc_mask: np.ndarray | None = None

        center_slot = self.window_size // 2
        for slot, frame_idx in enumerate(win.window_indices):
            rec = self.frames[frame_idx]
            chosen = self._sample_indices(rec.point_indices)
            raw_feats = self.X[chosen]
            feats = (raw_feats - self.feature_means) / (self.feature_stds + 1e-6)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(feats.T.copy())
            xyz_frames.append(np.nan_to_num(self.xyz[chosen], nan=0.0, posinf=0.0, neginf=0.0).T.copy())
            rd_patch_frames.append(self.patch_store.get_many(chosen, patch_kind="rd"))
            if self.use_ra_patches:
                ra_patch_frames.append(self.patch_store.get_many(chosen, patch_kind="ra"))
            doppler_values.append(self._doppler_values(raw_feats))
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = self.y[chosen].copy()
                center_weights = (
                    self.sample_weights[chosen].copy() if self.sample_weights is not None
                    else np.ones(len(chosen), dtype=np.float32)
                )
                if self.acc_targets is not None and self.acc_mask is not None:
                    center_acc_targets = self.acc_targets[chosen].copy()
                    center_acc_mask = self.acc_mask[chosen].copy()

        if center_labels is None or center_weights is None:
            raise RuntimeError("Center labels were not populated.")

        window_features = torch.from_numpy(np.stack(feat_frames, axis=0))
        window_xyz = torch.from_numpy(np.stack(xyz_frames, axis=0))
        center_labels_t = torch.from_numpy(center_labels)
        frame_meta = torch.from_numpy(self._build_frame_meta(frame_recs, doppler_values))
        rd_patches = torch.from_numpy(np.stack(rd_patch_frames, axis=0))
        ra_patches = torch.from_numpy(np.stack(ra_patch_frames, axis=0)) if self.use_ra_patches else None
        sample_weights = torch.from_numpy(center_weights)
        if self.acc_label_col is None:
            if self.use_ra_patches:
                return (
                    window_features,
                    window_xyz,
                    center_labels_t,
                    frame_meta,
                    rd_patches,
                    ra_patches,
                    sample_weights,
                )
            return (
                window_features,
                window_xyz,
                center_labels_t,
                frame_meta,
                rd_patches,
                sample_weights,
            )
        if center_acc_targets is None or center_acc_mask is None:
            raise RuntimeError("Center accumulatability targets were not populated.")
        if self.use_ra_patches:
            return (
                window_features,
                window_xyz,
                center_labels_t,
                frame_meta,
                rd_patches,
                ra_patches,
                sample_weights,
                torch.from_numpy(center_acc_targets),
                torch.from_numpy(center_acc_mask),
            )
        return (
            window_features,
            window_xyz,
            center_labels_t,
            frame_meta,
            rd_patches,
            sample_weights,
            torch.from_numpy(center_acc_targets),
            torch.from_numpy(center_acc_mask),
        )


class HybridRDTemporalEvalDataset(_BaseHybridRDTemporalDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        win = self.windows[idx]
        center_slot = self.window_size // 2
        feat_frames: list[torch.Tensor] = []
        xyz_frames: list[torch.Tensor] = []
        rd_patch_frames: list[torch.Tensor] = []
        ra_patch_frames: list[torch.Tensor] = []
        doppler_values: list[np.ndarray] = []
        frame_recs: list[FrameRecord] = []
        center_labels: torch.Tensor | None = None
        center_acc_targets: torch.Tensor | None = None
        center_acc_mask: torch.Tensor | None = None
        center_rec: FrameRecord | None = None

        for slot, frame_idx in enumerate(win.window_indices):
            rec = self.frames[frame_idx]
            pidx = rec.point_indices
            raw_feats = self.X[pidx]
            feats = (raw_feats - self.feature_means) / (self.feature_stds + 1e-6)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feat_frames.append(torch.from_numpy(feats.T.copy()))
            xyz_frames.append(torch.from_numpy(np.nan_to_num(self.xyz[pidx], nan=0.0, posinf=0.0, neginf=0.0).T.copy()))
            rd_patch_frames.append(torch.from_numpy(self.patch_store.get_many(pidx, patch_kind="rd")))
            if self.use_ra_patches:
                ra_patch_frames.append(torch.from_numpy(self.patch_store.get_many(pidx, patch_kind="ra")))
            doppler_values.append(self._doppler_values(raw_feats))
            frame_recs.append(rec)
            if slot == center_slot:
                center_labels = torch.from_numpy(self.y[pidx].copy())
                if self.acc_targets is not None and self.acc_mask is not None:
                    center_acc_targets = torch.from_numpy(self.acc_targets[pidx].copy())
                    center_acc_mask = torch.from_numpy(self.acc_mask[pidx].copy())
                center_rec = rec

        if center_labels is None or center_rec is None:
            raise RuntimeError("Center frame metadata was not populated.")

        out = {
            "window_features": feat_frames,
            "window_xyz": xyz_frames,
            "window_rd_patches": rd_patch_frames,
            "frame_meta": torch.from_numpy(self._build_frame_meta(frame_recs, doppler_values)),
            "center_labels": center_labels,
            "session": center_rec.session,
            "video_frame_index": int(center_rec.video_frame_index),
            "radar_frame_num": int(center_rec.radar_frame_num),
            "n_points": int(len(center_rec.point_indices)),
        }
        if self.use_ra_patches:
            out["window_ra_patches"] = ra_patch_frames
        if self.acc_label_col is not None:
            if center_acc_targets is None or center_acc_mask is None:
                raise RuntimeError("Center accumulatability targets were not populated.")
            out["acc_targets"] = center_acc_targets
            out["acc_mask"] = center_acc_mask
        return out


def hybrid_rd_eval_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("HybridRDTemporalEvalDataset is intended for batch_size=1.")
    return batch[0]
