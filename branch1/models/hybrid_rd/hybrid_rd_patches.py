"""Utilities for hybrid point-cloud + local range-Doppler patch features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


DEFAULT_RD_PATCH_DOPPLER_BINS = 17
DEFAULT_RD_PATCH_RANGE_BINS = 7
RD_FRAME_FILENAME_TEMPLATE = "rd_frame_{frame_index:06d}.npz"


@dataclass(frozen=True)
class RDPatchConfig:
    """Shape and source settings for point-aligned RD patches."""

    doppler_bins: int = DEFAULT_RD_PATCH_DOPPLER_BINS
    range_bins: int = DEFAULT_RD_PATCH_RANGE_BINS
    channel_name: str = "rd_power"

    def __post_init__(self) -> None:
        if self.doppler_bins < 1 or self.doppler_bins % 2 == 0:
            raise ValueError("doppler_bins must be an odd positive integer.")
        if self.range_bins < 1 or self.range_bins % 2 == 0:
            raise ValueError("range_bins must be an odd positive integer.")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.doppler_bins), int(self.range_bins))


@dataclass(frozen=True)
class RDPatchStats:
    """Patch extraction accounting for a batch of point rows."""

    n_rows: int
    n_valid: int
    n_missing_frame: int
    n_missing_bin: int
    n_failed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "n_rows": int(self.n_rows),
            "n_valid": int(self.n_valid),
            "n_missing_frame": int(self.n_missing_frame),
            "n_missing_bin": int(self.n_missing_bin),
            "n_failed": int(self.n_failed),
        }


def rd_frame_path(
    sidecar_root: str | Path,
    session_name: str,
    frame_index: int,
    *,
    sidecar_subdir: str = "hybrid_rd",
) -> Path:
    """Return the standard RD sidecar frame path for a session/frame."""

    return (
        Path(sidecar_root)
        / str(session_name)
        / sidecar_subdir
        / "frames"
        / RD_FRAME_FILENAME_TEMPLATE.format(frame_index=int(frame_index))
    )


def load_rd_sidecar(npz_path: str | Path, *, channel_name: str = "rd_power") -> dict[str, np.ndarray]:
    """Load one RD sidecar frame from disk."""

    path = Path(npz_path)
    with np.load(path) as payload:
        if channel_name not in payload:
            raise KeyError(f"RD sidecar {path} does not contain '{channel_name}'.")
        out = {
            channel_name: np.asarray(payload[channel_name], dtype=np.float32),
            "range_axis_m": np.asarray(payload["range_axis_m"], dtype=np.float32),
            "doppler_axis_mps": np.asarray(payload["doppler_axis_mps"], dtype=np.float32),
            "frame_index": np.asarray(payload["frame_index"]).astype(np.int64),
            "timestamp_ms": np.asarray(payload["timestamp_ms"]).astype(np.int64),
        }
    return out


def extract_rd_patch(
    rd_power: np.ndarray,
    *,
    doppler_bin: int,
    range_bin: int,
    config: RDPatchConfig | None = None,
) -> np.ndarray:
    """Extract a centered RD patch.

    Doppler indices wrap because they live on the FFT-shifted Doppler axis.
    Range indices are zero-padded because range is not periodic.
    """

    cfg = config or RDPatchConfig()
    rd = np.asarray(rd_power, dtype=np.float32)
    if rd.ndim != 2:
        raise ValueError(f"rd_power must have shape [doppler, range], got {rd.shape}")

    n_doppler, n_range = rd.shape
    d_center = int(doppler_bin)
    r_center = int(range_bin)
    if not (0 <= d_center < n_doppler):
        raise IndexError(f"doppler_bin {d_center} outside [0, {n_doppler})")
    if not (0 <= r_center < n_range):
        raise IndexError(f"range_bin {r_center} outside [0, {n_range})")

    d_half = cfg.doppler_bins // 2
    r_half = cfg.range_bins // 2
    d_indices = (np.arange(d_center - d_half, d_center + d_half + 1) % n_doppler).astype(np.int64)
    r_indices = np.arange(r_center - r_half, r_center + r_half + 1, dtype=np.int64)

    patch = np.zeros(cfg.shape, dtype=np.float32)
    valid_r = (r_indices >= 0) & (r_indices < n_range)
    if np.any(valid_r):
        patch[:, valid_r] = rd[np.ix_(d_indices, r_indices[valid_r])]
    return patch


def extract_patches_for_bins(
    rd_power: np.ndarray,
    doppler_bins: Sequence[int],
    range_bins: Sequence[int],
    *,
    config: RDPatchConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract patches for parallel Doppler/range bin sequences."""

    cfg = config or RDPatchConfig()
    if len(doppler_bins) != len(range_bins):
        raise ValueError("doppler_bins and range_bins must have the same length.")

    patches = np.zeros((len(doppler_bins), 1, cfg.doppler_bins, cfg.range_bins), dtype=np.float32)
    valid = np.zeros(len(doppler_bins), dtype=bool)
    for i, (d_bin, r_bin) in enumerate(zip(doppler_bins, range_bins)):
        try:
            patches[i, 0] = extract_rd_patch(
                rd_power,
                doppler_bin=int(d_bin),
                range_bin=int(r_bin),
                config=cfg,
            )
            valid[i] = True
        except Exception:
            valid[i] = False
    return patches, valid


def resolve_frame_index(frame_num: int, *, frame_number_offset: int = 0) -> int:
    """Map labeled CSV frame numbers to exported sidecar frame indices."""

    return int(frame_num) - int(frame_number_offset)


def require_patch_columns(columns: Sequence[str]) -> None:
    """Validate that a point table has the columns needed for RD patch joins."""

    required = {"range_bin", "doppler_bin"}
    has_session = bool({"session", "session_name", "session_id"} & set(columns))
    has_frame = bool({"radar_frame_num", "frame_num", "frame"} & set(columns))
    missing = sorted(required - set(columns))
    if not has_session:
        missing.append("session|session_name|session_id")
    if not has_frame:
        missing.append("radar_frame_num|frame_num|frame")
    if missing:
        raise ValueError(
            "Point CSV is missing RD patch provenance columns: "
            + ", ".join(missing)
            + ". Re-run or update adc_to_pointcloud_v6.py with richer detection export."
        )


def frame_column(columns: Sequence[str]) -> str:
    for candidate in ("radar_frame_num", "frame_num", "frame"):
        if candidate in columns:
            return candidate
    raise ValueError("Could not find frame column in point table.")


def session_column(columns: Sequence[str]) -> str:
    for candidate in ("session", "session_name", "session_id"):
        if candidate in columns:
            return candidate
    raise ValueError("Could not find session column in point table.")


def patch_manifest_payload(
    *,
    patch_config: RDPatchConfig,
    sidecar_root: str | Path | Sequence[str | Path],
    sidecar_subdir: str | Sequence[str],
    frame_number_offset: int,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(sidecar_root, (list, tuple)):
        sidecar_root_payload = [str(Path(p).resolve()) for p in sidecar_root]
    else:
        sidecar_root_payload = str(Path(sidecar_root).resolve())

    if isinstance(sidecar_subdir, (list, tuple)):
        sidecar_subdir_payload: str | list[str] = [str(s) for s in sidecar_subdir]
    else:
        sidecar_subdir_payload = str(sidecar_subdir)

    payload: dict[str, object] = {
        "patch_source": patch_config.channel_name,
        "patch_shape": [1, patch_config.doppler_bins, patch_config.range_bins],
        "patch_axes": ["channel", "doppler", "range"],
        "sidecar_root": sidecar_root_payload,
        "sidecar_subdir": sidecar_subdir_payload,
        "frame_number_offset": int(frame_number_offset),
        "doppler_edge_mode": "wrap",
        "range_edge_mode": "zero_pad",
    }
    if extra:
        payload.update(dict(extra))
    return payload
