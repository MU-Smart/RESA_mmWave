"""Generate Branch 3 polar BEV labels from processed session folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


MAX_RANGE_M = 12.0
AZ_RANGE_DEG = 45.0
N_RANGE_BINS = 128
N_AZ_BINS = 128
TEMPORAL_HALF_WIN = 2
MIN_Z_M = -0.851
MIN_SNR = 5.0
OCCUPANCY_THRESH = 1

LABEL_REMAP = {
    "wall": "structure",
    "door": "structure",
    "pillar": "structure",
    "box_like": "structure",
}
STRUCTURE_LABELS = {"structure", "wall", "door", "pillar", "box_like"}
LABEL_CSV_NAMES = (
    "labeled_radar_points_v4_fused.csv",
    "labeled_radar_points_v4.csv",
)


def build_polar_edges() -> tuple[np.ndarray, np.ndarray]:
    range_edges = np.linspace(0.0, MAX_RANGE_M, N_RANGE_BINS + 1)
    az_edges = np.linspace(-np.radians(AZ_RANGE_DEG), np.radians(AZ_RANGE_DEG), N_AZ_BINS + 1)
    return range_edges, az_edges


def xyz_to_polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    range_m = np.sqrt(x**2 + y**2)
    az_rad = np.arctan2(x, np.maximum(y, 1e-6))
    return range_m, az_rad


def points_to_occupancy(
    x: np.ndarray,
    y: np.ndarray,
    range_edges: np.ndarray,
    az_edges: np.ndarray,
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((N_RANGE_BINS, N_AZ_BINS), dtype=np.int32)
    range_m, az_rad = xyz_to_polar(x, y)
    r_idx = np.searchsorted(range_edges[1:-1], range_m).astype(np.int32)
    a_idx = np.searchsorted(az_edges[1:-1], az_rad).astype(np.int32)
    valid = (
        (r_idx >= 0)
        & (r_idx < N_RANGE_BINS)
        & (a_idx >= 0)
        & (a_idx < N_AZ_BINS)
        & (range_m <= MAX_RANGE_M)
        & (np.abs(az_rad) <= np.radians(AZ_RANGE_DEG))
    )
    grid = np.zeros((N_RANGE_BINS, N_AZ_BINS), dtype=np.int32)
    np.add.at(grid, (r_idx[valid], a_idx[valid]), 1)
    return grid


def occupancy_to_label(occ: np.ndarray) -> np.ndarray:
    label = np.ones((N_RANGE_BINS, N_AZ_BINS), dtype=np.int8)
    for az in range(N_AZ_BINS):
        hits = np.where(occ[:, az] >= OCCUPANCY_THRESH)[0]
        if len(hits) > 0:
            label[hits[0] :, az] = 0
    return label


def load_session_csv(session_dir: Path) -> pd.DataFrame | None:
    path = next((session_dir / name for name in LABEL_CSV_NAMES if (session_dir / name).exists()), None)
    if path is None:
        return None
    df = pd.read_csv(path, low_memory=False)
    label_col = "bucket" if "bucket" in df.columns else "bucket_4class"
    required = {"x", "y", "z", label_col, "radar_frame_num"}
    if not required.issubset(df.columns):
        return None
    df["bucket"] = df[label_col].replace(LABEL_REMAP)
    df = df[df["z"] > MIN_Z_M].reset_index(drop=True)
    if "snr" in df.columns:
        df = df[df["snr"] >= MIN_SNR].reset_index(drop=True)
    return df


def get_structure_xy(
    df: pd.DataFrame,
    center_frame: int,
    half_win: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = df["radar_frame_num"].to_numpy()
    mask = (
        (frames >= center_frame - half_win)
        & (frames <= center_frame + half_win)
        & df["bucket"].isin(STRUCTURE_LABELS)
    )
    sub = df[mask]
    if len(sub) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return sub["x"].to_numpy(dtype=np.float32), sub["y"].to_numpy(dtype=np.float32)


def generate_session_labels(session_dir: Path, force: bool = False) -> int:
    label_dir = session_dir / "bev_labels"
    meta_path = session_dir / "bev_label_meta.json"

    df = load_session_csv(session_dir)
    if df is None:
        return 0

    unique_frames = sorted(int(x) for x in df["radar_frame_num"].dropna().unique().tolist())
    if not unique_frames:
        return 0

    if not force and label_dir.exists() and meta_path.exists():
        if len(list(label_dir.glob("*.npy"))) >= len(unique_frames):
            return 0

    label_dir.mkdir(parents=True, exist_ok=True)
    range_edges, az_edges = build_polar_edges()
    n_written = 0
    n_low_cov = 0

    for frame_num in unique_frames:
        out_path = label_dir / f"{frame_num:06d}.npy"
        if not force and out_path.exists():
            n_written += 1
            continue
        x, y = get_structure_xy(df, frame_num, TEMPORAL_HALF_WIN)
        occ = points_to_occupancy(x, y, range_edges, az_edges)
        label = occupancy_to_label(occ)
        if int(np.sum(occ.max(axis=0) >= OCCUPANCY_THRESH)) < 5:
            n_low_cov += 1
        np.save(out_path, label)
        n_written += 1

    meta = {
        "session": session_dir.name,
        "n_frames": len(unique_frames),
        "n_written": n_written,
        "n_low_coverage": n_low_cov,
        "n_range_bins": N_RANGE_BINS,
        "n_az_bins": N_AZ_BINS,
        "max_range_m": MAX_RANGE_M,
        "az_range_deg": AZ_RANGE_DEG,
        "temporal_half_win": TEMPORAL_HALF_WIN,
        "frame_ids": unique_frames,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate polar BEV labels for Branch 3.")
    parser.add_argument("--session_dir", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.session_dir is None and args.root is None:
        parser.error("Provide --session_dir or --root")

    if args.session_dir:
        sessions = [Path(args.session_dir)]
    else:
        sessions = sorted(p for p in Path(args.root).iterdir() if p.is_dir() and p.name.startswith("session_"))

    print("=" * 70)
    print("branch3_label_bev.py  [Polar BEV label generation]")
    print(f"  Grid     : {N_RANGE_BINS}x{N_AZ_BINS}  (range x azimuth)")
    print(f"  Range    : 0-{MAX_RANGE_M}m   Azimuth: +/-{AZ_RANGE_DEG} deg")
    print(f"  Temporal : +/-{TEMPORAL_HALF_WIN} frames")
    print(f"  Sessions : {len(sessions)}")
    print("=" * 70)

    total = 0
    for session_dir in tqdm(sessions, desc="Sessions"):
        count = generate_session_labels(session_dir, force=args.force)
        total += count
        if count > 0:
            tqdm.write(f"  {session_dir.name}: {count} labels written")

    print(f"\n[DONE] {total} label files written across {len(sessions)} sessions.")


if __name__ == "__main__":
    main()
