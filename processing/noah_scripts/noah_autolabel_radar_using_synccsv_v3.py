"""
noah_autolabel_radar_using_synccsv_v3.py

Stage 2 of the auto-labeling pipeline.

For each synchronized (video frame, radar frame) pair:
  1. Loads the ADE20K label map saved by Stage 1 (seg/<frame_index:06d>.npy).
  2. Projects each radar point into the image using calibrated R, t, K.
  3. Reads the majority label in a small pixel neighborhood around the projection.
  4. Maps the ADE20K label to one of the navigation-relevant buckets:
       wall, floor, door, pillar, human, box_like
  5. Applies quality filters:
       - Ignores overhead / non-navigable classes (ceiling, lights, pipes, etc.)
       - Discards projections where the neighborhood majority is < min_majority
         (projection sits on a label boundary — unreliable)
       - For non-human classes, discards projections in the upper ROI of the
         image (likely hitting ceiling geometry reflected on glass, lights, etc.)
  6. Writes labeled_radar_points_v3.csv with one row per labeled radar point.

Filter progression across versions:
  v1 — no filtering
  v2 — ROI top-fraction filter applied to ALL categories
  v3 — ROI top-fraction filter applied only to non-human categories  ← this file

Usage:
    python noah_autolabel_radar_using_synccsv_v3.py
        --processing_session_dir "C:\\...\\Processing\\session_..."
        --sync_csv               "C:\\...\\Synchronized\\session_...\\synchronized_session_....csv"
        --radar_csv              "C:\\...\\Processing\\session_...\\session_....csv"
        --meta_json              "C:\\...\\Processing\\session_...\\meta_data.json"
        --extrinsics_json        "C:\\...\\results\\radar_camera_extrinsics.json"
        [--neigh_r 1]
        [--min_majority 0.60]
        [--roi_top_frac 0.35]
        [--out_csv labeled_radar_points_v3.csv]

Output columns:
    video_frame_index, radar_frame_num, u, v,
    ade_id, ade_name, bucket, maj_frac,
    x, y, z, cam_z,
    doppler (if available), snr (if available)

Dependencies:
    pip install numpy pandas tqdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Segmentation metadata
# =============================================================================

def load_seg_meta(processing_session_dir: Path) -> dict:
    """Load id2label mapping written by Stage 1 (noah_export_labelmaps_from_processing.py)."""
    meta_path = processing_session_dir / "seg_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing seg_meta.json in {processing_session_dir}\n"
            f"Run noah_export_labelmaps_from_processing.py first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("id2label", {})


# =============================================================================
# Calibration loaders
# =============================================================================

def load_intrinsics_from_meta(meta_json: Path) -> np.ndarray:
    """Return 3x3 float32 camera intrinsics matrix K from meta_data.json."""
    d = json.loads(meta_json.read_text(encoding="utf-8"))
    K = np.array(d["realsense_calibration"]["rgb"]["K"], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K matrix; got shape {K.shape}")
    return K


def load_Tcr_from_extrinsics(extrinsics_json: Path) -> np.ndarray:
    """
    Return 4x4 float32 homogeneous transform T_camera_radar that maps
    a point from radar frame → camera frame.
    """
    d = json.loads(extrinsics_json.read_text(encoding="utf-8"))
    R = np.array(d["R_radar_to_camera"], dtype=np.float32).reshape(3, 3)
    t = np.array(d["t_radar_to_camera"], dtype=np.float32).reshape(3)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# =============================================================================
# Projection helper
# =============================================================================

def project_points(
    points_r: np.ndarray,   # (N, 3) radar-frame XYZ
    K: np.ndarray,           # (3, 3) camera intrinsics
    T_cr: np.ndarray,        # (4, 4) radar → camera homogeneous transform
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project N radar points into image coordinates.

    Returns:
      uv  : (N, 2) float32 — pixel coordinates (NaN for behind-camera points)
      zc  : (N,)  float32  — depth in camera frame (negative means behind camera)
    """
    N  = points_r.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pr_h = np.concatenate([points_r.astype(np.float32), ones], axis=1)  # (N, 4)

    pc_h = (T_cr @ pr_h.T).T      # (N, 4)
    pc   = pc_h[:, :3]
    zc   = pc[:, 2]

    uv = np.full((N, 2), np.nan, dtype=np.float32)
    valid = zc > 0.05
    if valid.any():
        x = pc[valid, 0]
        y = pc[valid, 1]
        z = pc[valid, 2]
        uv[valid, 0] = K[0, 0] * (x / z) + K[0, 2]
        uv[valid, 1] = K[1, 1] * (y / z) + K[1, 2]

    return uv, zc


# =============================================================================
# Neighborhood majority-label helper
# =============================================================================

def majority_label(
    label_map: np.ndarray,
    u: int, v: int,
    r: int = 1,
) -> tuple[int, float]:
    """
    Return (dominant_label_id, majority_fraction) within a (2r+1)×(2r+1)
    pixel patch centered at (u, v) in label_map.

    A low majority_fraction means the projection sits on a label boundary;
    such points should be discarded.
    """
    H, W = label_map.shape
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)

    patch = label_map[v0:v1, u0:u1].reshape(-1)
    vals, counts = np.unique(patch, return_counts=True)
    best  = int(vals[np.argmax(counts)])
    frac  = float(np.max(counts) / patch.size)
    return best, frac


# =============================================================================
# ADE class filtering and bucket mapping
# =============================================================================

# Classes that are non-navigable overhead structure — never useful for ground
# navigation feedback.  Exact ADE20K label names (lowercase).
IGNORE_ADE_EXACT = frozenset({
    "ceiling",
    "light, light source",
    "building, edifice",
    "window",
    "signboard, sign",
    "bulletin board",
    "sky",
    "fan",
    "hood, exhaust hood",
})

# Substring matches — any ADE class whose name contains one of these is ignored.
IGNORE_ADE_CONTAINS = (
    "lamp",
    "lighting",
    "pipe",
    "duct",
    "sprinkler",
    "ceiling",
    "chandelier",
    "sconce",
)


def should_ignore_ade(ade_name: str) -> bool:
    """Return True if this ADE class should never be used as a radar label."""
    if not ade_name:
        return False
    n = ade_name.lower().strip()
    if n in IGNORE_ADE_EXACT:
        return True
    return any(sub in n for sub in IGNORE_ADE_CONTAINS)


def is_person_label(ade_name: str) -> bool:
    return "person" in (ade_name or "").lower()


# Navigation bucket mapping.  Order of checks matters — more specific first.
# Buckets: wall | floor | door | pillar | human | box_like
def map_bucket(ade_name: str) -> str:
    """Map an ADE20K class name to a coarse navigation-relevant bucket."""
    n = (ade_name or "").lower()

    if "person" in n:
        return "human"

    # Vertical structural supports
    if any(kw in n for kw in ("column", "pillar", "pole", "post")):
        return "pillar"

    # Walls (includes partition, railing treated as wall-like barrier)
    if "wall" in n or "partition" in n:
        return "wall"

    # Floors — navigate on these, not obstacles in themselves
    if "floor" in n or "carpet" in n or "rug" in n or "mat" in n:
        return "floor"

    # Doors and doorways
    if "door" in n:
        return "door"

    # Stairs/escalators — treat as box_like obstacle for now
    if any(kw in n for kw in ("stair", "escalator", "step")):
        return "box_like"

    # Everything else: furniture, equipment, boxes, screens, etc.
    return "box_like"


# =============================================================================
# Radar CSV column discovery
# =============================================================================

def pick_radar_columns(df: pd.DataFrame) -> dict:
    """
    Robustly identify the relevant columns from the radar point-cloud CSV
    regardless of minor naming differences between pipeline versions.
    """
    cols = set(df.columns)

    def first_present(candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    frame   = first_present(["frame_num", "radar_frame_num", "radar_frame", "frame"])
    x       = first_present(["x", "x_m", "x_radar", "X"])
    y       = first_present(["y", "y_m", "y_radar", "Y"])
    z       = first_present(["z", "z_m", "z_radar", "Z"])
    doppler = first_present(["doppler", "doppler_mps", "velocity", "vr"])
    snr     = first_present(["snr", "SNR", "power_snr", "snr_db"])

    if frame is None:
        raise ValueError(
            f"Could not find radar frame column. "
            f"Columns available: {list(df.columns)}"
        )
    if not all([x, y, z]):
        raise ValueError(
            f"Could not find radar xyz columns. "
            f"Columns available: {list(df.columns)}"
        )

    return {
        "frame":   frame,
        "x": x, "y": y, "z": z,
        "doppler": doppler,
        "snr":     snr,
    }


# =============================================================================
# Semicolon-list parser for radar_frame_nums column in sync CSV
# =============================================================================

def parse_semicolon_int_list(val) -> list[int]:
    """Parse '12;13;14' or nan into [12, 13, 14]."""
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    result = []
    for p in s.split(";"):
        p = p.strip()
        if p:
            try:
                result.append(int(p))
            except ValueError:
                pass
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Auto-label radar points using OneFormer segmentation maps (v3)."
    )
    ap.add_argument("--processing_session_dir", required=True,
                    help="Processing session directory (contains seg/ and meta_data.json)")
    ap.add_argument("--sync_csv", required=True,
                    help="synchronized_<stem>.csv from temporal_align_single")
    ap.add_argument("--radar_csv", required=True,
                    help="<stem>.csv radar point cloud (all frames)")
    ap.add_argument("--meta_json", required=True,
                    help="meta_data.json for camera intrinsics")
    ap.add_argument("--extrinsics_json", required=True,
                    help="radar_camera_extrinsics.json from calibration solve")
    ap.add_argument("--neigh_r", type=int, default=1,
                    help="Pixel neighborhood radius for majority-label vote. Default: 1")
    ap.add_argument("--min_majority", type=float, default=0.60,
                    help="Minimum majority fraction to accept a label (0–1). Default: 0.60")
    ap.add_argument("--roi_top_frac", type=float, default=0.35,
                    help="Fraction from top of image to ignore for non-human labels. "
                         "Filters out ceiling/pipe false hits. Default: 0.35")
    ap.add_argument("--out_csv", default=None,
                    help="Output CSV path. Default: <processing_session_dir>/labeled_radar_points_v3.csv")
    args = ap.parse_args()

    processing_dir = Path(args.processing_session_dir).resolve()

    print("=" * 70)
    print("noah_autolabel_radar_using_synccsv_v3.py")
    print("=" * 70)

    # ---- Validate seg/ directory ----
    seg_dir = processing_dir / "seg"
    if not seg_dir.exists():
        raise FileNotFoundError(
            f"Missing seg/ directory in {processing_dir}\n"
            f"Run noah_export_labelmaps_from_processing.py first."
        )
    n_seg_files = sum(1 for _ in seg_dir.glob("*.npy"))
    print(f"  seg/    : {n_seg_files} label maps found")

    # ---- Load calibration ----
    id2label = load_seg_meta(processing_dir)
    K        = load_intrinsics_from_meta(Path(args.meta_json))
    T_cr     = load_Tcr_from_extrinsics(Path(args.extrinsics_json))
    print(f"  Intrinsics loaded: fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  Extrinsics loaded: T_cr shape={T_cr.shape}")

    # ---- Load synchronized CSV ----
    sync = pd.read_csv(args.sync_csv)
    for c in ("video_frame_index", "radar_frame_nums"):
        if c not in sync.columns:
            raise ValueError(f"sync_csv missing column '{c}'. "
                             f"Available: {list(sync.columns)}")
    print(f"  sync_csv: {len(sync)} rows")

    # ---- Load radar CSV ----
    radar    = pd.read_csv(args.radar_csv)
    rcol     = pick_radar_columns(radar)
    radar[rcol["frame"]] = radar[rcol["frame"]].astype(int)
    radar_by_frame = {k: v for k, v in radar.groupby(rcol["frame"])}
    print(f"  radar_csv: {len(radar)} points across {len(radar_by_frame)} frames")
    print(f"  Columns: frame='{rcol['frame']}'  x='{rcol['x']}'  "
          f"doppler='{rcol['doppler']}'  snr='{rcol['snr']}'")

    # ---- Filter config ----
    print(f"\n  Filter config:")
    print(f"    neigh_r        = {args.neigh_r}  (pixel neighborhood radius)")
    print(f"    min_majority   = {args.min_majority}  (label boundary rejection)")
    print(f"    roi_top_frac   = {args.roi_top_frac}  (non-human upper-image cutoff)")

    # ---- Process frames ----
    out_rows = []

    # Rejection counters
    rej = {
        "not_matched":       0,
        "no_seg_file":       0,
        "no_radar_frames":   0,
        "off_image":         0,
        "low_majority":      0,
        "ignored_ade":       0,
        "roi_cutoff":        0,
    }

    for _, srow in tqdm(sync.iterrows(), total=len(sync), desc="Auto-label v3"):
        # Only process matched frames
        if "status" in sync.columns and str(srow.get("status", "")).strip() != "matched":
            rej["not_matched"] += 1
            continue

        vfi      = int(srow["video_frame_index"])
        seg_path = seg_dir / f"{vfi:06d}.npy"
        if not seg_path.exists():
            rej["no_seg_file"] += 1
            continue

        label_map = np.load(seg_path).astype(np.int32)
        H, W      = label_map.shape
        roi_v_min = int(H * args.roi_top_frac)

        radar_frames = parse_semicolon_int_list(srow["radar_frame_nums"])
        if not radar_frames:
            rej["no_radar_frames"] += 1
            continue

        for rf in radar_frames:
            grp = radar_by_frame.get(rf)
            if grp is None:
                continue

            pts = grp[[rcol["x"], rcol["y"], rcol["z"]]].to_numpy(dtype=np.float32)
            if pts.shape[0] == 0:
                continue

            uv, zc = project_points(pts, K, T_cr)

            for i in range(len(pts)):
                u_f, v_f = uv[i]

                # Discard behind-camera or failed projections
                if not (np.isfinite(u_f) and np.isfinite(v_f)):
                    rej["off_image"] += 1
                    continue

                ui, vi = int(round(u_f)), int(round(v_f))
                if not (0 <= ui < W and 0 <= vi < H):
                    rej["off_image"] += 1
                    continue

                # Majority-label vote
                lid, maj = majority_label(label_map, ui, vi, r=args.neigh_r)
                if maj < args.min_majority:
                    rej["low_majority"] += 1
                    continue

                ade_name = id2label.get(str(lid), str(lid))

                # Filter non-navigable overhead classes
                if should_ignore_ade(ade_name):
                    rej["ignored_ade"] += 1
                    continue

                # ROI cutoff for non-human classes
                if vi < roi_v_min and not is_person_label(ade_name):
                    rej["roi_cutoff"] += 1
                    continue

                bucket = map_bucket(ade_name)

                rec = {
                    "video_frame_index": vfi,
                    "radar_frame_num":   int(rf),
                    "u":        ui,
                    "v":        vi,
                    "ade_id":   int(lid),
                    "ade_name": ade_name,
                    "bucket":   bucket,
                    "maj_frac": round(float(maj), 4),
                    "x":        round(float(pts[i, 0]), 6),
                    "y":        round(float(pts[i, 1]), 6),
                    "z":        round(float(pts[i, 2]), 6),
                    "cam_z":    round(float(zc[i]), 5),
                }

                if rcol["doppler"] is not None:
                    rec["doppler"] = float(grp.iloc[i][rcol["doppler"]])
                if rcol["snr"] is not None:
                    rec["snr"] = float(grp.iloc[i][rcol["snr"]])

                out_rows.append(rec)

    # ---- Save output ----
    out_df  = pd.DataFrame(out_rows)
    out_csv = args.out_csv or str(processing_dir / "labeled_radar_points_v3.csv")
    out_df.to_csv(out_csv, index=False)

    print(f"\n[DONE] {len(out_df)} labeled points written → {out_csv}")

    # ---- Summary ----
    print(f"\n  Rejection breakdown:")
    total_considered = sum(rej.values()) + len(out_rows)
    for reason, count in rej.items():
        pct = 100 * count / total_considered if total_considered > 0 else 0
        print(f"    {reason:<22}: {count:6d}  ({pct:.1f}%)")

    if len(out_df) > 0:
        print(f"\n  Bucket distribution:")
        print(out_df["bucket"].value_counts(dropna=False).to_string())

        print(f"\n  Top ADE20K labels:")
        print(out_df["ade_name"].value_counts(dropna=False).head(15).to_string())

    print(f"\nNext step:")
    print(f"  python qc_labeled_radar.py --labeled_csv \"{out_csv}\"")
    print(f"  python visualize_radar_on_video_with_seg.py")
    print(f"    --processing_session_dir \"{processing_dir}\"")
    print(f"    --labeled_csv \"{out_csv}\"")


if __name__ == "__main__":
    main()
