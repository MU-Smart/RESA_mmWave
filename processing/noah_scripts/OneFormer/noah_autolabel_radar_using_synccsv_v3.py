"""
noah_autolabel_radar_batch_by_session_v3.py

Batch/session-by-session version of Stage 2 of the auto-labeling pipeline.

What it does:
  - Walks the root Processing/ folder and finds each session directory.
  - For each processing session, finds the matching session folder in Synchronized/.
  - Finds the matching synchronized_*.csv and radar point-cloud CSV for that session.
  - Runs the same v3 auto-labeling logic as the single-session script.
  - Writes one labeled_radar_points_v3.csv per session.
  - Writes a batch summary CSV at the Processing root.

Typical usage:
    python noah_autolabel_radar_batch_by_session_v3.py \
        --processing_root_dir "C:\\...\\Processing" \
        --synchronized_root_dir "C:\\...\\Synchronized" \
        --extrinsics_json "C:\\...\\results\\radar_camera_extrinsics.json"

Optional:
        [--session_glob "session_*"]
        [--neigh_r 1]
        [--min_majority 0.60]
        [--roi_top_frac 0.35]
        [--out_name labeled_radar_points_v3.csv]
        [--summary_csv batch_autolabel_summary_v3.csv]
        [--stop_on_error]

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
    """Load id2label mapping written by Stage 1."""
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
    T[:3, 3] = t
    return T


# =============================================================================
# Projection helper
# =============================================================================


def project_points(
    points_r: np.ndarray,
    K: np.ndarray,
    T_cr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project N radar points into image coordinates.

    Returns:
      uv  : (N, 2) float32 — pixel coordinates (NaN for behind-camera points)
      zc  : (N,)  float32  — depth in camera frame (negative means behind camera)
    """
    N = points_r.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pr_h = np.concatenate([points_r.astype(np.float32), ones], axis=1)

    pc_h = (T_cr @ pr_h.T).T
    pc = pc_h[:, :3]
    zc = pc[:, 2]

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
    u: int,
    v: int,
    r: int = 1,
) -> tuple[int, float]:
    """
    Return (dominant_label_id, majority_fraction) within a (2r+1)×(2r+1)
    pixel patch centered at (u, v) in label_map.
    """
    H, W = label_map.shape
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)

    patch = label_map[v0:v1, u0:u1].reshape(-1)
    vals, counts = np.unique(patch, return_counts=True)
    best = int(vals[np.argmax(counts)])
    frac = float(np.max(counts) / patch.size)
    return best, frac


# =============================================================================
# ADE class filtering and bucket mapping
# =============================================================================


IGNORE_ADE_EXACT = frozenset(
    {
        "ceiling",
        "light, light source",
        "building, edifice",
        "window",
        "signboard, sign",
        "bulletin board",
        "sky",
        "fan",
        "hood, exhaust hood",
    }
)

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
    if not ade_name:
        return False
    n = ade_name.lower().strip()
    if n in IGNORE_ADE_EXACT:
        return True
    return any(sub in n for sub in IGNORE_ADE_CONTAINS)



def is_person_label(ade_name: str) -> bool:
    return "person" in (ade_name or "").lower()



def map_bucket(ade_name: str) -> str:
    """Map an ADE20K class name to a coarse navigation-relevant bucket."""
    n = (ade_name or "").lower()

    if "person" in n:
        return "human"
    if any(kw in n for kw in ("column", "pillar", "pole", "post")):
        return "pillar"
    if "wall" in n or "partition" in n:
        return "wall"
    if "floor" in n or "carpet" in n or "rug" in n or "mat" in n:
        return "floor"
    if "door" in n:
        return "door"
    if any(kw in n for kw in ("stair", "escalator", "step")):
        return "box_like"
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

    frame = first_present(["frame_num", "radar_frame_num", "radar_frame", "frame"])
    x = first_present(["x", "x_m", "x_radar", "X"])
    y = first_present(["y", "y_m", "y_radar", "Y"])
    z = first_present(["z", "z_m", "z_radar", "Z"])
    doppler = first_present(["doppler", "doppler_mps", "velocity", "vr"])
    snr = first_present(["snr", "SNR", "power_snr", "snr_db"])

    if frame is None:
        raise ValueError(
            f"Could not find radar frame column. Columns available: {list(df.columns)}"
        )
    if not all([x, y, z]):
        raise ValueError(
            f"Could not find radar xyz columns. Columns available: {list(df.columns)}"
        )

    return {
        "frame": frame,
        "x": x,
        "y": y,
        "z": z,
        "doppler": doppler,
        "snr": snr,
    }


# =============================================================================
# Sync helpers
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
# Session discovery helpers
# =============================================================================


def find_session_dirs(processing_root_dir: Path, session_glob: str) -> list[Path]:
    sessions = [p for p in processing_root_dir.glob(session_glob) if p.is_dir()]
    sessions.sort()
    return sessions



def find_sync_csv_for_session(sync_session_dir: Path, session_name: str) -> Path:
    preferred = sync_session_dir / f"synchronized_{session_name}.csv"
    if preferred.exists():
        return preferred

    matches = sorted(sync_session_dir.glob("synchronized_*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No synchronized_*.csv found in {sync_session_dir} for session {session_name}"
        )
    raise FileNotFoundError(
        f"Multiple synchronized_*.csv files found in {sync_session_dir}; "
        f"could not choose one automatically."
    )



def find_radar_csv_for_session(processing_session_dir: Path, session_name: str) -> Path:
    preferred = processing_session_dir / f"{session_name}.csv"
    if preferred.exists():
        return preferred

    excluded_names = {
        "labeled_radar_points_v3.csv",
        "batch_autolabel_summary_v3.csv",
    }
    matches = [
        p
        for p in sorted(processing_session_dir.glob("*.csv"))
        if p.name not in excluded_names and not p.name.startswith("synchronized_")
    ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No radar CSV found in {processing_session_dir} for session {session_name}"
        )
    raise FileNotFoundError(
        f"Multiple candidate radar CSV files found in {processing_session_dir}; "
        f"could not choose one automatically."
    )


# =============================================================================
# Core processing for one session
# =============================================================================


def process_one_session(
    processing_session_dir: Path,
    sync_csv: Path,
    radar_csv: Path,
    extrinsics_json: Path,
    neigh_r: int,
    min_majority: float,
    roi_top_frac: float,
    out_csv: Path,
) -> dict:
    processing_session_dir = processing_session_dir.resolve()
    meta_json = processing_session_dir / "meta_data.json"
    seg_dir = processing_session_dir / "seg"

    if not seg_dir.exists():
        raise FileNotFoundError(
            f"Missing seg/ directory in {processing_session_dir}\n"
            f"Run noah_export_labelmaps_from_processing.py first."
        )
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing meta_data.json in {processing_session_dir}")

    n_seg_files = sum(1 for _ in seg_dir.glob("*.npy"))
    id2label = load_seg_meta(processing_session_dir)
    K = load_intrinsics_from_meta(meta_json)
    T_cr = load_Tcr_from_extrinsics(extrinsics_json)

    sync = pd.read_csv(sync_csv)
    for c in ("video_frame_index", "radar_frame_nums"):
        if c not in sync.columns:
            raise ValueError(
                f"sync_csv missing column '{c}'. Available: {list(sync.columns)}"
            )

    radar = pd.read_csv(radar_csv)
    rcol = pick_radar_columns(radar)
    radar[rcol["frame"]] = radar[rcol["frame"]].astype(int)
    radar_by_frame = {k: v for k, v in radar.groupby(rcol["frame"])}

    out_rows = []
    rej = {
        "not_matched": 0,
        "no_seg_file": 0,
        "no_radar_frames": 0,
        "off_image": 0,
        "low_majority": 0,
        "ignored_ade": 0,
        "roi_cutoff": 0,
    }

    for _, srow in tqdm(sync.iterrows(), total=len(sync), desc=processing_session_dir.name, leave=False):
        if "status" in sync.columns and str(srow.get("status", "")).strip() != "matched":
            rej["not_matched"] += 1
            continue

        vfi = int(srow["video_frame_index"])
        seg_path = seg_dir / f"{vfi:06d}.npy"
        if not seg_path.exists():
            rej["no_seg_file"] += 1
            continue

        label_map = np.load(seg_path).astype(np.int32)
        H, W = label_map.shape
        roi_v_min = int(H * roi_top_frac)

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
                if not (np.isfinite(u_f) and np.isfinite(v_f)):
                    rej["off_image"] += 1
                    continue

                ui, vi = int(round(u_f)), int(round(v_f))
                if not (0 <= ui < W and 0 <= vi < H):
                    rej["off_image"] += 1
                    continue

                lid, maj = majority_label(label_map, ui, vi, r=neigh_r)
                if maj < min_majority:
                    rej["low_majority"] += 1
                    continue

                ade_name = id2label.get(str(lid), str(lid))
                if should_ignore_ade(ade_name):
                    rej["ignored_ade"] += 1
                    continue

                if vi < roi_v_min and not is_person_label(ade_name):
                    rej["roi_cutoff"] += 1
                    continue

                rec = {
                    "video_frame_index": vfi,
                    "radar_frame_num": int(rf),
                    "u": ui,
                    "v": vi,
                    "ade_id": int(lid),
                    "ade_name": ade_name,
                    "bucket": map_bucket(ade_name),
                    "maj_frac": round(float(maj), 4),
                    "x": round(float(pts[i, 0]), 6),
                    "y": round(float(pts[i, 1]), 6),
                    "z": round(float(pts[i, 2]), 6),
                    "cam_z": round(float(zc[i]), 5),
                }

                if rcol["doppler"] is not None:
                    rec["doppler"] = float(grp.iloc[i][rcol["doppler"]])
                if rcol["snr"] is not None:
                    rec["snr"] = float(grp.iloc[i][rcol["snr"]])

                out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_csv, index=False)

    total_considered = sum(rej.values()) + len(out_rows)
    summary = {
        "session_name": processing_session_dir.name,
        "processing_session_dir": str(processing_session_dir),
        "sync_csv": str(sync_csv),
        "radar_csv": str(radar_csv),
        "meta_json": str(meta_json),
        "out_csv": str(out_csv),
        "n_seg_files": int(n_seg_files),
        "n_sync_rows": int(len(sync)),
        "n_radar_points": int(len(radar)),
        "n_radar_frames": int(len(radar_by_frame)),
        "labeled_points": int(len(out_df)),
        "total_considered": int(total_considered),
        "rej_not_matched": int(rej["not_matched"]),
        "rej_no_seg_file": int(rej["no_seg_file"]),
        "rej_no_radar_frames": int(rej["no_radar_frames"]),
        "rej_off_image": int(rej["off_image"]),
        "rej_low_majority": int(rej["low_majority"]),
        "rej_ignored_ade": int(rej["ignored_ade"]),
        "rej_roi_cutoff": int(rej["roi_cutoff"]),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "status": "ok",
        "error": "",
    }
    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Batch auto-label radar points session by session using OneFormer segmentation maps (v3)."
    )
    ap.add_argument(
        "--processing_root_dir",
        required=True,
        help="Root Processing directory containing session_* subfolders",
    )
    ap.add_argument(
        "--synchronized_root_dir",
        required=True,
        help="Root Synchronized directory containing matching session_* subfolders",
    )
    ap.add_argument(
        "--extrinsics_json",
        required=True,
        help="radar_camera_extrinsics.json from calibration solve",
    )
    ap.add_argument(
        "--session_glob",
        default="session_*",
        help="Glob for session directories inside Processing root. Default: session_*",
    )
    ap.add_argument(
        "--neigh_r",
        type=int,
        default=1,
        help="Pixel neighborhood radius for majority-label vote. Default: 1",
    )
    ap.add_argument(
        "--min_majority",
        type=float,
        default=0.60,
        help="Minimum majority fraction to accept a label (0–1). Default: 0.60",
    )
    ap.add_argument(
        "--roi_top_frac",
        type=float,
        default=0.35,
        help="Fraction from top of image to ignore for non-human labels. Default: 0.35",
    )
    ap.add_argument(
        "--out_name",
        default="labeled_radar_points_v3.csv",
        help="Per-session output filename written inside each Processing/session_*/ folder",
    )
    ap.add_argument(
        "--summary_csv",
        default=None,
        help="Optional batch summary CSV path. Default: <processing_root_dir>/batch_autolabel_summary_v3.csv",
    )
    ap.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one session fails. Default: continue and summarize errors.",
    )
    args = ap.parse_args()

    processing_root_dir = Path(args.processing_root_dir).resolve()
    synchronized_root_dir = Path(args.synchronized_root_dir).resolve()
    extrinsics_json = Path(args.extrinsics_json).resolve()

    if not processing_root_dir.exists():
        raise FileNotFoundError(f"Processing root not found: {processing_root_dir}")
    if not synchronized_root_dir.exists():
        raise FileNotFoundError(f"Synchronized root not found: {synchronized_root_dir}")
    if not extrinsics_json.exists():
        raise FileNotFoundError(f"Extrinsics JSON not found: {extrinsics_json}")

    sessions = find_session_dirs(processing_root_dir, args.session_glob)
    if not sessions:
        raise FileNotFoundError(
            f"No session directories found in {processing_root_dir} matching '{args.session_glob}'"
        )

    summary_csv = (
        Path(args.summary_csv).resolve()
        if args.summary_csv
        else processing_root_dir / "batch_autolabel_summary_v3.csv"
    )

    print("=" * 80)
    print("noah_autolabel_radar_batch_by_session_v3.py")
    print("=" * 80)
    print(f"Processing root   : {processing_root_dir}")
    print(f"Synchronized root : {synchronized_root_dir}")
    print(f"Extrinsics        : {extrinsics_json}")
    print(f"Sessions found    : {len(sessions)}")
    print("\nFilter config:")
    print(f"  neigh_r      = {args.neigh_r}")
    print(f"  min_majority = {args.min_majority}")
    print(f"  roi_top_frac = {args.roi_top_frac}")
    print(f"  out_name     = {args.out_name}")
    print()

    batch_rows: list[dict] = []

    for processing_session_dir in tqdm(sessions, desc="Sessions"):
        session_name = processing_session_dir.name
        print(f"\n--- [{session_name}] ---")
        try:
            sync_session_dir = synchronized_root_dir / session_name
            if not sync_session_dir.exists():
                raise FileNotFoundError(
                    f"Missing synchronized session folder: {sync_session_dir}"
                )

            sync_csv = find_sync_csv_for_session(sync_session_dir, session_name)
            radar_csv = find_radar_csv_for_session(processing_session_dir, session_name)
            out_csv = processing_session_dir / args.out_name

            print(f"sync_csv   : {sync_csv.name}")
            print(f"radar_csv  : {radar_csv.name}")
            print(f"output_csv : {out_csv.name}")

            summary = process_one_session(
                processing_session_dir=processing_session_dir,
                sync_csv=sync_csv,
                radar_csv=radar_csv,
                extrinsics_json=extrinsics_json,
                neigh_r=args.neigh_r,
                min_majority=args.min_majority,
                roi_top_frac=args.roi_top_frac,
                out_csv=out_csv,
            )
            batch_rows.append(summary)
            print(f"[DONE] labeled_points={summary['labeled_points']}")

        except Exception as e:
            err_row = {
                "session_name": session_name,
                "processing_session_dir": str(processing_session_dir),
                "sync_csv": "",
                "radar_csv": "",
                "meta_json": str(processing_session_dir / 'meta_data.json'),
                "out_csv": str(processing_session_dir / args.out_name),
                "n_seg_files": 0,
                "n_sync_rows": 0,
                "n_radar_points": 0,
                "n_radar_frames": 0,
                "labeled_points": 0,
                "total_considered": 0,
                "rej_not_matched": 0,
                "rej_no_seg_file": 0,
                "rej_no_radar_frames": 0,
                "rej_off_image": 0,
                "rej_low_majority": 0,
                "rej_ignored_ade": 0,
                "rej_roi_cutoff": 0,
                "fx": np.nan,
                "fy": np.nan,
                "cx": np.nan,
                "cy": np.nan,
                "status": "error",
                "error": str(e),
            }
            batch_rows.append(err_row)
            print(f"[ERROR] {e}")
            if args.stop_on_error:
                raise

    summary_df = pd.DataFrame(batch_rows)
    summary_df.to_csv(summary_csv, index=False)

    ok_count = int((summary_df["status"] == "ok").sum()) if len(summary_df) else 0
    err_count = int((summary_df["status"] == "error").sum()) if len(summary_df) else 0
    total_labeled = int(summary_df["labeled_points"].fillna(0).sum()) if len(summary_df) else 0

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Sessions processed : {len(summary_df)}")
    print(f"Succeeded          : {ok_count}")
    print(f"Failed             : {err_count}")
    print(f"Total labeled pts  : {total_labeled}")
    print(f"Summary CSV        : {summary_csv}")
    print("\nPer-session labeled CSVs were written inside each Processing/session_* folder.")


if __name__ == "__main__":
    main()
