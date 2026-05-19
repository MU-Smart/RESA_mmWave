"""
noah_autolabel_radar_using_synccsv_v4_depth_gated.py

Depth-gated variant of the v4_bestmatch autolabeler.

Core fix for foreground-occlusion misclassification (MASTER.md §10F):
    The v4 baseline projects each radar point into 2D image space and reads
    the OneFormer segment at that pixel — ignoring depth.  A radar return
    from a wall BEHIND a person therefore receives the "person" label.

    This script adds a depth-occlusion gate:
      - For every projected radar point at pixel (u, v) with camera-frame
        depth zc metres, it loads the synchronized RealSense depth frame and
        samples the minimum valid depth in a small neighbourhood around (u, v).
      - If that minimum depth is more than `--depth_occlusion_m` metres CLOSER
        than zc, the label at (u, v) belongs to a foreground occluder, not
        the radar point.  The label is rejected (counted as rej_depth_occluded).

    Depth frames are expected at:
        <processing_session_dir>/depth/<vfi:06d>.npy   (uint16, mm)
    This matches the storage format written by dca_realsense_recorder_v7.py.
    The depth scale is read from meta_data.json (field: depth_scale_m_per_unit);
    it defaults to 0.001 (1 mm per raw unit) if absent.

    When no depth frame is available for a video frame index, the gate is
    skipped for that point (conservative: label is kept).

All other logic (batch session discovery, sync CSV matching, ADE20K bucket
mapping, majority-label vote, ROI cutoff) is identical to v4_bestmatch.

Typical usage:
    python noah_autolabel_radar_using_synccsv_v4_depth_gated.py \\
        --processing_root_dir /path/to/data-4-30/Processing \\
        --synchronized_root_dir /path/to/data-4-30/Processing \\
        --assign_mode best_per_radar \\
        --max_time_diff_ms 35 \\
        --depth_occlusion_m 0.20 \\
        --depth_patch_r 2

Dependencies:
    pip install numpy pandas tqdm opencv-python
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

_DEFAULT_EXTRINSICS_JSON = (
    Path(__file__).parents[2] / "jetson_nav_pipeline/config/radar_camera_extrinsics.json"
)
_DEFAULT_DEPTH_SCALE = 0.001  # metres per raw uint16 unit (RealSense default)


# =============================================================================
# Depth helpers
# =============================================================================


def load_depth_scale(meta_json: Path) -> float:
    """Read depth_scale_m_per_unit from meta_data.json; fall back to 0.001."""
    try:
        d = json.loads(meta_json.read_text(encoding="utf-8"))
        return float(d.get("realsense_calibration", {}).get("depth_scale_m_per_unit",
                                                             _DEFAULT_DEPTH_SCALE))
    except Exception:
        return _DEFAULT_DEPTH_SCALE


class DepthFrameCache:
    """LRU-style cache for depth frames, keyed by (session_dir, frame_index).

    Keeps the last `capacity` unique frames in memory.  A capacity of ~4 is
    enough since radar frames are matched one-to-one with video frames.
    """

    def __init__(self, depth_dir: Path, depth_scale: float, capacity: int = 4):
        self._depth_dir = depth_dir
        self._scale = depth_scale
        self._cache: dict[int, np.ndarray | None] = {}
        self._order: list[int] = []
        self._capacity = capacity

    def get(self, frame_index: int) -> np.ndarray | None:
        """Return (H, W) float32 depth in metres, or None if frame missing."""
        if frame_index in self._cache:
            return self._cache[frame_index]

        path = self._depth_dir / f"{frame_index:06d}.npy"
        if path.exists():
            raw = np.load(path)  # uint16
            depth_m = raw.astype(np.float32) * self._scale
        else:
            depth_m = None

        # Evict oldest if at capacity
        if len(self._order) >= self._capacity:
            old = self._order.pop(0)
            del self._cache[old]

        self._cache[frame_index] = depth_m
        self._order.append(frame_index)
        return depth_m


def depth_min_in_patch(
    depth_m: np.ndarray,
    u: int,
    v: int,
    r: int,
) -> float:
    """
    Return the minimum *valid* (> 0) depth in a (2r+1)×(2r+1) patch
    centred at (u, v).  Returns 0.0 if no valid depth pixels exist.

    Using the minimum rather than the mean ensures we detect any foreground
    object present in the neighbourhood, even if it covers only a few pixels.
    """
    H, W = depth_m.shape
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[patch > 0]
    return float(valid.min()) if valid.size > 0 else 0.0


# =============================================================================
# Everything below is copied verbatim from v4_bestmatch except where noted
# with  # <DEPTH-GATE>  comments.
# =============================================================================


def load_seg_meta(processing_session_dir: Path) -> dict:
    meta_path = processing_session_dir / "seg_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing seg_meta.json in {processing_session_dir}\n"
            f"Run noah_export_labelmaps_from_processing_v2.py first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("id2label", {})


def load_intrinsics_from_meta(meta_json: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(meta_json.read_text(encoding="utf-8"))
    rgb = d["realsense_calibration"]["rgb"]
    if "K" in rgb:
        K = np.array(rgb["K"], dtype=np.float32)
    else:
        fx, fy = float(rgb["fx"]), float(rgb["fy"])
        cx, cy = float(rgb["cx"]), float(rgb["cy"])
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 K matrix; got shape {K.shape}")
    dist = np.array(rgb.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float32)
    return K, dist


def load_Tcr_from_extrinsics(extrinsics_json: Path) -> np.ndarray:
    d = json.loads(extrinsics_json.read_text(encoding="utf-8"))
    R = np.array(d["R_radar_to_camera"], dtype=np.float32).reshape(3, 3)
    t = np.array(d["t_radar_to_camera"], dtype=np.float32).reshape(3)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def project_points(
    points_r: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_cr: np.ndarray,
    use_distortion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    N = points_r.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pr_h = np.concatenate([points_r.astype(np.float32), ones], axis=1)
    pc_h = (T_cr @ pr_h.T).T
    pc = pc_h[:, :3]
    zc = pc[:, 2]
    uv = np.full((N, 2), np.nan, dtype=np.float32)
    valid = zc > 0.05
    if valid.any():
        if use_distortion:
            pts_valid = pc[valid].reshape(-1, 1, 3).astype(np.float64)
            pts_2d, _ = cv2.projectPoints(
                pts_valid,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                K.astype(np.float64),
                dist.astype(np.float64),
            )
            uv[valid] = pts_2d.reshape(-1, 2).astype(np.float32)
        else:
            x = pc[valid, 0]
            y = pc[valid, 1]
            z = pc[valid, 2]
            uv[valid, 0] = K[0, 0] * (x / z) + K[0, 2]
            uv[valid, 1] = K[1, 1] * (y / z) + K[1, 2]
    return uv, zc


def majority_label(label_map: np.ndarray, u: int, v: int, r: int = 1) -> tuple[int, float]:
    H, W = label_map.shape
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    patch = label_map[v0:v1, u0:u1].reshape(-1)
    vals, counts = np.unique(patch, return_counts=True)
    best = int(vals[np.argmax(counts)])
    frac = float(np.max(counts) / patch.size)
    return best, frac


IGNORE_ADE_EXACT = frozenset({
    "ceiling", "light, light source", "building, edifice", "window",
    "signboard, sign", "bulletin board", "sky", "fan", "hood, exhaust hood",
})
IGNORE_ADE_CONTAINS = ("lamp", "lighting", "pipe", "duct", "sprinkler",
                       "ceiling", "chandelier", "sconce")


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


def pick_radar_columns(df: pd.DataFrame) -> dict:
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
    range_bin = first_present(["range_bin"])
    doppler_bin = first_present(["doppler_bin"])
    if frame is None:
        raise ValueError(f"Could not find radar frame column. Available: {list(df.columns)}")
    if not all([x, y, z]):
        raise ValueError(f"Could not find radar xyz columns. Available: {list(df.columns)}")
    return {"frame": frame, "x": x, "y": y, "z": z, "doppler": doppler, "snr": snr,
            "range_bin": range_bin, "doppler_bin": doppler_bin}


def parse_semicolon_int_list(val) -> list[int]:
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


def parse_semicolon_float_list(val) -> list[float]:
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
                result.append(float(p))
            except ValueError:
                pass
    return result


def build_best_video_for_radar_frame(
    sync: pd.DataFrame,
    max_time_diff_us: int | None = None,
) -> tuple[dict[int, dict], dict]:
    best_map: dict[int, dict] = {}
    stats = {
        "candidate_links": 0,
        "links_missing_diff": 0,
        "links_over_time_gate": 0,
        "links_nonbest_duplicate": 0,
        "unique_radar_frames_selected": 0,
    }
    for _, srow in sync.iterrows():
        if "status" in sync.columns and str(srow.get("status", "")).strip() != "matched":
            continue
        try:
            vfi = int(srow["video_frame_index"])
        except Exception:
            continue
        radar_frames = parse_semicolon_int_list(srow.get("radar_frame_nums"))
        time_diffs = parse_semicolon_float_list(srow.get("time_diffs_us"))
        if not radar_frames:
            continue
        for idx, rf in enumerate(radar_frames):
            stats["candidate_links"] += 1
            diff_us = int(round(time_diffs[idx])) if idx < len(time_diffs) else None
            if diff_us is None:
                stats["links_missing_diff"] += 1
            if max_time_diff_us is not None and diff_us is not None and diff_us > max_time_diff_us:
                stats["links_over_time_gate"] += 1
                continue
            prev = best_map.get(rf)
            if prev is None:
                best_map[rf] = {"video_frame_index": vfi,
                                "time_diff_us": diff_us if diff_us is not None else 10**18}
                continue
            prev_diff = int(prev.get("time_diff_us", 10**18))
            cand_diff = int(diff_us if diff_us is not None else 10**18)
            if cand_diff < prev_diff:
                best_map[rf] = {"video_frame_index": vfi, "time_diff_us": cand_diff}
            stats["links_nonbest_duplicate"] += 1
    stats["unique_radar_frames_selected"] = len(best_map)
    return best_map, stats


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
        f"Multiple synchronized_*.csv files in {sync_session_dir}; cannot choose automatically."
    )


def find_radar_csv_for_session(processing_session_dir: Path, session_name: str) -> Path:
    preferred = processing_session_dir / f"{session_name}.csv"
    if preferred.exists():
        return preferred
    excluded_names = {"labeled_radar_points_v4.csv", "batch_autolabel_summary_v3.csv"}
    matches = [
        p for p in sorted(processing_session_dir.glob("*.csv"))
        if p.name not in excluded_names and not p.name.startswith("synchronized_")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No radar CSV found in {processing_session_dir} for session {session_name}"
        )
    raise FileNotFoundError(
        f"Multiple candidate radar CSVs in {processing_session_dir}; cannot choose automatically."
    )


# =============================================================================
# Core per-session processor — depth-gated version
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
    assign_mode: str = "all_within_window",
    max_time_diff_ms: float | None = None,
    use_distortion: bool = False,
    # <DEPTH-GATE> new parameters
    depth_occlusion_m: float = 0.20,
    depth_patch_r: int = 2,
) -> dict:
    processing_session_dir = processing_session_dir.resolve()
    meta_json = processing_session_dir / "meta_data.json"
    seg_dir = processing_session_dir / "seg"
    # <DEPTH-GATE> depth directory (may not exist — gate is skipped gracefully)
    depth_dir = processing_session_dir / "depth"

    if not seg_dir.exists():
        raise FileNotFoundError(
            f"Missing seg/ directory in {processing_session_dir}\n"
            f"Run noah_export_labelmaps_from_processing_v2.py first."
        )
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing meta_data.json in {processing_session_dir}")

    n_seg_files = sum(1 for _ in seg_dir.glob("*.npy"))
    id2label = load_seg_meta(processing_session_dir)
    K, dist = load_intrinsics_from_meta(meta_json)
    T_cr = load_Tcr_from_extrinsics(extrinsics_json)

    # <DEPTH-GATE> set up depth cache (or disable if no depth dir)
    depth_available = depth_dir.is_dir()
    depth_scale = load_depth_scale(meta_json) if depth_available else _DEFAULT_DEPTH_SCALE
    depth_cache = DepthFrameCache(depth_dir, depth_scale) if depth_available else None

    sync = pd.read_csv(sync_csv)
    for c in ("video_frame_index", "radar_frame_nums"):
        if c not in sync.columns:
            raise ValueError(f"sync_csv missing column '{c}'. Available: {list(sync.columns)}")

    radar = pd.read_csv(radar_csv)
    rcol = pick_radar_columns(radar)
    radar[rcol["frame"]] = radar[rcol["frame"]].astype(int)
    radar_by_frame = {k: v for k, v in radar.groupby(rcol["frame"])}

    range_bin_present = rcol["range_bin"] is not None
    doppler_bin_present = rcol["doppler_bin"] is not None
    if range_bin_present and doppler_bin_present:
        print(f"  [BINS] range_bin='{rcol['range_bin']}' doppler_bin='{rcol['doppler_bin']}' — will be written to output")
    elif range_bin_present:
        print(f"  [BINS] range_bin='{rcol['range_bin']}' found; doppler_bin MISSING — doppler_bin omitted from output")
    elif doppler_bin_present:
        print(f"  [BINS] doppler_bin='{rcol['doppler_bin']}' found; range_bin MISSING — range_bin omitted from output")
    else:
        print(f"  [BINS] WARNING: range_bin and doppler_bin not found in radar CSV — both omitted from output")
        print(f"         This session is invalid for the active April-28-only RD patch pipeline.")
        print(f"         Regenerate radar CSVs with 3branch_adc_to_pointcloud.py to preserve native bins.")

    out_rows = []
    rej = {
        "not_matched": 0,
        "no_seg_file": 0,
        "no_radar_frames": 0,
        "missing_radar_group": 0,
        "off_image": 0,
        "low_majority": 0,
        "ignored_ade": 0,
        "roi_cutoff": 0,
        "depth_occluded": 0,  # <DEPTH-GATE> new rejection reason
    }

    max_time_diff_us = None
    if max_time_diff_ms is not None:
        max_time_diff_us = int(round(float(max_time_diff_ms) * 1000.0))

    def _emit_point(vfi, rf, match_time_diff_us, label_map, H, W, roi_v_min, pts, i, uv, zc):
        """Inner function shared between assign modes.  Returns rec dict or None."""
        u_f, v_f = uv[i]
        if not (np.isfinite(u_f) and np.isfinite(v_f)):
            rej["off_image"] += 1
            return None

        ui, vi = int(round(u_f)), int(round(v_f))
        if not (0 <= ui < W and 0 <= vi < H):
            rej["off_image"] += 1
            return None

        # <DEPTH-GATE> occlusion check before ADE lookup
        if depth_cache is not None and depth_occlusion_m > 0:
            depth_frame = depth_cache.get(vfi)
            if depth_frame is not None:
                min_d = depth_min_in_patch(depth_frame, ui, vi, depth_patch_r)
                if min_d > 0 and min_d < (float(zc[i]) - depth_occlusion_m):
                    rej["depth_occluded"] += 1
                    return None

        lid, maj = majority_label(label_map, ui, vi, r=neigh_r)
        if maj < min_majority:
            rej["low_majority"] += 1
            return None

        ade_name = id2label.get(str(lid), str(lid))
        if should_ignore_ade(ade_name):
            rej["ignored_ade"] += 1
            return None

        if vi < roi_v_min and not is_person_label(ade_name):
            rej["roi_cutoff"] += 1
            return None

        rec = {
            "video_frame_index": vfi,
            "radar_frame_num": int(rf),
            "match_time_diff_us": int(match_time_diff_us) if match_time_diff_us is not None else -1,
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
            rec["doppler"] = float(radar_by_frame[rf].iloc[i][rcol["doppler"]])
        if rcol["snr"] is not None:
            rec["snr"] = float(radar_by_frame[rf].iloc[i][rcol["snr"]])
        if rcol["range_bin"] is not None:
            rec["range_bin"] = int(radar_by_frame[rf].iloc[i][rcol["range_bin"]])
        if rcol["doppler_bin"] is not None:
            rec["doppler_bin"] = int(radar_by_frame[rf].iloc[i][rcol["doppler_bin"]])
        return rec

    if assign_mode == "best_per_radar":
        best_map, match_stats = build_best_video_for_radar_frame(
            sync, max_time_diff_us=max_time_diff_us
        )
        seg_cache: dict[int, tuple] = {}

        for rf, match in tqdm(sorted(best_map.items()), total=len(best_map),
                               desc=processing_session_dir.name, leave=False):
            vfi = int(match["video_frame_index"])
            seg_tuple = seg_cache.get(vfi)
            if seg_tuple is None:
                seg_path = seg_dir / f"{vfi:06d}.npy"
                if not seg_path.exists():
                    rej["no_seg_file"] += 1
                    continue
                label_map = np.load(seg_path).astype(np.int32)
                H, W = label_map.shape
                roi_v_min = int(H * roi_top_frac)
                seg_tuple = (label_map, H, W, roi_v_min)
                seg_cache[vfi] = seg_tuple

            label_map, H, W, roi_v_min = seg_tuple
            grp = radar_by_frame.get(rf)
            if grp is None:
                rej["missing_radar_group"] += 1
                continue

            pts = grp[[rcol["x"], rcol["y"], rcol["z"]]].to_numpy(dtype=np.float32)
            if pts.shape[0] == 0:
                continue

            uv, zc = project_points(pts, K, dist, T_cr, use_distortion=use_distortion)

            for i in range(len(pts)):
                rec = _emit_point(vfi, rf, match.get("time_diff_us"),
                                   label_map, H, W, roi_v_min, pts, i, uv, zc)
                if rec is not None:
                    # Fix doppler/snr/bins: use grp row i
                    if rcol["doppler"] is not None:
                        rec["doppler"] = float(grp.iloc[i][rcol["doppler"]])
                    if rcol["snr"] is not None:
                        rec["snr"] = float(grp.iloc[i][rcol["snr"]])
                    if rcol["range_bin"] is not None:
                        rec["range_bin"] = int(grp.iloc[i][rcol["range_bin"]])
                    if rcol["doppler_bin"] is not None:
                        rec["doppler_bin"] = int(grp.iloc[i][rcol["doppler_bin"]])
                    out_rows.append(rec)
    else:
        match_stats = {k: 0 for k in ["candidate_links", "links_missing_diff",
                                        "links_over_time_gate", "links_nonbest_duplicate",
                                        "unique_radar_frames_selected"]}
        for _, srow in tqdm(sync.iterrows(), total=len(sync),
                             desc=processing_session_dir.name, leave=False):
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
                    rej["missing_radar_group"] += 1
                    continue

                pts = grp[[rcol["x"], rcol["y"], rcol["z"]]].to_numpy(dtype=np.float32)
                if pts.shape[0] == 0:
                    continue

                uv, zc = project_points(pts, K, dist, T_cr, use_distortion=use_distortion)

                for i in range(len(pts)):
                    rec = _emit_point(vfi, rf, None, label_map, H, W, roi_v_min, pts, i, uv, zc)
                    if rec is not None:
                        if rcol["doppler"] is not None:
                            rec["doppler"] = float(grp.iloc[i][rcol["doppler"]])
                        if rcol["snr"] is not None:
                            rec["snr"] = float(grp.iloc[i][rcol["snr"]])
                        if rcol["range_bin"] is not None:
                            rec["range_bin"] = int(grp.iloc[i][rcol["range_bin"]])
                        if rcol["doppler_bin"] is not None:
                            rec["doppler_bin"] = int(grp.iloc[i][rcol["doppler_bin"]])
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
        "rej_missing_radar_group": int(rej["missing_radar_group"]),
        "rej_off_image": int(rej["off_image"]),
        "rej_low_majority": int(rej["low_majority"]),
        "rej_ignored_ade": int(rej["ignored_ade"]),
        "rej_roi_cutoff": int(rej["roi_cutoff"]),
        "rej_depth_occluded": int(rej["depth_occluded"]),  # <DEPTH-GATE>
        "range_bin_present": range_bin_present,
        "doppler_bin_present": doppler_bin_present,
        "depth_available": depth_available,
        "depth_scale_m_per_unit": depth_scale,
        "depth_occlusion_m": depth_occlusion_m,
        "depth_patch_r": depth_patch_r,
        "assign_mode": assign_mode,
        "max_time_diff_ms": float(max_time_diff_ms) if max_time_diff_ms is not None else np.nan,
        "candidate_links": int(match_stats.get("candidate_links", 0)),
        "links_missing_diff": int(match_stats.get("links_missing_diff", 0)),
        "links_over_time_gate": int(match_stats.get("links_over_time_gate", 0)),
        "links_nonbest_duplicate": int(match_stats.get("links_nonbest_duplicate", 0)),
        "unique_radar_frames_selected": int(match_stats.get("unique_radar_frames_selected", 0)),
        "projection_model": "opencv_distorted" if use_distortion else "pinhole_extrinsics",
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
        description=(
            "Depth-gated batch autolabeler. Identical to v4_bestmatch but adds a "
            "per-point depth-occlusion check to prevent foreground objects from "
            "mislabeling background radar returns."
        )
    )
    ap.add_argument("--processing_root_dir", required=True,
                    help="Root Processing directory containing session_* subfolders")
    ap.add_argument("--synchronized_root_dir", required=True,
                    help="Root Synchronized directory containing matching session_* subfolders")
    ap.add_argument("--extrinsics_json", default=str(_DEFAULT_EXTRINSICS_JSON),
                    help=f"Path to radar_camera_extrinsics.json. Default: {_DEFAULT_EXTRINSICS_JSON}")
    ap.add_argument("--session_glob", default="session_*",
                    help="Glob for session directories inside Processing root. Default: session_*")
    ap.add_argument("--neigh_r", type=int, default=1,
                    help="Pixel neighbourhood radius for majority-label vote. Default: 1")
    ap.add_argument("--min_majority", type=float, default=0.60,
                    help="Minimum majority fraction to accept a label (0–1). Default: 0.60")
    ap.add_argument("--roi_top_frac", type=float, default=0.35,
                    help="Fraction from top of image to ignore for non-human labels. Default: 0.35")
    ap.add_argument("--assign_mode", choices=["best_per_radar", "all_within_window"],
                    default="all_within_window")
    ap.add_argument("--max_time_diff_ms", type=float, default=None,
                    help="Max radar-video time difference in ms (best_per_radar mode only). Default: disabled")
    ap.add_argument("--use_distortion", action="store_true",
                    help="Apply camera distortion coefficients during projection.")
    # <DEPTH-GATE> new arguments
    ap.add_argument(
        "--depth_occlusion_m", type=float, default=0.20,
        help=(
            "Minimum depth gap (metres) between the depth sensor reading and the radar "
            "camera-frame depth to trigger occlusion rejection. "
            "If the depth image shows an object >= this many metres CLOSER than the radar "
            "return, the label is discarded. Set 0 to disable. Default: 0.20"
        ),
    )
    ap.add_argument(
        "--depth_patch_r", type=int, default=2,
        help=(
            "Radius of the depth patch sampled around each projected pixel. "
            "The minimum valid depth in a (2r+1)×(2r+1) neighbourhood is used, "
            "making the gate robust to depth sensor holes. Default: 2"
        ),
    )
    ap.add_argument("--out_name", default="labeled_radar_points_v4.csv",
                    help="Per-session output filename. Default: labeled_radar_points_v4.csv")
    ap.add_argument("--summary_csv", default=None,
                    help="Optional batch summary CSV. Default: <processing_root>/batch_autolabel_summary_depth_gated.csv")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Stop immediately on any session failure.")
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
        else processing_root_dir / "batch_autolabel_summary_depth_gated.csv"
    )

    print("=" * 80)
    print("noah_autolabel_radar_using_synccsv_v4_depth_gated.py")
    print("=" * 80)
    print(f"Processing root   : {processing_root_dir}")
    print(f"Synchronized root : {synchronized_root_dir}")
    print(f"Extrinsics        : {extrinsics_json}")
    print(f"Sessions found    : {len(sessions)}")
    print("\nFilter config:")
    print(f"  neigh_r           = {args.neigh_r}")
    print(f"  min_majority      = {args.min_majority}")
    print(f"  roi_top_frac      = {args.roi_top_frac}")
    print(f"  assign_mode       = {args.assign_mode}")
    print(f"  max_dt_ms         = {args.max_time_diff_ms if args.max_time_diff_ms is not None else 'disabled'}")
    print(f"  projection        = {'opencv_distorted' if args.use_distortion else 'pinhole_extrinsics'}")
    print(f"  depth_occlusion_m = {args.depth_occlusion_m}  ({'disabled' if args.depth_occlusion_m <= 0 else 'active'})")
    print(f"  depth_patch_r     = {args.depth_patch_r}")
    print(f"  out_name          = {args.out_name}")
    print()

    batch_rows: list[dict] = []

    for processing_session_dir in tqdm(sessions, desc="Sessions"):
        session_name = processing_session_dir.name
        print(f"\n--- [{session_name}] ---")
        try:
            sync_session_dir = synchronized_root_dir / session_name
            if not sync_session_dir.exists():
                raise FileNotFoundError(f"Missing synchronized session folder: {sync_session_dir}")

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
                assign_mode=args.assign_mode,
                max_time_diff_ms=args.max_time_diff_ms,
                use_distortion=args.use_distortion,
                depth_occlusion_m=args.depth_occlusion_m,
                depth_patch_r=args.depth_patch_r,
            )
            batch_rows.append(summary)
            depth_rej = summary["rej_depth_occluded"]
            total = summary["total_considered"]
            pct = 100.0 * depth_rej / max(total, 1)
            bins_status = (
                "range_bin+doppler_bin=OK"
                if summary["range_bin_present"] and summary["doppler_bin_present"]
                else "range_bin+doppler_bin=MISSING"
            )
            print(f"[DONE] labeled_points={summary['labeled_points']}  "
                  f"depth_occluded={depth_rej} ({pct:.1f}% of considered)  "
                  f"{bins_status}")

        except Exception as e:
            err_row = {
                "session_name": session_name,
                "processing_session_dir": str(processing_session_dir),
                "status": "error",
                "error": str(e),
                "labeled_points": 0,
                "rej_depth_occluded": 0,
                "depth_available": False,
                "depth_occlusion_m": args.depth_occlusion_m,
                "depth_patch_r": args.depth_patch_r,
            }
            batch_rows.append(err_row)
            print(f"[ERROR] {e}")
            if args.stop_on_error:
                raise

    summary_df = pd.DataFrame(batch_rows)
    summary_df.to_csv(summary_csv, index=False)

    ok_count = int((summary_df.get("status", pd.Series(["ok"] * len(summary_df))) == "ok").sum())
    err_count = int((summary_df.get("status", pd.Series()) == "error").sum())
    total_labeled = int(summary_df.get("labeled_points", pd.Series([0])).fillna(0).sum())
    total_depth_rej = int(summary_df.get("rej_depth_occluded", pd.Series([0])).fillna(0).sum())

    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    sessions_with_bins = int(
        summary_df.get("range_bin_present", pd.Series([False] * len(summary_df)))
        .fillna(False).astype(bool).sum()
    )
    print(f"Sessions processed  : {len(summary_df)}")
    print(f"Succeeded           : {ok_count}")
    print(f"Failed              : {err_count}")
    print(f"Total labeled pts   : {total_labeled}")
    print(f"Depth-occluded rej  : {total_depth_rej}  (foreground-misclassification gate)")
    print(f"Sessions w/ bins    : {sessions_with_bins}/{ok_count}  (range_bin+doppler_bin in output)")
    if sessions_with_bins < ok_count:
        print(f"  WARNING: {ok_count - sessions_with_bins} session(s) missing bin columns —")
        print(f"           invalid for the active April-28-only RD patch pipeline.")
        print(f"           Regenerate radar CSVs with 3branch_adc_to_pointcloud.py to preserve native bins.")
    print(f"Summary CSV         : {summary_csv}")


if __name__ == "__main__":
    main()
