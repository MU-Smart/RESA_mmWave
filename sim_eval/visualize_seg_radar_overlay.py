#!/usr/bin/env python3
"""
visualize_seg_radar_overlay.py
==============================

Three-panel diagnostic overlay for inspecting radar point projection quality
and label assignment correctness.

Panels (side-by-side MP4):
  LEFT:   Color video frame + radar points (fill = assigned bucket)
  CENTER: OneFormer segmentation overlay + same radar points
  RIGHT:  Depth frame + same radar points  [optional, skipped if unavailable]

Each radar point is drawn with:
  - Fill color  = assigned bucket label (what went into labeled_radar_points_v4.csv)
  - Ring color  = bucket inferred from seg map at that pixel (what the image says)
  - Ring is white when they agree, red when they disagree

This directly answers: "Is the projection landing in the right segment,
and is the label assignment logic working correctly?"

Modes (--mode):
  precomputed  [default]  Use (u, v) already in labeled CSV — exact pixels used
                          during autolabeling. Best for auditing label quality.
  reproject               Re-project XYZ through extrinsics + intrinsics. Use
                          this when you want to check alignment with different
                          calibration parameters.

Session directory structure expected:
  session_dir/
    session_*_color.mp4
    session_*_color_timestamps.csv
    seg/{frame_idx:06d}.npy          (OneFormer per-pixel ADE class ID maps)
    seg_meta.json                    (id2label, written by export_labelmaps)
    labeled_radar_points_v4.csv      (u, v, bucket, ade_id, ade_name, cam_z)
    meta_data.json                   (camera intrinsics — for reproject mode)
    depth/{frame_idx:06d}.npy        [optional, uint16 mm]
    session_*_depth_vis.mp4          [optional fallback]

Usage:
  # Default: precomputed mode, 2-panel (color + seg)
  python visualize_seg_radar_overlay.py \\
    --session-dir LLM_ML/data/data-4-30/Processing/session_2026-04-28_17-53-50 \\
    --out-mp4 /tmp/seg_overlay_17-53-50.mp4

  # With depth panel, limited to frames 0-200
  python visualize_seg_radar_overlay.py \\
    --session-dir LLM_ML/data/data-4-30/Processing/session_2026-04-28_17-53-50 \\
    --out-mp4 /tmp/seg_overlay_17-53-50.mp4 \\
    --with-depth \\
    --start-frame 0 --end-frame 200

  # Reproject mode with custom extrinsics
  python visualize_seg_radar_overlay.py \\
    --session-dir LLM_ML/data/data-4-30/Processing/session_2026-04-28_17-53-50 \\
    --out-mp4 /tmp/seg_overlay_reproject.mp4 \\
    --mode reproject \\
    --extrinsics LLM_ML/jetson_nav_pipeline/config/radar_camera_extrinsics.json

  # Batch: all sessions in a Processing root
  python visualize_seg_radar_overlay.py \\
    --processing-root LLM_ML/data/data-4-30/Processing \\
    --out-dir /tmp/seg_overlays \\
    --session-prefix session_2026-04-28_17-
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Bucket colour map  (BGR)
# ---------------------------------------------------------------------------

BUCKET_BGR: Dict[str, Tuple[int, int, int]] = {
    "human":     (40,  40,  255),   # red
    "person":    (40,  40,  255),
    "structure": (255, 200,  40),   # amber/cyan
    "wall":      (255, 200,  40),
    "door":      (255, 200,  40),
    "pillar":    (255, 200,  40),
    "box_like":  (255, 200,  40),
    "floor":     (80,  220,  80),   # green
    "unknown":   (180, 180, 180),   # grey
}
DISAGREE_RING = (0,   0,  255)   # red ring = label mismatch
AGREE_RING    = (255, 255, 255)  # white ring = label matches seg

ADE_HUMAN_NAMES = {"person", "people", "human", "man", "woman", "child",
                   "pedestrian", "rider"}
ADE_FLOOR_NAMES = {"floor", "flooring", "rug", "carpet", "mat", "ground",
                   "pavement", "sidewalk", "path", "road"}
ADE_STRUCT_NAMES = {"wall", "door", "doorframe", "pillar", "column",
                    "ceiling", "cabinet", "shelf", "window", "fence",
                    "railing", "stairs", "step", "box", "bag", "sofa",
                    "table", "desk", "chair", "book", "monitor", "counter"}


def _bucket_color(bucket: str) -> Tuple[int, int, int]:
    return BUCKET_BGR.get(str(bucket).lower(), BUCKET_BGR["unknown"])


def _ade_name_to_bucket(ade_name: str) -> str:
    n = str(ade_name).lower().strip()
    if any(h in n for h in ADE_HUMAN_NAMES):
        return "human"
    if any(f in n for f in ADE_FLOOR_NAMES):
        return "floor"
    if any(s in n for s in ADE_STRUCT_NAMES):
        return "structure"
    return "unknown"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _find_first(directory: Path, *patterns: str) -> Optional[Path]:
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            return matches[0]
    return None


def _require(path: Optional[Path], desc: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Required file not found: {desc} (looked in {path})")
    return path


def _load_timestamps(ts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ts_path)
    if "frame_index" not in df.columns:
        df["frame_index"] = np.arange(len(df))
    for col in ("timestamp_ms", "timestamp", "timestamp_us"):
        if col in df.columns:
            df["timestamp_ms"] = df[col].astype(float) * (1.0 if col != "timestamp_us" else 1e-3)
            break
    else:
        raise ValueError(f"{ts_path} has no timestamp column")
    return df[["frame_index", "timestamp_ms"]].astype({"frame_index": int, "timestamp_ms": float})


def _nearest_depth_map(color_ts: pd.DataFrame, depth_ts: pd.DataFrame,
                       max_dt_ms: float = 250.0) -> Dict[int, Optional[int]]:
    d_times  = depth_ts["timestamp_ms"].to_numpy()
    d_frames = depth_ts["frame_index"].to_numpy()
    order    = np.argsort(d_times)
    d_times, d_frames = d_times[order], d_frames[order]

    out: Dict[int, Optional[int]] = {}
    for _, row in color_ts.iterrows():
        cf, ct = int(row["frame_index"]), float(row["timestamp_ms"])
        j = int(np.searchsorted(d_times, ct))
        cands = [k for k in (j-1, j, j+1) if 0 <= k < len(d_times)]
        if not cands:
            out[cf] = None
            continue
        best = min(cands, key=lambda k: abs(d_times[k] - ct))
        out[cf] = int(d_frames[best]) if abs(d_times[best] - ct) <= max_dt_ms else None
    return out


# ---------------------------------------------------------------------------
# Segmentation coloriser
# ---------------------------------------------------------------------------

class SegColouriser:
    """Converts ADE20K class-ID maps to BGR colour images."""

    def __init__(self, id2label: Dict[str, str]):
        # id2label keys are strings ("0", "1", ...)
        self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        self._cache: Dict[int, np.ndarray] = {}   # seg_id -> BGR colour

    def _colour_for_id(self, seg_id: int) -> np.ndarray:
        if seg_id not in self._cache:
            name = self.id2label.get(seg_id, "")
            bucket = _ade_name_to_bucket(name)
            self._cache[seg_id] = np.array(_bucket_color(bucket), dtype=np.uint8)
        return self._cache[seg_id]

    def render(self, seg_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Return a solid BGR image of the seg map coloured by bucket."""
        H, W = seg_map.shape
        out = np.zeros((H, W, 3), dtype=np.uint8)
        for seg_id in np.unique(seg_map):
            out[seg_map == seg_id] = self._colour_for_id(int(seg_id))
        return out

    def bucket_at_pixel(self, seg_map: np.ndarray, u: int, v: int) -> str:
        if 0 <= v < seg_map.shape[0] and 0 <= u < seg_map.shape[1]:
            seg_id = int(seg_map[v, u])
            name = self.id2label.get(seg_id, "")
            return _ade_name_to_bucket(name)
        return "unknown"


def _load_seg_map(seg_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    p = seg_dir / f"{frame_idx:06d}.npy"
    if not p.exists():
        return None
    arr = np.load(str(p))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int32)


# ---------------------------------------------------------------------------
# Reprojection helpers (used only in --mode reproject)
# ---------------------------------------------------------------------------

def _load_intrinsics(meta_path: Path, W: int, H: int
                     ) -> Tuple[float, float, float, float, Optional[np.ndarray]]:
    """Returns (fx, fy, cx, cy, dist_coeffs_or_None)."""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # Handle realsense_calibration.rgb nesting (from prepare_autolabel.py)
        cal = meta.get("realsense_calibration", {})
        rgb = cal.get("rgb", {}) if isinstance(cal, dict) else {}
        if rgb:
            fx = float(rgb.get("fx", 0)) or 0.9 * W
            fy = float(rgb.get("fy", 0)) or fx
            cx = float(rgb.get("cx", W / 2))
            cy = float(rgb.get("cy", H / 2))
            dist = rgb.get("dist_coeffs")
            dist_arr = np.array(dist, dtype=np.float64) if dist else None
            return fx, fy, cx, cy, dist_arr
        for key in ("color_intrinsics", "rgb_intrinsics", "intrinsics"):
            if key in meta and isinstance(meta[key], dict):
                d = meta[key]
                fx = float(d.get("fx", 0)) or 0.9 * W
                fy = float(d.get("fy", 0)) or fx
                cx = float(d.get("ppx", d.get("cx", W / 2)))
                cy = float(d.get("ppy", d.get("cy", H / 2)))
                dist = d.get("dist_coeffs") or d.get("distortion_coefficients")
                dist_arr = np.array(dist, dtype=np.float64) if dist else None
                return fx, fy, cx, cy, dist_arr
        for fk, ck in (("fx","cx"), ("color_fx","color_cx")):
            if fk in meta:
                return (float(meta[fk]), float(meta.get("fy", meta[fk])),
                        float(meta.get(ck, W/2)), float(meta.get("cy", H/2)), None)
    return 0.9*W, 0.9*W, W/2.0, H/2.0, None


def _load_extrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    # Support both flat and nested JSON
    for d in ([data] + [v for v in data.values() if isinstance(v, dict)]):
        # 4x4 transform matrix
        for key in ("T", "transform", "T_radar_to_camera", "radar_to_camera"):
            if key in d:
                T = np.array(d[key], dtype=np.float64)
                if T.size == 16:
                    T = T.reshape(4, 4)
                if T.shape == (4, 4):
                    return T[:3, :3], T[:3, 3]
        # Separate R + t (handles R_radar_to_camera / t_radar_to_camera)
        R_val = next((d[k] for k in (
            "R_radar_to_camera", "R", "rotation", "rotation_matrix") if k in d), None)
        t_val = next((d[k] for k in (
            "t_radar_to_camera", "t", "translation", "translation_vector") if k in d), None)
        if R_val is not None and t_val is not None:
            R = np.array(R_val, dtype=np.float64).reshape(3, 3)
            t = np.array(t_val, dtype=np.float64).reshape(3)
            return R, t
    raise ValueError(f"Cannot parse extrinsics from {path}")


def _reproject(xyz: np.ndarray, R: np.ndarray, t: np.ndarray,
               fx: float, fy: float, cx: float, cy: float,
               W: int, H: int,
               dist_coeffs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project radar XYZ into image coordinates, optionally with lens distortion."""
    if len(xyz) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=bool)

    if dist_coeffs is not None:
        # Use OpenCV's full projection including distortion
        rvec, _ = cv2.Rodrigues(R)
        pts_img, _ = cv2.projectPoints(
            xyz.astype(np.float64), rvec, t.reshape(3, 1),
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
            dist_coeffs,
        )
        uv = pts_img.reshape(-1, 2)
        cam_z = (R @ xyz.T).T[:, 2]
        valid = (cam_z > 0.05) & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, valid

    cam = (R @ xyz.T).T + t
    Z = cam[:, 2]
    valid = Z > 0.05
    u = fx * cam[:, 0] / np.maximum(Z, 1e-6) + cx
    v = fy * cam[:, 1] / np.maximum(Z, 1e-6) + cy
    valid &= np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return np.stack([u, v], axis=1), valid


# ---------------------------------------------------------------------------
# Raw-point-cloud helpers (for --show-all-points)
# ---------------------------------------------------------------------------

def _load_sync_map(session_dir: Path) -> Dict[int, List[int]]:
    """Parse synchronized_session_*.csv → {video_frame_index: [radar_frame_nums]}."""
    sync_csv = _find_first(session_dir, "synchronized_*.csv")
    if sync_csv is None:
        return {}
    df = pd.read_csv(sync_csv)
    result: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        try:
            vfi = int(row["video_frame_index"])
        except (KeyError, ValueError):
            continue
        rnums_str = str(row.get("radar_frame_nums", "") or "")
        rnums = [int(x) for x in rnums_str.split(";") if x.strip().lstrip("-").isdigit()]
        result[vfi] = rnums
    return result


def _load_raw_radar_by_rframe(session_dir: Path) -> Dict[int, np.ndarray]:
    """Load raw session CSV (the big one) → {radar_frame_num: xyz (N,3)}."""
    # session_2026-*.csv but NOT synchronized_* or labeled_*
    raw_csv = None
    for p in session_dir.glob("session_*.csv"):
        if "synchronized" not in p.name and "labeled" not in p.name:
            raw_csv = p
            break
    if raw_csv is None:
        return {}
    try:
        df = pd.read_csv(raw_csv, usecols=["radar_frame_num", "x", "y", "z"])
    except (ValueError, KeyError):
        return {}
    result: Dict[int, np.ndarray] = {}
    for rfnum, grp in df.groupby("radar_frame_num"):
        result[int(rfnum)] = grp[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return result


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], scale: float = 0.5) -> None:
    x, y = xy
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 1, cv2.LINE_AA)


def _draw_legend(img: np.ndarray) -> None:
    H = img.shape[0]
    entries = [
        ("human",     BUCKET_BGR["human"]),
        ("structure", BUCKET_BGR["structure"]),
        ("floor",     BUCKET_BGR["floor"]),
        ("unknown",   BUCKET_BGR["unknown"]),
    ]
    y0 = H - 90
    cv2.rectangle(img, (8, y0 - 18), (145, y0 + 14 + 20 * len(entries)), (0,0,0), -1)
    for i, (name, col) in enumerate(entries):
        yy = y0 + i * 20
        cv2.circle(img, (22, yy), 6, col, -1, cv2.LINE_AA)
        cv2.circle(img, (22, yy), 6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, name, (34, yy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (255,255,255), 1, cv2.LINE_AA)


RAW_PT_COLOR = (0, 220, 220)   # cyan — raw radar points not in labeled CSV


def render_frame(
    color_frame: np.ndarray,
    seg_frame: Optional[np.ndarray],      # colorised seg BGR, or None
    depth_frame: Optional[np.ndarray],    # BGR depth vis frame, or None
    points: pd.DataFrame,                 # rows with u, v, bucket (possibly seg_bucket_at_pixel)
    seg_colouriser: Optional[SegColouriser],
    seg_map: Optional[np.ndarray],        # raw seg map for per-point lookup
    frame_idx: int,
    point_radius: int = 5,
    seg_alpha: float = 0.55,
    raw_pts_uv: Optional[np.ndarray] = None,   # (N,2) all-raw projected points
) -> np.ndarray:
    H, W = color_frame.shape[:2]

    color_out = color_frame.copy()

    # Blend seg overlay
    if seg_frame is not None:
        seg_rs = cv2.resize(seg_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_out = cv2.addWeighted(color_frame, 1 - seg_alpha, seg_rs, seg_alpha, 0)
    else:
        seg_out = np.full_like(color_frame, 60)

    depth_out = depth_frame.copy() if depth_frame is not None else None

    # Draw raw (unfiltered) points as small cyan dots first (under labeled points)
    if raw_pts_uv is not None and len(raw_pts_uv) > 0:
        for uv in raw_pts_uv:
            u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if not (0 <= u < W and 0 <= v < H):
                continue
            cv2.circle(color_out, (u, v), 2, RAW_PT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(seg_out,   (u, v), 2, RAW_PT_COLOR, -1, cv2.LINE_AA)
            if depth_out is not None:
                cv2.circle(depth_out, (u, v), 2, RAW_PT_COLOR, -1, cv2.LINE_AA)

    n_agree = n_disagree = 0

    for _, row in points.iterrows():
        try:
            u = int(round(float(row["u"])))
            v = int(round(float(row["v"])))
        except Exception:
            continue
        if not (0 <= u < W and 0 <= v < H):
            continue

        assigned_bucket = str(row.get("bucket", "unknown")).lower()
        fill = _bucket_color(assigned_bucket)

        # Compare against seg map at this pixel
        if seg_map is not None and seg_colouriser is not None:
            seg_bucket = seg_colouriser.bucket_at_pixel(seg_map, u, v)
            ring = AGREE_RING if seg_bucket == assigned_bucket else DISAGREE_RING
            if seg_bucket == assigned_bucket:
                n_agree += 1
            else:
                n_disagree += 1
        else:
            ring = AGREE_RING

        # Draw on color panel
        cv2.circle(color_out, (u, v), point_radius,     ring,  1,  cv2.LINE_AA)
        cv2.circle(color_out, (u, v), point_radius - 1, fill, -1,  cv2.LINE_AA)

        # Draw on seg panel
        cv2.circle(seg_out,   (u, v), point_radius,     ring,  1,  cv2.LINE_AA)
        cv2.circle(seg_out,   (u, v), point_radius - 1, fill, -1,  cv2.LINE_AA)

        if depth_out is not None:
            cv2.circle(depth_out, (u, v), point_radius,     ring,  1,  cv2.LINE_AA)
            cv2.circle(depth_out, (u, v), point_radius - 1, fill, -1,  cv2.LINE_AA)

    total = len(points)
    n_raw = len(raw_pts_uv) if raw_pts_uv is not None else 0
    agree_pct = 100 * n_agree / max(total, 1)

    raw_info = f"  raw={n_raw}" if n_raw > 0 else ""
    _draw_label(color_out, f"Color  frame={frame_idx}  labeled={total}{raw_info}", (10, 24))
    _draw_legend(color_out)
    if n_raw > 0:
        _draw_label(color_out, "cyan=raw(unfiltered)  fill=labeled bucket", (10, 46), scale=0.4)

    seg_txt = f"Seg overlay  agree={agree_pct:.0f}%  mismatch={n_disagree}"
    _draw_label(seg_out, seg_txt, (10, 24))
    _draw_label(seg_out, "fill=assigned  ring=seg(white=ok red=bad)", (10, 46), scale=0.4)

    panels = [color_out, seg_out]
    if depth_out is not None:
        _draw_label(depth_out, f"Depth  frame={frame_idx}", (10, 24))
        panels.append(depth_out)

    # Ensure all panels are same size
    panels = [cv2.resize(p, (W, H)) for p in panels]
    return np.concatenate(panels, axis=1)


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(
    session_dir: Path,
    out_mp4: Path,
    *,
    mode: str = "precomputed",
    with_depth: bool = False,
    show_all_points: bool = False,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    point_radius: int = 5,
    seg_alpha: float = 0.55,
    fps: float = 15.0,
    label_csv_name: str = "labeled_radar_points_v4.csv",
    extrinsics_path: Optional[Path] = None,
    max_depth_dt_ms: float = 250.0,
) -> Dict:
    session_dir = session_dir.resolve()

    # --- required inputs ---
    color_mp4 = _find_first(session_dir, "*_color.mp4", "*.mp4")
    _require(color_mp4, f"color MP4 in {session_dir}")

    label_csv = session_dir / label_csv_name
    _require(label_csv, label_csv_name)

    seg_dir = session_dir / "seg"
    seg_meta_path = session_dir / "seg_meta.json"
    meta_path = session_dir / "meta_data.json"

    # --- load labels ---
    labels = pd.read_csv(label_csv)
    labels["video_frame_index"] = pd.to_numeric(labels.get("video_frame_index",
                                                             labels.get("video_frame_idx", np.nan)),
                                                 errors="coerce")
    labels = labels.dropna(subset=["video_frame_index"]).copy()
    labels["video_frame_index"] = labels["video_frame_index"].astype(int)

    if mode == "precomputed":
        for col in ("u", "v"):
            if col not in labels.columns:
                raise ValueError(f"labeled CSV missing '{col}' column — use --mode reproject")
        labels["u"] = pd.to_numeric(labels["u"], errors="coerce")
        labels["v"] = pd.to_numeric(labels["v"], errors="coerce")
        labels = labels.dropna(subset=["u", "v"]).reset_index(drop=True)

    by_frame: Dict[int, pd.DataFrame] = {
        k: g.reset_index(drop=True) for k, g in labels.groupby("video_frame_index")
    }

    # --- seg coloriser ---
    seg_colouriser: Optional[SegColouriser] = None
    if seg_dir.is_dir() and seg_meta_path.exists():
        id2label = json.loads(seg_meta_path.read_text()).get("id2label", {})
        seg_colouriser = SegColouriser(id2label)
    else:
        print(f"[WARN] No seg/ dir or seg_meta.json in {session_dir.name} — seg panel will be blank")

    # --- depth setup ---
    depth_npy_dir = session_dir / "depth"
    depth_vis_mp4_path = _find_first(session_dir, "*_depth_vis.mp4")
    depth_ts_map: Dict[int, Optional[int]] = {}
    depth_vis_cap: Optional[cv2.VideoCapture] = None

    if with_depth:
        if not depth_npy_dir.is_dir() and depth_vis_mp4_path is None:
            print(f"[WARN] --with-depth set but no depth/ or depth_vis MP4 found; skipping depth panel")
            with_depth = False
        elif depth_npy_dir.is_dir():
            pass  # will load per-frame .npy
        elif depth_vis_mp4_path is not None:
            color_ts_path = _find_first(session_dir, "*_color_timestamps.csv")
            depth_ts_path = _find_first(session_dir, "*_depth_timestamps.csv")
            if color_ts_path and depth_ts_path:
                color_ts = _load_timestamps(color_ts_path)
                depth_ts = _load_timestamps(depth_ts_path)
                depth_ts_map = _nearest_depth_map(color_ts, depth_ts, max_depth_dt_ms)
            depth_vis_cap = cv2.VideoCapture(str(depth_vis_mp4_path))

    # --- extrinsics + intrinsics (needed for show_all_points or reproject mode) ---
    R = t = None
    fx = fy = cx = cy = None
    dist_coeffs: Optional[np.ndarray] = None

    needs_reproject = (mode == "reproject") or show_all_points
    if needs_reproject:
        if extrinsics_path is None:
            default = Path(__file__).resolve().parents[1] / "config" / "radar_camera_extrinsics.json"
            extrinsics_path = default if default.exists() else None
        if extrinsics_path is None:
            if show_all_points:
                print(f"[WARN] --show-all-points requires extrinsics; pass --extrinsics or put "
                      f"radar_camera_extrinsics.json in config/. Skipping raw points.")
                show_all_points = False
            else:
                raise ValueError("--mode reproject requires --extrinsics path")
        if extrinsics_path is not None:
            R, t = _load_extrinsics(extrinsics_path)

    # --- open video (needed for frame dimensions before loading intrinsics) ---
    cap = cv2.VideoCapture(str(color_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {color_mp4}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or fps
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if needs_reproject and R is not None:
        fx, fy, cx, cy, dist_coeffs = _load_intrinsics(meta_path, frame_w, frame_h)

    # --- raw point cloud setup (--show-all-points) ---
    raw_by_rframe: Dict[int, np.ndarray] = {}
    sync_map: Dict[int, List[int]] = {}
    if show_all_points:
        print(f"  Loading raw radar CSV for {session_dir.name}...")
        raw_by_rframe = _load_raw_radar_by_rframe(session_dir)
        sync_map = _load_sync_map(session_dir)
        if not raw_by_rframe:
            print(f"  [WARN] No raw session CSV found — --show-all-points disabled")
            show_all_points = False
        else:
            print(f"  Loaded {sum(len(v) for v in raw_by_rframe.values())} raw pts "
                  f"across {len(raw_by_rframe)} radar frames")

    end = (total_frames - 1) if end_frame is None else min(end_frame, total_frames - 1)
    if end < start_frame:
        end = total_frames - 1

    n_panels = 3 if with_depth else 2
    out_w = frame_w * n_panels

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output writer: {out_mp4}")

    stats = {"frames": 0, "agree": 0, "disagree": 0, "no_seg": 0}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_idx in tqdm(range(start_frame, end + 1), desc=session_dir.name, leave=False):
        ok, color = cap.read()
        if not ok:
            break

        pts = by_frame.get(frame_idx, pd.DataFrame())

        # Reproject XYZ if needed (labeled points)
        if mode == "reproject" and len(pts) > 0 and R is not None:
            xyz = pts[["x", "y", "z"]].to_numpy(dtype=np.float64)
            uv, valid = _reproject(xyz, R, t, fx, fy, cx, cy, frame_w, frame_h, dist_coeffs)
            pts = pts.copy()
            pts["u"] = uv[:, 0]
            pts["v"] = uv[:, 1]
            pts = pts[valid].reset_index(drop=True)

        # Build raw (all-radar) point projections for this video frame
        raw_pts_uv: Optional[np.ndarray] = None
        if show_all_points and R is not None:
            rframe_nums = sync_map.get(frame_idx, [])
            xyz_chunks = [raw_by_rframe[rf] for rf in rframe_nums if rf in raw_by_rframe]
            if xyz_chunks:
                xyz_all = np.vstack(xyz_chunks)
                uv_all, valid_all = _reproject(
                    xyz_all, R, t, fx, fy, cx, cy, frame_w, frame_h, dist_coeffs)
                raw_pts_uv = uv_all[valid_all]

        # Load seg map
        seg_map: Optional[np.ndarray] = None
        seg_frame: Optional[np.ndarray] = None
        if seg_colouriser is not None and seg_dir.is_dir():
            seg_map = _load_seg_map(seg_dir, frame_idx)
            if seg_map is not None:
                seg_frame = seg_colouriser.render(seg_map, alpha=1.0)
                if seg_frame.shape[:2] != (frame_h, frame_w):
                    seg_frame = cv2.resize(seg_frame, (frame_w, frame_h),
                                           interpolation=cv2.INTER_NEAREST)
                    seg_map_rs = cv2.resize(seg_map.astype(np.float32), (frame_w, frame_h),
                                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    seg_map = seg_map_rs

        # Load depth frame
        depth_frame: Optional[np.ndarray] = None
        if with_depth:
            if depth_npy_dir.is_dir():
                dp = depth_npy_dir / f"{frame_idx:06d}.npy"
                if dp.exists():
                    raw = np.load(str(dp)).astype(np.float32)
                    # uint16 mm → normalise for display
                    raw = np.clip(raw / 5000.0 * 255, 0, 255).astype(np.uint8)
                    depth_frame = cv2.cvtColor(cv2.applyColorMap(raw, cv2.COLORMAP_TURBO),
                                               cv2.COLOR_BGR2BGR)
            elif depth_vis_cap is not None:
                di = depth_ts_map.get(frame_idx)
                if di is not None:
                    depth_vis_cap.set(cv2.CAP_PROP_POS_FRAMES, di)
                    ok_d, df = depth_vis_cap.read()
                    if ok_d:
                        depth_frame = df

        composed = render_frame(
            color_frame=color,
            seg_frame=seg_frame,
            depth_frame=depth_frame if with_depth else None,
            points=pts,
            seg_colouriser=seg_colouriser,
            seg_map=seg_map,
            frame_idx=frame_idx,
            point_radius=point_radius,
            seg_alpha=seg_alpha,
            raw_pts_uv=raw_pts_uv,
        )

        writer.write(composed)
        stats["frames"] += 1
        if seg_map is None:
            stats["no_seg"] += 1

    cap.release()
    writer.release()
    if depth_vis_cap is not None:
        depth_vis_cap.release()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Seg + radar overlay diagnostic visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Session target
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--session-dir", type=Path,
                     help="Single session directory")
    grp.add_argument("--processing-root", type=Path,
                     help="Root containing multiple session_* directories (batch mode)")

    # Output
    ap.add_argument("--out-mp4", type=Path, default=None,
                    help="Output MP4 path (single-session mode)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory for batch mode")

    # Mode
    ap.add_argument("--mode", choices=["precomputed", "reproject"], default="precomputed",
                    help="precomputed: use (u,v) from labeled CSV; reproject: project XYZ via extrinsics")
    ap.add_argument("--extrinsics", type=Path, default=None,
                    help="Extrinsics JSON (required for --mode reproject)")

    # Display
    ap.add_argument("--with-depth", action="store_true",
                    help="Add depth panel (requires depth/ npy dir or depth_vis MP4)")
    ap.add_argument("--show-all-points", action="store_true",
                    help="Overlay ALL raw radar points (cyan) before labeled points; "
                         "needs raw session CSV and extrinsics")
    ap.add_argument("--seg-alpha", type=float, default=0.55,
                    help="Blend weight for seg overlay on color frame (0=invisible, 1=opaque)")
    ap.add_argument("--point-radius", type=int, default=5)
    ap.add_argument("--fps", type=float, default=15.0)

    # Frame range
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)

    # Data
    ap.add_argument("--label-csv-name", default="labeled_radar_points_v4.csv")
    ap.add_argument("--session-prefix", default="session_",
                    help="Filter sessions by prefix in batch mode")

    args = ap.parse_args()

    common_kw = dict(
        mode=args.mode,
        with_depth=args.with_depth,
        show_all_points=args.show_all_points,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        point_radius=args.point_radius,
        seg_alpha=args.seg_alpha,
        fps=args.fps,
        label_csv_name=args.label_csv_name,
        extrinsics_path=args.extrinsics,
    )

    if args.session_dir:
        out = args.out_mp4
        if out is None:
            out = args.session_dir / f"{args.session_dir.name}_seg_radar_overlay.mp4"
        stats = process_session(args.session_dir, out, **common_kw)
        print(f"\nDone: {out}")
        print(f"Frames={stats['frames']}  no_seg={stats['no_seg']}")

    else:
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = args.processing_root / "seg_radar_overlays"
        out_dir.mkdir(parents=True, exist_ok=True)

        sessions = sorted(
            p for p in args.processing_root.iterdir()
            if p.is_dir() and p.name.startswith(args.session_prefix)
        )
        print(f"Batch mode: {len(sessions)} sessions → {out_dir}")

        for s in sessions:
            out_mp4 = out_dir / f"{s.name}_seg_radar_overlay.mp4"
            try:
                stats = process_session(s, out_mp4, **common_kw)
                print(f"  {s.name}: frames={stats['frames']} no_seg={stats['no_seg']}  → {out_mp4.name}")
            except Exception as e:
                print(f"  {s.name}: SKIP — {e}")


if __name__ == "__main__":
    main()
