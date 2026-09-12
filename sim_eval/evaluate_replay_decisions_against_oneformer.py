#!/usr/bin/env python3
"""
evaluate_replay_decisions_against_oneformer.py
==============================================

Compare radar-only replay navigation decisions against a synchronized
OneFormer + depth camera-reference decision.

Design goal
-----------
Keep branch3_replay_simulation.py as the radar-only inference producer.
This script is a separate evaluator:

    replay frame_preds.csv
      + OneFormer seg/*.npy
      + RealSense depth/*.npy
      + synchronized_*.csv
      + scene_pipeline.aggregate_scene / compute_nav_decision
    -> decision-level metrics

It does NOT segment replay videos. It consumes the existing OneFormer label maps
that were generated during the labeling pipeline.

Typical usage
-------------
Run replay first from the canonical pipeline folder, for example:

    python branch3_replay_simulation.py ^
      --input-mode model ^
      --sessions-dir path/to/Processing ^
      --model-pt models/3branch_best_model.pt ^
      --output replay_decision_eval ^
      --export-frame-predictions replay_decision_eval/frame_preds.csv ^
      --export-point-predictions replay_decision_eval/point_preds.csv

Then evaluate decisions:

    python evaluate_replay_decisions_against_oneformer.py ^
      --frame-preds replay_decision_eval/frame_preds.csv ^
      --processing-root path/to/Processing ^
      --sync-root path/to/Synchronized ^
      --extrinsics-json config/radar_camera_extrinsics.json ^
      --out-dir replay_decision_eval/oneformer_decision_eval

Optional: pass --metadata-json only for neutral grouping fields such as
building, split/type, environment notes, and tags. This evaluator does not use
old Policy-B label-remapping metadata.

Outputs
-------
    decision_eval_report.json
    per_frame_decision_eval.csv
    per_session_decision_eval.csv
    per_building_decision_eval.csv
    decision_confusion_matrix.csv
    failure_cases.csv

Notes
-----
The camera reference is an automatic RGB-D semantic reference, not perfect
human ground truth. By default camera humans have doppler=0.0, so exact
"approaching_person -> stop" agreement is not expected unless you choose
--camera-human-motion approaching or add a future motion estimator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Constants and class mapping
# =============================================================================

ACTION_ORDER = ["continue_straight", "slow_down", "stop", "turn_left", "turn_right", "unknown"]

# Broad ADE/OneFormer names that should be treated as blocking structure for
# navigation-decision evaluation. This is intentionally conservative.
STRUCTURE_TOKENS = (
    "wall", "door", "pillar", "column", "stair", "staircase", "railing", "handrail",
    "chair", "seat", "table", "desk", "counter", "cabinet", "shelf", "sofa", "bench",
    "booth", "box", "cart", "trash", "garbage", "bin", "window", "glass", "partition",
    "bookcase", "refrigerator", "plant", "screen", "monitor", "sign", "fence",
)
FLOOR_TOKENS = ("floor", "ground", "rug", "carpet", "mat")
HUMAN_TOKENS = ("person", "human", "man", "woman", "boy", "girl", "people")


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def normalize_action(primary_action: Any, best_direction: Any = None) -> str:
    """Normalize nav decision action for a clean confusion matrix."""
    a = str(primary_action or "unknown").strip().lower()
    d = str(best_direction or "").strip().lower()
    if a in {"turn_away", "veer", "veer_away", "turn"}:
        if d in {"left", "turn_left", "veer_left"}:
            return "turn_left"
        if d in {"right", "turn_right", "veer_right"}:
            return "turn_right"
        return "stop"  # blocked but no reliable side: conservative stop
    if a in {"continue", "continue_straight", "clear_path"}:
        return "continue_straight"
    if a in {"slow", "slow_down", "caution"}:
        return "slow_down"
    if a in {"stop", "halt"}:
        return "stop"
    if a in {"turn_left", "veer_left"}:
        return "turn_left"
    if a in {"turn_right", "veer_right"}:
        return "turn_right"
    return "unknown"


def compare_actions(expected: str, radar: str) -> dict:
    """Return exact/safety error labels for a camera-reference vs radar action."""
    expected = normalize_action(expected)
    radar = normalize_action(radar)
    exact = expected == radar
    wrong_direction = False
    unsafe = False
    conservative = False

    if exact:
        safe = True
    elif expected == "continue_straight":
        # Radar over-warns on a camera-clear frame: safe but conservative.
        safe = radar in {"slow_down", "stop", "turn_left", "turn_right"}
        conservative = safe
        unsafe = not safe
    elif expected == "slow_down":
        # Stop/turn are conservative; continue misses the hazard.
        safe = radar in {"slow_down", "stop", "turn_left", "turn_right"}
        conservative = radar in {"stop", "turn_left", "turn_right"}
        unsafe = radar == "continue_straight" or radar == "unknown"
    elif expected == "stop":
        # For a stop-reference frame, anything less than stop is treated unsafe.
        safe = radar == "stop"
        unsafe = not safe
    elif expected in {"turn_left", "turn_right"}:
        opposite = "turn_right" if expected == "turn_left" else "turn_left"
        if radar == "stop":
            safe = True
            conservative = True
        elif radar == expected:
            safe = True
        elif radar == opposite:
            safe = False
            unsafe = True
            wrong_direction = True
        else:
            safe = False
            unsafe = True
    else:
        safe = exact
        unsafe = not exact

    return {
        "exact_correct": bool(exact),
        "safe_correct": bool(safe),
        "unsafe_error": bool(unsafe),
        "conservative_error": bool(conservative),
        "wrong_direction_error": bool(wrong_direction),
    }


def session_hhmmss(session: str) -> str:
    m = re.search(r"session_\d{4}-\d{2}-\d{2}[_-](\d{2})[-_](\d{2})[-_](\d{2})", str(session))
    if m:
        return "".join(m.groups())
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", str(session))
    return m.group(1) if m else ""


# =============================================================================
# Metadata
# =============================================================================


def load_session_metadata(path: Optional[Path]) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sessions = data.get("sessions", data)
    out: dict[str, dict] = {}
    for k, v in sessions.items():
        kk = str(k).zfill(6)
        row = dict(v)
        row.setdefault("building", "unknown")
        row.setdefault("type", "unknown")
        row.setdefault("tags", [])
        out[kk] = row
    return out


def metadata_for_session(session: str, metadata: dict) -> dict:
    key = session_hhmmss(session)
    row = metadata.get(key, {})
    return {
        "session_hhmmss": key,
        "metadata_matched": bool(row),
        "building": row.get("building", "unknown"),
        "metadata_type": row.get("type", row.get("metadata_type", "unknown")),
        "env_type": row.get("env_type", ""),
        "rig_speed": row.get("rig_speed", ""),
        "turning": row.get("turning", ""),
        "obs": row.get("obs", ""),
        "tags": "|".join(row.get("tags", [])) if isinstance(row.get("tags", []), list) else str(row.get("tags", "")),
    }


# =============================================================================
# Scene pipeline import
# =============================================================================


def resolve_pipeline_dir(pipeline_dir: Optional[Path]) -> Path:
    """Find the canonical folder that contains scene_pipeline.py.

    The evaluator may live directly in canon/ or inside canon/tools/.  If the
    user does not pass --pipeline-dir, try the script folder, its parent, and
    the current working directory before failing.
    """
    if pipeline_dir is not None:
        return Path(pipeline_dir)
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, Path.cwd(), Path.cwd().parent]
    for c in candidates:
        if (c / "scene_pipeline.py").exists():
            return c
    return here


def load_scene_pipeline(pipeline_dir: Path):
    scene_path = Path(pipeline_dir) / "scene_pipeline.py"
    if not scene_path.exists():
        raise FileNotFoundError(
            f"Could not find scene_pipeline.py at {scene_path}. "
            "Pass --pipeline-dir pointing to the canonical pipeline folder."
        )
    spec = importlib.util.spec_from_file_location("scene_pipeline_eval", str(scene_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scene_pipeline_eval"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Session/sync/seg/depth utilities
# =============================================================================


def find_session_dir(processing_root: Path, session: str) -> Path:
    d = processing_root / session
    if d.exists():
        return d
    candidates = list(processing_root.rglob(session))
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Could not find session directory for {session} under {processing_root}")


def find_sync_csv(sync_root: Optional[Path], session_dir: Path, session: str) -> Optional[Path]:
    roots: list[Path] = []
    if sync_root:
        roots.extend([sync_root / session, sync_root])
    roots.extend([session_dir, session_dir.parent, session_dir.parent.parent if session_dir.parent else session_dir])
    patterns = ["synchronized_*.csv", "*synchronized*.csv", "*_sync*.csv"]
    for root in roots:
        if not root or not root.exists():
            continue
        for pat in patterns:
            matches = sorted(root.glob(pat))
            if matches:
                return matches[0]
        # one level below, common Synchronized/session_* layout
        sub = root / session
        if sub.exists():
            for pat in patterns:
                matches = sorted(sub.glob(pat))
                if matches:
                    return matches[0]
    return None


def choose_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    # fuzzy contains fallback
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


class SyncMapper:
    def __init__(self, sync_csv: Optional[Path]) -> None:
        self.sync_csv = sync_csv
        self.df: Optional[pd.DataFrame] = None
        self.radar_col: Optional[str] = None
        self.video_col: Optional[str] = None
        if sync_csv and sync_csv.exists():
            self.df = pd.read_csv(sync_csv)
            self.radar_col = choose_col(
                self.df.columns,
                ["radar_frame_num", "radar_frame", "frame_num", "frame", "radar_idx", "radar_index"],
            )
            self.video_col = choose_col(
                self.df.columns,
                [
                    "video_frame_index", "color_frame_index", "camera_frame_index",
                    "rgb_frame_index", "frame_index", "color_idx", "video_idx",
                ],
            )
            if self.radar_col is not None:
                self.df[self.radar_col] = pd.to_numeric(self.df[self.radar_col], errors="coerce")
            if self.video_col is not None:
                self.df[self.video_col] = pd.to_numeric(self.df[self.video_col], errors="coerce")

    def video_index_for_radar_frame(self, radar_frame_num: int) -> Optional[int]:
        if self.df is None or self.radar_col is None or self.video_col is None:
            # Fallback: assume same frame index.
            return int(radar_frame_num) if radar_frame_num >= 0 else None
        valid = self.df[[self.radar_col, self.video_col]].dropna()
        if len(valid) == 0:
            return int(radar_frame_num) if radar_frame_num >= 0 else None
        exact = valid[valid[self.radar_col].astype(int) == int(radar_frame_num)]
        if len(exact):
            return int(exact.iloc[0][self.video_col])
        diffs = np.abs(valid[self.radar_col].to_numpy(dtype=float) - float(radar_frame_num))
        j = int(np.argmin(diffs))
        if diffs[j] <= 1.0:
            return int(valid.iloc[j][self.video_col])
        return None


def candidate_frame_paths(session_dir: Path, subdirs: list[str], idx: int, suffixes: list[str]) -> Iterable[Path]:
    names = [f"{idx:06d}", f"{idx:05d}", f"{idx:04d}", str(idx)]
    for sd in subdirs:
        d = session_dir / sd
        if d.exists():
            for n in names:
                for s in suffixes:
                    yield d / f"{n}{s}"
    # fallback recursive search can be expensive but okay for a few misses
    for n in names[:2]:
        for s in suffixes:
            for p in session_dir.rglob(f"{n}{s}"):
                yield p


def find_seg_path(session_dir: Path, video_idx: int, seg_dir_name: str) -> Optional[Path]:
    subdirs = [seg_dir_name, "seg", "labelmaps", "label_maps", "segmentation", "oneformer"]
    for p in candidate_frame_paths(session_dir, subdirs, video_idx, [".npy"]):
        if p.exists():
            return p
    return None


def find_depth_path(session_dir: Path, video_idx: int, depth_dir_name: str) -> Optional[Path]:
    subdirs = [depth_dir_name, "depth", "frames_depth", "depth_frames", "aligned_depth"]
    # Include session-specific depth folders.
    for d in session_dir.glob("frames_depth*"):
        if d.is_dir():
            subdirs.append(d.name)
    for p in candidate_frame_paths(session_dir, subdirs, video_idx, [".npy"]):
        if p.exists():
            return p
    return None


def load_seg_meta(session_dir: Path) -> dict[int, str]:
    meta_path = session_dir / "seg_meta.json"
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    id2label = data.get("id2label") or data.get("id_to_label") or data.get("labels") or {}
    out: dict[int, str] = {}
    if isinstance(id2label, dict):
        for k, v in id2label.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    return out


def label_id_to_class(label_id: int, id2label: dict[int, str]) -> Optional[str]:
    name = id2label.get(int(label_id), str(label_id)).strip().lower()
    if any(tok in name for tok in HUMAN_TOKENS):
        return "human"
    if any(tok in name for tok in FLOOR_TOKENS):
        return "floor"
    if any(tok in name for tok in STRUCTURE_TOKENS):
        return "structure"
    return None


# =============================================================================
# Camera intrinsics/extrinsics
# =============================================================================


def _find_intrinsics_dict(meta: dict) -> Optional[dict]:
    for key in ["rgb_intrinsics", "color_intrinsics", "intrinsics", "camera_intrinsics"]:
        val = meta.get(key)
        if isinstance(val, dict):
            return val
    # Some recorders store top-level values.
    if all(k in meta for k in ["fx", "fy", "cx", "cy"]):
        return meta
    return None


def load_intrinsics(session_dir: Path, img_shape: Tuple[int, int], allow_approx: bool = False) -> dict:
    meta_path = session_dir / "meta_data.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    intr = _find_intrinsics_dict(meta)
    if intr:
        fx = _safe_float(intr.get("fx", intr.get("focal_x")), float("nan"))
        fy = _safe_float(intr.get("fy", intr.get("focal_y")), float("nan"))
        cx = _safe_float(intr.get("cx", intr.get("ppx")), float("nan"))
        cy = _safe_float(intr.get("cy", intr.get("ppy")), float("nan"))
        if all(np.isfinite([fx, fy, cx, cy])):
            return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "source": str(meta_path)}
    if not allow_approx:
        raise ValueError(f"Could not find RGB intrinsics in {meta_path}; pass --allow-approx-intrinsics to approximate.")
    h, w = img_shape
    f = float(max(h, w))
    return {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0, "source": "approx"}


def load_depth_scale(session_dir: Path, cli_scale: Optional[float]) -> float:
    if cli_scale is not None:
        return float(cli_scale)
    meta_path = session_dir / "meta_data.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ["depth_scale_m_per_unit", "depth_scale", "depth_unit_m"]:
                if k in meta:
                    v = float(meta[k])
                    return v
        except Exception:
            pass
    return 0.001


def _extract_matrix(data: dict, candidates: list[str]) -> Optional[np.ndarray]:
    for k in candidates:
        if k in data:
            arr = np.asarray(data[k], dtype=float)
            if arr.size == 9:
                return arr.reshape(3, 3)
    return None


def _extract_vector(data: dict, candidates: list[str]) -> Optional[np.ndarray]:
    for k in candidates:
        if k in data:
            arr = np.asarray(data[k], dtype=float).reshape(-1)
            if arr.size >= 3:
                return arr[:3].astype(float)
    return None


def load_extrinsics(path: Optional[Path], direction: str) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Extrinsics JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    R = _extract_matrix(data, [
        "R", "rotation", "rotation_matrix", "radar_to_camera_R", "radar_to_camera_rotation",
        "camera_to_radar_R", "camera_to_radar_rotation",
    ])
    t = _extract_vector(data, [
        "t", "translation", "translation_m", "radar_to_camera_t", "radar_to_camera_translation",
        "camera_to_radar_t", "camera_to_radar_translation",
    ])
    # Nested common patterns
    if R is None or t is None:
        for key in ["radar_to_camera", "camera_to_radar", "extrinsics"]:
            val = data.get(key)
            if isinstance(val, dict):
                R = R if R is not None else _extract_matrix(val, ["R", "rotation", "rotation_matrix"])
                t = t if t is not None else _extract_vector(val, ["t", "translation", "translation_m"])
    if R is None or t is None:
        raise ValueError(f"Could not parse R/t from extrinsics JSON: {p}")
    return {"R": R.astype(float), "t": t.astype(float), "direction": direction, "source": str(p)}


def camera_to_radar_points(cam_xyz: np.ndarray, extr: Optional[dict], assume_aligned: bool) -> np.ndarray:
    if cam_xyz.size == 0:
        return cam_xyz.reshape(0, 3)
    if extr is None:
        if not assume_aligned:
            raise ValueError("No extrinsics provided. Pass --extrinsics-json or --assume-camera-radar-aligned.")
        # Approximate RealSense camera coords (x right, y down, z forward) to radar coords
        # (x right, y forward, z up).
        out = np.empty_like(cam_xyz, dtype=float)
        out[:, 0] = cam_xyz[:, 0]
        out[:, 1] = cam_xyz[:, 2]
        out[:, 2] = -cam_xyz[:, 1]
        return out
    R = extr["R"]
    t = extr["t"].reshape(1, 3)
    direction = extr.get("direction", "radar_to_camera")
    if direction == "radar_to_camera":
        # cam = R @ radar + t  -> radar = R.T @ (cam - t)
        return (R.T @ (cam_xyz - t).T).T
    if direction == "camera_to_radar":
        return (R @ cam_xyz.T).T + t
    raise ValueError(f"Unknown extrinsics direction: {direction}")


# =============================================================================
# Camera-reference scene construction
# =============================================================================


def sample_class_pixels(mask: np.ndarray, id2label: dict[int, str], *, sample_stride: int, human_sample_stride: int,
                        max_points_per_class: int) -> dict[str, np.ndarray]:
    class_masks: dict[str, np.ndarray] = {}
    for label_id in np.unique(mask):
        cls = label_id_to_class(int(label_id), id2label)
        if cls is None:
            continue
        m = mask == label_id
        if cls in class_masks:
            class_masks[cls] |= m
        else:
            class_masks[cls] = m.copy()

    out: dict[str, np.ndarray] = {}
    for cls, m in class_masks.items():
        coords = np.argwhere(m)  # rows: v, u
        if coords.size == 0:
            continue
        stride = human_sample_stride if cls == "human" else sample_stride
        stride = max(int(stride), 1)
        coords = coords[::stride]
        if len(coords) > max_points_per_class:
            idx = np.linspace(0, len(coords) - 1, max_points_per_class).astype(int)
            coords = coords[idx]
        out[cls] = coords.astype(np.int32)
    return out


def build_camera_reference_points(
    *,
    session_dir: Path,
    video_idx: int,
    scene_pipeline,
    extrinsics: Optional[dict],
    assume_aligned: bool,
    seg_dir_name: str,
    depth_dir_name: str,
    depth_scale_override: Optional[float],
    allow_approx_intrinsics: bool,
    sample_stride: int,
    human_sample_stride: int,
    max_points_per_class: int,
    min_depth_m: float,
    max_depth_m: float,
    camera_human_motion: str,
) -> tuple[list[dict], dict]:
    seg_path = find_seg_path(session_dir, video_idx, seg_dir_name)
    depth_path = find_depth_path(session_dir, video_idx, depth_dir_name)
    info = {
        "video_frame_index": video_idx,
        "seg_path": str(seg_path) if seg_path else "",
        "depth_path": str(depth_path) if depth_path else "",
        "camera_points": 0,
        "camera_structure_points": 0,
        "camera_floor_points": 0,
        "camera_human_points": 0,
        "camera_reference_status": "ok",
    }
    if seg_path is None:
        info["camera_reference_status"] = "missing_seg"
        return [], info
    if depth_path is None:
        info["camera_reference_status"] = "missing_depth"
        return [], info

    mask = np.load(seg_path)
    depth_raw = np.load(depth_path)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if depth_raw.ndim > 2:
        depth_raw = depth_raw.squeeze()
    if mask.ndim != 2 or depth_raw.ndim != 2:
        info["camera_reference_status"] = "bad_seg_or_depth_shape"
        return [], info

    id2label = load_seg_meta(session_dir)
    if not id2label:
        # Fallback: if no id2label exists, only meaningful when labels already match numeric aliases.
        info["camera_reference_status"] = "missing_seg_meta"
        return [], info

    intr = load_intrinsics(session_dir, mask.shape, allow_approx=allow_approx_intrinsics)
    depth_scale = load_depth_scale(session_dir, depth_scale_override)
    sampled = sample_class_pixels(
        mask,
        id2label,
        sample_stride=sample_stride,
        human_sample_stride=human_sample_stride,
        max_points_per_class=max_points_per_class,
    )

    points: list[dict] = []
    h_seg, w_seg = mask.shape
    h_d, w_d = depth_raw.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    for cls, coords_vu in sampled.items():
        if len(coords_vu) == 0:
            continue
        v = coords_vu[:, 0].astype(float)
        u = coords_vu[:, 1].astype(float)
        # Map segmentation pixels to depth image pixels when dimensions differ.
        ud = np.clip(np.round(u * (w_d / max(w_seg, 1))).astype(int), 0, w_d - 1)
        vd = np.clip(np.round(v * (h_d / max(h_seg, 1))).astype(int), 0, h_d - 1)
        depth_m = depth_raw[vd, ud].astype(float) * depth_scale
        valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
        if not np.any(valid):
            continue
        u_valid = u[valid]
        v_valid = v[valid]
        z_cam = depth_m[valid]
        x_cam = (u_valid - cx) * z_cam / fx
        y_cam = (v_valid - cy) * z_cam / fy
        cam_xyz = np.column_stack([x_cam, y_cam, z_cam]).astype(float)
        radar_xyz = camera_to_radar_points(cam_xyz, extrinsics, assume_aligned=assume_aligned)

        # Keep only navigation-relevant forward space.
        keep = (
            np.isfinite(radar_xyz).all(axis=1)
            & (radar_xyz[:, 1] >= 0.1)
            & (radar_xyz[:, 1] <= max_depth_m)
        )
        radar_xyz = radar_xyz[keep]
        if radar_xyz.size == 0:
            continue

        for x, y, z in radar_xyz:
            doppler = -0.5 if (cls == "human" and camera_human_motion == "approaching") else 0.0
            points.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "pred_class": cls,
                "doppler": float(doppler),
                "snr": 1.0,
                "spatiotemporal_weight": 1.0,
            })

    info["camera_points"] = len(points)
    info["camera_structure_points"] = sum(1 for p in points if p["pred_class"] == "structure")
    info["camera_floor_points"] = sum(1 for p in points if p["pred_class"] == "floor")
    info["camera_human_points"] = sum(1 for p in points if p["pred_class"] == "human")
    if len(points) == 0:
        info["camera_reference_status"] = "no_points_after_depth_filter"
    return points, info


# =============================================================================
# Metrics and outputs
# =============================================================================


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        n = len(g)
        row.update({
            "frames": int(n),
            "exact_correct": int(g["exact_correct"].sum()),
            "safe_correct": int(g["safe_correct"].sum()),
            "unsafe_errors": int(g["unsafe_error"].sum()),
            "conservative_errors": int(g["conservative_error"].sum()),
            "wrong_direction_errors": int(g["wrong_direction_error"].sum()),
            "exact_accuracy": float(g["exact_correct"].mean()) if n else float("nan"),
            "safe_accuracy": float(g["safe_correct"].mean()) if n else float("nan"),
            "unsafe_error_rate": float(g["unsafe_error"].mean()) if n else float("nan"),
            "conservative_error_rate": float(g["conservative_error"].mean()) if n else float("nan"),
            "wrong_direction_rate": float(g["wrong_direction_error"].mean()) if n else float("nan"),
        })
        for action in ACTION_ORDER:
            row[f"expected_{action}"] = int((g["camera_expected_action"] == action).sum())
            row[f"radar_{action}"] = int((g["radar_action"] == action).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def decision_confusion(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(0, index=ACTION_ORDER, columns=ACTION_ORDER)
    return pd.crosstab(
        pd.Categorical(df["camera_expected_action"], categories=ACTION_ORDER),
        pd.Categorical(df["radar_action"], categories=ACTION_ORDER),
        dropna=False,
    )


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# =============================================================================
# Main evaluation
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate radar replay decisions against OneFormer/depth camera-reference decisions.")
    ap.add_argument("--frame-preds", required=True, type=Path, help="CSV from branch3_replay_simulation.py --export-frame-predictions")
    ap.add_argument("--processing-root", required=True, type=Path, help="Processing root containing session_* folders")
    ap.add_argument("--sync-root", default=None, type=Path, help="Optional Synchronized root containing session_* synchronized_*.csv")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--pipeline-dir", default=None, type=Path, help="Canonical pipeline folder containing scene_pipeline.py; default auto-detects script dir/parent/cwd")
    ap.add_argument("--metadata-json", default=None, type=Path, help="Optional neutral session metadata for grouping only: building/type/env/tags. No label-policy fields are used.")
    ap.add_argument("--extrinsics-json", default=None, type=Path, help="Radar-camera extrinsics JSON. Default: try processing session/config paths; otherwise use --assume-camera-radar-aligned.")
    ap.add_argument("--extrinsics-direction", default="radar_to_camera", choices=["radar_to_camera", "camera_to_radar"])
    ap.add_argument("--assume-camera-radar-aligned", action="store_true", help="Fallback approximation: camera x->radar x, camera z->radar y, -camera y->radar z")
    ap.add_argument("--allow-approx-intrinsics", action="store_true")
    ap.add_argument("--seg-dir-name", default="seg")
    ap.add_argument("--depth-dir-name", default="depth")
    ap.add_argument("--depth-scale", default=None, type=float, help="Override depth scale in meters per unit; default from meta_data.json or 0.001")
    ap.add_argument("--min-depth-m", type=float, default=0.30)
    ap.add_argument("--max-depth-m", type=float, default=5.0)
    ap.add_argument("--sample-stride", type=int, default=16, help="Pixel sampling stride for structure/floor masks")
    ap.add_argument("--human-sample-stride", type=int, default=6, help="Pixel sampling stride for human masks")
    ap.add_argument("--max-points-per-class", type=int, default=2500)
    ap.add_argument("--camera-human-motion", default="stationary", choices=["stationary", "approaching"], help="Camera reference has no radar Doppler; default stationary means nearby humans usually map to slow_down rather than stop.")
    ap.add_argument("--limit-frames", type=int, default=0, help="Debug: only process first N frame-pred rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = resolve_pipeline_dir(args.pipeline_dir)
    scene_pipeline = load_scene_pipeline(pipeline_dir)
    metadata = load_session_metadata(args.metadata_json)

    # Load global extrinsics if provided. If not provided, try common locations near pipeline dir.
    extrinsics = None
    extr_path = args.extrinsics_json
    if extr_path is None:
        for candidate in [
            pipeline_dir / "config" / "radar_camera_extrinsics.json",
            pipeline_dir / "radar_camera_extrinsics.json",
            args.processing_root.parent / "calibration" / "results" / "radar_camera_extrinsics.json",
        ]:
            if candidate.exists():
                extr_path = candidate
                break
    if extr_path is not None and Path(extr_path).exists():
        extrinsics = load_extrinsics(Path(extr_path), args.extrinsics_direction)
    elif not args.assume_camera_radar_aligned:
        raise FileNotFoundError(
            "No extrinsics JSON found. Pass --extrinsics-json or use --assume-camera-radar-aligned for a rough debug run."
        )

    frame_df = pd.read_csv(args.frame_preds)
    if args.limit_frames and args.limit_frames > 0:
        frame_df = frame_df.head(args.limit_frames).copy()
    required = {"session", "frame_num", "primary_action"}
    missing = sorted(required - set(frame_df.columns))
    if missing:
        raise ValueError(f"Frame predictions CSV is missing required columns: {missing}")

    sync_cache: dict[str, SyncMapper] = {}
    session_dir_cache: dict[str, Path] = {}
    rows: list[dict] = []

    for idx, r in frame_df.iterrows():
        session = str(r["session"])
        frame_num = _safe_int(r.get("frame_num"), -1)
        md = metadata_for_session(session, metadata)
        try:
            if session not in session_dir_cache:
                session_dir_cache[session] = find_session_dir(args.processing_root, session)
            session_dir = session_dir_cache[session]
            if session not in sync_cache:
                sync_cache[session] = SyncMapper(find_sync_csv(args.sync_root, session_dir, session))
            video_idx = sync_cache[session].video_index_for_radar_frame(frame_num)
            if video_idx is None:
                raise RuntimeError(f"Could not map radar frame {frame_num} to video frame")

            cam_points, info = build_camera_reference_points(
                session_dir=session_dir,
                video_idx=video_idx,
                scene_pipeline=scene_pipeline,
                extrinsics=extrinsics,
                assume_aligned=args.assume_camera_radar_aligned,
                seg_dir_name=args.seg_dir_name,
                depth_dir_name=args.depth_dir_name,
                depth_scale_override=args.depth_scale,
                allow_approx_intrinsics=args.allow_approx_intrinsics,
                sample_stride=args.sample_stride,
                human_sample_stride=args.human_sample_stride,
                max_points_per_class=args.max_points_per_class,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                camera_human_motion=args.camera_human_motion,
            )
            if info.get("camera_reference_status") != "ok":
                # Still write a row so coverage problems are visible.
                camera_scene = scene_pipeline.aggregate_scene([], window_frames=1)
            else:
                camera_scene = scene_pipeline.aggregate_scene(cam_points, window_frames=1)
            camera_decision = scene_pipeline.compute_nav_decision(camera_scene)
            camera_action = normalize_action(camera_decision.get("primary_action"), camera_decision.get("best_direction"))

        except Exception as exc:
            info = {
                "video_frame_index": -1,
                "seg_path": "",
                "depth_path": "",
                "camera_points": 0,
                "camera_structure_points": 0,
                "camera_floor_points": 0,
                "camera_human_points": 0,
                "camera_reference_status": f"error:{type(exc).__name__}:{exc}",
            }
            camera_decision = {
                "primary_action": "unknown",
                "reason": "camera_reference_error",
                "urgency": "unknown",
                "best_direction": "unknown",
                "center_clear": None,
                "left_clear": None,
                "right_clear": None,
                "center_points": 0,
                "nearest_human_ft": float("inf"),
                "n_humans": 0,
            }
            camera_action = "unknown"

        radar_action = normalize_action(r.get("primary_action"), r.get("best_direction"))
        cmp = compare_actions(camera_action, radar_action)

        row = {
            "session": session,
            "frame_num": frame_num,
            "frame_idx": _safe_int(r.get("frame_idx"), -1),
            "radar_time_ms": _safe_int(r.get("radar_time_ms"), -1),
            **md,
            "video_frame_index": info.get("video_frame_index", -1),
            "camera_expected_action": camera_action,
            "camera_expected_primary_action_raw": camera_decision.get("primary_action"),
            "camera_expected_reason": camera_decision.get("reason"),
            "camera_expected_urgency": camera_decision.get("urgency"),
            "camera_expected_best_direction": camera_decision.get("best_direction"),
            "camera_center_clear": camera_decision.get("center_clear"),
            "camera_left_clear": camera_decision.get("left_clear"),
            "camera_right_clear": camera_decision.get("right_clear"),
            "camera_center_points": camera_decision.get("center_points"),
            "camera_nearest_human_ft": camera_decision.get("nearest_human_ft"),
            "camera_n_humans": camera_decision.get("n_humans"),
            "camera_points": info.get("camera_points", 0),
            "camera_structure_points": info.get("camera_structure_points", 0),
            "camera_floor_points": info.get("camera_floor_points", 0),
            "camera_human_points": info.get("camera_human_points", 0),
            "camera_reference_status": info.get("camera_reference_status", "unknown"),
            "seg_path": info.get("seg_path", ""),
            "depth_path": info.get("depth_path", ""),
            "radar_action": radar_action,
            "radar_primary_action_raw": r.get("primary_action"),
            "radar_reason": r.get("reason"),
            "radar_urgency": r.get("urgency"),
            "radar_best_direction": r.get("best_direction"),
            "radar_center_clear": r.get("center_clear"),
            "radar_left_clear": r.get("left_clear"),
            "radar_right_clear": r.get("right_clear"),
            "radar_blocking_confidence": r.get("blocking_confidence"),
            "radar_center_effective_points": r.get("center_effective_points"),
            "radar_human_count": r.get("human_count"),
            "radar_center_human_present": r.get("center_human_present"),
            **cmp,
        }
        rows.append(row)

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(out_dir / "per_frame_decision_eval.csv", index=False)

    valid_df = eval_df[eval_df["camera_expected_action"] != "unknown"].copy()
    confusion = decision_confusion(valid_df)
    confusion.to_csv(out_dir / "decision_confusion_matrix.csv")

    session_summary = summarize_group(valid_df, ["session", "building", "metadata_type"])
    session_summary.to_csv(out_dir / "per_session_decision_eval.csv", index=False)

    building_summary = summarize_group(valid_df, ["building"])
    building_summary.to_csv(out_dir / "per_building_decision_eval.csv", index=False)

    # Basic tag summary: explode pipe-separated tags.
    tag_rows = []
    if "tags" in valid_df.columns:
        all_tags = sorted({t for tags in valid_df["tags"].fillna("") for t in str(tags).split("|") if t})
        for tag in all_tags:
            g = valid_df[valid_df["tags"].fillna("").apply(lambda x: tag in str(x).split("|"))]
            if len(g):
                s = summarize_group(g.assign(tag=tag), ["tag"])
                if len(s):
                    tag_rows.append(s.iloc[0].to_dict())
    pd.DataFrame(tag_rows).to_csv(out_dir / "per_tag_decision_eval.csv", index=False)

    failures = valid_df[(~valid_df["safe_correct"]) | (valid_df["conservative_error"]) | (valid_df["wrong_direction_error"])].copy()
    failures.to_csv(out_dir / "failure_cases.csv", index=False)

    total = int(len(valid_df))
    coverage = {
        "input_frames": int(len(eval_df)),
        "valid_camera_reference_frames": total,
        "skipped_or_error_frames": int(len(eval_df) - total),
        "camera_reference_status_counts": eval_df["camera_reference_status"].value_counts(dropna=False).to_dict() if len(eval_df) else {},
    }
    overall = {
        "frames": total,
        "exact_correct": int(valid_df["exact_correct"].sum()) if total else 0,
        "safe_correct": int(valid_df["safe_correct"].sum()) if total else 0,
        "unsafe_errors": int(valid_df["unsafe_error"].sum()) if total else 0,
        "conservative_errors": int(valid_df["conservative_error"].sum()) if total else 0,
        "wrong_direction_errors": int(valid_df["wrong_direction_error"].sum()) if total else 0,
        "exact_decision_accuracy": float(valid_df["exact_correct"].mean()) if total else None,
        "safe_decision_accuracy": float(valid_df["safe_correct"].mean()) if total else None,
        "unsafe_error_rate": float(valid_df["unsafe_error"].mean()) if total else None,
        "conservative_error_rate": float(valid_df["conservative_error"].mean()) if total else None,
        "wrong_direction_error_rate": float(valid_df["wrong_direction_error"].mean()) if total else None,
    }
    report = {
        "inputs": {
            "frame_preds": str(args.frame_preds),
            "processing_root": str(args.processing_root),
            "sync_root": str(args.sync_root) if args.sync_root else None,
            "metadata_json": str(args.metadata_json) if args.metadata_json else None,
            "metadata_json_usage": "optional grouping only; no Policy-B or label-remap fields are used",
            "extrinsics_json": str(extr_path) if extr_path else None,
            "extrinsics_direction": args.extrinsics_direction,
            "assume_camera_radar_aligned": bool(args.assume_camera_radar_aligned),
            "camera_human_motion": args.camera_human_motion,
        },
        "coverage": coverage,
        "overall": overall,
        "action_order": ACTION_ORDER,
        "confusion_matrix": confusion.to_dict(),
        "by_building": building_summary.to_dict(orient="records") if len(building_summary) else [],
        "notes": [
            "Camera-reference decisions are automatic RGB-D semantic references, not perfect human labels.",
            "This new-system evaluator uses canonical replay outputs and scene_pipeline.py decision logic only; old Policy-B metadata is not required or applied.",
            "By default camera human motion is stationary, so stop-vs-slow_down exact agreement may be conservative/limited.",
            "safe_decision_accuracy separates conservative behavior from unsafe behavior.",
        ],
    }
    (out_dir / "decision_eval_report.json").write_text(json.dumps(json_sanitize(report), indent=2), encoding="utf-8")

    print("=" * 80)
    print("Replay decision evaluation against OneFormer/depth reference")
    print("=" * 80)
    print(f"Frame predictions : {args.frame_preds}")
    print(f"Output directory  : {out_dir}")
    print(f"Valid frames      : {total:,} / {len(eval_df):,}")
    if total:
        print(f"Exact accuracy    : {overall['exact_decision_accuracy'] * 100:.2f}%")
        print(f"Safe accuracy     : {overall['safe_decision_accuracy'] * 100:.2f}%")
        print(f"Unsafe error rate : {overall['unsafe_error_rate'] * 100:.2f}%")
        print(f"Conservative rate : {overall['conservative_error_rate'] * 100:.2f}%")
    print(f"Report            : {out_dir / 'decision_eval_report.json'}")


if __name__ == "__main__":
    main()
