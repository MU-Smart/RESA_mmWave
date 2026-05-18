"""
branch3_replay_simulation.py
============================
Replay recorded radar sessions through the integrated 3branch pipeline and
write an MP4 simulation artifact.

Input note:
  Replay is model inference from the raw ADC-derived point cloud CSV produced
  by adc_to_pointcloud_v6.py:
      <session>/<session>.csv

  The replay path intentionally supports only the current RD-patch Branch 1
  runtime contract. Checkpoints must declare uses_rd_patch=True. Legacy
  <session>_radar_tensors.npz files remain supported by the RD/RA runtime and
  Branch 3 U-Net path, but oracle/labeled-point replay and plain pointcloud
  checkpoints are no longer accepted here.

Updated to use the current 3branch Jetson implementation:
  - Imports from 3branch_map_builder, 3branch_nav_decision,
    3branch_scene_aggregator, 3branch_llm_guidance, 3branch_unet
  - L1 geo-floor threshold: y < 5.0m, z < 0.20m, |doppler| < 1.5 m/s
  - L2 blocking z-floor:    z > -0.85m (valid point-cloud lower bound)
  - L3 blocking y-ceiling:  y < 2.0m
  - Doppler-aware blocking weights match the current live loop.
  - MapBuilder is a daemon Thread; push_classified_points is non-blocking.
    Replay calls _queue.join() after each push to maintain synchronous
    semantics for per-frame anomaly queries.

Pipeline per frame (mirrors 3branch_navigation_loop.py):
  1. Branch 1 validation: MapBuilder persist_score lookup before update
  2. Geo floor correction       (L1: y < 5.0m, z < 0.20m)
  3. MapBuilder.push_classified_points  (joined synchronously)
  4. Doppler-aware blocking detection (L2: z > -0.85m, L3: y < 2.0m)
  5. MapBuilder.query_anomalies -> scene["map_anomalies"]
  6. Branch 3 scene["unet_freespace"] from the same companion
     <session>_radar_tensors.npz and U-Net checkpoint path used by the live loop
  7. compute_nav_decision
  8. build_scene_description

Usage:
    # Real model inference from raw ADC-derived point CSV:
    python3 branch3_replay_simulation.py \\
        --sessions-dir LLM_ML/data/data-4-30/Processing \\
        --max-sessions 5

    # Every 5th session, useful for broad sampling:
    python3 branch3_replay_simulation.py \\
        --sessions-dir LLM_ML/data/data-4-30/Processing \\
        --session-selection stride --session-stride 5 --max-sessions 9

    # Seeded random subset, not sequential:
    python3 branch3_replay_simulation.py \\
        --sessions-dir LLM_ML/data/data-4-30/Processing \\
        --session-selection random --max-sessions 8 --session-seed 7

    # All preprocessed sessions, integrated UI:
    python3 branch3_replay_simulation.py \\
        --sessions-dir LLM_ML/data/jetson_pull_2026-04-22 \\
        --ui-style integrated --output out.mp4
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import io
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import uuid

import cv2
import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

THIS_DIR     = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR
BRANCH3_DIR  = THIS_DIR
REPO_ROOT    = PIPELINE_DIR.parents[1]
DATA_DIR     = REPO_ROOT / "LLM_ML" / "data" / "jetson_pull_2026-04-22"
DEFAULT_CFG_PATH  = THIS_DIR / "config" / "profile_objdet.cfg"
DEFAULT_UNET_PT   = THIS_DIR / "models" / "unet_best_model.pt"
DEFAULT_EXTRINSICS_JSON = THIS_DIR / "config" / "radar_camera_extrinsics.json"

MODEL_PT_CANDIDATES = [
    THIS_DIR / "models" / "3branch_best_model.pt",
]

for path in (REPO_ROOT, PIPELINE_DIR, PIPELINE_DIR / "perception", PIPELINE_DIR / "models"):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ---------------------------------------------------------------------------
# Importlib loader for 3branch_* modules
# ---------------------------------------------------------------------------

def _load_mod(alias: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(alias, str(filepath))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_load_mod("branch_frame_encoder", THIS_DIR / "models" / "3branch_frame_encoder.py")
_load_mod("branch_common",       THIS_DIR / "models" / "3branch_common.py")
_map_mod  = _load_mod("branch_map_builder",   THIS_DIR / "scene_pipeline.py")
_nav_mod  = _load_mod("branch_nav_decision",  THIS_DIR / "scene_pipeline.py")
_agg_mod  = _load_mod("branch_scene_agg",     THIS_DIR / "scene_pipeline.py")
_llm_mod  = _load_mod("branch_llm_guidance",  THIS_DIR / "guidance.py")
branch3_unet = _load_mod("branch3_unet",      THIS_DIR / "models" / "branch3_unet.py")

MapBuilder           = _map_mod.MapBuilder
compute_nav_decision = _nav_mod.compute_nav_decision
aggregate_scene      = _agg_mod.aggregate_scene
build_scene_description = _llm_mod.build_scene_description

FRAME_INTERVAL_S = 0.1

_BUCKET_TO_CLASS: dict[str, str] = {
    "wall":      "structure",
    "box_like":  "structure",
    "door":      "structure",
    "staircase": "structure",
    "posting":   "structure",
    "obstacle":  "structure",
    "pillar":    "structure",
    "structure": "structure",
    "floor":     "floor",
    "human":     "human",
    "person":    "human",
}

MIN_VALID_Z = -0.851
MIN_RANGE_M = 0.3
MAX_RANGE_M = 5.0

_REPLAY_NAV_LOOP_MOD = None


def _default_model_pt() -> Path:
    for candidate in MODEL_PT_CANDIDATES:
        if candidate.exists():
            return candidate
    return MODEL_PT_CANDIDATES[0]


def _load_replay_nav_loop():
    """Load the canonical RD-patch Branch 1 inference helpers lazily."""
    global _REPLAY_NAV_LOOP_MOD
    if _REPLAY_NAV_LOOP_MOD is None:
        os.environ.setdefault("NAV_LOG_DIR", "/tmp/branch3_replay_logs")
        os.environ.setdefault("NAV_POINTCLOUD_RD_FRAME_OFFSET", "0")
        pillar_model_dir = REPO_ROOT / "LLM_ML" / "model_stuff" / "pillar_elongation_5class"
        pillar_model_dir_str = str(pillar_model_dir)
        if pillar_model_dir_str not in sys.path:
            sys.path.insert(0, pillar_model_dir_str)
        _REPLAY_NAV_LOOP_MOD = _load_mod(
            "navigation_loop",
            THIS_DIR / "navigation_loop.py",
        )
    return _REPLAY_NAV_LOOP_MOD


def _configure_nav_loop_for_replay(
    nav_loop,
    *,
    model_pt: Path,
    mb: MapBuilder | None,
) -> None:
    """Point the live-loop inference helpers at local replay artifacts."""
    model_pt = Path(model_pt).resolve()
    if not model_pt.exists():
        raise FileNotFoundError(
            f"Branch 1 model checkpoint not found: {model_pt}. "
            "Pass --model-pt or place 3branch_best_model.pt under 3branch/models/."
        )
    prev_model_pt = getattr(nav_loop, "MODEL_PT", None)
    if str(prev_model_pt) != str(model_pt):
        nav_loop._model = None
        nav_loop._feat_means = None
        nav_loop._feat_stds = None
        nav_loop._bucket_order = None
        nav_loop._feature_cols = None
        if hasattr(nav_loop, "_uses_rd_patch"):
            nav_loop._uses_rd_patch = False
        if hasattr(nav_loop, "_has_acc_head"):
            nav_loop._has_acc_head = False
        if hasattr(nav_loop, "_model_pt_resolved"):
            nav_loop._model_pt_resolved = False
        if hasattr(nav_loop, "_unet_model"):
            nav_loop._unet_model = None
        if hasattr(nav_loop, "_unet_loaded"):
            nav_loop._unet_loaded = False
        if hasattr(nav_loop, "_unet_mod"):
            nav_loop._unet_mod = None
        if hasattr(nav_loop, "_unet_checkpoint"):
            nav_loop._unet_checkpoint = None
    nav_loop.MODEL_PT = str(model_pt)
    if hasattr(nav_loop, "NAV_PERCEPTION_MODE"):
        nav_loop.NAV_PERCEPTION_MODE = "pointcloud_rd_patch"
    nav_loop.UNET_PT = str(DEFAULT_UNET_PT)
    nav_loop.MMWAVE_CFG_PATH = str(DEFAULT_CFG_PATH)
    nav_loop.ML_MODEL_DIR = str(THIS_DIR)
    nav_loop.LLM_DIR = str(THIS_DIR)
    nav_loop._map_builder = mb


def _load_extrinsics_summary(path: Path) -> dict:
    summary = {"path": str(path), "available": False}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        summary.update({
            "available": True,
            "translation_magnitude_m": data.get("translation_magnitude_m"),
            "inlier_rms_m": data.get("inlier_rms_m"),
            "n_inliers": data.get("n_inliers"),
            "n_total_correspondences": data.get("n_total_correspondences"),
            "method": data.get("method"),
        })
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _load_rd_patch_checkpoint(model_pt: Path) -> dict:
    """Load and validate the only Branch 1 checkpoint family replay supports."""
    if not model_pt.exists():
        raise FileNotFoundError(
            f"Branch 1 RD-patch checkpoint not found: {model_pt}. "
            "Pass --model-pt explicitly."
        )
    ckpt = torch.load(model_pt, map_location="cpu", weights_only=False)
    if not bool(ckpt.get("uses_rd_patch", False)):
        raise ValueError(
            f"Unsupported Branch 1 checkpoint for replay: {model_pt}. "
            "branch3_replay_simulation now accepts only RD-patch checkpoints "
            "with uses_rd_patch=True."
        )
    missing = [
        key for key in ("feature_cols", "bucket_order")
        if not ckpt.get(key)
    ]
    if missing:
        raise ValueError(
            f"RD-patch checkpoint is missing required metadata {missing}: {model_pt}"
        )
    return ckpt


def _load_model_frames(csv_path: Path, *, model_pt: Path, mb: MapBuilder | None) -> list[tuple[int, list[dict]]]:
    """Run the canonical RD-patch Branch 1 runtime and group predictions by frame."""
    if csv_path.name == "labeled_radar_points_v4.csv":
        raise ValueError(
            "Replay requires the raw ADC-derived <session>.csv, not "
            "labeled_radar_points_v4.csv."
        )
    _load_rd_patch_checkpoint(model_pt)
    nav_loop = _load_replay_nav_loop()
    _configure_nav_loop_for_replay(
        nav_loop,
        model_pt=model_pt,
        mb=mb,
    )
    points = nav_loop.run_inference_on_csv(str(csv_path), session_dir=str(csv_path.parent))
    by_frame: dict[int, list[dict]] = {}
    for pt in points:
        p = dict(pt)
        raw_cls = str(p.get("pred_class", "")).strip().lower()
        p["pred_class"] = _BUCKET_TO_CLASS.get(raw_cls, raw_cls)
        for cls_name in ("structure", "floor", "human"):
            live_key = f"pred_prob_{cls_name}"
            export_key = f"prob_{cls_name}"
            if export_key not in p and live_key in p:
                p[export_key] = p[live_key]
            elif live_key not in p and export_key in p:
                p[live_key] = p[export_key]
        if "range_m" not in p:
            p["range_m"] = float(
                np.sqrt(
                    float(p.get("x", 0.0)) ** 2
                    + float(p.get("y", 0.0)) ** 2
                    + float(p.get("z", 0.0)) ** 2
                )
            )
        by_frame.setdefault(int(p.get("frame", 0)), []).append(p)
    return sorted(by_frame.items())


def _align_frames_to_radar_timestamps(
    labeled_frames: list[tuple[int, list[dict]]],
    radar_frame_indices: np.ndarray | None,
) -> list[tuple[int, list[dict]]]:
    """Use the full radar timeline, filling unlabeled frames with no points."""
    if radar_frame_indices is None or len(radar_frame_indices) == 0:
        return labeled_frames
    by_frame = {int(frame_num): points for frame_num, points in labeled_frames}
    return [(int(frame_num), by_frame.get(int(frame_num), []))
            for frame_num in radar_frame_indices]


def _frame_times_for_replay(
    frame_data: list[tuple[int, list[dict]]],
    radar_frame_indices: np.ndarray | None,
    radar_times: np.ndarray | None,
) -> np.ndarray | None:
    """Return one radar timestamp per replay frame, keyed by actual frame number."""
    if (
        radar_frame_indices is None
        or radar_times is None
        or len(radar_frame_indices) != len(radar_times)
    ):
        return None
    time_by_frame = {
        int(frame_num): int(time_ms)
        for frame_num, time_ms in zip(radar_frame_indices, radar_times)
    }
    times = [time_by_frame.get(int(frame_num)) for frame_num, _ in frame_data]
    if any(t is None for t in times):
        return None
    return np.asarray(times, dtype=np.int64)


def _raw_session_csv_path(session_dir: Path) -> Path:
    canonical = session_dir / f"{session_dir.name}.csv"
    if canonical.exists():
        return canonical
    candidates = [
        p for p in sorted(session_dir.glob("*.csv"))
        if p.name not in {
            "labeled_radar_points_v4.csv",
            f"{session_dir.name}_radar_timestamps.csv",
            f"{session_dir.name}_color_timestamps.csv",
            f"{session_dir.name}_depth_timestamps.csv",
        }
        and not p.name.startswith("synchronized_")
        and not p.name.endswith("_timestamps.csv")
    ]
    if candidates:
        return candidates[0]
    return canonical


def _find_raw_session_csvs(sessions_dir: Path) -> list[tuple[str, Path]]:
    results = []
    for d in sorted(sessions_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("session_"):
            continue
        csv_path = _raw_session_csv_path(d)
        if csv_path.exists():
            results.append((d.name, csv_path))
    return results


def _select_session_subset(
    sessions: list[tuple[str, Path]],
    *,
    mode: str,
    max_sessions: int,
    stride: int,
    seed: int,
) -> list[tuple[str, Path]]:
    if not sessions:
        return sessions

    if mode == "sequential":
        subset = sessions
    elif mode == "stride":
        if stride <= 0:
            raise ValueError("--session-stride must be > 0")
        subset = sessions[::stride]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        limit = max_sessions if max_sessions and max_sessions > 0 else len(sessions)
        limit = min(limit, len(sessions))
        if limit >= len(sessions):
            order = rng.permutation(len(sessions))
            subset = [sessions[int(i)] for i in order]
        else:
            idxs = rng.choice(len(sessions), size=limit, replace=False)
            subset = [sessions[int(i)] for i in np.sort(idxs)]
    else:
        raise ValueError(f"Unknown session selection mode: {mode}")

    cap = max_sessions if max_sessions and max_sessions > 0 else len(subset)
    return subset[:cap]


def _session_output_path(base: Path, session_name: str, n_sessions: int, ui_style: str) -> Path:
    """Resolve one MP4 path per session when output is a directory or batch run."""
    if base.suffix.lower() == ".mp4" and n_sessions == 1:
        return base
    if base.suffix.lower() == ".mp4":
        return base.parent / f"{base.stem}_{session_name}.mp4"
    return base / f"{session_name}_{ui_style}.mp4"


def _debug_snapshot_root(base_output: Path) -> Path:
    """Root folder for sampled debug frames, kept inside the chosen output tree."""
    if base_output.suffix.lower() == ".mp4":
        return base_output.parent / f"{base_output.stem}_debug_frame_samples"
    return base_output / "debug_frame_samples"


# ---------------------------------------------------------------------------
# Branch 2 — replay defaults tuned against April 28 sessions after restoring
# full-frame timeline handling. The main lever is stronger floor recovery,
# while keeping the more permissive blocking thresholds from the newer logic.
# ---------------------------------------------------------------------------

GEO_FLOOR_Z_THRESH  = 0.20
GEO_FLOOR_Y_MAX     = 5.0
GEO_FLOOR_RECLASSIFY_ENABLED = False
BLOCKING_Z_FLOOR    = MIN_VALID_Z
BLOCKING_Y_CEILING  = 2.0
GEO_FLOOR_DOPPLER_SANITY_MPS = 1.5

BLOCKING_DOPPLER_MIN_ANCHORS        = 6
BLOCKING_DOPPLER_RANSAC_ITERS       = 24
BLOCKING_DOPPLER_INLIER_MPS         = 0.18
BLOCKING_DOPPLER_DYNAMIC_RESID_MPS  = 0.45
BLOCKING_STATIC_WEIGHT              = 0.35
BLOCKING_AMBIG_WEIGHT               = 0.70
BLOCKING_DYNAMIC_WEIGHT             = 1.00
BLOCKING_CENTER_EFFECTIVE_THRESH    = 5.0
BLOCKING_SIDE_EFFECTIVE_THRESH      = 7.0
BLOCKING_CENTER_DYNAMIC_THRESH      = 3
BLOCKING_SIDE_DYNAMIC_THRESH        = 4


def _apply_threshold_overrides(args) -> None:
    global GEO_FLOOR_Z_THRESH
    global GEO_FLOOR_Y_MAX
    global GEO_FLOOR_RECLASSIFY_ENABLED
    global BLOCKING_Z_FLOOR
    global BLOCKING_Y_CEILING
    global GEO_FLOOR_DOPPLER_SANITY_MPS
    global BLOCKING_STATIC_WEIGHT
    global BLOCKING_AMBIG_WEIGHT
    global BLOCKING_DYNAMIC_WEIGHT
    global BLOCKING_CENTER_EFFECTIVE_THRESH
    global BLOCKING_SIDE_EFFECTIVE_THRESH
    global BLOCKING_CENTER_DYNAMIC_THRESH
    global BLOCKING_SIDE_DYNAMIC_THRESH

    GEO_FLOOR_Z_THRESH = float(args.geo_floor_z)
    GEO_FLOOR_Y_MAX = float(args.geo_floor_y_max)
    GEO_FLOOR_RECLASSIFY_ENABLED = bool(args.enable_geo_floor_reclassify)
    BLOCKING_Z_FLOOR = float(args.blocking_z_floor)
    BLOCKING_Y_CEILING = float(args.blocking_y_ceiling)
    GEO_FLOOR_DOPPLER_SANITY_MPS = float(args.geo_floor_doppler_max)
    BLOCKING_STATIC_WEIGHT = float(args.blocking_static_weight)
    BLOCKING_AMBIG_WEIGHT = float(args.blocking_ambig_weight)
    BLOCKING_DYNAMIC_WEIGHT = float(args.blocking_dynamic_weight)
    BLOCKING_CENTER_EFFECTIVE_THRESH = float(args.center_effective_thresh)
    BLOCKING_SIDE_EFFECTIVE_THRESH = float(args.side_effective_thresh)
    BLOCKING_CENTER_DYNAMIC_THRESH = int(args.center_dynamic_thresh)
    BLOCKING_SIDE_DYNAMIC_THRESH = int(args.side_dynamic_thresh)


def _apply_geo_floor_correction(points: list[dict]) -> tuple[list[dict], int]:
    if not GEO_FLOOR_RECLASSIFY_ENABLED:
        return list(points), 0

    corrected, count = [], 0
    for p in points:
        if (
            p.get("pred_class") in ("structure", "human")
            and float(p.get("y", 99.0)) < GEO_FLOOR_Y_MAX
            and float(p.get("z", 0.0)) < GEO_FLOOR_Z_THRESH
            and abs(float(p.get("doppler", 0.0))) < GEO_FLOOR_DOPPLER_SANITY_MPS
        ):
            p = dict(p)
            p["pred_class"] = "floor"
            count += 1
        corrected.append(p)
    return corrected, count


def _fit_static_doppler_model(struct_points: list[dict]) -> tuple[float, float] | None:
    anchors = [
        (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
            float(p.get("doppler", 0.0)),
        )
        for p in struct_points
        if float(p.get("y", 0.0)) > 0.3
        and abs(float(p.get("doppler", 99.0))) < 1.5
    ]
    if len(anchors) < BLOCKING_DOPPLER_MIN_ANCHORS:
        return None

    arr = np.asarray(anchors, dtype=np.float64)
    xs = arr[:, 0]
    ys = arr[:, 1]
    dopplers = arr[:, 2]
    az = np.arctan2(xs, np.maximum(ys, 0.1))
    A = np.column_stack([np.sin(az), np.cos(az)])
    b = -dopplers
    n = len(b)
    if n < 2:
        return None

    best_mask = None
    best_inliers = 0
    best_resid = np.inf
    rng = np.random.default_rng(42)

    for _ in range(BLOCKING_DOPPLER_RANSAC_ITERS):
        idx = rng.choice(n, size=2, replace=False)
        try:
            v, _, _, _ = np.linalg.lstsq(A[idx], b[idx], rcond=None)
        except Exception:
            continue
        residuals = np.abs(A @ v - b)
        mask = residuals < BLOCKING_DOPPLER_INLIER_MPS
        inliers = int(mask.sum())
        median_resid = float(np.median(residuals[mask])) if inliers else np.inf
        if inliers > best_inliers or (inliers == best_inliers and median_resid < best_resid):
            best_mask = mask
            best_inliers = inliers
            best_resid = median_resid

    if best_mask is None or best_inliers < BLOCKING_DOPPLER_MIN_ANCHORS:
        return None

    try:
        v_refined, _, _, _ = np.linalg.lstsq(A[best_mask], b[best_mask], rcond=None)
    except Exception:
        return None
    return float(v_refined[0]), float(v_refined[1])


def _blocking_doppler_weights(xs: np.ndarray,
                              ys: np.ndarray,
                              dopplers: np.ndarray,
                              static_model: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(len(xs), dtype=np.float32)
    dynamic_mask = np.zeros(len(xs), dtype=bool)
    if static_model is None or len(xs) == 0:
        return weights, dynamic_mask

    vx, vy = static_model
    az = np.arctan2(xs, np.maximum(ys, 0.1))
    expected = -(vx * np.sin(az) + vy * np.cos(az))
    residual = np.abs(dopplers - expected)

    static_like = residual <= BLOCKING_DOPPLER_INLIER_MPS
    dynamic_like = residual >= BLOCKING_DOPPLER_DYNAMIC_RESID_MPS
    ambiguous = ~(static_like | dynamic_like)

    weights[static_like] = BLOCKING_STATIC_WEIGHT
    weights[ambiguous] = BLOCKING_AMBIG_WEIGHT
    weights[dynamic_like] = BLOCKING_DYNAMIC_WEIGHT
    dynamic_mask = dynamic_like
    return weights.astype(np.float32), dynamic_mask


# ---------------------------------------------------------------------------
# Ghost persistence tracker — prevents ghost markers from flickering when
# ego estimation temporarily drops out.
# ---------------------------------------------------------------------------

class GhostPersistenceTracker:
    """Maintains a decaying memory of ghost point positions across frames.

    When ego-Doppler directness is available, the tracker is updated with
    current ghost positions.  When ego drops out, previously seen ghost
    positions are carried forward with exponential decay so the overlay
    does not flicker.

    Parameters
    ----------
    decay_per_frame : float
        Multiplicative decay applied each frame (0 < decay <= 1).
        Lower values = faster fade-out.
    max_age_frames : int
        Maximum number of frames a ghost position is retained after its
        last observation.
    merge_radius_m : float
        Spatial radius for merging nearby ghost positions (prevents
        duplicate entries from frame-to-frame jitter).
    """

    def __init__(
        self,
        decay_per_frame: float = 0.85,
        max_age_frames: int = 15,
        merge_radius_m: float = 0.30,
    ) -> None:
        self.decay = float(decay_per_frame)
        self.max_age = int(max_age_frames)
        self.merge_r = float(merge_radius_m)
        # Each entry: {"x": float, "y": float, "strength": float, "age": int}
        self._ghosts: list[dict] = []

    def update(self, points: list[dict]) -> None:
        """Update tracker with current-frame ghost candidates.

        Call this *after* directness annotation so that p_dir_doppler is
        available on each point.
        """
        # Decay and age existing ghosts
        for g in self._ghosts:
            g["strength"] *= self.decay
            g["age"] += 1
        # Prune expired ghosts
        self._ghosts = [
            g for g in self._ghosts
            if g["strength"] > 0.05 and g["age"] < self.max_age
        ]

        # Add new ghost candidates from current frame
        for p in points:
            p_dir = float(p.get("p_dir_doppler", 1.0))
            if p_dir >= 0.15:
                continue
            px = float(p.get("x", 0.0))
            py = float(p.get("y", 0.0))
            # Merge with existing nearby ghost
            merged = False
            for g in self._ghosts:
                dx = g["x"] - px
                dy = g["y"] - py
                if (dx * dx + dy * dy) < (self.merge_r * self.merge_r):
                    g["strength"] = min(g["strength"] + 0.3, 1.0)
                    g["age"] = 0
                    merged = True
                    break
            if not merged:
                self._ghosts.append({
                    "x": px, "y": py,
                    "strength": 1.0,
                    "age": 0,
                })

    def get_active(self, min_strength: float = 0.20) -> list[tuple[float, float]]:
        """Return list of (x, y) tuples for active ghost positions."""
        return [
            (g["x"], g["y"])
            for g in self._ghosts
            if g["strength"] >= min_strength
        ]

    def reset(self) -> None:
        self._ghosts.clear()


# ---------------------------------------------------------------------------
# Doppler residual annotation
# ---------------------------------------------------------------------------

def _annotate_doppler_residuals(
    points: list[dict],
    static_model: tuple[float, float] | None,
) -> None:
    """Attach doppler_residual / doppler_expected fields in-place to structure and human points.

    doppler_residual = measured_doppler − ego_expected_doppler.
    Negative = target approaching faster than ego motion predicts (dynamic
    inbound); positive = receding faster than predicted.  When no static ego
    model is available the raw Doppler is used as the residual (ego = 0).
    """
    for pt in points:
        if pt.get("pred_class") not in ("structure", "human"):
            continue
        px_v  = float(pt.get("x", 0.0))
        py_v  = float(pt.get("y", 0.0))
        dop_v = float(pt.get("doppler", 0.0))
        if static_model is not None:
            vx_m, vy_m = static_model
            az_v    = float(np.arctan2(px_v, max(py_v, 0.1)))
            exp_dop = -(vx_m * np.sin(az_v) + vy_m * np.cos(az_v))
        else:
            exp_dop = 0.0
        signed_r = dop_v - exp_dop
        pt["doppler_expected"]     = round(float(exp_dop), 3)
        pt["doppler_residual"]     = round(float(signed_r), 3)
        pt["doppler_residual_abs"] = round(abs(float(signed_r)), 3)
        pt["doppler_ego_model"]    = static_model is not None


def _apply_improved_blocking(points: list[dict], scene: dict) -> dict:
    struct_pts = [p for p in points if p.get("pred_class") == "structure"]
    if not struct_pts:
        scene["doppler_residuals"] = {"ego_model_available": False, "n_structure_points": 0}
        return scene
    try:
        static_model = _fit_static_doppler_model(struct_pts)

        # Annotate all structure and human points with signed Doppler residuals
        # (used by BEV vector rendering and captured in debug JSON).
        _annotate_doppler_residuals(points, static_model)
        n_struct = len(struct_pts)
        all_abs_resid = np.array(
            [abs(float(p.get("doppler_residual", 0.0))) for p in struct_pts],
            dtype=np.float32,
        )
        scene["doppler_residuals"] = {
            "ego_model_available": static_model is not None,
            "vx_mps": round(float(static_model[0]), 3) if static_model else None,
            "vy_mps": round(float(static_model[1]), 3) if static_model else None,
            "n_structure_points": n_struct,
            "n_static":  int(np.sum(all_abs_resid <= BLOCKING_DOPPLER_INLIER_MPS)),
            "n_dynamic": int(np.sum(all_abs_resid >= BLOCKING_DOPPLER_DYNAMIC_RESID_MPS)),
            "residual_mean_mps": round(float(np.mean(all_abs_resid)), 3) if n_struct else 0.0,
            "residual_max_mps":  round(float(np.max(all_abs_resid)),  3) if n_struct else 0.0,
            "residual_p75_mps":  round(float(np.percentile(all_abs_resid, 75)), 3) if n_struct else 0.0,
        }

        arr = np.array(
            [[p["x"], p["y"], p["z"], float(p.get("doppler", 0.0))]
             for p in struct_pts],
            dtype=np.float32,
        )
        xs, ys, zs, ds = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        in_zone = (ys >= 0.5) & (ys < BLOCKING_Y_CEILING)
        if not np.any(in_zone):
            return scene
        bx, by, bz, bd = xs[in_zone], ys[in_zone], zs[in_zone], ds[in_zone]
        az    = np.degrees(np.arctan2(np.abs(bx), np.maximum(by, 0.1)))
        above = bz > BLOCKING_Z_FLOOR
        weights, dynamic = _blocking_doppler_weights(bx, by, bd, static_model)

        center_mask = (az <= 20.0) & (np.abs(bx) < 0.5) & above
        left_mask = (bx <= 0) & (az > 20.0) & (az <= 60.0) & above
        right_mask = (bx > 0) & (az > 20.0) & (az <= 60.0) & above

        c_cnt = int(np.sum(center_mask))
        l_cnt = int(np.sum(left_mask))
        r_cnt = int(np.sum(right_mask))
        c_eff = float(np.sum(weights[center_mask]))
        l_eff = float(np.sum(weights[left_mask]))
        r_eff = float(np.sum(weights[right_mask]))
        c_dyn = int(np.sum(dynamic[center_mask]))
        l_dyn = int(np.sum(dynamic[left_mask]))
        r_dyn = int(np.sum(dynamic[right_mask]))

        scene.setdefault("blocking", {})
        scene["blocking"]["center_points"] = c_cnt
        scene["blocking"]["left_points"]   = l_cnt
        scene["blocking"]["right_points"]  = r_cnt
        scene["blocking"]["center_effective_points"] = round(c_eff, 1)
        scene["blocking"]["left_effective_points"]   = round(l_eff, 1)
        scene["blocking"]["right_effective_points"]  = round(r_eff, 1)
        scene["blocking"]["center_dynamic_points"] = c_dyn
        scene["blocking"]["left_dynamic_points"]   = l_dyn
        scene["blocking"]["right_dynamic_points"]  = r_dyn
        scene["blocking"]["source"] = "raw_structure_geometry"
        scene["blocking"]["z_floor_m"] = BLOCKING_Z_FLOOR
        scene["blocking"]["y_ceiling_m"] = BLOCKING_Y_CEILING
        scene["open_directions_confidence"] = "observed"

        center_blocked = (
            c_eff >= BLOCKING_CENTER_EFFECTIVE_THRESH
            or c_dyn >= BLOCKING_CENTER_DYNAMIC_THRESH
        )
        left_blocked = (
            l_eff >= BLOCKING_SIDE_EFFECTIVE_THRESH
            or l_dyn >= BLOCKING_SIDE_DYNAMIC_THRESH
        )
        right_blocked = (
            r_eff >= BLOCKING_SIDE_EFFECTIVE_THRESH
            or r_dyn >= BLOCKING_SIDE_DYNAMIC_THRESH
        )
        scene["open_directions"] = {
            "left_clear":   not left_blocked,
            "center_clear": not center_blocked,
            "right_clear":  not right_blocked,
        }
        center_conf = (
            "strong"
            if c_eff >= 8.0 or c_dyn >= 4
            else "weak"
            if c_cnt > 0 or c_eff > 0.0 or c_dyn > 0
            else "none"
        )
        scene["blocking"]["center_confidence"] = center_conf
        scene["open_directions_confidence"] = (
            "observed" if center_conf == "strong" else "weak" if center_conf == "weak" else "unknown"
        )
        scene.setdefault("evidence_channels", {}).setdefault("blocking", {})
        scene["evidence_channels"]["blocking"].update({
            "source": "raw_structure_geometry",
            "confidence": center_conf,
            "raw_points": c_cnt,
            "effective_points": round(c_eff, 1),
            "dynamic_points": c_dyn,
            "left_effective_points": round(l_eff, 1),
            "right_effective_points": round(r_eff, 1),
        })
    except Exception as exc:
        print(f"    [warn] Improved blocking failed: {exc}", flush=True)
    return scene


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

_ACTION_COLOR = {
    "stop":       "#c0392b",
    "veer_left":  "#e67e22",
    "veer_right": "#e67e22",
    "slow_down":  "#f1c40f",
    "continue":   "#27ae60",
}
_CLASS_COLOR  = {"structure": "#e74c3c", "floor": "#95a5a6", "human": "#2ecc71"}
_CLASS_MARKER = {"structure": "s", "floor": ".", "human": "^"}

# April 28 replay points are physically concentrated near the forward axis.
# A wide +/-3.5m lateral plot made valid detections look like a thin line.
# Keep the preview focused on the navigation-relevant envelope; this affects
# only video rendering, not inference.
RADAR_PREVIEW_X_HALF_M = 2.0
RADAR_PREVIEW_Y_MAX_M  = 4.8


def _render_frame(
    session_label: str,
    frame_num: int,
    frame_idx: int,
    n_frames: int,
    sim_time_s: float,
    points: list[dict],
    scene: dict,
    decision: dict,
    map_occ: dict,
    guidance_text: str,
    guidance_fired: bool,
    frame_width: int,
    frame_height: int,
    ghost_positions: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    # ── palette ──────────────────────────────────────────────────────────────
    BG_PAGE  = "#080c18"
    BG_PANEL = "#0d1122"
    BG_CARD  = "#131828"
    BORDER   = "#1e2640"
    T_PRI    = "#dde4f5"
    T_SEC    = "#6e7d9e"
    T_DIM    = "#353f58"
    GRID_C   = "#111830"

    _ACOL = {
        "stop": "#cf3434", "veer_left": "#d4821e",
        "veer_right": "#d4821e", "slow_down": "#c9a500", "continue": "#27a85e",
    }
    _CCOL = {"structure": "#e05252", "floor": "#556070", "human": "#2ece74"}
    _CMRK = {"structure": "s", "floor": ".", "human": "^"}

    action     = decision.get("primary_action", "?")
    urgency    = decision.get("urgency", "")
    action_col = _ACOL.get(action, "#556070")

    fig = plt.figure(figsize=(frame_width / 100, frame_height / 100), dpi=100)
    fig.patch.set_facecolor(BG_PAGE)

    gs = GridSpec(1, 2, figure=fig, width_ratios=[57, 43],
                  wspace=0.06, left=0.06, right=0.98, top=0.91, bottom=0.09)
    ax_bev  = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    # ── header text (figure-level) ────────────────────────────────────────────
    short = session_label[-38:] if len(session_label) > 38 else session_label
    fig.text(0.06, 0.96, short,
             color=T_PRI, fontsize=9, fontweight="bold", va="bottom",
             fontfamily="monospace")
    fig.text(0.06, 0.926, f"frame {frame_num:03d}  ·  {frame_idx+1}/{n_frames}  ·  t={sim_time_s:.1f}s",
             color=T_SEC, fontsize=7, va="bottom", fontfamily="monospace")
    fig.text(0.98, 0.955, f"▶  {action.upper().replace('_', ' ')}",
             color=action_col, fontsize=10.5, fontweight="bold",
             va="center", ha="right")
    if urgency:
        fig.text(0.98, 0.924, urgency.upper(),
                 color=action_col, fontsize=6.5, alpha=0.72, va="center", ha="right")

    # ── BEV ──────────────────────────────────────────────────────────────────
    ax_bev.set_facecolor(BG_PANEL)
    ax_bev.set_xlim(RADAR_PREVIEW_X_HALF_M, -RADAR_PREVIEW_X_HALF_M)
    ax_bev.set_ylim(0.0, RADAR_PREVIEW_Y_MAX_M)
    ax_bev.set_xlabel("← Left  ·  Lateral (m)  ·  Right →",
                      color=T_SEC, fontsize=7.5, labelpad=2)
    ax_bev.set_ylabel("Forward (m)", color=T_SEC, fontsize=7.5, labelpad=2)
    ax_bev.tick_params(colors=T_SEC, labelsize=7, pad=2)
    for spine in ax_bev.spines.values():
        spine.set_edgecolor(BORDER)
    ax_bev.grid(True, color=GRID_C, linestyle="-", linewidth=0.4, alpha=1.0)

    od = scene.get("open_directions", {})
    dirs_unknown = (
        scene.get("open_directions_confidence") == "unknown"
        or decision.get("reason") == "low_confidence"
    )

    def _zone_shade(clear_key):
        if dirs_unknown:
            return "#10152a", 0.55
        return ("#0b2218", 0.55) if od.get(clear_key, True) else ("#220c0c", 0.60)

    lc, la = _zone_shade("left_clear")
    cc, ca = _zone_shade("center_clear")
    rc, ra = _zone_shade("right_clear")
    ax_bev.axvspan(0.5,  RADAR_PREVIEW_X_HALF_M,  color=lc, alpha=la, zorder=0)
    ax_bev.axvspan(-0.5, 0.5,                     color=cc, alpha=ca, zorder=0)
    ax_bev.axvspan(-RADAR_PREVIEW_X_HALF_M, -0.5, color=rc, alpha=ra, zorder=0)

    for xv in (-0.5, 0.5):
        ax_bev.axvline(xv, color="#2a3460", lw=0.9, ls="--", alpha=0.75, zorder=1)

    # range arcs
    theta = np.linspace(-np.pi / 2, np.pi / 2, 220)
    for r, rlbl in ((1.0, ""), (1.5, "1.5 m"), (2.5, ""), (3.5, "3.5 m")):
        if r > RADAR_PREVIEW_Y_MAX_M:
            continue
        rx = r * np.sin(theta)
        ry = r * np.cos(theta)
        mask = np.abs(rx) <= RADAR_PREVIEW_X_HALF_M
        ax_bev.plot(rx[mask], ry[mask], color="#1e2b50",
                    lw=0.9, ls=":", alpha=0.90, zorder=1)
        if rlbl:
            ax_bev.text(RADAR_PREVIEW_X_HALF_M - 0.07, r + 0.06, rlbl,
                        color="#2c3a60", fontsize=6.0, ha="right", va="bottom")

    for xt, zt in ((1.35, "LEFT"), (0.0, "CTR"), (-1.35, "RIGHT")):
        ax_bev.text(xt, 0.20, zt, color="#354060", fontsize=6.5,
                    ha="center", va="bottom", fontweight="bold")

    # radar origin with pulse ring
    ax_bev.add_patch(plt.Circle((0, 0), 0.20, color="#3a8fd9",
                                fill=False, lw=0.9, alpha=0.30, zorder=9))
    ax_bev.scatter([0], [0], marker="^", s=160, color="#3a8fd9",
                   zorder=10, edgecolors="#90c8f0", linewidths=1.0)

    # direction arrow — glow + solid
    direction = decision.get("best_direction", "center")
    adx = {"left": 1.0, "right": -1.0}.get(direction, 0.0)
    ady = 0.55 if direction in ("left", "right") else (1.2 if direction == "center" else 0.0)
    if ady > 0:
        arw = _ACOL.get(action, "#27a85e")
        ax_bev.annotate("", xy=(adx, ady), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=arw,
                                        lw=5.5, alpha=0.14), zorder=6)
        ax_bev.annotate("", xy=(adx, ady), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=arw,
                                        lw=2.0, alpha=0.96), zorder=7)

    # point cloud — glow + core
    by_class: dict[str, list] = {"structure": [], "floor": [], "human": []}
    ghost_pts: list[tuple[float, float]] = []
    for p in points:
        cls = p.get("pred_class", "floor")
        if cls in by_class:
            by_class[cls].append((float(p["x"]), float(p["y"])))
        # Mark probable ghost returns: low p_dir_doppler indicates Doppler-inconsistent return
        p_dir = float(p.get("p_dir_doppler", 1.0))
        if p_dir < 0.15:
            ghost_pts.append((float(p["x"]), float(p["y"])))
    for cls, pts_xy in by_class.items():
        if not pts_xy:
            continue
        col = _CCOL[cls]
        px_ = [p[0] for p in pts_xy]
        py_ = [p[1] for p in pts_xy]
        s   = 22 if cls == "floor" else 58
        ax_bev.scatter(px_, py_, c=col, marker=_CMRK[cls],
                       s=s * 4.5, alpha=0.09, zorder=3, linewidths=0)
        ax_bev.scatter(px_, py_, c=col, marker=_CMRK[cls],
                       s=s, alpha=0.88 if cls != "floor" else 0.58,
                       zorder=4,
                       edgecolors="#ffffff" if cls == "human" else "none",
                       linewidths=0.5 if cls == "human" else 0)

    # Ghost overlay: diamond markers on top of model-predicted markers
    if ghost_pts:
        gx = [p[0] for p in ghost_pts]
        gy = [p[1] for p in ghost_pts]
        ax_bev.scatter(gx, gy, marker="D", c="#d4a017", s=72, alpha=0.75,
                       zorder=5, edgecolors="#ffcc00", linewidths=0.8)

    # Doppler residual gradient vectors — drawn above scatter layers.
    _draw_doppler_vectors(ax_bev, points, scale_mps_to_m=0.55, alpha=0.78, zorder=6)

    leg_handles = [mpatches.Patch(color=_CCOL[c], label=c)
                   for c in ("structure", "floor", "human") if by_class.get(c)]
    if ghost_pts:
        leg_handles.append(mpatches.Patch(color="#d4a017", label="ghost"))
    if leg_handles:
        ax_bev.legend(handles=leg_handles, loc="upper right", fontsize=7,
                      facecolor=BG_CARD, labelcolor=T_PRI, framealpha=0.90,
                      edgecolor=BORDER, borderpad=0.5, handlelength=1.2)

    ax_bev.text(0.02, 0.98, "RADAR BEV",
                transform=ax_bev.transAxes, color=T_DIM, fontsize=7,
                fontweight="bold", va="top", fontfamily="monospace")

    # ── progress bar ──────────────────────────────────────────────────────────
    bar_ax = fig.add_axes([0.06, 0.033, 0.595, 0.018])
    bar_ax.set_facecolor("#0d1020")
    bar_ax.set_xlim(0, max(n_frames, 1))
    bar_ax.set_ylim(0, 1)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for spine in bar_ax.spines.values():
        spine.set_edgecolor(BORDER)
    bar_ax.barh(0.5, frame_idx + 1, height=1.0,
                color=action_col, align="center", alpha=0.70)
    for frac in np.linspace(0.1, 0.9, 9):
        bar_ax.axvline(frac * n_frames, color="#181f38", lw=0.7)
    bar_ax.text(max(n_frames, 1) / 2, 0.5,
                f"{frame_idx+1} / {n_frames}  ·  t={sim_time_s:.1f}s",
                color=T_PRI, fontsize=6.2, ha="center", va="center",
                fontfamily="monospace")

    # ── INFO PANEL ────────────────────────────────────────────────────────────
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    def _sec(y, title, col):
        ax_info.text(0.04, y, title, color=col, fontsize=7.0, fontweight="bold",
                     va="top", transform=ax_info.transAxes, fontfamily="monospace")
        ax_info.plot([0.04, 0.96], [y - 0.027, y - 0.027],
                     color=col, lw=0.55, alpha=0.40,
                     transform=ax_info.transAxes, clip_on=False)
        return y - 0.032

    def _kv(y, key, val, vc=T_PRI, fs=6.8):
        ax_info.text(0.06, y, key, color=T_SEC, fontsize=fs, va="top",
                     transform=ax_info.transAxes, fontfamily="monospace")
        ax_info.text(0.50, y, str(val), color=vc, fontsize=fs, va="top",
                     transform=ax_info.transAxes, fontfamily="monospace")
        return y - 0.030

    def _bar(y, label, value, max_val, col, bar_h=0.020):
        x0, bw = 0.40, 0.54
        ax_info.text(0.06, y - bar_h / 2 + 0.004, label,
                     color=T_DIM, fontsize=5.8, va="center",
                     transform=ax_info.transAxes, fontfamily="monospace")
        ax_info.add_patch(Rectangle(
            (x0, y - bar_h), bw, bar_h,
            facecolor="#0e1528", transform=ax_info.transAxes, clip_on=False, zorder=2))
        frac = min(max(float(value) / max(float(max_val), 1e-9), 0.0), 1.0)
        if frac > 0:
            ax_info.add_patch(Rectangle(
                (x0, y - bar_h), bw * frac, bar_h,
                facecolor=col, alpha=0.78,
                transform=ax_info.transAxes, clip_on=False, zorder=3))
        ax_info.text(x0 + bw + 0.02, y - bar_h / 2 + 0.002,
                     f"{value:.2f}", color=T_SEC, fontsize=5.6, va="center",
                     transform=ax_info.transAxes, fontfamily="monospace")
        return y - bar_h - 0.007

    def _sep(y):
        ax_info.plot([0.02, 0.98], [y, y], color=BORDER, lw=0.5, alpha=0.8,
                     transform=ax_info.transAxes, clip_on=False)
        return y - 0.012

    y = 0.985

    # ── ACTION ───────────────────────────────────────────────────────────────
    y = _sec(y, "ACTION", action_col)
    ax_info.text(0.06, y, action.upper().replace("_", " "),
                 color=action_col, fontsize=15, fontweight="bold",
                 va="top", transform=ax_info.transAxes)
    y -= 0.065
    y = _kv(y, "reason",    decision.get("reason", "—"),          vc=T_PRI)
    y = _kv(y, "direction", decision.get("best_direction", "—"),  vc=T_PRI)
    y = _kv(y, "urgency",   urgency or "—",                       vc=T_SEC)
    y = _kv(y, "blk_conf",  decision.get("blocking_confidence", "—"), vc=T_SEC)
    y = _sep(y)

    # ── DIRECTIONS ────────────────────────────────────────────────────────────
    y = _sec(y, "DIRECTIONS", "#4a90d9")
    dirs_labels = [("LEFT", "left_clear"), ("CTR", "center_clear"), ("RIGHT", "right_clear")]
    box_w, box_h = 0.275, 0.065
    box_y = y - box_h - 0.008
    for i, (lbl, key) in enumerate(dirs_labels):
        if dirs_unknown:
            fc, tc, vt = "#2a2200", "#c8a400", "?"
        else:
            clear = od.get(key, True)
            fc = "#0c2818" if clear else "#1e0808"
            tc = "#3dd880" if clear else "#d05050"
            vt = "CLR" if clear else "BLK"
        bx = 0.04 + i * (box_w + 0.018)
        ax_info.add_patch(Rectangle(
            (bx, box_y), box_w, box_h,
            facecolor=fc, transform=ax_info.transAxes, clip_on=False, zorder=2))
        ax_info.add_patch(Rectangle(
            (bx, box_y), box_w, box_h,
            fill=False, edgecolor=tc, linewidth=0.9,
            transform=ax_info.transAxes, clip_on=False, zorder=3))
        ax_info.text(bx + box_w / 2, box_y + box_h - 0.006, lbl,
                     color=T_DIM, fontsize=6.0, fontweight="bold",
                     ha="center", va="top", transform=ax_info.transAxes)
        ax_info.text(bx + box_w / 2, box_y + 0.010, vt,
                     color=tc, fontsize=8.5, fontweight="bold",
                     ha="center", va="bottom", transform=ax_info.transAxes)
    y = box_y - 0.014
    y = _sep(y)

    # ── BRANCHES ─────────────────────────────────────────────────────────────
    y = _sec(y, "BRANCHES", "#9b67d0")
    anom = scene.get("map_anomalies", {})
    fs   = scene.get("unet_freespace", {})
    sh   = float(anom.get("struct_history", 0))
    fh   = float(anom.get("free_history",  0))
    novel_occ  = bool(anom.get("novel_occupancy",  False))
    novel_free = bool(anom.get("novel_free_space", False))

    y = _bar(y, "B2 struct", sh / (sh + 5.0) if sh > 0 else 0.0, 1.0, "#d9841a")
    y = _bar(y, "B2 free",   fh / (fh + 5.0) if fh > 0 else 0.0, 1.0, "#3498db")

    occ_c  = "#e05252" if novel_occ  else T_DIM
    free_c = "#2ece74" if novel_free else T_DIM
    ax_info.text(0.06, y,
                 f"novel_occ={'Y' if novel_occ else 'n'}   "
                 f"novel_free={'Y' if novel_free else 'n'}",
                 color=T_DIM, fontsize=6.5, va="top",
                 transform=ax_info.transAxes, fontfamily="monospace")
    if novel_occ:
        ax_info.text(0.178, y, "Y", color="#e05252", fontsize=6.5, va="top",
                     transform=ax_info.transAxes, fontfamily="monospace")
    if novel_free:
        ax_info.text(0.540, y, "Y", color="#2ece74", fontsize=6.5, va="top",
                     transform=ax_info.transAxes, fontfamily="monospace")
    y -= 0.028

    if fs.get("available"):
        y = _bar(y, "B3 CTR",  float(fs.get("center", 0)), 1.0, "#2ece74")
        y = _bar(y, "B3 LEFT", float(fs.get("left",   0)), 1.0, "#4a90d9")
        y = _bar(y, "B3 RIGHT",float(fs.get("right",  0)), 1.0, "#4a90d9")
        if decision.get("unet_disagree_center"):
            ax_info.text(0.06, y, "▲ UNet disagrees center",
                         color="#c9a500", fontsize=6.5, fontweight="bold",
                         va="top", transform=ax_info.transAxes)
            y -= 0.025
    else:
        ax_info.text(0.06, y, f"B3 {fs.get('reason', 'unavailable')[:26]}",
                     color=T_DIM, fontsize=6.5, va="top",
                     transform=ax_info.transAxes, fontfamily="monospace")
        y -= 0.025
    y = _sep(y)

    # ── POINTS ───────────────────────────────────────────────────────────────
    counts = {c: sum(1 for p in points if p.get("pred_class") == c)
              for c in ("structure", "floor", "human")}
    y = _sec(y, "POINTS", T_SEC)
    for i, (cls, col) in enumerate(
            [("structure", _CCOL["structure"]),
             ("floor",     _CCOL["floor"]),
             ("human",     _CCOL["human"])]):
        bx = 0.04 + i * 0.315
        ax_info.text(bx, y - 0.004, f"{counts[cls]:3d}",
                     color=col, fontsize=12, fontweight="bold",
                     va="top", transform=ax_info.transAxes, fontfamily="monospace")
        ax_info.text(bx, y - 0.042, cls[:5],
                     color=T_DIM, fontsize=5.8, va="top",
                     transform=ax_info.transAxes)
    y -= 0.072
    y = _sep(y)

    # ── GUIDANCE ─────────────────────────────────────────────────────────────
    gc = "#2ece74" if guidance_fired else T_SEC
    y = _sec(y, f"GUIDANCE {'[live]' if guidance_fired else '[cached]'}", gc)
    words = (guidance_text or "Analysing scene...").split()
    glines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 30:
            if cur:
                glines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        glines.append(cur)
    for li, lg in enumerate(glines[:4]):
        if y < 0.02:
            break
        prefix = "“" if li == 0 else " "
        suffix = "”" if (li == len(glines) - 1 or li == 3) else ""
        ax_info.text(0.06, y, f"{prefix}{lg}{suffix}",
                     color=gc, fontsize=7.0, va="top",
                     transform=ax_info.transAxes, style="italic")
        y -= 0.036

    # ── render ────────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    if img.shape[:2] != (frame_height, frame_width):
        img = cv2.resize(img, (frame_width, frame_height))
    return img


def _draw_doppler_vectors(
    ax,
    points: list[dict],
    *,
    scale_mps_to_m: float = 0.55,
    min_residual_mps: float = 0.08,
    alpha: float = 0.74,
    zorder: int = 5,
) -> None:
    """Draw ego-compensated Doppler residual gradient vectors on a BEV axes.

    Each arrow originates at the point's (x, y) position and extends in the
    radial direction (along the radar LOS) by an amount proportional to the
    signed Doppler residual:

      doppler_residual < 0  →  approaching faster than ego predicts
                               →  arrow points toward radar origin
      doppler_residual > 0  →  receding faster than ego predicts
                               →  arrow points away from origin

    Color gradient:
      green  (#2ece74) – effectively static  |residual| ≤ inlier threshold
      yellow (#c9a500) – ambiguous
      red    (#e05252) – dynamic             |residual| ≥ dynamic threshold

    Arrows are drawn for structure and human points that have been annotated
    by _annotate_doppler_residuals.  Points with |residual| < min_residual_mps
    are skipped to avoid visual clutter from near-static returns.
    """
    vecs: list[tuple[float, float, float, float, float]] = []
    for p in points:
        if "doppler_residual" not in p:
            continue
        if p.get("pred_class") not in ("structure", "human"):
            continue
        px_v = float(p["x"])
        py_v = float(p["y"])
        dr   = float(p["doppler_residual"])
        if abs(dr) < min_residual_mps:
            continue
        r = float(np.hypot(px_v, py_v))
        if r < 0.05:
            continue
        # Radial velocity component projected onto 2-D BEV plane.
        # dr < 0  (approaching) → U/V have opposite sign to x/y → arrow toward origin
        u = dr * px_v / r * scale_mps_to_m
        v = dr * py_v / r * scale_mps_to_m
        vecs.append((px_v, py_v, u, v, abs(dr)))

    if not vecs:
        return

    xs_v  = np.array([w[0] for w in vecs])
    ys_v  = np.array([w[1] for w in vecs])
    us_v  = np.array([w[2] for w in vecs])
    vs_v  = np.array([w[3] for w in vecs])
    mag_v = np.array([w[4] for w in vecs])

    # Normalise magnitude against the dynamic residual threshold for coloring.
    norm_mag = np.clip(mag_v / max(BLOCKING_DOPPLER_DYNAMIC_RESID_MPS, 1e-6), 0.0, 1.0)

    # Two-segment gradient: green → yellow (0–0.5) and yellow → red (0.5–1.0)
    colors = []
    for nm in norm_mag:
        if nm < 0.5:
            t = nm * 2.0          # 0 → 1 across the green-to-yellow segment
            r_c = 0.19 + 0.60 * t  # 0.19 → 0.79
            g_c = 0.81 - 0.18 * t  # 0.81 → 0.63
            b_c = 0.45 - 0.17 * t  # 0.45 → 0.28
        else:
            t = (nm - 0.5) * 2.0  # 0 → 1 across the yellow-to-red segment
            r_c = 0.79 + 0.09 * t  # 0.79 → 0.88
            g_c = 0.63 - 0.44 * t  # 0.63 → 0.19
            b_c = 0.28 - 0.12 * t  # 0.28 → 0.16
        colors.append((r_c, g_c, b_c))

    ax.quiver(
        xs_v, ys_v, us_v, vs_v,
        color=colors,
        scale=1.0, scale_units="xy",
        width=0.007, headwidth=3.5, headlength=4.5, headaxislength=4.0,
        alpha=alpha, zorder=zorder,
    )


def _write_mp4(frames: list[np.ndarray], output_path: Path, fps: float = 10.0) -> None:
    if not frames:
        print("  [video] No frames to write.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"  [video] Could not open VideoWriter for {output_path}")
        return
    for frame in frames:
        writer.write(frame)
    for _ in range(max(1, int(fps * 2))):
        writer.write(frames[-1])
    writer.release()
    print(f"  [video] {len(frames)} frames -> {output_path}  "
          f"({w}x{h} @ {fps}fps, ~{len(frames)/fps:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Branch 3 replay helpers. This mirrors 3branch_navigation_loop.run_unet_freespace:
# load the companion <session>_radar_tensors.npz, read rd_{frame_num}, run the
# fixed checkpoint with no MapBuilder prior, and compute the same near-range
# sector means.
# ---------------------------------------------------------------------------

_UNET_LEFT_COLS   = (0,   36)
_UNET_CENTER_COLS = (36,  92)
_UNET_RIGHT_COLS  = (92,  128)
_UNET_RANGE_BINS  = (0,   64)


def _unavailable_unet(reason: str = "live_unet_unavailable") -> dict:
    return {"available": False, "center": 0.0, "left": 0.0, "right": 0.0,
            "reason": reason, "source": "live_loop_equivalent"}


def _load_live_npz_for_session(csv_path: Path):
    npz_path = csv_path.parent / f"{csv_path.parent.name}_radar_tensors.npz"
    if not npz_path.exists():
        return None
    try:
        return np.load(str(npz_path), allow_pickle=False)
    except Exception:
        return None


class LiveBranch3Runner:
    def __init__(self, unet_pt: Path = DEFAULT_UNET_PT) -> None:
        self.unet_pt = Path(unet_pt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = False
        self.reason = ""
        self.model = None
        if not self.unet_pt.exists():
            self.reason = "unet_checkpoint_not_found"
            return
        try:
            _, model = branch3_unet.load_unet_checkpoint(
                self.unet_pt, map_location=str(self.device))
            model.to(self.device)
            model.eval()
            self.model = model
            self.ready = True
        except Exception as exc:
            self.reason = f"unet_load_failed:{exc}"

    def infer(self, npz_data, frame_num: int) -> dict:
        if not self.ready or self.model is None:
            return _unavailable_unet(self.reason or "unet_not_loaded")
        rd_key = f"rd_{int(frame_num)}"
        if npz_data is None or rd_key not in npz_data:
            return _unavailable_unet("rd_tensor_missing")
        try:
            rd_cube = npz_data[rd_key]
            inp = branch3_unet.rd_cube_to_input(rd_cube, use_prior=False)
            inp = inp.to(self.device)
            with torch.no_grad():
                prob_map = self.model(inp).squeeze(0).detach().cpu().numpy()
            if prob_map.shape != (128, 128):
                return _unavailable_unet("unexpected_unet_output_shape")

            r0, r1 = _UNET_RANGE_BINS
            prob_near = prob_map[r0:r1, :]
            l0, l1 = _UNET_LEFT_COLS
            c0, c1 = _UNET_CENTER_COLS
            rr0, rr1 = _UNET_RIGHT_COLS
            left_prob = float(np.nanmean(prob_near[:, l0:l1]))
            center_prob = float(np.nanmean(prob_near[:, c0:c1]))
            right_prob = float(np.nanmean(prob_near[:, rr0:rr1]))
            return {
                "available": True,
                "center": center_prob if np.isfinite(center_prob) else 0.0,
                "left": left_prob if np.isfinite(left_prob) else 0.0,
                "right": right_prob if np.isfinite(right_prob) else 0.0,
                "source": "live_loop_equivalent",
                "frame": int(frame_num),
            }
        except Exception as exc:
            return _unavailable_unet(f"unet_inference_failed:{exc}")


# ---------------------------------------------------------------------------
# Color video helpers (unchanged from old version)
# ---------------------------------------------------------------------------

def _load_frame_timestamps(path: Path):
    try:
        frame_indices, times = [], []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                frame_indices.append(int(row["frame_index"]))
                times.append(int(row["timestamp_ms"]))
        return np.asarray(frame_indices, np.int64), np.asarray(times, np.int64)
    except Exception:
        return None


def _find_color_video(session_dir: Path) -> Path | None:
    c = session_dir / f"{session_dir.name}_color.mp4"
    if c.exists():
        return c
    for f in session_dir.glob("*_color.mp4"):
        return f
    return None


def _probe_color_panel_height(session_list, width):
    for _, csv_path in session_list:
        vid = _find_color_video(csv_path.parent)
        if vid is None:
            continue
        cap = cv2.VideoCapture(str(vid))
        if cap.isOpened():
            vw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            vh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            if vw > 0:
                return round(vh * width / vw)
    return 0


def _read_next_color_frame(cap, width, height):
    if cap is None or not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    return frame


# ---------------------------------------------------------------------------
# MapBuilder-backed persist_score summary
# ---------------------------------------------------------------------------

def _persist_summary(mb: MapBuilder, points: list[dict]) -> dict:
    if not points:
        return {"max": 0.0, "mean": 0.0, "nonzero": 0}
    xyz    = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float32)
    scores = mb.get_point_occupancy_scores(xyz, radius_m=0.30)
    return {
        "max":     float(np.max(scores))  if len(scores) else 0.0,
        "mean":    float(np.mean(scores)) if len(scores) else 0.0,
        "nonzero": int(np.sum(scores > 1e-6)),
    }


# ---------------------------------------------------------------------------
# Per-frame 3branch processing
# ---------------------------------------------------------------------------

class EgoTrajectoryState:
    """
    Causal replay-only ego-motion smoother/integrator.

    This is not a global SLAM trajectory. It is a body-frame trajectory summary
    derived from radar range-rate ego velocity, used to make Doppler residuals
    less frame-noisy and to expose motion provenance in evaluator artifacts.
    """

    def __init__(self, *, alpha: float = 0.35, max_dt_s: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.max_dt_s = float(max_dt_s)
        self.last_time_s: float | None = None
        self.vx_smooth = 0.0
        self.vy_smooth = 0.0
        self.x_body_m = 0.0
        self.y_body_m = 0.0
        self.distance_m = 0.0
        self.initialized = False

    def update(self, instant: dict, frame_time_s: float | None) -> dict:
        available = bool(instant.get("available", False))
        raw_vx = float(instant.get("vx_mps", 0.0))
        raw_vy = float(instant.get("vy_mps", 0.0))

        dt_s = 0.0
        if frame_time_s is not None and self.last_time_s is not None:
            dt_s = float(frame_time_s - self.last_time_s)
            if dt_s < 0.0 or dt_s > self.max_dt_s:
                dt_s = 0.0

        if available:
            if not self.initialized:
                self.vx_smooth = raw_vx
                self.vy_smooth = raw_vy
                self.initialized = True
            else:
                self.vx_smooth = self.alpha * raw_vx + (1.0 - self.alpha) * self.vx_smooth
                self.vy_smooth = self.alpha * raw_vy + (1.0 - self.alpha) * self.vy_smooth

            if dt_s > 0.0:
                dx = self.vx_smooth * dt_s
                dy = self.vy_smooth * dt_s
                self.x_body_m += dx
                self.y_body_m += dy
                self.distance_m += float(np.hypot(dx, dy))

        if frame_time_s is not None:
            self.last_time_s = float(frame_time_s)

        smooth_speed = float(np.hypot(self.vx_smooth, self.vy_smooth))
        payload = dict(instant)
        payload.update({
            "available": bool(available and self.initialized),
            "source": "scene_doppler_ransac_smoothed",
            "vx_mps": float(self.vx_smooth if self.initialized else raw_vx),
            "vy_mps": float(self.vy_smooth if self.initialized else raw_vy),
            "speed_mps": smooth_speed if self.initialized else float(instant.get("speed_mps", 0.0)),
            "raw_vx_mps": raw_vx,
            "raw_vy_mps": raw_vy,
            "raw_speed_mps": float(instant.get("speed_mps", float(np.hypot(raw_vx, raw_vy)))),
            "dt_s": round(dt_s, 4),
            "smoothing_alpha": self.alpha,
            "trajectory": self.snapshot(),
        })
        return payload

    def snapshot(self) -> dict:
        return {
            "x_body_m": round(float(self.x_body_m), 3),
            "y_body_m": round(float(self.y_body_m), 3),
            "distance_m": round(float(self.distance_m), 3),
            "vx_smooth_mps": round(float(self.vx_smooth), 3),
            "vy_smooth_mps": round(float(self.vy_smooth), 3),
            "speed_smooth_mps": round(float(np.hypot(self.vx_smooth, self.vy_smooth)), 3),
            "initialized": bool(self.initialized),
        }


def _process_frame_3branch(
    raw_points: list[dict],
    frame_num: int,
    mb: MapBuilder,
    *,
    npz_data,
    branch3_runner: LiveBranch3Runner,
    frame_time_s: float | None = None,
    ego_trajectory: EgoTrajectoryState | None = None,
    ghost_tracker: GhostPersistenceTracker | None = None,
    session_dir: Path | None = None,
) -> tuple[list[dict], dict, dict, dict]:
    # Mirror the live navigation loop: process the current radar frame only.
    # Temporal memory belongs in MapBuilder, not in a replay-only merged cloud.
    persist = _persist_summary(mb, raw_points)

    corrected, _geo_count = _apply_geo_floor_correction(raw_points)
    branch1_input_feature_cols = (
        "ego_doppler_residual_mps",
        "ego_residual_abs_z",
        "ego_inlier_flag",
        "p_ego",
        "p_dir_doppler",
        "rd_entropy",
        "rd_doppler_spread",
        "rd_anisotropy",
        "rd_peak_ratio",
    )
    for pt in corrected:
        for col in branch1_input_feature_cols:
            if col in pt:
                pt[f"branch1_input_{col}"] = pt[col]

    # Step 1 — Runtime ego-Doppler directness annotation (shadow mode)
    try:
        from perception.directness_runtime import (
            DirectnessConfig,
            estimate_ego_velocity as _estimate_ego_vel,
            annotate_directness as _annotate_dir,
            build_directness_evidence as _build_dir_evidence,
        )
        _dir_config = DirectnessConfig()
        _ego_est = _estimate_ego_vel(corrected, _dir_config)
        corrected = _annotate_dir(corrected, _ego_est, _dir_config)
        _dir_evidence = _build_dir_evidence(corrected, _ego_est)
        # Update ghost persistence tracker with current-frame ghost candidates
        if ghost_tracker is not None:
            ghost_tracker.update(corrected)
    except Exception:
        _ego_est = {"available": False, "vx_mps": 0.0, "vy_mps": 0.0, "speed_mps": 0.0,
                    "n_candidate_points": 0, "n_static_points": 0, "residual_mad_mps": float("inf"),
                    "inlier_fraction": 0.0, "sector_entropy": 0.0, "sector_coverage": 0.0, "p_ego": 0.0}
        _dir_evidence = {}

    # Step 3 — Local RD scalar feature extraction (debug mode)
    _rd_scalar_evidence: dict = {"available": False, "points_with_scalars": 0}
    if session_dir is not None and corrected:
        try:
            from models.hybrid_rd_runtime import (
                RuntimeRDPatchConfig,
                extract_patches_and_scalars_for_frame_df as _rd_scalars_for_df,
            )
            import pandas as _pd_rd
            _rd_config = RuntimeRDPatchConfig()
            _frame_df = _pd_rd.DataFrame([
                {
                    "range_bin": pt.get("range_bin", 0),
                    "doppler_bin": pt.get("doppler_bin", 0),
                }
                for pt in corrected
            ])
            if "range_bin" in _frame_df.columns and "doppler_bin" in _frame_df.columns:
                _patches, _scalars = _rd_scalars_for_df(
                    _frame_df,
                    session_dir=session_dir,
                    frame_num=int(frame_num),
                    config=_rd_config,
                )
                for i, (pt, sc) in enumerate(zip(corrected, _scalars)):
                    for k, v in sc.items():
                        pt[k] = v
                # Per-frame summary by class
                _by_cls: dict[str, dict[str, list]] = {}
                for pt, sc in zip(corrected, _scalars):
                    cls = str(pt.get("pred_class", "unknown"))
                    _by_cls.setdefault(cls, {k: [] for k in sc})
                    for k, v in sc.items():
                        _by_cls[cls][k].append(v)
                _cls_means: dict[str, dict[str, float]] = {}
                for cls, col_lists in _by_cls.items():
                    _cls_means[cls] = {
                        k: round(float(sum(v) / len(v)), 4)
                        for k, v in col_lists.items() if v
                    }
                _rd_scalar_evidence = {
                    "available": True,
                    "points_with_scalars": len(_scalars),
                    "by_class": _cls_means,
                    # Flat human summaries for HNM scoring
                    "human_rd_entropy_mean": _cls_means.get("human", {}).get("rd_entropy"),
                    "human_rd_doppler_spread_mean": _cls_means.get("human", {}).get("rd_doppler_spread"),
                    "human_rd_anisotropy_mean": _cls_means.get("human", {}).get("rd_anisotropy"),
                    "human_wheel_sideband_mean": _cls_means.get("human", {}).get("wheel_sideband_score_local"),
                }
        except Exception:
            pass

    ego_instant = mb.estimate_ego_velocity(corrected) if mb is not None else {
        "available": False,
        "vx_mps": 0.0,
        "vy_mps": 0.0,
        "speed_mps": 0.0,
        "n_static_points": 0,
        "n_candidate_points": 0,
        "source": "unavailable",
    }
    ego_velocity = (
        ego_trajectory.update(ego_instant, frame_time_s)
        if ego_trajectory is not None else ego_instant
    )

    if corrected:
        mb.push_classified_points(corrected)
        # Drain the queue synchronously so anomaly queries see the update
        mb._queue.join()

    _acc_evidence: dict = {"available": any("p_acc" in p for p in corrected)}
    if _acc_evidence["available"]:
        _acc_by_cls: dict[str, list[float]] = {}
        for pt in corrected:
            if "p_acc" not in pt:
                continue
            cls = str(pt.get("pred_class", "unknown"))
            _acc_by_cls.setdefault(cls, []).append(float(pt.get("p_acc", 1.0)))
        _acc_evidence.update({
            "by_class": {
                cls: round(float(np.mean(vals)), 4)
                for cls, vals in _acc_by_cls.items() if vals
            },
            "human_mean_p_acc": (
                round(float(np.mean(_acc_by_cls["human"])), 4)
                if _acc_by_cls.get("human") else None
            ),
            "structure_mean_p_acc": (
                round(float(np.mean(_acc_by_cls["structure"])), 4)
                if _acc_by_cls.get("structure") else None
            ),
        })

    scene = aggregate_scene(corrected, window_frames=1, ego_velocity_mps=ego_velocity)
    # Floor recovery is a semantic aid. It must not erase low-Z structure
    # returns before the corridor-blocking geometry pass; April 28 point
    # clouds use z=0 at radar height, so real blocking returns commonly sit
    # below zero.
    scene = _apply_improved_blocking(raw_points, scene)
    scene["map_anomalies"]  = mb.query_anomalies(0.0, 1.0, radius_m=0.5)
    scene["unet_freespace"] = branch3_runner.infer(npz_data, int(frame_num))
    scene.setdefault("evidence_channels", {}).setdefault("directness", {}).update(_dir_evidence)
    scene.setdefault("evidence_channels", {}).setdefault("accumulatability", {}).update(_acc_evidence)
    scene.setdefault("evidence_channels", {}).setdefault("rd_scalars", {}).update(_rd_scalar_evidence)
    scene.setdefault("evidence_channels", {}).setdefault("map", {})
    scene["evidence_channels"]["map"].update({
        "available": True,
        "struct_history": scene["map_anomalies"].get("struct_history", 0.0),
        "free_history": scene["map_anomalies"].get("free_history", 0.0),
        "has_history": scene["map_anomalies"].get("has_history", False),
        "ego_motion_available": bool(ego_velocity.get("available", False)),
        "ego_vx_mps": round(float(ego_velocity.get("vx_mps", 0.0)), 3),
        "ego_vy_mps": round(float(ego_velocity.get("vy_mps", 0.0)), 3),
        "ego_speed_mps": round(float(ego_velocity.get("speed_mps", 0.0)), 3),
        "ego_raw_vx_mps": round(float(ego_velocity.get("raw_vx_mps", ego_velocity.get("vx_mps", 0.0))), 3),
        "ego_raw_vy_mps": round(float(ego_velocity.get("raw_vy_mps", ego_velocity.get("vy_mps", 0.0))), 3),
        "ego_raw_speed_mps": round(float(ego_velocity.get("raw_speed_mps", ego_velocity.get("speed_mps", 0.0))), 3),
        "ego_static_points": int(ego_velocity.get("n_static_points", 0)),
        "ego_candidate_points": int(ego_velocity.get("n_candidate_points", 0)),
    })
    # Step 4 — MapBuilder weighted-update mode stats
    scene.setdefault("evidence_channels", {}).setdefault("map_update", {})
    scene["evidence_channels"]["map_update"].update(dict(mb._map_update_stats))
    scene.setdefault("evidence_channels", {}).setdefault("unet", {})
    scene["evidence_channels"]["unet"].update({
        "available": bool(scene["unet_freespace"].get("available", False)),
        "center": scene["unet_freespace"].get("center", 0.0),
        "left": scene["unet_freespace"].get("left", 0.0),
        "right": scene["unet_freespace"].get("right", 0.0),
        "reason": scene["unet_freespace"].get("reason"),
    })

    decision = compute_nav_decision(scene)
    return corrected, scene, decision, persist


# ---------------------------------------------------------------------------
# Validation overlay stamp
# ---------------------------------------------------------------------------

def _put_text(img, text, xy, color):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _stamp_validation_overlay(img, scene, decision, persist) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    strip_h = 88
    y0 = max(0, h - strip_h)

    # Background strip
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (6, 8, 18), -1)
    cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)
    cv2.line(out, (0, y0), (w, y0), (40, 55, 100), 1)

    anom = scene.get("map_anomalies", {})
    fs   = scene.get("unet_freespace", {})

    def _badge(x, y, label, value, ok_color, dim_color, active):
        col = ok_color if active else dim_color
        text = f"{label}: {value}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pad = 6
        cv2.rectangle(out, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      (12, 16, 32), -1)
        cv2.rectangle(out, (x - pad, y - th - pad), (x + tw + pad, y + pad),
                      col, 1, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    col, 1, cv2.LINE_AA)
        return x + tw + pad * 2 + 10

    # Row 1: Branch 1 — persist scores
    ry = y0 + 26
    x  = 18
    cv2.putText(out, "B1", (x, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (80, 100, 160), 1, cv2.LINE_AA)
    x += 28
    x = _badge(x, ry, "persist_max",  f"{persist['max']:.3f}",
               (100, 210, 110), (80, 90, 110), persist["max"] > 0.01)
    x = _badge(x, ry, "mean",  f"{persist['mean']:.3f}",
               (100, 210, 110), (80, 90, 110), persist["mean"] > 0.01)
    x = _badge(x, ry, "nonzero",  str(persist["nonzero"]),
               (100, 210, 110), (80, 90, 110), persist["nonzero"] > 0)

    # Row 2: Branch 2 — map state
    ry = y0 + 56
    x  = 18
    cv2.putText(out, "B2", (x, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (80, 100, 160), 1, cv2.LINE_AA)
    x += 28
    x = _badge(x, ry, "struct_hist", f"{anom.get('struct_history', 0):.1f}",
               (70, 180, 255), (80, 90, 110), anom.get("struct_history", 0) > 0)
    x = _badge(x, ry, "free_hist", f"{anom.get('free_history', 0):.1f}",
               (70, 180, 255), (80, 90, 110), anom.get("free_history", 0) > 0)
    novel_occ  = bool(anom.get("novel_occupancy", False))
    novel_free = bool(anom.get("novel_free_space", False))
    x = _badge(x, ry, "novel_occ",  "Y" if novel_occ  else "n",
               (80, 80, 220), (70, 78, 100), novel_occ)
    x = _badge(x, ry, "novel_free", "Y" if novel_free else "n",
               (80, 200, 120), (70, 78, 100), novel_free)

    # Row 3 (right side of row 2): Branch 3 — UNet
    if fs.get("available"):
        x = _badge(x, ry, "B3_C", f"{fs.get('center', 0):.2f}",
                   (80, 220, 120), (80, 90, 110), True)
        x = _badge(x, ry, "L",    f"{fs.get('left',   0):.2f}",
                   (80, 160, 220), (80, 90, 110), True)
        x = _badge(x, ry, "R",    f"{fs.get('right',  0):.2f}",
                   (80, 160, 220), (80, 90, 110), True)
        if decision.get("unet_disagree_center"):
            _badge(x, ry, "DISAGREE", "", (40, 200, 200), (80, 90, 110), True)
    else:
        _badge(x, ry, "B3", f"unavail:{fs.get('reason', '?')[:18]}",
               (130, 130, 145), (100, 105, 120), False)

    return out


# ---------------------------------------------------------------------------
# Integrated-frame compositor (color + radar panel overlay)
# ---------------------------------------------------------------------------

def _alpha_rect(frame, x0, y0, x1, y1, color, alpha):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


def _fit_text(text, *, font_scale, thickness, max_width):
    if max_width <= 0:
        return text
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    if size[0] <= max_width:
        return text
    base = text.strip()
    while len(base) > 3:
        base = base[:-1].rstrip()
        candidate = base + "..."
        size, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if size[0] <= max_width:
            return candidate
    return "..."


def _draw_chip(frame, x, y, text, *, fill, text_color=(255, 255, 255),
               max_width=None, font_scale=0.46, thickness=1,
               pad_x=10, pad_y=7) -> int:
    text = _fit_text(text, font_scale=font_scale, thickness=thickness,
                     max_width=max_width if max_width is not None else 10_000)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         font_scale, thickness)
    w = tw + pad_x * 2
    h = th + baseline + pad_y * 2
    _alpha_rect(frame, x, y, x + w, y + h, fill, 0.84)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 32, 52), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad_x, y + pad_y + th),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                thickness, cv2.LINE_AA)
    return w


def _render_radar_panel(session_label, frame_num, frame_idx, n_frames, sim_time_s,
                        points, scene, decision, map_occ, persist,
                        guidance_text, guidance_fired,
                        panel_width, panel_height,
                        ghost_positions: list[tuple[float, float]] | None = None) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    BG_PANEL = "#0a0e1c"
    BG_CARD  = "#101828"
    BORDER   = "#1a2238"
    T_PRI    = "#dde4f5"
    T_SEC    = "#6272a4"
    T_DIM    = "#303850"
    GRID_C   = "#0e1428"
    _ACOL = {
        "stop": "#cf3434", "veer_left": "#d4821e",
        "veer_right": "#d4821e", "slow_down": "#c9a500", "continue": "#27a85e",
    }
    _CCOL = {"structure": "#e05252", "floor": "#4a5568", "human": "#2ece74"}
    _CMRK = {"structure": "s", "floor": ".", "human": "^"}

    action     = decision.get("primary_action", "?")
    action_col = _ACOL.get(action, "#556070")

    fig = plt.figure(figsize=(panel_width / 100, panel_height / 100), dpi=100)
    fig.patch.set_facecolor(BG_PANEL)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.10)

    ax.set_facecolor(BG_PANEL)
    ax.set_xlim(RADAR_PREVIEW_X_HALF_M, -RADAR_PREVIEW_X_HALF_M)
    ax.set_ylim(0.0, RADAR_PREVIEW_Y_MAX_M)
    ax.set_xlabel("Lateral (m)", color=T_SEC, fontsize=7, labelpad=2)
    ax.set_ylabel("Forward (m)", color=T_SEC, fontsize=7, labelpad=2)
    ax.tick_params(colors=T_SEC, labelsize=6, pad=1)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, color=GRID_C, linestyle="-", linewidth=0.4, alpha=1.0)

    od = scene.get("open_directions", {})
    dirs_unknown = (
        scene.get("open_directions_confidence") == "unknown"
        or decision.get("reason") == "low_confidence"
    )

    def _zone(clear_key):
        if dirs_unknown:
            return "#0e1228", 0.55
        return ("#091a10", 0.55) if od.get(clear_key, True) else ("#1a0808", 0.60)

    lc, la = _zone("left_clear")
    cc, ca = _zone("center_clear")
    rc, ra = _zone("right_clear")
    ax.axvspan(0.5,  RADAR_PREVIEW_X_HALF_M,  color=lc, alpha=la, zorder=0)
    ax.axvspan(-0.5, 0.5,                     color=cc, alpha=ca, zorder=0)
    ax.axvspan(-RADAR_PREVIEW_X_HALF_M, -0.5, color=rc, alpha=ra, zorder=0)

    for xv in (-0.5, 0.5):
        ax.axvline(xv, color="#202840", lw=0.8, ls="--", alpha=0.7, zorder=1)

    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    for r in (1.0, 1.5, 2.5, 3.5):
        if r > RADAR_PREVIEW_Y_MAX_M:
            continue
        rx = r * np.sin(theta)
        ry = r * np.cos(theta)
        mask = np.abs(rx) <= RADAR_PREVIEW_X_HALF_M
        ax.plot(rx[mask], ry[mask], color="#172030",
                lw=0.8, ls=":", alpha=0.90, zorder=1)

    for xt, zt in ((1.35, "L"), (0.0, "C"), (-1.35, "R")):
        ax.text(xt, 0.18, zt, color=T_DIM, fontsize=6,
                ha="center", va="bottom", fontweight="bold")

    ax.add_patch(plt.Circle((0, 0), 0.18, color="#3a8fd9",
                             fill=False, lw=0.8, alpha=0.28, zorder=9))
    ax.scatter([0], [0], marker="^", s=130, color="#3a8fd9",
               zorder=10, edgecolors="#90c8f0", linewidths=0.8)

    direction = decision.get("best_direction", "center")
    adx = {"left": 1.0, "right": -1.0}.get(direction, 0.0)
    ady = 0.5 if direction in ("left", "right") else (1.0 if direction == "center" else 0.0)
    if ady > 0:
        arw = _ACOL.get(action, "#27a85e")
        ax.annotate("", xy=(adx, ady), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=arw, lw=5.0, alpha=0.12), zorder=6)
        ax.annotate("", xy=(adx, ady), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=arw, lw=2.0, alpha=0.95), zorder=7)

    by_class: dict[str, list] = {"structure": [], "floor": [], "human": []}
    ghost_pts: list[tuple[float, float]] = []
    for p in points:
        cls = p.get("pred_class", "floor")
        if cls in by_class:
            by_class[cls].append((float(p["x"]), float(p["y"])))
        # Mark probable ghost returns: low p_dir_doppler indicates Doppler-inconsistent return
        p_dir = float(p.get("p_dir_doppler", 1.0))
        if p_dir < 0.15:
            ghost_pts.append((float(p["x"]), float(p["y"])))
    for cls, pts_xy in by_class.items():
        if not pts_xy:
            continue
        col = _CCOL[cls]
        px_ = [p[0] for p in pts_xy]
        py_ = [p[1] for p in pts_xy]
        s = 20 if cls == "floor" else 50
        ax.scatter(px_, py_, c=col, marker=_CMRK[cls],
                   s=s * 4, alpha=0.08, zorder=3, linewidths=0)
        ax.scatter(px_, py_, c=col, marker=_CMRK[cls],
                   s=s, alpha=0.88 if cls != "floor" else 0.55,
                   zorder=4,
                   edgecolors="#ffffff" if cls == "human" else "none",
                   linewidths=0.4 if cls == "human" else 0)

    # Ghost overlay: diamond markers on top of model-predicted markers
    if ghost_pts:
        gx = [p[0] for p in ghost_pts]
        gy = [p[1] for p in ghost_pts]
        ax.scatter(gx, gy, marker="D", c="#d4a017", s=65, alpha=0.70,
                   zorder=5, edgecolors="#ffcc00", linewidths=0.7)

    # Doppler residual gradient vectors — drawn after all scatter layers so
    # arrows sit above point markers.  Only annotated points get arrows.
    _draw_doppler_vectors(ax, points, scale_mps_to_m=0.50, alpha=0.76, zorder=6)

    leg_handles = [mpatches.Patch(color=_CCOL[c], label=c)
                   for c in ("structure", "floor", "human") if by_class.get(c)]
    if ghost_pts:
        leg_handles.append(mpatches.Patch(color="#d4a017", label="ghost"))
    if leg_handles:
        ax.legend(handles=leg_handles, loc="upper right", fontsize=5.5,
                  facecolor=BG_CARD, labelcolor=T_PRI, framealpha=0.88,
                  edgecolor=BORDER, borderpad=0.4, handlelength=1.0)

    counts = {cls: len(pts_xy) for cls, pts_xy in by_class.items()}
    ax.text(0.02, 0.98,
            f"RADAR  f{frame_num:03d}  ({frame_idx+1}/{n_frames})",
            transform=ax.transAxes, color=T_DIM, fontsize=7,
            fontweight="bold", va="top", fontfamily="monospace")
    ax.text(0.98, 0.98, f"t={sim_time_s:.1f}s",
            transform=ax.transAxes, color=T_SEC, fontsize=6.5, va="top", ha="right")
    ax.text(0.02, 0.89,
            f"S={counts['structure']} F={counts['floor']} H={counts['human']}",
            transform=ax.transAxes, color=T_SEC, fontsize=6.5, va="top",
            fontfamily="monospace")

    action_label = action.upper().replace("_", " ")
    ax.text(0.02, 0.03,
            f"{action_label}  persist={persist['max']:.2f}",
            transform=ax.transAxes, color=action_col, fontsize=7,
            fontweight="bold", va="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#08101e",
                      edgecolor=action_col, alpha=0.90, linewidth=0.7))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    if img.shape[:2] != (panel_height, panel_width):
        img = cv2.resize(img, (panel_width, panel_height))
    return img


def _compose_integrated_frame(
    color_frame, radar_panel, *, session_label, frame_num, n_frames,
    frame_idx, sim_time_s, decision, scene, persist, guidance_text, guidance_fired,
) -> np.ndarray:
    out  = color_frame.copy()
    h, w = out.shape[:2]
    top_h    = max(80, h // 8)
    bottom_h = max(88, h // 7)
    margin   = max(14, h // 44)

    # Top and bottom dark overlays
    _alpha_rect(out, 0, 0, w, top_h,        (5, 7, 16), 0.72)
    _alpha_rect(out, 0, h - bottom_h, w, h, (5, 7, 16), 0.75)

    # Accent line under top bar
    cv2.line(out, (0, top_h), (w, top_h), (35, 50, 95), 1)
    cv2.line(out, (0, h - bottom_h), (w, h - bottom_h), (35, 50, 95), 1)

    action     = str(decision.get("primary_action", "?")).lower()
    urgency    = str(decision.get("urgency", ""))
    action_col = _ACTION_COLOR.get(action, "#6c757d")
    action_bgr = _hex_to_bgr(action_col)

    # Session + frame info
    short = session_label[-40:] if len(session_label) > 40 else session_label
    cv2.putText(out, short, (margin, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 238, 255), 1, cv2.LINE_AA)
    cv2.putText(out,
                f"f{frame_num:03d}  ({frame_idx+1}/{n_frames})  t={sim_time_s:.1f}s",
                (margin, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 125, 165), 1, cv2.LINE_AA)

    # Action badge — top right
    action_label = f"  {action.upper().replace('_', ' ')}  "
    if urgency:
        action_label += f"| {urgency.upper()}  "
    badge_x = w - margin - 300
    _draw_chip(out, badge_x, 12, action_label,
               fill=action_bgr, text_color=(255, 255, 255),
               max_width=280, font_scale=0.54, pad_x=14, pad_y=8)

    # Branch status chips — lower part of top bar
    anom = scene.get("map_anomalies", {})
    fs   = scene.get("unet_freespace", {})

    novel_occ  = bool(anom.get("novel_occupancy",  False))
    novel_free = bool(anom.get("novel_free_space", False))
    b1_txt = f"B1  persist {persist['max']:.2f} / {persist['nonzero']}"
    b2_occ  = "Y" if novel_occ  else "n"
    b2_free = "Y" if novel_free else "n"
    b2_txt = f"B2  occ={b2_occ}  free={b2_free}  sh={anom.get('struct_history', 0):.1f}"

    b1_fill = (28, 80, 48)
    b2_fill = (100, 62, 18) if novel_occ else (30, 58, 100)

    chip_y = top_h - 32
    x = margin
    x += _draw_chip(out, x, chip_y, b1_txt, fill=b1_fill,
                    max_width=max(180, w // 5), font_scale=0.42, pad_x=10, pad_y=6)
    x += 8
    x += _draw_chip(out, x, chip_y, b2_txt, fill=b2_fill,
                    max_width=max(220, w // 4), font_scale=0.42, pad_x=10, pad_y=6)
    x += 8
    if fs.get("available"):
        b3_txt  = (f"B3  C={fs.get('center',0):.2f}  "
                   f"L={fs.get('left',0):.2f}  R={fs.get('right',0):.2f}")
        b3_fill = (20, 90, 55)
        if decision.get("unet_disagree_center"):
            b3_txt += "  ▲"
            b3_fill = (70, 80, 18)
    else:
        b3_txt  = f"B3  {fs.get('reason', 'unavail')[:20]}"
        b3_fill = (55, 58, 68)
    _draw_chip(out, x, chip_y, b3_txt, fill=b3_fill,
               max_width=max(220, w // 4), font_scale=0.42, pad_x=10, pad_y=6)

    # Radar panel — pinned top-right of live area
    ph, pw_panel = radar_panel.shape[:2]
    px = w - pw_panel - margin
    py = top_h + margin
    if py + ph > h - bottom_h - margin:
        py = max(top_h + margin, h - bottom_h - margin - ph)

    # Panel shadow
    _alpha_rect(out, px - 4, py - 4, px + pw_panel + 4, py + ph + 4, (0, 0, 0), 0.40)
    panel_region = out[py:py + ph, px:px + pw_panel]
    cv2.addWeighted(radar_panel, 0.96, panel_region, 0.04, 0, panel_region)
    cv2.rectangle(out, (px, py), (px + pw_panel, py + ph),
                  (50, 65, 105), 1, cv2.LINE_AA)
    # Accent top edge on panel
    cv2.line(out, (px, py), (px + pw_panel, py), action_bgr, 2)

    # Bottom bar — guidance
    od = scene.get("open_directions", {})
    dirs_unknown = (
        scene.get("open_directions_confidence") == "unknown"
        or decision.get("reason") == "low_confidence"
    )

    def _dir_txt(key):
        if dirs_unknown:
            return "?"
        return "CLR" if od.get(key, True) else "BLK"

    def _dir_col(key):
        if dirs_unknown:
            return (160, 140, 40)
        return (60, 200, 100) if od.get(key, True) else (80, 70, 200)

    dir_y = h - bottom_h + 30
    cv2.putText(out, "DIRS:", (margin, dir_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 105, 145), 1, cv2.LINE_AA)
    for i, (lbl, key) in enumerate(
            [("LEFT", "left_clear"), ("CTR", "center_clear"), ("RIGHT", "right_clear")]):
        dx = margin + 55 + i * 110
        cv2.putText(out, lbl, (dx, dir_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 82, 115), 1, cv2.LINE_AA)
        cv2.putText(out, _dir_txt(key), (dx, dir_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, _dir_col(key), 1, cv2.LINE_AA)

    guide_col = (80, 210, 100) if guidance_fired else (155, 165, 192)
    gy = h - bottom_h + 30
    guide_x = margin + 380
    cv2.putText(out, "GUIDANCE" if guidance_fired else "cached",
                (guide_x, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                guide_col, 1, cv2.LINE_AA)
    wrapped = textwrap.wrap(guidance_text or "Analysing scene...", width=56)
    for li, line in enumerate(wrapped[:2]):
        cv2.putText(out, line, (guide_x, gy + 18 + li * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 224, 238), 1, cv2.LINE_AA)

    reason_txt = (
        f"{action.upper().replace('_', ' ')}  /  "
        f"{decision.get('reason', '?')}  /  "
        f"block={decision.get('blocking_confidence', scene.get('blocking', {}).get('center_confidence', '?'))}"
    )
    cv2.putText(out, reason_txt, (margin, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 95, 135), 1, cv2.LINE_AA)

    # Progress bar at very bottom
    bar_y  = h - 6
    bar_x0 = margin
    bar_x1 = w - margin
    cv2.rectangle(out, (bar_x0, bar_y - 5), (bar_x1, bar_y), (22, 28, 50), -1)
    prog = int((frame_idx + 1) / max(n_frames, 1) * (bar_x1 - bar_x0))
    cv2.rectangle(out, (bar_x0, bar_y - 5), (bar_x0 + prog, bar_y), action_bgr, -1)
    cv2.putText(out, f"{frame_idx+1}/{n_frames}",
                (w - margin - 76, bar_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 200, 225), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------

def _session_list_from_args(args) -> list[tuple[str, Path]]:
    if args.csv is not None:
        if args.csv.name == "labeled_radar_points_v4.csv":
            raise ValueError(
                "Replay requires a raw ADC-derived session CSV, not "
                "labeled_radar_points_v4.csv."
            )
        return [(args.csv.parent.name, args.csv)]
    if args.session is not None:
        return [(args.session.name, _raw_session_csv_path(args.session))]
    sessions = _find_raw_session_csvs(args.sessions_dir)
    if getattr(args, "session_names", None):
        requested = [n.strip() for n in args.session_names.split(",") if n.strip()]
        name_to_entry = {name: entry for name, entry in sessions}
        filtered = [(n, name_to_entry[n]) for n in requested if n in name_to_entry]
        missing = [n for n in requested if n not in name_to_entry]
        if missing:
            raise ValueError(f"Session names not found under {args.sessions_dir}: {missing}")
        return filtered
    return _select_session_subset(
        sessions,
        mode=args.session_selection,
        max_sessions=args.max_sessions,
        stride=args.session_stride,
        seed=args.session_seed,
    )


# ---------------------------------------------------------------------------
# Per-frame / per-point prediction export
# ---------------------------------------------------------------------------

FRAME_PRED_FIELDNAMES: list[str] = [
    "run_id", "session", "frame_idx", "frame_num", "radar_time_ms", "input_mode",
    "branch1_ready", "n_raw_points",
    "n_pred_structure", "n_pred_floor", "n_pred_human",
    "left_clear", "center_clear", "right_clear",
    "blocking_confidence", "center_effective_points", "center_dynamic_points",
    "floor_detected", "floor_confidence",
    "human_count", "center_human_present",
    "unet_available", "unet_center", "unet_left", "unet_right",
    "map_has_history", "map_novel_occupancy", "map_novel_free_space",
    "primary_action", "reason", "urgency", "best_direction",
    "reused_semantics", "errors",
]

POINT_PRED_FIELDNAMES: list[str] = [
    "run_id", "session", "frame_num", "point_idx",
    "x", "y", "z", "range_m", "doppler", "snr",
    "range_bin", "doppler_bin",
    "pred_class",
    "pred_prob_structure", "pred_prob_floor", "pred_prob_human",
    "prob_structure", "prob_floor", "prob_human",
    "p_acc", "acc_confidence",
    "ego_doppler_residual_mps", "ego_residual_abs_z", "ego_inlier_flag",
    "p_ego", "p_dir_doppler",
    "rd_entropy", "rd_doppler_spread", "rd_anisotropy", "rd_peak_ratio",
    "branch1_input_ego_doppler_residual_mps",
    "branch1_input_ego_residual_abs_z",
    "branch1_input_ego_inlier_flag",
    "branch1_input_p_ego",
    "branch1_input_p_dir_doppler",
    "branch1_input_rd_entropy",
    "branch1_input_rd_doppler_spread",
    "branch1_input_rd_anisotropy",
    "branch1_input_rd_peak_ratio",
    "feature_status",
]


def _build_frame_prediction_row(
    run_id: str,
    session: str,
    input_mode: str,
    state: dict,
) -> dict:
    scene    = state.get("scene", {})
    decision = state.get("decision", {})
    od       = scene.get("open_directions", {})
    blocking = scene.get("blocking", {})
    fs       = scene.get("unet_freespace", {})
    anom     = scene.get("map_anomalies", {})
    humans   = scene.get("humans", {})
    detections = humans.get("detections", []) or []
    n_humans = len(detections)
    center_human = any(d.get("sector") == "center" for d in detections)
    corrected = state.get("corrected", [])
    pc = state.get("point_counts", {})
    branch1_ready = bool(corrected) and not state.get("reused_semantics", False)
    return {
        "run_id":                   run_id,
        "session":                  session,
        "frame_idx":                state.get("frame_idx"),
        "frame_num":                state.get("frame_num"),
        "radar_time_ms":            state.get("radar_time_ms"),
        "input_mode":               input_mode,
        "branch1_ready":            branch1_ready,
        "n_raw_points":             state.get("n_raw_points", len(corrected)),
        "n_pred_structure":         pc.get("structure", 0),
        "n_pred_floor":             pc.get("floor", 0),
        "n_pred_human":             pc.get("human", 0),
        "left_clear":               od.get("left_clear"),
        "center_clear":             od.get("center_clear"),
        "right_clear":              od.get("right_clear"),
        "blocking_confidence":      (
            decision.get("blocking_confidence")
            or blocking.get("center_confidence")
        ),
        "center_effective_points":  blocking.get("center_effective_points"),
        "center_dynamic_points":    blocking.get("center_dynamic_points"),
        "floor_detected":           decision.get("floor_detected"),
        "floor_confidence":         decision.get("floor_confidence"),
        "human_count":              n_humans,
        "center_human_present":     center_human,
        "unet_available":           bool(fs.get("available", False)),
        "unet_center":              fs.get("center"),
        "unet_left":                fs.get("left"),
        "unet_right":               fs.get("right"),
        "map_has_history":          bool(float(anom.get("struct_history", 0)) > 0),
        "map_novel_occupancy":      bool(anom.get("novel_occupancy", False)),
        "map_novel_free_space":     bool(anom.get("novel_free_space", False)),
        "primary_action":           decision.get("primary_action"),
        "reason":                   decision.get("reason"),
        "urgency":                  decision.get("urgency"),
        "best_direction":           decision.get("best_direction"),
        "reused_semantics":         bool(state.get("reused_semantics", False)),
        "errors":                   "",
    }


def _build_point_prediction_rows(
    run_id: str,
    session: str,
    state: dict,
) -> list[dict]:
    frame_num = state.get("frame_num")
    rows = []
    for i, pt in enumerate(state.get("corrected", [])):
        rows.append({
            "run_id":         run_id,
            "session":        session,
            "frame_num":      frame_num,
            "point_idx":      i,
            "x":              pt.get("x"),
            "y":              pt.get("y"),
            "z":              pt.get("z"),
            "range_m":        pt.get("range_m"),
            "doppler":        pt.get("doppler"),
            "snr":            pt.get("snr"),
            "range_bin":      pt.get("range_bin"),
            "doppler_bin":    pt.get("doppler_bin"),
            "pred_class":     pt.get("pred_class"),
            "pred_prob_structure": pt.get("pred_prob_structure", pt.get("prob_structure")),
            "pred_prob_floor":     pt.get("pred_prob_floor", pt.get("prob_floor")),
            "pred_prob_human":     pt.get("pred_prob_human", pt.get("prob_human")),
            "prob_structure":      pt.get("prob_structure", pt.get("pred_prob_structure")),
            "prob_floor":          pt.get("prob_floor", pt.get("pred_prob_floor")),
            "prob_human":          pt.get("prob_human", pt.get("pred_prob_human")),
            "p_acc":          pt.get("p_acc"),
            "acc_confidence": pt.get("acc_confidence"),
            "ego_doppler_residual_mps": pt.get("ego_doppler_residual_mps"),
            "ego_residual_abs_z":       pt.get("ego_residual_abs_z"),
            "ego_inlier_flag":          pt.get("ego_inlier_flag"),
            "p_ego":                    pt.get("p_ego"),
            "p_dir_doppler":            pt.get("p_dir_doppler"),
            "rd_entropy":               pt.get("rd_entropy"),
            "rd_doppler_spread":        pt.get("rd_doppler_spread"),
            "rd_anisotropy":            pt.get("rd_anisotropy"),
            "rd_peak_ratio":            pt.get("rd_peak_ratio"),
            "branch1_input_ego_doppler_residual_mps": pt.get("branch1_input_ego_doppler_residual_mps"),
            "branch1_input_ego_residual_abs_z":       pt.get("branch1_input_ego_residual_abs_z"),
            "branch1_input_ego_inlier_flag":          pt.get("branch1_input_ego_inlier_flag"),
            "branch1_input_p_ego":                    pt.get("branch1_input_p_ego"),
            "branch1_input_p_dir_doppler":            pt.get("branch1_input_p_dir_doppler"),
            "branch1_input_rd_entropy":               pt.get("branch1_input_rd_entropy"),
            "branch1_input_rd_doppler_spread":        pt.get("branch1_input_rd_doppler_spread"),
            "branch1_input_rd_anisotropy":            pt.get("branch1_input_rd_anisotropy"),
            "branch1_input_rd_peak_ratio":            pt.get("branch1_input_rd_peak_ratio"),
            "feature_status": pt.get("feature_status", ""),
        })
    return rows


class _PredExporter:
    """Collect per-frame and per-point prediction rows and write CSVs at flush time."""

    def __init__(
        self,
        frame_csv_path: "Path | None",
        point_csv_path: "Path | None",
        run_id: str,
        input_mode: str,
    ) -> None:
        self.frame_csv_path = frame_csv_path
        self.point_csv_path = point_csv_path
        self.run_id = run_id
        self.input_mode = input_mode
        self._frame_rows: list[dict] = []
        self._point_rows: list[dict] = []
        self.enabled = bool(frame_csv_path or point_csv_path)

    def append(self, session: str, state: dict) -> None:
        if not self.enabled:
            return
        if self.frame_csv_path:
            self._frame_rows.append(
                _build_frame_prediction_row(self.run_id, session, self.input_mode, state)
            )
        if self.point_csv_path:
            self._point_rows.extend(
                _build_point_prediction_rows(self.run_id, session, state)
            )

    def flush(self) -> dict:
        """Write accumulated rows to disk and return a summary dict for stats."""
        result: dict = {}
        if self.frame_csv_path and self._frame_rows:
            self.frame_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.frame_csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=FRAME_PRED_FIELDNAMES, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(self._frame_rows)
            print(
                f"[pred] {len(self._frame_rows)} frame rows -> {self.frame_csv_path}",
                flush=True,
            )
            result["frame_predictions_csv"] = str(self.frame_csv_path)
            result["frame_predictions_rows"] = len(self._frame_rows)
        if self.point_csv_path and self._point_rows:
            self.point_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.point_csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=POINT_PRED_FIELDNAMES, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(self._point_rows)
            print(
                f"[pred] {len(self._point_rows)} point rows -> {self.point_csv_path}",
                flush=True,
            )
            result["point_predictions_csv"] = str(self.point_csv_path)
            result["point_predictions_rows"] = len(self._point_rows)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--sessions-dir", type=Path, default=DATA_DIR)
    src.add_argument("--session",      type=Path)
    src.add_argument("--csv",          type=Path)
    parser.add_argument("--model-pt", type=Path, default=_default_model_pt(),
                        help="Branch 1 RD-patch checkpoint.")
    parser.add_argument("--unet-pt", type=Path, default=DEFAULT_UNET_PT,
                        help="U-Net freespace checkpoint for Branch 3 (default: %(default)s).")
    parser.add_argument("--extrinsics-json", type=Path, default=DEFAULT_EXTRINSICS_JSON,
                        help="Radar-to-camera extrinsics metadata to record for evaluation provenance.")
    parser.add_argument("--max-sessions",       type=int, default=5)
    parser.add_argument("--session-selection",
                        choices=("sequential", "stride", "random"),
                        default="sequential",
                        help="How to choose sessions from the sorted session list.")
    parser.add_argument("--session-names", type=str, default=None,
                        help="Comma-separated list of session names to replay (overrides --session-selection). "
                             "Names must match directory names under --sessions-dir.")
    parser.add_argument("--session-stride", type=int, default=5,
                        help="When --session-selection stride is used, take every Nth session.")
    parser.add_argument("--session-seed", type=int, default=42,
                        help="Seed for --session-selection random.")
    parser.add_argument("--guidance-interval-s", type=float, default=5.0)
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--ui-style",
                        choices=("integrated", "stacked", "radar"), default="integrated")
    parser.add_argument("--output",      type=Path,
                        default=THIS_DIR / "branch3_simulation.mp4")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--fps",         type=float, default=10.0)
    parser.add_argument("--width",       type=int, default=1280)
    parser.add_argument("--height",      type=int, default=720)
    parser.add_argument("--no-video",    action="store_true")
    parser.add_argument("--no-color-video", action="store_true",
                        help="Disable color video compositing.")
    parser.add_argument("--geo-floor-z", type=float, default=GEO_FLOOR_Z_THRESH)
    parser.add_argument("--geo-floor-y-max", type=float, default=GEO_FLOOR_Y_MAX)
    parser.add_argument("--geo-floor-doppler-max", type=float, default=GEO_FLOOR_DOPPLER_SANITY_MPS)
    parser.add_argument("--enable-geo-floor-reclassify", action="store_true",
                        help="Destructively relabel low static structure/human points as floor. Disabled by default for April 28 data.")
    parser.add_argument("--blocking-z-floor", type=float, default=BLOCKING_Z_FLOOR)
    parser.add_argument("--blocking-y-ceiling", type=float, default=BLOCKING_Y_CEILING)
    parser.add_argument("--blocking-static-weight", type=float, default=BLOCKING_STATIC_WEIGHT)
    parser.add_argument("--blocking-ambig-weight", type=float, default=BLOCKING_AMBIG_WEIGHT)
    parser.add_argument("--blocking-dynamic-weight", type=float, default=BLOCKING_DYNAMIC_WEIGHT)
    parser.add_argument("--center-effective-thresh", type=float, default=BLOCKING_CENTER_EFFECTIVE_THRESH)
    parser.add_argument("--side-effective-thresh", type=float, default=BLOCKING_SIDE_EFFECTIVE_THRESH)
    parser.add_argument("--center-dynamic-thresh", type=int, default=BLOCKING_CENTER_DYNAMIC_THRESH)
    parser.add_argument("--side-dynamic-thresh", type=int, default=BLOCKING_SIDE_DYNAMIC_THRESH)
    parser.add_argument("--empty-frame-mode", choices=("zero", "hold_last"), default="hold_last")
    parser.add_argument("--continuous-map-across-sessions", action="store_true",
                        help="Keep one MapBuilder across all replayed sessions. Default resets map state per session.")
    parser.add_argument("--map-update-mode",
                        choices=["uniform", "ego_only", "ego_directness", "learned_acc"],
                        default="uniform",
                        help="Map accumulation weighting mode: uniform (A0, default), ego_only (A1), "
                             "ego_directness (A2). ego_directness suppresses low-directness ghost points "
                             "from persistent map accumulation. learned_acc uses the optional Step 8 "
                             "p_acc head when present.")
    parser.add_argument("--ego-smoothing-alpha", type=float, default=0.35,
                        help="Causal smoothing factor for radar ego velocity before Doppler compensation.")
    parser.add_argument("--ego-max-dt-s", type=float, default=0.5,
                        help="Maximum frame interval integrated into the replay ego trajectory.")
    parser.add_argument("--debug-frame-snapshots", action="store_true",
                        help="Save sampled rendered frames and a manifest JSON under the output tree.")
    parser.add_argument("--debug-frame-snapshot-stride", type=int, default=10,
                        help="Save every Nth rendered frame when debug snapshots are enabled.")
    parser.add_argument("--export-frame-predictions", type=Path, default=None, metavar="CSV",
                        help="Write per-frame prediction rows to this CSV path after all sessions complete.")
    parser.add_argument("--export-point-predictions", type=Path, default=None, metavar="CSV",
                        help="Write per-point prediction rows (one row per emitted radar point) to this CSV path.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Stable identifier for this evaluation run recorded in every exported row. "
                             "Auto-generated UUID if omitted.")
    args = parser.parse_args()
    _apply_threshold_overrides(args)

    session_list = _session_list_from_args(args)
    for _name, csv_path in session_list:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
    _load_rd_patch_checkpoint(args.model_pt)

    branch3_runner = LiveBranch3Runner(args.unet_pt)
    extrinsics_summary = _load_extrinsics_summary(args.extrinsics_json)

    _map_update_mode = getattr(args, "map_update_mode", "uniform")

    continuous_mb = None
    if args.continuous_map_across_sessions:
        continuous_path = Path(f"/tmp/branch3_replay_grid_{os.getpid()}_continuous.npz")
        try:
            continuous_path.unlink()
        except FileNotFoundError:
            pass
        continuous_mb = MapBuilder(save_path=continuous_path, save_every=1_000_000,
                                   map_update_mode=_map_update_mode)
        continuous_mb.start()

    base_output = args.output
    debug_root = _debug_snapshot_root(base_output)
    stacked_color_panel_height = 0
    if args.ui_style == "stacked" and not args.no_video and not args.no_color_video:
        stacked_color_panel_height = _probe_color_panel_height(session_list, args.width)
        if stacked_color_panel_height:
            print(f"Color video detected — vertical output: "
                  f"{args.width}x{stacked_color_panel_height} over {args.width}x{args.height}")
        else:
            print("No color videos found — radar-only output.")

    counts: dict[str, int] = {}
    stats = {
        "sessions": [name for name, _ in session_list],
        "session_inputs": {name: str(path) for name, path in session_list},
        "session_selection_mode": args.session_selection,
        "session_selection_stride": int(args.session_stride),
        "session_selection_seed": int(args.session_seed),
        "branch3_mode": "live_loop_equivalent",
        "input_mode": "rd_patch_model",
        "ui_style": args.ui_style,
        "frames": 0,
        "ego_motion_available_frames": 0,
        "ego_smoothing_alpha": float(args.ego_smoothing_alpha),
        "ego_max_dt_s": float(args.ego_max_dt_s),
        "ego_trajectory_by_session": {},
        "branch3_available_frames": 0,
        "unet_disagree_frames": 0,
        "novel_occupancy_frames": 0,
        "novel_free_space_frames": 0,
        "persist_nonzero_frames": 0,
        "decision_counts": counts,
        "reason_counts": {},
        "human_motion_counts": {
            "detections": 0,
            "raw_approaching": 0,
            "ego_relative_approaching": 0,
            "raw_approach_suppressed_by_ego": 0,
        },
        "open_direction_counts": {},
        "debug_frame_snapshots": {},
        "output": str(args.output),
        "cfg_path": str(DEFAULT_CFG_PATH),
        "model_pt": str(args.model_pt),
        "unet_pt": str(args.unet_pt),
        "radar_camera_extrinsics": extrinsics_summary,
        "real_unet_ready": bool(branch3_runner.ready),
        "real_unet_reason": branch3_runner.reason or None,
        "debug_frame_snapshots_enabled": bool(args.debug_frame_snapshots),
        "debug_frame_snapshot_stride": int(args.debug_frame_snapshot_stride),
        "debug_frame_snapshot_root": str(debug_root),
        "continuous_map_across_sessions": bool(args.continuous_map_across_sessions),
        "map_update_mode": _map_update_mode,
        "run_id": None,
        "export_frame_predictions": None,
        "export_point_predictions": None,
    }

    run_id = args.run_id or str(uuid.uuid4())
    stats["run_id"] = run_id
    pred_exporter = _PredExporter(
        frame_csv_path=args.export_frame_predictions,
        point_csv_path=args.export_point_predictions,
        run_id=run_id,
        input_mode="rd_patch_model",
    )

    print("=" * 72)
    print("3branch replay simulation")
    print(f"sessions={len(session_list)} branch3_mode=live_loop_equivalent "
          f"input_mode=rd_patch_model output={args.output}")
    print(f"session_selection={args.session_selection} "
          f"stride={args.session_stride} seed={args.session_seed}")
    print(f"model_pt={args.model_pt}")
    print("model input: raw ADC-derived <session>.csv files")
    print(f"extrinsics={args.extrinsics_json} "
          f"available={extrinsics_summary.get('available', False)}")
    print(f"thresholds: geo_floor_y<{GEO_FLOOR_Y_MAX}m  "
          f"geo_floor_z<{GEO_FLOOR_Z_THRESH}m  "
          f"geo_floor_doppler<{GEO_FLOOR_DOPPLER_SANITY_MPS}m/s  "
          f"geo_floor_reclass={'on' if GEO_FLOOR_RECLASSIFY_ENABLED else 'off'}  "
          f"blocking_z>{BLOCKING_Z_FLOOR}m  blocking_y<{BLOCKING_Y_CEILING}m  "
          f"doppler_static_weight={BLOCKING_STATIC_WEIGHT}")
    print("=" * 72)

    guidance_every = max(1, int(args.guidance_interval_s / FRAME_INTERVAL_S))
    last_guidance  = "Analysing scene..."
    last_semantic_state = None

    use_color_sync = False  # set per-session below

    for sess_idx, (session_name, csv_path) in enumerate(session_list, start=1):
        if continuous_mb is not None:
            mb = continuous_mb
            map_mode = "continuous"
        else:
            map_path = Path(f"/tmp/branch3_replay_grid_{os.getpid()}_{sess_idx}.npz")
            try:
                map_path.unlink()
            except FileNotFoundError:
                pass
            mb = MapBuilder(save_path=map_path, save_every=1_000_000,
                            map_update_mode=_map_update_mode)
            mb.start()
            map_mode = "session_reset"

        last_guidance = "Analysing scene..."
        last_semantic_state = None
        ego_trajectory = EgoTrajectoryState(
            alpha=args.ego_smoothing_alpha,
            max_dt_s=args.ego_max_dt_s,
        )
        ghost_tracker = GhostPersistenceTracker()

        session_output = _session_output_path(
            base_output, session_name, len(session_list), args.ui_style)
        args.output = session_output
        if not args.no_video:
            stats.setdefault("session_outputs", {})[session_name] = str(session_output)
        video_frames: list[np.ndarray] = []
        debug_enabled = bool(args.debug_frame_snapshots and args.debug_frame_snapshot_stride > 0)
        debug_stride = max(1, int(args.debug_frame_snapshot_stride))
        debug_session_dir = debug_root / session_name
        debug_samples: list[dict] = []

        def _capture_debug_snapshot(image: np.ndarray, state: dict,
                                    *, output_frame_idx: int,
                                    video_time_ms: int | None,
                                    source: str) -> None:
            if not debug_enabled:
                return
            if output_frame_idx != 0 and output_frame_idx % debug_stride != 0:
                return
            debug_session_dir.mkdir(parents=True, exist_ok=True)
            sample_idx = len(debug_samples)
            frame_num = int(state["frame_num"])
            frame_idx = int(state["frame_idx"])
            radar_time_ms = state.get("radar_time_ms")
            sim_time_s = float(state.get("sim_time_s", 0.0))
            img_name = (
                f"sample_{sample_idx:04d}_out{output_frame_idx:05d}_"
                f"f{frame_num:03d}.png"
            )
            img_path = debug_session_dir / img_name
            ok = cv2.imwrite(str(img_path), image)
            debug_samples.append({
                "sample_idx": sample_idx,
                "source": source,
                "image": img_name,
                "image_path": str(img_path),
                "output_frame_idx": output_frame_idx,
                "frame_idx": frame_idx,
                "frame_num": frame_num,
                "radar_time_ms": int(radar_time_ms) if radar_time_ms is not None else None,
                "video_time_ms": int(video_time_ms) if video_time_ms is not None else None,
                "sim_time_s": sim_time_s,
                "point_count": int(state.get("n_points", 0)),
                "point_counts": state.get("point_counts", {}),
                "decision": {
                    "primary_action": state["decision"].get("primary_action"),
                    "reason": state["decision"].get("reason"),
                    "urgency": state["decision"].get("urgency"),
                    "best_direction": state["decision"].get("best_direction"),
                    "left_clear": state["decision"].get("left_clear"),
                    "center_clear": state["decision"].get("center_clear"),
                    "right_clear": state["decision"].get("right_clear"),
                    "floor_detected": state["decision"].get("floor_detected"),
                    "floor_confidence": state["decision"].get("floor_confidence"),
                    "confidence": state["decision"].get("confidence"),
                    "blocking_confidence": state["decision"].get("blocking_confidence"),
                    "nearest_human_ft": state["decision"].get("nearest_human_ft"),
                    "nearest_human_sector": state["decision"].get("nearest_human_sector"),
                    "nearest_human_motion": state["decision"].get("nearest_human_motion"),
                    "nearest_human_raw_motion": state["decision"].get("nearest_human_raw_motion"),
                    "nearest_human_speed_mps": state["decision"].get("nearest_human_speed_mps"),
                    "nearest_human_raw_speed_mps": state["decision"].get("nearest_human_raw_speed_mps"),
                    "nearest_human_points": state["decision"].get("nearest_human_points"),
                    "n_humans": state["decision"].get("n_humans"),
                    "unet_disagree_center": state["decision"].get("unet_disagree_center"),
                },
                "scene": {
                    "confidence": state["scene"].get("confidence"),
                    "open_directions_confidence": state["scene"].get("open_directions_confidence"),
                    "class_counts": state["scene"].get("class_counts"),
                    "ego_motion": state["scene"].get("ego_motion"),
                    "humans": state["scene"].get("humans"),
                    "structure": state["scene"].get("structure"),
                    "floor": state["scene"].get("floor"),
                    "evidence_channels": state["scene"].get("evidence_channels"),
                    "open_directions": state["scene"].get("open_directions"),
                    "blocking": state["scene"].get("blocking"),
                    "map_anomalies": state["scene"].get("map_anomalies"),
                    "unet_freespace": state["scene"].get("unet_freespace"),
                    "reused_semantics": state.get("reused_semantics", False),
                    "doppler_residuals": state["scene"].get("doppler_residuals"),
                    "directness": state["scene"].get("evidence_channels", {}).get("directness", {}),
                },
                "persist": state.get("persist", {}),
                "current_decision_text": (
                    f"{state['decision'].get('primary_action')} / "
                    f"{state['decision'].get('reason')} / "
                    f"block={state['decision'].get('blocking_confidence')}"
                ),
                "guidance_text": state.get("guidance_text", ""),
                "ok": bool(ok),
            })

        session_dir = csv_path.parent
        npz_data = _load_live_npz_for_session(csv_path)

        radar_ts     = _load_frame_timestamps(session_dir / f"{session_name}_radar_timestamps.csv")
        color_ts     = _load_frame_timestamps(session_dir / f"{session_name}_color_timestamps.csv")
        radar_indices = radar_ts[0] if radar_ts is not None else None
        radar_times  = radar_ts[1] if radar_ts is not None else None
        color_times  = color_ts[1] if color_ts is not None else None
        predicted_frame_data = _load_model_frames(
            csv_path, model_pt=args.model_pt, mb=mb)
        frame_data = _align_frames_to_radar_timestamps(
            predicted_frame_data, radar_indices)
        source_frame_count = len(predicted_frame_data)
        source_frame_kind = "model-predicted"
        frame_times = _frame_times_for_replay(frame_data, radar_indices, radar_times)

        if radar_indices is not None and len(frame_data) != source_frame_count:
            missing_count = len(frame_data) - source_frame_count
            print(
                f"[{sess_idx}/{len(session_list)}] {session_name}: "
                f"{len(frame_data)} radar frames "
                f"({source_frame_count} {source_frame_kind}, {missing_count} empty)",
                flush=True,
            )
        else:
            print(
                f"[{sess_idx}/{len(session_list)}] {session_name}: "
                f"{len(frame_data)} {source_frame_kind} frames"
            )

        color_cap    = None
        color_fw     = 0
        color_fh     = 0
        color_fps    = float(args.fps)

        if not args.no_color_video:
            vid = _find_color_video(session_dir)
            if vid is not None:
                color_cap = cv2.VideoCapture(str(vid))
                if not color_cap.isOpened():
                    print(f"  [color] Could not open {vid.name}")
                    color_cap = None
                else:
                    color_fw  = int(color_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    color_fh  = int(color_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    color_fps = float(color_cap.get(cv2.CAP_PROP_FPS) or args.fps or 10.0)
                    print(f"  [color] {vid.name} @ {color_fps:.2f}fps")

        def _process_radar_state(radar_idx: int, *, render_panel: bool) -> dict:
            nonlocal last_guidance, last_semantic_state
            frame_num, raw_pts = frame_data[radar_idx]
            radar_time_ms = int(frame_times[radar_idx]) if frame_times is not None and radar_idx < len(frame_times) else None
            frame_time_s = (
                float(radar_time_ms) / 1000.0
                if radar_time_ms is not None else radar_idx * FRAME_INTERVAL_S
            )
            corrected, scene, decision, persist = _process_frame_3branch(
                raw_pts, int(frame_num), mb,
                npz_data=npz_data,
                branch3_runner=branch3_runner,
                frame_time_s=frame_time_s,
                ego_trajectory=ego_trajectory,
                ghost_tracker=ghost_tracker,
                session_dir=session_dir,
            )
            reused_semantics = False
            if (
                args.empty_frame_mode == "hold_last"
                and not corrected
                and last_semantic_state is not None
            ):
                fallback_scene = copy.deepcopy(last_semantic_state["scene"])
                fallback_scene["unet_freespace"] = scene.get("unet_freespace", {})
                fallback_scene["map_anomalies"] = scene.get("map_anomalies", {})
                fallback_scene["confidence"] = "low"
                fallback_scene["open_directions_confidence"] = "unknown"
                scene = fallback_scene
                decision = compute_nav_decision(scene)
                persist = last_semantic_state["persist"]
                corrected = last_semantic_state["corrected"]
                reused_semantics = True
            elif corrected:
                last_semantic_state = {
                    "scene": copy.deepcopy(scene),
                    "persist": persist,
                    "corrected": corrected,
                }

            _update_stats(stats, counts, scene, decision, persist)
            guidance_fired = (radar_idx % guidance_every) == 0
            if guidance_fired:
                last_guidance = build_scene_description(
                    {"scene": scene, "decision": decision})
                _print_frame_summary(frame_num, decision, persist, scene,
                                     args.verbose, last_guidance)
            ghost_positions = ghost_tracker.get_active() if ghost_tracker else []
            state = {
                "frame_num":     int(frame_num),
                "frame_idx":     int(radar_idx),
                "sim_time_s":    radar_idx * FRAME_INTERVAL_S,
                "radar_time_ms":  radar_time_ms,
                "corrected":     corrected,
                "n_points":      int(len(corrected)),
                "n_raw_points":  int(len(raw_pts)),
                "point_counts":  {
                    cls: int(sum(1 for p in corrected if p.get("pred_class") == cls))
                    for cls in ("structure", "floor", "human")
                },
                "scene":         scene,
                "ego_motion":    scene.get("ego_motion", {}),
                "decision":      decision,
                "persist":       persist,
                "guidance_text": last_guidance,
                "guidance_fired": guidance_fired,
                "reused_semantics": reused_semantics,
                "ghost_positions": ghost_positions,
            }
            pred_exporter.append(session_name, state)
            if render_panel:
                map_occ   = mb.query_occupancy(0.0, 1.5, radius_m=0.5)
                panel_w   = max(420, min(int((color_fw or args.width)  * 0.44), 620))
                panel_h   = max(280, min(int((color_fh or args.height) * 0.50), 390))
                state["radar_panel"] = _render_radar_panel(
                    session_label=session_name, frame_num=int(frame_num),
                    frame_idx=radar_idx,
                    n_frames=len(frame_data), sim_time_s=radar_idx * FRAME_INTERVAL_S,
                    points=corrected, scene=scene, decision=decision, map_occ=map_occ,
                    persist=persist, guidance_text=last_guidance,
                    guidance_fired=guidance_fired,
                    panel_width=panel_w, panel_height=panel_h)
            return state

        def _render_radar_frame(
            radar_idx: int,
            *,
            output_frame_idx: int | None = None,
            capture_debug: bool = True,
        ) -> np.ndarray:
            state = _process_radar_state(radar_idx, render_panel=False)
            map_occ = mb.query_occupancy(0.0, 1.5, radius_m=0.5)
            img = _render_frame(
                session_label=session_name, frame_num=state["frame_num"],
                frame_idx=state["frame_idx"],
                n_frames=len(frame_data), sim_time_s=state["sim_time_s"],
                points=state["corrected"], scene=state["scene"], decision=state["decision"],
                map_occ=map_occ, guidance_text=state["guidance_text"],
                guidance_fired=state["guidance_fired"],
                frame_width=args.width, frame_height=args.height,
                ghost_positions=state.get("ghost_positions"))
            stamped = _stamp_validation_overlay(
                img, state["scene"], state["decision"], state["persist"])
            if output_frame_idx is None:
                output_frame_idx = state["frame_idx"]
            if capture_debug:
                _capture_debug_snapshot(
                    stamped, state,
                    output_frame_idx=int(output_frame_idx),
                    video_time_ms=state.get("radar_time_ms"),
                    source="radar")
            return stamped

        def _advance_radar_state(radar_idx: int) -> dict:
            return _process_radar_state(radar_idx, render_panel=True)

        use_color_sync = (
            not args.no_video
            and color_cap is not None
            and frame_times is not None
            and color_times is not None
            and len(frame_times) > 0
            and len(color_times) > 0
            and len(frame_data) > 0
        )

        if use_color_sync and args.ui_style == "integrated" and color_fw > 0 and color_fh > 0:
            _run_integrated_writer(
                frame_data, color_times, frame_times, color_cap,
                color_fw, color_fh, color_fps, args,
                session_name, _advance_radar_state,
                debug_capture_fn=_capture_debug_snapshot)

        elif use_color_sync and args.ui_style == "stacked" and stacked_color_panel_height > 0:
            _run_stacked_writer(
                frame_data, color_times, frame_times, color_cap,
                stacked_color_panel_height, color_fps, args,
                session_name, _render_radar_frame,
                debug_capture_fn=_capture_debug_snapshot)

        else:
            for fi, _ in enumerate(frame_data):
                if args.no_video:
                    # When skipping video, still render stride-th frames for debug snapshots
                    if debug_enabled and (fi == 0 or fi % debug_stride == 0):
                        _render_radar_frame(fi, output_frame_idx=fi)
                    else:
                        _process_radar_state(fi, render_panel=False)
                else:
                    stamped = _render_radar_frame(fi, output_frame_idx=fi)
                    video_frames.append(stamped)

        if color_cap is not None:
            color_cap.release()

        if not args.no_video and not use_color_sync:
            session_output.parent.mkdir(parents=True, exist_ok=True)
            _write_mp4(video_frames, session_output, fps=args.fps)

        if debug_enabled:
            debug_session_dir.mkdir(parents=True, exist_ok=True)
            debug_manifest = {
                "session": session_name,
                "session_output": str(session_output),
                "source_csv": str(csv_path),
                "enabled": True,
                "stride": debug_stride,
                "map_mode": map_mode,
                "ego_trajectory": ego_trajectory.snapshot(),
                "captured": len(debug_samples),
                "samples": debug_samples,
            }
            manifest_path = debug_session_dir / "debug_frames.json"
            # Convert numpy types to native Python for JSON serialization;
            # also replace non-finite floats (inf, -inf, nan) with None (→ JSON null)
            # since Python's json.dumps emits literal Infinity/NaN which is not valid JSON.
            def _convert(obj):
                if isinstance(obj, dict):
                    return {k: _convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_convert(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    v = float(obj)
                    return None if not math.isfinite(v) else v
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, float):
                    return None if not math.isfinite(obj) else obj
                return obj
            manifest_path.write_text(json.dumps(_convert(debug_manifest), indent=2), encoding="utf-8")
            stats.setdefault("debug_frame_snapshots", {})[session_name] = {
                "dir": str(debug_session_dir),
                "manifest": str(manifest_path),
                "captured": len(debug_samples),
            }

        stats.setdefault("ego_trajectory_by_session", {})[session_name] = ego_trajectory.snapshot()

        args.output = base_output

    pred_export_result = pred_exporter.flush()
    stats.update(pred_export_result)

    print("summary:")
    print(json.dumps(stats, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")



# ---------------------------------------------------------------------------
# Stats / print helpers (extracted to reduce duplication)
# ---------------------------------------------------------------------------

def _update_stats(stats, counts, scene, decision, persist):
    stats["frames"] += 1
    action = str(decision.get("primary_action", "unknown"))
    counts[action] = counts.get(action, 0) + 1
    reason = str(decision.get("reason", "unknown"))
    reason_counts = stats.setdefault("reason_counts", {})
    reason_counts[reason] = reason_counts.get(reason, 0) + 1
    block_conf = str(decision.get(
        "blocking_confidence",
        scene.get("blocking", {}).get("center_confidence", "unknown"),
    ))
    block_counts = stats.setdefault("blocking_confidence_counts", {})
    block_counts[block_conf] = block_counts.get(block_conf, 0) + 1
    od = scene.get("open_directions", {})
    if (
        scene.get("open_directions_confidence") == "unknown"
        or decision.get("reason") == "low_confidence"
    ):
        blocked = "unknown"
    else:
        blocked = "".join(
            name
            for name, clear in (
                ("L", bool(od.get("left_clear", True))),
                ("C", bool(od.get("center_clear", True))),
                ("R", bool(od.get("right_clear", True))),
            )
            if not clear
        ) or "all_clear"
    direction_counts = stats.setdefault("open_direction_counts", {})
    direction_counts[blocked] = direction_counts.get(blocked, 0) + 1
    fs   = scene.get("unet_freespace", {})
    anom = scene.get("map_anomalies",  {})
    human_motion_counts = stats.setdefault("human_motion_counts", {
        "detections": 0,
        "raw_approaching": 0,
        "ego_relative_approaching": 0,
        "raw_approach_suppressed_by_ego": 0,
    })
    for det in scene.get("humans", {}).get("detections", []) or []:
        raw_motion = det.get("raw_motion", det.get("motion"))
        rel_motion = det.get("motion")
        human_motion_counts["detections"] += 1
        if raw_motion == "approaching":
            human_motion_counts["raw_approaching"] += 1
        if rel_motion == "approaching":
            human_motion_counts["ego_relative_approaching"] += 1
        if raw_motion == "approaching" and rel_motion != "approaching":
            human_motion_counts["raw_approach_suppressed_by_ego"] += 1
    if scene.get("ego_motion", {}).get("available"):
        stats["ego_motion_available_frames"] = stats.get("ego_motion_available_frames", 0) + 1
    if fs.get("available"):
        stats["branch3_available_frames"] += 1
    if decision.get("unet_disagree_center"):
        stats["unet_disagree_frames"] += 1
    if anom.get("novel_occupancy"):
        stats["novel_occupancy_frames"] += 1
    if anom.get("novel_free_space"):
        stats["novel_free_space_frames"] += 1
    if persist["nonzero"] > 0:
        stats["persist_nonzero_frames"] += 1


def _print_frame_summary(frame_num, decision, persist, scene, verbose, guidance_text):
    fs = scene.get("unet_freespace", {})
    print(
        f"  f{int(frame_num):03d} "
        f"{decision.get('primary_action','?'):<10} "
        f"{decision.get('reason',''):<34} "
        f"persist_max={persist['max']:.2f} "
        f"b3={'on' if fs.get('available') else 'off'}",
        flush=True)
    if verbose:
        for line in guidance_text.splitlines():
            print(f"    {line}")


def _run_integrated_writer(frame_data, color_times, frame_times, color_cap,
                            color_fw, color_fh, color_fps, args,
                            session_name, advance_fn,
                            debug_capture_fn=None):
    """Write integrated-style MP4 (color + radar panel overlay)."""
    radar_idx     = 0
    current_state = advance_fn(0)
    radar_idx     = 1
    output_writer = None
    last_composite = None
    last_color_frame = None
    last_video_time_ms = None
    output_count   = 0

    def _compose_state(color_frame, state):
        return _compose_integrated_frame(
            color_frame, state["radar_panel"],
            session_label=session_name,
            frame_num=int(state["frame_num"]),
            frame_idx=int(state["frame_idx"]),
            n_frames=len(frame_data),
            sim_time_s=float(state["sim_time_s"]),
            decision=state["decision"],
            scene=state["scene"],
            persist=state["persist"],
            guidance_text=str(state["guidance_text"]),
            guidance_fired=bool(state["guidance_fired"]),
        )

    for color_time_ms in color_times:
        while (radar_idx < len(frame_data)
               and radar_idx < len(frame_times)
               and int(frame_times[radar_idx]) <= int(color_time_ms)):
            current_state = advance_fn(radar_idx)
            radar_idx += 1
        cf = _read_next_color_frame(color_cap, color_fw, color_fh)
        if cf is None:
            break
        composite = _compose_state(cf, current_state)
        if output_writer is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_writer = cv2.VideoWriter(
                str(args.output), fourcc,
                color_fps if color_fps > 0 else args.fps,
                (color_fw, color_fh))
            if not output_writer.isOpened():
                print(f"  [video] Could not open VideoWriter for {args.output}")
                return
        output_writer.write(composite)
        if debug_capture_fn is not None:
            debug_capture_fn(
                composite, current_state,
                output_frame_idx=output_count,
                video_time_ms=int(color_time_ms),
                source="integrated")
        last_color_frame = cf
        last_composite = composite
        last_video_time_ms = int(color_time_ms)
        output_count += 1

    fps_out = color_fps if color_fps > 0 else args.fps
    frame_interval_ms = 1000.0 / max(fps_out, 1.0)
    while (
        output_writer is not None
        and last_color_frame is not None
        and radar_idx < len(frame_data)
    ):
        current_time_ms = int(frame_times[radar_idx]) if frame_times is not None else None
        current_state = advance_fn(radar_idx)
        radar_idx += 1
        composite = _compose_state(last_color_frame, current_state)
        if last_video_time_ms is not None and current_time_ms is not None:
            repeat = max(1, int(round((current_time_ms - last_video_time_ms) / frame_interval_ms)))
        else:
            repeat = 1
        for _ in range(repeat):
            output_writer.write(composite)
            if debug_capture_fn is not None:
                debug_capture_fn(
                    composite, current_state,
                    output_frame_idx=output_count,
                    video_time_ms=current_time_ms,
                    source="integrated")
            output_count += 1
        last_composite = composite
        if current_time_ms is not None:
            last_video_time_ms = current_time_ms

    if output_writer is not None and last_composite is not None:
        pad_frames  = max(1, int(fps_out * 2))
        for _ in range(pad_frames):
            output_writer.write(last_composite)
        output_writer.release()
        total = output_count + pad_frames
        print(f"  [video] {total} frames -> {args.output}  "
              f"({color_fw}x{color_fh} @ {fps_out:.2f}fps, "
              f"~{total/max(fps_out,1):.0f}s)", flush=True)


def _run_stacked_writer(frame_data, color_times, frame_times, color_cap,
                         stacked_height, color_fps, args,
                         session_name, render_fn,
                         debug_capture_fn=None):
    """Write stacked MP4 (color above, radar below)."""
    radar_idx     = 0
    current_sim   = render_fn(0, capture_debug=False)
    radar_idx     = 1
    output_writer = None
    last_composite = None
    last_color_frame = None
    last_video_time_ms = None
    output_count   = 0
    out_h          = args.height + stacked_height
    out_w          = args.width

    for color_time_ms in color_times:
        while (radar_idx < len(frame_data)
               and radar_idx < len(frame_times)
               and int(frame_times[radar_idx]) <= int(color_time_ms)):
            current_sim = render_fn(radar_idx, capture_debug=False)
            radar_idx += 1
        cf = _read_next_color_frame(color_cap, args.width, stacked_height)
        if cf is None:
            break
        composite = np.vstack([cf, current_sim])
        if output_writer is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_writer = cv2.VideoWriter(
                str(args.output), fourcc,
                color_fps if color_fps > 0 else args.fps,
                (out_w, out_h))
            if not output_writer.isOpened():
                print(f"  [video] Could not open VideoWriter for {args.output}")
                return
        output_writer.write(composite)
        last_color_frame = cf
        last_composite = composite
        last_video_time_ms = int(color_time_ms)
        output_count += 1

    fps_out = color_fps if color_fps > 0 else args.fps
    frame_interval_ms = 1000.0 / max(fps_out, 1.0)
    while (
        output_writer is not None
        and last_color_frame is not None
        and radar_idx < len(frame_data)
    ):
        current_time_ms = int(frame_times[radar_idx]) if frame_times is not None else None
        current_sim = render_fn(radar_idx, capture_debug=False)
        radar_idx += 1
        composite = np.vstack([last_color_frame, current_sim])
        if last_video_time_ms is not None and current_time_ms is not None:
            repeat = max(1, int(round((current_time_ms - last_video_time_ms) / frame_interval_ms)))
        else:
            repeat = 1
        for _ in range(repeat):
            output_writer.write(composite)
            output_count += 1
        last_composite = composite
        if current_time_ms is not None:
            last_video_time_ms = current_time_ms

    if output_writer is not None and last_composite is not None:
        pad_frames = max(1, int(fps_out * 2))
        for _ in range(pad_frames):
            output_writer.write(last_composite)
        output_writer.release()
        total = output_count + pad_frames
        print(f"  [video] {total} frames -> {args.output}  "
              f"({out_w}x{out_h} @ {fps_out:.2f}fps, "
              f"~{total/max(fps_out,1):.0f}s)", flush=True)


if __name__ == "__main__":
    main()
