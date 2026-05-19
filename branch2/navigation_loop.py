"""
navigation_loop.py
==================
Continuous navigation loop for Jetson deployment.

Pipeline per session:
    radar_only_recorder  →  adc_to_pointcloud_v6
    →  Z filter + range gate + checkpoint-driven feature computation
    →  checkpoint-selected inference (DGCNN or KPConv)
    →  scene_aggregator  →  [async LLM thread]  →  Piper TTS  →  delete

Model: kpconv_combined_apr28_apr22_v1
    - RDPatchTemporalSegmenter wrapping KPConvTemporalSegmenter
    - class order read from checkpoint; current Branch 1 checkpoints may use
      either structure/floor/human or structure/floor/human_candidate/ghost_return
    - 29 features: directness_soft_structural (13 base + 7 soft structural + 5 directness
      + 4 RD scalar; 9 dims zero-filled — see runtime_feature_gap_resolution_plan.md)
    - 64 points/frame, window_size=3, RD/RA patches enabled
    - val_bal_acc: 73.31% at epoch 26

Usage
-----
    python3 navigation_loop.py
    (set NAV_PERCEPTION_MODE=pointcloud_rd_patch for RD/RA-patch checkpoint)
Ctrl+C stops cleanly.
"""

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from scene_pipeline import get_logger, get_log_path
log = get_logger("Loop")

_PIPELINE_DIR = Path(__file__).resolve().parent
_PIPELINE_3BRANCH_DIR = _PIPELINE_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_ROOT     = "/home/ryan/nav_sessions/"
MMWAVE_CFG_PATH = "/home/ryan/xwr/profile_objdet.cfg"
ML_MODEL_DIR    = "/home/ryan/xwr/ml_model"
LLM_DIR         = "/home/ryan/xwr/llm"

# ADC settings
ADC_PFA     = 1e-2
ADC_MIN_SNR = 3.0

# Perception mode. Hybrid RD/RA remains a point-cloud branch; dense tensor
# modes are intentionally not enabled in this implementation.
NAV_PERCEPTION_MODE = os.environ.get("NAV_PERCEPTION_MODE", "pointcloud").strip().lower()

# Model checkpoint. Prefer the 3branch KPConv checkpoint, but fall back to the
# last known working point-cloud checkpoint when the 3branch artifact has not
# been synced to the Jetson yet.
BRANCH3_MODEL_PT = str(_PIPELINE_DIR / "models" / "3branch_best_model.pt")
LEGACY_BRANCH3_MODEL_PT = os.path.join(ML_MODEL_DIR, "models", "3branch_best_model.pt")
LEGACY_MODEL_PT  = os.path.join(ML_MODEL_DIR, "models", "84.pt")
BEST_MODEL_PT    = os.path.join(ML_MODEL_DIR, "models", "best_model.pt")
_EXPLICIT_POINTCLOUD_MODEL_PT = os.environ.get("NAV_POINTCLOUD_MODEL_PT")
_EXPLICIT_RD_PATCH_MODEL_PT = os.environ.get("NAV_POINTCLOUD_RD_PATCH_MODEL_PT")
BASE_MODEL_PT = _EXPLICIT_POINTCLOUD_MODEL_PT or BRANCH3_MODEL_PT
MODEL_PT = (
    _EXPLICIT_RD_PATCH_MODEL_PT or BASE_MODEL_PT
    if NAV_PERCEPTION_MODE == "pointcloud_rd_patch"
    else BASE_MODEL_PT
)

# Hybrid RD/RA patch sidecar settings
HYBRID_RD_SUBDIR = os.environ.get("NAV_POINTCLOUD_RD_PATCH_SUBDIR", "hybrid_rd")
HYBRID_RD_DOPPLER_BINS = int(os.environ.get("NAV_POINTCLOUD_RD_PATCH_DOPPLER_BINS", "17"))
HYBRID_RD_RANGE_BINS = int(os.environ.get("NAV_POINTCLOUD_RD_PATCH_RANGE_BINS", "7"))
_HYBRID_RD_FRAME_OFFSET_ENV = os.environ.get("NAV_POINTCLOUD_RD_FRAME_OFFSET")
HYBRID_RD_FRAME_OFFSET = int(_HYBRID_RD_FRAME_OFFSET_ENV or "1")
HYBRID_RD_CLUTTER_MODE = os.environ.get("NAV_POINTCLOUD_RD_CLUTTER_MODE", "off")
HYBRID_RD_LINEAR_POWER = os.environ.get("NAV_POINTCLOUD_RD_LINEAR_POWER", "0").strip().lower() in {"1", "true", "yes"}
HYBRID_RD_OVERWRITE_SIDECARS = os.environ.get("NAV_POINTCLOUD_RD_OVERWRITE_SIDECARS", "0").strip().lower() in {"1", "true", "yes"}

# Branch 3 — optional BEVUNet free-space inference. It stays disabled at
# runtime when the checkpoint is missing or sidecars are incompatible.
UNET_ENABLED = os.environ.get("NAV_UNET_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
UNET_PT = os.environ.get("NAV_UNET_PT", str(_PIPELINE_DIR / "models" / "unet_best_model.pt"))
UNET_SECTOR_AXIS = os.environ.get("NAV_UNET_SECTOR_AXIS", "columns")

# LLM — WSL Ollama via reverse SSH tunnel (WSL:11434 → Jetson:11435)
LLM_MODEL    = "llama3.2:3b"
LLM_MIN_WAIT = 8.0
OLLAMA_HOST  = "http://127.0.0.1:11435"

# Piper TTS
PIPER_BIN   = Path("/home/ryan/miniconda3/envs/mmwave/bin/piper")
PIPER_VOICE = Path("/home/ryan/piper_voices/en_US-lessac-medium.onnx")

# Scene window — single session to avoid stale data
SCENE_WINDOW = 3

# Queue warning threshold
MAX_QUEUE_WARN = 3

# Physical Z filter — remove below-floor multipath
MIN_VALID_Z = -0.851  # -(radar_height 0.813m + 1.5in tolerance)

# Range gate — navigationally relevant window
MIN_RANGE_M = 0.3   # below = noise or wheelchair body
MAX_RANGE_M = 5.0   # beyond = too far to act on now

# Soft structural RANSAC iterations (fewer = faster at inference)
SOFT_RANSAC_ITERS = 150

# Fast guidance fallback cadence
FAST_GUIDANCE_MIN_WAIT = 3.0

# Branch 2 — MapBuilder
MAP_ENABLED        = True    # set False to disable MapBuilder entirely
MAP_CELL_M         = 0.20    # metres per occupancy cell
MAP_RANGE_M        = 10.0    # ±10 m grid extent
MAP_DECAY          = 0.97    # per-session grid decay
MAP_SCORE_RADIUS_M = 0.30    # radius for per-point occupancy score lookup
MAP_UPDATE_MODE    = os.environ.get("NAV_MAP_UPDATE_MODE", "uniform")

# =============================================================================
# Module path setup
# =============================================================================

sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PIPELINE_DIR / "perception"))
sys.path.insert(0, str(_PIPELINE_DIR / "models"))
sys.path.insert(0, ML_MODEL_DIR)
sys.path.insert(0, LLM_DIR)

# =============================================================================
# Model loader
# =============================================================================

_model        = None
_feat_means   = None
_feat_stds    = None
_bucket_order = None
_feature_cols = None
_window_size  = 3
_n_points     = 128
_uses_rd_patch = False
_uses_ra_patch = False
_has_acc_head = False
_model_pt_resolved = False
_ra_patch_source = "true_ra"
_ra_patch_az_mode = "tx0_only"
_ra_patch_az_fft_size = 64
_ra_patch_log_scale = True
_ra_patch_normalize_by_frame_max = True
_ra_patch_row_edge_mode = "zero_pad"

# Branch 2 — MapBuilder singleton (initialized in main())
_map_builder = None

# Branch 3 — BEVUNet singleton (loaded lazily)
_unet_model = None
_unet_loaded = False
_unet_mod = None
_unet_checkpoint = None


def get_map_builder():
    """Return the running MapBuilder instance, or None if disabled."""
    return _map_builder


def _resolve_model_pt() -> str:
    """Resolve MODEL_PT, preferring 3branch but preserving the working fallback."""
    global MODEL_PT, _model_pt_resolved
    if _model_pt_resolved:
        return MODEL_PT

    explicit = (
        _EXPLICIT_RD_PATCH_MODEL_PT
        if NAV_PERCEPTION_MODE == "pointcloud_rd_patch"
        else _EXPLICIT_POINTCLOUD_MODEL_PT
    )
    if explicit:
        _model_pt_resolved = True
        return MODEL_PT

    for candidate in (MODEL_PT, BRANCH3_MODEL_PT, LEGACY_BRANCH3_MODEL_PT, LEGACY_MODEL_PT, BEST_MODEL_PT):
        if candidate and os.path.exists(candidate):
            if candidate != MODEL_PT:
                log.warning("Preferred 3branch checkpoint missing; using fallback model.",
                            preferred=MODEL_PT,
                            fallback=candidate)
            MODEL_PT = candidate
            _model_pt_resolved = True
            return MODEL_PT

    _model_pt_resolved = True
    return MODEL_PT


def _unet_unavailable(reason: str = "") -> dict:
    out = {"available": False, "center": 0.0, "left": 0.0, "right": 0.0}
    if reason:
        out["reason"] = reason
    return out


def get_unet_model():
    """Return the optional Branch 3 BEVUNet model, or None when unavailable."""
    global _unet_model, _unet_loaded, _unet_mod, _unet_checkpoint
    if _unet_loaded:
        return _unet_model

    _unet_loaded = True
    if not UNET_ENABLED:
        log.info("Branch 3: UNet disabled by NAV_UNET_ENABLED.")
        return None
    if not UNET_PT or not os.path.exists(UNET_PT):
        log.info("Branch 3: UNet checkpoint not found; free-space branch disabled.",
                 path=UNET_PT)
        return None

    try:
        import importlib
        import torch

        _unet_mod = importlib.import_module("branch3_unet")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _unet_checkpoint, model = _unet_mod.load_unet_checkpoint(UNET_PT, map_location=str(device))
        model.to(device)
        model.eval()
        _unet_model = model
        log.info("Branch 3: BEVUNet loaded.",
                 checkpoint=UNET_PT,
                 params=model.n_params(),
                 in_channels=getattr(model, "in_channels", None),
                 device=str(device))
    except Exception as e:
        log.warning(f"Branch 3: BEVUNet load failed: {e}; free-space branch disabled.")
        _unet_model = None

    return _unet_model


def get_model():
    global _model, _feat_means, _feat_stds, _bucket_order, _feature_cols
    global _window_size, _n_points
    global _uses_rd_patch, _uses_ra_patch, _has_acc_head
    global _ra_patch_source, _ra_patch_az_mode, _ra_patch_az_fft_size
    global _ra_patch_log_scale, _ra_patch_normalize_by_frame_max, _ra_patch_row_edge_mode
    global HYBRID_RD_FRAME_OFFSET

    if _model is not None:
        return _model

    import torch
    import numpy as np

    model_pt = _resolve_model_pt()
    log.info("Loading model...", path=model_pt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Inference device: {device}")
    checkpoint = torch.load(model_pt, map_location=device, weights_only=False)
    checkpoint_state = checkpoint["model_state"]

    _feat_means   = np.array(checkpoint["feature_means"], dtype=np.float32)
    _feat_stds    = np.array(checkpoint["feature_stds"],  dtype=np.float32)
    _feat_stds    = np.where(_feat_stds < 1e-6, 1.0, _feat_stds)
    _bucket_order = checkpoint.get("bucket_order", ["structure", "floor", "human"])
    _feature_cols = checkpoint.get("feature_cols")
    _window_size  = int(checkpoint.get("window_size", 3))
    _n_points     = int(checkpoint.get("points_per_frame", 128))

    _uses_rd_patch = bool(checkpoint.get("uses_rd_patch", False))
    _uses_ra_patch = bool(checkpoint.get("uses_ra_patch", checkpoint.get("uses_ra_patches", False)))
    _ra_patch_source = str(checkpoint.get("ra_patch_source", "true_ra"))
    _ra_patch_az_mode = str(checkpoint.get("ra_patch_az_mode", "tx0_only"))
    _ra_patch_az_fft_size = int(checkpoint.get("ra_patch_az_fft_size", 64))
    _ra_patch_log_scale = bool(checkpoint.get("ra_patch_log_scale", True))
    _ra_patch_normalize_by_frame_max = bool(checkpoint.get("ra_patch_normalize_by_frame_max", True))
    _ra_patch_row_edge_mode = str(checkpoint.get("ra_patch_row_edge_mode", "zero_pad"))
    if _uses_rd_patch and _HYBRID_RD_FRAME_OFFSET_ENV is None:
        HYBRID_RD_FRAME_OFFSET = int(checkpoint.get("rd_patch_frame_number_offset", HYBRID_RD_FRAME_OFFSET))
    _has_acc_head = bool(
        checkpoint.get(
            "has_acc_head",
            any(str(k).startswith("acc_head.") for k in checkpoint_state.keys()),
        )
    )
    if NAV_PERCEPTION_MODE == "pointcloud_rd_patch" and not _uses_rd_patch:
        raise ValueError(
            "NAV_PERCEPTION_MODE=pointcloud_rd_patch requires a checkpoint with "
            "uses_rd_patch=true."
        )
    if NAV_PERCEPTION_MODE == "pointcloud" and _uses_rd_patch:
        raise ValueError(
            "Loaded checkpoint uses RD patches, but NAV_PERCEPTION_MODE=pointcloud. "
            "Set NAV_PERCEPTION_MODE=pointcloud_rd_patch or use a base point-cloud checkpoint."
        )
    if _uses_ra_patch and not _uses_rd_patch:
        raise ValueError("RA patch checkpoints require RD patches as well.")

    encoder_type = checkpoint.get("encoder_type", "dgcnn")
    rd_patch_embed_dim = int(checkpoint.get("rd_patch_embed_dim", 0)) if _uses_rd_patch else 0
    if _uses_rd_patch and rd_patch_embed_dim <= 0:
        raise ValueError("RD-patch checkpoint must include a positive rd_patch_embed_dim.")
    ra_patch_embed_dim = int(checkpoint.get("ra_patch_embed_dim", 0)) if _uses_ra_patch else 0
    if _uses_ra_patch and ra_patch_embed_dim <= 0:
        ra_patch_embed_dim = rd_patch_embed_dim
    model_n_features = len(_feature_cols) + rd_patch_embed_dim + ra_patch_embed_dim
    log.info(f"Checkpoint: encoder={encoder_type} features={len(_feature_cols)} "
             f"points={_n_points} window={_window_size} rd_patch={_uses_rd_patch} "
             f"ra_patch={_uses_ra_patch} ra_source={_ra_patch_source} "
             f"rd_frame_offset={HYBRID_RD_FRAME_OFFSET}")

    if encoder_type == "kpconv":
        from models.radar_3dgcnn_common_kpconv import KPConvTemporalSegmenter
        base_model = KPConvTemporalSegmenter(
            n_features      = model_n_features,
            n_classes       = len(_bucket_order),
            window_size     = _window_size,
            k               = int(checkpoint.get("k", 20)),
            emb_dims        = int(checkpoint.get("emb_dims", 256)),
            temporal_layers = int(checkpoint.get("temporal_layers", 2)),
            temporal_heads  = int(checkpoint.get("temporal_heads", 4)),
            dropout         = 0.0,
            frame_meta_dim  = int(checkpoint.get("frame_meta_dim", 5)),
            gate_hidden     = int(checkpoint.get("gate_hidden", 128)),
            radius_1        = float(checkpoint.get("radius_1", 0.35)),
            radius_2        = float(checkpoint.get("radius_2", 0.50)),
            radius_3        = float(checkpoint.get("radius_3", 0.70)),
            sigma_factor    = float(checkpoint.get("sigma_factor", 2.5)),
            n_kernel_points = int(checkpoint.get("n_kernel_points", 15)),
        )
    else:
        from models.radar_3dgcnn_common_doorwall import DopplerAwareTemporalDGCNNSegmenter
        base_model = DopplerAwareTemporalDGCNNSegmenter(
            n_features      = model_n_features,
            n_classes       = len(_bucket_order),
            k               = int(checkpoint.get("k", 28)),
            emb_dims        = int(checkpoint.get("emb_dims", 256)),
            dropout         = 0.0,
            window_size     = _window_size,
            temporal_layers = int(checkpoint.get("temporal_layers", 2)),
            temporal_heads  = int(checkpoint.get("temporal_heads", 4)),
            frame_meta_dim  = int(checkpoint.get("frame_meta_dim", 5)),
            gate_hidden     = int(checkpoint.get("gate_hidden", 128)),
        )

    if _uses_rd_patch:
        from models.hybrid_rd_model import RDPatchTemporalSegmenter
        model = RDPatchTemporalSegmenter(
            base_model,
            patch_channels=int(checkpoint.get("rd_patch_channels", 1)),
            patch_embed_dim=rd_patch_embed_dim,
            use_ra_patches=_uses_ra_patch,
            patch_hidden_channels=tuple(checkpoint.get("rd_patch_hidden_channels", (8, 16))),
            has_acc_head=_has_acc_head,
            acc_head_hidden_dim=int(checkpoint.get("acc_head_hidden_dim", 64)),
        )
    else:
        model = base_model

    state_dict = checkpoint_state
    if hasattr(model, "base_model") and not any(
        k.startswith("base_model.") for k in state_dict
    ):
        # The checkpoint was stripped of its base_model. prefix (expected for
        # the bare standalone 3branch loop).  Add the prefix back so the
        # wrapped RDPatchTemporalSegmenter can load it.
        _prefixed: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("rd_patch_encoder.") or k.startswith("ra_patch_encoder.") or k.startswith("acc_head."):
                _prefixed[k] = v
            else:
                _prefixed["base_model." + k] = v
        state_dict = _prefixed

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    _model = model

    log.info("Model loaded.",
             encoder=encoder_type,
             params=sum(p.numel() for p in model.parameters()),
             device=str(device),
             bucket_order=str(_bucket_order),
             has_acc_head=_has_acc_head,
             uses_ra_patch=_uses_ra_patch,
             n_features=len(_feature_cols),
             n_points=_n_points)
    return _model


# =============================================================================
# ADC processor
# =============================================================================

_processor = None


def get_processor():
    global _processor
    if _processor is None:
        from perception.adc_to_pointcloud_v6 import build_processor
        log.info("Building RadarProcessor...", cfg=MMWAVE_CFG_PATH)
        _processor = build_processor(
            cfg_path=MMWAVE_CFG_PATH,
            pfa=ADC_PFA,
            min_snr=ADC_MIN_SNR,
        )
        log.info("RadarProcessor ready.")
    return _processor


# =============================================================================
# Camera calibration
# =============================================================================

def _load_camera_calibration_for_session(session_dir: "str | None") -> "dict | None":
    """Load extrinsics + intrinsics for one session.

    Returns a dict with R, t, K, dist, img_shape, extrinsics_source,
    intrinsics_source — or None if the extrinsics asset is missing.
    """
    try:
        import json as _json
        from pathlib import Path as _Path
        from perception.calibration import load_calibration

        sd = _Path(session_dir) if session_dir else _Path(".")
        calib = load_calibration(sd)

        img_h, img_w = 360, 640
        meta_path = sd / "meta_data.json"
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            img_w = int(meta.get("width", img_w))
            img_h = int(meta.get("height", img_h))
        calib["img_shape"] = (img_h, img_w)
        return calib
    except Exception as exc:
        log.warning("Camera calibration unavailable; projection skipped.", reason=str(exc))
        return None


# =============================================================================
# Feature computation at inference time
#
# dca_v10 expects 20 features:
#   Base (13): x, y, z, range_m, azimuth_deg, elevation_deg, doppler, snr,
#              local_density, persist_score, z_norm_range,
#              frame_doppler_abs, frame_doppler_std
#   Soft structural (7): z_above_floor, corridor_margin, wall_anomaly,
#                        floor_ang, wall_ang, has_floor, has_wall
# =============================================================================

_xyz_session_buffer: deque = deque(maxlen=SCENE_WINDOW)


def reset_session_buffers() -> None:
    """Prevent stale geometry/scene state from leaking across session files."""
    _xyz_session_buffer.clear()
    _scene_window.clear()


def _normalize_vector(vec: "np.ndarray") -> "np.ndarray":
    import numpy as np

    vec = np.asarray(vec, dtype=np.float32)
    denom = float(np.max(np.abs(vec))) if vec.size else 0.0
    if denom < 1e-6:
        denom = 1.0
    return (vec / denom).astype(np.float32)


# =============================================================================
# Column rename map — ADC pipeline output → model feature names
# =============================================================================

_ADC_RENAME = {
    "doppler_mps": "doppler",
    "v": "doppler",
    "power_snr": "snr",
}


def _rename_adc_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rename ADC pipeline columns to the model feature names when needed."""
    renames = {}
    for src, dst in _ADC_RENAME.items():
        if src in df.columns and dst not in df.columns:
            renames[src] = dst
    if renames:
        df = df.rename(columns=renames)
    return df


def _compute_local_density_inference(df: "pd.DataFrame", frame_col: str,
                                     radius_m: float = 0.5) -> "pd.DataFrame":
    """Approximate training-time local_density per frame at inference."""
    import numpy as np

    densities = np.zeros(len(df), dtype=np.float32)

    for _, grp in df.groupby(frame_col, sort=False):
        idx = grp.index.to_numpy()
        pts = grp[["x", "y", "z"]].to_numpy(dtype=np.float32)
        n = len(pts)

        if n <= 1:
            densities[idx] = 0.0
            continue

        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        counts = (dist < radius_m).sum(-1) - 1
        counts = np.maximum(counts, 0)
        densities[idx] = (np.log1p(counts) / np.log1p(100)).astype(np.float32)

    df = df.copy()
    df["local_density"] = densities
    return df


def _compute_persist_score_inference(df: "pd.DataFrame", frame_col: str,
                                     voxel_size: float = 0.15) -> "pd.DataFrame":
    """Approximate persist_score from the current inference session."""
    import numpy as np

    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    voxels = np.floor(pts / voxel_size).astype(np.int32)

    frame_keys = df[frame_col].astype(int).to_numpy()
    unique_frames = len(np.unique(frame_keys))
    if unique_frames == 0:
        df = df.copy()
        df["persist_score"] = 0.0
        return df

    vox_frame_count: dict = {}
    for i in range(len(df)):
        key = (int(voxels[i, 0]), int(voxels[i, 1]), int(voxels[i, 2]))
        vox_frame_count.setdefault(key, set()).add(int(frame_keys[i]))

    persist = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        key = (int(voxels[i, 0]), int(voxels[i, 1]), int(voxels[i, 2]))
        persist[i] = len(vox_frame_count[key]) / unique_frames

    df = df.copy()
    df["persist_score"] = persist.astype(np.float32)
    return df


def normalize_inference_schema(df: "pd.DataFrame", required_feature_cols: list[str] | None = None) -> "pd.DataFrame":
    """Map ADC output columns onto the checkpoint feature schema without hardcoding a model."""
    import numpy as np
    import pandas as pd

    df = df.copy()
    required_feature_cols = required_feature_cols or []

    df = _rename_adc_columns(df)

    if "radar_frame_num" not in df.columns:
        if "frame_num" in df.columns:
            df["radar_frame_num"] = df["frame_num"]
        elif "frame" in df.columns:
            df["radar_frame_num"] = df["frame"]

    numeric_candidates = [
        "x", "y", "z", "range_m", "azimuth_deg", "elevation_deg",
        "doppler", "snr", "local_density", "persist_score",
        "frame_num", "radar_frame_num", "range_bin", "doppler_bin",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "range_m" not in df.columns and all(c in df.columns for c in ("x", "y", "z")):
        df["range_m"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

    if "azimuth_deg" not in df.columns and all(c in df.columns for c in ("x", "y")):
        df["azimuth_deg"] = np.degrees(np.arctan2(df["x"], np.maximum(df["y"], 1e-6))).astype(np.float32)

    if "elevation_deg" not in df.columns and all(c in df.columns for c in ("x", "y", "z")):
        r_xy = np.sqrt(df["x"]**2 + df["y"]**2)
        df["elevation_deg"] = np.degrees(np.arctan2(df["z"], r_xy + 1e-6)).astype(np.float32)

    if "doppler" not in df.columns:
        df["doppler"] = 0.0
    if "snr" not in df.columns:
        df["snr"] = 0.0

    for col in set(required_feature_cols) | {"doppler", "snr", "range_m", "azimuth_deg", "elevation_deg", "radar_frame_num"}:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if col == "radar_frame_num":
                df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)

    for col in ("range_bin", "doppler_bin"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(-1).astype(np.int64)

    return df


def add_computed_features(df: "pd.DataFrame", required_feature_cols: list[str] | None = None) -> "pd.DataFrame":
    """Compute all checkpoint-driven features missing from the ADC pipeline."""
    import numpy as np

    required_feature_cols = required_feature_cols or []
    df = _rename_adc_columns(df)

    if "range_m" not in df.columns:
        df["range_m"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2).astype(np.float32)

    if "z_norm_range" in required_feature_cols and "z_norm_range" not in df.columns:
        df["z_norm_range"] = (
            df["z"].to_numpy(dtype=np.float32) /
            (df["range_m"].to_numpy(dtype=np.float32) + 1e-6)
        )

    if "azimuth_deg" not in df.columns:
        df["azimuth_deg"] = np.degrees(
            np.arctan2(df["x"].to_numpy(dtype=np.float32), np.maximum(df["y"].to_numpy(dtype=np.float32), 1e-6))
        ).astype(np.float32)

    if "elevation_deg" not in df.columns:
        r_xy = np.sqrt(df["x"].to_numpy(dtype=np.float32)**2 + df["y"].to_numpy(dtype=np.float32)**2)
        df["elevation_deg"] = np.degrees(
            np.arctan2(df["z"].to_numpy(dtype=np.float32), r_xy + 1e-6)
        ).astype(np.float32)

    frame_col_candidates = ("radar_frame_num", "frame_num", "frame")
    frame_col = next((c for c in frame_col_candidates if c in df.columns), None)

    if "doppler" in df.columns and frame_col:
        grp = df.groupby(frame_col)["doppler"]
        if "frame_doppler_abs" in required_feature_cols and "frame_doppler_abs" not in df.columns:
            df["frame_doppler_abs"] = grp.transform(lambda x: x.abs().mean()).astype(np.float32)
        if "frame_doppler_std" in required_feature_cols and "frame_doppler_std" not in df.columns:
            df["frame_doppler_std"] = grp.transform("std").fillna(0).astype(np.float32)
    elif "doppler" in df.columns:
        if "frame_doppler_abs" in required_feature_cols and "frame_doppler_abs" not in df.columns:
            df["frame_doppler_abs"] = df["doppler"].abs().astype(np.float32)
        if "frame_doppler_std" in required_feature_cols and "frame_doppler_std" not in df.columns:
            df["frame_doppler_std"] = np.zeros(len(df), dtype=np.float32)
    else:
        df["doppler"] = np.float32(0.0)
        if "frame_doppler_abs" in required_feature_cols and "frame_doppler_abs" not in df.columns:
            df["frame_doppler_abs"] = np.float32(0.0)
        if "frame_doppler_std" in required_feature_cols and "frame_doppler_std" not in df.columns:
            df["frame_doppler_std"] = np.float32(0.0)

    if "local_density" in required_feature_cols and "local_density" not in df.columns:
        if frame_col:
            df = _compute_local_density_inference(df, frame_col, radius_m=0.5)
        else:
            df["local_density"] = np.float32(0.0)

    if "persist_score" in required_feature_cols and "persist_score" not in df.columns:
        used_map_scores = False
        mb = get_map_builder()
        if mb is not None and all(c in df.columns for c in ("x", "y", "z")):
            try:
                xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
                scores = mb.get_point_occupancy_scores(xyz, radius_m=MAP_SCORE_RADIUS_M)
                if len(scores) == len(df) and float(np.max(scores)) > 1e-6:
                    df["persist_score"] = scores.astype(np.float32)
                    used_map_scores = True
                    log.debug("Persist score: using MapBuilder occupancy scores.",
                              max=f"{float(np.max(scores)):.3f}")
            except Exception as e:
                log.debug(f"Persist score: MapBuilder lookup failed: {e}")

        if used_map_scores:
            pass
        elif frame_col:
            df = _compute_persist_score_inference(df, frame_col, voxel_size=0.15)
        else:
            df["persist_score"] = np.float32(0.0)

    soft_cols = [
        "z_above_floor", "corridor_margin", "wall_anomaly",
        "floor_ang", "wall_ang", "has_floor", "has_wall"
    ]
    missing_soft = [c for c in soft_cols if c in required_feature_cols and c not in df.columns]

    if missing_soft:
        session_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        _xyz_session_buffer.append(session_xyz)

        try:
            from concurrent.futures import ThreadPoolExecutor
            from perception.corridor_soft_structural import process_frame_group

            aggregated_xyz = (
                np.concatenate(list(_xyz_session_buffer), axis=0)
                if len(_xyz_session_buffer) > 0
                else session_xyz
            )

            t0 = time.monotonic()

            def _run_soft():
                return process_frame_group(
                    anchor_pts=session_xyz,
                    aggregated_pts=aggregated_xyz,
                    floor_dist_thresh=0.06,
                    wall_dist_thresh=0.08,
                    ransac_iters=SOFT_RANSAC_ITERS,
                    compute_normals=True,
                    normal_k=8,
                )

            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_run_soft)
                _, features = future.result(timeout=3.0)

            elapsed = time.monotonic() - t0
            has_floor = float(features.get("has_floor", [0])[0])
            log.debug(f"Soft structural: {elapsed:.2f}s has_floor={has_floor:.0f}")

            for col in missing_soft:
                if col in features and len(features[col]) == len(df):
                    df[col] = features[col]
                else:
                    df[col] = np.float32(0.0)

        except Exception as e:
            log.warning(f"Soft structural failed: {e} — using zeros")
            for col in missing_soft:
                df[col] = np.float32(0.0)

    return df


def build_frame_meta_from_buffer(frame_buffer: "deque") -> "np.ndarray":
    import numpy as np

    records = list(frame_buffer)
    if not records:
        return np.zeros((0, 5), dtype=np.float32)

    center = records[len(records) // 2]
    center_frame = int(center[2])
    dt_radar = np.array([int(rec[2]) - center_frame for rec in records], dtype=np.float32)
    dt_video = dt_radar.copy()
    mean_d = np.array([float(np.mean(rec[3])) if len(rec[3]) else 0.0 for rec in records], dtype=np.float32)
    abs_mean_d = np.array([float(np.mean(np.abs(rec[3]))) if len(rec[3]) else 0.0 for rec in records], dtype=np.float32)
    std_d = np.array([float(np.std(rec[3])) if len(rec[3]) else 0.0 for rec in records], dtype=np.float32)

    meta = np.stack([
        _normalize_vector(dt_video),
        _normalize_vector(dt_radar),
        _normalize_vector(mean_d),
        _normalize_vector(abs_mean_d),
        _normalize_vector(std_d),
    ], axis=1)
    return meta.astype(np.float32)


# =============================================================================
# Per-session inference
# =============================================================================

def _build_runtime_rd_patch_config(*, include_rd_cube: bool = False):
    from models.hybrid_rd_runtime import RuntimeRDPatchConfig

    return RuntimeRDPatchConfig(
        doppler_bins=HYBRID_RD_DOPPLER_BINS,
        range_bins=HYBRID_RD_RANGE_BINS,
        frame_number_offset=HYBRID_RD_FRAME_OFFSET,
        sidecar_subdir=HYBRID_RD_SUBDIR,
        clutter_mode=HYBRID_RD_CLUTTER_MODE,
        log_scale=not HYBRID_RD_LINEAR_POWER,
        include_rd_cube=include_rd_cube,
    )


def _build_runtime_ra_patch_config():
    from models.hybrid_rd_runtime import RuntimeRAPatchConfig

    return RuntimeRAPatchConfig(
        azimuth_bins=9,
        range_bins=7,
        frame_number_offset=HYBRID_RD_FRAME_OFFSET,
        source=_ra_patch_source,
        az_mode=_ra_patch_az_mode,
        az_fft_size=_ra_patch_az_fft_size,
        log_scale=_ra_patch_log_scale,
        normalize_by_frame_max=_ra_patch_normalize_by_frame_max,
        row_edge_mode=_ra_patch_row_edge_mode,
    )


RD_SCALAR_FEATURE_COLS = (
    "rd_entropy",
    "rd_doppler_spread",
    "rd_anisotropy",
    "rd_peak_ratio",
)
DIRECTNESS_FEATURE_COLS = (
    "ego_doppler_residual_mps",
    "ego_residual_abs_z",
    "ego_inlier_flag",
    "p_ego",
    "p_dir_doppler",
)


def _feature_set(feature_cols) -> set[str]:
    return set(str(c) for c in (feature_cols or []))


def _inject_pre_inference_directness(frame_df, feature_cols) -> None:
    needed = _feature_set(feature_cols) & set(DIRECTNESS_FEATURE_COLS)
    if not needed or len(frame_df) == 0:
        return
    try:
        from perception.directness_runtime import (
            DirectnessConfig,
            annotate_directness,
            estimate_ego_velocity_no_class,
        )

        cfg = DirectnessConfig()
        point_dicts = []
        for _, row in frame_df.iterrows():
            point_dicts.append({
                "x": float(row.get("x", 0.0)),
                "y": float(row.get("y", 0.0)),
                "z": float(row.get("z", 0.0)),
                "doppler": float(row.get("doppler", 0.0)),
                "snr": float(row.get("snr", 1.0)),
            })
        ego = estimate_ego_velocity_no_class(point_dicts, cfg)
        annotated = annotate_directness(point_dicts, ego, cfg)
        for col in DIRECTNESS_FEATURE_COLS:
            if col in needed:
                frame_df[col] = np.asarray(
                    [float(p.get(col, 0.0)) for p in annotated],
                    dtype=np.float32,
                )
    except Exception as exc:
        log.debug(f"Pre-inference directness feature injection failed: {exc}")


def run_inference_on_csv(csv_path: str, session_dir: str | None = None) -> list:
    import torch
    import numpy as np
    import pandas as pd
    from collections import Counter

    model = get_model()
    reset_session_buffers()

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return []

    df = normalize_inference_schema(df, _feature_cols or [])

    rd_patch_config = None
    rd_patch_extractor = None
    rd_patch_scalar_extractor = None
    rdra_patch_extractor = None
    ra_patch_config = None
    if _uses_rd_patch:
        if session_dir is None:
            raise ValueError("Hybrid RD/RA patch inference requires session_dir.")
        missing = [c for c in ("range_bin", "doppler_bin") if c not in df.columns]
        if missing:
            raise ValueError(
                "Hybrid RD/RA patch inference requires ADC CSV columns: "
                + ", ".join(missing)
                + ". Enable richer range_bin/doppler_bin export in adc_to_pointcloud_v6.py."
            )
        from models.hybrid_rd_runtime import (
            export_session_sidecars,
            extract_rdra_patches_and_scalars_for_frame_df,
            extract_patches_and_scalars_for_frame_df,
            extract_patches_for_frame_df,
        )

        ra_source = str(_ra_patch_source).strip().lower()
        need_rd_cube = bool(_uses_ra_patch and ra_source in {"true_ra", "rd_cube", "beamformed"})
        rd_patch_config = _build_runtime_rd_patch_config(include_rd_cube=need_rd_cube)
        export_session_sidecars(
            session_dir,
            cfg_path=MMWAVE_CFG_PATH,
            config=rd_patch_config,
            overwrite=HYBRID_RD_OVERWRITE_SIDECARS,
        )
        rd_patch_extractor = extract_patches_for_frame_df
        rd_patch_scalar_extractor = extract_patches_and_scalars_for_frame_df
        if _uses_ra_patch:
            ra_patch_config = _build_runtime_ra_patch_config()
            rdra_patch_extractor = extract_rdra_patches_and_scalars_for_frame_df

    # Z filter — remove below-floor multipath ghost returns
    df = df[df["z"] > MIN_VALID_Z].reset_index(drop=True)

    # Range gate — keep only navigationally relevant points
    df = df[
        (df["range_m"] >= MIN_RANGE_M) & (df["range_m"] <= MAX_RANGE_M)
    ].reset_index(drop=True)

    if len(df) == 0:
        return []

    frame_col = None
    for c in ("radar_frame_num", "frame_num", "frame"):
        if c in df.columns:
            frame_col = c
            break
    if frame_col is None:
        return []

    if frame_col != "radar_frame_num":
        df["radar_frame_num"] = pd.to_numeric(df[frame_col], errors="coerce").fillna(0).astype(np.int64)
        frame_col = "radar_frame_num"

    df = add_computed_features(df, _feature_cols or [])

    runtime_injected_cols = set(RD_SCALAR_FEATURE_COLS) | set(DIRECTNESS_FEATURE_COLS)
    for col in (_feature_cols or []):
        if col in runtime_injected_cols and col not in df.columns:
            df[col] = 0.0

    missing_cols = [col for col in (_feature_cols or []) if col not in df.columns]
    if missing_cols:
        log.warning("Missing checkpoint feature columns after normalization; zero-filling.", columns=",".join(missing_cols))
        for col in missing_cols:
            df[col] = 0.0

    # --- radar → camera projection ---
    _cam_calib = _load_camera_calibration_for_session(session_dir)
    if _cam_calib is not None:
        from perception.calibration import project_radar_to_image as _project_r2i
        pts_radar = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        uv, zc, cam_valid = _project_r2i(
            pts_radar,
            _cam_calib["R"],
            _cam_calib["t"],
            _cam_calib["K"],
            _cam_calib["dist"],
            _cam_calib["img_shape"],
        )
        df["cam_u"] = uv[:, 0].astype(np.float32)
        df["cam_v"] = uv[:, 1].astype(np.float32)
        df["cam_z"] = zc.astype(np.float64)
        df["cam_valid"] = cam_valid
        log.debug(
            "Camera projection applied.",
            extrinsics=_cam_calib["extrinsics_source"],
            intrinsics=_cam_calib["intrinsics_source"],
            n_valid=int(cam_valid.sum()),
            n_total=len(cam_valid),
        )

    results = []
    frames = sorted(df[frame_col].unique())
    frame_buffer = deque(maxlen=_window_size)
    all_preds = []
    device = next(model.parameters()).device
    center_idx = _window_size // 2
    last_emitted_frame_num: int | None = None
    session_tensor_cache: dict[Path, object] = {}

    with torch.no_grad():
        for frame_num in frames:
            frame_df = df[df[frame_col] == frame_num].copy()
            if len(frame_df) == 0:
                continue

            rd_patches = None
            ra_patches = None
            if _uses_rd_patch:
                if _uses_ra_patch:
                    rd_patches, ra_patches, rd_scalars = rdra_patch_extractor(
                        frame_df,
                        session_dir=session_dir,
                        frame_num=int(frame_num),
                        rd_config=rd_patch_config,
                        ra_config=ra_patch_config,
                        session_tensor_cache=session_tensor_cache,
                    )
                else:
                    if _feature_set(_feature_cols) & set(RD_SCALAR_FEATURE_COLS):
                        rd_patches, rd_scalars = rd_patch_scalar_extractor(
                            frame_df,
                            session_dir=session_dir,
                            frame_num=int(frame_num),
                            config=rd_patch_config,
                        )
                    else:
                        rd_patches = rd_patch_extractor(
                            frame_df,
                            session_dir=session_dir,
                            frame_num=int(frame_num),
                            config=rd_patch_config,
                        )
                        rd_scalars = None
                if rd_scalars is not None:
                    for col in RD_SCALAR_FEATURE_COLS:
                        if col in (_feature_cols or []):
                            frame_df[col] = np.asarray(
                                [float(s.get(col, 0.0)) for s in rd_scalars],
                                dtype=np.float32,
                            )

            _inject_pre_inference_directness(frame_df, _feature_cols)

            feats_raw = frame_df[_feature_cols].to_numpy(dtype=np.float32)
            feats_norm = (feats_raw - _feat_means) / (_feat_stds + 1e-6)
            feats_norm = np.nan_to_num(feats_norm, nan=0.0, posinf=0.0, neginf=0.0)
            xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            dopp_vals = frame_df["doppler"].to_numpy(dtype=np.float32) if "doppler" in frame_df.columns else np.zeros(len(frame_df), dtype=np.float32)

            frame_buffer.append((feats_norm, xyz, int(frame_num), dopp_vals, rd_patches, ra_patches, frame_df))
            while len(frame_buffer) < _window_size:
                frame_buffer.appendleft(frame_buffer[0])

            feat_list = [
                torch.from_numpy(rec[0].T.copy()).unsqueeze(0).to(device)
                for rec in frame_buffer
            ]
            xyz_list = [
                torch.from_numpy(rec[1].T.copy()).unsqueeze(0).to(device)
                for rec in frame_buffer
            ]
            frame_meta_np = build_frame_meta_from_buffer(frame_buffer)
            frame_meta = torch.from_numpy(frame_meta_np).unsqueeze(0).to(device)

            if _uses_rd_patch:
                rd_patch_list = [
                    torch.from_numpy(rec[4].copy()).unsqueeze(0).to(device)
                    for rec in frame_buffer
                ]
                if _uses_ra_patch:
                    ra_patch_list = [
                        torch.from_numpy(rec[5].copy()).unsqueeze(0).to(device)
                        for rec in frame_buffer
                    ]
                    model_out = model(feat_list, xyz_list, frame_meta, rd_patch_list, ra_patch_list)
                else:
                    model_out = model(feat_list, xyz_list, frame_meta, rd_patch_list)
            else:
                model_out = model(feat_list, xyz_list, frame_meta)
            if isinstance(model_out, dict):
                logits = model_out["sem_logits"]
                acc_logits = model_out.get("acc_logits")
            else:
                logits = model_out
                acc_logits = None
            preds = logits.squeeze(0).argmax(dim=0).detach().cpu().numpy()
            # Softmax class probabilities for confidence scoring and directness
            softmax = torch.softmax(logits.squeeze(0), dim=0).detach().cpu().numpy()  # (C, N)
            pred_conf = softmax.max(axis=0)  # confidence for argmax class
            def _class_prob(name: str, *, aliases: tuple[str, ...] = ()) -> np.ndarray:
                for candidate in (name, *aliases):
                    if candidate in _bucket_order:
                        return softmax[_bucket_order.index(candidate)]
                return np.zeros(softmax.shape[1], dtype=np.float32)

            prob_structure = _class_prob("structure")
            prob_floor = _class_prob("floor")
            prob_human_candidate = _class_prob("human_candidate", aliases=("human",))
            prob_human = prob_human_candidate
            prob_ghost_return = _class_prob("ghost_return")
            p_acc = None
            if acc_logits is not None:
                p_acc = torch.sigmoid(acc_logits.squeeze(0).squeeze(0)).detach().cpu().numpy()

            # KPConvTemporalSegmenter predicts only the center frame of the
            # temporal window. Bind labels back to that buffered frame, not the
            # newest frame just appended above.
            center_rec = frame_buffer[center_idx]
            _, _, center_frame_num, _, _, _, center_frame_df = center_rec
            if last_emitted_frame_num is not None and center_frame_num == last_emitted_frame_num:
                continue

            if len(preds) != len(center_frame_df):
                log.warning(
                    "Prediction count mismatch (center frame); trimming to frame size.",
                    preds=len(preds),
                    frame_points=len(center_frame_df),
                    center_frame=center_frame_num,
                    current_frame=frame_num,
                )
            n_out = min(len(preds), len(center_frame_df))
            pred_labels = [_bucket_order[int(preds[i])] for i in range(n_out)]
            all_preds.extend(pred_labels)

            doppler_col = "doppler" if "doppler" in center_frame_df.columns else None
            snr_col = "snr" if "snr" in center_frame_df.columns else None
            has_cam = "cam_u" in center_frame_df.columns
            provenance_int_cols = (
                "radar_frame_num",
                "frame_num",
                "range_bin",
                "doppler_bin",
            )
            emitted_feature_cols = RD_SCALAR_FEATURE_COLS + DIRECTNESS_FEATURE_COLS

            for i, (_, row) in enumerate(center_frame_df.iloc[:n_out].iterrows()):
                entry = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "pred_class": pred_labels[i],
                    "pred_conf": round(float(pred_conf[i]), 4),
                    "pred_prob_structure": round(float(prob_structure[i]), 4),
                    "pred_prob_floor": round(float(prob_floor[i]), 4),
                    "pred_prob_human": round(float(prob_human[i]), 4),
                    "pred_prob_human_candidate": round(float(prob_human_candidate[i]), 4),
                    "pred_prob_ghost_return": round(float(prob_ghost_return[i]), 4),
                    "doppler": float(row[doppler_col]) if doppler_col else 0.0,
                    "snr": float(row[snr_col]) if snr_col else 0.0,
                    "frame": int(center_frame_num),
                }
                if p_acc is not None:
                    acc_prob = float(p_acc[i])
                    entry["p_acc"] = round(acc_prob, 4)
                    entry["acc_confidence"] = round(abs(acc_prob - 0.5) * 2.0, 4)
                for col in provenance_int_cols:
                    if col in center_frame_df.columns:
                        value = row[col]
                        try:
                            if value == value:
                                entry[col] = int(value)
                        except (TypeError, ValueError):
                            pass
                for col in emitted_feature_cols:
                    if col in center_frame_df.columns:
                        value = row[col]
                        try:
                            if value == value:
                                entry[col] = float(value)
                        except (TypeError, ValueError):
                            pass
                if has_cam:
                    u = row["cam_u"]
                    v = row["cam_v"]
                    entry["cam_u"] = float(u) if u == u else None  # NaN check
                    entry["cam_v"] = float(v) if v == v else None
                    entry["cam_z"] = float(row["cam_z"])
                    entry["cam_valid"] = bool(row["cam_valid"])
                results.append(entry)

            last_emitted_frame_num = center_frame_num

    counts = Counter(all_preds)
    log.debug("Predictions", total=len(all_preds), **counts)

    return results


# =============================================================================
# Piper TTS
# =============================================================================

class PiperTTS:
    """Compatibility wrapper around the standalone TTS engine with backend fallback."""

    def __init__(self):
        from guidance import TTSEngine

        self._tts = TTSEngine(model=str(PIPER_VOICE), backend="auto")

    def speak(self, text: str) -> None:
        log.info("[NAV]", guidance=text)
        self._tts.speak_async(text)

    def stop(self) -> None:
        return None


# =============================================================================
# Async LLM thread
# =============================================================================

_scene_queue: queue.Queue = queue.Queue(maxsize=1)


def publish_scene(scene: dict) -> None:
    try:
        _scene_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        _scene_queue.put_nowait(scene)
    except queue.Full:
        pass


def llm_worker(stop_event: threading.Event) -> None:
    log.info("LLM worker starting...")

    llm_engine = None
    use_llm = False

    try:
        from guidance import GuidanceEngine
        llm_engine = GuidanceEngine(
            model=LLM_MODEL,
            min_interval_s=LLM_MIN_WAIT,
            ollama_host=OLLAMA_HOST,
            change_threshold=False,
        )
        log.info("LLM guidance engine ready.", model=LLM_MODEL, host=OLLAMA_HOST)
        use_llm = True
    except Exception as e:
        log.warning(f"LLM unavailable ({e}), using rule-based guidance.")

    from guidance import FastGuidanceEngine
    fast_engine = FastGuidanceEngine(min_interval_s=FAST_GUIDANCE_MIN_WAIT)

    tts = PiperTTS()
    tts.speak("Navigation system ready.")
    log.info("LLM worker ready.", use_llm=use_llm)

    while not stop_event.is_set():
        try:
            scene = _scene_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            guidance = ""
            if llm_engine is not None:
                guidance = llm_engine.get_guidance(scene) or ""
            if not guidance:
                guidance = fast_engine.get_guidance(scene) or ""
            if guidance:
                log.info("Guidance", text=guidance)
                tts.speak(guidance)
        except Exception as e:
            log.error(f"Guidance error: {e}")

    log.info("LLM worker stopped.")


# =============================================================================
# Scene window
# =============================================================================

_scene_window: deque = deque()  # current-session points only


# =============================================================================
# ADC processing
# =============================================================================

def run_adc_to_pointcloud(session_dir: str) -> tuple:
    from perception.adc_to_pointcloud_v6 import process_session_fast

    log.adc_start(Path(session_dir).name)
    t0 = time.monotonic()

    try:
        processor = get_processor()
        success, detections, csv_path = process_session_fast(processor, session_dir)
        elapsed = time.monotonic() - t0

        if not success:
            log.adc_error(Path(session_dir).name, csv_path or "unknown error")
            return False, 0, ""

        log.adc_done(Path(session_dir).name, elapsed, detections, success=True)
        return True, detections, csv_path

    except Exception as e:
        log.adc_error(Path(session_dir).name, str(e))
        return False, 0, ""


# =============================================================================
# Deletion
# =============================================================================

def delete_session(session_dir: str) -> None:
    try:
        shutil.rmtree(session_dir)
        log.delete_done(Path(session_dir).name)
    except Exception as e:
        log.delete_error(Path(session_dir).name, str(e))


# =============================================================================
# Branch 3 — BEVUNet free-space inference
# =============================================================================

def run_unet_freespace(session_dir: str, frame_num: int) -> dict:
    """
    Run optional Branch 3 BEVUNet free-space inference for one representative
    frame. Returns nav_decision-compatible safe defaults when unavailable.
    """
    unet = get_unet_model()
    if unet is None:
        return _unet_unavailable("unet_unavailable")

    try:
        import numpy as np
        import torch
        from models.hybrid_rd_runtime import export_session_sidecars

        config = _build_runtime_rd_patch_config(include_rd_cube=True)
        export_session_sidecars(
            session_dir,
            cfg_path=MMWAVE_CFG_PATH,
            config=config,
            overwrite=HYBRID_RD_OVERWRITE_SIDECARS,
        )

        frame_index = int(frame_num) - int(config.frame_number_offset)
        frame_path = (
            Path(session_dir)
            / config.sidecar_subdir
            / "frames"
            / f"rd_frame_{frame_index:06d}.npz"
        )
        if not frame_path.exists():
            return _unet_unavailable("rd_sidecar_missing")

        mb = get_map_builder()
        prior = None
        if mb is not None:
            try:
                prior = mb.get_polar_prior()
            except Exception as e:
                log.debug(f"Branch 3: polar prior unavailable: {e}")

        in_channels = int(getattr(unet, "in_channels", 25))
        with np.load(frame_path) as payload:
            if _unet_mod is None:
                return _unet_unavailable("unet_module_missing")
            inp = _unet_mod.input_from_npz_payload(
                payload,
                prior,
                in_channels=in_channels,
            )

        if inp is None:
            return _unet_unavailable("incompatible_rd_sidecar")

        device = next(unet.parameters()).device
        inp = inp.to(device)
        with torch.no_grad():
            prob_map = unet(inp).squeeze(0).detach().cpu().numpy()

        fs = _unet_mod.sector_probabilities(prob_map, axis=UNET_SECTOR_AXIS)
        fs["frame"] = int(frame_num)
        fs["sector_axis"] = UNET_SECTOR_AXIS
        return fs

    except Exception as e:
        log.warning(f"Branch 3: UNet freespace failed for frame {frame_num}: {e}")
        return _unet_unavailable("unet_inference_failed")


# =============================================================================
# Processing loop
# =============================================================================

def processing_loop(ready_queue: queue.Queue, stop_event: threading.Event) -> None:
    sessions_ok     = 0
    sessions_failed = 0

    log.info("Processing loop started. Warming up components...")
    get_processor()
    get_model()
    get_unet_model()

    from scene_pipeline import aggregate_scene

    # Verify soft structural import at startup so failures are visible
    try:
        from perception.corridor_soft_structural import process_frame_group
        log.info("Soft structural: corridor_soft_structural.py loaded OK.")
    except Exception as e:
        log.warning(f"Soft structural import FAILED: {e} — all soft features will be zero")

    log.info("Processing loop ready. Waiting for sessions...")

    while not (stop_event.is_set() and ready_queue.empty()):

        qsize = ready_queue.qsize()
        if qsize >= MAX_QUEUE_WARN:
            log.queue_warning(qsize)

        try:
            session_dir = ready_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        log.info("Processing session", session=Path(session_dir).name)

        success, detections, csv_path = run_adc_to_pointcloud(session_dir)

        if not success:
            sessions_failed += 1
        else:
            sessions_ok += 1
            try:
                t0     = time.monotonic()
                points = run_inference_on_csv(csv_path, session_dir=session_dir)
                elapsed_inf = time.monotonic() - t0
                log.debug("Inference done",
                          points=len(points),
                          elapsed=f"{elapsed_inf:.2f}s")

                import numpy as np

                # --- Per-sector structure breakdown ---
                struct_pts = [p for p in points if p["pred_class"] == "structure"]
                floor_pts  = [p for p in points if p["pred_class"] == "floor"]
                human_pts  = [p for p in points if p["pred_class"] == "human"]

                ZONE_L = -0.5
                ZONE_R =  0.5

                if struct_pts:
                    import math
                    # Azimuth-based sector debug — matches corridor_open_directions logic
                    CENTER_AZ = 20.0
                    SIDE_AZ   = 60.0
                    in_block  = [p for p in struct_pts if 0.3 <= p["y"] < 1.0]
                    for label, condition in [
                        ("center_az", lambda p: math.degrees(math.atan2(abs(p["x"]), max(p["y"], 0.1))) <= CENTER_AZ),
                        ("left_az",   lambda p: p["x"] <= 0 and CENTER_AZ < math.degrees(math.atan2(abs(p["x"]), max(p["y"], 0.1))) <= SIDE_AZ),
                        ("right_az",  lambda p: p["x"] >  0 and CENTER_AZ < math.degrees(math.atan2(abs(p["x"]), max(p["y"], 0.1))) <= SIDE_AZ),
                    ]:
                        blk = [p for p in in_block if condition(p)]
                        all_s = [p for p in struct_pts if condition(p)]
                        if all_s:
                            log.debug(f"Struct/{label}",
                                      total=len(all_s),
                                      within_1m=len(blk),
                                      y_min=round(min(p["y"] for p in all_s), 2),
                                      y_med=round(sorted(p["y"] for p in all_s)[len(all_s)//2], 2))

                # --- Floor distribution ---
                if floor_pts:
                    fy = sorted(p["y"] for p in floor_pts)
                    log.debug("Floor dist",
                              count=len(floor_pts),
                              y_min=round(fy[0], 2),
                              y_med=round(fy[len(fy)//2], 2),
                              y_max=round(fy[-1], 2))

                # --- Human detections ---
                if human_pts:
                    hy = sorted(p["y"] for p in human_pts)
                    log.debug("Human dist",
                              count=len(human_pts),
                              y_min=round(hy[0], 2),
                              y_med=round(hy[len(hy)//2], 2))

                # Geo floor correction — reclassify near-field structure
                # points that match floor geometry. KPConv never saw floor in
                # training (Z filter cut floor from labels), so near-field
                # low-Z, low-Doppler returns are often misclassified as structure.
                # Signature: y < 2.5m, z < 0.05m, |doppler| < 0.2 m/s
                corrected_points = []
                geo_floor_count  = 0
                for p in points:
                    if (p.get("pred_class") == "structure"
                            and float(p.get("y", 99.0)) < 2.5
                            and float(p.get("z", 0.0))  < 0.05
                            and abs(float(p.get("doppler", 0.0))) < 0.2):
                        p = dict(p)
                        p["pred_class"] = "floor"
                        geo_floor_count += 1
                    corrected_points.append(p)
                if geo_floor_count:
                    log.debug("Geo floor correction",
                              reclassified=geo_floor_count,
                              total_floor=sum(1 for p in corrected_points
                                              if p["pred_class"] == "floor"))
                points = corrected_points

                # Step 1 — Runtime ego-Doppler directness annotation (shadow mode)
                # Annotates points with ego_expected_doppler_mps, ego_doppler_residual_mps,
                # ego_residual_abs_z, ego_inlier_flag, p_ego, p_dir_doppler.
                # Does NOT change map behavior yet — shadow mode only.
                try:
                    from perception.directness_runtime import (
                        DirectnessConfig,
                        estimate_ego_velocity as _estimate_ego_vel,
                        annotate_directness as _annotate_dir,
                        build_directness_evidence as _build_dir_evidence,
                    )
                    _dir_config = DirectnessConfig()
                    _ego_est = _estimate_ego_vel(points, _dir_config)
                    points = _annotate_dir(points, _ego_est, _dir_config)
                    _dir_evidence = _build_dir_evidence(points, _ego_est)
                    log.debug("Directness annotation done",
                              ego_avail=_ego_est.get("available", False),
                              p_ego=round(_ego_est.get("p_ego", 0.0), 3),
                              n_candidates=_ego_est.get("n_candidate_points", 0))
                except Exception as _de:
                    log.debug(f"Directness annotation failed (shadow mode): {_de}")

                # Branch 2 — push corrected points to MapBuilder (non-blocking)
                mb = get_map_builder()
                if mb is not None and points:
                    mb.push_classified_points(points)

                _scene_window.clear()
                _scene_window.extend(points)
                scene = aggregate_scene(list(_scene_window), window_frames=SCENE_WINDOW)
                # Attach directness evidence channel to scene
                try:
                    scene.setdefault("evidence_channels", {})["directness"] = _dir_evidence
                except Exception:
                    pass

                # Improved blocking detection — azimuth-based, z > 0.05m gate,
                # extends range to 0.5–2.5m (vs original 0.3–1.0m).
                try:
                    _struct_pts = [p for p in points if p.get("pred_class") == "structure"]
                    if _struct_pts:
                        import numpy as _np2
                        _arr = _np2.array(
                            [[p["x"], p["y"], p["z"]] for p in _struct_pts],
                            dtype=_np2.float32,
                        )
                        _xs, _ys, _zs = _arr[:, 0], _arr[:, 1], _arr[:, 2]
                        _in_zone = (_ys >= 0.5) & (_ys < 2.5)
                        if _np2.any(_in_zone):
                            _bx = _xs[_in_zone]
                            _by = _ys[_in_zone]
                            _bz = _zs[_in_zone]
                            _az = _np2.degrees(_np2.arctan2(
                                _np2.abs(_bx), _np2.maximum(_by, 0.1)
                            ))
                            _above = _bz > 0.05
                            _c_cnt = int(_np2.sum(
                                (_az <= 20.0) & (_np2.abs(_bx) < 0.5) & _above
                            ))
                            _l_cnt = int(_np2.sum(
                                (_bx <= 0) & (_az > 20.0) & (_az <= 60.0) & _above
                            ))
                            _r_cnt = int(_np2.sum(
                                (_bx >  0) & (_az > 20.0) & (_az <= 60.0) & _above
                            ))
                            scene["blocking"]["center_points"] = _c_cnt
                            scene["blocking"]["left_points"]   = _l_cnt
                            scene["blocking"]["right_points"]  = _r_cnt
                            scene["open_directions"] = {
                                "left_clear":   _l_cnt < 20,
                                "center_clear": _c_cnt < 15,
                                "right_clear":  _r_cnt < 20,
                            }
                except Exception as _be:
                    log.debug(f"Improved blocking calc failed: {_be}")

                # Branch 2 — enrich scene with MapBuilder anomaly flags
                mb = get_map_builder()
                if mb is not None:
                    try:
                        scene["map_anomalies"] = mb.query_anomalies(0.0, 1.0, radius_m=0.5)
                    except Exception as _me:
                        log.debug(f"MapBuilder anomaly query failed: {_me}")

                # Branch 3 — optional BEVUNet free-space inference.
                try:
                    last_frame = int(points[-1]["frame"]) if points else 0
                    scene["unet_freespace"] = run_unet_freespace(session_dir, last_frame)
                    if scene["unet_freespace"].get("available"):
                        _fs = scene["unet_freespace"]
                        log.debug("Branch 3 UNet",
                                  center=f"{_fs['center']:.2f}",
                                  left=f"{_fs['left']:.2f}",
                                  right=f"{_fs['right']:.2f}")
                except Exception as _ue:
                    log.debug(f"Branch 3 UNet scene enrichment failed: {_ue}")
                    scene["unet_freespace"] = _unet_unavailable("unet_scene_failed")

                od = scene.get("open_directions", {})
                log.debug("Scene dict",
                          structure_detected=scene.get("structure", {}).get("detected"),
                          structure_elements=len(scene.get("structure", {}).get("elements", [])),
                          floor_detected=scene.get("floor", {}).get("detected"),
                          humans_detected=scene.get("humans", {}).get("detected"),
                          center_clear=od.get("center_clear"),
                          left_clear=od.get("left_clear"),
                          right_clear=od.get("right_clear"))

                publish_scene(scene)

            except Exception as e:
                log.error(f"Inference/scene error: {e}")

        delete_session(session_dir)
        ready_queue.task_done()
        log.loop_summary(sessions_ok, sessions_failed)

    log.info("Processing loop stopped.", ok=sessions_ok, failed=sessions_failed)


# =============================================================================
# Main
# =============================================================================

def main():
    model_pt = _resolve_model_pt()
    if NAV_PERCEPTION_MODE not in {"pointcloud", "pointcloud_rd_patch"}:
        raise SystemExit(
            f"[ERROR] NAV_PERCEPTION_MODE={NAV_PERCEPTION_MODE!r} is not implemented in this loop. "
            "Dense RD / RD-scene modes are shelved for a later iteration."
        )

    log.info("=" * 55)
    log.info("Radar Navigation Loop — checkpoint-driven model selection")
    log.info(f"Perception  : {NAV_PERCEPTION_MODE}")
    log.info(f"Output root : {OUTPUT_ROOT}")
    log.info(f"CFG         : {MMWAVE_CFG_PATH}")
    log.info(f"Model       : {model_pt}")
    log.info(f"UNet        : {UNET_PT if UNET_ENABLED else 'disabled'}")
    log.info(f"LLM model   : {LLM_MODEL}")
    log.info(f"LLM host    : {OLLAMA_HOST}")
    log.info(f"Range gate  : {MIN_RANGE_M}-{MAX_RANGE_M}m "
             f"({MIN_RANGE_M*3.28:.1f}-{MAX_RANGE_M*3.28:.1f}ft)")
    log.info(f"Log file    : {get_log_path()}")
    log.info("=" * 55)

    if not os.path.exists(MMWAVE_CFG_PATH):
        raise SystemExit(f"[ERROR] CFG not found: {MMWAVE_CFG_PATH}")
    if not os.path.exists(model_pt):
        raise SystemExit(f"[ERROR] Model not found: {model_pt}")

    log.info("Ollama will be checked lazily by the LLM worker; rule-based fallback remains available.")

    # Branch 2 — start MapBuilder
    global _map_builder
    if MAP_ENABLED:
        try:
            from scene_pipeline import MapBuilder
            _map_builder = MapBuilder(
                cell_m  = MAP_CELL_M,
                range_m = MAP_RANGE_M,
                decay   = MAP_DECAY,
                session_s = float(os.environ.get("NAV_SESSION_DURATION_S", "1.5")),
                map_update_mode=MAP_UPDATE_MODE,
            )
            _map_builder.start()
            log.info("MapBuilder started.", cell_m=MAP_CELL_M, range_m=MAP_RANGE_M, map_update_mode=MAP_UPDATE_MODE)
        except Exception as e:
            log.warning(f"MapBuilder failed to start: {e} — map features disabled.")
            _map_builder = None

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ready_queue = queue.Queue()
    stop_event  = threading.Event()

    def _handle_stop(sig, frame):
        if not stop_event.is_set():
            log.shutdown("Ctrl+C / SIGTERM")
            stop_event.set()

    signal.signal(signal.SIGINT,  _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    llm_thread = threading.Thread(
        target=llm_worker,
        args=(stop_event,),
        name="LLMWorker",
        daemon=True,
    )
    llm_thread.start()

    try:
        from perception.radar_only_recorder import record_continuous
    except ImportError:
        raise SystemExit("[ERROR] radar_only_recorder.py not found.")

    recorder_thread = threading.Thread(
        target=record_continuous,
        args=(ready_queue, stop_event),
        name="RecorderThread",
        daemon=True,
    )
    recorder_thread.start()

    try:
        processing_loop(ready_queue, stop_event)
    finally:
        recorder_thread.join(timeout=15.0)
        llm_thread.join(timeout=10.0)

        while not ready_queue.empty():
            try:
                delete_session(ready_queue.get_nowait())
            except queue.Empty:
                break

        try:
            for p in Path(OUTPUT_ROOT).iterdir():
                if p.is_dir() and p.name.startswith("session_"):
                    shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

        log.info("Navigation loop exited cleanly.")


def _self_test():
    """
    Smoke-test the Branch 2 per-session logic (geo floor correction and improved
    blocking detection) against hardcoded point fixtures.  No radar or Ollama needed.
    Run with: python navigation_loop.py --self-test
    """
    import numpy as np
    print("--- navigation_loop --self-test ---")

    # --- Geo floor correction ---
    raw_pts = [
        # Should be reclassified: low-z, near, slow Doppler
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.02, "doppler": 0.05, "snr": 10.0},
        # Should stay structure: z too high
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.30, "doppler": 0.05, "snr": 10.0},
        # Should stay structure: Doppler too high
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.02, "doppler": 0.50, "snr": 10.0},
        # Should stay structure: y too far
        {"pred_class": "structure", "x": 0.0, "y": 3.0, "z": 0.02, "doppler": 0.05, "snr": 10.0},
        # Floor point — unchanged
        {"pred_class": "floor",     "x": 0.0, "y": 1.5, "z": 0.01, "doppler": 0.00, "snr": 8.0},
    ]
    corrected = []
    geo_count = 0
    for p in raw_pts:
        if (p.get("pred_class") == "structure"
                and float(p.get("y", 99.0)) < 2.5
                and float(p.get("z", 0.0))  < 0.05
                and abs(float(p.get("doppler", 0.0))) < 0.2):
            p = dict(p); p["pred_class"] = "floor"; geo_count += 1
        corrected.append(p)

    assert geo_count == 1, f"Expected 1 geo-floor reclassification, got {geo_count}"
    assert corrected[0]["pred_class"] == "floor",      "Point 0 should be reclassified to floor"
    assert corrected[1]["pred_class"] == "structure",  "Point 1 z too high — stay structure"
    assert corrected[2]["pred_class"] == "structure",  "Point 2 doppler too high — stay structure"
    assert corrected[3]["pred_class"] == "structure",  "Point 3 y too far — stay structure"
    print(f"PASS: geo floor correction reclassified {geo_count}/4 structure points")

    # --- Improved blocking detection ---
    struct_pts = [
        # Center: azimuth ≤20°, |x|<0.5, z>0.05, y in [0.5, 2.5)
        {"x": 0.1, "y": 1.0, "z": 0.3, "pred_class": "structure"},
        {"x": 0.0, "y": 0.8, "z": 0.4, "pred_class": "structure"},
        # Left: x<=0, 20°<az<=60°, z>0.05
        {"x": -0.8, "y": 1.0, "z": 0.3, "pred_class": "structure"},
        # Right: x>0, 20°<az<=60°, z>0.05
        {"x":  0.9, "y": 1.0, "z": 0.3, "pred_class": "structure"},
        # Floor-level — should be excluded by z>0.05 gate
        {"x": 0.1, "y": 1.0, "z": 0.01, "pred_class": "structure"},
    ]
    arr = np.array([[p["x"], p["y"], p["z"]] for p in struct_pts], dtype=np.float32)
    xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
    in_zone = (ys >= 0.5) & (ys < 2.5)
    bx, by, bz = xs[in_zone], ys[in_zone], zs[in_zone]
    az = np.degrees(np.arctan2(np.abs(bx), np.maximum(by, 0.1)))
    above = bz > 0.05
    c_cnt = int(np.sum((az <= 20.0) & (np.abs(bx) < 0.5) & above))
    l_cnt = int(np.sum((bx <= 0) & (az > 20.0) & (az <= 60.0) & above))
    r_cnt = int(np.sum((bx >  0) & (az > 20.0) & (az <= 60.0) & above))

    assert c_cnt == 2, f"Expected 2 center points, got {c_cnt}"
    assert l_cnt == 1, f"Expected 1 left point, got {l_cnt}"
    assert r_cnt == 1, f"Expected 1 right point, got {r_cnt}"
    print(f"PASS: improved blocking detection — center={c_cnt}, left={l_cnt}, right={r_cnt}")

    # --- MapBuilder integration (if available) ---
    try:
        from scene_pipeline import MapBuilder
        mb = MapBuilder()
        mb.push_classified_points(struct_pts)
        anomalies = mb.query_anomalies(0.0, 1.0, radius_m=0.5)
        assert isinstance(anomalies, dict), "query_anomalies should return a dict"
        assert "has_history" in anomalies
        print(f"PASS: MapBuilder push_classified_points + query_anomalies → {anomalies}")
    except ImportError:
        print("SKIP: map_builder not importable in this environment")

    print("\nAll --self-test assertions passed.")


if __name__ == "__main__":
    import sys as _sys
    if "--self-test" in _sys.argv:
        _self_test()
    else:
        main()
