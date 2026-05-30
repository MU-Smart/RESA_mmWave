"""Strict live navigation loop for Jetson deployment.

Pipeline per session:
    radar recorder -> ADC point cloud -> strict feature preparation
    -> checkpoint-selected inference -> scene aggregation -> decision
    -> LLM/rule guidance -> TTS -> session cleanup

Only LLM/TTS are allowed to degrade at runtime. Perception, features, model
selection, MapBuilder startup, and decision generation fail explicitly.
"""

from __future__ import annotations

import importlib
import queue
import shutil
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Direct script execution starts with branch2/ on sys.path. Add the repo root
# only so the shared config package can be imported; canonical paths live there.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import (
    ADC_TO_POINTCLOUD_MODULE,
    BRANCH3_UNET_MODULE,
    CORRIDOR_SOFT_STRUCTURAL_MODULE,
    DGCNN_MODEL_MODULE,
    DIRECTNESS_RUNTIME_MODULE,
    GUIDANCE_MODULE,
    HYBRID_RD_MODEL_MODULE,
    HYBRID_RD_RUNTIME_MODULE,
    KPCONV_MODEL_MODULE,
    RADAR_RECORDER_MODULE,
    SCENE_PIPELINE_MODULE,
)
from navigation_config import (
    DEFAULT_NAVIGATION_CONFIG,
    NavigationConfig,
    RDPatchConfig,
    config_summary,
    install_import_paths,
    validate_static_config,
)


install_import_paths(DEFAULT_NAVIGATION_CONFIG)

_scene_pipeline = importlib.import_module(SCENE_PIPELINE_MODULE)
compute_nav_decision = _scene_pipeline.compute_nav_decision
get_log_path = _scene_pipeline.get_log_path
get_logger = _scene_pipeline.get_logger


log = get_logger("Loop")


SOFT_STRUCTURAL_FEATURES = {
    "z_above_floor",
    "corridor_margin",
    "wall_anomaly",
    "floor_ang",
    "wall_ang",
    "has_floor",
    "has_wall",
}
RD_SCALAR_FEATURES = {
    "rd_entropy",
    "rd_doppler_spread",
    "rd_anisotropy",
    "rd_peak_ratio",
}
DIRECTNESS_FEATURES = {
    "ego_doppler_residual_mps",
    "ego_residual_abs_z",
    "ego_inlier_flag",
    "p_ego",
    "p_dir_doppler",
}
ADC_RENAME = {
    "doppler_mps": "doppler",
    "v": "doppler",
    "power_snr": "snr",
}


@dataclass
class CheckpointContract:
    payload: dict[str, Any]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    bucket_order: list[str]
    feature_cols: list[str]
    window_size: int
    n_points: int
    uses_rd_patch: bool
    uses_ra_patch: bool
    has_acc_head: bool
    ra_patch_source: str
    ra_patch_az_mode: str
    ra_patch_az_fft_size: int
    ra_patch_log_scale: bool
    ra_patch_normalize_by_frame_max: bool
    ra_patch_row_edge_mode: str


@dataclass
class ModelContext:
    model: Any
    contract: CheckpointContract


@dataclass
class UNetContext:
    payload: dict[str, Any]
    model: Any
    module: Any
    device: Any
    checkpoint: Path


@dataclass
class NavigationRuntime:
    config: NavigationConfig
    processor: Any
    model_ctx: ModelContext
    map_builder: Any | None
    unet_ctx: UNetContext | None = None


def _feature_set(feature_cols: list[str]) -> set[str]:
    return {str(col) for col in feature_cols}


def _require_columns(df, columns: set[str], *, context: str) -> None:
    missing = sorted(col for col in columns if col not in df.columns)
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _assert_finite(df, columns: set[str], *, context: str) -> None:
    for col in sorted(columns):
        values = df[col].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{context} has non-finite values in {col!r}")


def _checkpoint_contract(payload: dict[str, Any], config: NavigationConfig) -> CheckpointContract:
    required = {"model_state", "feature_means", "feature_stds", "feature_cols"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"Checkpoint missing required field(s): {missing}")

    feature_cols = [str(col) for col in payload["feature_cols"]]
    if not feature_cols:
        raise ValueError("Checkpoint feature_cols is empty")

    feature_means = np.asarray(payload["feature_means"], dtype=np.float32)
    feature_stds = np.asarray(payload["feature_stds"], dtype=np.float32)
    if len(feature_means) != len(feature_cols) or len(feature_stds) != len(feature_cols):
        raise ValueError(
            "Checkpoint feature statistics must match feature_cols length "
            f"({len(feature_cols)} cols, {len(feature_means)} means, {len(feature_stds)} stds)"
        )
    feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)

    uses_rd_patch = bool(payload.get("uses_rd_patch", False))
    uses_ra_patch = bool(payload.get("uses_ra_patch", payload.get("uses_ra_patches", False)))
    if config.perception_mode == "pointcloud_rd_patch" and not uses_rd_patch:
        raise ValueError("pointcloud_rd_patch mode requires a checkpoint with uses_rd_patch=true")
    if config.perception_mode == "pointcloud" and uses_rd_patch:
        raise ValueError("Loaded checkpoint uses RD patches, but config perception_mode='pointcloud'")
    if uses_ra_patch and not uses_rd_patch:
        raise ValueError("RA patch checkpoints require RD patches")

    expected_offset = payload.get("rd_patch_frame_number_offset")
    if uses_rd_patch and expected_offset is not None:
        expected_offset = int(expected_offset)
        actual_offset = int(config.rd_patch.frame_number_offset)
        if expected_offset != actual_offset:
            raise ValueError(
                "RD frame offset mismatch: "
                f"checkpoint={expected_offset}, config={actual_offset}"
            )

    checkpoint_state = payload["model_state"]
    has_acc_head = bool(
        payload.get(
            "has_acc_head",
            any(str(key).startswith("acc_head.") for key in checkpoint_state.keys()),
        )
    )

    return CheckpointContract(
        payload=payload,
        feature_means=feature_means,
        feature_stds=feature_stds,
        bucket_order=[str(label) for label in payload.get("bucket_order", ["structure", "floor", "human"])],
        feature_cols=feature_cols,
        window_size=int(payload.get("window_size", 3)),
        n_points=int(payload.get("points_per_frame", 128)),
        uses_rd_patch=uses_rd_patch,
        uses_ra_patch=uses_ra_patch,
        has_acc_head=has_acc_head,
        ra_patch_source=str(payload.get("ra_patch_source", "true_ra")),
        ra_patch_az_mode=str(payload.get("ra_patch_az_mode", "tx0_only")),
        ra_patch_az_fft_size=int(payload.get("ra_patch_az_fft_size", 64)),
        ra_patch_log_scale=bool(payload.get("ra_patch_log_scale", True)),
        ra_patch_normalize_by_frame_max=bool(payload.get("ra_patch_normalize_by_frame_max", True)),
        ra_patch_row_edge_mode=str(payload.get("ra_patch_row_edge_mode", "zero_pad")),
    )


def _import_attr(module_name: str, attr_name: str) -> Any:
    return getattr(importlib.import_module(module_name), attr_name)


def load_model(config: NavigationConfig) -> ModelContext:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(config.model_pt, map_location=device, weights_only=False)
    contract = _checkpoint_contract(payload, config)
    state_dict = payload["model_state"]

    encoder_type = str(payload.get("encoder_type", "dgcnn")).strip().lower()
    rd_patch_embed_dim = int(payload.get("rd_patch_embed_dim", 0)) if contract.uses_rd_patch else 0
    if contract.uses_rd_patch and rd_patch_embed_dim <= 0:
        raise ValueError("RD-patch checkpoint must include a positive rd_patch_embed_dim")
    ra_patch_embed_dim = int(payload.get("ra_patch_embed_dim", 0)) if contract.uses_ra_patch else 0
    if contract.uses_ra_patch and ra_patch_embed_dim <= 0:
        ra_patch_embed_dim = rd_patch_embed_dim

    n_features = len(contract.feature_cols) + rd_patch_embed_dim + ra_patch_embed_dim

    if encoder_type == "kpconv":
        model_cls = _import_attr(KPCONV_MODEL_MODULE, "KPConvTemporalSegmenter")
        base_model = model_cls(
            n_features=n_features,
            n_classes=len(contract.bucket_order),
            window_size=contract.window_size,
            k=int(payload.get("k", 20)),
            emb_dims=int(payload.get("emb_dims", 256)),
            temporal_layers=int(payload.get("temporal_layers", 2)),
            temporal_heads=int(payload.get("temporal_heads", 4)),
            dropout=0.0,
            frame_meta_dim=int(payload.get("frame_meta_dim", 5)),
            gate_hidden=int(payload.get("gate_hidden", 128)),
            radius_1=float(payload.get("radius_1", 0.35)),
            radius_2=float(payload.get("radius_2", 0.50)),
            radius_3=float(payload.get("radius_3", 0.70)),
            sigma_factor=float(payload.get("sigma_factor", 2.5)),
            n_kernel_points=int(payload.get("n_kernel_points", 15)),
        )
    elif encoder_type == "dgcnn":
        model_cls = _import_attr(DGCNN_MODEL_MODULE, "DopplerAwareTemporalDGCNNSegmenter")
        base_model = model_cls(
            n_features=n_features,
            n_classes=len(contract.bucket_order),
            k=int(payload.get("k", 28)),
            emb_dims=int(payload.get("emb_dims", 256)),
            dropout=0.0,
            window_size=contract.window_size,
            temporal_layers=int(payload.get("temporal_layers", 2)),
            temporal_heads=int(payload.get("temporal_heads", 4)),
            frame_meta_dim=int(payload.get("frame_meta_dim", 5)),
            gate_hidden=int(payload.get("gate_hidden", 128)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint encoder_type={encoder_type!r}")

    if contract.uses_rd_patch:
        wrapper_cls = _import_attr(HYBRID_RD_MODEL_MODULE, "RDPatchTemporalSegmenter")
        model = wrapper_cls(
            base_model,
            patch_channels=int(payload.get("rd_patch_channels", 1)),
            patch_embed_dim=rd_patch_embed_dim,
            use_ra_patches=contract.uses_ra_patch,
            patch_hidden_channels=tuple(payload.get("rd_patch_hidden_channels", (8, 16))),
            has_acc_head=contract.has_acc_head,
            acc_head_hidden_dim=int(payload.get("acc_head_hidden_dim", 64)),
        )
    else:
        model = base_model

    if hasattr(model, "base_model") and not any(str(key).startswith("base_model.") for key in state_dict):
        prefixed: dict[str, Any] = {}
        for key, value in state_dict.items():
            key_str = str(key)
            if key_str.startswith(("rd_patch_encoder.", "ra_patch_encoder.", "acc_head.")):
                prefixed[key_str] = value
            else:
                prefixed["base_model." + key_str] = value
        state_dict = prefixed

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    log.info(
        "Model loaded.",
        path=config.model_pt,
        encoder=encoder_type,
        device=str(device),
        features=len(contract.feature_cols),
        points=contract.n_points,
        window=contract.window_size,
        rd_patch=contract.uses_rd_patch,
        ra_patch=contract.uses_ra_patch,
        buckets=str(contract.bucket_order),
    )
    return ModelContext(model=model, contract=contract)


def load_unet(config: NavigationConfig) -> UNetContext | None:
    """Load the Branch 3 U-Net checkpoint selected by NavigationConfig."""
    if not config.unet.enabled:
        return None

    import torch

    checkpoint = Path(config.unet.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint.exists():
        raise FileNotFoundError(
            "Branch 3 U-Net checkpoint is required but missing.\n"
            f"unet_checkpoint={checkpoint}\n"
            f"device={device}\n"
            f"perception_mode={config.perception_mode}"
        )

    unet_module = importlib.import_module(BRANCH3_UNET_MODULE)
    try:
        payload, model = unet_module.load_unet_checkpoint(checkpoint, map_location=str(device))
        model.to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Branch 3 U-Net checkpoint.\n"
            f"unet_checkpoint={checkpoint}\n"
            f"device={device}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc

    log.info("Branch 3 U-Net loaded.", path=checkpoint, device=str(device))
    return UNetContext(
        payload=payload,
        model=model,
        module=unet_module,
        device=device,
        checkpoint=checkpoint,
    )


def build_processor(config: NavigationConfig) -> Any:
    adc_mod = importlib.import_module(ADC_TO_POINTCLOUD_MODULE)
    RadarProcessor = adc_mod.RadarProcessor
    parse_mmwave_cfg = adc_mod.parse_mmwave_cfg

    radar_cfg = parse_mmwave_cfg(config.mmwave_cfg_path)
    processor = RadarProcessor(
        cfg_path=config.mmwave_cfg_path,
        cfg=radar_cfg,
        pfa=config.adc.pfa,
        min_snr=config.adc.min_snr,
        min_range_bin=config.adc.min_range_bin,
        clutter_mode=config.adc.clutter_mode,
        range_window=config.adc.range_window,
        doppler_window=config.adc.doppler_window,
        enable_aoa=config.adc.enable_aoa,
        max_range_m=config.adc.max_range_m,
        max_abs_azimuth_deg=config.adc.max_abs_azimuth_deg,
        frame_number_offset=config.rd_patch.frame_number_offset,
        write_rd_sidecars=config.perception_mode == "pointcloud_rd_patch",
        rd_sidecar_subdir=config.rd_patch.sidecar_subdir,
        rd_sidecar_clutter_mode=config.rd_patch.clutter_mode,
        rd_log_scale=not config.rd_patch.linear_power,
    )
    log.info("RadarProcessor ready.", cfg=config.mmwave_cfg_path)
    return processor


def build_map_builder(config: NavigationConfig) -> Any | None:
    if not config.map.enabled:
        return None
    MapBuilder = _import_attr(SCENE_PIPELINE_MODULE, "MapBuilder")

    valid_modes = {
        MapBuilder.MAP_UPDATE_MODE_UNIFORM,
        MapBuilder.MAP_UPDATE_MODE_EGO_ONLY,
        MapBuilder.MAP_UPDATE_MODE_EGO_DIRECTNESS,
        MapBuilder.MAP_UPDATE_MODE_LEARNED_ACC,
    }
    if config.map.update_mode not in valid_modes:
        raise ValueError(f"Unsupported MapBuilder update_mode={config.map.update_mode!r}")

    builder = MapBuilder(
        cell_m=config.map.cell_m,
        range_m=config.map.range_m,
        decay=config.map.decay,
        session_s=config.map.session_s,
        map_update_mode=config.map.update_mode,
    )
    builder.start()
    log.info(
        "MapBuilder started.",
        cell_m=config.map.cell_m,
        range_m=config.map.range_m,
        update_mode=config.map.update_mode,
    )
    return builder


def validate_feature_contract(runtime: NavigationRuntime) -> None:
    features = _feature_set(runtime.model_ctx.contract.feature_cols)
    map_mode = runtime.config.map.update_mode if runtime.config.map.enabled else "uniform"
    if features & SOFT_STRUCTURAL_FEATURES:
        importlib.import_module(CORRIDOR_SOFT_STRUCTURAL_MODULE)
    if features & DIRECTNESS_FEATURES:
        importlib.import_module(DIRECTNESS_RUNTIME_MODULE)
    if map_mode in {"ego_only", "ego_directness", "learned_acc"}:
        importlib.import_module(DIRECTNESS_RUNTIME_MODULE)
    if map_mode == "learned_acc" and not runtime.model_ctx.contract.has_acc_head:
        raise ValueError("MapBuilder update_mode='learned_acc' requires a checkpoint with an acc head")
    if features & RD_SCALAR_FEATURES and not runtime.model_ctx.contract.uses_rd_patch:
        raise ValueError("RD scalar features require an RD-patch checkpoint")
    if runtime.config.features.persist_score_source == "mapbuilder" and runtime.map_builder is None:
        raise ValueError("MapBuilder persist_score source requires a running MapBuilder")
    if runtime.config.unet.enabled and runtime.unet_ctx is None:
        raise ValueError("Branch 3 U-Net is enabled but no U-Net runtime context was loaded")


def build_runtime(config: NavigationConfig) -> NavigationRuntime:
    install_import_paths(config)
    validate_static_config(config)
    processor = build_processor(config)
    model_ctx = load_model(config)
    unet_ctx = load_unet(config)
    map_builder = None
    try:
        map_builder = build_map_builder(config)
        runtime = NavigationRuntime(
            config=config,
            processor=processor,
            model_ctx=model_ctx,
            map_builder=map_builder,
            unet_ctx=unet_ctx,
        )
        validate_feature_contract(runtime)
        return runtime
    except Exception:
        if map_builder is not None:
            map_builder.stop()
        raise


def _rename_adc_columns(df):
    renames = {
        src: dst
        for src, dst in ADC_RENAME.items()
        if src in df.columns and dst not in df.columns
    }
    return df.rename(columns=renames) if renames else df


def normalize_inference_schema(df, feature_cols: list[str]):
    import pandas as pd

    out = _rename_adc_columns(df.copy())
    if "radar_frame_num" not in out.columns:
        if "frame_num" in out.columns:
            out["radar_frame_num"] = out["frame_num"]
        elif "frame" in out.columns:
            out["radar_frame_num"] = out["frame"]

    _require_columns(out, {"x", "y", "z", "doppler", "snr", "radar_frame_num"}, context="ADC CSV")

    numeric_cols = {
        "x",
        "y",
        "z",
        "doppler",
        "snr",
        "range_m",
        "azimuth_deg",
        "elevation_deg",
        "radar_frame_num",
        "frame_num",
        "frame",
        "range_bin",
        "doppler_bin",
    } | _feature_set(feature_cols)

    for col in sorted(numeric_cols & set(out.columns)):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "range_m" not in out.columns:
        out["range_m"] = np.sqrt(out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2)
    if "azimuth_deg" not in out.columns:
        out["azimuth_deg"] = np.degrees(np.arctan2(out["x"], np.maximum(out["y"], 1e-6)))
    if "elevation_deg" not in out.columns:
        r_xy = np.sqrt(out["x"] ** 2 + out["y"] ** 2)
        out["elevation_deg"] = np.degrees(np.arctan2(out["z"], r_xy + 1e-6))

    out["radar_frame_num"] = out["radar_frame_num"].astype("int64")
    for col in ("range_bin", "doppler_bin"):
        if col in out.columns:
            out[col] = out[col].astype("int64")

    required_now = {"x", "y", "z", "doppler", "snr", "range_m", "radar_frame_num"}
    _assert_finite(out, required_now, context="ADC CSV")
    return out


def _compute_local_density(df, radius_m: float):
    densities = np.zeros(len(df), dtype=np.float32)
    for _, group in df.groupby("radar_frame_num", sort=False):
        idx = group.index.to_numpy()
        pts = group[["x", "y", "z"]].to_numpy(dtype=np.float32)
        if len(pts) <= 1:
            continue
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        counts = np.maximum((dist < radius_m).sum(-1) - 1, 0)
        densities[idx] = (np.log1p(counts) / np.log1p(100)).astype(np.float32)
    out = df.copy()
    out["local_density"] = densities
    return out


def _compute_session_persist_score(df, voxel_size_m: float):
    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    voxels = np.floor(pts / voxel_size_m).astype(np.int32)
    frames = df["radar_frame_num"].astype(int).to_numpy()
    unique_frames = len(np.unique(frames))
    if unique_frames == 0:
        raise ValueError("Cannot compute persist_score without frame ids")

    voxel_frame_count: dict[tuple[int, int, int], set[int]] = {}
    for i in range(len(df)):
        key = tuple(int(v) for v in voxels[i])
        voxel_frame_count.setdefault(key, set()).add(int(frames[i]))

    persist = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        key = tuple(int(v) for v in voxels[i])
        persist[i] = len(voxel_frame_count[key]) / unique_frames

    out = df.copy()
    out["persist_score"] = persist
    return out


def _compute_map_persist_score(runtime: NavigationRuntime, df):
    if runtime.map_builder is None:
        raise RuntimeError("MapBuilder is not running")
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    scores = runtime.map_builder.get_point_occupancy_scores(
        xyz,
        radius_m=runtime.config.map.score_radius_m,
    )
    if len(scores) != len(df):
        raise RuntimeError("MapBuilder returned an unexpected persist_score length")
    out = df.copy()
    out["persist_score"] = np.asarray(scores, dtype=np.float32)
    return out


def _compute_soft_structural(runtime: NavigationRuntime, df, missing_features: set[str]):
    module = importlib.import_module(CORRIDOR_SOFT_STRUCTURAL_MODULE)
    session_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    _, features = module.process_frame_group(
        anchor_pts=session_xyz,
        aggregated_pts=session_xyz,
        floor_dist_thresh=0.06,
        wall_dist_thresh=0.08,
        ransac_iters=runtime.config.features.soft_ransac_iters,
        compute_normals=True,
        normal_k=8,
    )

    out = df.copy()
    for col in sorted(missing_features):
        values = features.get(col)
        if values is None or len(values) != len(out):
            raise RuntimeError(f"Soft structural feature {col!r} was not produced")
        out[col] = np.asarray(values, dtype=np.float32)
    return out


def add_computed_features(runtime: NavigationRuntime, df):
    feature_cols = runtime.model_ctx.contract.feature_cols
    features = _feature_set(feature_cols)
    out = df.copy()

    if "z_norm_range" in features and "z_norm_range" not in out.columns:
        out["z_norm_range"] = out["z"].to_numpy(dtype=np.float32) / (
            out["range_m"].to_numpy(dtype=np.float32) + 1e-6
        )

    if "frame_doppler_abs" in features and "frame_doppler_abs" not in out.columns:
        out["frame_doppler_abs"] = (
            out.groupby("radar_frame_num")["doppler"].transform(lambda x: x.abs().mean()).astype(np.float32)
        )
    if "frame_doppler_std" in features and "frame_doppler_std" not in out.columns:
        out["frame_doppler_std"] = (
            out.groupby("radar_frame_num")["doppler"].transform("std").fillna(0.0).astype(np.float32)
        )
    if "local_density" in features and "local_density" not in out.columns:
        out = _compute_local_density(out, runtime.config.features.local_density_radius_m)
    if "persist_score" in features and "persist_score" not in out.columns:
        if runtime.config.features.persist_score_source == "session_voxel":
            out = _compute_session_persist_score(out, runtime.config.features.persist_voxel_size_m)
        elif runtime.config.features.persist_score_source == "mapbuilder":
            out = _compute_map_persist_score(runtime, out)
        else:
            raise ValueError(f"Unsupported persist_score_source={runtime.config.features.persist_score_source!r}")

    missing_soft = (features & SOFT_STRUCTURAL_FEATURES) - set(out.columns)
    if missing_soft:
        out = _compute_soft_structural(runtime, out, missing_soft)

    return out


def build_frame_meta(frame_buffer: deque) -> np.ndarray:
    def normalize(vec):
        vec = np.asarray(vec, dtype=np.float32)
        denom = float(np.max(np.abs(vec))) if vec.size else 0.0
        if denom < 1e-6:
            denom = 1.0
        return (vec / denom).astype(np.float32)

    records = list(frame_buffer)
    if not records:
        return np.zeros((0, 5), dtype=np.float32)

    center = records[len(records) // 2]
    center_frame = int(center[2])
    dt_radar = np.array([int(rec[2]) - center_frame for rec in records], dtype=np.float32)
    dt_video = dt_radar.copy()
    mean_d = np.array([float(np.mean(rec[3])) if len(rec[3]) else 0.0 for rec in records], dtype=np.float32)
    abs_mean_d = np.array(
        [float(np.mean(np.abs(rec[3]))) if len(rec[3]) else 0.0 for rec in records],
        dtype=np.float32,
    )
    std_d = np.array([float(np.std(rec[3])) if len(rec[3]) else 0.0 for rec in records], dtype=np.float32)

    return np.stack(
        [
            normalize(dt_video),
            normalize(dt_radar),
            normalize(mean_d),
            normalize(abs_mean_d),
            normalize(std_d),
        ],
        axis=1,
    ).astype(np.float32)


def build_runtime_rd_patch_config(config: RDPatchConfig, *, include_rd_cube: bool = False):
    RuntimeRDPatchConfig = _import_attr(HYBRID_RD_RUNTIME_MODULE, "RuntimeRDPatchConfig")

    return RuntimeRDPatchConfig(
        doppler_bins=config.doppler_bins,
        range_bins=config.range_bins,
        frame_number_offset=config.frame_number_offset,
        sidecar_subdir=config.sidecar_subdir,
        clutter_mode=config.clutter_mode,
        log_scale=not config.linear_power,
        include_rd_cube=include_rd_cube,
    )


def build_runtime_ra_patch_config(contract: CheckpointContract, config: RDPatchConfig):
    RuntimeRAPatchConfig = _import_attr(HYBRID_RD_RUNTIME_MODULE, "RuntimeRAPatchConfig")

    return RuntimeRAPatchConfig(
        azimuth_bins=9,
        range_bins=7,
        frame_number_offset=config.frame_number_offset,
        source=contract.ra_patch_source,
        az_mode=contract.ra_patch_az_mode,
        az_fft_size=contract.ra_patch_az_fft_size,
        log_scale=contract.ra_patch_log_scale,
        normalize_by_frame_max=contract.ra_patch_normalize_by_frame_max,
        row_edge_mode=contract.ra_patch_row_edge_mode,
    )


def inject_directness_features(frame_df, feature_cols: list[str]) -> None:
    needed = _feature_set(feature_cols) & DIRECTNESS_FEATURES
    if not needed:
        return

    directness_mod = importlib.import_module(DIRECTNESS_RUNTIME_MODULE)
    DirectnessConfig = directness_mod.DirectnessConfig
    annotate_directness = directness_mod.annotate_directness
    estimate_ego_velocity_no_class = directness_mod.estimate_ego_velocity_no_class

    cfg = DirectnessConfig()
    points = [
        {
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "doppler": float(row["doppler"]),
            "snr": float(row["snr"]),
        }
        for _, row in frame_df.iterrows()
    ]
    ego = estimate_ego_velocity_no_class(points, cfg)
    annotated = annotate_directness(points, ego, cfg)
    for col in sorted(needed):
        if any(col not in point for point in annotated):
            raise RuntimeError(f"Directness feature {col!r} was not produced")
        frame_df[col] = np.asarray([float(point[col]) for point in annotated], dtype=np.float32)


def verify_frame_features(frame_df, feature_cols: list[str]) -> None:
    missing = [col for col in feature_cols if col not in frame_df.columns]
    if missing:
        raise ValueError(f"Frame missing model feature columns: {missing}")
    _assert_finite(frame_df, set(feature_cols), context="model features")


def run_adc_to_pointcloud(runtime: NavigationRuntime, session_dir: str) -> tuple[bool, int, str]:
    process_session_fast = _import_attr(ADC_TO_POINTCLOUD_MODULE, "process_session_fast")

    log.adc_start(Path(session_dir).name)
    t0 = time.monotonic()
    success, detections, csv_path = process_session_fast(runtime.processor, session_dir)
    elapsed = time.monotonic() - t0
    if not success:
        log.adc_error(Path(session_dir).name, csv_path or "unknown error")
        return False, 0, ""
    log.adc_done(Path(session_dir).name, elapsed, detections, success=True)
    return True, detections, csv_path


def run_inference_on_csv(runtime: NavigationRuntime, csv_path: str, session_dir: str) -> list[dict[str, Any]]:
    import pandas as pd
    import torch
    from collections import Counter

    contract = runtime.model_ctx.contract
    model = runtime.model_ctx.model
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return []

    df = normalize_inference_schema(df, contract.feature_cols)

    if contract.uses_rd_patch:
        _require_columns(df, {"range_bin", "doppler_bin"}, context="RD patch ADC CSV")
        rd_runtime_mod = importlib.import_module(HYBRID_RD_RUNTIME_MODULE)
        export_session_sidecars = rd_runtime_mod.export_session_sidecars
        extract_patches_and_scalars_for_frame_df = rd_runtime_mod.extract_patches_and_scalars_for_frame_df
        extract_patches_for_frame_df = rd_runtime_mod.extract_patches_for_frame_df
        extract_rdra_patches_and_scalars_for_frame_df = rd_runtime_mod.extract_rdra_patches_and_scalars_for_frame_df

        ra_source = contract.ra_patch_source.strip().lower()
        need_rd_cube = bool(
            runtime.config.unet.enabled
            or (contract.uses_ra_patch and ra_source in {"true_ra", "rd_cube", "beamformed"})
        )
        rd_config = build_runtime_rd_patch_config(runtime.config.rd_patch, include_rd_cube=need_rd_cube)
        export_session_sidecars(
            session_dir,
            cfg_path=runtime.config.mmwave_cfg_path,
            config=rd_config,
            overwrite=runtime.config.rd_patch.overwrite_sidecars,
        )
        ra_config = build_runtime_ra_patch_config(contract, runtime.config.rd_patch) if contract.uses_ra_patch else None
    else:
        extract_patches_for_frame_df = None
        extract_patches_and_scalars_for_frame_df = None
        extract_rdra_patches_and_scalars_for_frame_df = None
        rd_config = None
        ra_config = None

    df = df[df["z"] > runtime.config.min_valid_z].reset_index(drop=True)
    df = df[(df["range_m"] >= runtime.config.min_range_m) & (df["range_m"] <= runtime.config.max_range_m)].reset_index(drop=True)
    if len(df) == 0:
        return []

    df = add_computed_features(runtime, df)

    static_runtime_cols = RD_SCALAR_FEATURES | DIRECTNESS_FEATURES
    global_required = [col for col in contract.feature_cols if col not in static_runtime_cols]
    missing_global = [col for col in global_required if col not in df.columns]
    if missing_global:
        raise ValueError(f"Prepared dataframe missing model feature columns: {missing_global}")

    results: list[dict[str, Any]] = []
    frames = sorted(df["radar_frame_num"].unique())
    frame_buffer: deque = deque(maxlen=contract.window_size)
    center_idx = contract.window_size // 2
    last_emitted_frame_num: int | None = None
    all_preds: list[str] = []
    device = next(model.parameters()).device
    session_tensor_cache: dict[Path, object] = {}
    need_rd_scalars = bool(_feature_set(contract.feature_cols) & RD_SCALAR_FEATURES)

    with torch.no_grad():
        for frame_num in frames:
            frame_df = df[df["radar_frame_num"] == frame_num].copy()
            if len(frame_df) == 0:
                continue

            rd_patches = None
            ra_patches = None
            if contract.uses_rd_patch:
                if contract.uses_ra_patch:
                    rd_patches, ra_patches, rd_scalars = extract_rdra_patches_and_scalars_for_frame_df(
                        frame_df,
                        session_dir=session_dir,
                        frame_num=int(frame_num),
                        rd_config=rd_config,
                        ra_config=ra_config,
                        session_tensor_cache=session_tensor_cache,
                    )
                elif need_rd_scalars:
                    rd_patches, rd_scalars = extract_patches_and_scalars_for_frame_df(
                        frame_df,
                        session_dir=session_dir,
                        frame_num=int(frame_num),
                        config=rd_config,
                    )
                else:
                    rd_patches = extract_patches_for_frame_df(
                        frame_df,
                        session_dir=session_dir,
                        frame_num=int(frame_num),
                        config=rd_config,
                    )
                    rd_scalars = None

                if need_rd_scalars:
                    if rd_scalars is None or len(rd_scalars) != len(frame_df):
                        raise RuntimeError("RD scalar extractor did not return one scalar record per point")
                    for col in sorted(RD_SCALAR_FEATURES & _feature_set(contract.feature_cols)):
                        if any(col not in scalar for scalar in rd_scalars):
                            raise RuntimeError(f"RD scalar feature {col!r} was not produced")
                        frame_df[col] = np.asarray(
                            [float(scalar[col]) for scalar in rd_scalars],
                            dtype=np.float32,
                        )

            inject_directness_features(frame_df, contract.feature_cols)
            verify_frame_features(frame_df, contract.feature_cols)

            feats_raw = frame_df[contract.feature_cols].to_numpy(dtype=np.float32)
            feats_norm = (feats_raw - contract.feature_means) / (contract.feature_stds + 1e-6)
            feats_norm = np.nan_to_num(feats_norm, nan=0.0, posinf=0.0, neginf=0.0)
            xyz = frame_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
            doppler = frame_df["doppler"].to_numpy(dtype=np.float32)

            frame_buffer.append((feats_norm, xyz, int(frame_num), doppler, rd_patches, ra_patches, frame_df))
            while len(frame_buffer) < contract.window_size:
                frame_buffer.appendleft(frame_buffer[0])

            feat_list = [
                torch.from_numpy(record[0].T.copy()).unsqueeze(0).to(device)
                for record in frame_buffer
            ]
            xyz_list = [
                torch.from_numpy(record[1].T.copy()).unsqueeze(0).to(device)
                for record in frame_buffer
            ]
            frame_meta = torch.from_numpy(build_frame_meta(frame_buffer)).unsqueeze(0).to(device)

            if contract.uses_rd_patch:
                rd_patch_list = [
                    torch.from_numpy(record[4].copy()).unsqueeze(0).to(device)
                    for record in frame_buffer
                ]
                if contract.uses_ra_patch:
                    ra_patch_list = [
                        torch.from_numpy(record[5].copy()).unsqueeze(0).to(device)
                        for record in frame_buffer
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
            softmax = torch.softmax(logits.squeeze(0), dim=0).detach().cpu().numpy()
            pred_conf = softmax.max(axis=0)

            def class_prob(name: str, *aliases: str) -> np.ndarray:
                for candidate in (name, *aliases):
                    if candidate in contract.bucket_order:
                        return softmax[contract.bucket_order.index(candidate)]
                return np.zeros(softmax.shape[1], dtype=np.float32)

            prob_structure = class_prob("structure")
            prob_floor = class_prob("floor")
            prob_human = class_prob("human_candidate", "human")
            prob_ghost = class_prob("ghost_return")
            p_acc = (
                torch.sigmoid(acc_logits.squeeze(0).squeeze(0)).detach().cpu().numpy()
                if acc_logits is not None
                else None
            )

            center_rec = frame_buffer[center_idx]
            _, _, center_frame_num, _, _, _, center_frame_df = center_rec
            if last_emitted_frame_num is not None and center_frame_num == last_emitted_frame_num:
                continue
            if len(preds) != len(center_frame_df):
                raise RuntimeError(
                    "Prediction count mismatch: "
                    f"preds={len(preds)} frame_points={len(center_frame_df)} "
                    f"center_frame={center_frame_num}"
                )

            pred_labels = [contract.bucket_order[int(preds[i])] for i in range(len(preds))]
            all_preds.extend(pred_labels)
            emitted_feature_cols = tuple(sorted(RD_SCALAR_FEATURES | DIRECTNESS_FEATURES))
            provenance_int_cols = ("radar_frame_num", "frame_num", "range_bin", "doppler_bin")

            for i, (_, row) in enumerate(center_frame_df.iterrows()):
                entry: dict[str, Any] = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                    "pred_class": pred_labels[i],
                    "pred_conf": round(float(pred_conf[i]), 4),
                    "pred_prob_structure": round(float(prob_structure[i]), 4),
                    "pred_prob_floor": round(float(prob_floor[i]), 4),
                    "pred_prob_human": round(float(prob_human[i]), 4),
                    "pred_prob_human_candidate": round(float(prob_human[i]), 4),
                    "pred_prob_ghost_return": round(float(prob_ghost[i]), 4),
                    "doppler": float(row["doppler"]),
                    "snr": float(row["snr"]),
                    "frame": int(center_frame_num),
                }
                if p_acc is not None:
                    acc_prob = float(p_acc[i])
                    entry["p_acc"] = round(acc_prob, 4)
                    entry["acc_confidence"] = round(abs(acc_prob - 0.5) * 2.0, 4)
                for col in provenance_int_cols:
                    if col in center_frame_df.columns:
                        entry[col] = int(row[col])
                for col in emitted_feature_cols:
                    if col in center_frame_df.columns:
                        entry[col] = float(row[col])
                results.append(entry)

            last_emitted_frame_num = center_frame_num

    log.debug("Predictions", total=len(all_preds), **Counter(all_preds))
    return results


class PiperTTS:
    """Compatibility wrapper around the TTS engine. TTS fallback is allowed."""

    def __init__(self, config: NavigationConfig):
        TTSEngine = _import_attr(GUIDANCE_MODULE, "TTSEngine")
        self._tts = TTSEngine(model=str(config.guidance.piper_voice), backend="auto")

    def speak(self, text: str) -> None:
        log.info("[NAV]", guidance=text)
        self._tts.speak_async(text)


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


def llm_worker(config: NavigationConfig, stop_event: threading.Event) -> None:
    log.info("LLM worker starting...")
    llm_engine = None

    try:
        GuidanceEngine = _import_attr(GUIDANCE_MODULE, "GuidanceEngine")
        llm_engine = GuidanceEngine(
            model=config.guidance.llm_model,
            min_interval_s=config.guidance.llm_min_wait_s,
            ollama_host=config.guidance.ollama_host,
            change_threshold=False,
        )
        log.info("LLM guidance engine ready.", model=config.guidance.llm_model, host=config.guidance.ollama_host)
    except Exception as exc:
        log.warning(f"LLM unavailable ({exc}); using rule-based guidance.")

    FastGuidanceEngine = _import_attr(GUIDANCE_MODULE, "FastGuidanceEngine")

    fast_engine = FastGuidanceEngine(min_interval_s=config.guidance.fast_min_wait_s)
    tts = PiperTTS(config)
    tts.speak("Navigation system ready.")

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
        except Exception as exc:
            log.error(f"Guidance error: {exc}")

    log.info("LLM worker stopped.")


def apply_geo_floor_correction(points: list[dict[str, Any]], config: NavigationConfig) -> list[dict[str, Any]]:
    if not config.geo_floor.enabled:
        return points

    corrected: list[dict[str, Any]] = []
    count = 0
    for point in points:
        if (
            point.get("pred_class") == "structure"
            and float(point.get("y", 99.0)) < config.geo_floor.max_y_m
            and float(point.get("z", 0.0)) < config.geo_floor.max_z_m
            and abs(float(point.get("doppler", 0.0))) < config.geo_floor.max_abs_doppler_mps
        ):
            point = dict(point)
            point["pred_class"] = "floor"
            count += 1
        corrected.append(point)

    if count:
        log.debug("Geo floor correction", reclassified=count)
    return corrected


def apply_blocking_override(scene: dict, points: list[dict[str, Any]], config: NavigationConfig) -> None:
    cfg = config.blocking
    struct_pts = [point for point in points if point.get("pred_class") == "structure"]
    center_count = 0
    left_count = 0
    right_count = 0

    if struct_pts:
        arr = np.asarray([[p["x"], p["y"], p["z"]] for p in struct_pts], dtype=np.float32)
        xs, ys, zs = arr[:, 0], arr[:, 1], arr[:, 2]
        in_zone = (ys >= cfg.min_y_m) & (ys < cfg.max_y_m)
        if np.any(in_zone):
            bx = xs[in_zone]
            by = ys[in_zone]
            bz = zs[in_zone]
            az = np.degrees(np.arctan2(np.abs(bx), np.maximum(by, 0.1)))
            above = bz > cfg.min_obstacle_z_m
            center_count = int(np.sum((az <= cfg.center_azimuth_deg) & (np.abs(bx) < cfg.center_half_width_m) & above))
            left_count = int(np.sum((bx <= 0) & (az > cfg.center_azimuth_deg) & (az <= cfg.side_azimuth_deg) & above))
            right_count = int(np.sum((bx > 0) & (az > cfg.center_azimuth_deg) & (az <= cfg.side_azimuth_deg) & above))

    scene.setdefault("blocking", {})
    scene["blocking"]["center_points"] = center_count
    scene["blocking"]["left_points"] = left_count
    scene["blocking"]["right_points"] = right_count
    scene["open_directions"] = {
        "left_clear": left_count < cfg.left_clear_threshold,
        "center_clear": center_count < cfg.center_clear_threshold,
        "right_clear": right_count < cfg.right_clear_threshold,
    }


def annotate_points_for_map(runtime: NavigationRuntime, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if runtime.map_builder is None or runtime.config.map.update_mode == "uniform" or not points:
        return points

    required = {"p_ego"}
    if runtime.config.map.update_mode in {"ego_directness", "learned_acc"}:
        required.add("p_dir_doppler")
    if runtime.config.map.update_mode == "learned_acc":
        required.add("p_acc")

    if all(all(field in point for field in required) for point in points):
        return points

    directness_mod = importlib.import_module(DIRECTNESS_RUNTIME_MODULE)
    DirectnessConfig = directness_mod.DirectnessConfig
    annotate_directness = directness_mod.annotate_directness
    estimate_ego_velocity = directness_mod.estimate_ego_velocity

    cfg = DirectnessConfig()
    ego = estimate_ego_velocity(points, cfg)
    annotated = annotate_directness(points, ego, cfg)

    for point in annotated:
        missing = sorted(col for col in required if col not in point)
        if missing:
            raise RuntimeError(f"MapBuilder update evidence missing: {missing}")
    return annotated


def unet_npz_path_for_session(session_dir: str | Path) -> Path:
    session_path = Path(session_dir)
    return session_path / f"{session_path.name}_radar_tensors.npz"


def load_unet_npz_for_session(session_dir: str | Path, *, csv_path: str | Path | None = None):
    """Load the archived Branch 3 replay tensor bundle for one session."""
    session_path = Path(session_dir)
    npz_path = unet_npz_path_for_session(session_path)
    if not npz_path.exists():
        details = [
            "Branch 3 U-Net tensor file is required but missing.",
            f"session_dir={session_path}",
            f"expected_npz={npz_path}",
        ]
        if csv_path is not None:
            details.append(f"csv_path={csv_path}")
        raise FileNotFoundError("\n".join(details))
    try:
        return np.load(str(npz_path), allow_pickle=False)
    except Exception as exc:
        details = [
            "Failed to load Branch 3 U-Net tensor file.",
            f"session_dir={session_path}",
            f"npz_path={npz_path}",
            f"error={type(exc).__name__}: {exc}",
        ]
        if csv_path is not None:
            details.insert(2, f"csv_path={csv_path}")
        raise RuntimeError("\n".join(details)) from exc


def _point_key_sample(points: list[dict[str, Any]], limit: int = 3) -> list[list[str]]:
    return [sorted(str(key) for key in point.keys()) for point in points[:limit]]


def _select_unet_frame(points: list[dict[str, Any]]) -> int:
    frames: list[int] = []
    for point in points:
        for key in ("frame", "frame_num", "radar_frame_num"):
            if key in point:
                frames.append(int(point[key]))
                break
    if not frames:
        raise RuntimeError(
            "Cannot select a Branch 3 U-Net frame; classified points do not carry frame metadata.\n"
            f"point_count={len(points)}\n"
            f"point_key_sample={_point_key_sample(points)}"
        )
    return max(frames)


def _load_unet_rd_cube(
    runtime: NavigationRuntime,
    frame_num: int,
    *,
    npz_data: Any | None = None,
    session_dir: str | Path | None = None,
) -> tuple[np.ndarray, str]:
    if npz_data is not None:
        rd_key = f"rd_{int(frame_num)}"
        if rd_key not in npz_data:
            available = list(npz_data.files) if hasattr(npz_data, "files") else []
            raise KeyError(
                "Branch 3 U-Net RD tensor is missing for frame.\n"
                f"frame_num={int(frame_num)}\n"
                f"required_key={rd_key}\n"
                f"available_key_count={len(available)}\n"
                f"available_key_sample={available[:8]}"
            )
        return np.asarray(npz_data[rd_key], dtype=np.complex64), "session_npz"

    if session_dir is None:
        raise RuntimeError(f"Branch 3 U-Net needs session_dir or npz_data for frame {int(frame_num)}")

    load_frame_rd_cube = _import_attr(HYBRID_RD_RUNTIME_MODULE, "load_frame_rd_cube")
    rd_config = build_runtime_rd_patch_config(runtime.config.rd_patch, include_rd_cube=True)
    ra_config = build_runtime_ra_patch_config(runtime.model_ctx.contract, runtime.config.rd_patch)
    try:
        rd_cube = load_frame_rd_cube(
            session_dir,
            int(frame_num),
            config=ra_config,
            rd_config=rd_config,
        )
        return np.asarray(rd_cube, dtype=np.complex64), "live_sidecar"
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Branch 3 U-Net RD cube for live navigation.\n"
            f"session_dir={session_dir}\n"
            f"frame_num={int(frame_num)}\n"
            f"rd_sidecar_subdir={runtime.config.rd_patch.sidecar_subdir}\n"
            f"frame_number_offset={runtime.config.rd_patch.frame_number_offset}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc


def _unet_prior(runtime: NavigationRuntime) -> np.ndarray | None:
    if not runtime.config.unet.use_map_prior:
        return None
    if runtime.map_builder is None:
        raise RuntimeError("unet.use_map_prior=True requires a running MapBuilder")
    return runtime.map_builder.get_polar_prior(shape=(32, 256))


def run_unet_freespace(
    runtime: NavigationRuntime,
    frame_num: int,
    *,
    npz_data: Any | None = None,
    session_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the canonical Branch 3 U-Net path for one radar frame."""
    if not runtime.config.unet.enabled:
        raise RuntimeError("Branch 3 U-Net inference was requested but unet.enabled=False")
    if runtime.unet_ctx is None:
        raise RuntimeError("Branch 3 U-Net inference was requested before the U-Net context was loaded")

    import torch

    rd_cube, rd_source = _load_unet_rd_cube(
        runtime,
        int(frame_num),
        npz_data=npz_data,
        session_dir=session_dir,
    )
    cfg = runtime.config.unet
    try:
        inp = runtime.unet_ctx.module.rd_cube_to_input(
            rd_cube,
            prior=_unet_prior(runtime),
            use_prior=cfg.use_map_prior,
        ).to(runtime.unet_ctx.device)
        with torch.no_grad():
            prob_map = runtime.unet_ctx.model(inp).squeeze(0).detach().cpu().numpy()
    except Exception as exc:
        raise RuntimeError(
            "Branch 3 U-Net inference failed.\n"
            f"frame_num={int(frame_num)}\n"
            f"rd_source={rd_source}\n"
            f"unet_checkpoint={runtime.unet_ctx.checkpoint}\n"
            f"device={runtime.unet_ctx.device}\n"
            f"use_map_prior={cfg.use_map_prior}\n"
            f"error={type(exc).__name__}: {exc}"
        ) from exc

    expected_shape = tuple(int(v) for v in cfg.expected_output_shape)
    if tuple(prob_map.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected Branch 3 U-Net output shape.\n"
            f"frame_num={int(frame_num)}\n"
            f"actual_shape={tuple(prob_map.shape)}\n"
            f"expected_shape={expected_shape}\n"
            f"unet_checkpoint={runtime.unet_ctx.checkpoint}"
        )

    r0, r1 = cfg.near_range_bins
    prob_near = prob_map[int(r0):int(r1), :]
    sectors = runtime.unet_ctx.module.sector_probabilities(prob_near, axis=cfg.sector_axis)
    values = {
        "left": float(sectors.get("left", 0.0)),
        "center": float(sectors.get("center", 0.0)),
        "right": float(sectors.get("right", 0.0)),
    }
    bad = {key: value for key, value in values.items() if not np.isfinite(value)}
    if bad:
        raise RuntimeError(
            "Branch 3 U-Net produced non-finite sector probability values.\n"
            f"frame_num={int(frame_num)}\n"
            f"sector_values={values}\n"
            f"unet_checkpoint={runtime.unet_ctx.checkpoint}"
        )

    threshold = float(cfg.free_threshold)
    return {
        "available": True,
        "center": values["center"],
        "left": values["left"],
        "right": values["right"],
        "center_free": values["center"] >= threshold,
        "left_free": values["left"] >= threshold,
        "right_free": values["right"] >= threshold,
        "free_threshold": threshold,
        "free_space_confidence": "high",
        "source": "navigation_loop_unet",
        "rd_source": rd_source,
        "frame": int(frame_num),
        "sector_axis": cfg.sector_axis,
        "range_bins": (int(r0), int(r1)),
        "use_map_prior": bool(cfg.use_map_prior),
    }


def attach_unet_freespace(
    runtime: NavigationRuntime,
    scene: dict[str, Any],
    frame_num: int,
    *,
    npz_data: Any | None = None,
    session_dir: str | Path | None = None,
) -> None:
    if runtime.config.unet.enabled:
        scene["unet_freespace"] = run_unet_freespace(
            runtime,
            int(frame_num),
            npz_data=npz_data,
            session_dir=session_dir,
        )


def assemble_navigation_scene(
    runtime: NavigationRuntime,
    points: list[dict[str, Any]],
    *,
    window_frames: int | None = None,
    ego_velocity_mps: dict[str, Any] | None = None,
    update_map: bool = True,
    join_map_update: bool = False,
    compute_decision: bool = True,
) -> dict[str, Any]:
    aggregate_scene = _import_attr(SCENE_PIPELINE_MODULE, "aggregate_scene")
    scene_window = int(window_frames if window_frames is not None else runtime.config.scene_window)

    if update_map and runtime.map_builder is not None and points:
        runtime.map_builder.push_classified_points(points)
        if join_map_update:
            map_queue = getattr(runtime.map_builder, "_queue", None)
            if map_queue is not None:
                map_queue.join()

    kwargs: dict[str, Any] = {"window_frames": scene_window}
    if ego_velocity_mps is not None:
        kwargs["ego_velocity_mps"] = ego_velocity_mps
    scene = aggregate_scene(points, **kwargs)
    apply_blocking_override(scene, points, runtime.config)
    if runtime.map_builder is not None:
        scene["map_anomalies"] = runtime.map_builder.query_anomalies(0.0, 1.0, radius_m=0.5)
    if compute_decision:
        scene["decision"] = compute_nav_decision(scene)
    return scene


def delete_session(session_dir: str) -> None:
    try:
        shutil.rmtree(session_dir)
        log.delete_done(Path(session_dir).name)
    except Exception as exc:
        log.delete_error(Path(session_dir).name, str(exc))


def process_session(runtime: NavigationRuntime, session_dir: str) -> bool:
    success, _, csv_path = run_adc_to_pointcloud(runtime, session_dir)
    if not success:
        return False

    t0 = time.monotonic()
    points = run_inference_on_csv(runtime, csv_path, session_dir=session_dir)
    points = apply_geo_floor_correction(points, runtime.config)
    points = annotate_points_for_map(runtime, points)
    elapsed = time.monotonic() - t0
    log.debug("Inference done", points=len(points), elapsed=f"{elapsed:.2f}s")

    scene = assemble_navigation_scene(runtime, points, compute_decision=False)
    if points:
        attach_unet_freespace(
            runtime,
            scene,
            _select_unet_frame(points),
            session_dir=session_dir,
        )
    scene["decision"] = compute_nav_decision(scene)

    od = scene.get("open_directions", {})
    log.debug(
        "Scene decision",
        action=scene["decision"].get("primary_action"),
        reason=scene["decision"].get("reason"),
        center_clear=od.get("center_clear"),
        left_clear=od.get("left_clear"),
        right_clear=od.get("right_clear"),
    )
    publish_scene(scene)
    return True


def processing_loop(runtime: NavigationRuntime, ready_queue: queue.Queue, stop_event: threading.Event) -> None:
    sessions_ok = 0
    sessions_failed = 0
    log.info("Processing loop ready. Waiting for sessions...")

    while not (stop_event.is_set() and ready_queue.empty()):
        qsize = ready_queue.qsize()
        if qsize >= runtime.config.max_queue_warn:
            log.queue_warning(qsize)

        try:
            session_dir = ready_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            log.info("Processing session", session=Path(session_dir).name)
            if process_session(runtime, session_dir):
                sessions_ok += 1
            else:
                sessions_failed += 1
        except Exception as exc:
            sessions_failed += 1
            log.error(f"Session failed: {exc}")
        finally:
            delete_session(session_dir)
            ready_queue.task_done()
            log.loop_summary(sessions_ok, sessions_failed)

    log.info("Processing loop stopped.", ok=sessions_ok, failed=sessions_failed)


def main(config: NavigationConfig = DEFAULT_NAVIGATION_CONFIG) -> None:
    runtime = build_runtime(config)

    log.info("=" * 55)
    log.info("Radar Navigation Loop - canonical strict profile")
    for key, value in config_summary(config).items():
        log.info(f"{key:20s}: {value}")
    log.info(f"{'log_file':20s}: {get_log_path()}")
    log.info("=" * 55)

    config.output_root.mkdir(parents=True, exist_ok=True)
    ready_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def handle_stop(sig, frame):
        if not stop_event.is_set():
            log.shutdown("Ctrl+C / SIGTERM")
            stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    llm_thread = threading.Thread(
        target=llm_worker,
        args=(config, stop_event),
        name="LLMWorker",
        daemon=True,
    )
    llm_thread.start()

    recorder_mod = importlib.import_module(RADAR_RECORDER_MODULE)
    recorder_mod.OUTPUT_ROOT = str(config.output_root)
    recorder_mod.MMWAVE_CFG_PATH = str(config.mmwave_cfg_path)
    recorder_mod.SESSION_DURATION_S = float(config.session_duration_s)
    record_continuous = recorder_mod.record_continuous

    recorder_thread = threading.Thread(
        target=record_continuous,
        args=(ready_queue, stop_event),
        name="RecorderThread",
        daemon=True,
    )
    recorder_thread.start()

    try:
        processing_loop(runtime, ready_queue, stop_event)
    finally:
        stop_event.set()
        recorder_thread.join(timeout=15.0)
        llm_thread.join(timeout=10.0)
        if runtime.map_builder is not None:
            runtime.map_builder.stop()

        while not ready_queue.empty():
            try:
                delete_session(ready_queue.get_nowait())
            except queue.Empty:
                break

        for path in config.output_root.glob("session_*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

        log.info("Navigation loop exited cleanly.")


def _self_test() -> None:
    aggregate_scene = _import_attr(SCENE_PIPELINE_MODULE, "aggregate_scene")

    print("--- navigation_loop --self-test ---")
    config = DEFAULT_NAVIGATION_CONFIG

    raw_points = [
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.02, "doppler": 0.05, "snr": 10.0},
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.30, "doppler": 0.05, "snr": 10.0},
        {"pred_class": "structure", "x": 0.0, "y": 1.0, "z": 0.02, "doppler": 0.50, "snr": 10.0},
        {"pred_class": "structure", "x": 0.0, "y": 3.0, "z": 0.02, "doppler": 0.05, "snr": 10.0},
        {"pred_class": "floor", "x": 0.0, "y": 1.5, "z": 0.01, "doppler": 0.00, "snr": 8.0},
    ]
    corrected = apply_geo_floor_correction(raw_points, config)
    assert sum(1 for point in corrected if point["pred_class"] == "floor") == 2
    assert corrected[0]["pred_class"] == "floor"
    assert corrected[1]["pred_class"] == "structure"
    print("PASS: geo floor correction")

    struct_points = [
        {"x": 0.1, "y": 1.0, "z": 0.3, "pred_class": "structure", "doppler": 0.0, "snr": 8.0},
        {"x": 0.0, "y": 0.8, "z": 0.4, "pred_class": "structure", "doppler": 0.0, "snr": 8.0},
        {"x": -0.8, "y": 1.0, "z": 0.3, "pred_class": "structure", "doppler": 0.0, "snr": 8.0},
        {"x": 0.9, "y": 1.0, "z": 0.3, "pred_class": "structure", "doppler": 0.0, "snr": 8.0},
        {"x": 0.1, "y": 1.0, "z": 0.01, "pred_class": "structure", "doppler": 0.0, "snr": 8.0},
    ]
    scene = aggregate_scene(struct_points, window_frames=config.scene_window)
    apply_blocking_override(scene, struct_points, config)
    assert scene["blocking"]["center_points"] == 2
    assert scene["blocking"]["left_points"] == 1
    assert scene["blocking"]["right_points"] == 1
    scene["decision"] = compute_nav_decision(scene)
    assert "primary_action" in scene["decision"]
    print("PASS: blocking override + nav decision")

    print("All --self-test assertions passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        main()
