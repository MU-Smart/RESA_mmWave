"""Canonical configuration for the live navigation loop.

The loop imports one frozen configuration object and does not read environment
variables. Edit this file when the deployed model or radar profile changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from config.config import (
    CFG_PATH_DEFAULT,
    LLM_DIR_DEFAULT,
    ML_MODEL_DIR_DEFAULT,
    NAV_MODEL_PT_DEFAULT,
    NAV_OUTPUT_ROOT_DEFAULT,
    PIPER_VOICE_DEFAULT,
    REPO_ROOT,
    UNET_CHECKPOINT_DEFAULT,
    install_import_paths as install_repo_import_paths,
)


PerceptionMode = Literal["pointcloud", "pointcloud_rd_patch"]
PersistScoreSource = Literal["session_voxel", "mapbuilder"]


@dataclass(frozen=True)
class ADCConfig:
    pfa: float = 1e-2
    min_snr: float = 3.0
    min_range_bin: int = 18
    clutter_mode: str = "off"
    range_window: str = "hann"
    doppler_window: str = "hann"
    enable_aoa: bool = True
    max_range_m: float | None = None
    max_abs_azimuth_deg: float | None = None


@dataclass(frozen=True)
class RDPatchConfig:
    sidecar_subdir: str = "hybrid_rd"
    doppler_bins: int = 17
    range_bins: int = 7
    frame_number_offset: int = 1
    clutter_mode: str = "off"
    linear_power: bool = False
    overwrite_sidecars: bool = False


@dataclass(frozen=True)
class MapConfig:
    enabled: bool = True
    cell_m: float = 0.20
    range_m: float = 10.0
    decay: float = 0.97
    session_s: float = 1.5
    score_radius_m: float = 0.30
    update_mode: str = "uniform"


@dataclass(frozen=True)
class FeatureConfig:
    persist_score_source: PersistScoreSource = "session_voxel"
    local_density_radius_m: float = 0.50
    persist_voxel_size_m: float = 0.15
    soft_ransac_iters: int = 150


@dataclass(frozen=True)
class GeoFloorCorrectionConfig:
    enabled: bool = True
    max_y_m: float = 2.5
    max_z_m: float = 0.05
    max_abs_doppler_mps: float = 0.2


@dataclass(frozen=True)
class BlockingConfig:
    min_y_m: float = 0.5
    max_y_m: float = 2.5
    min_obstacle_z_m: float = 0.05
    center_azimuth_deg: float = 20.0
    side_azimuth_deg: float = 60.0
    center_half_width_m: float = 0.5
    left_clear_threshold: int = 20
    center_clear_threshold: int = 15
    right_clear_threshold: int = 20


@dataclass(frozen=True)
class GuidanceConfig:
    llm_model: str = "llama3.2:3b"
    llm_min_wait_s: float = 8.0
    fast_min_wait_s: float = 3.0
    ollama_host: str = "http://127.0.0.1:11435"
    piper_voice: Path = PIPER_VOICE_DEFAULT


@dataclass(frozen=True)
class UNetConfig:
    enabled: bool = True
    checkpoint: Path = UNET_CHECKPOINT_DEFAULT
    sector_axis: str = "columns"
    expected_output_shape: tuple[int, int] = (128, 128)
    near_range_bins: tuple[int, int] = (0, 64)
    free_threshold: float = 0.5
    use_map_prior: bool = False


@dataclass(frozen=True)
class NavigationConfig:
    output_root: Path = NAV_OUTPUT_ROOT_DEFAULT
    mmwave_cfg_path: Path = CFG_PATH_DEFAULT
    model_pt: Path = NAV_MODEL_PT_DEFAULT
    perception_mode: PerceptionMode = "pointcloud_rd_patch"
    session_duration_s: float = 1.5
    ml_model_dir: Path = ML_MODEL_DIR_DEFAULT
    llm_dir: Path = LLM_DIR_DEFAULT
    scene_window: int = 3
    max_queue_warn: int = 3
    min_valid_z: float = -0.851
    min_range_m: float = 0.3
    max_range_m: float = 5.0
    adc: ADCConfig = ADCConfig()
    rd_patch: RDPatchConfig = RDPatchConfig()
    features: FeatureConfig = FeatureConfig()
    map: MapConfig = MapConfig()
    geo_floor: GeoFloorCorrectionConfig = GeoFloorCorrectionConfig()
    blocking: BlockingConfig = BlockingConfig()
    unet: UNetConfig = UNetConfig()
    guidance: GuidanceConfig = GuidanceConfig()


DEFAULT_NAVIGATION_CONFIG = NavigationConfig()


def branch3_replay_config(base: NavigationConfig = DEFAULT_NAVIGATION_CONFIG) -> NavigationConfig:
    """Return the canonical replay profile: Branch 1 RD-patch + Branch 3 U-Net."""
    return replace(
        base,
        perception_mode="pointcloud_rd_patch",
        unet=replace(base.unet, enabled=True),
    )


BRANCH3_REPLAY_CONFIG = branch3_replay_config()


def install_import_paths(config: NavigationConfig) -> None:
    """Install the canonical local import roots for the live loop."""
    install_repo_import_paths(
        ml_model_dir=config.ml_model_dir,
        llm_dir=config.llm_dir,
    )


def validate_static_config(config: NavigationConfig) -> None:
    """Validate static paths and config values before runtime construction."""
    if config.perception_mode not in {"pointcloud", "pointcloud_rd_patch"}:
        raise ValueError(f"Unsupported perception_mode={config.perception_mode!r}")
    if config.unet.enabled and config.perception_mode != "pointcloud_rd_patch":
        raise ValueError("Branch 3 U-Net requires perception_mode='pointcloud_rd_patch'")
    if not config.mmwave_cfg_path.exists():
        raise FileNotFoundError(f"mmWave cfg not found: {config.mmwave_cfg_path}")
    if not config.model_pt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {config.model_pt}")
    if config.unet.enabled and not config.unet.checkpoint.exists():
        raise FileNotFoundError(f"Branch 3 U-Net checkpoint not found: {config.unet.checkpoint}")
    if config.min_range_m >= config.max_range_m:
        raise ValueError("min_range_m must be less than max_range_m")
    if config.scene_window <= 0:
        raise ValueError("scene_window must be positive")
    if config.session_duration_s <= 0.0:
        raise ValueError("session_duration_s must be positive")
    if config.features.persist_score_source == "mapbuilder" and not config.map.enabled:
        raise ValueError("persist_score_source='mapbuilder' requires MapBuilder to be enabled")
    if config.unet.enabled:
        if config.unet.sector_axis not in {"columns", "rows", "global"}:
            raise ValueError(f"Unsupported unet.sector_axis={config.unet.sector_axis!r}")
        out_h, out_w = config.unet.expected_output_shape
        if out_h <= 0 or out_w <= 0:
            raise ValueError("unet.expected_output_shape must contain positive dimensions")
        r0, r1 = config.unet.near_range_bins
        if r0 < 0 or r1 <= r0 or r1 > out_h:
            raise ValueError(
                "unet.near_range_bins must be an increasing range within "
                f"expected output height {out_h}; got {config.unet.near_range_bins}"
            )
        if not (0.0 <= config.unet.free_threshold <= 1.0):
            raise ValueError("unet.free_threshold must be between 0.0 and 1.0")


def validate_branch3_replay_config(config: NavigationConfig = BRANCH3_REPLAY_CONFIG) -> None:
    """Validate the canonical replay profile and report actionable debug context."""
    errors: list[str] = []
    if config.perception_mode != "pointcloud_rd_patch":
        errors.append(f"perception_mode={config.perception_mode!r}; expected 'pointcloud_rd_patch'")
    if not config.unet.enabled:
        errors.append("unet.enabled=False; replay requires Branch 3 U-Net")
    try:
        validate_static_config(config)
    except Exception as exc:
        errors.append(f"static config validation failed: {type(exc).__name__}: {exc}")
    if errors:
        details = [
            "Replay navigation configuration is invalid.",
            "The simulator does not accept CLI overrides for navigation behavior.",
            "Update branch2/navigation_config.py or config/config.py, then rerun.",
            "",
            f"repo_root={REPO_ROOT}",
            f"perception_mode={config.perception_mode}",
            f"mmwave_cfg_path={config.mmwave_cfg_path}",
            f"model_pt={config.model_pt}",
            f"unet_enabled={config.unet.enabled}",
            f"unet_checkpoint={config.unet.checkpoint}",
            f"unet_sector_axis={config.unet.sector_axis}",
            f"unet_expected_output_shape={config.unet.expected_output_shape}",
            f"unet_near_range_bins={config.unet.near_range_bins}",
            f"map_update_mode={config.map.update_mode}",
            "",
            "Errors:",
            *[f"- {err}" for err in errors],
        ]
        raise RuntimeError("\n".join(details))


def config_summary(config: NavigationConfig) -> dict[str, object]:
    """Return stable startup fields for logging."""
    return {
        "perception": config.perception_mode,
        "output_root": str(config.output_root),
        "cfg": str(config.mmwave_cfg_path),
        "model": str(config.model_pt),
        "session_duration_s": config.session_duration_s,
        "range_gate_m": f"{config.min_range_m}-{config.max_range_m}",
        "map_enabled": config.map.enabled,
        "map_update_mode": config.map.update_mode,
        "persist_score_source": config.features.persist_score_source,
        "unet_enabled": config.unet.enabled,
        "unet_checkpoint": str(config.unet.checkpoint),
        "unet_sector_axis": config.unet.sector_axis,
        "unet_near_range_bins": str(config.unet.near_range_bins),
        "llm_model": config.guidance.llm_model,
        "llm_host": config.guidance.ollama_host,
    }
