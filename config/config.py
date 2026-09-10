"""
Shared path configuration.

Usage:
    from config.config import REPO_ROOT, CFG_PATH_DEFAULT, DATA_ROOT_DEFAULT, load_mod
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ── Repo root ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

BRANCH1_DIR = REPO_ROOT / "branch1"
BRANCH2_DIR = REPO_ROOT / "branch2"
BRANCH3_DIR = REPO_ROOT / "branch3"
BRANCH1_PROCESSING_DIR = BRANCH1_DIR / "processing"
BRANCH1_CALIB_DIR = BRANCH1_DIR / "calib"
BRANCH1_GEOMETRY_DIR = BRANCH1_DIR / "inputs" / "geometry"
BRANCH1_RECORDERS_DIR = BRANCH1_DIR / "inputs" / "recorders"
REPO_MODELS_DIR = REPO_ROOT / "models"
ML_MODEL_DIR_DEFAULT = Path("/home/ryan/xwr/ml_model")
LLM_DIR_DEFAULT = Path("/home/ryan/xwr/llm")
NAV_OUTPUT_ROOT_DEFAULT = Path("/home/ryan/nav_sessions")
NAV_MODEL_PT_DEFAULT = Path("/home/ryan/xwr/ml_model/models/3branch_best_model.pt")
UNET_CHECKPOINT_DEFAULT = REPO_MODELS_DIR / "unet_best_model.pt"
PIPER_VOICE_DEFAULT = Path("/home/ryan/piper_voices/en_US-lessac-medium.onnx")


def install_import_paths(
    *,
    ml_model_dir: Path = ML_MODEL_DIR_DEFAULT,
    llm_dir: Path = LLM_DIR_DEFAULT,
) -> None:
    """Install canonical local import roots."""
    paths = (
        REPO_ROOT,
        BRANCH1_DIR,
        BRANCH2_DIR,
        BRANCH3_DIR,
        BRANCH1_PROCESSING_DIR,
        BRANCH1_CALIB_DIR,
        BRANCH1_GEOMETRY_DIR,
        BRANCH1_RECORDERS_DIR,
        REPO_MODELS_DIR,
        ml_model_dir,
        llm_dir,
    )
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


install_import_paths()


# ── Model sub-package directories (for reference) ──────────
MODELS_ROOT = BRANCH1_DIR / "models"

MODEL_DIRS = {
    "hybrid_rd": MODELS_ROOT / "hybrid_rd",
    "3dgcnn":    MODELS_ROOT / "3dgcnn",
    "kpconv":    MODELS_ROOT / "kpconv",
}


# Canonical module names. Use these instead of legacy bare imports that rely on
# branch directories being inserted directly into sys.path.
SCENE_PIPELINE_MODULE = "branch2.scene_pipeline"
DIRECTNESS_RUNTIME_MODULE = "branch2.directness_runtime"
GUIDANCE_MODULE = "branch3.guidance"
BRANCH3_UNET_MODULE = "branch3.branch3_unet"
BRANCH3_LABEL_BEV_MODULE = "branch3.branch3_label_bev"
BRANCH3_TRAIN_UNET_MODULE = "branch3.train_unet"
ADC_TO_POINTCLOUD_MODULE = "branch1.processing.adc_to_pointcloud_v6"
RADAR_RECORDER_MODULE = "branch1.inputs.recorders.radar_only_recorder"
CORRIDOR_SOFT_STRUCTURAL_MODULE = "branch1.inputs.geometry.corridor_soft_structural"
HYBRID_RD_RUNTIME_MODULE = "branch1.models.hybrid_rd.hybrid_rd_runtime"
HYBRID_RD_MODEL_MODULE = "branch1.models.hybrid_rd.hybrid_rd_model"
KPCONV_MODEL_MODULE = "branch1.models.3dgcnn.radar_3dgcnn_common_kpconv"
DGCNN_MODEL_MODULE = "branch1.models.3dgcnn.radar_3dgcnn_common_doorwall"


# ── Data / config file defaults ────────────────────────────
CFG_PATH_DEFAULT       = REPO_ROOT / "config" / "profile_objdet.cfg"
EXTRINSICS_PATH        = REPO_ROOT / "config" / "radar_camera_extrinsics.json"
INTRINSICS_PATH        = REPO_ROOT / "config" / "camera_intrinsics_d435i.json"
DATA_ROOT_DEFAULT      = REPO_ROOT / "LLM_ML" / "data" / "data5-11"
REPLAY_DATA_DIR_DEFAULT = REPO_ROOT / "LLM_ML" / "data" / "jetson_pull_2026-04-22"
LABEL_CSV_NAME_DEFAULT = "labeled_radar_points_v4.csv"


# ── Dynamic module loading (no __init__.py required) ───────
def load_mod(name: str, rel: str):
    """Load a module directly from a relative path using importlib.util.

    Args:
        name: Module name to assign (used for identification)
        rel:  Relative path from repo root (e.g. "branch1/models/3dgcnn/file.py")

    Returns:
        The loaded module object.

    Example:
        from config.config import REPO_ROOT, load_mod

        kpconv_mod = load_mod("kpconv", "branch1/models/3dgcnn/radar_3dgcnn_common_kpconv.py")
        KPConvTemporalSegmenter = kpconv_mod.KPConvTemporalSegmenter
    """
    p = REPO_ROOT / rel
    s = importlib.util.spec_from_file_location(name, p)
    if s is None or s.loader is None:
        raise ImportError(f"Cannot create spec for {p}")
    m = importlib.util.module_from_spec(s)
    # Must be registered before exec: dataclasses under `from __future__ import
    # annotations` resolve field types via sys.modules[cls.__module__], which
    # is this module's own name. Without this, exec_module() raises inside any
    # @dataclass in the loaded file ("'NoneType' object has no attribute
    # '__dict__'").
    sys.modules[name] = m
    try:
        s.loader.exec_module(m)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return m
