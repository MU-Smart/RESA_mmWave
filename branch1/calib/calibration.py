"""Shared radar-to-camera calibration loaders and projection utilities.

Intrinsics priority (highest to lowest):
  1. session meta_data.json  → realsense_calibration.rgb
  2. repo camera_intrinsics.npz  (CALIBRATION/…/Spatial_Calibration/)
  3. config/camera_intrinsics_d435i.json  (values from the same npz, always present)

Extrinsics source:
  config/radar_camera_extrinsics.json  (extracted from RADAR_CAMERA_EXTRINSICS_HANDOFF.md)
  or any path supplied by the caller.

Processing order required by the handoff doc:
  1. decode radar data → radar-frame 3D points
  2. apply R_radar_to_camera, t_radar_to_camera          ← this module
  3. project with camera intrinsics K / dist              ← this module
  4. run OneFormer segmentation in camera image domain
  5. transfer labels back to radar points if needed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_THIS_DIR = Path(__file__).parent
_CONFIG_DIR = _THIS_DIR.parent / "config"

# Canonical assets shipped with the repo.
DEFAULT_EXTRINSICS_JSON: Path = _CONFIG_DIR / "radar_camera_extrinsics.json"
_DEFAULT_D435I_JSON: Path = _CONFIG_DIR / "camera_intrinsics_d435i.json"

# Measured .npz in the CALIBRATION tree (may not exist in all checkouts).
_REPO_ROOT = _THIS_DIR.parent.parent
_REPO_NPZ: Path = (
    _REPO_ROOT / "CALIBRATION/Preprocessing Scripts/archive/Spatial_Calibration/camera_intrinsics.npz"
)


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def load_extrinsics(path: Path) -> Optional[dict]:
    """Load R and t from a radar-camera extrinsics JSON.

    Returns {"R": (3,3) float64, "t": (3,1) float64, "source": str}
    or None on any failure.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        R = np.asarray(data["R_radar_to_camera"], dtype=np.float64)
        t = np.asarray(data["t_radar_to_camera"], dtype=np.float64).reshape(3, 1)
        if R.shape != (3, 3):
            raise ValueError(f"R shape {R.shape} != (3, 3)")
        return {"R": R, "t": t, "source": str(path)}
    except Exception as exc:
        print(f"[calibration] Could not load extrinsics from {path}: {exc}", flush=True)
        return None


def load_intrinsics_npz(path: Path) -> Optional[dict]:
    """Load K and dist from a camera_intrinsics.npz file.

    Returns {"K": (3,3) float64, "dist": (n,) float64, "source": str}
    or None on any failure.
    """
    try:
        npz = np.load(str(path))
        K = np.asarray(npz["K"], dtype=np.float64)
        dist = np.asarray(npz["dist"], dtype=np.float64).reshape(-1)
        if K.shape != (3, 3):
            raise ValueError(f"K shape {K.shape} != (3, 3)")
        return {"K": K, "dist": dist, "source": str(path)}
    except Exception as exc:
        print(f"[calibration] Could not load intrinsics from {path}: {exc}", flush=True)
        return None


def load_intrinsics_from_meta(meta_path: Path) -> Optional[dict]:
    """Load K and distortion from a session meta_data.json.

    Accepts either a 3×3 matrix under "K" or scalar keys fx/fy/cx/cy.
    Returns {"K": (3,3) float64, "dist": (n,) float64, "source": str}
    or None if the calibration block is absent or malformed.
    """
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rgb = meta.get("realsense_calibration", {}).get("rgb", {})
        if "K" in rgb:
            K = np.asarray(rgb["K"], dtype=np.float64)
        else:
            fx = rgb.get("fx")
            fy = rgb.get("fy")
            cx = rgb.get("cx")
            cy = rgb.get("cy")
            if None in (fx, fy, cx, cy):
                return None
            K = np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        dist = np.asarray(
            rgb.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64
        ).reshape(-1)
        if K.shape != (3, 3):
            raise ValueError(f"K shape {K.shape} != (3, 3)")
        return {"K": K, "dist": dist, "source": f"{meta_path} [session meta]"}
    except Exception as exc:
        print(f"[calibration] Could not load intrinsics from {meta_path}: {exc}", flush=True)
        return None


def get_default_d435i_intrinsics() -> dict:
    """Load D435i intrinsics from config/camera_intrinsics_d435i.json.

    These values were sourced from the measured .npz in the CALIBRATION tree.
    Always succeeds (falls back to hardcoded constants if the JSON is missing).

    Returns {"K": (3,3) float64, "dist": (n,) float64, "source": str}
    """
    try:
        data = json.loads(_DEFAULT_D435I_JSON.read_text(encoding="utf-8"))
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        return {"K": K, "dist": dist, "source": str(_DEFAULT_D435I_JSON)}
    except Exception:
        # Last-resort hardcoded constants (approximate; prefer the JSON).
        K = np.array(
            [[1362.27, 0.0, 743.69], [0.0, 1359.20, 230.51], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist = np.array(
            [-0.5132, 6.0888, -0.00369, 0.0161, -35.097], dtype=np.float64
        )
        return {"K": K, "dist": dist, "source": "hardcoded D435i fallback"}


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_calibration(
    session_dir: Path,
    extrinsics_json: Optional[Path] = None,
) -> dict:
    """Load extrinsics + intrinsics for a session, applying the priority chain.

    Extrinsics: extrinsics_json arg → DEFAULT_EXTRINSICS_JSON.
    Intrinsics:  session meta_data.json → repo .npz → D435i JSON fallback.

    Raises FileNotFoundError if no extrinsics source can be found
    (the handoff doc forbids skipping this step).

    Returns {
        "R":      (3,3) float64,
        "t":      (3,1) float64,
        "K":      (3,3) float64,
        "dist":   (n,)  float64,
        "extrinsics_source": str,
        "intrinsics_source": str,
    }
    """
    # --- extrinsics (required) ---
    ext_path = extrinsics_json if extrinsics_json is not None else DEFAULT_EXTRINSICS_JSON
    ext = load_extrinsics(ext_path)
    if ext is None:
        raise FileNotFoundError(
            f"Radar-camera extrinsics not found at {ext_path}. "
            "Cannot project radar points into the camera frame without them."
        )

    # --- intrinsics (priority chain) ---
    # Note: the repo .npz is intentionally NOT used here because it stores
    # values at the calibration resolution (1280x720) without a scale factor.
    # dense_ra_video_simulation.py handles that scaling explicitly.
    # load_calibration always resolves to a resolution-correct K.
    intr: Optional[dict] = None

    meta_path = session_dir / "meta_data.json"
    if meta_path.exists():
        intr = load_intrinsics_from_meta(meta_path)

    if intr is None:
        intr = get_default_d435i_intrinsics()

    return {
        "R": ext["R"],
        "t": ext["t"],
        "K": intr["K"],
        "dist": intr["dist"],
        "extrinsics_source": ext["source"],
        "intrinsics_source": intr["source"],
    }


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_radar_to_image(
    pts_radar: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    img_shape: tuple[int, int],
    *,
    use_distortion: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project radar-frame 3D points into image pixel coordinates.

    Unlike the rendering helper in dense_ra_video_simulation, this function
    preserves the original point ordering: every input point gets an entry in
    the output arrays so the results can be joined back to per-point data frames.

    Args:
        pts_radar: (N, 3) float array — x, y, z in radar frame.
        R:         (3, 3) rotation matrix (radar → camera).
        t:         (3, 1) translation vector (radar → camera).
        K:         (3, 3) camera intrinsic matrix.
        dist:      (n,)   distortion coefficients.
        img_shape: (height, width) of the target image.
        use_distortion: Apply OpenCV distortion projection. Defaults to False
                        to match the Branch 1 autolabel path, which projects
                        into the recorded/label-map frame with a pinhole model.

    Returns:
        uv:    (N, 2) float32 — pixel coords [u, v]; NaN for invalid points.
        zc:    (N,)   float64 — depth in camera frame (negative = behind camera).
        valid: (N,)   bool    — True where the point projects inside the image
                                and is in front of the camera.
    """
    N = pts_radar.shape[0]
    uv = np.full((N, 2), np.nan, dtype=np.float32)
    zc = np.full(N, np.nan, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)

    if N == 0:
        return uv, zc, valid

    # radar → camera frame
    pts_cam = (R @ pts_radar.T + t).T  # (N, 3)
    zc = pts_cam[:, 2]

    in_front = zc > 0.1
    if not np.any(in_front):
        return uv, zc, valid

    if use_distortion:
        pts_front = pts_cam[in_front].reshape(-1, 1, 3).astype(np.float64)
        pts_2d, _ = cv2.projectPoints(
            pts_front,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K.astype(np.float64),
            dist.astype(np.float64),
        )
        pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)
    else:
        pts_front = pts_cam[in_front]
        z = np.maximum(pts_front[:, 2], 1e-6)
        pts_2d = np.empty((pts_front.shape[0], 2), dtype=np.float32)
        pts_2d[:, 0] = (K[0, 0] * (pts_front[:, 0] / z) + K[0, 2]).astype(np.float32)
        pts_2d[:, 1] = (K[1, 1] * (pts_front[:, 1] / z) + K[1, 2]).astype(np.float32)
    uv[in_front] = pts_2d

    # clip to image bounds
    h, w = img_shape[:2]
    in_bounds = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h)
    )
    idx = np.where(in_front)[0]
    valid[idx[in_bounds]] = True

    return uv, zc, valid


def project_radar_to_image_filtered(
    pts_radar: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Project radar points and return only those that land inside the image.

    Equivalent to dense_ra_video_simulation._project_radar_to_image.
    Intended for rendering overlays where invalid points are simply dropped.

    Returns (M, 2) float32 pixel coordinates for the M valid points.
    """
    uv, _, valid = project_radar_to_image(pts_radar, R, t, K, dist, img_shape)
    return uv[valid]
