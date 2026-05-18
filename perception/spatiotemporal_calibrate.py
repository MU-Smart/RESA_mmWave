#!/usr/bin/env python3
"""Standalone spatiotemporal radar-camera calibration MVP.

This script is a conservative first pass inspired by Wise et al.,
"A Continuous-Time Approach for 3D Radar-to-Camera Extrinsic Calibration".
It does not implement Lie-group B-splines yet. Instead, it keeps the key
targetless idea that is useful for this codebase:

    camera_velocity_camera ~= R_cr * radar_velocity_radar
                              - omega_camera x t_cr

where R_cr and t_cr are the radar-to-camera extrinsics used by the current
projection code. The static calibration result is used as a strong prior and
as a fallback so this branch can be evaluated without changing the live
navigation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
DEFAULT_STATIC_PRIOR = REPO_ROOT / "LLM_ML/jetson_nav_pipeline/config/radar_camera_extrinsics.json"
DEFAULT_ADC_SCRIPT = REPO_ROOT / "LLM_ML/jetson_nav_pipeline/3branch/3branch_adc_to_pointcloud.py"
DEFAULT_ADC_CFG = REPO_ROOT / "LLM_ML/jetson_nav_pipeline/3branch/profile_objdet.cfg"
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "results"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _get_float(row: dict[str, str], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return float(default)


def _get_int(row: dict[str, str], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return int(default)


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_to_euler_zyx_deg(rotation: np.ndarray) -> dict[str, float]:
    euler = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_euler("ZYX", degrees=True)
    return {"yaw": float(euler[0]), "pitch": float(euler[1]), "roll": float(euler[2])}


def _load_timestamp_csv(path: Path) -> list[int]:
    values: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "timestamp_us" in row and row["timestamp_us"] not in (None, ""):
                values.append(int(float(row["timestamp_us"])))
            elif "timestamp_ms" in row and row["timestamp_ms"] not in (None, ""):
                values.append(int(float(row["timestamp_ms"])) * 1000)
            elif "timestamp_ns" in row and row["timestamp_ns"] not in (None, ""):
                values.append(int(float(row["timestamp_ns"])) // 1000)
            else:
                raise ValueError(f"No timestamp column found in {path}")
    return values


def _pointcloud_csv_path(session_dir: Path) -> Path | None:
    candidates = [
        session_dir / "radar_pointcloud.csv",
        session_dir / f"{session_dir.name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_sessions(root: Path | None, session_args: list[str], limit: int | None) -> list[Path]:
    sessions: list[Path] = []
    if session_args:
        for item in session_args:
            path = Path(item)
            if not path.is_absolute():
                if root is None:
                    path = Path.cwd() / path
                else:
                    path = root / path
            sessions.append(path.resolve())
    elif root is not None:
        sessions = sorted(p.resolve() for p in root.iterdir() if p.is_dir() and p.name.startswith("session_"))
    if limit is not None:
        sessions = sessions[: max(0, int(limit))]
    return sessions


def _preprocess_if_needed(
    session_dir: Path,
    *,
    enabled: bool,
    adc_script: Path,
    adc_cfg: Path,
    min_snr: float,
    pfa: float,
    force: bool,
) -> None:
    if _pointcloud_csv_path(session_dir) is not None and not force:
        return
    if not enabled and not force:
        raise FileNotFoundError(
            f"No point-cloud CSV found in {session_dir}. Re-run with --preprocess-missing "
            "or generate the CSV with 3branch_adc_to_pointcloud.py."
        )
    cmd = [
        sys.executable,
        str(adc_script),
        "--session",
        str(session_dir),
        "--cfg",
        str(adc_cfg),
        "--min-snr",
        str(min_snr),
        "--pfa",
        str(pfa),
    ]
    if force:
        cmd.append("--force")
    print(f"[preprocess] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


# ---------------------------------------------------------------------------
# Static prior
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StaticPrior:
    path: Path | None
    rotation: np.ndarray
    translation_m: np.ndarray
    loaded: bool


def load_static_prior(path: Path | None) -> StaticPrior:
    if path is None or not path.exists():
        return StaticPrior(path=path, rotation=np.eye(3, dtype=np.float64), translation_m=np.zeros(3), loaded=False)
    payload = _read_json(path)
    rotation = np.asarray(payload.get("R_radar_to_camera", np.eye(3)), dtype=np.float64)
    translation = np.asarray(payload.get("t_radar_to_camera", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"Static prior rotation has shape {rotation.shape}, expected (3, 3)")
    return StaticPrior(path=path, rotation=rotation, translation_m=translation, loaded=True)


# ---------------------------------------------------------------------------
# Radar ego-motion from DCA point CSV
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RadarDetection:
    timestamp_us: int
    frame_num: int
    xyz: np.ndarray
    doppler_mps: float
    snr: float
    range_m: float


@dataclass(slots=True)
class RadarFrame:
    frame_num: int
    timestamp_us: int
    detections: list[RadarDetection]


@dataclass(slots=True)
class RadarVelocityEstimate:
    frame_num: int
    timestamp_us: int
    velocity_radar_mps: np.ndarray
    speed_mps: float
    n_total: int
    n_inliers: int
    residual_rms_mps: float
    condition_number: float
    observable: bool


@dataclass(slots=True)
class RadarMotion:
    estimates: list[RadarVelocityEstimate]

    @property
    def timestamps_us(self) -> np.ndarray:
        return np.asarray([item.timestamp_us for item in self.estimates], dtype=np.int64)

    @property
    def velocities_mps(self) -> np.ndarray:
        if not self.estimates:
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray([item.velocity_radar_mps for item in self.estimates], dtype=np.float64)

    @property
    def speeds_mps(self) -> np.ndarray:
        return np.asarray([item.speed_mps for item in self.estimates], dtype=np.float64)

    @property
    def observable_mask(self) -> np.ndarray:
        return np.asarray([item.observable for item in self.estimates], dtype=bool)

    @property
    def observable_fraction(self) -> float:
        if not self.estimates:
            return 0.0
        return float(np.mean(self.observable_mask))


def load_radar_frames(session_dir: Path) -> list[RadarFrame]:
    csv_path = _pointcloud_csv_path(session_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No point-cloud CSV found in {session_dir}")

    grouped: dict[int, list[RadarDetection]] = {}
    timestamps: dict[int, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_num = _get_int(row, "frame_num", "radar_frame_num")
            timestamp_us = _get_int(row, "timestamp_us")
            x = _get_float(row, "x")
            y = _get_float(row, "y")
            z = _get_float(row, "z")
            xyz = np.array([x, y, z], dtype=np.float64)
            range_m = _get_float(row, "range_m", default=float(np.linalg.norm(xyz)))
            det = RadarDetection(
                timestamp_us=timestamp_us,
                frame_num=frame_num,
                xyz=xyz,
                doppler_mps=_get_float(row, "doppler_mps", "doppler", "v"),
                snr=_get_float(row, "snr", "power_snr", default=1.0),
                range_m=range_m,
            )
            grouped.setdefault(frame_num, []).append(det)
            timestamps.setdefault(frame_num, timestamp_us)

    return [
        RadarFrame(frame_num=frame_num, timestamp_us=timestamps[frame_num], detections=grouped[frame_num])
        for frame_num in sorted(grouped)
    ]


def _solve_weighted_velocity(unit_vectors: np.ndarray, radial_mps: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(unit_vectors, dtype=np.float64)
    b = np.asarray(radial_mps, dtype=np.float64)
    w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 1e-6, None)).reshape(-1, 1)
    aw = a * w
    bw = b * w[:, 0]
    solution, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    cond = float(np.linalg.cond(aw.T @ aw)) if len(aw) >= 3 else float("inf")
    return solution.astype(np.float64), cond


def estimate_radar_frame_velocity(
    frame: RadarFrame,
    *,
    min_points: int,
    min_snr: float,
    min_range_m: float,
    max_range_m: float,
    max_abs_doppler_mps: float,
    residual_threshold_mps: float,
    ransac_iterations: int,
    doppler_sign: float,
    max_speed_mps: float,
    max_frame_residual_mps: float,
    min_inlier_fraction: float,
) -> RadarVelocityEstimate:
    if not frame.detections:
        return RadarVelocityEstimate(frame.frame_num, frame.timestamp_us, np.zeros(3), 0.0, 0, 0, float("inf"), float("inf"), False)

    xyz = np.asarray([det.xyz for det in frame.detections], dtype=np.float64)
    doppler = np.asarray([det.doppler_mps for det in frame.detections], dtype=np.float64)
    snr = np.asarray([det.snr for det in frame.detections], dtype=np.float64)
    ranges = np.asarray([det.range_m for det in frame.detections], dtype=np.float64)
    ranges = np.where(np.isfinite(ranges) & (ranges > 1e-3), ranges, np.linalg.norm(xyz, axis=1))
    valid = (
        np.all(np.isfinite(xyz), axis=1)
        & np.isfinite(doppler)
        & np.isfinite(snr)
        & (snr >= float(min_snr))
        & (ranges >= float(min_range_m))
        & (ranges <= float(max_range_m))
        & (np.abs(doppler) <= float(max_abs_doppler_mps))
    )
    if int(valid.sum()) < min_points:
        return RadarVelocityEstimate(frame.frame_num, frame.timestamp_us, np.zeros(3), 0.0, int(valid.sum()), 0, float("inf"), float("inf"), False)

    xyz = xyz[valid]
    ranges = ranges[valid]
    unit_vectors = xyz / ranges[:, None]
    radial = float(doppler_sign) * doppler[valid]
    weights = np.clip(snr[valid] / max(float(np.median(snr[valid])), 1e-3), 0.1, 8.0)

    rng = np.random.default_rng(frame.frame_num + 17)
    best_mask = np.zeros(len(unit_vectors), dtype=bool)
    best_velocity = np.zeros(3, dtype=np.float64)
    best_cond = float("inf")
    for _ in range(max(1, int(ransac_iterations))):
        sample = rng.choice(len(unit_vectors), size=3, replace=False)
        try:
            velocity, cond = _solve_weighted_velocity(unit_vectors[sample], radial[sample], weights[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(unit_vectors @ velocity - radial)
        mask = residuals < float(residual_threshold_mps)
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
            best_velocity = velocity
            best_cond = cond

    if int(best_mask.sum()) >= 3:
        velocity, cond = _solve_weighted_velocity(unit_vectors[best_mask], radial[best_mask], weights[best_mask])
    else:
        velocity, cond = _solve_weighted_velocity(unit_vectors, radial, weights)
    residuals = np.abs(unit_vectors @ velocity - radial)
    inliers = residuals < float(residual_threshold_mps)
    use_residuals = residuals[inliers] if np.any(inliers) else residuals
    rms = float(np.sqrt(np.mean(use_residuals**2))) if len(use_residuals) else float("inf")
    inlier_fraction = float(inliers.sum() / max(1, len(inliers)))
    speed = float(np.linalg.norm(velocity))
    observable = bool(
        int(inliers.sum()) >= min_points
        and math.isfinite(cond)
        and cond < 1e4
        and speed <= float(max_speed_mps)
        and rms <= float(max_frame_residual_mps)
        and inlier_fraction >= float(min_inlier_fraction)
    )
    return RadarVelocityEstimate(
        frame_num=frame.frame_num,
        timestamp_us=frame.timestamp_us,
        velocity_radar_mps=velocity,
        speed_mps=speed,
        n_total=int(len(unit_vectors)),
        n_inliers=int(inliers.sum()),
        residual_rms_mps=rms,
        condition_number=float(cond if math.isfinite(cond) else best_cond),
        observable=observable,
    )


def estimate_radar_motion(session_dir: Path, args: argparse.Namespace) -> RadarMotion:
    frames = load_radar_frames(session_dir)
    estimates = [
        estimate_radar_frame_velocity(
            frame,
            min_points=args.min_radar_points,
            min_snr=args.min_snr,
            min_range_m=args.min_range_m,
            max_range_m=args.max_range_m,
            max_abs_doppler_mps=args.max_abs_doppler_mps,
            residual_threshold_mps=args.radar_residual_threshold_mps,
            ransac_iterations=args.radar_ransac_iterations,
            doppler_sign=args.doppler_sign,
            max_speed_mps=args.max_radar_speed_mps,
            max_frame_residual_mps=args.max_radar_frame_residual_mps,
            min_inlier_fraction=args.min_radar_inlier_fraction,
        )
        for frame in frames
    ]
    return RadarMotion(estimates=estimates)


# ---------------------------------------------------------------------------
# RGB-D odometry and local camera motion
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class RGBDSession:
    session_dir: Path
    color_video_path: Path
    color_timestamps_us: list[int]
    depth_timestamps_us: list[int]
    depth_indices: list[int]
    intrinsics: CameraIntrinsics
    depth_scale_m_per_unit: float
    depth_available: bool

    @property
    def frame_count(self) -> int:
        if not self.depth_available:
            return len(self.color_timestamps_us)
        return min(len(self.color_timestamps_us), len(self.depth_indices))


@dataclass(slots=True)
class RGBDFrame:
    frame_index: int
    color_timestamp_us: int
    depth_timestamp_us: int
    color_bgr: np.ndarray
    depth_m: np.ndarray | None


@dataclass(slots=True)
class RelativePose:
    from_index: int
    to_index: int
    timestamp_us: int
    rotation_curr_prev: np.ndarray
    translation_curr_prev: np.ndarray
    success: bool
    n_tracks: int
    n_inliers: int
    rms_error_m: float


@dataclass(slots=True)
class CameraOdometry:
    timestamps_us: np.ndarray
    rotations_wc: np.ndarray
    translations_wc: np.ndarray
    relative_poses: list[RelativePose]
    motion_source: str = "rgbd"
    scale_factor: float = 1.0

    @property
    def valid_fraction(self) -> float:
        if not self.relative_poses:
            return 0.0
        return float(np.mean([pose.success for pose in self.relative_poses]))


@dataclass(slots=True)
class CameraTrajectory:
    timestamps_us: np.ndarray
    rotations_wc: np.ndarray
    translations_wc: np.ndarray

    def _clamp(self, timestamp_us: int) -> int:
        return int(np.clip(timestamp_us, int(self.timestamps_us[0]), int(self.timestamps_us[-1])))

    def rotation_at(self, timestamp_us: int) -> np.ndarray:
        ts = float(self._clamp(timestamp_us))
        if len(self.timestamps_us) == 1:
            return self.rotations_wc[0].copy()
        slerp = Slerp(self.timestamps_us.astype(np.float64), Rotation.from_matrix(self.rotations_wc))
        return slerp([ts]).as_matrix()[0]

    def translation_at(self, timestamp_us: int) -> np.ndarray:
        ts = self._clamp(timestamp_us)
        if len(self.timestamps_us) == 1:
            return self.translations_wc[0].copy()
        return np.asarray(
            [np.interp(ts, self.timestamps_us, self.translations_wc[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )

    def _velocity_indices(self, timestamp_us: int, window: int = 1) -> tuple[int, int]:
        if len(self.timestamps_us) < 2:
            return 0, 0
        idx = int(np.searchsorted(self.timestamps_us, self._clamp(timestamp_us)))
        i0 = max(0, idx - window)
        i1 = min(len(self.timestamps_us) - 1, idx + window)
        if i0 == i1:
            i0 = max(0, i0 - 1)
            i1 = min(len(self.timestamps_us) - 1, i1 + 1)
        return i0, i1

    def linear_velocity_camera_at(self, timestamp_us: int, window: int = 1) -> np.ndarray:
        i0, i1 = self._velocity_indices(timestamp_us, window=window)
        if i0 == i1:
            return np.zeros(3, dtype=np.float64)
        dt = (int(self.timestamps_us[i1]) - int(self.timestamps_us[i0])) * 1e-6
        if dt <= 0.0:
            return np.zeros(3, dtype=np.float64)
        v_world = (self.translations_wc[i1] - self.translations_wc[i0]) / dt
        rotation_wc = self.rotation_at(timestamp_us)
        return rotation_wc.T @ v_world

    def angular_velocity_camera_at(self, timestamp_us: int, window: int = 1) -> np.ndarray:
        i0, i1 = self._velocity_indices(timestamp_us, window=window)
        if i0 == i1:
            return np.zeros(3, dtype=np.float64)
        dt = (int(self.timestamps_us[i1]) - int(self.timestamps_us[i0])) * 1e-6
        if dt <= 0.0:
            return np.zeros(3, dtype=np.float64)
        rel = self.rotations_wc[i0].T @ self.rotations_wc[i1]
        return Rotation.from_matrix(rel).as_rotvec() / dt

    def sample_speed_signal(self) -> np.ndarray:
        return np.asarray(
            [np.linalg.norm(self.linear_velocity_camera_at(int(ts))) for ts in self.timestamps_us],
            dtype=np.float64,
        )


def _intrinsics_from_meta(session_dir: Path) -> tuple[CameraIntrinsics, float]:
    meta_path = session_dir / "meta_data.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta_data.json in {session_dir}")
    meta = _read_json(meta_path)
    calib = meta.get("realsense_calibration", {})
    rgb = calib.get("rgb") or calib.get("color") or {}
    if "K" in rgb:
        k = np.asarray(rgb["K"], dtype=np.float64)
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    else:
        fx = float(rgb.get("fx", 0.0))
        fy = float(rgb.get("fy", 0.0))
        cx = float(rgb.get("cx", 0.0))
        cy = float(rgb.get("cy", 0.0))
    width = int(rgb.get("width", meta.get("width", 1280)))
    height = int(rgb.get("height", meta.get("height", 720)))
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid RGB intrinsics in {meta_path}")
    scale = float(meta.get("depth_scale_m_per_unit", calib.get("depth_scale_m_per_unit", 0.001)))
    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy), scale


def _pair_depth_indices(color_ts: list[int], depth_ts: list[int]) -> list[int]:
    if not color_ts or not depth_ts:
        return []
    if len(color_ts) == len(depth_ts):
        return list(range(len(color_ts)))
    indices: list[int] = []
    depth_idx = 0
    for ts in color_ts:
        while depth_idx + 1 < len(depth_ts):
            cur = abs(depth_ts[depth_idx] - ts)
            nxt = abs(depth_ts[depth_idx + 1] - ts)
            if nxt <= cur:
                depth_idx += 1
            else:
                break
        indices.append(depth_idx)
    return indices


def load_rgbd_session(session_dir: Path, *, allow_monocular: bool) -> RGBDSession:
    stem = session_dir.name
    intrinsics, depth_scale = _intrinsics_from_meta(session_dir)
    color_video = session_dir / f"{stem}_color.mp4"
    color_ts = session_dir / f"{stem}_color_timestamps.csv"
    depth_ts = session_dir / f"{stem}_depth_timestamps.csv"
    if not color_video.exists():
        raise FileNotFoundError(f"Missing color video: {color_video}")
    if not color_ts.exists():
        raise FileNotFoundError(f"Missing color timestamp CSV in {session_dir}")
    color_timestamps = _load_timestamp_csv(color_ts)
    depth_dir = session_dir / "depth"
    depth_available = bool(depth_ts.exists() and depth_dir.exists() and any(depth_dir.glob("*.npy")))
    if depth_available:
        depth_timestamps = _load_timestamp_csv(depth_ts)
        depth_indices = _pair_depth_indices(color_timestamps, depth_timestamps)
    elif allow_monocular:
        depth_timestamps = color_timestamps
        depth_indices = list(range(len(color_timestamps)))
    else:
        raise FileNotFoundError(f"Missing depth timestamp CSV or depth frames in {session_dir}")
    return RGBDSession(
        session_dir=session_dir,
        color_video_path=color_video,
        color_timestamps_us=color_timestamps,
        depth_timestamps_us=depth_timestamps,
        depth_indices=depth_indices,
        intrinsics=intrinsics,
        depth_scale_m_per_unit=depth_scale,
        depth_available=depth_available,
    )


def _load_depth_m(session: RGBDSession, depth_index: int) -> np.ndarray:
    path = session.session_dir / "depth" / f"{depth_index:06d}.npy"
    depth_raw = np.load(path)
    return depth_raw.astype(np.float32) * float(session.depth_scale_m_per_unit)


def iter_rgbd_frames(session_dir: Path, *, step: int, max_frames: int | None, allow_monocular: bool) -> Iterable[RGBDFrame]:
    session = load_rgbd_session(session_dir, allow_monocular=allow_monocular)
    capture = cv2.VideoCapture(str(session.color_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open color video: {session.color_video_path}")
    try:
        frame_idx = 0
        yielded = 0
        while frame_idx < session.frame_count:
            ok, color_bgr = capture.read()
            if not ok:
                break
            if frame_idx % max(1, int(step)) != 0:
                frame_idx += 1
                continue
            depth_idx = session.depth_indices[frame_idx]
            yield RGBDFrame(
                frame_index=frame_idx,
                color_timestamp_us=session.color_timestamps_us[frame_idx],
                depth_timestamp_us=session.depth_timestamps_us[depth_idx],
                color_bgr=color_bgr,
                depth_m=_load_depth_m(session, depth_idx) if session.depth_available else None,
            )
            yielded += 1
            frame_idx += 1
            if max_frames is not None and yielded >= int(max_frames):
                break
    finally:
        capture.release()


def _backproject(points_xy: np.ndarray, depth_m: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape[:2]
    u = np.clip(np.round(points_xy[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(points_xy[:, 1]).astype(int), 0, h - 1)
    z = depth_m[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.2) & (z < 10.0)
    x = (u.astype(np.float64) - float(intr.cx)) * z / float(intr.fx)
    y = (v.astype(np.float64) - float(intr.cy)) * z / float(intr.fy)
    return np.column_stack([x, y, z]).astype(np.float64), valid


def _rigid_transform(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ca = points_a.mean(axis=0)
    cb = points_b.mean(axis=0)
    aa = points_a - ca
    bb = points_b - cb
    u, _, vt = np.linalg.svd(aa.T @ bb)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = cb - rotation @ ca
    return rotation, translation


def _ransac_rigid(points_a: np.ndarray, points_b: np.ndarray, *, iterations: int, threshold_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_a) < 3:
        raise ValueError("At least three 3D correspondences are required")
    rng = np.random.default_rng(11)
    best_mask = np.zeros(len(points_a), dtype=bool)
    best_r = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(len(points_a), size=3, replace=False)
        try:
            r, t = _rigid_transform(points_a[sample], points_b[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm((points_a @ r.T) + t - points_b, axis=1)
        mask = residuals < float(threshold_m)
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask
            best_r = r
            best_t = t
    if int(best_mask.sum()) >= 3:
        best_r, best_t = _rigid_transform(points_a[best_mask], points_b[best_mask])
    return best_r, best_t, best_mask


def _track_features(prev: RGBDFrame, curr: RGBDFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    prev_gray = cv2.cvtColor(prev.color_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr.color_bgr, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(args.odom_max_corners),
        qualityLevel=float(args.odom_quality_level),
        minDistance=float(args.odom_min_distance_px),
    )
    if features is None or len(features) < 12:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    if next_pts is None or status is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), 0
    prev_pts = features.reshape(-1, 2)
    curr_pts = next_pts.reshape(-1, 2)
    tracked = status.reshape(-1).astype(bool)
    return prev_pts[tracked], curr_pts[tracked], int(len(prev_pts))


def estimate_relative_pose_monocular(prev: RGBDFrame, curr: RGBDFrame, intr: CameraIntrinsics, args: argparse.Namespace) -> RelativePose:
    prev_pts, curr_pts, _ = _track_features(prev, curr, args)
    if len(prev_pts) < 12:
        return RelativePose(prev.frame_index, curr.frame_index, curr.color_timestamp_us, np.eye(3), np.zeros(3), False, len(prev_pts), 0, float("inf"))

    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    try:
        essential, inlier_mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=float(args.mono_ransac_threshold_px),
        )
        if essential is None:
            raise ValueError("essential matrix failed")
        _, rotation, translation, pose_mask = cv2.recoverPose(essential, prev_pts, curr_pts, camera_matrix)
        mask = pose_mask.reshape(-1).astype(bool) if pose_mask is not None else inlier_mask.reshape(-1).astype(bool)
        n_inliers = int(mask.sum())
        success = bool(n_inliers >= int(args.odom_min_inliers))
        rms = 0.0 if success else float("inf")
        return RelativePose(
            prev.frame_index,
            curr.frame_index,
            curr.color_timestamp_us,
            rotation.astype(np.float64),
            translation.reshape(3).astype(np.float64),
            success,
            len(prev_pts),
            n_inliers,
            rms,
        )
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return RelativePose(prev.frame_index, curr.frame_index, curr.color_timestamp_us, np.eye(3), np.zeros(3), False, len(prev_pts), 0, float("inf"))


def estimate_relative_pose(prev: RGBDFrame, curr: RGBDFrame, intr: CameraIntrinsics, args: argparse.Namespace) -> RelativePose:
    if prev.depth_m is None or curr.depth_m is None:
        return estimate_relative_pose_monocular(prev, curr, intr, args)

    prev_pts, curr_pts, _ = _track_features(prev, curr, args)
    if len(prev_pts) < 12:
        return RelativePose(prev.frame_index, curr.frame_index, curr.color_timestamp_us, np.eye(3), np.zeros(3), False, len(prev_pts), 0, float("inf"))

    prev_xyz, prev_valid = _backproject(prev_pts, prev.depth_m, intr)
    curr_xyz, curr_valid = _backproject(curr_pts, curr.depth_m, intr)
    valid = prev_valid & curr_valid
    prev_xyz = prev_xyz[valid]
    curr_xyz = curr_xyz[valid]
    if len(prev_xyz) < 8:
        return RelativePose(prev.frame_index, curr.frame_index, curr.color_timestamp_us, np.eye(3), np.zeros(3), False, len(prev_pts), 0, float("inf"))

    try:
        r, t, inliers = _ransac_rigid(
            prev_xyz,
            curr_xyz,
            iterations=args.odom_ransac_iterations,
            threshold_m=args.odom_ransac_threshold_m,
        )
        residuals = np.linalg.norm((prev_xyz @ r.T) + t - curr_xyz, axis=1)
        rms = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if np.any(inliers) else float(np.sqrt(np.mean(residuals**2)))
        success = bool(int(inliers.sum()) >= int(args.odom_min_inliers))
    except (ValueError, np.linalg.LinAlgError):
        r = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        inliers = np.zeros(len(prev_xyz), dtype=bool)
        rms = float("inf")
        success = False
    return RelativePose(prev.frame_index, curr.frame_index, curr.color_timestamp_us, r, t, success, len(prev_pts), int(inliers.sum()), rms)


def run_rgbd_odometry(session_dir: Path, args: argparse.Namespace) -> CameraOdometry:
    rgbd = load_rgbd_session(session_dir, allow_monocular=bool(args.allow_monocular))
    frames = list(iter_rgbd_frames(session_dir, step=args.rgbd_step, max_frames=args.max_rgbd_frames, allow_monocular=bool(args.allow_monocular)))
    if len(frames) < 2:
        raise ValueError(f"Need at least two RGB-D frames in {session_dir}")

    pose_wc = np.eye(4, dtype=np.float64)
    rotations_wc = [pose_wc[:3, :3].copy()]
    translations_wc = [pose_wc[:3, 3].copy()]
    timestamps_us = [frames[0].color_timestamp_us]
    relative: list[RelativePose] = []

    for prev, curr in zip(frames[:-1], frames[1:]):
        rel = estimate_relative_pose(prev, curr, rgbd.intrinsics, args)
        relative.append(rel)
        t_curr_prev = np.eye(4, dtype=np.float64)
        t_curr_prev[:3, :3] = rel.rotation_curr_prev
        t_curr_prev[:3, 3] = rel.translation_curr_prev
        if rel.success:
            pose_wc = pose_wc @ np.linalg.inv(t_curr_prev)
        rotations_wc.append(pose_wc[:3, :3].copy())
        translations_wc.append(pose_wc[:3, 3].copy())
        timestamps_us.append(curr.color_timestamp_us)

    return CameraOdometry(
        timestamps_us=np.asarray(timestamps_us, dtype=np.int64),
        rotations_wc=np.asarray(rotations_wc, dtype=np.float64),
        translations_wc=np.asarray(translations_wc, dtype=np.float64),
        relative_poses=relative,
        motion_source="rgbd" if rgbd.depth_available else "monocular",
        scale_factor=1.0,
    )


# ---------------------------------------------------------------------------
# Spatiotemporal residual assembly and solve
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemporalInitialization:
    offset_us: int
    score: float
    sample_period_us: int
    overlap_samples: int
    source: str = "unknown"


@dataclass(slots=True)
class SessionMotionData:
    session_name: str
    radar_motion: RadarMotion
    camera_odometry: CameraOdometry
    camera_trajectory: CameraTrajectory
    temporal_init: TemporalInitialization


@dataclass(slots=True)
class PairedMotion:
    radar_velocity_r: np.ndarray
    camera_velocity_c: np.ndarray
    camera_omega_c: np.ndarray
    provenance: list[tuple[str, int]]


def _interp_scalar(ts_us: np.ndarray, values: np.ndarray, query_us: np.ndarray) -> np.ndarray:
    return np.interp(query_us, ts_us, values)


def estimate_time_offset(
    radar_timestamps_us: np.ndarray,
    radar_speeds_mps: np.ndarray,
    camera_timestamps_us: np.ndarray,
    camera_speeds_mps: np.ndarray,
    *,
    max_offset_us: int,
    sample_period_us: int | None = None,
) -> TemporalInitialization:
    radar_timestamps_us = np.asarray(radar_timestamps_us, dtype=np.int64)
    camera_timestamps_us = np.asarray(camera_timestamps_us, dtype=np.int64)
    radar_speeds_mps = np.asarray(radar_speeds_mps, dtype=np.float64).reshape(-1)
    camera_speeds_mps = np.asarray(camera_speeds_mps, dtype=np.float64).reshape(-1)
    if len(radar_timestamps_us) < 4 or len(camera_timestamps_us) < 4:
        return TemporalInitialization(0, float("-inf"), 0, 0, "speed_correlation")
    if sample_period_us is None:
        radar_step = int(np.median(np.diff(radar_timestamps_us))) if len(radar_timestamps_us) > 1 else 100000
        camera_step = int(np.median(np.diff(camera_timestamps_us))) if len(camera_timestamps_us) > 1 else radar_step
        sample_period_us = max(1, min(radar_step, camera_step))

    best = TemporalInitialization(0, float("-inf"), int(sample_period_us), 0, "speed_correlation")
    candidates = np.arange(-int(max_offset_us), int(max_offset_us) + int(sample_period_us), int(sample_period_us), dtype=np.int64)
    for offset_us in candidates:
        start = max(int(radar_timestamps_us[0]), int(camera_timestamps_us[0] - offset_us))
        stop = min(int(radar_timestamps_us[-1]), int(camera_timestamps_us[-1] - offset_us))
        if stop - start < 3 * int(sample_period_us):
            continue
        query = np.arange(start, stop + 1, int(sample_period_us), dtype=np.int64)
        if len(query) < 4:
            continue
        radar_sig = _interp_scalar(radar_timestamps_us, radar_speeds_mps, query)
        camera_sig = _interp_scalar(camera_timestamps_us, camera_speeds_mps, query + offset_us)
        radar_sig = radar_sig - float(np.mean(radar_sig))
        camera_sig = camera_sig - float(np.mean(camera_sig))
        denom = float(np.linalg.norm(radar_sig) * np.linalg.norm(camera_sig))
        if denom <= 1e-9:
            continue
        score = float(np.dot(radar_sig, camera_sig) / denom)
        if score > best.score:
            best = TemporalInitialization(int(offset_us), score, int(sample_period_us), int(len(query)), "speed_correlation")
    return best


def _overlap_sample_count(radar_timestamps_us: np.ndarray, camera_timestamps_us: np.ndarray, offset_us: int) -> int:
    if len(radar_timestamps_us) == 0 or len(camera_timestamps_us) == 0:
        return 0
    query = np.asarray(radar_timestamps_us, dtype=np.int64) + int(offset_us)
    cam_t0 = int(np.min(camera_timestamps_us))
    cam_t1 = int(np.max(camera_timestamps_us))
    return int(np.sum((query >= cam_t0) & (query <= cam_t1)))


def estimate_start_time_offset(
    session_dir: Path,
    radar_timestamps_us: np.ndarray,
    camera_timestamps_us: np.ndarray,
) -> TemporalInitialization:
    stem = session_dir.name
    radar_csv = session_dir / f"{stem}_radar_timestamps.csv"
    color_csv = session_dir / f"{stem}_color_timestamps.csv"
    source = "trajectory_start"
    try:
        radar_source = _load_timestamp_csv(radar_csv)
        camera_source = _load_timestamp_csv(color_csv)
        if radar_source and camera_source:
            radar_t0 = int(radar_source[0])
            camera_t0 = int(camera_source[0])
            source = "timestamp_csv_start"
        else:
            radar_t0 = int(radar_timestamps_us[0])
            camera_t0 = int(camera_timestamps_us[0])
    except (FileNotFoundError, ValueError, IndexError):
        if len(radar_timestamps_us) == 0 or len(camera_timestamps_us) == 0:
            return TemporalInitialization(0, float("-inf"), 0, 0, source)
        radar_t0 = int(radar_timestamps_us[0])
        camera_t0 = int(camera_timestamps_us[0])

    offset_us = int(camera_t0 - radar_t0)
    radar_step = int(np.median(np.diff(radar_timestamps_us))) if len(radar_timestamps_us) > 1 else 0
    camera_step = int(np.median(np.diff(camera_timestamps_us))) if len(camera_timestamps_us) > 1 else 0
    sample_period_us = max(0, min(x for x in [radar_step, camera_step] if x > 0)) if (radar_step > 0 or camera_step > 0) else 0
    overlap = _overlap_sample_count(radar_timestamps_us, camera_timestamps_us, offset_us)
    return TemporalInitialization(offset_us, 1.0, sample_period_us, overlap, source)


def choose_temporal_initialization(
    session_dir: Path,
    radar: RadarMotion,
    trajectory: CameraTrajectory,
    args: argparse.Namespace,
) -> TemporalInitialization:
    start = estimate_start_time_offset(session_dir, radar.timestamps_us, trajectory.timestamps_us)
    speed = estimate_time_offset(
        radar.timestamps_us,
        radar.speeds_mps,
        trajectory.timestamps_us,
        trajectory.sample_speed_signal(),
        max_offset_us=int(args.max_offset_ms * 1000),
    )

    if args.temporal_init == "speed":
        return speed if speed.overlap_samples > 0 else start
    if args.temporal_init == "blend" and speed.overlap_samples > 0 and start.overlap_samples > 0:
        weight = float(np.clip(args.temporal_start_weight, 0.0, 1.0))
        offset_us = int(round(weight * start.offset_us + (1.0 - weight) * speed.offset_us))
        overlap = _overlap_sample_count(radar.timestamps_us, trajectory.timestamps_us, offset_us)
        return TemporalInitialization(
            offset_us,
            float((start.score + speed.score) * 0.5),
            int(start.sample_period_us or speed.sample_period_us),
            overlap,
            "blend_start_speed",
        )
    return start if start.overlap_samples > 0 else speed


def gather_pairs(
    session_data: list[SessionMotionData],
    offset_us: int,
    *,
    min_speed_mps: float,
    max_camera_speed_mps: float = 0.0,
    max_camera_omega_radps: float = 0.0,
    max_pair_speed_delta_mps: float = 0.0,
    max_pair_speed_ratio: float = 0.0,
) -> PairedMotion:
    radar_chunks: list[np.ndarray] = []
    camera_chunks: list[np.ndarray] = []
    omega_chunks: list[np.ndarray] = []
    provenance: list[tuple[str, int]] = []
    for data in session_data:
        radar_ts = data.radar_motion.timestamps_us
        radar_vel = data.radar_motion.velocities_mps
        radar_mask = data.radar_motion.observable_mask & (data.radar_motion.speeds_mps >= float(min_speed_mps))
        if len(radar_ts) == 0:
            continue
        query = radar_ts.astype(np.int64) + int(offset_us)
        cam_t0 = int(data.camera_trajectory.timestamps_us[0])
        cam_t1 = int(data.camera_trajectory.timestamps_us[-1])
        valid = radar_mask & (query >= cam_t0) & (query <= cam_t1)
        idxs = np.flatnonzero(valid)
        if len(idxs) == 0:
            continue
        cam_v = np.asarray([data.camera_trajectory.linear_velocity_camera_at(int(query[i])) for i in idxs], dtype=np.float64)
        cam_w = np.asarray([data.camera_trajectory.angular_velocity_camera_at(int(query[i])) for i in idxs], dtype=np.float64)
        finite = np.all(np.isfinite(cam_v), axis=1) & np.all(np.isfinite(cam_w), axis=1)
        if not np.any(finite):
            continue
        idxs = idxs[finite]
        radar_selected = radar_vel[idxs]
        cam_selected = cam_v[finite]
        omega_selected = cam_w[finite]

        pair_valid = np.ones(len(idxs), dtype=bool)
        radar_speed = np.linalg.norm(radar_selected, axis=1)
        camera_speed = np.linalg.norm(cam_selected, axis=1)
        camera_omega = np.linalg.norm(omega_selected, axis=1)
        if max_camera_speed_mps > 0.0:
            pair_valid &= camera_speed <= float(max_camera_speed_mps)
        if max_camera_omega_radps > 0.0:
            pair_valid &= camera_omega <= float(max_camera_omega_radps)
        if max_pair_speed_delta_mps > 0.0:
            pair_valid &= np.abs(radar_speed - camera_speed) <= float(max_pair_speed_delta_mps)
        if max_pair_speed_ratio > 0.0:
            lo = np.maximum(np.minimum(radar_speed, camera_speed), float(min_speed_mps))
            hi = np.maximum(radar_speed, camera_speed)
            pair_valid &= (hi / lo) <= float(max_pair_speed_ratio)
        if not np.any(pair_valid):
            continue

        kept_idxs = idxs[pair_valid]
        radar_chunks.append(radar_selected[pair_valid])
        camera_chunks.append(cam_selected[pair_valid])
        omega_chunks.append(omega_selected[pair_valid])
        provenance.extend((data.session_name, int(i)) for i in kept_idxs)
    if not radar_chunks:
        empty = np.zeros((0, 3), dtype=np.float64)
        return PairedMotion(empty, empty, empty, [])
    return PairedMotion(np.vstack(radar_chunks), np.vstack(camera_chunks), np.vstack(omega_chunks), provenance)


def solve_wahba_rotation(radar_vectors: np.ndarray, camera_vectors: np.ndarray, *, min_speed_mps: float = 0.03) -> tuple[np.ndarray, int, float]:
    radar_vectors = np.asarray(radar_vectors, dtype=np.float64)
    camera_vectors = np.asarray(camera_vectors, dtype=np.float64)
    if radar_vectors.shape != camera_vectors.shape or radar_vectors.ndim != 2 or radar_vectors.shape[1] != 3:
        return np.eye(3, dtype=np.float64), 0, float("inf")
    rn = np.linalg.norm(radar_vectors, axis=1)
    cn = np.linalg.norm(camera_vectors, axis=1)
    valid = np.isfinite(rn) & np.isfinite(cn) & (rn >= min_speed_mps) & (cn >= min_speed_mps)
    if int(valid.sum()) < 3:
        return np.eye(3, dtype=np.float64), int(valid.sum()), float("inf")
    ru = radar_vectors[valid] / rn[valid, None]
    cu = camera_vectors[valid] / cn[valid, None]
    weights = np.minimum(rn[valid], cn[valid])
    cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(len(ru)):
        cov += float(weights[i]) * np.outer(cu[i], ru[i])
    u, _, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = u @ vt
    aligned = (rotation @ radar_vectors[valid].T).T
    rms = float(np.sqrt(np.mean(np.sum((camera_vectors[valid] - aligned) ** 2, axis=1))))
    return rotation, int(valid.sum()), rms


def _pair_filter_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "max_camera_speed_mps": float(args.max_camera_speed_mps),
        "max_camera_omega_radps": float(args.max_camera_omega_radps),
        "max_pair_speed_delta_mps": float(args.max_pair_speed_delta_mps),
        "max_pair_speed_ratio": float(args.max_pair_speed_ratio),
    }


def _residual_metrics(rotation: np.ndarray, translation: np.ndarray, pairs: PairedMotion) -> tuple[float, float, float]:
    if len(pairs.radar_velocity_r) == 0:
        return float("inf"), float("inf"), float("inf")
    predicted = (rotation @ pairs.radar_velocity_r.T).T - np.cross(pairs.camera_omega_c, translation.reshape(1, 3))
    norms = np.linalg.norm(pairs.camera_velocity_c - predicted, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.mean(norms)), float(np.max(norms))


def _excitation_summary(pairs: PairedMotion) -> dict[str, Any]:
    def summarize(values: np.ndarray) -> dict[str, Any]:
        if len(values) < 3:
            return {"singular_values": [], "effective_rank": 0, "condition": float("inf")}
        centered = values - np.mean(values, axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        tol = max(float(s[0]) * 0.05, 1e-6) if len(s) else 1e-6
        rank = int(np.sum(s > tol))
        cond = float(s[0] / max(float(s[-1]), 1e-9)) if len(s) else float("inf")
        return {"singular_values": [float(x) for x in s], "effective_rank": rank, "condition": cond}

    return {
        "radar_velocity": summarize(pairs.radar_velocity_r),
        "camera_velocity": summarize(pairs.camera_velocity_c),
        "camera_angular_velocity": summarize(pairs.camera_omega_c),
    }


def solve_spatiotemporal(
    session_data: list[SessionMotionData],
    static_prior: StaticPrior,
    args: argparse.Namespace,
) -> dict[str, Any]:
    temporal_offsets = [item.temporal_init.offset_us for item in session_data if item.temporal_init.overlap_samples > 0]
    init_offset_us = int(np.median(temporal_offsets)) if temporal_offsets else 0
    pair_filter_kwargs = _pair_filter_kwargs(args)
    init_pairs = gather_pairs(session_data, init_offset_us, min_speed_mps=args.min_motion_speed_mps, **pair_filter_kwargs)

    wahba_r, wahba_pairs, wahba_rms = solve_wahba_rotation(init_pairs.radar_velocity_r, init_pairs.camera_velocity_c)
    init_rotation = static_prior.rotation if args.init_rotation == "static" or wahba_pairs < 3 else wahba_r
    init_translation = static_prior.translation_m.copy()
    x0 = np.concatenate([
        Rotation.from_matrix(init_rotation).as_rotvec(),
        init_translation,
        [init_offset_us * 1e-6],
    ])

    time_prior_s = init_offset_us * 1e-6
    rot_prior = static_prior.rotation
    trans_prior = static_prior.translation_m

    def objective(vector: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(vector[:3]).as_matrix()
        translation = vector[3:6]
        offset_s = time_prior_s if args.time_mode == "fixed" else float(vector[6])
        offset_us = int(round(offset_s * 1e6))
        pairs = gather_pairs(session_data, offset_us, min_speed_mps=args.min_motion_speed_mps, **pair_filter_kwargs)
        if len(pairs.radar_velocity_r) == 0:
            residuals = np.array([1000.0], dtype=np.float64)
        else:
            predicted = (rotation @ pairs.radar_velocity_r.T).T - np.cross(pairs.camera_omega_c, translation.reshape(1, 3))
            residuals = (pairs.camera_velocity_c - predicted).reshape(-1)
        if static_prior.loaded and args.rotation_prior_weight > 0.0:
            rot_delta = Rotation.from_matrix(rotation @ rot_prior.T).as_rotvec()
            residuals = np.concatenate([residuals, float(args.rotation_prior_weight) * rot_delta])
        if static_prior.loaded and args.translation_prior_weight > 0.0:
            residuals = np.concatenate([residuals, float(args.translation_prior_weight) * (translation - trans_prior)])
        if args.time_mode == "solve" and args.time_prior_weight > 0.0:
            residuals = np.concatenate([residuals, [float(args.time_prior_weight) * (float(vector[6]) - time_prior_s)]])
        return residuals

    lower = np.full(7, -np.inf, dtype=np.float64)
    upper = np.full(7, np.inf, dtype=np.float64)
    if args.time_mode == "fixed":
        lower[6] = time_prior_s - 1e-9
        upper[6] = time_prior_s + 1e-9
    else:
        lower[6] = -float(args.max_offset_ms) / 1000.0
        upper[6] = float(args.max_offset_ms) / 1000.0

    lsq = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(args.loss_scale_mps),
        x_scale="jac",
        max_nfev=int(args.max_nfev),
    )

    est_rotation = Rotation.from_rotvec(lsq.x[:3]).as_matrix()
    est_translation = lsq.x[3:6].astype(np.float64)
    est_offset_us = int(round((time_prior_s if args.time_mode == "fixed" else float(lsq.x[6])) * 1e6))
    solved_pairs = gather_pairs(session_data, est_offset_us, min_speed_mps=args.min_motion_speed_mps, **pair_filter_kwargs)
    residual_rms, residual_mean, residual_max = _residual_metrics(est_rotation, est_translation, solved_pairs)

    fallback_reason = "targetless_result_accepted"
    use_static = False
    if not bool(lsq.success):
        fallback_reason = "least_squares_failed"
        use_static = static_prior.loaded
    elif len(solved_pairs.radar_velocity_r) < int(args.min_paired_samples):
        fallback_reason = "insufficient_paired_motion_samples"
        use_static = static_prior.loaded
    elif not math.isfinite(residual_rms) or residual_rms > float(args.max_residual_rms_mps):
        fallback_reason = "residual_rms_too_high"
        use_static = static_prior.loaded

    accepted_rotation = static_prior.rotation if use_static else est_rotation
    accepted_translation = static_prior.translation_m if use_static else est_translation

    return {
        "method": "spatiotemporal_velocity_mvp",
        "R_radar_to_camera": accepted_rotation.tolist(),
        "t_radar_to_camera": accepted_translation.tolist(),
        "euler_ZYX_deg": rotation_to_euler_zyx_deg(accepted_rotation),
        "translation_magnitude_m": float(np.linalg.norm(accepted_translation)),
        "time_offset_ms": float(est_offset_us / 1000.0),
        "estimated_R_radar_to_camera": est_rotation.tolist(),
        "estimated_t_radar_to_camera": est_translation.tolist(),
        "estimated_euler_ZYX_deg": rotation_to_euler_zyx_deg(est_rotation),
        "static_prior": {
            "loaded": bool(static_prior.loaded),
            "path": None if static_prior.path is None else str(static_prior.path),
            "R_radar_to_camera": static_prior.rotation.tolist(),
            "t_radar_to_camera": static_prior.translation_m.tolist(),
        },
        "fallback": {
            "use_static": bool(use_static),
            "reason": fallback_reason,
        },
        "solver": {
            "success": bool(lsq.success),
            "message": str(lsq.message),
            "cost": float(lsq.cost),
            "nfev": int(lsq.nfev),
            "loss": "soft_l1",
            "loss_scale_mps": float(args.loss_scale_mps),
        },
        "initialization": {
            "rotation_source": args.init_rotation,
            "time_mode": args.time_mode,
            "median_time_offset_ms": float(init_offset_us / 1000.0),
            "wahba_pairs": int(wahba_pairs),
            "wahba_rms_mps": float(wahba_rms),
        },
        "pair_filters": pair_filter_kwargs,
        "paired_motion_samples": int(len(solved_pairs.radar_velocity_r)),
        "residual_rms_mps": float(residual_rms),
        "residual_mean_mps": float(residual_mean),
        "residual_max_mps": float(residual_max),
        "excitation": _excitation_summary(solved_pairs),
        "temporal_initializations": [
            {
                "session": item.session_name,
                "offset_us": int(item.temporal_init.offset_us),
                "score": float(item.temporal_init.score),
                "sample_period_us": int(item.temporal_init.sample_period_us),
                "overlap_samples": int(item.temporal_init.overlap_samples),
                "source": item.temporal_init.source,
            }
            for item in session_data
        ],
    }


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _summarize_radar_motion(radar: RadarMotion) -> dict[str, Any]:
    estimates = radar.estimates
    inliers = np.asarray([item.n_inliers for item in estimates], dtype=np.float64) if estimates else np.zeros(0)
    residuals = np.asarray([item.residual_rms_mps for item in estimates if item.observable], dtype=np.float64)
    speeds = radar.speeds_mps
    return {
        "frames": int(len(estimates)),
        "observable_fraction": float(radar.observable_fraction),
        "median_speed_mps": float(np.median(speeds)) if len(speeds) else 0.0,
        "median_inliers": float(np.median(inliers)) if len(inliers) else 0.0,
        "median_residual_rms_mps": float(np.median(residuals)) if len(residuals) else float("inf"),
    }


def _estimate_monocular_scale(
    radar: RadarMotion,
    trajectory: CameraTrajectory,
    temporal: TemporalInitialization,
    args: argparse.Namespace,
) -> float:
    if args.monocular_scale_mode == "none":
        return 1.0
    ratios: list[float] = []
    radar_ts = radar.timestamps_us
    radar_speeds = radar.speeds_mps
    mask = radar.observable_mask & (radar_speeds >= float(args.min_motion_speed_mps))
    for ts, radar_speed in zip(radar_ts[mask], radar_speeds[mask]):
        query = int(ts) + int(temporal.offset_us)
        if query < int(trajectory.timestamps_us[0]) or query > int(trajectory.timestamps_us[-1]):
            continue
        camera_speed = float(np.linalg.norm(trajectory.linear_velocity_camera_at(query)))
        if math.isfinite(camera_speed) and camera_speed > 1e-6:
            ratios.append(float(radar_speed) / camera_speed)
    if len(ratios) < int(args.min_monocular_scale_pairs):
        return 1.0
    scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
    return float(np.clip(scale, float(args.min_monocular_scale), float(args.max_monocular_scale)))


def _scale_odometry_translations(odom: CameraOdometry, scale: float) -> CameraOdometry:
    return CameraOdometry(
        timestamps_us=odom.timestamps_us,
        rotations_wc=odom.rotations_wc,
        translations_wc=odom.translations_wc * float(scale),
        relative_poses=odom.relative_poses,
        motion_source=odom.motion_source,
        scale_factor=float(scale),
    )


def _summarize_odometry(odom: CameraOdometry) -> dict[str, Any]:
    tracks = np.asarray([pose.n_tracks for pose in odom.relative_poses], dtype=np.float64) if odom.relative_poses else np.zeros(0)
    inliers = np.asarray([pose.n_inliers for pose in odom.relative_poses], dtype=np.float64) if odom.relative_poses else np.zeros(0)
    rms = np.asarray([pose.rms_error_m for pose in odom.relative_poses if pose.success], dtype=np.float64)
    path_length = float(np.sum(np.linalg.norm(np.diff(odom.translations_wc, axis=0), axis=1))) if len(odom.translations_wc) >= 2 else 0.0
    return {
        "frames": int(len(odom.timestamps_us)),
        "valid_fraction": float(odom.valid_fraction),
        "path_length_m": path_length,
        "median_tracks": float(np.median(tracks)) if len(tracks) else 0.0,
        "median_inliers": float(np.median(inliers)) if len(inliers) else 0.0,
        "median_rms_error_m": float(np.median(rms)) if len(rms) else float("inf"),
        "motion_source": odom.motion_source,
        "scale_factor": float(odom.scale_factor),
    }


def process_session(session_dir: Path, args: argparse.Namespace) -> tuple[SessionMotionData | None, dict[str, Any]]:
    info: dict[str, Any] = {"session": session_dir.name, "path": str(session_dir)}
    try:
        _preprocess_if_needed(
            session_dir,
            enabled=bool(args.preprocess_missing),
            adc_script=args.adc_script,
            adc_cfg=args.adc_cfg,
            min_snr=args.min_snr,
            pfa=args.cfar_pfa,
            force=bool(args.force_preprocess),
        )
        radar = estimate_radar_motion(session_dir, args)
        odom = run_rgbd_odometry(session_dir, args)
        trajectory = CameraTrajectory(
            timestamps_us=odom.timestamps_us,
            rotations_wc=odom.rotations_wc,
            translations_wc=odom.translations_wc,
        )
        temporal = choose_temporal_initialization(session_dir, radar, trajectory, args)
        if odom.motion_source == "monocular":
            scale = _estimate_monocular_scale(radar, trajectory, temporal, args)
            odom = _scale_odometry_translations(odom, scale)
            trajectory = CameraTrajectory(
                timestamps_us=odom.timestamps_us,
                rotations_wc=odom.rotations_wc,
                translations_wc=odom.translations_wc,
            )
        data = SessionMotionData(
            session_name=session_dir.name,
            radar_motion=radar,
            camera_odometry=odom,
            camera_trajectory=trajectory,
            temporal_init=temporal,
        )
        info["radar_motion"] = _summarize_radar_motion(radar)
        info["camera_odometry"] = _summarize_odometry(odom)
        info["temporal_init"] = {
            "offset_ms": float(temporal.offset_us / 1000.0),
            "score": float(temporal.score),
            "overlap_samples": int(temporal.overlap_samples),
            "source": temporal.source,
        }
        info["usable"] = True
        return data, info
    except Exception as exc:
        info["usable"] = False
        info["error"] = str(exc)
        return None, info


def save_pairs_csv(path: Path, pairs: PairedMotion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session",
            "radar_index",
            "radar_vx",
            "radar_vy",
            "radar_vz",
            "camera_vx",
            "camera_vy",
            "camera_vz",
            "camera_wx",
            "camera_wy",
            "camera_wz",
        ])
        for i, (session_name, radar_idx) in enumerate(pairs.provenance):
            writer.writerow(
                [session_name, radar_idx]
                + [float(x) for x in pairs.radar_velocity_r[i]]
                + [float(x) for x in pairs.camera_velocity_c[i]]
                + [float(x) for x in pairs.camera_omega_c[i]]
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--processing-root", type=Path, default=None, help="Root containing session_* directories.")
    parser.add_argument("--session", action="append", default=[], help="Session directory or name under --processing-root. May repeat.")
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--static-prior", type=Path, default=DEFAULT_STATIC_PRIOR)

    parser.add_argument("--preprocess-missing", action="store_true", help="Run current 3branch DCA preprocessing if point CSV is missing.")
    parser.add_argument("--force-preprocess", action="store_true", help="Force DCA point-cloud regeneration.")
    parser.add_argument("--adc-script", type=Path, default=DEFAULT_ADC_SCRIPT)
    parser.add_argument("--adc-cfg", type=Path, default=DEFAULT_ADC_CFG)
    parser.add_argument("--cfar-pfa", type=float, default=1e-2)

    parser.add_argument("--min-snr", type=float, default=3.0)
    parser.add_argument("--min-range-m", type=float, default=0.3)
    parser.add_argument("--max-range-m", type=float, default=8.0)
    parser.add_argument("--max-abs-doppler-mps", type=float, default=3.0)
    parser.add_argument("--doppler-sign", type=float, default=-1.0)
    parser.add_argument("--min-radar-points", type=int, default=6)
    parser.add_argument("--radar-residual-threshold-mps", type=float, default=0.35)
    parser.add_argument("--radar-ransac-iterations", type=int, default=64)
    parser.add_argument("--max-radar-speed-mps", type=float, default=3.0)
    parser.add_argument("--max-radar-frame-residual-mps", type=float, default=0.25)
    parser.add_argument("--min-radar-inlier-fraction", type=float, default=0.2)

    parser.add_argument("--rgbd-step", type=int, default=3)
    parser.add_argument("--max-rgbd-frames", type=int, default=180)
    parser.add_argument("--odom-max-corners", type=int, default=800)
    parser.add_argument("--odom-quality-level", type=float, default=0.01)
    parser.add_argument("--odom-min-distance-px", type=float, default=7.0)
    parser.add_argument("--odom-ransac-iterations", type=int, default=96)
    parser.add_argument("--odom-ransac-threshold-m", type=float, default=0.08)
    parser.add_argument("--odom-min-inliers", type=int, default=6)
    parser.add_argument("--allow-monocular", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mono-ransac-threshold-px", type=float, default=1.5)
    parser.add_argument("--monocular-scale-mode", choices=["none", "radar_median"], default="radar_median")
    parser.add_argument("--min-monocular-scale-pairs", type=int, default=20)
    parser.add_argument("--min-monocular-scale", type=float, default=0.01)
    parser.add_argument("--max-monocular-scale", type=float, default=5.0)

    parser.add_argument("--max-offset-ms", type=float, default=2000.0)
    parser.add_argument("--temporal-init", choices=["start", "speed", "blend"], default="start")
    parser.add_argument("--temporal-start-weight", type=float, default=0.85)
    parser.add_argument("--time-mode", choices=["fixed", "solve"], default="fixed")
    parser.add_argument("--min-motion-speed-mps", type=float, default=0.05)
    parser.add_argument("--max-camera-speed-mps", type=float, default=0.0)
    parser.add_argument("--max-camera-omega-radps", type=float, default=0.0)
    parser.add_argument("--max-pair-speed-delta-mps", type=float, default=0.0)
    parser.add_argument("--max-pair-speed-ratio", type=float, default=0.0)
    parser.add_argument("--init-rotation", choices=["static", "wahba"], default="static")
    parser.add_argument("--rotation-prior-weight", type=float, default=0.35)
    parser.add_argument("--translation-prior-weight", type=float, default=4.0)
    parser.add_argument("--time-prior-weight", type=float, default=5.0)
    parser.add_argument("--loss-scale-mps", type=float, default=0.2)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--min-paired-samples", type=int, default=40)
    parser.add_argument("--max-residual-rms-mps", type=float, default=0.45)
    parser.add_argument("--save-pairs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.processing_root is None and not args.session:
        raise SystemExit("[ERROR] Provide --processing-root or at least one --session")
    root = args.processing_root.resolve() if args.processing_root is not None else None
    sessions = _resolve_sessions(root, args.session, args.session_limit)
    if not sessions:
        raise SystemExit("[ERROR] No sessions resolved")

    static_prior = load_static_prior(args.static_prior.resolve() if args.static_prior is not None else None)
    print(f"[setup] sessions={len(sessions)} static_prior_loaded={static_prior.loaded}", flush=True)
    print(f"[setup] output_dir={args.output_dir}", flush=True)

    usable_data: list[SessionMotionData] = []
    diagnostics: dict[str, Any] = {
        "method": "spatiotemporal_velocity_mvp",
        "sessions": {},
        "paper_basis": {
            "velocity_residual": "v_camera ~= R_cr * v_radar - omega_camera x t_cr",
            "mvp_simplification": "finite-difference RGB-D egomotion instead of Lie-group B-splines",
        },
    }

    for idx, session_dir in enumerate(sessions, start=1):
        print(f"[session {idx}/{len(sessions)}] {session_dir.name}", flush=True)
        data, info = process_session(session_dir, args)
        diagnostics["sessions"][session_dir.name] = info
        if data is not None:
            usable_data.append(data)
            radar_obs = info["radar_motion"]["observable_fraction"]
            odom_valid = info["camera_odometry"]["valid_fraction"]
            temp_ms = info["temporal_init"]["offset_ms"]
            print(
                "[session]"
                f" usable radar_obs={radar_obs:.3f}"
                f" odom_valid={odom_valid:.3f}"
                f" time_offset_ms={temp_ms:.1f}",
                flush=True,
            )
        else:
            print(f"[session] skipped: {info.get('error')}", flush=True)

    if not usable_data:
        diagnostics["solve"] = {"success": False, "error": "no usable sessions"}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "spatiotemporal_diagnostics.json", diagnostics)
        raise SystemExit("[ERROR] No usable sessions")

    result = solve_spatiotemporal(usable_data, static_prior, args)
    diagnostics["solve"] = {
        "success": bool(result["solver"]["success"]),
        "paired_motion_samples": int(result["paired_motion_samples"]),
        "residual_rms_mps": float(result["residual_rms_mps"]),
        "fallback": result["fallback"],
        "excitation": result["excitation"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "spatiotemporal_radar_camera_extrinsics.json"
    diag_path = args.output_dir / "spatiotemporal_diagnostics.json"
    _write_json(result_path, result)
    _write_json(diag_path, diagnostics)

    if args.save_pairs:
        offset_us = int(round(float(result["time_offset_ms"]) * 1000.0))
        pairs = gather_pairs(usable_data, offset_us, min_speed_mps=args.min_motion_speed_mps, **_pair_filter_kwargs(args))
        save_pairs_csv(args.output_dir / "spatiotemporal_motion_pairs.csv", pairs)

    print(f"[done] result={result_path}", flush=True)
    print(f"[done] diagnostics={diag_path}", flush=True)
    print(
        "[done]"
        f" paired={result['paired_motion_samples']}"
        f" residual_rms_mps={result['residual_rms_mps']:.4f}"
        f" time_offset_ms={result['time_offset_ms']:.1f}"
        f" fallback={result['fallback']['use_static']}:{result['fallback']['reason']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
