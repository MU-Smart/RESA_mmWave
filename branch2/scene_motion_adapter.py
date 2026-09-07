# author: Demetrius Hullum Scott
"""
motion-aware scene aggregation helpers. 
integrates into replay/simulation scripts, comparing current-frame aggregation
against motion-compensated temporal window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class MotionAwareConfig:
    window_frames: int = 4
    doppler_sign: float = -1.0
    min_ego_points: int = 8
    max_abs_doppler_mps: float = 2.5
    ransac_iterations: int = 64
    static_residual_mps: float = 0.25
    dynamic_residual_mps: float = 0.55
    min_static_inlier_fraction: float = 0.25
    max_ego_speed_mps: float = 2.5
    grid_size_m: float = 0.35
    min_persist_frames: int = 2
    immediate_hazard_y_m: float = 1.20
    max_history_y_m: float = 5.0
    keep_ambiguous_current: bool = True


@dataclass(slots=True)
class FramePacket:
    frame_num: int
    timestamp_us: int
    points: list[dict[str, Any]]


@dataclass(slots=True)
class EgoMotionEstimate:
    available: bool
    vx_mps: float
    vy_mps: float
    speed_mps: float
    n_candidate_points: int
    n_static_points: int
    residual_rms_mps: float
    inlier_fraction: float
    source: str = "spatiotemporal_scene_doppler"

    def as_payload(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "source": self.source,
            "vx_mps": round(float(self.vx_mps), 3),
            "vy_mps": round(float(self.vy_mps), 3),
            "speed_mps": round(float(self.speed_mps), 3),
            "n_static_points": int(self.n_static_points),
            "n_candidate_points": int(self.n_candidate_points),
            "residual_rms_mps": round(float(self.residual_rms_mps), 3),
            "inlier_fraction": round(float(self.inlier_fraction), 3),
        }


@dataclass(slots=True)
class MotionAwareWindow:
    points: list[dict[str, Any]]
    ego_motion: EgoMotionEstimate
    diagnostics: dict[str, Any]


def _range_xy(point: dict[str, Any]) -> float:
    return float(np.hypot(float(point.get("x", 0.0)), float(point.get("y", 0.0))))


def _unit_xy(point: dict[str, Any]) -> tuple[float, float] | None:
    rng = _range_xy(point)
    if not np.isfinite(rng) or rng <= 1e-3:
        return None
    return float(point.get("x", 0.0)) / rng, float(point.get("y", 0.0)) / rng


def _solve_velocity(unit_vectors: np.ndarray, radial_mps: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        solution, *_ = np.linalg.lstsq(unit_vectors, radial_mps, rcond=None)
        return solution.astype(np.float64)
    w = np.sqrt(np.clip(weights.astype(np.float64), 1e-6, None)).reshape(-1, 1)
    solution, *_ = np.linalg.lstsq(unit_vectors * w, radial_mps * w[:, 0], rcond=None)
    return solution.astype(np.float64)


def estimate_ego_velocity(points: list[dict[str, Any]], config: MotionAwareConfig) -> EgoMotionEstimate:
    """Estimate 2D rig ego velocity from static-world radar Doppler returns."""

    candidates: list[dict[str, Any]] = []
    for point in points:
        cls = str(point.get("pred_class", "")).lower()
        if cls not in {"structure", "floor"}:
            continue
        doppler = float(point.get("doppler", 0.0))
        if not np.isfinite(doppler) or abs(doppler) > float(config.max_abs_doppler_mps):
            continue
        unit = _unit_xy(point)
        if unit is None:
            continue
        candidates.append(point)

    if len(candidates) < int(config.min_ego_points):
        return EgoMotionEstimate(False, 0.0, 0.0, 0.0, len(candidates), 0, float("inf"), 0.0)

    unit_vectors = np.asarray([_unit_xy(point) for point in candidates], dtype=np.float64)
    radial = np.asarray([float(config.doppler_sign) * float(point.get("doppler", 0.0)) for point in candidates], dtype=np.float64)
    snr = np.asarray([float(point.get("snr", 1.0)) for point in candidates], dtype=np.float64)
    weights = np.clip(snr / max(float(np.nanmedian(snr)), 1e-3), 0.1, 8.0)

    rng = np.random.default_rng(101 + len(candidates))
    best_mask = np.zeros(len(candidates), dtype=bool)
    for _ in range(max(1, int(config.ransac_iterations))):
        if len(candidates) >= 2:
            sample = rng.choice(len(candidates), size=2, replace=False)
        else:
            sample = np.arange(len(candidates))
        try:
            velocity = _solve_velocity(unit_vectors[sample], radial[sample], weights[sample])
        except np.linalg.LinAlgError:
            continue
        residuals = np.abs(unit_vectors @ velocity - radial)
        mask = residuals <= float(config.static_residual_mps)
        if int(mask.sum()) > int(best_mask.sum()):
            best_mask = mask

    if int(best_mask.sum()) >= 2:
        velocity = _solve_velocity(unit_vectors[best_mask], radial[best_mask], weights[best_mask])
    else:
        velocity = _solve_velocity(unit_vectors, radial, weights)
    residuals = np.abs(unit_vectors @ velocity - radial)
    inliers = residuals <= float(config.static_residual_mps)
    use_residuals = residuals[inliers] if np.any(inliers) else residuals
    residual_rms = float(np.sqrt(np.mean(use_residuals**2))) if len(use_residuals) else float("inf")
    speed = float(np.linalg.norm(velocity))
    inlier_fraction = float(inliers.sum() / max(1, len(candidates)))
    available = bool(
        int(inliers.sum()) >= int(config.min_ego_points)
        and speed <= float(config.max_ego_speed_mps)
        and inlier_fraction >= float(config.min_static_inlier_fraction)
        and np.isfinite(residual_rms)
    )
    return EgoMotionEstimate(
        available=available,
        vx_mps=float(velocity[0] if available else 0.0),
        vy_mps=float(velocity[1] if available else 0.0),
        speed_mps=float(speed if available else 0.0),
        n_candidate_points=int(len(candidates)),
        n_static_points=int(inliers.sum()),
        residual_rms_mps=residual_rms,
        inlier_fraction=inlier_fraction,
    )


def annotate_ego_doppler(
    points: list[dict[str, Any]],
    ego: EgoMotionEstimate,
    config: MotionAwareConfig,
) -> list[dict[str, Any]]:
    """Add expected stationary-world Doppler and residual labels to points."""

    annotated: list[dict[str, Any]] = []
    for point in points:
        out = dict(point)
        unit = _unit_xy(out)
        doppler = float(out.get("doppler", 0.0))
        if not ego.available or unit is None or not np.isfinite(doppler):
            out.update({
                "ego_expected_doppler_mps": 0.0,
                "ego_doppler_residual_mps": 0.0,
                "motion_label": "unknown",
                "spatiotemporal_weight": 1.0,
            })
            annotated.append(out)
            continue

        expected = -(float(ego.vx_mps) * unit[0] + float(ego.vy_mps) * unit[1])
        residual = doppler - expected
        abs_residual = abs(float(residual))
        if abs_residual <= float(config.static_residual_mps):
            label = "static_world"
            weight = 1.0
        elif abs_residual >= float(config.dynamic_residual_mps):
            label = "dynamic_or_ghost"
            weight = 0.15 if out.get("pred_class") in {"structure", "floor"} else 1.0
        else:
            label = "ambiguous_motion"
            weight = 0.45 if out.get("pred_class") in {"structure", "floor"} else 1.0
        out.update({
            "ego_expected_doppler_mps": round(float(expected), 4),
            "ego_doppler_residual_mps": round(float(residual), 4),
            "motion_label": label,
            "spatiotemporal_weight": float(weight),
        })
        annotated.append(out)
    return annotated


def _integrate_frame_poses(frames: list[FramePacket], estimates: list[EgoMotionEstimate]) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    if not frames:
        return poses
    pose = np.zeros(2, dtype=np.float64)
    poses[int(frames[0].frame_num)] = pose.copy()
    for prev, cur, ego in zip(frames[:-1], frames[1:], estimates[:-1]):
        dt = max(0.0, (int(cur.timestamp_us) - int(prev.timestamp_us)) * 1e-6)
        if ego.available and dt <= 0.5:
            pose = pose + np.asarray([ego.vx_mps, ego.vy_mps], dtype=np.float64) * dt
        poses[int(cur.frame_num)] = pose.copy()
    return poses


def _grid_key(point: dict[str, Any], config: MotionAwareConfig) -> tuple[str, int, int, int]:
    cell = max(float(config.grid_size_m), 1e-3)
    return (
        str(point.get("pred_class", "")),
        int(np.floor(float(point.get("x", 0.0)) / cell)),
        int(np.floor(float(point.get("y", 0.0)) / cell)),
        int(np.floor(float(point.get("z", 0.0)) / cell)),
    )


def _attach_persistence(points: list[dict[str, Any]], config: MotionAwareConfig) -> None:
    frame_sets: dict[tuple[str, int, int, int], set[int]] = {}
    for point in points:
        if point.get("pred_class") not in {"structure", "floor"}:
            continue
        if point.get("motion_label") not in {"static_world", "ambiguous_motion", "unknown"}:
            continue
        frame_sets.setdefault(_grid_key(point, config), set()).add(int(point.get("source_frame_num", 0)))
    for point in points:
        count = len(frame_sets.get(_grid_key(point, config), set()))
        point["temporal_persistence_frames"] = int(count)


def build_motion_aware_window(frames: list[FramePacket], config: MotionAwareConfig | None = None) -> MotionAwareWindow:
    """Create an ego-compensated, Doppler-annotated point window for aggregation."""

    config = config or MotionAwareConfig()
    frames = sorted(frames[-int(config.window_frames):], key=lambda item: item.timestamp_us)
    if not frames:
        return MotionAwareWindow([], EgoMotionEstimate(False, 0.0, 0.0, 0.0, 0, 0, float("inf"), 0.0), {})

    estimates = [estimate_ego_velocity(frame.points, config) for frame in frames]
    poses = _integrate_frame_poses(frames, estimates)
    anchor = frames[-1]
    anchor_pose = poses.get(int(anchor.frame_num), np.zeros(2, dtype=np.float64))
    anchor_ego = estimates[-1]

    compensated: list[dict[str, Any]] = []
    for frame, ego in zip(frames, estimates):
        frame_pose = poses.get(int(frame.frame_num), anchor_pose)
        delta = anchor_pose - frame_pose
        annotated = annotate_ego_doppler(frame.points, ego, config)
        for point in annotated:
            out = dict(point)
            out["source_frame_num"] = int(frame.frame_num)
            out["anchor_frame_num"] = int(anchor.frame_num)
            out["motion_compensated"] = True
            out["x"] = float(out.get("x", 0.0)) - float(delta[0])
            out["y"] = float(out.get("y", 0.0)) - float(delta[1])
            compensated.append(out)

    _attach_persistence(compensated, config)

    kept: list[dict[str, Any]] = []
    suppressed = 0
    anchor_frame = int(anchor.frame_num)
    for point in compensated:
        cls = str(point.get("pred_class", ""))
        label = str(point.get("motion_label", "unknown"))
        is_anchor = int(point.get("source_frame_num", -1)) == anchor_frame
        y = float(point.get("y", 0.0))
        persist = int(point.get("temporal_persistence_frames", 0))

        keep = False
        if cls == "human":
            keep = is_anchor
        elif cls in {"structure", "floor"}:
            if y > float(config.max_history_y_m):
                keep = False
            elif y <= float(config.immediate_hazard_y_m) and is_anchor:
                keep = label != "dynamic_or_ghost"
            elif persist >= int(config.min_persist_frames) and label in {"static_world", "ambiguous_motion", "unknown"}:
                keep = True
            elif is_anchor and label == "static_world":
                keep = True
            elif is_anchor and config.keep_ambiguous_current and label == "ambiguous_motion":
                keep = True
        else:
            keep = is_anchor

        if keep:
            kept.append(point)
        else:
            suppressed += 1

    diagnostics = {
        "window_frames": int(len(frames)),
        "input_points": int(sum(len(frame.points) for frame in frames)),
        "output_points": int(len(kept)),
        "suppressed_points": int(suppressed),
        "anchor_frame": int(anchor.frame_num),
        "ego_available_frames": int(sum(1 for est in estimates if est.available)),
        "anchor_ego": anchor_ego.as_payload(),
        "persistent_points": int(sum(1 for p in kept if int(p.get("temporal_persistence_frames", 0)) >= int(config.min_persist_frames))),
        "dynamic_or_ghost_points": int(sum(1 for p in compensated if p.get("motion_label") == "dynamic_or_ghost")),
    }
    return MotionAwareWindow(kept, anchor_ego, diagnostics)


def aggregate_motion_aware_scene(
    frames: list[FramePacket],
    aggregate_scene: Callable[..., dict[str, Any]],
    config: MotionAwareConfig | None = None,
) -> tuple[dict[str, Any], MotionAwareWindow]:
    """Run the existing scene aggregator on a motion-aware point window."""

    window = build_motion_aware_window(frames, config)
    scene = aggregate_scene(
        window.points,
        window_frames=window.diagnostics.get("window_frames", 1),
        ego_velocity_mps=window.ego_motion.as_payload() if window.ego_motion.available else None,
    )
    scene.setdefault("evidence_channels", {}).setdefault("spatiotemporal", {})
    scene["evidence_channels"]["spatiotemporal"].update(window.diagnostics)
    scene["spatiotemporal"] = window.diagnostics
    return scene, window
