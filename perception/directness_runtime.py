#!/usr/bin/env python3
"""Runtime ego-Doppler directness annotation for shadow-mode integration.

Annotates classified radar points with ego-Doppler residual statistics, per-point
directness scores, and frame-level reliability diagnostics.  This module is designed
to be called from both live `navigation_loop.py` and offline
`branch3_replay_simulation.py` with the same contract.

All thresholds and parameters are config-driven to support replay ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DirectnessConfig:
    doppler_sign: float = -1.0
    """Sign convention: -1 means raw Doppler is negative for approaching
    targets.  Multiply by this factor to convert to radial velocity in the
    range-rate sense (positive = receding)."""

    max_abs_doppler_mps: float = 2.5
    """Maximum absolute Doppler (m/s) considered for ego estimation."""

    min_ego_points: int = 8
    """Minimum candidate points required to attempt ego velocity estimation."""

    ransac_iterations: int = 64
    """Number of RANSAC iterations for robust ego-velocity fitting."""

    static_residual_mps: float = 0.25
    """Residual threshold (m/s) below which a point is a static-world inlier."""

    dynamic_residual_mps: float = 0.55
    """Residual threshold (m/s) above which a point is considered dynamic/ghost."""

    min_static_inlier_fraction: float = 0.25
    """Minimum fraction of candidate points that must be static inliers for
    ego estimate to be considered reliable."""

    max_ego_speed_mps: float = 2.5
    """Maximum plausible ego speed (m/s).  Estimates above this are rejected."""

    ego_sigma_floor_mps: float = 0.05
    """Minimum robust residual scale (m/s) to prevent division by zero."""

    p_ego_residual_scale_mps: float = 0.35
    """Residual MAD scale (m/s) used in the p_ego exponential decay term."""

    sector_coverage_min: float = 0.15
    """Minimum fraction of azimuth sectors that must contain candidate points
    for full sector-coverage credit."""

    n_azimuth_sectors: int = 6
    """Number of equal-width azimuth sectors for coverage estimation."""


# ---------------------------------------------------------------------------
# Ego velocity estimation (RANSAC least-squares on structure/floor points)
# ---------------------------------------------------------------------------


def _unit_xy(point: dict[str, Any]) -> tuple[float, float] | None:
    rng = float(np.hypot(float(point.get("x", 0.0)), float(point.get("y", 0.0))))
    if not np.isfinite(rng) or rng <= 1e-3:
        return None
    return float(point.get("x", 0.0)) / rng, float(point.get("y", 0.0)) / rng


def _solve_velocity(
    unit_vectors: np.ndarray,
    radial_mps: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if weights is None:
        solution, *_ = np.linalg.lstsq(unit_vectors, radial_mps, rcond=None)
        return solution.astype(np.float64)
    w = np.sqrt(np.clip(weights.astype(np.float64), 1e-6, None)).reshape(-1, 1)
    solution, *_ = np.linalg.lstsq(unit_vectors * w, radial_mps * w[:, 0], rcond=None)
    return solution.astype(np.float64)


def estimate_ego_velocity(
    points: list[dict[str, Any]],
    config: DirectnessConfig | None = None,
) -> dict[str, Any]:
    """Estimate 2D rig ego velocity from static-world radar Doppler returns.

    Returns a dict with keys:
        available, vx_mps, vy_mps, speed_mps,
        n_candidate_points, n_static_points,
        residual_mad_mps, inlier_fraction,
        sector_entropy, sector_coverage, p_ego
    """
    config = config or DirectnessConfig()

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

    n_candidates = len(candidates)

    if n_candidates < int(config.min_ego_points):
        return {
            "available": False,
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "speed_mps": 0.0,
            "n_candidate_points": n_candidates,
            "n_static_points": 0,
            "residual_mad_mps": float("inf"),
            "inlier_fraction": 0.0,
            "sector_entropy": 0.0,
            "sector_coverage": 0.0,
            "p_ego": 0.0,
        }

    unit_vectors = np.asarray([_unit_xy(p) for p in candidates], dtype=np.float64)
    radial = np.asarray(
        [float(config.doppler_sign) * float(p.get("doppler", 0.0)) for p in candidates],
        dtype=np.float64,
    )
    snr = np.asarray([float(p.get("snr", 1.0)) for p in candidates], dtype=np.float64)
    weights = np.clip(snr / max(float(np.nanmedian(snr)), 1e-3), 0.1, 8.0)

    rng = np.random.default_rng(101 + n_candidates)
    best_mask = np.zeros(n_candidates, dtype=bool)
    for _ in range(max(1, int(config.ransac_iterations))):
        if n_candidates >= 2:
            sample = rng.choice(n_candidates, size=2, replace=False)
        else:
            sample = np.arange(n_candidates)
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
    residual_mad = float(
        np.median(np.abs(use_residuals - np.median(use_residuals)))
    ) if len(use_residuals) else float("inf")
    speed = float(np.linalg.norm(velocity))
    inlier_fraction = float(inliers.sum() / max(1, n_candidates))

    # Sector coverage
    az = np.arctan2(
        np.asarray([float(p.get("x", 0.0)) for p in candidates]),
        np.maximum(np.asarray([float(p.get("y", 0.0)) for p in candidates]), 0.1),
    )
    n_sectors = int(config.n_azimuth_sectors)
    sector_bins = np.linspace(-np.pi / 2, np.pi / 2, n_sectors + 1)
    sector_counts = np.zeros(n_sectors, dtype=np.int32)
    for a in az:
        idx = np.digitize(a, sector_bins) - 1
        if 0 <= idx < n_sectors:
            sector_counts[idx] += 1
    sector_coverage = float(np.sum(sector_counts > 0)) / max(n_sectors, 1)
    sector_probs = sector_counts.astype(np.float64) / max(sector_counts.sum(), 1)
    sector_entropy = float(
        -np.sum(sector_probs[sector_probs > 0] * np.log(sector_probs[sector_probs > 0] + 1e-12))
        / max(np.log(n_sectors), 1e-12)
    )

    available = bool(
        int(inliers.sum()) >= int(config.min_ego_points)
        and speed <= float(config.max_ego_speed_mps)
        and inlier_fraction >= float(config.min_static_inlier_fraction)
        and np.isfinite(residual_mad)
    )

    # p_ego: continuous reliability in [0, 1]
    if available:
        p_ego = float(
            np.clip(inlier_fraction, 0.0, 1.0)
            * np.exp(-residual_mad / max(float(config.p_ego_residual_scale_mps), 1e-6))
            * np.clip(sector_coverage / max(float(config.sector_coverage_min), 1e-6), 0.0, 1.0)
        )
    else:
        p_ego = 0.0

    return {
        "available": available,
        "vx_mps": float(velocity[0]) if available else 0.0,
        "vy_mps": float(velocity[1]) if available else 0.0,
        "speed_mps": float(speed) if available else 0.0,
        "n_candidate_points": n_candidates,
        "n_static_points": int(inliers.sum()),
        "residual_mad_mps": round(residual_mad, 4) if np.isfinite(residual_mad) else float("inf"),
        "inlier_fraction": round(inlier_fraction, 4),
        "sector_entropy": round(sector_entropy, 4),
        "sector_coverage": round(sector_coverage, 4),
        "p_ego": round(p_ego, 4),
    }


def estimate_ego_velocity_no_class(
    points: list[dict[str, Any]],
    config: DirectnessConfig | None = None,
) -> dict[str, Any]:
    """Estimate 2D rig ego velocity before classification is available.

    This uses the same Doppler-RANSAC solver and reliability gates as
    ``estimate_ego_velocity()``, but treats every finite, in-range Doppler point
    as a candidate instead of requiring ``pred_class`` to be structure/floor.
    RANSAC provides the static-world consensus separation.
    """

    candidate_points: list[dict[str, Any]] = []
    for point in points:
        doppler = float(point.get("doppler", 0.0))
        if not np.isfinite(doppler):
            continue
        unit = _unit_xy(point)
        if unit is None:
            continue
        out = dict(point)
        out["pred_class"] = "structure"
        candidate_points.append(out)
    return estimate_ego_velocity(candidate_points, config)


# ---------------------------------------------------------------------------
# Per-point directness annotation
# ---------------------------------------------------------------------------


def annotate_directness(
    points: list[dict[str, Any]],
    ego: dict[str, Any],
    config: DirectnessConfig | None = None,
) -> list[dict[str, Any]]:
    """Annotate points with ego-Doppler directness fields.

    Adds to each point dict:
        ego_expected_doppler_mps
        ego_doppler_residual_mps
        ego_residual_abs_z
        ego_inlier_flag
        ego_reliable
        p_ego
        p_dir_doppler

    Returns a new list of annotated point dicts (input is not mutated).
    """
    config = config or DirectnessConfig()
    ego_available = bool(ego.get("available", False))
    vx = float(ego.get("vx_mps", 0.0))
    vy = float(ego.get("vy_mps", 0.0))
    residual_mad = float(ego.get("residual_mad_mps", 0.05))
    p_ego_val = float(ego.get("p_ego", 0.0))

    ego_sigma = max(residual_mad * 1.4826, float(config.ego_sigma_floor_mps))

    annotated: list[dict[str, Any]] = []
    for point in points:
        out = dict(point)
        unit = _unit_xy(out)
        doppler = float(out.get("doppler", 0.0))

        if not ego_available or unit is None or not np.isfinite(doppler):
            out.update({
                "ego_expected_doppler_mps": 0.0,
                "ego_doppler_residual_mps": 0.0,
                "ego_residual_abs_z": 0.0,
                "ego_inlier_flag": False,
                "ego_reliable": False,
                "p_ego": 0.0,
                "p_dir_doppler": 1.0,
            })
            annotated.append(out)
            continue

        expected = -(vx * unit[0] + vy * unit[1])
        residual = doppler - expected
        abs_z = abs(residual) / max(ego_sigma, 1e-6)
        inlier = abs(residual) <= float(config.static_residual_mps)
        # p_dir_doppler: Gaussian falloff, clipped at 4 sigma
        p_dir = float(np.exp(-0.5 * min(abs_z, 4.0) ** 2))

        out.update({
            "ego_expected_doppler_mps": round(float(expected), 4),
            "ego_doppler_residual_mps": round(float(residual), 4),
            "ego_residual_abs_z": round(float(abs_z), 4),
            "ego_inlier_flag": bool(inlier),
            "ego_reliable": ego_available,
            "p_ego": round(p_ego_val, 4),
            "p_dir_doppler": round(p_dir, 4),
        })
        annotated.append(out)

    return annotated


# ---------------------------------------------------------------------------
# Frame-level directness summary for evidence channels
# ---------------------------------------------------------------------------


def build_directness_evidence(
    points: list[dict[str, Any]],
    ego: dict[str, Any],
) -> dict[str, Any]:
    """Build a frame-level directness evidence summary dict.

    The output dict follows the runtime ``evidence_channels.directness``
    schema expected by ``branch3_replay_simulation.py`` debug snapshots.
    """
    human_pts = [p for p in points if p.get("pred_class") == "human"]
    struct_pts = [p for p in points if p.get("pred_class") == "structure"]

    human_residuals = [abs(float(p.get("ego_doppler_residual_mps", 0.0))) for p in human_pts]
    struct_residuals = [abs(float(p.get("ego_doppler_residual_mps", 0.0))) for p in struct_pts]
    human_p_dir = [float(p.get("p_dir_doppler", 1.0)) for p in human_pts]
    struct_p_dir = [float(p.get("p_dir_doppler", 1.0)) for p in struct_pts]
    dynamic_or_ghost = [p for p in points
                        if abs(float(p.get("ego_doppler_residual_mps", 0.0))) > 0.45]

    # Right sector (x > 0.35) human details
    right_human_pts = [p for p in human_pts if float(p.get("x", 0.0)) > 0.35]
    right_human_p_dir = [float(p.get("p_dir_doppler", 1.0)) for p in right_human_pts]

    return {
        "ego_available": bool(ego.get("available", False)),
        "p_ego": round(float(ego.get("p_ego", 0.0)), 4),
        "vx_mps": round(float(ego.get("vx_mps", 0.0)), 3),
        "vy_mps": round(float(ego.get("vy_mps", 0.0)), 3),
        "speed_mps": round(float(ego.get("speed_mps", 0.0)), 3),
        "candidate_points": int(ego.get("n_candidate_points", 0)),
        "static_points": int(ego.get("n_static_points", 0)),
        "static_inlier_fraction": round(float(ego.get("inlier_fraction", 0.0)), 4),
        "residual_mad_mps": round(float(ego.get("residual_mad_mps", 0.0)), 4),
        "sector_entropy": round(float(ego.get("sector_entropy", 0.0)), 4),
        "sector_coverage": round(float(ego.get("sector_coverage", 0.0)), 4),
        "n_dynamic_or_ghost": len(dynamic_or_ghost),
        "mean_p_dir_doppler": round(float(np.mean(human_p_dir)), 4) if human_p_dir else 0.0,
        "human_mean_ego_residual_mps": round(float(np.mean(human_residuals)), 4) if human_residuals else 0.0,
        "human_max_ego_residual_mps": round(float(np.max(human_residuals)), 4) if human_residuals else 0.0,
        "human_mean_ego_residual_abs_z": round(
            float(np.mean([float(p.get("ego_residual_abs_z", 0.0)) for p in human_pts])), 4
        ) if human_pts else 0.0,
        "human_mean_p_dir_doppler": round(float(np.mean(human_p_dir)), 4) if human_p_dir else 0.0,
        "structure_mean_p_dir_doppler": round(float(np.mean(struct_p_dir)), 4) if struct_p_dir else 0.0,
        "right_sector_human_count": len(right_human_pts),
        "right_sector_mean_p_dir_doppler": round(float(np.mean(right_human_p_dir)), 4) if right_human_p_dir else 0.0,
    }
