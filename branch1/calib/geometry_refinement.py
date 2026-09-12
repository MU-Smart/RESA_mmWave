#!/usr/bin/env python3
"""Stage 5: radar<->depth geometry refinement (simple projective residual).

See docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md (Stage 5 of that plan's staged
sequence) and the reference document's section 8:

    "Radar-to-depth geometry can refine the spatial transform after
    Doppler/trajectory alignment is stable... A simple projective residual
    is z(p_D) - D(u,v). A better refinement in structured indoor scenes is a
    point-to-plane residual against a local RGB-D surface normal."

This module implements the SIMPLE projective residual only -- the reference
document explicitly frames point-to-plane as "a better refinement", not a
requirement, and this is a refinement stage, not the initialization
mechanism (the module docstring's own framing). Point-to-plane (surface
normal estimation from the local depth patch) is a documented future
upgrade, not a corner cut -- see docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md
open items.

Residual model, reusing this repo's existing "aligned depth in the color
pixel grid" convention (spatiotemporal_calibrate.py::_backproject already
assumes this -- CameraIntrinsics there is RGB/color intrinsics, and depth
images are indexed by nearest color frame via DepthLookup):

    p_C = R_cr @ p_R + t_cr                        (candidate radar->camera)
    (u, v) = project(p_C, color intrinsics)
    z_pred = p_C.z
    z_meas = depth_image[v, u]                      (aligned depth, meters)
    r_G = z_pred - z_meas

This is a REFINEMENT on top of Stage 4's (R_cr, t_cr) -- not a fresh solve.
Time offset is held fixed at the Stage 4 value (radar<->camera geometry
does not depend on temporal alignment the way the Doppler/velocity
residuals do; re-solving beta here would conflate two different questions).
A tight prior toward the Stage 4 input keeps a small or noisy depth
correspondence set from dragging the transform far from an already-
converged Doppler solution -- see --geometry-warmstart-*-weight.

Design history (both bugs found by this module's own synthetic test, not
corner cuts -- see the test file this module ships alongside):

1. A first version re-derived correspondence VALIDITY (in-bounds
   projection, valid depth) on every solver iteration using the CURRENT
   candidate (R, t). That makes the residual vector's length change as
   points drift in and out of the image bounds between iterations --
   scipy.optimize.least_squares requires a fixed-length residual vector (it
   maintains a fixed-shape Jacobian), so this crashed partway through a
   solve with a shape-mismatch error once real depth data pushed some
   correspondences near an image edge.

2. The fix for (1) was to freeze BOTH membership and the measured depth
   value at the input (Stage 4) transform, recomputing only z_pred from the
   candidate transform each iteration. That fixed the crash, but
   accidentally made translation.x and translation.y COMPLETELY
   unobservable: p_c.z = (R @ p_radar + t)[2] depends only on t_z (t_x, t_y
   only ever affect p_c.x/p_c.y, never p_c.z, for ANY rotation or surface
   geometry) -- and with the measured depth also frozen, the objective's
   data term ends up with literally zero dependence on t_x/t_y. This is a
   correctness regression, not a hard test case: real point-to-surface
   depth residuals get their lateral-translation sensitivity from
   RE-PROJECTING at each iterate (t_x/t_y shift where in the image you
   sample, and a surface with any depth gradient returns a different value
   there) -- freezing the sample location severs exactly that coupling.

The correct design (implemented here): freeze only MEMBERSHIP (which
correspondences participate) and CACHE each participating correspondence's
full depth IMAGE once (avoiding repeated disk reads across iterations, see
DepthLookup.depth_at in spatiotemporal_calibrate.py). Every iteration then
re-projects using the CURRENT candidate (R, t) and re-samples the CACHED
image at the (clamped-to-bounds, never rejected) pixel location. This keeps
the residual vector's length fixed across the whole solve (clamping instead
of rejecting) while preserving genuine sensitivity to all 6 DOF, including
t_x/t_y.

Anti-circularity note (same as doppler_lever_arm.py): correspondences come
from Stage 2's RANSAC-accepted static radar detections only, never a
Branch 1 directness/ghost classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 10.0


@dataclass(slots=True)
class GeometryCorrespondence:
    p_radar: np.ndarray  # (3,) radar-frame xyz
    weight: float
    depth_image: np.ndarray  # cached full depth image (meters) for this detection's frame
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def gather_geometry_correspondences(
    session_data: Sequence[Any],
    warm_rotation: np.ndarray,
    warm_translation: np.ndarray,
    *,
    max_per_session: int,
    seed: int = 29,
) -> list[GeometryCorrespondence]:
    """Decides correspondence MEMBERSHIP once, using the input (Stage 4)
    transform for the initial validity check, and caches each kept
    correspondence's depth image (so the per-iteration objective never
    touches disk). See module docstring's design-history note for why
    membership is frozen but depth SAMPLING is not.

    session_data items are duck-typed (`.radar_motion.estimates`,
    `.depth_lookup` -- exactly spatiotemporal_calibrate.py's
    SessionMotionData, not imported here to avoid a circular import)."""
    out: list[GeometryCorrespondence] = []
    rng = np.random.default_rng(seed)
    for data in session_data:
        depth_lookup = getattr(data, "depth_lookup", None)
        if depth_lookup is None:
            continue
        intr = depth_lookup.intrinsics
        depth_image_cache: dict[int, np.ndarray] = {}
        session_candidates: list[GeometryCorrespondence] = []
        for est in data.radar_motion.estimates:
            accepted_xyz = est.accepted_xyz
            if len(accepted_xyz) == 0:
                continue
            snr = est.accepted_snr
            weights = np.clip(snr / max(float(np.median(snr)), 1e-3), 0.1, 8.0)
            ts = int(est.timestamp_us)
            for xyz, w in zip(accepted_xyz, weights):
                if _project_and_sample(warm_rotation, warm_translation, xyz, depth_lookup, ts) is None:
                    continue
                if ts not in depth_image_cache:
                    depth_img = depth_lookup.depth_at(ts)
                    if depth_img is None:
                        continue
                    depth_image_cache[ts] = depth_img
                session_candidates.append(
                    GeometryCorrespondence(
                        p_radar=xyz,
                        weight=float(w),
                        depth_image=depth_image_cache[ts],
                        fx=intr.fx,
                        fy=intr.fy,
                        cx=intr.cx,
                        cy=intr.cy,
                        width=intr.width,
                        height=intr.height,
                    )
                )
        if not session_candidates:
            continue
        if len(session_candidates) > max_per_session:
            idx = rng.choice(len(session_candidates), size=max_per_session, replace=False)
            session_candidates = [session_candidates[i] for i in idx]
        out.extend(session_candidates)
    return out


def _project_and_sample(
    rotation: np.ndarray,
    translation: np.ndarray,
    p_radar: np.ndarray,
    depth_lookup: Any,
    timestamp_us: int,
) -> tuple[float, float] | None:
    """Returns (predicted_z, measured_z), or None if out of bounds/invalid.
    Used ONLY for the one-time membership decision in
    gather_geometry_correspondences -- never inside the per-iteration
    objective (see _reproject_and_sample_cached for that)."""
    p_c = rotation @ p_radar + translation
    if p_c[2] <= MIN_VALID_DEPTH_M or p_c[2] >= MAX_VALID_DEPTH_M:
        return None
    intr = depth_lookup.intrinsics
    u = intr.fx * p_c[0] / p_c[2] + intr.cx
    v = intr.fy * p_c[1] / p_c[2] + intr.cy
    u_px = int(round(u))
    v_px = int(round(v))
    if u_px < 0 or u_px >= intr.width or v_px < 0 or v_px >= intr.height:
        return None
    depth_m = depth_lookup.depth_at(timestamp_us)
    if depth_m is None:
        return None
    measured = float(depth_m[v_px, u_px])
    if not np.isfinite(measured) or measured <= MIN_VALID_DEPTH_M or measured >= MAX_VALID_DEPTH_M:
        return None
    return float(p_c[2]), measured


def _reproject_and_sample_cached(
    rotation: np.ndarray,
    translation: np.ndarray,
    corr: GeometryCorrespondence,
) -> tuple[float, float | None]:
    """Re-projects using the CURRENT candidate transform and samples the
    CACHED depth image at the (clamped, never rejected) pixel location --
    this is what keeps t_x/t_y observable (see module docstring). Returns
    (z_pred, z_meas_or_None); z_meas is None only if the sampled depth
    value itself is invalid (rare in practice -- a real sensor hole/no
    return), in which case the caller contributes a neutral zero residual
    rather than dropping the correspondence (which would change the
    residual vector's length)."""
    p_c = rotation @ corr.p_radar + translation
    z_pred = float(p_c[2])
    z_safe = z_pred if abs(z_pred) > 1e-6 else 1e-6
    u = corr.fx * p_c[0] / z_safe + corr.cx
    v = corr.fy * p_c[1] / z_safe + corr.cy
    u_px = int(np.clip(round(u), 0, corr.width - 1))
    v_px = int(np.clip(round(v), 0, corr.height - 1))
    measured = float(corr.depth_image[v_px, u_px])
    if not np.isfinite(measured) or measured <= MIN_VALID_DEPTH_M or measured >= MAX_VALID_DEPTH_M:
        return z_pred, None
    return z_pred, measured


def refine_with_geometry(
    session_data: Sequence[Any],
    doppler_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """Stage 5 refinement, single-shot: gathers correspondences from
    `session_data` (each item's `.depth_lookup` must still be able to read
    its depth files -- i.e. they must not have been cleared from local
    disk yet) and solves in one call.

    For a rolling-batch driver where raw session data (including depth
    files) is downloaded, processed, and deleted in batches -- so no single
    point in time has every session's depth files simultaneously present --
    call `gather_geometry_correspondences` once per batch instead (while
    that batch's files are still on disk), concatenate the returned lists
    across batches, then call `solve_geometry_refinement_from_correspondences`
    once at the end. See docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md's
    notebook design for why this split exists."""
    warm_rotation = np.asarray(doppler_result["R_radar_to_camera"], dtype=np.float64)
    warm_translation = np.asarray(doppler_result["t_radar_to_camera"], dtype=np.float64)

    correspondences = gather_geometry_correspondences(
        session_data, warm_rotation, warm_translation, max_per_session=int(args.geometry_max_detections_per_session)
    )
    return solve_geometry_refinement_from_correspondences(correspondences, doppler_result, args)


def solve_geometry_refinement_from_correspondences(
    correspondences: list[GeometryCorrespondence],
    doppler_result: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """The solve half of Stage 5, decoupled from correspondence gathering
    (see refine_with_geometry's docstring for why -- rolling-batch drivers
    must gather per-batch, while depth files are still on disk, then call
    this once with the accumulated list after all batches are done and
    cleared)."""
    warm_rotation = np.asarray(doppler_result["R_radar_to_camera"], dtype=np.float64)
    warm_translation = np.asarray(doppler_result["t_radar_to_camera"], dtype=np.float64)

    if len(correspondences) < int(args.geometry_min_correspondences):
        return _fallback_result(
            doppler_result,
            reason=f"insufficient_geometry_correspondences ({len(correspondences)} < {args.geometry_min_correspondences})",
            n_correspondences=len(correspondences),
        )

    weight_arr = np.asarray([c.weight for c in correspondences], dtype=np.float64)

    def _residuals_data_term(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        values = np.empty(len(correspondences), dtype=np.float64)
        for i, corr in enumerate(correspondences):
            z_pred, measured = _reproject_and_sample_cached(rotation, translation, corr)
            values[i] = 0.0 if measured is None else (z_pred - measured)
        return weight_arr * values

    def objective(vector: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(vector[:3]).as_matrix()
        translation = vector[3:6]
        residuals = _residuals_data_term(rotation, translation)
        rot_delta = Rotation.from_matrix(rotation @ warm_rotation.T).as_rotvec()
        residuals = np.concatenate([residuals, float(args.geometry_warmstart_rotation_weight) * rot_delta])
        residuals = np.concatenate(
            [residuals, float(args.geometry_warmstart_translation_weight) * (translation - warm_translation)]
        )
        return residuals

    x0 = np.concatenate([Rotation.from_matrix(warm_rotation).as_rotvec(), warm_translation])
    lsq = least_squares(
        objective,
        x0,
        loss="huber",
        f_scale=float(args.geometry_loss_scale_m),
        x_scale="jac",
        max_nfev=int(args.max_nfev),
    )

    est_rotation = Rotation.from_rotvec(lsq.x[:3]).as_matrix()
    est_translation = lsq.x[3:6].astype(np.float64)

    final_residuals = _residuals_data_term(est_rotation, est_translation)
    residual_rms = float(np.sqrt(np.mean(final_residuals**2))) if len(final_residuals) else float("inf")

    fallback_reason = "geometry_result_accepted"
    use_doppler_input = False
    if not bool(lsq.success):
        fallback_reason = "least_squares_failed"
        use_doppler_input = True
    elif not np.isfinite(residual_rms) or residual_rms > float(args.geometry_max_residual_rms_m):
        fallback_reason = "residual_rms_too_high"
        use_doppler_input = True

    accepted_rotation = warm_rotation if use_doppler_input else est_rotation
    accepted_translation = warm_translation if use_doppler_input else est_translation

    return {
        "method": "geometry_refinement_stage5",
        "R_radar_to_camera": accepted_rotation.tolist(),
        "t_radar_to_camera": accepted_translation.tolist(),
        "translation_magnitude_m": float(np.linalg.norm(accepted_translation)),
        # geometry refinement holds time fixed -- carried through unchanged
        "time_offset_ms": doppler_result["time_offset_ms"],
        "estimated_R_radar_to_camera": est_rotation.tolist(),
        "estimated_t_radar_to_camera": est_translation.tolist(),
        "doppler_input": {
            "R_radar_to_camera": warm_rotation.tolist(),
            "t_radar_to_camera": warm_translation.tolist(),
        },
        "fallback": {"use_doppler_input": bool(use_doppler_input), "reason": fallback_reason},
        "solver": {
            "success": bool(lsq.success),
            "message": str(lsq.message),
            "cost": float(lsq.cost),
            "nfev": int(lsq.nfev),
            "loss": "huber",
            "loss_scale_m": float(args.geometry_loss_scale_m),
        },
        "n_correspondences": int(len(correspondences)),
        "residual_rms_m": residual_rms,
    }


def _fallback_result(doppler_result: dict[str, Any], *, reason: str, n_correspondences: int) -> dict[str, Any]:
    return {
        "method": "geometry_refinement_stage5",
        "R_radar_to_camera": doppler_result["R_radar_to_camera"],
        "t_radar_to_camera": doppler_result["t_radar_to_camera"],
        "translation_magnitude_m": doppler_result.get("translation_magnitude_m"),
        "time_offset_ms": doppler_result["time_offset_ms"],
        "fallback": {"use_doppler_input": True, "reason": reason},
        "n_correspondences": int(n_correspondences),
        "residual_rms_m": float("inf"),
    }
