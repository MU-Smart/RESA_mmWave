#!/usr/bin/env python3
"""Stage 4: per-detection Doppler residual with the full lever-arm term.

See docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md (Stage 4 of that plan's staged
sequence) and the reference document's section 7.1/12.1 pseudocode.

Upgrades Stage 3's per-frame-aggregated-velocity residual
(spatiotemporal_calibrate.py::solve_spatiotemporal) to use every
RANSAC-accepted radar detection individually:

    v_R_R(t) = R_cr^T @ (v_C(t) + omega_C(t) x t_cr)
    doppler_pred_j = dot(u_j, v_R_R(radar_timestamp_j + beta))
    r_j = accepted_doppler_mps_j - doppler_pred_j

This is the SAME kinematic model as Stage 3 (and as Wise et al.'s Eq. 16) --
Stage 4's actual contribution is using every accepted detection as its own
residual (far more data per frame) instead of first collapsing each frame to
one RANSAC-fitted velocity vector and discarding everything else. Stage 4 is
warm-started from Stage 3's converged (R_cr, t_cr, beta), per the reference
document's own recommended staged sequence (its section 11 table: Stage 3
output feeds Stage 4).

Anti-circularity note (reference document section 8.1): "accepted" here
means RANSAC-inlier static returns only, from
spatiotemporal_calibrate.py::estimate_radar_frame_velocity's own per-frame
RANSAC (Stage 2) -- never a Branch 1 directness/ghost classifier. This repo
does not yet have a directness classifier calibrated against this rig
generation, and depending on one here would create exactly the circular
dependency the reference document warns against.

Deliberate parameterization choice (documented, not an oversight): this
solves for (R_cr, t_cr) -- radar-to-CAMERA -- not (R_IR, t_IR) -- radar-to-
IMU -- even though docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md section 2.1
discusses the latter as appealing (T_DI/T_CD become fixed known constants,
shrinking the solve). The reason: this repo's continuous linear-velocity
source (RGB-D odometry, CameraTrajectory) and its validation ground truth
(config/radar_camera_extrinsics.json) are BOTH already expressed at the
camera origin. Re-deriving v_I(t) from v_C(t) would require rigid-body
velocity transport through the depth<->color and depth<->gyro lever arms
(~13-15mm each, from config/d435i_factory_extrinsics.json) -- doable in
principle, but it adds a fragile extra sign-convention-sensitive step for a
correction on the order of a few mm/s (omega * lever_arm, at typical
wheelchair angular rates), well below the RGB-D odometry's own noise floor.
Revisit if/when translation precision becomes the actual bottleneck.

Performance note: when time_mode == "fixed" (the default and the v1
recommendation from the reference document's header -- fix a single offset
rather than jointly solving clock scale/offset per iteration), v_C(t) and
omega_C(t) depend only on each detection's OWN fixed query timestamp, not on
the optimization variables -- so they are precomputed ONCE before calling
least_squares, not re-evaluated every Levenberg-Marquardt iteration. When
time_mode == "solve", beta is a free parameter and this cache is invalid;
that path re-evaluates trajectory interpolation per iteration and is
correspondingly slower -- use --max-doppler-detections to bound runtime if
so, or prefer "fixed" for large pooled datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class DopplerDetectionPool:
    unit_vectors_r: np.ndarray  # (N, 3) bearing unit vectors, radar frame
    radial_mps: np.ndarray  # (N,) sign-adjusted measured radial velocity
    weights: np.ndarray  # (N,) SNR-based only -- see module docstring anti-circularity note
    query_timestamps_us: np.ndarray  # (N,) int64, radar detection timestamp pre-offset
    session_index: np.ndarray  # (N,) int, index into the session_data sequence passed in
    provenance: list[tuple[str, int]]  # (session_name, frame_num), for diagnostics only

    def __len__(self) -> int:
        return int(len(self.radial_mps))


def gather_doppler_detections(
    session_data: Sequence[Any],
    *,
    max_detections: int | None = None,
    seed: int = 13,
) -> DopplerDetectionPool:
    """Pool RANSAC-accepted per-detection radar observations across sessions.

    `session_data` items are duck-typed: each needs `.session_name` and
    `.radar_motion.estimates` (a sequence of objects with `.timestamp_us`,
    `.frame_num`, `.accepted_xyz`, `.accepted_doppler_mps`, `.accepted_snr`
    -- exactly spatiotemporal_calibrate.py's SessionMotionData/RadarMotion/
    RadarVelocityEstimate, imported nowhere here to avoid a circular import;
    any object satisfying this shape works).
    """
    unit_chunks: list[np.ndarray] = []
    radial_chunks: list[np.ndarray] = []
    snr_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    session_idx_chunks: list[np.ndarray] = []
    provenance: list[tuple[str, int]] = []

    for session_idx, data in enumerate(session_data):
        for est in data.radar_motion.estimates:
            accepted_xyz = est.accepted_xyz
            if len(accepted_xyz) == 0:
                continue
            ranges = np.linalg.norm(accepted_xyz, axis=1)
            valid = ranges > 1e-3
            if not np.any(valid):
                continue
            n = int(valid.sum())
            unit_chunks.append(accepted_xyz[valid] / ranges[valid, None])
            radial_chunks.append(est.accepted_doppler_mps[valid])
            snr_chunks.append(est.accepted_snr[valid])
            ts_chunks.append(np.full(n, int(est.timestamp_us), dtype=np.int64))
            session_idx_chunks.append(np.full(n, session_idx, dtype=np.int64))
            provenance.extend((data.session_name, int(est.frame_num)) for _ in range(n))

    if not unit_chunks:
        return DopplerDetectionPool(
            unit_vectors_r=np.zeros((0, 3)),
            radial_mps=np.zeros(0),
            weights=np.zeros(0),
            query_timestamps_us=np.zeros(0, dtype=np.int64),
            session_index=np.zeros(0, dtype=np.int64),
            provenance=[],
        )

    unit_vectors_r = np.vstack(unit_chunks)
    radial_mps = np.concatenate(radial_chunks)
    snr = np.concatenate(snr_chunks)
    query_timestamps_us = np.concatenate(ts_chunks)
    session_index = np.concatenate(session_idx_chunks)
    weights = np.clip(snr / max(float(np.median(snr)), 1e-3), 0.1, 8.0)

    n_total = len(radial_mps)
    if max_detections is not None and n_total > max_detections:
        rng = np.random.default_rng(seed)
        keep = rng.choice(n_total, size=max_detections, replace=False)
        keep.sort()
        unit_vectors_r = unit_vectors_r[keep]
        radial_mps = radial_mps[keep]
        weights = weights[keep]
        query_timestamps_us = query_timestamps_us[keep]
        session_index = session_index[keep]
        provenance = [provenance[i] for i in keep]

    return DopplerDetectionPool(
        unit_vectors_r=unit_vectors_r,
        radial_mps=radial_mps,
        weights=weights,
        query_timestamps_us=query_timestamps_us,
        session_index=session_index,
        provenance=provenance,
    )


def _precompute_body_motion(
    session_data: Sequence[Any],
    pool: DopplerDetectionPool,
    offset_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    """v_C(t), omega_C(t) for every pooled detection at a FIXED offset.

    Valid only while offset_us is held fixed across the optimization (see
    module docstring's performance note). Each session supplies its own
    camera_trajectory / angular_velocity_at (the latter already
    IMU-vs-RGB-D-selected upstream, per --omega-source)."""
    n = len(pool)
    v_c = np.zeros((n, 3), dtype=np.float64)
    omega_c = np.zeros((n, 3), dtype=np.float64)
    # group by session to avoid repeated attribute lookups
    for session_idx in np.unique(pool.session_index):
        data = session_data[int(session_idx)]
        mask = pool.session_index == session_idx
        ts = pool.query_timestamps_us[mask] + int(offset_us)
        v_c[mask] = np.asarray(
            [data.camera_trajectory.linear_velocity_camera_at(int(t)) for t in ts], dtype=np.float64
        )
        omega_c[mask] = np.asarray(
            [data.angular_velocity_at(int(t)) for t in ts], dtype=np.float64
        )
    return v_c, omega_c


def _predicted_radial(
    rotation: np.ndarray,
    translation: np.ndarray,
    unit_vectors_r: np.ndarray,
    v_c: np.ndarray,
    omega_c: np.ndarray,
) -> np.ndarray:
    lever = np.cross(omega_c, translation.reshape(1, 3))
    v_body = v_c + lever
    v_radar = (rotation.T @ v_body.T).T
    return np.einsum("ij,ij->i", unit_vectors_r, v_radar)


def solve_doppler_lever_arm(
    session_data: Sequence[Any],
    static_prior: Any,
    warm_start: dict[str, Any],
    args: Any,
) -> dict[str, Any]:
    """Stage 4 solve. `static_prior` duck-types spatiotemporal_calibrate.py's
    StaticPrior (`.loaded`, `.rotation`, `.translation_m`). `warm_start` is
    Stage 3's result dict (must carry `R_radar_to_camera`, `t_radar_to_camera`,
    `time_offset_ms`) -- used both as x0 and as the fallback if Stage 4
    doesn't converge or degrades on held-out residual."""
    pool = gather_doppler_detections(session_data, max_detections=args.max_doppler_detections)

    warm_rotation = np.asarray(warm_start["R_radar_to_camera"], dtype=np.float64)
    warm_translation = np.asarray(warm_start["t_radar_to_camera"], dtype=np.float64)
    warm_offset_us = int(round(float(warm_start["time_offset_ms"]) * 1000.0))

    if len(pool) < int(args.min_doppler_detections):
        return _fallback_result(
            warm_start,
            reason=f"insufficient_doppler_detections ({len(pool)} < {args.min_doppler_detections})",
            pool_size=len(pool),
        )

    rot_prior = static_prior.rotation
    trans_prior = static_prior.translation_m
    time_prior_s = warm_offset_us * 1e-6

    x0 = np.concatenate([Rotation.from_matrix(warm_rotation).as_rotvec(), warm_translation, [time_prior_s]])

    lower = np.full(7, -np.inf, dtype=np.float64)
    upper = np.full(7, np.inf, dtype=np.float64)
    if args.time_mode == "fixed":
        lower[6] = time_prior_s - 1e-9
        upper[6] = time_prior_s + 1e-9
        v_c_cached, omega_c_cached = _precompute_body_motion(session_data, pool, warm_offset_us)
    else:
        lower[6] = -float(args.max_offset_ms) / 1000.0
        upper[6] = float(args.max_offset_ms) / 1000.0
        v_c_cached = omega_c_cached = None  # re-evaluated per-iteration below

    def objective(vector: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(vector[:3]).as_matrix()
        translation = vector[3:6]
        if args.time_mode == "fixed":
            v_c, omega_c = v_c_cached, omega_c_cached
        else:
            offset_us = int(round(float(vector[6]) * 1e6))
            v_c, omega_c = _precompute_body_motion(session_data, pool, offset_us)
        predicted = _predicted_radial(rotation, translation, pool.unit_vectors_r, v_c, omega_c)
        residuals = pool.weights * (pool.radial_mps - predicted)
        if static_prior.loaded and args.rotation_prior_weight > 0.0:
            rot_delta = Rotation.from_matrix(rotation @ rot_prior.T).as_rotvec()
            residuals = np.concatenate([residuals, float(args.rotation_prior_weight) * rot_delta])
        if static_prior.loaded and args.translation_prior_weight > 0.0:
            residuals = np.concatenate([residuals, float(args.translation_prior_weight) * (translation - trans_prior)])
        if args.time_mode == "solve" and args.time_prior_weight > 0.0:
            residuals = np.concatenate([residuals, [float(args.time_prior_weight) * (float(vector[6]) - time_prior_s)]])
        return residuals

    # KNOWN LIMITATION, not fixed here -- --time-mode solve's offset
    # dimension is effectively non-functional, and this is inherited from
    # Stage 3's pre-existing solve_spatiotemporal (same architecture, same
    # bug), not something introduced in this module. Root cause, confirmed
    # via a direct unit test recovering a known synthetic 25ms time offset
    # (it silently recovered 0.0 ms instead): the objective rounds the
    # time-offset parameter to integer MICROSECONDS before using it
    # (int(round(offset_s * 1e6))), matching every other timestamp in this
    # codebase's epoch-microsecond convention -- and CameraTrajectory's own
    # _clamp() does the same int(...) cast internally regardless of what's
    # passed in. scipy's least_squares only supports a RELATIVE
    # finite-difference step (confirmed against scipy's own
    # _compute_absolute_step: even an explicit diff_step is multiplied by
    # sign(x0)*abs(x0), with NO absolute-step option) -- and that relative
    # step is on the order of 1e-8 to 1e-7 seconds for any offset in a
    # plausible +-200ms range, i.e. a small fraction of a microsecond.
    # Every finite-difference probe therefore rounds to the SAME integer
    # microsecond as the baseline, making the numerical Jacobian's
    # time-offset column exactly zero -- least_squares reports immediate,
    # spurious convergence without ever exploring the time dimension,
    # REGARDLESS of the starting offset value (not just x0==0 -- verified
    # this also fails starting from -0.015s and 0.003s). An explicit
    # diff_step does NOT fix this (scipy always rescales it by |x0|, which
    # reintroduces the same vanishing-step problem) -- confirmed by testing
    # before writing this comment. A real fix requires either evaluating
    # the objective at continuous (non-integer-microsecond) timestamps
    # throughout the trajectory/IMU interpolation chain (touching
    # CameraTrajectory._clamp, pre-existing Stage 1-3 code, out of scope
    # for this Stage 4 addition) or switching to an analytic/complex-step
    # Jacobian that bypasses the rounding. Use --time-mode fixed (the
    # default, and the reference document's own v1 recommendation) until
    # this is addressed -- see docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md
    # open items.
    lsq = least_squares(
        objective,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(args.doppler_loss_scale_mps),
        x_scale="jac",
        max_nfev=int(args.max_nfev),
    )

    est_rotation = Rotation.from_rotvec(lsq.x[:3]).as_matrix()
    est_translation = lsq.x[3:6].astype(np.float64)
    est_offset_us = int(round((time_prior_s if args.time_mode == "fixed" else float(lsq.x[6])) * 1e6))

    if args.time_mode == "fixed":
        v_c_final, omega_c_final = v_c_cached, omega_c_cached
    else:
        v_c_final, omega_c_final = _precompute_body_motion(session_data, pool, est_offset_us)
    predicted_final = _predicted_radial(est_rotation, est_translation, pool.unit_vectors_r, v_c_final, omega_c_final)
    residual_all = pool.radial_mps - predicted_final
    residual_rms = float(np.sqrt(np.mean(residual_all**2))) if len(residual_all) else float("inf")
    residual_mean = float(np.mean(np.abs(residual_all))) if len(residual_all) else float("inf")
    residual_max = float(np.max(np.abs(residual_all))) if len(residual_all) else float("inf")

    fallback_reason = "doppler_result_accepted"
    use_warm_start = False
    if not bool(lsq.success):
        fallback_reason = "least_squares_failed"
        use_warm_start = True
    elif not np.isfinite(residual_rms) or residual_rms > float(args.max_residual_rms_mps):
        fallback_reason = "residual_rms_too_high"
        use_warm_start = True

    accepted_rotation = warm_rotation if use_warm_start else est_rotation
    accepted_translation = warm_translation if use_warm_start else est_translation
    accepted_offset_us = warm_offset_us if use_warm_start else est_offset_us

    return {
        "method": "doppler_lever_arm_stage4",
        "R_radar_to_camera": accepted_rotation.tolist(),
        "t_radar_to_camera": accepted_translation.tolist(),
        "translation_magnitude_m": float(np.linalg.norm(accepted_translation)),
        "time_offset_ms": float(accepted_offset_us / 1000.0),
        "estimated_R_radar_to_camera": est_rotation.tolist(),
        "estimated_t_radar_to_camera": est_translation.tolist(),
        "warm_start": {
            "R_radar_to_camera": warm_rotation.tolist(),
            "t_radar_to_camera": warm_translation.tolist(),
            "time_offset_ms": float(warm_offset_us / 1000.0),
            "source_method": warm_start.get("method", "unknown"),
        },
        "fallback": {"use_warm_start": bool(use_warm_start), "reason": fallback_reason},
        "solver": {
            "success": bool(lsq.success),
            "message": str(lsq.message),
            "cost": float(lsq.cost),
            "nfev": int(lsq.nfev),
            "loss": "soft_l1",
            "loss_scale_mps": float(args.doppler_loss_scale_mps),
            "time_mode": args.time_mode,
        },
        "n_detections_pooled": int(len(pool)),
        "residual_rms_mps": residual_rms,
        "residual_mean_mps": residual_mean,
        "residual_max_mps": residual_max,
    }


def _fallback_result(warm_start: dict[str, Any], *, reason: str, pool_size: int) -> dict[str, Any]:
    return {
        "method": "doppler_lever_arm_stage4",
        "R_radar_to_camera": warm_start["R_radar_to_camera"],
        "t_radar_to_camera": warm_start["t_radar_to_camera"],
        "translation_magnitude_m": float(np.linalg.norm(np.asarray(warm_start["t_radar_to_camera"]))),
        "time_offset_ms": warm_start["time_offset_ms"],
        "warm_start": {
            "R_radar_to_camera": warm_start["R_radar_to_camera"],
            "t_radar_to_camera": warm_start["t_radar_to_camera"],
            "time_offset_ms": warm_start["time_offset_ms"],
            "source_method": warm_start.get("method", "unknown"),
        },
        "fallback": {"use_warm_start": True, "reason": reason},
        "n_detections_pooled": int(pool_size),
        "residual_rms_mps": float("inf"),
        "residual_mean_mps": float("inf"),
        "residual_max_mps": float("inf"),
    }
