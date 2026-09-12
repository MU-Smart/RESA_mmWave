#!/usr/bin/env python3
"""Static AprilTag + trihedral-reflector radar-camera extrinsic calibration.

This reproduces the target-based method that produced the current
config/radar_camera_extrinsics.json (method: "SVD_Kabsch_RANSAC + TRF_t_only +
IRLS_t_only + Joint_Rt_IRLS", tagStandard41h12, tag_size 0.2032 m, reflector
recessed 0.1016 m behind the tag face along its normal) -- for use when the
physical radar/camera mount has changed slightly and the existing transform
should be treated as a *prior*, not thrown away.

Per session:
  1. For every radar frame, find the nearest color-video frame by timestamp.
  2. Detect the AprilTag (tagStandard41h12) in that frame, get its pose via
     pupil_apriltags (fx/fy/cx/cy PnP), and offset along the tag normal (away
     from the camera, into the mount) by --reflector-offset-m to get the
     trihedral reflector's 3-D position in the camera frame.
  3. Predict the reflector's radar-frame position using the *prior* extrinsics
     and gate the frame's radar detections around that prediction. The
     reflector's huge RCS means the highest-SNR detection in the gate should
     be the true reflector return; this is the radar-side correspondence.

Across all sessions, the (radar_xyz, camera_xyz) correspondences are solved
with the same generic rigid-registration primitives used by
spatiotemporal_calibrate.py (`_rigid_transform`, `_ransac_rigid`: closed-form
Kabsch/SVD wrapped in RANSAC), then refined with a robust (soft_l1)
least_squares pass that is *regularized toward the prior* transform via
--rotation-prior-weight / --translation-prior-weight -- the requested "prior"
mechanism. If the fresh estimate isn't well supported (too few correspondences
or residual too high), the result falls back to the prior unchanged, exactly
like spatiotemporal_calibrate.py's fallback contract.

Requires pupil_apriltags (tagStandard41h12 is not in OpenCV's built-in ArUco
dictionaries, which only cover the classic 16h5/25h9/36h10/36h11 families).

Usage:
  python3 -m branch1.calib.target_calibrate \\
    --processing-root data/raw/sessions \\
    --static-prior config/radar_camera_extrinsics.json \\
    --output-dir branch1/calib/results/session_2026-08-27_rig_update \\
    --save-correspondences
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch1.calib.spatiotemporal_calibrate import (  # noqa: E402
    RadarFrame,
    StaticPrior,
    _load_timestamp_csv,
    _pointcloud_csv_path,
    _rigid_transform,
    _ransac_rigid,
    _read_json,
    _write_json,
    load_radar_frames,
    load_static_prior,
    rotation_to_euler_zyx_deg,
)

DEFAULT_STATIC_PRIOR = REPO_ROOT / "config" / "radar_camera_extrinsics.json"
DEFAULT_OUTPUT_DIR = THIS_FILE.parent / "results" / "target_calibration"


# ---------------------------------------------------------------------------
# Per-frame tag -> reflector geometry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TagObservation:
    video_frame_index: int
    tag_id: int
    decision_margin: float
    reflector_camera_xyz: np.ndarray


# ---------------------------------------------------------------------------
# Pose-flip rejection: the target rig is static within a session, so the
# tag's true orientation should barely move frame-to-frame. pupil_apriltags'
# monocular PnP solve has a well-known planar pose ambiguity for near-head-on
# square tags ("more than one new minima found") -- it silently picks one of
# two geometrically distinct branches per frame, which shows up as a sudden
# large jump in pose_R relative to neighboring frames even though the tag
# never moved. Reject any frame whose orientation jumps too far from the
# last-accepted frame's orientation; a rejected frame doesn't overwrite the
# reference, so isolated flips don't cascade into rejecting good frames that
# follow. (Depth-based pose verification was tried first -- see
# DEPTH_TAG_POSE_PLAN.md -- but the depth sensor doesn't return usable range
# data at the tag's own surface on this hardware/target, so it never
# resolves the ambiguity in practice; dropped in favor of this approach.)
# ---------------------------------------------------------------------------


def _pose_angular_diff_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a @ rotation_b.T
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def _camera_intrinsics_from_meta(session_dir: Path) -> tuple[float, float, float, float]:
    meta = _read_json(session_dir / "meta_data.json")
    rgb = meta["realsense_calibration"]["rgb"]
    return float(rgb["fx"]), float(rgb["fy"]), float(rgb["cx"]), float(rgb["cy"])


def detect_tag_observations(
    session_dir: Path,
    wanted_video_indices: set[int],
    *,
    tag_family: str,
    tag_size_m: float,
    reflector_offset_m: float,
    min_decision_margin: float,
    tag_id: int | None,
    max_pose_flip_deg: float = 20.0,
) -> tuple[dict[int, TagObservation], int]:
    """Returns (observations, n_flip_rejected)."""
    from pupil_apriltags import Detector  # local import: optional heavy dep

    fx, fy, cx, cy = _camera_intrinsics_from_meta(session_dir)
    # nthreads=1: pupil_apriltags' internal threading has an intermittent
    # native-crash race under concurrent detect() calls; single-threaded
    # detection is plenty fast for a few hundred frames per session.
    detector = Detector(families=tag_family, nthreads=1, quad_decimate=1.0)

    video_path = session_dir / f"{session_dir.name}_color.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    observations: dict[int, TagObservation] = {}
    if not wanted_video_indices:
        cap.release()
        return observations, 0
    last_wanted = max(wanted_video_indices)

    last_accepted_rotation: np.ndarray | None = None
    n_flip_rejected = 0

    frame_idx = 0
    while frame_idx <= last_wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in wanted_video_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = detector.detect(
                gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=tag_size_m
            )
            candidates = [r for r in results if tag_id is None or r.tag_id == tag_id]
            candidates = [r for r in candidates if r.decision_margin >= min_decision_margin]
            if candidates:
                best = max(candidates, key=lambda r: r.decision_margin)

                is_flip = (
                    last_accepted_rotation is not None
                    and _pose_angular_diff_deg(best.pose_R, last_accepted_rotation) > max_pose_flip_deg
                )
                if is_flip:
                    n_flip_rejected += 1
                else:
                    last_accepted_rotation = best.pose_R
                    # Tag +Z (pose_R[:, 2]) points away from the camera, into
                    # the mount -- verified empirically: camera-in-tag-frame
                    # z < 0. The trihedral reflector sits recessed behind the
                    # printed face along that same direction.
                    reflector = best.pose_t.reshape(3) + reflector_offset_m * best.pose_R[:, 2]
                    observations[frame_idx] = TagObservation(
                        video_frame_index=frame_idx,
                        tag_id=int(best.tag_id),
                        decision_margin=float(best.decision_margin),
                        reflector_camera_xyz=reflector,
                    )
        frame_idx += 1
    cap.release()
    return observations, n_flip_rejected


# ---------------------------------------------------------------------------
# Radar-side reflector correspondence (gated max-SNR pick)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Correspondence:
    session: str
    video_frame_index: int
    radar_frame_num: int
    radar_xyz: np.ndarray
    camera_xyz: np.ndarray
    radar_snr: float
    gate_distance_m: float
    tag_decision_margin: float


def gather_session_correspondences(
    session_dir: Path,
    static_prior: StaticPrior,
    args: argparse.Namespace,
) -> tuple[list[Correspondence], dict[str, Any]]:
    name = session_dir.name
    info: dict[str, Any] = {"session": name}

    csv_path = _pointcloud_csv_path(session_dir)
    if csv_path is None:
        info["error"] = "missing point-cloud CSV"
        return [], info
    radar_frames = load_radar_frames(session_dir)
    if not radar_frames:
        info["error"] = "no radar frames"
        return [], info

    color_ts_path = session_dir / f"{name}_color_timestamps.csv"
    if not color_ts_path.exists():
        info["error"] = "missing color timestamps csv"
        return [], info
    video_ts_us = np.asarray(_load_timestamp_csv(color_ts_path), dtype=np.int64)

    max_dt_us = int(args.max_time_diff_ms * 1000)
    radar_to_video_idx: dict[int, int] = {}
    for f in radar_frames:
        diffs = np.abs(video_ts_us - int(f.timestamp_us))
        best_i = int(np.argmin(diffs))
        if diffs[best_i] <= max_dt_us:
            radar_to_video_idx[f.frame_num] = best_i

    wanted = set(radar_to_video_idx.values())
    observations, n_flip_rejected = detect_tag_observations(
        session_dir,
        wanted,
        tag_family=args.tag_family,
        tag_size_m=args.tag_size_m,
        reflector_offset_m=args.reflector_offset_m,
        min_decision_margin=args.min_decision_margin,
        tag_id=args.tag_id,
        max_pose_flip_deg=args.max_pose_flip_deg,
    )
    info["video_frames_checked"] = len(wanted)
    info["tag_observations"] = len(observations)
    info["pose_flip_rejected"] = n_flip_rejected

    radar_by_frame = {f.frame_num: f for f in radar_frames}
    correspondences: list[Correspondence] = []
    for radar_frame_num, video_idx in radar_to_video_idx.items():
        obs = observations.get(video_idx)
        if obs is None:
            continue
        predicted_radar = static_prior.rotation.T @ (obs.reflector_camera_xyz - static_prior.translation_m)
        frame = radar_by_frame[radar_frame_num]
        best_det = None
        best_dist = float("inf")
        for det in frame.detections:
            dist = float(np.linalg.norm(det.xyz - predicted_radar))
            if dist <= args.gate_radius_m and det.snr > (best_det.snr if best_det is not None else -np.inf):
                best_det = det
                best_dist = dist
        if best_det is None:
            continue
        correspondences.append(
            Correspondence(
                session=name,
                video_frame_index=video_idx,
                radar_frame_num=radar_frame_num,
                radar_xyz=best_det.xyz,
                camera_xyz=obs.reflector_camera_xyz,
                radar_snr=best_det.snr,
                gate_distance_m=best_dist,
                tag_decision_margin=obs.decision_margin,
            )
        )
    info["correspondences"] = len(correspondences)
    return correspondences, info


# ---------------------------------------------------------------------------
# Solve: RANSAC Kabsch init, then prior-regularized robust refinement
# ---------------------------------------------------------------------------


def _residual_metrics(rotation: np.ndarray, translation: np.ndarray, radar_pts: np.ndarray, camera_pts: np.ndarray) -> tuple[float, float, float]:
    if len(radar_pts) == 0:
        return float("inf"), float("inf"), float("inf")
    predicted = radar_pts @ rotation.T + translation
    norms = np.linalg.norm(camera_pts - predicted, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.mean(norms)), float(np.max(norms))


def solve_target_calibration(
    correspondences: list[Correspondence],
    static_prior: StaticPrior,
    args: argparse.Namespace,
) -> dict[str, Any]:
    radar_pts = np.asarray([c.radar_xyz for c in correspondences], dtype=np.float64)
    camera_pts = np.asarray([c.camera_xyz for c in correspondences], dtype=np.float64)
    n_total = len(correspondences)

    if n_total >= 3:
        ransac_r, ransac_t, ransac_mask = _ransac_rigid(
            radar_pts, camera_pts, iterations=args.ransac_iterations, threshold_m=args.ransac_threshold_m
        )
        n_inliers = int(ransac_mask.sum())
    else:
        ransac_r, ransac_t = static_prior.rotation.copy(), static_prior.translation_m.copy()
        ransac_mask = np.zeros(n_total, dtype=bool)
        n_inliers = 0

    init_rotation = ransac_r if n_inliers >= 3 else static_prior.rotation
    init_translation = ransac_t if n_inliers >= 3 else static_prior.translation_m
    rot_prior = static_prior.rotation
    trans_prior = static_prior.translation_m

    # Refine against the RANSAC inlier subset only, not all correspondences.
    # With this dataset's ~40-50% outlier rate, including outliers in the
    # soft_l1 objective (even down-weighted) was pulling the refined fit
    # away from a RANSAC solution that already clears the acceptance
    # thresholds on its own -- refinement should tighten the inlier fit and
    # blend it with the prior, not re-fight the outliers RANSAC already
    # rejected.
    if n_inliers >= 3:
        refine_radar_pts = radar_pts[ransac_mask]
        refine_camera_pts = camera_pts[ransac_mask]
    else:
        refine_radar_pts = radar_pts
        refine_camera_pts = camera_pts

    def objective(vector: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(vector[:3]).as_matrix()
        translation = vector[3:6]
        if len(refine_radar_pts) == 0:
            residuals = np.array([1000.0], dtype=np.float64)
        else:
            predicted = refine_radar_pts @ rotation.T + translation
            residuals = (refine_camera_pts - predicted).reshape(-1)
        if static_prior.loaded and args.rotation_prior_weight > 0.0:
            rot_delta = Rotation.from_matrix(rotation @ rot_prior.T).as_rotvec()
            residuals = np.concatenate([residuals, float(args.rotation_prior_weight) * rot_delta])
        if static_prior.loaded and args.translation_prior_weight > 0.0:
            residuals = np.concatenate([residuals, float(args.translation_prior_weight) * (translation - trans_prior)])
        return residuals

    x0 = np.concatenate([Rotation.from_matrix(init_rotation).as_rotvec(), init_translation])
    lsq = least_squares(
        objective, x0, loss="soft_l1", f_scale=float(args.loss_scale_m),
        x_scale="jac", max_nfev=int(args.max_nfev),
    )
    est_rotation = Rotation.from_rotvec(lsq.x[:3]).as_matrix()
    est_translation = lsq.x[3:6].astype(np.float64)

    # Re-run inlier classification against the refined transform for reporting.
    if n_total > 0:
        residual_norms = np.linalg.norm(camera_pts - (radar_pts @ est_rotation.T + est_translation), axis=1)
        final_inlier_mask = residual_norms <= args.ransac_threshold_m
    else:
        final_inlier_mask = np.zeros(0, dtype=bool)
    n_final_inliers = int(final_inlier_mask.sum())
    if n_final_inliers >= 3:
        inlier_rms, inlier_mean, inlier_max = _residual_metrics(
            est_rotation, est_translation, radar_pts[final_inlier_mask], camera_pts[final_inlier_mask]
        )
    else:
        inlier_rms, inlier_mean, inlier_max = float("inf"), float("inf"), float("inf")

    fallback_reason = "target_result_accepted"
    use_static = False
    if not bool(lsq.success):
        fallback_reason = "least_squares_failed"
        use_static = static_prior.loaded
    elif n_final_inliers < int(args.min_correspondences):
        fallback_reason = "insufficient_correspondences"
        use_static = static_prior.loaded
    elif not np.isfinite(inlier_rms) or inlier_rms > float(args.max_inlier_rms_m):
        fallback_reason = "inlier_rms_too_high"
        use_static = static_prior.loaded

    accepted_rotation = static_prior.rotation if use_static else est_rotation
    accepted_translation = static_prior.translation_m if use_static else est_translation

    return {
        "method": "AprilTag_reflector_SVD_Kabsch_RANSAC_prior_regularized_TRF",
        "R_radar_to_camera": accepted_rotation.tolist(),
        "t_radar_to_camera": accepted_translation.tolist(),
        "euler_ZYX_deg": rotation_to_euler_zyx_deg(accepted_rotation),
        "translation_magnitude_m": float(np.linalg.norm(accepted_translation)),
        "estimated_R_radar_to_camera": est_rotation.tolist(),
        "estimated_t_radar_to_camera": est_translation.tolist(),
        "estimated_euler_ZYX_deg": rotation_to_euler_zyx_deg(est_rotation),
        "static_prior": {
            "loaded": bool(static_prior.loaded),
            "path": None if static_prior.path is None else str(static_prior.path),
            "R_radar_to_camera": static_prior.rotation.tolist(),
            "t_radar_to_camera": static_prior.translation_m.tolist(),
        },
        "fallback": {"use_static": bool(use_static), "reason": fallback_reason},
        "solver": {
            "success": bool(lsq.success),
            "message": str(lsq.message),
            "cost": float(lsq.cost),
            "nfev": int(lsq.nfev),
            "loss": "soft_l1",
            "loss_scale_m": float(args.loss_scale_m),
            "rotation_prior_weight": float(args.rotation_prior_weight),
            "translation_prior_weight": float(args.translation_prior_weight),
        },
        "n_total_correspondences": int(n_total),
        "n_ransac_inliers": int(n_inliers),
        "ransac_threshold_m": float(args.ransac_threshold_m),
        "ransac_iterations": int(args.ransac_iterations),
        "n_inliers": n_final_inliers,
        "n_outliers": int(n_total - n_final_inliers),
        "inlier_rms_m": inlier_rms,
        "inlier_mean_m": inlier_mean,
        "inlier_max_m": inlier_max,
        "apriltag_family": args.tag_family,
        "apriltag_size_m": float(args.tag_size_m),
        "reflector_depth_offset_m": float(args.reflector_offset_m),
        "tag_normal_offset": True,
        "gate_radius_m": float(args.gate_radius_m),
    }


def save_correspondences_csv(path: Path, correspondences: list[Correspondence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "session", "video_frame_index", "radar_frame_num",
            "radar_x", "radar_y", "radar_z",
            "camera_x", "camera_y", "camera_z",
            "radar_snr", "gate_distance_m", "tag_decision_margin",
        ])
        for c in correspondences:
            writer.writerow([
                c.session, c.video_frame_index, c.radar_frame_num,
                *[float(v) for v in c.radar_xyz],
                *[float(v) for v in c.camera_xyz],
                float(c.radar_snr), float(c.gate_distance_m), float(c.tag_decision_margin),
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_sessions(root: Path | None, session_args: list[str]) -> list[Path]:
    if session_args:
        return [Path(s).resolve() for s in session_args]
    if root is not None:
        return sorted(p.resolve() for p in root.iterdir() if p.is_dir() and p.name.startswith("session_"))
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processing-root", type=Path, default=None)
    parser.add_argument("--session", action="append", default=[])
    parser.add_argument("--static-prior", type=Path, default=DEFAULT_STATIC_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag-family", default="tagStandard41h12")
    parser.add_argument("--tag-id", type=int, default=None, help="Restrict to one tag id. Default: any.")
    parser.add_argument("--tag-size-m", type=float, default=0.2032)
    parser.add_argument("--reflector-offset-m", type=float, default=0.1016)
    parser.add_argument("--min-decision-margin", type=float, default=15.0)
    parser.add_argument("--max-time-diff-ms", type=float, default=50.0, help="Max |video_ts - radar_ts| for a temporal match.")
    parser.add_argument("--gate-radius-m", type=float, default=0.5, help="Search radius around the prior-predicted reflector position.")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.1)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--rotation-prior-weight", type=float, default=0.15)
    parser.add_argument("--translation-prior-weight", type=float, default=0.5)
    parser.add_argument("--loss-scale-m", type=float, default=0.05)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--min-correspondences", type=int, default=30)
    parser.add_argument("--max-inlier-rms-m", type=float, default=0.15)
    parser.add_argument("--save-correspondences", action="store_true")
    parser.add_argument(
        "--max-pose-flip-deg", type=float, default=20.0,
        help=(
            "Reject a frame's monocular AprilTag pose if its orientation "
            "jumps more than this many degrees from the last-accepted "
            "frame's orientation (the rig is static, so this should only "
            "trip on the planar pose ambiguity's discrete branch flips, "
            "not real motion)."
        ),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.processing_root is None and not args.session:
        raise SystemExit("[ERROR] Provide --processing-root or at least one --session")
    sessions = _resolve_sessions(args.processing_root, args.session)
    if not sessions:
        raise SystemExit("[ERROR] No sessions resolved")

    static_prior = load_static_prior(args.static_prior.resolve() if args.static_prior is not None else None)
    print(f"[setup] sessions={len(sessions)} static_prior_loaded={static_prior.loaded}", flush=True)

    all_correspondences: list[Correspondence] = []
    diagnostics: dict[str, Any] = {"method": "target_calibrate", "sessions": {}}
    for idx, session_dir in enumerate(sessions, start=1):
        print(f"[session {idx}/{len(sessions)}] {session_dir.name}", flush=True)
        corr, info = gather_session_correspondences(session_dir, static_prior, args)
        diagnostics["sessions"][session_dir.name] = info
        all_correspondences.extend(corr)
        print(f"[session] {info}", flush=True)

    result = solve_target_calibration(all_correspondences, static_prior, args)
    diagnostics["solve"] = {
        "success": bool(result["solver"]["success"]),
        "n_total_correspondences": result["n_total_correspondences"],
        "n_inliers": result["n_inliers"],
        "inlier_rms_m": result["inlier_rms_m"],
        "fallback": result["fallback"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "target_radar_camera_extrinsics.json"
    diag_path = args.output_dir / "target_diagnostics.json"
    _write_json(result_path, result)
    _write_json(diag_path, diagnostics)
    if args.save_correspondences:
        save_correspondences_csv(args.output_dir / "target_correspondences.csv", all_correspondences)

    print(f"[done] result={result_path}", flush=True)
    print(f"[done] diagnostics={diag_path}", flush=True)
    print(
        "[done]"
        f" n_total={result['n_total_correspondences']}"
        f" n_inliers={result['n_inliers']}"
        f" inlier_rms_m={result['inlier_rms_m']:.4f}"
        f" fallback={result['fallback']['use_static']}:{result['fallback']['reason']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
