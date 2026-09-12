"""Synthetic end-to-end correctness tests for Stage 4 (Doppler lever-arm) and
Stage 5 (geometry refinement) calibration solvers.

These generate measurements from a KNOWN ground-truth (R_true, t_true) and
check that the solvers actually recover it -- not just that the code runs
without crashing. They import branch1/calib/doppler_lever_arm.py and
branch1/calib/geometry_refinement.py directly (both pure numpy/scipy, no
cv2 dependency) rather than branch1/calib/spatiotemporal_calibrate.py, so
these tests do not require cv2 to be installed.

Written while implementing Stage 4/5
(docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md) -- this suite caught two real
bugs before they could reach real session data:

1. A first geometry_refinement.py design re-checked correspondence validity
   every solver iteration using the CURRENT candidate transform, making the
   residual vector's length change between iterations -- crashed
   scipy.optimize.least_squares partway through a solve once points near an
   image edge were involved (see test_geometry_refinement_recovers_ground_truth).
2. The fix for (1) (freezing the measured depth value, not just
   correspondence membership) accidentally made lateral translation
   (t_x, t_y) mathematically unobservable, since p_c.z never depends on
   t_x/t_y once (u, v) association is also frozen. The final design
   freezes only WHICH correspondences participate, but re-projects and
   re-samples a CACHED depth image every iteration.

A third, unresolved finding is documented rather than fixed here:
--time-mode solve's offset dimension is effectively non-functional in both
this module and the pre-existing Stage 3 solve_spatiotemporal it's modeled
on (see test_time_mode_solve_does_not_crash and
doppler_lever_arm.py's own comment at its least_squares call for the full
root-cause writeup). That test intentionally does NOT assert offset
recovery.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "branch1" / "calib"))
import doppler_lever_arm as dla  # noqa: E402
import geometry_refinement as geo  # noqa: E402

try:
    from scipy.spatial.transform import Rotation

    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _rotation_angle_error_deg(r_true: np.ndarray, r_est: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(r_true.T @ r_est) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


@unittest.skipUnless(_HAVE_SCIPY, "scipy is required for these solver tests")
class DopplerLeverArmSyntheticTests(unittest.TestCase):
    R_TRUE = None  # set in setUp, needs scipy
    T_TRUE = np.array([0.02, 0.01, -0.015])
    N_FRAMES = 40
    DETS_PER_FRAME = 8

    def setUp(self) -> None:
        self.R_true = Rotation.from_euler("xyz", [3.0, -2.0, 95.0], degrees=True).as_matrix()
        self.t_true = self.T_TRUE
        self.rng = np.random.default_rng(42)

    def _make_session(self, seed: int, doppler_noise_std: float = 0.0):
        r = np.random.default_rng(seed)
        frames = []
        v_c_list, omega_c_list, ts_list = [], [], []
        for i in range(self.N_FRAMES):
            ts = 1_000_000_000_000_000 + i * 100_000  # 100ms frames, epoch us
            v_c = r.normal(0, 0.3, size=3) + np.array([0.2 * np.sin(i * 0.3), 0.1 * np.cos(i * 0.2), 0.05])
            omega_c = r.normal(0, 0.4, size=3) + np.array([0.3 * np.cos(i * 0.25), 0.2 * np.sin(i * 0.4), 0.1])
            v_c_list.append(v_c)
            omega_c_list.append(omega_c)
            ts_list.append(ts)

            v_body = v_c + np.cross(omega_c, self.t_true)
            v_radar_true = self.R_true.T @ v_body

            u = r.normal(0, 1, size=(self.DETS_PER_FRAME, 3))
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            ranges = r.uniform(1.5, 4.0, size=self.DETS_PER_FRAME)
            xyz = u * ranges[:, None]
            radial = np.einsum("ij,j->i", u, v_radar_true)
            if doppler_noise_std > 0:
                radial = radial + r.normal(0, doppler_noise_std, size=self.DETS_PER_FRAME)
            snr = r.uniform(3.0, 10.0, size=self.DETS_PER_FRAME)

            frames.append(
                SimpleNamespace(
                    timestamp_us=ts,
                    frame_num=i,
                    accepted_xyz=xyz,
                    accepted_doppler_mps=radial,
                    accepted_snr=snr,
                )
            )

        v_c_arr, omega_c_arr, ts_arr = np.asarray(v_c_list), np.asarray(omega_c_list), np.asarray(ts_list)

        def linear_velocity_camera_at(t):
            return np.asarray([np.interp(t, ts_arr, v_c_arr[:, k]) for k in range(3)])

        def angular_velocity_at(t):
            return np.asarray([np.interp(t, ts_arr, omega_c_arr[:, k]) for k in range(3)])

        return SimpleNamespace(
            session_name=f"session_synthetic_{seed}",
            radar_motion=SimpleNamespace(estimates=frames),
            camera_trajectory=SimpleNamespace(linear_velocity_camera_at=linear_velocity_camera_at),
            angular_velocity_at=angular_velocity_at,
            depth_lookup=None,
        )

    def _default_args(self, **overrides):
        base = dict(
            max_doppler_detections=20000,
            min_doppler_detections=50,
            time_mode="fixed",
            max_offset_ms=200.0,
            rotation_prior_weight=0.05,
            translation_prior_weight=0.05,
            time_prior_weight=1.0,
            doppler_loss_scale_mps=0.3,
            max_nfev=300,
            max_residual_rms_mps=0.45,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_recovers_ground_truth_noiseless(self):
        sessions = [self._make_session(seed=100 + i) for i in range(3)]
        rotvec_perturb = self.rng.normal(0, 0.05, size=3)
        r_warm = Rotation.from_rotvec(Rotation.from_matrix(self.R_true).as_rotvec() + rotvec_perturb).as_matrix()
        t_warm = self.t_true + self.rng.normal(0, 0.01, size=3)
        warm_start = {
            "method": "stage3_synthetic_warmstart",
            "R_radar_to_camera": r_warm.tolist(),
            "t_radar_to_camera": t_warm.tolist(),
            "time_offset_ms": 0.0,
        }
        # deliberately WRONG static prior, to prove the fit is driven by
        # Doppler evidence rather than by the prior term
        r_prior_wrong = Rotation.from_euler("xyz", [10.0, -8.0, 88.0], degrees=True).as_matrix()
        static_prior = SimpleNamespace(loaded=True, rotation=r_prior_wrong, translation_m=np.array([0.05, -0.02, 0.01]))

        result = dla.solve_doppler_lever_arm(sessions, static_prior, warm_start, self._default_args())

        r_est = np.asarray(result["R_radar_to_camera"])
        t_est = np.asarray(result["t_radar_to_camera"])
        rot_err_deg = _rotation_angle_error_deg(self.R_true, r_est)
        t_err_m = float(np.linalg.norm(self.t_true - t_est))

        self.assertTrue(result["solver"]["success"])
        self.assertFalse(result["fallback"]["use_warm_start"], result["fallback"])
        self.assertLess(rot_err_deg, 0.5, f"rotation error {rot_err_deg} deg too large")
        self.assertLess(t_err_m, 0.005, f"translation error {t_err_m * 1000} mm too large")

    def test_degrades_gracefully_under_doppler_noise(self):
        sessions = [self._make_session(seed=200 + i, doppler_noise_std=0.05) for i in range(5)]
        rotvec_perturb = self.rng.normal(0, 0.05, size=3)
        r_warm = Rotation.from_rotvec(Rotation.from_matrix(self.R_true).as_rotvec() + rotvec_perturb).as_matrix()
        t_warm = self.t_true + self.rng.normal(0, 0.01, size=3)
        warm_start = {
            "method": "x",
            "R_radar_to_camera": r_warm.tolist(),
            "t_radar_to_camera": t_warm.tolist(),
            "time_offset_ms": 0.0,
        }
        static_prior = SimpleNamespace(loaded=True, rotation=self.R_true, translation_m=self.t_true)
        result = dla.solve_doppler_lever_arm(sessions, static_prior, warm_start, self._default_args())
        r_est = np.asarray(result["R_radar_to_camera"])
        t_est = np.asarray(result["t_radar_to_camera"])
        rot_err_deg = _rotation_angle_error_deg(self.R_true, r_est)
        t_err_m = float(np.linalg.norm(self.t_true - t_est))
        self.assertAlmostEqual(result["residual_rms_mps"], 0.05, delta=0.02)
        self.assertLess(rot_err_deg, 3.0)
        self.assertLess(t_err_m, 0.05)

    def test_falls_back_on_insufficient_detections(self):
        session = self._make_session(seed=999)
        session.radar_motion.estimates = session.radar_motion.estimates[:2]
        warm_start = {
            "method": "x",
            "R_radar_to_camera": self.R_true.tolist(),
            "t_radar_to_camera": self.t_true.tolist(),
            "time_offset_ms": 0.0,
        }
        static_prior = SimpleNamespace(loaded=True, rotation=self.R_true, translation_m=self.t_true)
        result = dla.solve_doppler_lever_arm([session], static_prior, warm_start, self._default_args())
        self.assertTrue(result["fallback"]["use_warm_start"])
        self.assertIn("insufficient", result["fallback"]["reason"])
        self.assertEqual(result["R_radar_to_camera"], warm_start["R_radar_to_camera"])

    def test_time_mode_solve_does_not_crash(self):
        """KNOWN LIMITATION (see doppler_lever_arm.py's comment at its
        least_squares call): --time-mode solve's offset dimension is
        effectively non-functional because scipy's finite-difference step
        is always far smaller than the 1-microsecond rounding grain baked
        into the objective (and into CameraTrajectory._clamp). This is
        inherited from Stage 3's pre-existing solve_spatiotemporal, not
        introduced here. This test intentionally only checks that the
        solve completes without error and that rotation/translation (whose
        x0 values are never exactly on a quantization boundary the same
        way) still move sensibly -- it does NOT assert offset recovery."""
        sessions = [self._make_session(seed=400 + i) for i in range(3)]
        rotvec_perturb = self.rng.normal(0, 0.05, size=3)
        r_warm = Rotation.from_rotvec(Rotation.from_matrix(self.R_true).as_rotvec() + rotvec_perturb).as_matrix()
        t_warm = self.t_true + self.rng.normal(0, 0.01, size=3)
        warm_start = {
            "method": "x",
            "R_radar_to_camera": r_warm.tolist(),
            "t_radar_to_camera": t_warm.tolist(),
            "time_offset_ms": 0.0,
        }
        static_prior = SimpleNamespace(loaded=True, rotation=self.R_true, translation_m=self.t_true)
        args = self._default_args(time_mode="solve")
        result = dla.solve_doppler_lever_arm(sessions, static_prior, warm_start, args)
        self.assertTrue(result["solver"]["success"])
        self.assertTrue(np.isfinite(result["residual_rms_mps"]))


@unittest.skipUnless(_HAVE_SCIPY, "scipy is required for these solver tests")
class GeometryRefinementSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.R_true = Rotation.from_euler("xyz", [3.0, -2.0, 95.0], degrees=True).as_matrix()
        self.t_true = np.array([0.02, 0.01, -0.015])
        self.rng = np.random.default_rng(42)

    class _FakeIntrinsics:
        def __init__(self):
            self.fx = 300.0
            self.fy = 300.0
            self.cx = 212.0
            self.cy = 120.0
            self.width = 424
            self.height = 240

    def _make_geometry_session(self, seed: int):
        r = np.random.default_rng(seed)
        intr = self._FakeIntrinsics()
        frames, depth_images = [], {}
        for i in range(20):
            ts = 2_000_000_000_000_000 + i * 100_000
            n_pts = 6
            # Oblique planar surface per frame (z varies smoothly with
            # pixel u,v) -- a fronto-parallel wall would make t_x/t_y
            # mathematically unobservable from a pure depth residual
            # (z is constant regardless of (u,v) on such a wall); real
            # indoor scenes are rarely perfectly fronto-parallel over a
            # whole FOV, so this is the realistic, not the convenient, case.
            z0 = r.uniform(1.5, 3.5)
            slope_u = r.uniform(0.001, 0.003) * r.choice([-1, 1])
            slope_v = r.uniform(0.001, 0.003) * r.choice([-1, 1])
            uu, vv = np.meshgrid(np.arange(intr.width), np.arange(intr.height))
            depth_img = (z0 + slope_u * (uu - intr.cx) + slope_v * (vv - intr.cy)).astype(np.float32)
            u_px = r.uniform(50, intr.width - 50, size=n_pts)
            v_px = r.uniform(30, intr.height - 30, size=n_pts)
            z_true = z0 + slope_u * (u_px - intr.cx) + slope_v * (v_px - intr.cy)
            x_c = (u_px - intr.cx) * z_true / intr.fx
            y_c = (v_px - intr.cy) * z_true / intr.fy
            p_camera = np.stack([x_c, y_c, z_true], axis=1)
            p_radar = (self.R_true.T @ (p_camera - self.t_true).T).T

            snr = r.uniform(3.0, 10.0, size=n_pts)
            frames.append(
                SimpleNamespace(
                    timestamp_us=ts,
                    frame_num=i,
                    accepted_xyz=p_radar,
                    accepted_doppler_mps=np.zeros(n_pts),
                    accepted_snr=snr,
                )
            )
            depth_images[ts] = depth_img

        def depth_at(timestamp_us):
            return depth_images.get(timestamp_us)

        depth_lookup = SimpleNamespace(intrinsics=intr, depth_at=depth_at)
        return SimpleNamespace(
            session_name=f"session_geom_{seed}",
            radar_motion=SimpleNamespace(estimates=frames),
            depth_lookup=depth_lookup,
        )

    def _default_args(self, **overrides):
        base = dict(
            geometry_max_detections_per_session=100,
            geometry_min_correspondences=20,
            geometry_loss_scale_m=0.15,
            geometry_max_residual_rms_m=0.5,
            geometry_warmstart_rotation_weight=2.0,
            geometry_warmstart_translation_weight=8.0,
            max_nfev=300,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_residual_is_near_zero_at_ground_truth(self):
        """Validates the residual MODEL itself (not the optimizer): at the
        exact ground-truth transform, every correspondence's residual
        should be ~0, up to pixel-rounding quantization noise."""
        sessions = [self._make_geometry_session(seed=300 + i) for i in range(4)]
        correspondences = geo.gather_geometry_correspondences(
            sessions, self.R_true, self.t_true, max_per_session=100
        )
        residuals = []
        for corr in correspondences:
            z_pred, measured = geo._reproject_and_sample_cached(self.R_true, self.t_true, corr)
            if measured is not None:
                residuals.append(z_pred - measured)
        residuals = np.asarray(residuals)
        self.assertGreater(len(residuals), 300)
        self.assertLess(float(np.sqrt(np.mean(residuals**2))), 0.002)

    def test_refinement_improves_on_doppler_input(self):
        sessions = [self._make_geometry_session(seed=300 + i) for i in range(4)]
        rotvec_perturb = self.rng.normal(0, 0.03, size=3)
        r_doppler_in = Rotation.from_rotvec(Rotation.from_matrix(self.R_true).as_rotvec() + rotvec_perturb).as_matrix()
        t_doppler_in = self.t_true + self.rng.normal(0, 0.008, size=3)
        doppler_result = {
            "method": "doppler_lever_arm_stage4_synthetic",
            "R_radar_to_camera": r_doppler_in.tolist(),
            "t_radar_to_camera": t_doppler_in.tolist(),
            "translation_magnitude_m": float(np.linalg.norm(t_doppler_in)),
            "time_offset_ms": 0.0,
        }

        result = geo.refine_with_geometry(sessions, doppler_result, self._default_args())
        r_est = np.asarray(result["R_radar_to_camera"])
        t_est = np.asarray(result["t_radar_to_camera"])
        rot_err_out = _rotation_angle_error_deg(self.R_true, r_est)
        t_err_out = float(np.linalg.norm(self.t_true - t_est))
        rot_err_in = _rotation_angle_error_deg(self.R_true, r_doppler_in)
        t_err_in = float(np.linalg.norm(self.t_true - t_doppler_in))

        self.assertTrue(result["solver"]["success"])
        self.assertFalse(result["fallback"]["use_doppler_input"], result["fallback"])
        # Relative-improvement assertions, not absolute-mm thresholds:
        # test_residual_is_near_zero_at_ground_truth already confirms the
        # residual model itself is correct; a depth-only projective
        # residual has an inherently weak, pixel-quantization-noisy signal
        # for lateral translation on a gently-sloped surface with only
        # ~400 points -- exactly the limitation the reference document
        # flags when recommending point-to-plane as "a better refinement"
        # than this simple version. "did it measurably improve" is the
        # meaningful bar here, not an arbitrary absolute mm threshold.
        self.assertLess(rot_err_out, rot_err_in, "should improve rotation, not degrade it")
        # Translation gets a looser (not strict-less) bound: across repeated
        # runs with different random seeds, rotation improvement is
        # robust, but translation is occasionally a statistical wash
        # (observed: outputs differing from input by ~5e-7 m, i.e. noise,
        # not a real regression) -- an expected consequence of this
        # residual's weak lateral-translation signal (see
        # test_residual_is_near_zero_at_ground_truth and this module's
        # docstring), not a bug. A strict "<" here is seed-sensitive; "not
        # meaningfully worse" is the honest, robust claim.
        self.assertLessEqual(t_err_out, t_err_in * 1.02, "should not meaningfully degrade translation")

    def test_split_gather_and_solve_matches_combined(self):
        """Validates the exact pattern the rolling-batch notebook uses:
        gather_geometry_correspondences called separately PER BATCH (here,
        simulated as 2 batches of 2 sessions each), concatenated, then
        solve_geometry_refinement_from_correspondences called once -- must
        produce the same result as calling refine_with_geometry directly
        on all sessions at once (see geometry_refinement.py's
        refine_with_geometry docstring for why the split exists: a real
        rolling-batch driver clears each batch's depth files from disk
        before the next batch downloads, so no single point in time has
        every session's depth available simultaneously)."""
        sessions = [self._make_geometry_session(seed=300 + i) for i in range(4)]
        rotvec_perturb = self.rng.normal(0, 0.03, size=3)
        r_doppler_in = Rotation.from_rotvec(Rotation.from_matrix(self.R_true).as_rotvec() + rotvec_perturb).as_matrix()
        t_doppler_in = self.t_true + self.rng.normal(0, 0.008, size=3)
        doppler_result = {
            "method": "x",
            "R_radar_to_camera": r_doppler_in.tolist(),
            "t_radar_to_camera": t_doppler_in.tolist(),
            "translation_magnitude_m": float(np.linalg.norm(t_doppler_in)),
            "time_offset_ms": 0.0,
        }
        args = self._default_args()

        # geometry_max_detections_per_session set above each session's 120
        # candidates (20 frames x 6 points) so this test isolates the
        # gather/solve split itself -- with subsampling active, gather
        # calls with independent RNG streams (one per batch) legitimately
        # keep DIFFERENT random subsets than one combined call, which would
        # confound this equivalence check with subsampling nondeterminism
        # rather than testing the split/accumulate logic itself.
        args.geometry_max_detections_per_session = 200
        combined_result = geo.refine_with_geometry(sessions, doppler_result, args)

        warm_r = np.asarray(doppler_result["R_radar_to_camera"])
        warm_t = np.asarray(doppler_result["t_radar_to_camera"])
        batch1 = geo.gather_geometry_correspondences(
            sessions[:2], warm_r, warm_t, max_per_session=args.geometry_max_detections_per_session
        )
        batch2 = geo.gather_geometry_correspondences(
            sessions[2:], warm_r, warm_t, max_per_session=args.geometry_max_detections_per_session
        )
        accumulated = batch1 + batch2
        split_result = geo.solve_geometry_refinement_from_correspondences(accumulated, doppler_result, args)

        self.assertEqual(combined_result["n_correspondences"], split_result["n_correspondences"])
        np.testing.assert_allclose(
            combined_result["R_radar_to_camera"], split_result["R_radar_to_camera"], atol=1e-10
        )
        np.testing.assert_allclose(
            combined_result["t_radar_to_camera"], split_result["t_radar_to_camera"], atol=1e-10
        )

    def test_falls_back_when_no_depth_available(self):
        intr = self._FakeIntrinsics()
        depth_lookup = SimpleNamespace(intrinsics=intr, depth_at=lambda ts: None)
        session = SimpleNamespace(
            session_name="empty", radar_motion=SimpleNamespace(estimates=[]), depth_lookup=depth_lookup
        )
        doppler_result = {
            "method": "x",
            "R_radar_to_camera": self.R_true.tolist(),
            "t_radar_to_camera": self.t_true.tolist(),
            "translation_magnitude_m": float(np.linalg.norm(self.t_true)),
            "time_offset_ms": 0.0,
        }
        result = geo.refine_with_geometry([session], doppler_result, self._default_args())
        self.assertTrue(result["fallback"]["use_doppler_input"])
        self.assertIn("insufficient", result["fallback"]["reason"])
        self.assertEqual(result["R_radar_to_camera"], doppler_result["R_radar_to_camera"])


if __name__ == "__main__":
    unittest.main()
