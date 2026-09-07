"""
corridor_soft_structural.py

Soft structural scoring module for the NOAH radar navigation pipeline.

Replaces the hard wall/floor mask approach (structural_mask branch) with
continuous structural-membership probabilities that are passed as INPUT
FEATURES to the downstream classifier — not used as a gating decision.

Design philosophy
─────────────────
The structural mask branch failed because a binary routing decision
(is this point wall/floor? → remove it) is too brittle with sparse,
noisy mmWave data. Pillars get absorbed into walls, floor overclaims,
and the obstacle classifier starves.

This module instead computes per-point soft scores:
    p_wall   ∈ [0, 1]   — how wall-like is this point?
    p_floor  ∈ [0, 1]   — how floor-like is this point?
    d_wall   ∈ ℝ        — signed distance to nearest corridor wall plane (m)
    d_floor  ∈ ℝ        — signed distance to floor plane (m)
    wall_ang ∈ [0, π/2] — angle between point's local normal and wall normal
    floor_ang∈ [0, π/2] — angle between point's local normal and floor normal
    corridor_margin ∈ ℝ  — lateral distance from corridor centerline (m)
    z_above_floor  ∈ ℝ   — height above estimated floor (m)

These are appended as features to every point. The classifier (3DGCNN or
otherwise) then learns WHEN to trust the geometry and when to override it.
No points are ever removed.

Integration
───────────
This module is designed to be called from:
    build_temporal_corridor_soft_structural_dataset.py
which replaces:
    build_temporal_corridor_structural_mask_dataset.py

The build script loads the canonical training_dataset.csv, calls
compute_soft_structural_features() per frame group, and appends the
new columns before writing the augmented dataset.

The train script uses the same 3DGCNN architecture but with an expanded
input feature vector (original features + soft structural features).
The label space stays 5-class door_as_wall: wall, floor, pillar, human, box_like.
All points are classified — nothing is masked out.

Corridor geometry
─────────────────
Corridor planes are estimated from the temporally-aggregated point cloud
(rolling window of neighboring frames aligned to the anchor frame, same
as temporal_corridor_common.py). This module provides:

    1. fit_corridor_planes()  — RANSAC-based wall and floor plane fitting
                                on the aggregated cloud.
    2. compute_soft_structural_features() — per-point feature computation
                                            given the fitted planes.
    3. Pillar anomaly features — points near walls but deviating from the
                                 wall surface are scored as pillar candidates.

Dependencies: numpy, scipy (for KDTree in local normal estimation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Plane representation
# =============================================================================

@dataclass
class Plane:
    """Plane represented as (normal, offset): n · x + d = 0, ‖n‖ = 1."""
    normal: np.ndarray          # (3,) unit normal
    offset: float               # signed offset d
    inlier_frac: float = 0.0    # fraction of cloud that are inliers
    n_inliers: int = 0
    xy_elongation: float = 1.0  # λ₁/λ₂ of XY covariance of inliers
                                # >>1 = elongated (wall), ≈1 = compact (pillar)
    xy_span: float = 0.0       # max extent of inliers in XY plane (meters)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance from points (N, 3) to this plane."""
        return pts @ self.normal + self.offset

    def abs_distance(self, pts: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_distance(pts))


@dataclass
class CorridorPlanes:
    """Container for fitted corridor geometry."""
    floor: Optional[Plane] = None
    walls: list[Plane] = field(default_factory=list)   # elongated vertical planes (real walls)
    pillar_planes: list[Plane] = field(default_factory=list)  # compact vertical planes (pillars)
    centerline_y: float = 0.0       # lateral center between walls (in radar frame)
    corridor_width: float = 3.0     # estimated corridor width (m), fallback default
    floor_z: float = -0.5           # estimated floor height in radar Z (m)
    # Plane fit quality (inlier fractions) — exposed as features so the model can
    # continuously discount structural features when geometry was poorly constrained.
    wall_fit_max_inlier_frac: float = 0.0   # best wall plane's inlier frac; 0.0 = no wall
    floor_fit_inlier_frac: float = 0.0      # floor plane's inlier frac; 0.0 = no floor


# =============================================================================
# RANSAC plane fitting
# =============================================================================

def _ransac_plane(
    pts: np.ndarray,
    n_iter: int = 200,
    dist_thresh: float = 0.06,
    min_inlier_frac: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Plane]:
    """
    Fit a plane to pts (N, 3) using RANSAC.

    Returns the best Plane if min_inlier_frac is met, else None.
    """
    if len(pts) < 3:
        return None
    if rng is None:
        rng = np.random.default_rng(42)

    N = len(pts)
    best_n_inliers = 0
    best_normal = None
    best_offset = None

    for _ in range(n_iter):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-10:
            continue
        n = n / norm
        d = -np.dot(n, p0)

        dists = np.abs(pts @ n + d)
        n_inliers = int(np.sum(dists < dist_thresh))

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_normal = n.copy()
            best_offset = d

    if best_normal is None or best_n_inliers / N < min_inlier_frac:
        return None

    # Refine with inliers
    inlier_mask = np.abs(pts @ best_normal + best_offset) < dist_thresh
    inlier_pts = pts[inlier_mask]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        cov = np.cov((inlier_pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        best_normal = eigvecs[:, 0].copy()  # smallest eigenvalue = normal
        best_offset = -np.dot(best_normal, centroid)

    # Convention: floor normal should point up (+Z in radar frame)
    # Wall normals should point inward (toward corridor center)

    plane = Plane(
        normal=best_normal,
        offset=best_offset,
        inlier_frac=best_n_inliers / N,
        n_inliers=best_n_inliers,
    )
    return plane


def _compute_xy_elongation(pts: np.ndarray, plane: Plane, dist_thresh: float) -> Tuple[float, float]:
    """
    Compute the XY elongation ratio and span of a plane's inlier set.

    Returns:
        elongation : λ₁/λ₂ ratio of XY covariance eigenvalues.
                     >>1 = elongated (wall), ≈1 = compact (pillar)
        xy_span    : maximum pairwise distance in XY among inliers (meters)
    """
    inlier_mask = plane.abs_distance(pts) < dist_thresh
    inliers = pts[inlier_mask]

    if len(inliers) < 3:
        return 1.0, 0.0

    # Project onto XY plane (horizontal)
    xy = inliers[:, :2]  # X=lateral, Y=forward
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    cov = centered.T @ centered / len(centered)

    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # descending
        # Ratio of largest to smallest eigenvalue
        elongation = float(eigvals[0] / max(eigvals[1], 1e-10))
    except np.linalg.LinAlgError:
        elongation = 1.0

    # XY span: max extent in the dominant direction
    xy_span = float(np.max(xy, axis=0).max() - np.min(xy, axis=0).min())

    return elongation, xy_span


# =============================================================================
# Corridor plane fitting
# =============================================================================

def fit_corridor_planes(
    pts: np.ndarray,
    floor_dist_thresh: float = 0.06,
    floor_min_inlier_frac: float = 0.08,
    wall_dist_thresh: float = 0.08,
    wall_min_inlier_frac: float = 0.06,
    floor_normal_tol_deg: float = 30.0,
    wall_normal_tol_deg: float = 30.0,
    ransac_iters: int = 300,
    max_walls: int = 2,
    wall_min_elongation: float = 3.0,
    wall_min_xy_span: float = 0.8,
    max_pillar_planes: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> CorridorPlanes:
    """
    Fit floor and wall planes from a (temporally aggregated) point cloud.

    Coordinate convention (TI IWR1843 radar frame):
        X = lateral (positive right)
        Y = forward (range direction)
        Z = vertical (positive up)

    The floor is expected to be roughly horizontal (normal ≈ ±Z).
    Walls are expected to be roughly vertical (normal ≈ ±X or with small Z component).

    Strategy:
        1. Fit floor first among points with low Z (bottom portion of cloud).
        2. Remove floor inliers, then iteratively fit up to max_walls wall planes
           among remaining points with roughly vertical normals.

    Parameters
    ----------
    pts : (N, 3) float array — point cloud in radar frame (x, y, z)
    floor_dist_thresh : RANSAC inlier distance for floor
    floor_min_inlier_frac : min fraction of pts to accept a floor plane
    wall_dist_thresh : RANSAC inlier distance for walls
    wall_min_inlier_frac : min fraction of remaining pts to accept a wall
    floor_normal_tol_deg : max angle between fitted normal and +Z for floor
    wall_normal_tol_deg : max angle between fitted normal and XY-plane for wall
    ransac_iters : iterations per RANSAC call
    max_walls : max number of wall planes to fit
    rng : numpy random generator for reproducibility

    Returns
    -------
    CorridorPlanes with fitted planes (may have None floor, empty walls)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    result = CorridorPlanes()

    if len(pts) < 10:
        return result

    up = np.array([0.0, 0.0, 1.0])
    floor_tol_cos = np.cos(np.radians(floor_normal_tol_deg))
    wall_tol_sin = np.sin(np.radians(wall_normal_tol_deg))

    # ----- Step 1: Fit floor -----
    # Focus on points in the lower portion of the cloud.
    # Use the bottom 65th percentile rather than median — in corridor scenes
    # the median Z can be above floor level when walls dominate the cloud.
    z_vals = pts[:, 2]
    z_cutoff = np.percentile(z_vals, 65)
    floor_candidates = pts[z_vals <= z_cutoff]

    floor_plane = None
    if len(floor_candidates) >= 3:
        floor_plane = _ransac_plane(
            floor_candidates,
            n_iter=ransac_iters,
            dist_thresh=floor_dist_thresh,
            min_inlier_frac=floor_min_inlier_frac,
            rng=rng,
        )

    # Fallback: try all points if the restricted set failed
    if floor_plane is None and len(pts) >= 3:
        floor_plane = _ransac_plane(
            pts,
            n_iter=ransac_iters // 2,
            dist_thresh=floor_dist_thresh,
            min_inlier_frac=floor_min_inlier_frac * 0.5,
            rng=rng,
        )

    if floor_plane is not None:
        # Check that normal is roughly vertical
        cos_angle = abs(np.dot(floor_plane.normal, up))
        if cos_angle >= floor_tol_cos:
            # Ensure normal points up
            if np.dot(floor_plane.normal, up) < 0:
                floor_plane.normal = -floor_plane.normal
                floor_plane.offset = -floor_plane.offset
            result.floor = floor_plane
            # Estimate floor Z as the mean Z of inliers
            floor_dists = floor_plane.abs_distance(pts)
            floor_inlier_z = pts[floor_dists < floor_dist_thresh, 2]
            if len(floor_inlier_z) > 0:
                result.floor_z = float(np.median(floor_inlier_z))

    # ----- Step 2: Fit vertical planes (walls AND pillar candidates) -----
    # Remove floor inliers from the candidate set
    remaining_mask = np.ones(len(pts), dtype=bool)
    if result.floor is not None:
        remaining_mask = result.floor.abs_distance(pts) >= floor_dist_thresh

    n_vertical_attempts = max_walls + max_pillar_planes
    for _ in range(n_vertical_attempts):
        wall_candidates = pts[remaining_mask]
        if len(wall_candidates) < 3:
            break

        wall_plane = _ransac_plane(
            wall_candidates,
            n_iter=ransac_iters,
            dist_thresh=wall_dist_thresh,
            min_inlier_frac=wall_min_inlier_frac,
            rng=rng,
        )
        if wall_plane is None:
            break

        # Check that normal is roughly horizontal (vertical surface criterion)
        # A vertical surface normal should have small Z-component
        z_comp = abs(np.dot(wall_plane.normal, up))
        if z_comp > wall_tol_sin:
            # This plane is too horizontal — probably another floor / ceiling
            break

        # ---- Anti-pillar elongation check ----
        # Compute the XY eigenvalue ratio of the inlier set.
        # Walls are elongated in XY (λ₁/λ₂ >> 1, large span).
        # Pillars are compact in XY (λ₁/λ₂ ≈ 1, small span).
        elongation, xy_span = _compute_xy_elongation(pts, wall_plane, wall_dist_thresh)
        wall_plane.xy_elongation = elongation
        wall_plane.xy_span = xy_span

        # Remove this plane's inliers regardless of classification
        plane_inlier = wall_plane.abs_distance(pts) < wall_dist_thresh
        remaining_mask = remaining_mask & ~plane_inlier

        if elongation >= wall_min_elongation and xy_span >= wall_min_xy_span:
            # Elongated + large span → real wall
            if len(result.walls) < max_walls:
                result.walls.append(wall_plane)
        else:
            # Compact or small span → pillar candidate
            if len(result.pillar_planes) < max_pillar_planes:
                result.pillar_planes.append(wall_plane)

    # ----- Step 3: Record fit quality -----
    if result.floor is not None:
        result.floor_fit_inlier_frac = result.floor.inlier_frac
    if result.walls:
        result.wall_fit_max_inlier_frac = max(w.inlier_frac for w in result.walls)

    # ----- Step 4: Estimate corridor geometry -----
    if len(result.walls) == 2:
        # Two walls: corridor width and centerline
        # The centerline in the lateral (X) dimension is midway between
        # the two wall planes evaluated at a reference point (origin).
        d0 = result.walls[0].signed_distance(np.zeros((1, 3)))[0]
        d1 = result.walls[1].signed_distance(np.zeros((1, 3)))[0]
        result.corridor_width = abs(d0 - d1)
        result.centerline_y = (d0 + d1) / 2.0
    elif len(result.walls) == 1:
        # One wall: estimate corridor width from point spread
        wall_dists = result.walls[0].signed_distance(pts)
        result.corridor_width = float(np.percentile(np.abs(wall_dists), 95)) * 2
        result.centerline_y = 0.0

    return result


# =============================================================================
# Local normal estimation (for angular features)
# =============================================================================

def estimate_local_normals(
    pts: np.ndarray,
    k: int = 8,
    fallback_normal: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Estimate per-point local surface normals via PCA of k-nearest neighbors.

    Parameters
    ----------
    pts : (N, 3) float array
    k : number of neighbors (including self)
    fallback_normal : (3,) used when N < k; defaults to [0, 0, 1]

    Returns
    -------
    normals : (N, 3) unit normals (ambiguous sign — caller should orient)
    """
    from scipy.spatial import cKDTree

    N = len(pts)
    if fallback_normal is None:
        fallback_normal = np.array([0.0, 0.0, 1.0])

    if N < 3:
        return np.tile(fallback_normal, (N, 1))

    k_actual = min(k, N)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_actual)
    if k_actual < 3:
        return np.tile(fallback_normal, (N, 1))

    normals = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        neighbors = pts[idx[i]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        cov = centered.T @ centered / len(neighbors)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]  # smallest eigenvalue
        except np.linalg.LinAlgError:
            normals[i] = fallback_normal

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals = normals / norms

    return normals


# =============================================================================
# Soft structural feature computation
# =============================================================================

def compute_soft_structural_features(
    pts: np.ndarray,
    corridor: CorridorPlanes,
    compute_normals: bool = True,
    normal_k: int = 8,
) -> dict[str, np.ndarray]:
    """
    Compute per-point soft structural features given fitted corridor planes.

    Parameters
    ----------
    pts : (N, 3) float array — current-frame points in radar frame (x, y, z)
    corridor : CorridorPlanes — fitted from temporally-aggregated cloud
    compute_normals : whether to estimate local normals for angular features
    normal_k : neighbor count for local normal PCA

    Returns
    -------
    Dictionary of feature name → (N,) float arrays:
        p_floor          : soft floor membership [0, 1]
        p_wall           : soft wall membership [0, 1] (max over all walls)
        d_floor          : signed distance to floor plane (m)
        d_wall_min       : distance to nearest wall plane (m)
        d_wall_signed    : signed distance to nearest wall (m, for left/right)
        z_above_floor    : height above estimated floor (m)
        corridor_margin  : normalized lateral position [-1, 1] where ±1 = wall
        wall_anomaly     : pillar candidate score [0, 1]
                           high when close to wall azimuth but offset in range
        floor_ang        : angle (rad) between local normal and floor normal
        wall_ang         : angle (rad) between local normal and nearest wall normal
    """
    N = len(pts)
    features: dict[str, np.ndarray] = {}

    # ---- Floor features ----
    if corridor.floor is not None:
        d_floor = corridor.floor.signed_distance(pts)
        features["d_floor"] = d_floor.astype(np.float32)
        features["z_above_floor"] = (pts[:, 2] - corridor.floor_z).astype(np.float32)

        # Soft floor probability: Gaussian decay from floor plane
        # σ controls how quickly probability drops with distance.
        # Radar range resolution is 4.2cm but multipath/noise spread further.
        sigma_floor = 0.12  # ~12 cm
        features["p_floor"] = np.exp(
            -0.5 * (d_floor / sigma_floor) ** 2
        ).astype(np.float32)
    else:
        # No floor plane found — use neutral fallback values, NOT large sentinels.
        # 2.0m ≈ "far from any plausible floor" without wrecking feature scale.
        features["d_floor"] = np.full(N, 2.0, dtype=np.float32)
        features["z_above_floor"] = pts[:, 2].astype(np.float32)
        features["p_floor"] = np.zeros(N, dtype=np.float32)

    # ---- Wall features ----
    if corridor.walls:
        # Compute distance to each wall, take the nearest
        wall_dists = np.column_stack([
            w.abs_distance(pts) for w in corridor.walls
        ])  # (N, n_walls)
        wall_signed = np.column_stack([
            w.signed_distance(pts) for w in corridor.walls
        ])

        nearest_wall_idx = np.argmin(wall_dists, axis=1)
        d_wall_min = wall_dists[np.arange(N), nearest_wall_idx]
        d_wall_signed = wall_signed[np.arange(N), nearest_wall_idx]

        features["d_wall_min"] = d_wall_min.astype(np.float32)
        features["d_wall_signed"] = d_wall_signed.astype(np.float32)

        # Soft wall probability: Gaussian decay
        sigma_wall = 0.15  # ~15 cm — wider to account for radar noise
        features["p_wall"] = np.exp(
            -0.5 * (d_wall_min / sigma_wall) ** 2
        ).astype(np.float32)

        # Corridor margin: normalized lateral position
        # +1 = at wall, 0 = at centerline, -1 = at opposite wall
        half_width = max(corridor.corridor_width / 2.0, 0.1)
        if len(corridor.walls) >= 2:
            # Use signed distance from centerline
            # Approximate: project onto the dominant lateral direction (X)
            lateral = pts[:, 0] - corridor.centerline_y
            features["corridor_margin"] = np.clip(
                lateral / half_width, -1.0, 1.0
            ).astype(np.float32)
        else:
            features["corridor_margin"] = np.clip(
                d_wall_signed / half_width, -1.0, 1.0
            ).astype(np.float32)

        # ---- Wall anomaly / pillar candidate score ----
        # A point is a pillar candidate if:
        #   - It is laterally near a wall (small corridor_margin deviation)
        #   - But its range-distance from the wall plane is moderate
        #     (it protrudes into the corridor)
        # This catches pillars that sit against or near walls.
        #
        # Score = closeness_to_wall_lateral × protrusion_from_wall_surface
        # where closeness is high when the point is in the wall's lateral zone
        # and protrusion is high when the point is offset from the wall plane.

        sigma_lateral = 0.3  # how laterally close to wall = "near"
        lateral_closeness = np.exp(-0.5 * (d_wall_min / sigma_lateral) ** 2)

        # Protrusion: moderate distance from wall plane, not too close (= wall)
        # not too far (= clearly in corridor). Peak around 0.15–0.40m.
        protrusion = d_wall_min
        protrusion_score = protrusion * np.exp(-protrusion / 0.3)
        # Normalize to [0, 1] (peak at d=0.3 gives ~0.11, scale up)
        protrusion_score = np.clip(protrusion_score / 0.12, 0.0, 1.0)

        # Also boost if the point has significant height above floor
        # (pillars are vertical structures)
        if corridor.floor is not None:
            height = pts[:, 2] - corridor.floor_z
            height_factor = np.clip(height / 0.5, 0.0, 1.0)  # saturates at 0.5m
        else:
            height_factor = np.ones(N, dtype=np.float32)

        features["wall_anomaly"] = (
            lateral_closeness * protrusion_score * height_factor
        ).astype(np.float32)

    else:
        # No wall planes found — use neutral fallback values, NOT large sentinels.
        features["d_wall_min"] = np.full(N, 2.0, dtype=np.float32)
        features["d_wall_signed"] = np.zeros(N, dtype=np.float32)
        features["p_wall"] = np.zeros(N, dtype=np.float32)
        features["corridor_margin"] = np.zeros(N, dtype=np.float32)
        features["wall_anomaly"] = np.zeros(N, dtype=np.float32)

    # ---- Pillar plane features ----
    # These are computed from vertical planes that FAILED the elongation check
    # (compact in XY = pillar candidate). Gives the model explicit geometry
    # about nearby pillar-like structures.
    if corridor.pillar_planes:
        pillar_dists = np.column_stack([
            p.abs_distance(pts) for p in corridor.pillar_planes
        ])  # (N, n_pillar_planes)
        nearest_pillar_idx = np.argmin(pillar_dists, axis=1)
        d_pillar_min = pillar_dists[np.arange(N), nearest_pillar_idx]

        features["d_pillar_min"] = d_pillar_min.astype(np.float32)

        # Soft pillar-plane membership: Gaussian decay
        sigma_pillar = 0.15
        features["p_pillar_plane"] = np.exp(
            -0.5 * (d_pillar_min / sigma_pillar) ** 2
        ).astype(np.float32)

        # Elongation of nearest vertical surface (wall or pillar)
        # For each point, find the nearest vertical plane (wall or pillar)
        # and report its elongation ratio. Low = pillar, high = wall.
        all_vert_planes = corridor.walls + corridor.pillar_planes
        if all_vert_planes:
            all_vert_dists = np.column_stack([
                p.abs_distance(pts) for p in all_vert_planes
            ])
            nearest_vert_idx = np.argmin(all_vert_dists, axis=1)
            nearest_vert_elongation = np.array([
                all_vert_planes[idx].xy_elongation for idx in nearest_vert_idx
            ])
            # Log-scale elongation: log(1)=0 for pillar, log(10)=2.3 for wall
            features["nearest_vert_elongation"] = np.log1p(
                nearest_vert_elongation
            ).astype(np.float32)
        else:
            features["nearest_vert_elongation"] = np.zeros(N, dtype=np.float32)

        features["has_pillar_plane"] = np.full(N, 1.0, dtype=np.float32)
    else:
        features["d_pillar_min"] = np.full(N, 2.0, dtype=np.float32)
        features["p_pillar_plane"] = np.zeros(N, dtype=np.float32)
        features["has_pillar_plane"] = np.zeros(N, dtype=np.float32)

        # Elongation from walls only (no pillar planes detected)
        if corridor.walls:
            wall_dists_for_elong = np.column_stack([
                w.abs_distance(pts) for w in corridor.walls
            ])
            nearest_wall_for_elong = np.argmin(wall_dists_for_elong, axis=1)
            elong_vals = np.array([
                corridor.walls[idx].xy_elongation for idx in nearest_wall_for_elong
            ])
            features["nearest_vert_elongation"] = np.log1p(elong_vals).astype(np.float32)
        else:
            features["nearest_vert_elongation"] = np.zeros(N, dtype=np.float32)

    # ---- Angular features (optional, requires local normals) ----
    if compute_normals and N >= 3:
        normals = estimate_local_normals(pts, k=normal_k)

        if corridor.floor is not None:
            # Angle between local normal and floor normal
            cos_floor = np.abs(np.sum(normals * corridor.floor.normal, axis=1))
            cos_floor = np.clip(cos_floor, 0.0, 1.0)
            features["floor_ang"] = np.arccos(cos_floor).astype(np.float32)
        else:
            features["floor_ang"] = np.full(N, np.pi / 2, dtype=np.float32)

        if corridor.walls:
            # Angle to nearest wall normal
            nearest_wall_normal = np.array([
                corridor.walls[idx].normal for idx in nearest_wall_idx
            ])
            cos_wall = np.abs(np.sum(normals * nearest_wall_normal, axis=1))
            cos_wall = np.clip(cos_wall, 0.0, 1.0)
            features["wall_ang"] = np.arccos(cos_wall).astype(np.float32)
        else:
            features["wall_ang"] = np.full(N, np.pi / 2, dtype=np.float32)
    else:
        features["floor_ang"] = np.full(N, np.pi / 2, dtype=np.float32)
        features["wall_ang"] = np.full(N, np.pi / 2, dtype=np.float32)

    # ---- Plane fit quality features ----
    # Continuous inlier-fraction for each plane type (0.0 when not found).
    # Lets the model soft-weight structural features rather than relying solely
    # on the binary has_floor / has_wall flags.
    features["wall_plane_quality"] = np.full(
        N, corridor.wall_fit_max_inlier_frac, dtype=np.float32
    )
    features["floor_plane_quality"] = np.full(
        N, corridor.floor_fit_inlier_frac, dtype=np.float32
    )

    # ---- Plane availability indicators ----
    # Binary flags so the model knows when geometry features are from real
    # planes vs fallback values. Critical when ~40-50% of frames have no planes.
    features["has_floor"] = np.full(N, 1.0 if corridor.floor is not None else 0.0,
                                    dtype=np.float32)
    features["has_wall"] = np.full(N, 1.0 if corridor.walls else 0.0,
                                   dtype=np.float32)

    # ---- Mount-invariant coordinate features ----
    # Raw x, y, z, elevation_deg are mount-sensitive — if the radar shifts
    # position or tilt between sessions, these change for the same physical
    # scene. Replace with coordinates relative to fitted planes.
    #
    # z_rel: height above floor plane (= z_above_floor, already computed).
    #        When no floor: fall back to z relative to cloud centroid.
    # x_rel: lateral position relative to corridor center.
    #        When no walls: fall back to x relative to cloud centroid.
    # range_rel: range unchanged (mount-invariant — distance from radar).
    # elev_rel: elevation angle relative to floor plane normal.
    #        When no floor: fall back to raw elevation (atan2(z, xy_range)).

    # z_rel — already computed as z_above_floor, but provide a better
    # fallback for no-floor frames: use cloud centroid Z instead of raw Z
    if corridor.floor is not None:
        features["z_rel"] = features["z_above_floor"].copy()
    else:
        z_centroid = float(np.median(pts[:, 2])) if N > 0 else 0.0
        features["z_rel"] = (pts[:, 2] - z_centroid).astype(np.float32)

    # x_rel — lateral position relative to corridor center or cloud centroid
    if corridor.walls:
        if len(corridor.walls) >= 2:
            features["x_rel"] = (pts[:, 0] - corridor.centerline_y).astype(np.float32)
        else:
            # Single wall: distance from wall as lateral reference
            features["x_rel"] = corridor.walls[0].signed_distance(pts).astype(np.float32)
    else:
        x_centroid = float(np.median(pts[:, 0])) if N > 0 else 0.0
        features["x_rel"] = (pts[:, 0] - x_centroid).astype(np.float32)

    # y_rel — forward position relative to cloud centroid
    # (range direction is always relative to radar, but centering removes
    # any systematic offset from where the radar sits on the rig)
    y_centroid = float(np.median(pts[:, 1])) if N > 0 else 0.0
    features["y_rel"] = (pts[:, 1] - y_centroid).astype(np.float32)

    # elev_rel — elevation angle relative to floor plane
    if corridor.floor is not None:
        # Project each point onto the floor normal to get "height above floor"
        # then compute elevation as atan2(height, horizontal_range)
        floor_dist = corridor.floor.signed_distance(pts)  # height above floor plane
        xy_range = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        xy_range = np.maximum(xy_range, 1e-6)
        features["elev_rel"] = np.arctan2(floor_dist, xy_range).astype(np.float32)
    else:
        # Fallback: elevation from cloud centroid
        z_c = float(np.median(pts[:, 2])) if N > 0 else 0.0
        xy_range = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        xy_range = np.maximum(xy_range, 1e-6)
        features["elev_rel"] = np.arctan2(pts[:, 2] - z_c, xy_range).astype(np.float32)

    return features


# =============================================================================
# Feature column names (for dataset builders)
# =============================================================================

SOFT_STRUCTURAL_FEATURE_COLS = [
    "p_floor",
    "p_wall",
    "d_floor",
    "d_wall_min",
    "d_wall_signed",
    "z_above_floor",
    "corridor_margin",
    "wall_anomaly",
    "floor_ang",
    "wall_ang",
    "wall_plane_quality",
    "floor_plane_quality",
    "has_floor",
    "has_wall",
    "z_rel",
    "x_rel",
    "y_rel",
    "elev_rel",
    "d_pillar_min",
    "p_pillar_plane",
    "nearest_vert_elongation",
    "has_pillar_plane",
]
"""Ordered list of feature columns appended by this module."""


# =============================================================================
# Convenience: full pipeline for one frame group
# =============================================================================

def process_frame_group(
    anchor_pts: np.ndarray,
    aggregated_pts: np.ndarray,
    floor_dist_thresh: float = 0.06,
    wall_dist_thresh: float = 0.08,
    ransac_iters: int = 300,
    compute_normals: bool = True,
    normal_k: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[CorridorPlanes, dict[str, np.ndarray]]:
    """
    End-to-end: fit corridor planes on the aggregated cloud, then compute
    soft structural features for the anchor frame's points.

    Parameters
    ----------
    anchor_pts : (N, 3) — current-frame radar points (the ones being labeled)
    aggregated_pts : (M, 3) — temporally-aggregated cloud (anchor + neighbors,
                     motion-compensated back to anchor frame)
    floor_dist_thresh, wall_dist_thresh : RANSAC thresholds
    ransac_iters : RANSAC iterations per plane fit
    compute_normals : estimate local normals for angular features
    normal_k : KNN count for normal estimation
    rng : numpy RNG for reproducibility

    Returns
    -------
    corridor : CorridorPlanes (fitted planes, for diagnostics / visualization)
    features : dict of feature_name → (N,) arrays (one entry per anchor point)
    """
    corridor = fit_corridor_planes(
        aggregated_pts,
        floor_dist_thresh=floor_dist_thresh,
        wall_dist_thresh=wall_dist_thresh,
        ransac_iters=ransac_iters,
        rng=rng,
    )

    features = compute_soft_structural_features(
        anchor_pts,
        corridor,
        compute_normals=compute_normals,
        normal_k=normal_k,
    )

    return corridor, features


# =============================================================================
# Diagnostic summary (for build script logging)
# =============================================================================

def corridor_summary(corridor: CorridorPlanes) -> str:
    """One-line summary of fitted corridor geometry."""
    parts = []
    if corridor.floor is not None:
        parts.append(
            f"floor(z={corridor.floor_z:.3f}m, "
            f"inl={corridor.floor.inlier_frac:.0%})"
        )
    else:
        parts.append("floor(NONE)")

    for i, w in enumerate(corridor.walls):
        parts.append(
            f"wall{i}(inl={w.inlier_frac:.0%}, elong={w.xy_elongation:.1f}, span={w.xy_span:.2f}m)"
        )
    if not corridor.walls:
        parts.append("walls(NONE)")

    for i, p in enumerate(corridor.pillar_planes):
        parts.append(
            f"pillar{i}(inl={p.inlier_frac:.0%}, elong={p.xy_elongation:.1f}, span={p.xy_span:.2f}m)"
        )

    parts.append(f"width={corridor.corridor_width:.2f}m")
    return "  ".join(parts)
