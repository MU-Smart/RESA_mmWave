"""
radar_3dgcnn_common_doorwall_soft_structural.py

5-class door_as_wall with pillar elongation features.

Label space: wall (includes door), floor, pillar, human, box_like

Key innovation: the RANSAC wall fitting now includes an XY elongation check
that separates walls (elongated vertical planes, λ₁/λ₂ >> 1) from pillars
(compact vertical planes, λ₁/λ₂ ≈ 1). This gives the model explicit
geometric features for the wall/pillar distinction:
    - d_pillar_min: distance to nearest compact vertical plane
    - p_pillar_plane: soft membership in a compact vertical plane
    - nearest_vert_elongation: log elongation of nearest vertical surface
    - has_pillar_plane: whether any compact vertical planes were detected

Combined with the existing wall_anomaly, z_above_floor, and corridor_margin
features, the model now has strong geometric priors for pillar detection.
"""

import radar_3dgcnn_common_doorwall as _base

# -- 5-class door_as_wall label space (pillar stays separate) ---------------
BUCKET_ORDER = ["wall", "floor", "pillar", "human", "box_like"]
BUCKET_TO_IDX = {b: i for i, b in enumerate(BUCKET_ORDER)}
IDX_TO_BUCKET = {i: b for b, i in BUCKET_TO_IDX.items()}
N_CLASSES = len(BUCKET_ORDER)

LABEL_REMAP = {
    "door": "wall",
}

# -- Monkey-patch the base module's globals --------------------------------
_base.BUCKET_ORDER = BUCKET_ORDER
_base.BUCKET_TO_IDX = BUCKET_TO_IDX
_base.IDX_TO_BUCKET = IDX_TO_BUCKET
_base.N_CLASSES = N_CLASSES
_base.LABEL_REMAP = LABEL_REMAP

# -- Now import everything (with patched globals) --------------------------
from radar_3dgcnn_common_doorwall import *  # noqa: F401, F403

# -- Override feature list -------------------------------------------------
BASE_FEATURE_COLS = list(_base.FEATURE_COLS)

# Targeted features for 5-class with pillar elongation.
# Keep the wall/floor context separate from the pillar-specific branch so the
# trainer can run ablations without duplicating code.
CORE_SOFT_STRUCTURAL_COLS = [
    "z_above_floor",
    "corridor_margin",
    "wall_anomaly",
    "floor_ang",
    "wall_ang",
    "wall_plane_quality",
    "floor_plane_quality",
    "has_floor",
    "has_wall",
]

PILLAR_GEOMETRY_COLS = [
    "d_pillar_min",
    "p_pillar_plane",
    "nearest_vert_elongation",
    "has_pillar_plane",
]

FEATURE_COLS = BASE_FEATURE_COLS + CORE_SOFT_STRUCTURAL_COLS + PILLAR_GEOMETRY_COLS
N_FEATURES = len(FEATURE_COLS)
