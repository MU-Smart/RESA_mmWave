#!/usr/bin/env python3
"""Dataset building and validation tools for the radar navigation pipeline.

Subcommands:

    build-points       Build a labeled point dataset with session holdout split.
    build-patches      Build point-aligned RD/RA patch shards from labeled point CSVs and radar sidecars.
    validate           Validate point and patch datasets before training.

Usage:
    python dataset_tools.py build-points --help
    python dataset_tools.py build-patches --help
    python dataset_tools.py validate --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
CANON_DIR = THIS_FILE.parent.parent
REPO_ROOT = THIS_FILE.parents[2]
IO_ROOT = THIS_FILE.parents[3]
for path in (CANON_DIR, REPO_ROOT, IO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# =========================================================================
# Shared constants
# =========================================================================

SESSION_PREFIX_DEFAULT = "session_2026-04-28_"
CLASS_NAMES = {"structure", "floor", "human"}
FEATURE_COLS = [
    "x", "y", "z",
    "range_m", "azimuth_deg", "elevation_deg",
    "doppler", "snr",
    "local_density", "persist_score",
    "z_norm_range", "frame_doppler_abs", "frame_doppler_std",
    "z_above_floor", "corridor_margin", "wall_anomaly",
    "floor_ang", "wall_ang", "has_floor", "has_wall",
]

# =========================================================================
# build-points subcommand
# =========================================================================

LABEL_REMAP = {
    "wall": "structure",
    "door": "structure",
    "pillar": "structure",
    "box_like": "structure",
    "floor": "floor",
    "human": "human",
    "structure": "structure",
}
FRAME_COL = "radar_frame_num"
DENSITY_RADIUS_M = 0.5
WINDOW_SIZE = 3

META_COLS = [
    "session", "radar_frame_num", "label_frame_num",
    "range_bin", "doppler_bin",
    "bucket", "bucket_3class",
    "video_frame_index", "match_time_diff_us", "u", "cam_v_px",
    "ade_id", "ade_name", "maj_frac", "cam_z", "split",
]

FUSION_META_COLS = [
    "motion_label", "spatiotemporal_weight",
    "temporal_persistence_frames", "ghost_score",
    "static_confidence", "final_train_weight",
    "ego_available", "ego_doppler_residual_mps",
]

REVIEW_META_COLS = [
    "accumulatable_label", "return_type_label",
    "review_confidence", "review_source",
]

DEPTH_META_COLS = [
    "p_depth_match",
    "depth_corr_available", "depth_corr_in_fov",
    "depth_corr_valid_px", "depth_corr_patch_px",
    "depth_corr_radar_z_m", "depth_corr_min_m", "depth_corr_median_m",
    "depth_corr_selected_m", "depth_corr_residual_m", "depth_corr_abs_residual_m",
    "depth_corr_sigma_m", "depth_corr_quality",
    "depth_foreground_occluded", "depth_missing_reason",
    "depth_candidate_cluster_id", "depth_candidate_cluster_size",
    "depth_candidate_cluster_radius_m",
    "depth_candidate_cluster_min_depth_match",
    "depth_candidate_cluster_mean_p_dir_doppler",
    "matched_video_frame_index", "matched_depth_frame_index",
    "cam_u_depth", "cam_v_depth",
    "cam_semantic_bucket", "cam_semantic_conf",
]

TEACHER_META_COLS = [
    "depth_teacher_eligible",
    "depth_teacher_acc_label", "depth_teacher_return_type",
    "depth_teacher_weight", "depth_teacher_reason", "depth_teacher_source",
    "ghost_score_combined",
]

BRANCH1_POINT_LABEL_COLS = [
    "bucket_3class_legacy",
    "bucket_4class",
    "point_label_source",
    "point_label_rule",
    "point_label_weight",
    "branch1_label_version",
    "branch1_no_human_session",
]


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
    )


def _discover_sessions(root: Path, labeled_csv_name: str = "labeled_radar_points_v4.csv", session_prefix: str = SESSION_PREFIX_DEFAULT) -> list[Path]:
    sessions = []
    for session_dir in sorted(root.glob(f"{session_prefix}*")):
        if not session_dir.is_dir():
            continue
        raw_csv = session_dir / f"{session_dir.name}.csv"
        label_csv = session_dir / labeled_csv_name
        tensor_npz = session_dir / f"{session_dir.name}_radar_tensors.npz"
        sidecar_manifest = session_dir / "hybrid_rd" / "manifest.json"
        if raw_csv.exists() and label_csv.exists() and tensor_npz.exists() and sidecar_manifest.exists():
            # Skip sessions whose labeled CSV has no data rows (failed autolabeler runs)
            try:
                if sum(1 for _ in open(label_csv)) <= 1:
                    print(f"  [SKIP] {session_dir.name}: labeled CSV is empty")
                    continue
            except OSError:
                continue
            sessions.append(session_dir)
    return sessions


def _add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "radar_frame_num" not in out.columns and "frame_num" in out.columns:
        out["radar_frame_num"] = out["frame_num"]
    if "doppler" not in out.columns and "doppler_mps" in out.columns:
        out["doppler"] = out["doppler_mps"]
    if "snr" not in out.columns and "power_snr" in out.columns:
        out["snr"] = out["power_snr"]
    if "v_raw" not in out.columns:
        out["v_raw"] = _numeric(out, "v", 0.0)
    if "range_m" not in out.columns:
        out["range_m"] = np.sqrt(out["x"] ** 2 + out["y"] ** 2 + out["z"] ** 2)
    if "elevation_deg" not in out.columns:
        ratio = np.clip(out["z"] / out["range_m"].replace(0, np.nan), -1.0, 1.0)
        out["elevation_deg"] = np.degrees(np.arcsin(ratio.fillna(0.0)))
    out["z_norm_range"] = (out["z"] / out["range_m"].replace(0, np.nan)).fillna(0.0)
    out["doppler"] = _numeric(out, "doppler", 0.0).astype(np.float32)
    out["snr"] = _numeric(out, "snr", 0.0).astype(np.float32)
    return out


def _add_local_density(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    densities = np.zeros(len(out), dtype=np.float32)
    for _, grp in out.groupby(FRAME_COL, sort=False):
        pos = out.index.get_indexer(grp.index)
        pts = grp[["x", "y", "z"]].to_numpy(dtype=np.float32)
        if len(pts) > 1:
            diff = pts[:, None, :] - pts[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            counts = np.maximum((dists < DENSITY_RADIUS_M).sum(axis=1) - 1, 0)
            densities[pos] = (np.log1p(counts) / np.log1p(100)).astype(np.float32)
    out["local_density"] = densities
    return out


def _add_frame_doppler_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["frame_doppler_abs"] = (
        out.groupby(FRAME_COL)["doppler"].transform(lambda x: x.abs().mean()).astype(np.float32)
    )
    out["frame_doppler_std"] = (
        out.groupby(FRAME_COL)["doppler"].transform(lambda x: x.std()).fillna(0.0).astype(np.float32)
    )
    return out


def _add_soft_structural(df: pd.DataFrame, *, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    out = df.copy()
    soft_cols = [
        "z_above_floor", "corridor_margin", "wall_anomaly",
        "floor_ang", "wall_ang", "has_floor", "has_wall",
    ]
    for col in soft_cols:
        out[col] = 0.0
    out["floor_ang"] = np.pi / 2
    out["wall_ang"] = np.pi / 2

    frames = sorted(int(f) for f in out[FRAME_COL].dropna().unique())
    frame_pts = {
        frame: out[out[FRAME_COL] == frame][["x", "y", "z"]].to_numpy(dtype=np.float32)
        for frame in frames
    }
    rng = np.random.default_rng(42)
    half = window_size // 2
    for i, anchor in enumerate(frames):
        anchor_pts = frame_pts[anchor]
        if len(anchor_pts) < 3:
            continue
        neighbours = frames[max(0, i - half): i + half + 1]
        agg_pts = np.vstack([frame_pts[f] for f in neighbours if len(frame_pts[f]) > 0])
        try:
            from perception.corridor_soft_structural import process_frame_group
            _, feats = process_frame_group(anchor_pts, agg_pts, rng=rng)
        except Exception:
            continue
        mask = out[FRAME_COL] == anchor
        n = int(mask.sum())
        for col in soft_cols:
            values = feats.get(col)
            if values is not None and len(values) == n:
                out.loc[mask, col] = values
    return out


def _prepare_raw(
    raw_csv: Path,
    *,
    min_range_m: float | None,
    max_range_m: float | None,
    min_z_m: float | None,
    persist_score: float,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv)
    required = {FRAME_COL, "x", "y", "z", "range_bin", "doppler_bin"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{raw_csv} missing required raw columns: {sorted(missing)}")
    for col in ("x", "y", "z", "range_m", "azimuth_deg", "elevation_deg", "range_bin", "doppler_bin"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["x", "y", "z", FRAME_COL, "range_bin", "doppler_bin"]).copy()
    if min_range_m is not None:
        raw = raw[raw["range_m"] >= float(min_range_m)].copy()
    if max_range_m is not None:
        raw = raw[raw["range_m"] <= float(max_range_m)].copy()
    if min_z_m is not None:
        raw = raw[raw["z"] >= float(min_z_m)].copy()
    raw[FRAME_COL] = raw[FRAME_COL].astype(np.int64)
    raw["range_bin"] = raw["range_bin"].astype(np.int64)
    raw["doppler_bin"] = raw["doppler_bin"].astype(np.int64)
    raw = _add_base_features(raw)
    raw = _add_local_density(raw)
    raw = _add_frame_doppler_stats(raw)
    raw = _add_soft_structural(raw)
    raw["persist_score"] = float(persist_score)
    return raw


def _prepare_labels(label_csv: Path, session_name: str) -> pd.DataFrame:
    labels = pd.read_csv(label_csv)
    required = {FRAME_COL, "x", "y", "z", "bucket", "range_bin", "doppler_bin"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{label_csv} missing required label columns: {sorted(missing)}")
    labels = labels.copy()
    labels["session"] = session_name
    labels["bucket"] = labels["bucket"].astype(str).str.strip()
    if "bucket_3class" in labels.columns:
        labels["bucket_3class"] = labels["bucket_3class"].astype(str).str.strip()
        missing_3class = labels["bucket_3class"].isna() | ~labels["bucket_3class"].isin(CLASS_NAMES)
        labels.loc[missing_3class, "bucket_3class"] = labels.loc[missing_3class, "bucket"].map(LABEL_REMAP)
    else:
        labels["bucket_3class"] = labels["bucket"].map(LABEL_REMAP)
    if labels["bucket_3class"].isna().any():
        bad = sorted(labels.loc[labels["bucket_3class"].isna(), "bucket"].dropna().unique().tolist())
        raise ValueError(f"{label_csv} has unmapped bucket labels: {bad}")
    for col in (FRAME_COL, "range_bin", "doppler_bin"):
        labels[col] = pd.to_numeric(labels[col], errors="coerce")
    labels = labels.dropna(subset=[FRAME_COL, "range_bin", "doppler_bin", "x", "y", "z"]).copy()
    labels[FRAME_COL] = labels[FRAME_COL].astype(np.int64)
    labels["range_bin"] = labels["range_bin"].astype(np.int64)
    labels["doppler_bin"] = labels["doppler_bin"].astype(np.int64)
    if "v" in labels.columns and "cam_v_px" not in labels.columns:
        labels = labels.rename(columns={"v": "cam_v_px"})
    return labels


def _join_labels_to_raw(raw: pd.DataFrame, labels: pd.DataFrame, session_name: str) -> pd.DataFrame:
    label_j = labels.copy()
    for col in ("x", "y", "z"):
        label_j[f"_{col}j"] = pd.to_numeric(label_j[col], errors="coerce").round(4)
    keys = [FRAME_COL, "range_bin", "doppler_bin", "_xj", "_yj", "_zj"]
    label_cols = [
        "session", FRAME_COL, "range_bin", "doppler_bin", "_xj", "_yj", "_zj",
        "bucket", "bucket_3class", "video_frame_index", "match_time_diff_us",
        "u", "cam_v_px", "ade_id", "ade_name", "maj_frac", "cam_z",
    ] + FUSION_META_COLS + REVIEW_META_COLS + DEPTH_META_COLS + TEACHER_META_COLS + BRANCH1_POINT_LABEL_COLS
    label_cols = [c for c in label_cols if c in label_j.columns]

    def merge_with_raw_frame(raw_frame_col: str) -> pd.DataFrame:
        raw_j = raw.copy()
        if raw_frame_col != FRAME_COL:
            raw_j[FRAME_COL] = pd.to_numeric(raw_j[raw_frame_col], errors="coerce")
            raw_j = raw_j.dropna(subset=[FRAME_COL]).copy()
            raw_j[FRAME_COL] = raw_j[FRAME_COL].astype(np.int64)
        for col in ("x", "y", "z"):
            raw_j[f"_{col}j"] = pd.to_numeric(raw_j[col], errors="coerce").round(4)
        return raw_j.merge(
            label_j[label_cols],
            on=keys,
            how="inner",
            suffixes=("", "_label"),
        )

    attempts: list[tuple[str, pd.DataFrame]] = []
    frame_candidates = [FRAME_COL]
    if "frame_num" in raw.columns and "frame_num" not in frame_candidates:
        frame_candidates.append("frame_num")

    best_raw_frame_col = FRAME_COL
    best_merged = pd.DataFrame()
    for raw_frame_col in frame_candidates:
        merged = merge_with_raw_frame(raw_frame_col)
        attempts.append((raw_frame_col, merged))
        if len(merged) > len(best_merged):
            best_raw_frame_col = raw_frame_col
            best_merged = merged
        if len(merged) / max(len(labels), 1) >= 0.95:
            best_raw_frame_col = raw_frame_col
            best_merged = merged
            break

    merged = best_merged
    joined_fraction = len(merged) / max(len(labels), 1)
    if merged.empty:
        details = ", ".join(f"{frame_col}={len(candidate)}" for frame_col, candidate in attempts)
        raise ValueError(f"{session_name}: no labels matched raw radar rows ({details}).")
    if joined_fraction < 0.95:
        details = ", ".join(f"{frame_col}={len(candidate)}" for frame_col, candidate in attempts)
        raise ValueError(
            f"{session_name}: only {len(merged)}/{len(labels)} labels matched raw rows "
            f"({joined_fraction:.1%}). Tried raw frame columns: {details}. "
            f"Check parser, filters, and frame offset."
        )
    if best_raw_frame_col != FRAME_COL:
        print(
            f"  [FRAME] matched labels against raw {best_raw_frame_col}; "
            f"keeping label {FRAME_COL} for output/patch indexing",
            flush=True,
        )
    merged["session"] = session_name
    merged["label_frame_num"] = merged[FRAME_COL].astype(np.int64)
    return merged.drop(columns=["_xj", "_yj", "_zj"], errors="ignore")


def _validate_points(df: pd.DataFrame, *, train_sessions: list[str], val_sessions: list[str], session_prefix: str = SESSION_PREFIX_DEFAULT) -> None:
    allowed = set(train_sessions) | set(val_sessions)
    sessions = set(df["session"].astype(str).unique())
    if sessions != allowed:
        raise ValueError(f"Unexpected session set. Missing={sorted(allowed - sessions)} extra={sorted(sessions - allowed)}")
    if any(not s.startswith(session_prefix) for s in sessions):
        raise ValueError(f"Non-{session_prefix.rstrip('_')} session found in dataset.")
    split_sessions = df.groupby("session")["split"].nunique()
    bad_split = split_sessions[split_sessions != 1]
    if not bad_split.empty:
        raise ValueError(f"Sessions assigned to multiple splits: {bad_split.to_dict()}")
    actual_val = set(df.loc[df["split"] == "val", "session"].unique())
    if actual_val != set(val_sessions):
        raise ValueError(f"Validation session mismatch: {sorted(actual_val)} != {sorted(val_sessions)}")
    if df["bucket_3class"].isna().any() or not df["bucket_3class"].isin(CLASS_NAMES).all():
        raise ValueError("Null or invalid bucket_3class values remain.")
    if "bucket_4class" in df.columns:
        allowed_4class = {"structure", "floor", "human_candidate", "ghost_return"}
        if df["bucket_4class"].isna().any() or not df["bucket_4class"].isin(allowed_4class).all():
            raise ValueError("Null or invalid bucket_4class values remain.")
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Feature column has NaNs: {col}")
    if not df["range_bin"].between(0, 255).all():
        raise ValueError("range_bin outside [0, 255].")
    if not df["doppler_bin"].between(0, 31).all():
        raise ValueError("doppler_bin outside [0, 31].")
    zero_frac = float((df["range_bin"] == 0).mean())
    if zero_frac > 0.05:
        raise ValueError(f"range_bin collapsed near zero ({zero_frac:.1%} rows are zero).")
    key = ["session", FRAME_COL, "x", "y", "z", "range_bin", "doppler_bin"]
    conflicts = df.groupby(key, dropna=False)["bucket_3class"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting duplicate point labels: {len(conflicts)} groups.")


def _write_outputs(df: pd.DataFrame, out_dir: Path, train_sessions: list[str], val_sessions: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [
        c for c in (
            META_COLS
            + FUSION_META_COLS
            + REVIEW_META_COLS
            + DEPTH_META_COLS
            + TEACHER_META_COLS
            + BRANCH1_POINT_LABEL_COLS
            + FEATURE_COLS
        )
        if c in df.columns
    ]
    all_path = out_dir / "points.csv"
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    legacy_all_path = out_dir / "apr28_points.csv"
    legacy_train_path = out_dir / "apr28_train.csv"
    legacy_val_path = out_dir / "apr28_val.csv"
    df[out_cols].to_csv(all_path, index=False)
    df[df["split"] == "train"][out_cols].to_csv(train_path, index=False)
    df[df["split"] == "val"][out_cols].to_csv(val_path, index=False)
    # Backward-compatible aliases for existing scripts/docs that used the original
    # April-28-only filenames.
    df[out_cols].to_csv(legacy_all_path, index=False)
    df[df["split"] == "train"][out_cols].to_csv(legacy_train_path, index=False)
    df[df["split"] == "val"][out_cols].to_csv(legacy_val_path, index=False)
    fusion_cols_present = [c for c in FUSION_META_COLS if c in df.columns]
    review_cols_present = [c for c in REVIEW_META_COLS if c in df.columns]
    depth_cols_present = [c for c in DEPTH_META_COLS if c in df.columns]
    teacher_cols_present = [c for c in TEACHER_META_COLS if c in df.columns]
    branch1_point_label_cols_present = [c for c in BRANCH1_POINT_LABEL_COLS if c in df.columns]
    manifest = {
        "kind": "point_dataset",
        "point_csv": str(all_path),
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "legacy_point_csv": str(legacy_all_path),
        "legacy_train_csv": str(legacy_train_path),
        "legacy_val_csv": str(legacy_val_path),
        "source_sessions": train_sessions + val_sessions,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "frame_number_offset": 0,
        "label_remap": LABEL_REMAP,
        "feature_cols": FEATURE_COLS,
        "fusion_cols": fusion_cols_present,
        "review_cols": review_cols_present,
        "depth_cols": depth_cols_present,
        "teacher_cols": teacher_cols_present,
        "branch1_point_label_cols": branch1_point_label_cols_present,
        "has_final_train_weight": "final_train_weight" in df.columns,
        "has_depth_teacher": bool(teacher_cols_present),
        "accumulatable_counts": (
            df["accumulatable_label"].astype(str).str.lower().value_counts().to_dict()
            if "accumulatable_label" in df.columns else {}
        ),
        "depth_teacher_reason_counts": (
            df["depth_teacher_reason"].astype(str).value_counts().to_dict()
            if "depth_teacher_reason" in df.columns else {}
        ),
        "class_counts": {
            split: df[df["split"] == split]["bucket_3class"].value_counts().to_dict()
            for split in ("train", "val")
        },
        "class_counts_4class": (
            {
                split: df[df["split"] == split]["bucket_4class"].value_counts().to_dict()
                for split in ("train", "val")
            }
            if "bucket_4class" in df.columns else {}
        ),
        "n_rows": int(len(df)),
        "n_train_rows": int((df["split"] == "train").sum()),
        "n_val_rows": int((df["split"] == "val").sum()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "n_rows": manifest["n_rows"],
        "n_train_rows": manifest["n_train_rows"],
        "n_val_rows": manifest["n_val_rows"],
        "class_counts": manifest["class_counts"],
    }, indent=2))


def _build_build_points_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-points",
        help="Build a labeled point dataset with session holdout split.",
        description="Build a labeled point dataset with session holdout split.",
    )
    p.add_argument("--processing-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--holdout-last", type=int, default=5)
    p.add_argument("--persist-score", type=float, default=0.08)
    p.add_argument("--min-range-m", type=float, default=None)
    p.add_argument("--max-range-m", type=float, default=None)
    p.add_argument("--min-z-m", type=float, default=None)
    p.add_argument("--session-prefix", default=SESSION_PREFIX_DEFAULT,
                   help=f"Session directory name prefix for discovery (default: {SESSION_PREFIX_DEFAULT}).")
    p.add_argument(
        "--labeled-csv-name",
        default="labeled_radar_points_v4.csv",
        help="Filename of the per-session label CSV to consume (default: labeled_radar_points_v4.csv). "
             "Pass labeled_radar_points_v4_fused.csv to use spatiotemporal-weighted labels.",
    )


def _run_build_points(args: argparse.Namespace) -> None:
    sessions = _discover_sessions(args.processing_root.resolve(), labeled_csv_name=args.labeled_csv_name, session_prefix=args.session_prefix)
    if len(sessions) <= args.holdout_last:
        raise SystemExit(f"Need more than {args.holdout_last} sessions, found {len(sessions)}.")
    train_dirs = sessions[:-args.holdout_last]
    val_dirs = sessions[-args.holdout_last:]
    train_sessions = [p.name for p in train_dirs]
    val_sessions = [p.name for p in val_dirs]

    print(f"Sessions found  : {len(sessions)} (prefix: {args.session_prefix})")
    print(f"Train sessions   : {len(train_sessions)}")
    print(f"Val sessions     : {len(val_sessions)}")
    for session in val_sessions:
        print(f"  VAL {session}")

    pieces = []
    for split, split_dirs in (("train", train_dirs), ("val", val_dirs)):
        for session_dir in split_dirs:
            t0 = time.time()
            print(f"[{split}] {session_dir.name}", flush=True)
            raw = _prepare_raw(
                session_dir / f"{session_dir.name}.csv",
                min_range_m=args.min_range_m,
                max_range_m=args.max_range_m,
                min_z_m=args.min_z_m,
                persist_score=args.persist_score,
            )
            labels = _prepare_labels(session_dir / args.labeled_csv_name, session_dir.name)
            merged = _join_labels_to_raw(raw, labels, session_dir.name)
            merged["split"] = split
            pieces.append(merged)
            print(
                f"  rows={len(merged)} labels={len(labels)} "
                f"classes={merged['bucket_3class'].value_counts().to_dict()} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    df = pd.concat(pieces, ignore_index=True)
    _validate_points(df, train_sessions=train_sessions, val_sessions=val_sessions, session_prefix=args.session_prefix)
    _write_outputs(df, args.output_dir.resolve(), train_sessions, val_sessions)


# =========================================================================
# build-patches subcommand
# =========================================================================

from models.hybrid_rd_patches import (
    RDPatchConfig,
    extract_patches_for_bins,
    frame_column,
    patch_manifest_payload,
    rd_frame_path,
    require_patch_columns,
    resolve_frame_index,
    session_column,
)
from models.hybrid_rd_runtime import (
    RuntimeRDPatchConfig,
    RuntimeRAPatchConfig,
    compute_rd_patch_scalars,
    extract_rdra_patches_and_scalars_for_frame_df,
)


RD_SCALAR_COLS = [
    "rd_entropy",
    "rd_doppler_spread",
    "rd_anisotropy",
    "rd_peak_ratio",
]


def _write_patch_shard(
    shard_path: Path,
    rd_patches: list[np.ndarray],
    row_indices: list[int],
    *,
    ra_patches: list[np.ndarray] | None = None,
) -> int:
    if not rd_patches:
        return 0
    payload: dict[str, np.ndarray] = {
        "rd_patches": np.stack(rd_patches, axis=0).astype(np.float32),
        "source_row_indices": np.asarray(row_indices, dtype=np.int64),
    }
    if ra_patches is not None:
        payload["ra_patches"] = np.stack(ra_patches, axis=0).astype(np.float32)
    np.savez_compressed(shard_path, **payload)
    return int(payload["rd_patches"].shape[0])


def _normalize_paths(value: Path | str | list[Path] | list[str]) -> list[Path]:
    if isinstance(value, (Path, str)):
        return [Path(value)]
    return [Path(p) for p in value]


def _resolve_sidecar_frame_path(
    sidecar_roots: list[Path], sidecar_subdirs: list[str],
    session_name: str, export_frame: int,
) -> Path | None:
    for root in sidecar_roots:
        for subdir in sidecar_subdirs:
            candidate = rd_frame_path(root, session_name, export_frame, sidecar_subdir=subdir)
            if candidate.exists():
                return candidate
    return None


def build_patch_dataset(
    *,
    point_csv: Path,
    sidecar_root: Path | list[Path],
    output_dir: Path,
    sidecar_subdir: str | list[str] = ("hybrid_rd", "branch3_hybrid_rd"),
    frame_number_offset: int = 0,
    patch_config: RDPatchConfig | None = None,
    shard_size: int = 50000,
    drop_invalid: bool = False,
    require_all_valid: bool = False,
    allowed_session_prefix: str | None = None,
    extract_rd_scalars: bool = False,
) -> dict[str, object]:
    cfg = patch_config or RDPatchConfig()
    if shard_size < 1:
        raise ValueError("shard_size must be positive.")
    ra_cfg = RuntimeRAPatchConfig(frame_number_offset=frame_number_offset)

    df = pd.read_csv(point_csv, low_memory=False)
    require_patch_columns(df.columns)
    sess_col = session_column(df.columns)
    frame_col = frame_column(df.columns)
    if allowed_session_prefix:
        sessions = df[sess_col].astype(str)
        keep_mask = sessions.str.startswith(allowed_session_prefix)
        n_filtered_out = int((~keep_mask).sum())
        if not keep_mask.any():
            raise ValueError(f"No rows matched allowed session prefix '{allowed_session_prefix}'.")
        df = df.loc[keep_mask].reset_index(drop=True)
    else:
        n_filtered_out = 0
    sidecar_roots = [p.resolve() for p in _normalize_paths(sidecar_root)]
    sidecar_subdirs = [str(s) for s in _normalize_paths(sidecar_subdir)]

    out_dir = output_dir.resolve()
    patches_dir = out_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    out_df = df.copy()
    out_df["rd_patch_valid"] = False
    out_df["rd_patch_shard"] = ""
    out_df["rd_patch_index"] = -1
    out_df["ra_patch_valid"] = False
    out_df["ra_patch_shard"] = ""
    out_df["ra_patch_index"] = -1
    if extract_rd_scalars:
        for col in RD_SCALAR_COLS:
            out_df[col] = np.float32(0.0)

    sidecar_cache: dict[Path, dict[str, np.ndarray] | None] = {}
    session_tensor_cache: dict[Path, object] = {}
    current_rd_patches: list[np.ndarray] = []
    current_ra_patches: list[np.ndarray] = []
    current_rows: list[int] = []
    shard_records: list[dict[str, object]] = []
    shard_idx = 0

    n_missing_frame = 0
    n_missing_bin = 0
    n_failed = 0
    n_valid = 0

    def flush_shard() -> None:
        nonlocal shard_idx, current_rd_patches, current_ra_patches, current_rows
        if not current_rd_patches:
            return
        shard_name = f"rdra_patches_{shard_idx:05d}.npz"
        shard_path = patches_dir / shard_name
        n_written = _write_patch_shard(
            shard_path,
            current_rd_patches,
            current_rows,
            ra_patches=current_ra_patches,
        )
        shard_records.append({"path": str(shard_path.relative_to(out_dir)), "n_patches": n_written})
        shard_idx += 1
        current_rd_patches = []
        current_ra_patches = []
        current_rows = []

    grouped = out_df.groupby([sess_col, frame_col], sort=False)
    for (session_name, frame_num), group in grouped:
        export_frame = resolve_frame_index(int(frame_num), frame_number_offset=frame_number_offset)
        frame_path = _resolve_sidecar_frame_path(sidecar_roots, sidecar_subdirs, str(session_name), export_frame)
        if frame_path is None:
            n_missing_frame += int(len(group))
            continue
        if frame_path not in sidecar_cache:
            if frame_path.exists():
                with np.load(frame_path) as payload:
                    if cfg.channel_name not in payload:
                        raise KeyError(f"{frame_path} does not contain '{cfg.channel_name}'")
                    sidecar_cache[frame_path] = {
                        "rd_power": np.asarray(payload[cfg.channel_name], dtype=np.float32),
                        "doppler_axis_mps": (
                            np.asarray(payload["doppler_axis_mps"], dtype=np.float32)
                            if "doppler_axis_mps" in payload else np.asarray([], dtype=np.float32)
                        ),
                    }
            else:
                sidecar_cache[frame_path] = None

        sidecar = sidecar_cache[frame_path]
        if sidecar is None:
            n_missing_frame += int(len(group))
            continue

        range_bins = pd.to_numeric(group["range_bin"], errors="coerce")
        doppler_bins = pd.to_numeric(group["doppler_bin"], errors="coerce")
        valid_bin_mask = range_bins.notna() & doppler_bins.notna()
        n_missing_bin += int((~valid_bin_mask).sum())
        if not valid_bin_mask.any():
            continue

        valid_group = group[valid_bin_mask]
        rd_rt_cfg = RuntimeRDPatchConfig(
            doppler_bins=cfg.doppler_bins,
            range_bins=cfg.range_bins,
            frame_number_offset=frame_number_offset,
            sidecar_subdir=str(frame_path.parents[1].name),
            clutter_mode="off",
            log_scale=True,
            include_rd_cube=False,
        )
        try:
            rd_patches, ra_patches, scalars = extract_rdra_patches_and_scalars_for_frame_df(
                valid_group,
                session_dir=frame_path.parents[2],
                frame_num=int(frame_num),
                rd_config=rd_rt_cfg,
                ra_config=ra_cfg,
                session_tensor_cache=session_tensor_cache,
            )
        except Exception as exc:
            n_failed += int(len(valid_group))
            raise RuntimeError(f"{session_name} frame {frame_num}: failed to extract RD/RA patches: {exc}") from exc

        if rd_patches.shape[0] != len(valid_group) or ra_patches.shape[0] != len(valid_group):
            raise RuntimeError(
                f"{session_name} frame {frame_num}: patch count mismatch "
                f"rd={rd_patches.shape[0]} ra={ra_patches.shape[0]} rows={len(valid_group)}"
            )

        for out_idx, (row_index, rd_patch, ra_patch) in enumerate(
            zip(valid_group.index.to_numpy(), rd_patches, ra_patches)
        ):
            if len(current_rd_patches) >= shard_size:
                flush_shard()
            rel_shard = f"patches/rdra_patches_{shard_idx:05d}.npz"
            patch_index = len(current_rd_patches)
            current_rd_patches.append(rd_patch.astype(np.float32))
            current_ra_patches.append(ra_patch.astype(np.float32))
            current_rows.append(int(row_index))
            out_df.at[row_index, "rd_patch_valid"] = True
            out_df.at[row_index, "rd_patch_shard"] = rel_shard
            out_df.at[row_index, "rd_patch_index"] = patch_index
            out_df.at[row_index, "ra_patch_valid"] = True
            out_df.at[row_index, "ra_patch_shard"] = rel_shard
            out_df.at[row_index, "ra_patch_index"] = patch_index
            if extract_rd_scalars:
                for col in RD_SCALAR_COLS:
                    out_df.at[row_index, col] = np.float32(scalars[out_idx].get(col, 0.0))
            n_valid += 1

    flush_shard()

    if require_all_valid and n_valid != len(out_df):
        raise RuntimeError(
            f"Patch dataset contract failed: not every row received a valid patch. "
            f"rows={len(out_df)} valid={n_valid} missing_frame={n_missing_frame} "
            f"missing_bin={n_missing_bin} failed_patch={n_failed}"
        )

    if drop_invalid:
        out_df = out_df[out_df["rd_patch_valid"] & out_df["ra_patch_valid"]].reset_index(drop=True)

    csv_out = out_dir / "points_with_rd_patch_index.csv"
    out_df.to_csv(csv_out, index=False)

    branch1_label_extra: dict[str, object] = {}
    if "bucket_4class" in out_df.columns:
        branch1_label_extra = {
            "label_col": "bucket_4class",
            "bucket_order": ["structure", "floor", "human_candidate", "ghost_return"],
            "sample_weight_col": "point_label_weight" if "point_label_weight" in out_df.columns else None,
            "class_counts_4class": {
                str(split): out_df[out_df["split"] == split]["bucket_4class"].value_counts().to_dict()
                for split in sorted(out_df["split"].dropna().astype(str).unique())
            } if "split" in out_df.columns else out_df["bucket_4class"].value_counts().to_dict(),
            "point_label_source_counts": (
                out_df["point_label_source"].astype(str).value_counts().to_dict()
                if "point_label_source" in out_df.columns else {}
            ),
            "branch1_label_version": (
                str(out_df["branch1_label_version"].dropna().iloc[0])
                if "branch1_label_version" in out_df.columns and not out_df["branch1_label_version"].dropna().empty
                else None
            ),
            "uses_ra_patches": True,
            "rd_patch_shape": [1, cfg.doppler_bins, cfg.range_bins],
            "ra_patch_shape": [1, ra_cfg.azimuth_bins, ra_cfg.range_bins],
            "ra_patch_source": str(ra_cfg.source),
            "ra_az_mode": str(ra_cfg.az_mode),
            "ra_az_fft_size": int(ra_cfg.az_fft_size),
            "ra_log_scale": bool(ra_cfg.log_scale),
            "ra_normalize_by_frame_max": bool(ra_cfg.normalize_by_frame_max),
            "ra_row_edge_mode": str(ra_cfg.row_edge_mode),
        }
        depth_cols = [
            c for c in (DEPTH_META_COLS + TEACHER_META_COLS + ["p_depth_match", "ghost_score_combined"])
            if c in out_df.columns
        ]
        if depth_cols:
            depth_summary: dict[str, object] = {
                "depth_columns_present": sorted(set(depth_cols)),
                "has_depth_teacher": True,
            }
            if "branch1_depth_teacher_matched" in out_df.columns:
                depth_summary["matched_rows"] = int(
                    pd.to_numeric(out_df["branch1_depth_teacher_matched"], errors="coerce").fillna(0).sum()
                )
            if "depth_teacher_eligible" in out_df.columns:
                eligible = out_df["depth_teacher_eligible"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
                depth_summary["eligible_rows"] = int(eligible.sum())
            if "depth_teacher_acc_label" in out_df.columns:
                depth_summary["depth_teacher_acc_label_counts"] = (
                    out_df["depth_teacher_acc_label"].astype(str).str.lower().value_counts().to_dict()
                )
            if "depth_teacher_reason" in out_df.columns:
                depth_summary["depth_teacher_reason_counts"] = (
                    out_df["depth_teacher_reason"].astype(str).value_counts().head(20).to_dict()
                )
            if "depth_teacher_weight" in out_df.columns:
                weights = _numeric(out_df, "depth_teacher_weight", 0.0)
                depth_summary["depth_teacher_weight_nonzero_rows"] = int((weights > 0).sum())
                depth_summary["depth_teacher_weight_max"] = float(weights.max()) if len(weights) else 0.0
            branch1_label_extra["depth_teacher_summary"] = depth_summary

    manifest = patch_manifest_payload(
        patch_config=cfg,
        sidecar_root=sidecar_root,
        sidecar_subdir=sidecar_subdir,
        frame_number_offset=frame_number_offset,
        extra={
            "kind": "hybrid_rdra_patch_dataset",
            "patch_source": "rd_power_true_ra",
            "rd_patch_source": "hybrid_rd_power",
            "ra_patch_source": str(ra_cfg.source),
            "point_csv": str(point_csv.resolve()),
            "output_csv": str(csv_out.relative_to(out_dir)),
            "drop_invalid": bool(drop_invalid),
            "require_all_valid": bool(require_all_valid),
            "allowed_session_prefix": allowed_session_prefix,
            "extract_rd_scalars": bool(extract_rd_scalars),
            "rd_scalar_cols": RD_SCALAR_COLS if extract_rd_scalars else [],
            "uses_ra_patches": True,
            "rd_patch_shape": [1, cfg.doppler_bins, cfg.range_bins],
            "ra_patch_shape": [1, ra_cfg.azimuth_bins, ra_cfg.range_bins],
            "ra_az_mode": str(ra_cfg.az_mode),
            "ra_az_fft_size": int(ra_cfg.az_fft_size),
            "ra_log_scale": bool(ra_cfg.log_scale),
            "ra_normalize_by_frame_max": bool(ra_cfg.normalize_by_frame_max),
            "ra_row_edge_mode": str(ra_cfg.row_edge_mode),
            "shard_size": int(shard_size),
            "n_rows_input": int(len(df)),
            "n_rows_filtered_out": int(n_filtered_out),
            "n_rows_output": int(len(out_df)),
            "n_valid_patches": int(n_valid),
            "n_missing_frame": int(n_missing_frame),
            "n_missing_bin": int(n_missing_bin),
            "n_failed_patch": int(n_failed),
            "sidecar_roots": [str(p) for p in sidecar_roots],
            "sidecar_subdirs": sidecar_subdirs,
            "shards": shard_records,
            "branch1_labels": branch1_label_extra,
        },
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _build_build_patches_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-patches",
        help="Build point-aligned RD/RA patch shards from labeled point CSVs and radar sidecars.",
        description="Build point-aligned RD/RA patch shards from labeled point CSVs and radar sidecars.",
    )
    p.add_argument("--point-csv", required=True, type=Path, help="Labeled point CSV with range_bin/doppler_bin.")
    p.add_argument("--sidecar-root", required=True, nargs="+", type=Path,
                   help="One or more roots containing per-session RD sidecar directories and session tensor npz files.")
    p.add_argument("--output-dir", required=True, type=Path, help="Output dataset directory.")
    p.add_argument("--sidecar-subdir", nargs="+", default=["hybrid_rd", "branch3_hybrid_rd"],
                   help="One or more RD sidecar subdirectories to search.")
    p.add_argument("--frame-number-offset", type=int, default=1, help="CSV frame number minus sidecar frame_index.")
    p.add_argument("--patch-doppler-bins", type=int, default=17)
    p.add_argument("--patch-range-bins", type=int, default=7)
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--drop-invalid", action="store_true")
    p.add_argument("--require-all-valid", action="store_true")
    p.add_argument("--allowed-session-prefix", default=None,
                   help="Fail if any point row session does not start with this prefix.")
    p.add_argument("--extract-rd-scalars", action="store_true",
                   help="Append RD scalar feature columns derived from each extracted RD patch.")


def _run_build_patches(args: argparse.Namespace) -> None:
    manifest = build_patch_dataset(
        point_csv=args.point_csv,
        sidecar_root=args.sidecar_root,
        output_dir=args.output_dir,
        sidecar_subdir=args.sidecar_subdir,
        frame_number_offset=args.frame_number_offset,
        patch_config=RDPatchConfig(doppler_bins=args.patch_doppler_bins, range_bins=args.patch_range_bins),
        shard_size=args.shard_size,
        drop_invalid=args.drop_invalid,
        require_all_valid=args.require_all_valid,
        allowed_session_prefix=args.allowed_session_prefix,
        extract_rd_scalars=args.extract_rd_scalars,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "n_rows_input": manifest["n_rows_input"],
        "n_rows_output": manifest["n_rows_output"],
        "n_valid_patches": manifest["n_valid_patches"],
        "n_missing_frame": manifest["n_missing_frame"],
        "n_missing_bin": manifest["n_missing_bin"],
        "n_failed_patch": manifest["n_failed_patch"],
        "uses_ra_patches": manifest.get("branch1_labels", {}).get("uses_ra_patches", True),
    }, indent=2))


# =========================================================================
# validate subcommand
# =========================================================================


def _load_expected_sessions(manifest_path: Path | None) -> tuple[set[str] | None, set[str] | None]:
    if manifest_path is None:
        return None, None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "train_sessions" not in payload or "val_sessions" not in payload:
        return None, None
    train_sessions = set(payload.get("train_sessions", []))
    val_sessions = set(payload.get("val_sessions", []))
    if not train_sessions and not val_sessions:
        return None, None
    return train_sessions, val_sessions


def _validate_point_csv(
    df: pd.DataFrame,
    *,
    train_sessions: set[str] | None,
    val_sessions: set[str] | None,
    session_prefix: str = SESSION_PREFIX_DEFAULT,
    allow_train_only: bool = False,
) -> None:
    sessions = set(df["session"].astype(str).unique())
    bad_sessions = sorted(s for s in sessions if not s.startswith(session_prefix))
    if bad_sessions:
        raise ValueError(f"Non-{session_prefix.rstrip('_')} sessions found: {bad_sessions[:10]}")
    if train_sessions is not None and val_sessions is not None:
        expected = train_sessions | val_sessions
        if sessions != expected:
            raise ValueError(f"Session mismatch. Missing={sorted(expected - sessions)} extra={sorted(sessions - expected)}")
        actual_val = set(df.loc[df["split"] == "val", "session"].astype(str).unique())
        if actual_val != val_sessions:
            raise ValueError(f"Validation sessions changed: {sorted(actual_val)} != {sorted(val_sessions)}")
        train_overlap = set(df.loc[df["split"] == "train", "session"].astype(str).unique()) & val_sessions
        if train_overlap:
            raise ValueError(f"Validation sessions leaked into train split: {sorted(train_overlap)}")
    actual_splits = set(df["split"].dropna().astype(str).unique())
    allowed_splits = {"train", "val"}
    extra_splits = actual_splits - allowed_splits
    if extra_splits:
        raise ValueError(f"Unexpected split values: {sorted(extra_splits)}")
    required_splits = {"train"} if allow_train_only else {"train", "val"}
    missing_splits = required_splits - actual_splits
    if missing_splits:
        if allow_train_only:
            raise ValueError(f"Expected split=train; missing split(s): {sorted(missing_splits)}")
        raise ValueError("Expected exactly split=train and split=val.")
    if df["bucket_3class"].isna().any() or not df["bucket_3class"].isin(CLASS_NAMES).all():
        raise ValueError("Invalid bucket_3class values.")
    if "bucket_4class" in df.columns:
        allowed_4class = {"structure", "floor", "human_candidate", "ghost_return"}
        if df["bucket_4class"].isna().any() or not df["bucket_4class"].isin(allowed_4class).all():
            raise ValueError("Invalid bucket_4class values.")
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"NaNs in feature column: {col}")
    if not pd.to_numeric(df["range_bin"], errors="coerce").between(0, 255).all():
        raise ValueError("range_bin outside [0, 255].")
    if not pd.to_numeric(df["doppler_bin"], errors="coerce").between(0, 31).all():
        raise ValueError("doppler_bin outside [0, 31].")
    zero_range_frac = float((pd.to_numeric(df["range_bin"], errors="coerce") == 0).mean())
    if zero_range_frac > 0.05:
        raise ValueError(f"range_bin appears collapsed to zero: {zero_range_frac:.1%}")
    key = ["session", "radar_frame_num", "x", "y", "z", "range_bin", "doppler_bin"]
    conflicts = df.groupby(key, dropna=False)["bucket_3class"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting labels for identical radar points: {len(conflicts)} groups")
    if "bucket_4class" in df.columns:
        conflicts_4class = df.groupby(key, dropna=False)["bucket_4class"].nunique()
        conflicts_4class = conflicts_4class[conflicts_4class > 1]
        if not conflicts_4class.empty:
            raise ValueError(f"Conflicting four-class labels for identical radar points: {len(conflicts_4class)} groups")


def _validate_patch_index(df: pd.DataFrame, dataset_root: Path) -> None:
    required = {"rd_patch_valid", "rd_patch_shard", "rd_patch_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Patch-index columns missing: {sorted(missing)}")
    if not df["rd_patch_valid"].astype(bool).all():
        raise ValueError("Some rows have rd_patch_valid=False.")
    has_ra = {"ra_patch_valid", "ra_patch_shard", "ra_patch_index"}.issubset(df.columns)
    if has_ra:
        if not df["ra_patch_valid"].astype(bool).all():
            raise ValueError("Some rows have ra_patch_valid=False.")
        if not (df["rd_patch_shard"].astype(str) == df["ra_patch_shard"].astype(str)).all():
            raise ValueError("RD and RA shard paths do not match row-for-row.")
        if not (pd.to_numeric(df["rd_patch_index"], errors="coerce") == pd.to_numeric(df["ra_patch_index"], errors="coerce")).all():
            raise ValueError("RD and RA patch indices do not match row-for-row.")
    shard_counts = df["rd_patch_shard"].astype(str).value_counts()
    for rel_path, expected_count in shard_counts.items():
        shard_path = dataset_root / rel_path
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing patch shard: {shard_path}")
        with np.load(shard_path) as payload:
            rd_patch_arr = np.asarray(payload["rd_patches"])
            n_patches = int(rd_patch_arr.shape[0])
            if not np.isfinite(rd_patch_arr).all():
                raise ValueError(f"Patch shard {rel_path} contains non-finite RD values.")
            if has_ra:
                if "ra_patches" not in payload:
                    raise ValueError(f"Patch shard {rel_path} is missing 'ra_patches'.")
                ra_patch_arr = np.asarray(payload["ra_patches"])
                if int(ra_patch_arr.shape[0]) != n_patches:
                    raise ValueError(f"RD/RA shard length mismatch in {rel_path}.")
                if not np.isfinite(ra_patch_arr).all():
                    raise ValueError(f"Patch shard {rel_path} contains non-finite RA values.")
        max_idx = int(pd.to_numeric(df.loc[df["rd_patch_shard"].astype(str) == rel_path, "rd_patch_index"]).max())
        if max_idx >= n_patches:
            raise ValueError(f"Patch index exceeds shard size in {rel_path}: max={max_idx} n={n_patches}")
        if expected_count > n_patches:
            raise ValueError(f"More CSV rows than patches referenced for {rel_path}: rows={expected_count} n={n_patches}")


def _build_validate_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "validate",
        help="Validate point and patch datasets before training.",
        description="Validate point and patch datasets before training.",
    )
    p.add_argument("--point-csv", required=True, type=Path)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--dataset-root", type=Path, default=None, help="Patch dataset root; enables shard checks.")
    p.add_argument("--session-prefix", default=SESSION_PREFIX_DEFAULT,
                   help=f"Expected session prefix for validation (default: {SESSION_PREFIX_DEFAULT}).")
    p.add_argument(
        "--allow-train-only",
        action="store_true",
        help="Allow datasets whose split column contains train rows only. Use for streaming training batches validated against a separate external validation dataset.",
    )


def _run_validate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.point_csv, low_memory=False)
    train_sessions, val_sessions = _load_expected_sessions(args.manifest)
    _validate_point_csv(
        df,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        session_prefix=args.session_prefix,
        allow_train_only=args.allow_train_only,
    )
    if args.dataset_root is not None:
        _validate_patch_index(df, args.dataset_root)
    print(json.dumps({
        "point_csv": str(args.point_csv),
        "rows": int(len(df)),
        "sessions": int(df["session"].nunique()),
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": {
            split: df[df["split"] == split]["bucket_3class"].value_counts().to_dict()
            for split in ("train", "val")
        },
        "class_counts_4class": (
            {
                split: df[df["split"] == split]["bucket_4class"].value_counts().to_dict()
                for split in ("train", "val")
            }
            if "bucket_4class" in df.columns else {}
        ),
        "patch_shards_checked": bool(args.dataset_root),
    }, indent=2))


# =========================================================================
# CLI dispatch
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_build_points_parser(subparsers)
    _build_build_patches_parser(subparsers)
    _build_validate_parser(subparsers)

    args = parser.parse_args()

    if args.command == "build-points":
        _run_build_points(args)
    elif args.command == "build-patches":
        _run_build_patches(args)
    elif args.command == "validate":
        _run_validate(args)


if __name__ == "__main__":
    main()
