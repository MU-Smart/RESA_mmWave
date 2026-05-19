#!/usr/bin/env python3
"""Prepare and audit Branch 1 four-class RD-patch training datasets.

The output label contract is:

    structure / floor / human_candidate / ghost_return

This tool is intentionally conservative. It does not infer ghosts from missing
depth alone. It can promote person-like or structure-like points to
``ghost_return`` only through explicit, high-confidence evidence from:

* controlled no-human sessions,
* depth-teacher consensus when eligible,
* or high ghost-score / low-directness fallback rules.

The primary use case is a clean apr28-only rebuild of the current Branch 1
architecture. Session allowlisting is supported so older mixed-session artifacts
can be treated as obsolete during dataset materialization.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FOUR_CLASS_ORDER = ["structure", "floor", "human_candidate", "ghost_return"]
RAW_TO_3CLASS = {
    "wall": "structure",
    "door": "structure",
    "pillar": "structure",
    "box_like": "structure",
    "floor": "floor",
    "human": "human",
    "structure": "structure",
}
THREE_TO_FOUR = {
    "structure": "structure",
    "floor": "floor",
    "human": "human_candidate",
}
DEFAULT_NO_HUMAN_PREFIXES = [
    "session_2026-04-21_",
    "session_2026-04-22_",
]
DEFAULT_GHOST_SCORE_THRESHOLD = 0.70
DEFAULT_P_DIR_GHOST_THRESHOLD = 0.20
GHOST_RETURN_TYPES = {
    "ghost",
    "ghost_return",
    "wall_multipath",
    "floor_bounce",
    "rig_scatter",
    "wheel_microdoppler",
    "depth_occluded",
    "no_human_person_like",
    "controlled_no_human_person_like",
}
DEPTH_TEACHER_COLS = [
    "p_depth_match",
    "depth_corr_available",
    "depth_corr_in_fov",
    "depth_corr_valid_px",
    "depth_corr_patch_px",
    "depth_corr_radar_z_m",
    "depth_corr_min_m",
    "depth_corr_median_m",
    "depth_corr_selected_m",
    "depth_corr_residual_m",
    "depth_corr_abs_residual_m",
    "depth_corr_sigma_m",
    "depth_corr_quality",
    "depth_foreground_occluded",
    "depth_missing_reason",
    "depth_candidate_cluster_id",
    "depth_candidate_cluster_size",
    "depth_candidate_cluster_radius_m",
    "depth_candidate_cluster_min_depth_match",
    "depth_candidate_cluster_mean_p_dir_doppler",
    "matched_video_frame_index",
    "matched_depth_frame_index",
    "cam_u_depth",
    "cam_v_depth",
    "cam_semantic_bucket",
    "cam_semantic_conf",
    "depth_teacher_eligible",
    "depth_teacher_acc_label",
    "depth_teacher_return_type",
    "depth_teacher_weight",
    "depth_teacher_reason",
    "depth_teacher_source",
    "ghost_score_combined",
]
DEPTH_MERGE_KEY_COLS = ["session", "radar_frame_num", "range_bin", "doppler_bin", "x", "y", "z"]


def _clean_strings(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _lower_strings(series: pd.Series) -> pd.Series:
    return _clean_strings(series).str.lower()


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .astype(np.float32)
    )


def _truthy(series: pd.Series) -> pd.Series:
    return _lower_strings(series).isin({"1", "true", "yes", "y"})


def _ensure_legacy_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bucket_3class" not in out.columns:
        if "bucket" not in out.columns:
            raise ValueError("Dataset needs either bucket_3class or bucket.")
        out["bucket_3class"] = _lower_strings(out["bucket"]).map(RAW_TO_3CLASS)
    out["bucket_3class"] = _lower_strings(out["bucket_3class"])
    bad = sorted(set(out["bucket_3class"].dropna()) - set(THREE_TO_FOUR))
    if bad:
        raise ValueError(f"Unmapped bucket_3class labels: {bad}")
    out["bucket_3class_legacy"] = out["bucket_3class"]
    return out


def _no_human_mask(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.Series:
    sessions = _clean_strings(df["session"])
    mask = pd.Series(False, index=df.index)
    for prefix in prefixes:
        if prefix:
            mask |= sessions.str.startswith(str(prefix))
    return mask


def _session_prefix_mask(df: pd.DataFrame, prefix: str | None) -> pd.Series:
    if not prefix:
        return pd.Series(True, index=df.index)
    return _clean_strings(df["session"]).str.startswith(prefix)


def _base_weight(df: pd.DataFrame) -> pd.Series:
    if "final_train_weight" in df.columns:
        return _numeric(df, "final_train_weight", 1.0).clip(0.0, 1.0)
    return pd.Series(1.0, index=df.index, dtype=np.float32)


def _merge_depth_teacher_columns(
    df: pd.DataFrame,
    depth_df: pd.DataFrame,
    *,
    round_decimals: int = 4,
) -> tuple[pd.DataFrame, dict[str, object]]:
    missing = [c for c in DEPTH_MERGE_KEY_COLS if c not in df.columns or c not in depth_df.columns]
    if missing:
        raise ValueError(f"Depth merge requires key columns in both datasets: missing {missing}")
    depth_cols = [c for c in DEPTH_TEACHER_COLS if c in depth_df.columns and c not in df.columns]
    existing_depth_cols = [c for c in DEPTH_TEACHER_COLS if c in df.columns]
    if not depth_cols and existing_depth_cols:
        return df.copy(), {
            "source": "already_present",
            "depth_columns_added": [],
            "depth_columns_already_present": existing_depth_cols,
            "matched_rows": int(df[existing_depth_cols].notna().any(axis=1).sum()),
            "source_rows": 0,
            "duplicate_depth_keys_dropped": 0,
        }
    if not depth_cols:
        return df.copy(), {
            "source": "none",
            "depth_columns_added": [],
            "depth_columns_already_present": [],
            "matched_rows": 0,
            "source_rows": int(len(depth_df)),
            "duplicate_depth_keys_dropped": 0,
        }

    def keyed(src: pd.DataFrame) -> pd.DataFrame:
        out = src.copy()
        out["_merge_session"] = _clean_strings(out["session"])
        for col in ("radar_frame_num", "range_bin", "doppler_bin"):
            out[f"_merge_{col}"] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        for col in ("x", "y", "z"):
            out[f"_merge_{col}"] = pd.to_numeric(out[col], errors="coerce").round(int(round_decimals))
        return out

    left = keyed(df)
    right = keyed(depth_df)
    merge_keys = [
        "_merge_session",
        "_merge_radar_frame_num",
        "_merge_range_bin",
        "_merge_doppler_bin",
        "_merge_x",
        "_merge_y",
        "_merge_z",
    ]
    before = len(right)
    right = right[merge_keys + depth_cols].drop_duplicates(subset=merge_keys, keep="first")
    merged = left.merge(right, on=merge_keys, how="left")
    matched = int(merged[depth_cols].notna().any(axis=1).sum())
    merged = merged.drop(columns=merge_keys, errors="ignore")
    merged["branch1_depth_teacher_matched"] = merged[depth_cols].notna().any(axis=1).astype(np.int8)
    return merged, {
        "source": "merged",
        "depth_columns_added": depth_cols,
        "depth_columns_already_present": existing_depth_cols,
        "matched_rows": matched,
        "source_rows": int(len(depth_df)),
        "duplicate_depth_keys_dropped": int(before - len(right)),
        "round_decimals": int(round_decimals),
    }


def materialize_fourclass(
    df: pd.DataFrame,
    *,
    no_human_prefixes: list[str],
    no_human_val_sessions: set[str],
    allowed_session_prefix: str | None,
    use_depth_teacher: bool,
    min_depth_teacher_weight: float,
    use_accumulatable_no: bool,
    ghost_weight: float,
    ghost_score_threshold: float,
    p_dir_ghost_threshold: float,
) -> pd.DataFrame:
    out = _ensure_legacy_labels(df)
    if allowed_session_prefix:
        keep_mask = _session_prefix_mask(out, allowed_session_prefix)
        if not keep_mask.any():
            raise ValueError(f"No rows matched allowed session prefix '{allowed_session_prefix}'.")
        out = out.loc[keep_mask].reset_index(drop=True)
    out["bucket_4class"] = out["bucket_3class"].map(THREE_TO_FOUR)
    out["point_label_source"] = np.where(
        out["bucket_4class"].eq("human_candidate"),
        "camera_semantic_human_candidate",
        "camera_semantic",
    )
    out["point_label_rule"] = "legacy_3class_remap"
    out["point_label_weight"] = _base_weight(out)

    if "accumulatable_label" not in out.columns:
        out["accumulatable_label"] = "uncertain"
    else:
        accum = _lower_strings(out["accumulatable_label"])
        out["accumulatable_label"] = np.where(accum.isin({"yes", "no"}), accum, "uncertain")

    if "return_type_label" not in out.columns:
        out["return_type_label"] = "unknown"
    else:
        return_type = _lower_strings(out["return_type_label"])
        out["return_type_label"] = np.where(return_type.eq("nan") | return_type.eq(""), "unknown", return_type)

    ghost_mask = pd.Series(False, index=out.index)
    ghost_source = pd.Series("", index=out.index, dtype=object)
    ghost_rule = pd.Series("", index=out.index, dtype=object)

    no_human = _no_human_mask(out, no_human_prefixes)
    person_like_no_human = no_human & out["bucket_3class"].eq("human")
    ghost_mask |= person_like_no_human
    ghost_source.loc[person_like_no_human] = "controlled_no_human_session"
    ghost_rule.loc[person_like_no_human] = "no_human_session_person_like_label"

    if use_depth_teacher and {"depth_teacher_eligible", "depth_teacher_acc_label"}.issubset(out.columns):
        eligible = _truthy(out["depth_teacher_eligible"])
        acc_label = _lower_strings(out["depth_teacher_acc_label"])
        acc_no = acc_label.eq("no")
        acc_yes = acc_label.eq("yes")
        teacher_weight = _numeric(out, "depth_teacher_weight", 0.0)
        depth_supported = eligible & acc_yes & (teacher_weight >= float(min_depth_teacher_weight))
        out.loc[depth_supported, "point_label_source"] = "depth_teacher_supported"
        out.loc[depth_supported, "point_label_rule"] = "eligible_depth_teacher_yes"
        out.loc[depth_supported, "point_label_weight"] = np.maximum(
            _numeric(out.loc[depth_supported], "point_label_weight", 1.0).to_numpy(dtype=np.float32),
            teacher_weight.loc[depth_supported].to_numpy(dtype=np.float32),
        )
        out.loc[depth_supported, "accumulatable_label"] = "yes"
        out.loc[
            depth_supported & _lower_strings(out["return_type_label"]).isin({"", "unknown", "nan"}),
            "return_type_label",
        ] = "direct"

        depth_ghost = eligible & acc_no & (teacher_weight >= float(min_depth_teacher_weight))
        ghost_mask |= depth_ghost
        ghost_source.loc[depth_ghost & ghost_source.eq("")] = "depth_teacher"
        ghost_rule.loc[depth_ghost & ghost_rule.eq("")] = "eligible_depth_teacher_no"

    if use_accumulatable_no and "accumulatable_label" in out.columns:
        acc_no = _lower_strings(out["accumulatable_label"]).eq("no")
        return_type = _lower_strings(out["return_type_label"])
        review_ghost = acc_no & return_type.isin(GHOST_RETURN_TYPES)
        ghost_mask |= review_ghost
        ghost_source.loc[review_ghost & ghost_source.eq("")] = "accumulatability_label"
        ghost_rule.loc[review_ghost & ghost_rule.eq("")] = "accumulatable_no_ghost_return_type"

    ghost_signal = _numeric(out, "ghost_score_combined", np.nan)
    if ghost_signal.isna().all():
        ghost_signal = _numeric(out, "ghost_score", 0.0)
    else:
        ghost_signal = ghost_signal.fillna(_numeric(out, "ghost_score", 0.0))
    p_dir_signal = _numeric(out, "p_dir_doppler", _numeric(out, "static_confidence", 1.0))
    directness_ghost = (
        ghost_signal >= float(ghost_score_threshold)
    ) & (
        p_dir_signal < float(p_dir_ghost_threshold)
    )
    directness_ghost &= ~ghost_mask
    ghost_mask |= directness_ghost
    ghost_source.loc[directness_ghost & ghost_source.eq("")] = "directness_ghost_rule"
    ghost_rule.loc[directness_ghost & ghost_rule.eq("")] = "high_ghost_score_low_directness"

    out.loc[ghost_mask, "bucket_4class"] = "ghost_return"
    out.loc[ghost_mask, "point_label_source"] = ghost_source.loc[ghost_mask].replace("", "generated_ghost_rule")
    out.loc[ghost_mask, "point_label_rule"] = ghost_rule.loc[ghost_mask].replace("", "generated_ghost_rule")
    depth_ghost_weight = _numeric(out, "depth_teacher_weight", 0.0) if "depth_teacher_weight" in out.columns else pd.Series(0.0, index=out.index)
    out.loc[ghost_mask, "point_label_weight"] = np.where(
        ghost_source.loc[ghost_mask].eq("depth_teacher"),
        depth_ghost_weight.loc[ghost_mask].clip(lower=0.0, upper=1.0),
        float(ghost_weight),
    )
    out.loc[ghost_mask, "accumulatable_label"] = "no"
    out.loc[ghost_mask & _lower_strings(out["return_type_label"]).isin({"", "unknown", "nan"}), "return_type_label"] = (
        "controlled_no_human_person_like"
    )

    if no_human_val_sessions:
        sessions = _clean_strings(out["session"])
        out.loc[sessions.isin(no_human_val_sessions), "split"] = "val"

    out["branch1_label_version"] = "branch1_4class_v1"
    out["branch1_no_human_session"] = no_human.astype(np.int8)
    out["point_label_weight"] = _numeric(out, "point_label_weight", 1.0).clip(0.0, 1.0)
    return out


def build_audit(df: pd.DataFrame) -> dict[str, object]:
    warnings: list[str] = []
    split_counts = df["split"].astype(str).value_counts().to_dict() if "split" in df.columns else {}
    class_counts_by_split = {
        split: {cls: int(count) for cls, count in sub["bucket_4class"].value_counts().to_dict().items()}
        for split, sub in df.groupby("split", sort=True)
    } if "split" in df.columns else {"all": df["bucket_4class"].value_counts().to_dict()}

    for split in ("train", "val"):
        counts = class_counts_by_split.get(split, {})
        for cls in FOUR_CLASS_ORDER:
            n = int(counts.get(cls, 0))
            if n == 0:
                warnings.append(f"{split} split has no {cls} rows.")
            elif cls == "ghost_return" and n < 25:
                warnings.append(f"{split} split has only {n} ghost_return rows.")

    legacy_to_four = pd.crosstab(df["bucket_3class_legacy"], df["bucket_4class"]).to_dict()
    ghost_by_session = (
        df[df["bucket_4class"].eq("ghost_return")]
        .groupby(["split", "session"], dropna=False)
        .size()
        .reset_index(name="ghost_rows")
        .sort_values(["split", "ghost_rows", "session"], ascending=[True, False, True])
    )
    source_counts = df["point_label_source"].astype(str).value_counts().to_dict()
    rule_counts = df["point_label_rule"].astype(str).value_counts().to_dict()

    acc_counts = (
        df["accumulatable_label"].astype(str).str.lower().value_counts().to_dict()
        if "accumulatable_label" in df.columns else {}
    )
    depth_cols_present = [c for c in DEPTH_TEACHER_COLS if c in df.columns]
    depth_summary: dict[str, object] = {
        "depth_columns_present": depth_cols_present,
        "has_depth_teacher": bool(depth_cols_present),
    }
    if depth_cols_present:
        if "branch1_depth_teacher_matched" in df.columns:
            depth_summary["matched_rows"] = int(pd.to_numeric(df["branch1_depth_teacher_matched"], errors="coerce").fillna(0).sum())
        if "depth_teacher_eligible" in df.columns:
            eligible = _truthy(df["depth_teacher_eligible"])
            depth_summary["eligible_rows"] = int(eligible.sum())
            if int(eligible.sum()) == 0:
                warnings.append("Depth teacher columns are present, but no rows are depth-teacher eligible.")
        if "depth_teacher_acc_label" in df.columns:
            depth_summary["depth_teacher_acc_label_counts"] = (
                df["depth_teacher_acc_label"].astype(str).str.lower().value_counts().to_dict()
            )
        if "depth_teacher_reason" in df.columns:
            depth_summary["depth_teacher_reason_counts"] = (
                df["depth_teacher_reason"].astype(str).value_counts().head(20).to_dict()
            )
        if "depth_teacher_weight" in df.columns:
            w = _numeric(df, "depth_teacher_weight", 0.0)
            depth_summary["depth_teacher_weight_nonzero_rows"] = int((w > 0).sum())
            depth_summary["depth_teacher_weight_max"] = float(w.max()) if len(w) else 0.0
    return {
        "n_rows": int(len(df)),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "bucket_order": FOUR_CLASS_ORDER,
        "class_counts_by_split": class_counts_by_split,
        "legacy_3class_to_4class": legacy_to_four,
        "point_label_source_counts": source_counts,
        "point_label_rule_counts": rule_counts,
        "accumulatable_label_counts": acc_counts,
        "ghost_return_rows": int(df["bucket_4class"].eq("ghost_return").sum()),
        "human_candidate_rows": int(df["bucket_4class"].eq("human_candidate").sum()),
        "ghost_return_sessions": int(ghost_by_session["session"].nunique()) if not ghost_by_session.empty else 0,
        "depth_teacher_summary": depth_summary,
        "warnings": warnings,
    }


def write_audit_files(df: pd.DataFrame, out_dir: Path, *, prefix: str = "branch1_fourclass") -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(df)
    (out_dir / f"{prefix}_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    pd.crosstab([df["split"], df["bucket_3class_legacy"]], df["bucket_4class"]).to_csv(
        out_dir / f"{prefix}_legacy_to_4class.csv"
    )
    (
        df.groupby(["split", "session", "bucket_4class"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["split", "session", "bucket_4class"])
        .to_csv(out_dir / f"{prefix}_counts_by_session.csv", index=False)
    )
    (
        df.groupby(["split", "point_label_source", "point_label_rule", "bucket_4class"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["split", "bucket_4class", "rows"], ascending=[True, True, False])
        .to_csv(out_dir / f"{prefix}_counts_by_source.csv", index=False)
    )
    return audit


def mirror_patch_roots(source_root: Path, out_dir: Path, df: pd.DataFrame) -> dict[str, str]:
    if "rd_patch_shard" not in df.columns:
        return {}
    roots = sorted(
        {
            Path(str(rel)).parts[0]
            for rel in df["rd_patch_shard"].dropna().astype(str)
            if rel and rel.lower() != "nan"
        }
    )
    mirrored: dict[str, str] = {}
    for root in roots:
        src = (source_root / root).resolve()
        src_link = source_root / root
        dst = out_dir / root
        if not src.exists():
            if src_link.is_symlink():
                target = os.readlink(src_link)
                mirrored[root] = f"missing_target:{target}"
                continue
            raise FileNotFoundError(f"Patch root referenced by CSV does not exist: {src}")
        if dst.exists() or dst.is_symlink():
            mirrored[root] = os.readlink(dst) if dst.is_symlink() else str(dst)
            continue
        try:
            dst.symlink_to(src, target_is_directory=src.is_dir())
            mirrored[root] = str(src)
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            mirrored[root] = str(dst)
    return mirrored


def _run_label_dataset(args: argparse.Namespace) -> None:
    source_csv = args.source_csv.resolve()
    source_root = args.source_root.resolve() if args.source_root else source_csv.parent.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv, low_memory=False)
    if args.allowed_session_prefix:
        keep_mask = _session_prefix_mask(df, args.allowed_session_prefix)
        if not keep_mask.any():
            raise ValueError(f"No rows matched allowed session prefix '{args.allowed_session_prefix}'.")
        df = df.loc[keep_mask].reset_index(drop=True)
    depth_merge: dict[str, object] = {"source": "none"}
    if args.depth_teacher_csv is not None:
        depth_df = pd.read_csv(args.depth_teacher_csv.resolve(), low_memory=False)
        if args.allowed_session_prefix:
            depth_keep = _session_prefix_mask(depth_df, args.allowed_session_prefix)
            depth_df = depth_df.loc[depth_keep].reset_index(drop=True)
        df, depth_merge = _merge_depth_teacher_columns(
            df,
            depth_df,
            round_decimals=args.depth_merge_round_decimals,
        )
    no_human_prefixes = args.no_human_session_prefix or DEFAULT_NO_HUMAN_PREFIXES
    val_sessions = set(args.no_human_val_session or [])
    out = materialize_fourclass(
        df,
        no_human_prefixes=no_human_prefixes,
        no_human_val_sessions=val_sessions,
        use_depth_teacher=not args.disable_depth_teacher,
        min_depth_teacher_weight=args.min_depth_teacher_weight,
        use_accumulatable_no=args.use_accumulatable_no,
        ghost_weight=args.ghost_weight,
        allowed_session_prefix=args.allowed_session_prefix,
        ghost_score_threshold=args.ghost_score_threshold,
        p_dir_ghost_threshold=args.p_dir_ghost_threshold,
    )

    bad_labels = sorted(set(out["bucket_4class"].dropna()) - set(FOUR_CLASS_ORDER))
    if bad_labels:
        raise ValueError(f"Invalid bucket_4class labels after materialization: {bad_labels}")
    if "split" not in out.columns:
        raise ValueError("Dataset must contain split column before four-class materialization.")

    output_csv = out_dir / "points_with_rd_patch_index.csv"
    out.to_csv(output_csv, index=False)
    mirrored = mirror_patch_roots(source_root, out_dir, out)
    if (source_root / "manifest.json").exists():
        shutil.copy2(source_root / "manifest.json", out_dir / "source_manifest.json")

    audit = write_audit_files(out, out_dir)
    manifest = {
        "kind": "branch1_fourclass_rd_patch_dataset",
        "label_version": "branch1_4class_v1",
        "source_csv": str(source_csv),
        "source_root": str(source_root),
        "output_csv": str(output_csv),
        "bucket_order": FOUR_CLASS_ORDER,
        "label_col": "bucket_4class",
        "sample_weight_col": "point_label_weight",
        "no_human_session_prefixes": no_human_prefixes,
        "no_human_val_sessions": sorted(val_sessions),
        "use_depth_teacher": not args.disable_depth_teacher,
        "depth_teacher_csv": str(args.depth_teacher_csv.resolve()) if args.depth_teacher_csv else None,
        "depth_teacher_merge": depth_merge,
        "use_accumulatable_no": bool(args.use_accumulatable_no),
        "allowed_session_prefix": args.allowed_session_prefix,
        "ghost_score_threshold": float(args.ghost_score_threshold),
        "p_dir_ghost_threshold": float(args.p_dir_ghost_threshold),
        "patch_roots": mirrored,
        "audit": audit,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out_dir / "branch1_fourclass_label_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_csv": str(output_csv),
        "output_dir": str(out_dir),
        "split_counts": audit["split_counts"],
        "class_counts_by_split": audit["class_counts_by_split"],
        "ghost_return_rows": audit["ghost_return_rows"],
        "depth_teacher_merge": depth_merge,
        "depth_teacher_summary": audit.get("depth_teacher_summary", {}),
        "warnings": audit["warnings"],
    }, indent=2, default=str))


def _run_audit_dataset(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.dataset, low_memory=False)
    if "bucket_4class" not in df.columns:
        raise ValueError("Audit dataset requires bucket_4class.")
    audit = write_audit_files(df, args.output_dir.resolve())
    print(json.dumps(audit, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    label = sub.add_parser("label-dataset", help="Create a Branch 1 four-class RD-patch dataset.")
    label.add_argument("--source-csv", required=True, type=Path)
    label.add_argument("--source-root", type=Path, default=None)
    label.add_argument("--output-dir", required=True, type=Path)
    label.add_argument("--allowed-session-prefix", default=None)
    label.add_argument("--depth-teacher-csv", type=Path, default=None)
    label.add_argument("--depth-merge-round-decimals", type=int, default=4)
    label.add_argument("--no-human-session-prefix", action="append", default=None)
    label.add_argument("--no-human-val-session", action="append", default=None)
    label.add_argument("--disable-depth-teacher", action="store_true")
    label.add_argument("--min-depth-teacher-weight", type=float, default=0.5)
    label.add_argument("--ghost-score-threshold", type=float, default=DEFAULT_GHOST_SCORE_THRESHOLD)
    label.add_argument("--p-dir-ghost-threshold", type=float, default=DEFAULT_P_DIR_GHOST_THRESHOLD)
    label.add_argument(
        "--use-accumulatable-no",
        action="store_true",
        help="Also map high-confidence accumulatable_label=no ghost return types to ghost_return.",
    )
    label.add_argument("--ghost-weight", type=float, default=1.0)
    label.set_defaults(func=_run_label_dataset)

    audit = sub.add_parser("audit-dataset", help="Audit an existing Branch 1 four-class dataset.")
    audit.add_argument("--dataset", required=True, type=Path)
    audit.add_argument("--output-dir", required=True, type=Path)
    audit.set_defaults(func=_run_audit_dataset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
