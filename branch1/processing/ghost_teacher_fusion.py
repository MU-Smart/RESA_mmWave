#!/usr/bin/env python3
"""Fuse depth-correspondence and directness evidence into ghost teacher labels.

This tool is intentionally conservative.  Depth evidence can only create hard
`accumulatable_label=no` labels when the session passed depth QA and depth
contradictions are cluster-verified and supported by radar-only ghost evidence.
Manual review labels, when present, take precedence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEPTH_JOIN_COLS = [
    "p_depth_match",
    "depth_corr_available",
    "depth_corr_in_fov",
    "depth_corr_patch_px",
    "depth_corr_valid_px",
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
]

OUTPUT_TEACHER_COLS = [
    "depth_teacher_eligible",
    "depth_teacher_acc_label",
    "depth_teacher_return_type",
    "depth_teacher_weight",
    "depth_teacher_reason",
    "depth_teacher_source",
    "ghost_score_combined",
]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _round_join_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "radar_frame_num" not in out.columns and "frame_num" in out.columns:
        out["radar_frame_num"] = out["frame_num"]
    for col in ("radar_frame_num", "range_bin", "doppler_bin"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1).astype(np.int64)
    for col in ("x", "y", "z"):
        if col in out.columns:
            out[f"_{col}j"] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def _merge_depth(labels: pd.DataFrame, depth: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels_j = _round_join_cols(labels)
    depth_j = _round_join_cols(depth)
    keys = ["radar_frame_num", "range_bin", "doppler_bin", "_xj", "_yj", "_zj"]
    missing = [c for c in keys if c not in labels_j.columns or c not in depth_j.columns]
    if missing:
        return labels.copy(), {"depth_rows_matched": 0, "depth_match_fraction": 0.0, "missing_join_cols": missing}

    keep_cols = keys + [c for c in DEPTH_JOIN_COLS if c in depth_j.columns]
    depth_keep = depth_j[keep_cols].drop_duplicates(keys, keep="first")
    merged = labels_j.merge(depth_keep, on=keys, how="left", suffixes=("", "_depth"))
    matched = int(merged["p_depth_match"].notna().sum()) if "p_depth_match" in merged.columns else 0
    merged = merged.drop(columns=["_xj", "_yj", "_zj"], errors="ignore")
    return merged, {
        "depth_rows_matched": matched,
        "depth_match_fraction": float(matched / max(len(labels), 1)),
        "missing_join_cols": [],
    }


def _manual_acc(row: pd.Series) -> str:
    value = str(row.get("accumulatable_label", "") or "").strip().lower()
    return value if value in {"yes", "no", "uncertain"} else ""


def _return_type_for_no(row: pd.Series) -> str:
    if int(float(row.get("depth_foreground_occluded", 0) or 0)):
        return "depth_occluded"
    sem = str(row.get("cam_semantic_bucket", "") or "").strip().lower()
    bucket = str(row.get("bucket_3class", row.get("bucket", "")) or "").strip().lower()
    if sem == "floor" or bucket == "floor":
        return "floor_bounce"
    if sem == "structure" or bucket == "structure":
        return "wall_multipath"
    return "unknown"


def _compute_teacher_row(row: pd.Series, *, eligible: bool, args: argparse.Namespace) -> dict[str, Any]:
    manual = _manual_acc(row)
    p_depth = float(row.get("p_depth_match", args.depth_unknown_floor) or args.depth_unknown_floor)
    p_dir = float(row.get("p_dir_doppler", row.get("static_confidence", 1.0)) or 1.0)
    static_conf = float(row.get("static_confidence", 0.5) or 0.5)
    ghost_score = float(row.get("ghost_score", 0.5) or 0.5)
    missing_reason = str(row.get("depth_missing_reason", "unknown") or "unknown")
    cluster_size = int(float(row.get("depth_candidate_cluster_size", 0) or 0))
    cluster_radius = float(row.get("depth_candidate_cluster_radius_m", 999.0) or 999.0)
    cluster_min_depth = float(row.get("depth_candidate_cluster_min_depth_match", 1.0) or 1.0)
    foreground = bool(int(float(row.get("depth_foreground_occluded", 0) or 0)))

    p_depth_for_score = p_depth if eligible and missing_reason == "ok" else float(args.depth_unknown_floor)
    p_dir_clip = float(np.clip(p_dir, 0.0, 1.0))
    static_clip = float(np.clip(static_conf, 0.0, 1.0))
    p_depth_clip = float(np.clip(p_depth_for_score, 0.0, 1.0))
    ghost_score_combined = 1.0 - (
        max(p_dir_clip, 1e-4) ** float(args.w_doppler)
        * max(p_depth_clip, 1e-4) ** float(args.w_depth)
        * max(static_clip, 1e-4) ** float(args.w_static)
    )
    ghost_score_combined = float(np.clip(max(ghost_score_combined, ghost_score), 0.0, 1.0))

    if manual:
        return_type = str(row.get("return_type_label", "") or "").strip().lower()
        return {
            "depth_teacher_eligible": bool(eligible),
            "depth_teacher_acc_label": manual,
            "depth_teacher_return_type": return_type or ("direct" if manual == "yes" else "unknown"),
            "depth_teacher_weight": 1.0 if manual in {"yes", "no"} else 0.0,
            "depth_teacher_reason": "manual_review",
            "depth_teacher_source": "manual_review",
            "ghost_score_combined": round(ghost_score_combined, 5),
            "accumulatable_label": manual,
            "return_type_label": return_type or ("direct" if manual == "yes" else "unknown"),
        }

    depth_cluster_verified = (
        cluster_size >= int(args.min_cluster_size)
        and cluster_radius <= float(args.max_cluster_radius_m)
        and cluster_min_depth < float(args.depth_no_threshold)
    )
    radar_suspicious = p_dir < float(args.p_dir_no_threshold) or ghost_score >= float(args.ghost_score_no_threshold)

    label = "uncertain"
    return_type = "unknown"
    weight = 0.0
    reason = "uncertain"
    if (
        eligible
        and missing_reason == "ok"
        and p_depth < float(args.depth_no_threshold)
        and depth_cluster_verified
        and radar_suspicious
    ):
        label = "no"
        return_type = _return_type_for_no(row)
        weight = float(args.hard_no_weight)
        reason = "depth_cluster_and_radar_suspicious"
    elif (
        eligible
        and missing_reason == "ok"
        and p_depth >= float(args.depth_yes_threshold)
        and p_dir >= float(args.p_dir_yes_threshold)
        and not foreground
    ):
        label = "yes"
        return_type = "direct"
        weight = float(args.hard_yes_weight)
        reason = "depth_and_directness_agree"
    elif not eligible:
        reason = "session_depth_qa_failed"
    elif missing_reason != "ok":
        reason = f"depth_missing:{missing_reason}"
    elif p_depth < float(args.depth_no_threshold) and not depth_cluster_verified:
        reason = "depth_low_cluster_unverified"
    elif p_depth < float(args.depth_no_threshold) and not radar_suspicious:
        reason = "depth_low_radar_not_suspicious"

    return {
        "depth_teacher_eligible": bool(eligible),
        "depth_teacher_acc_label": label,
        "depth_teacher_return_type": return_type,
        "depth_teacher_weight": round(float(weight), 5),
        "depth_teacher_reason": reason,
        "depth_teacher_source": "depth_directness_rule",
        "ghost_score_combined": round(ghost_score_combined, 5),
        "accumulatable_label": label,
        "return_type_label": return_type,
    }


def _ensure_processing_assets(src_session: Path, dst_session: Path, *, copy_assets: bool) -> None:
    dst_session.mkdir(parents=True, exist_ok=True)
    asset_names = [
        f"{src_session.name}.csv",
        f"{src_session.name}_radar_tensors.npz",
        "hybrid_rd",
        "seg",
        "seg_meta.json",
        "meta_data.json",
        f"synchronized_{src_session.name}.csv",
        f"{src_session.name}_radar_timestamps.csv",
        f"{src_session.name}_depth_timestamps.csv",
        f"{src_session.name}_color_timestamps.csv",
    ]
    for name in asset_names:
        src = src_session / name
        if not src.exists():
            continue
        dst = dst_session / name
        if dst.exists() or dst.is_symlink():
            continue
        if copy_assets:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        else:
            try:
                os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
            except OSError:
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)


def fuse_session(
    *,
    processing_session: Path,
    depth_session: Path,
    out_session: Path,
    label_csv_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    src_label = processing_session / label_csv_name
    if not src_label.exists():
        raise FileNotFoundError(f"Missing label CSV: {src_label}")
    labels = pd.read_csv(src_label, low_memory=False)

    depth_csv = depth_session / "points_with_depth_correspondence.csv"
    depth_report = depth_session / "depth_correspondence_report.json"
    if depth_csv.exists():
        depth = pd.read_csv(depth_csv, low_memory=False)
        merged, merge_stats = _merge_depth(labels, depth)
    else:
        merged = labels.copy()
        merge_stats = {"depth_rows_matched": 0, "depth_match_fraction": 0.0, "missing_join_cols": []}
    report = _load_json(depth_report)
    eligible = bool(report.get("depth_teacher_eligible", False))

    teacher_rows = [_compute_teacher_row(row, eligible=eligible, args=args) for _, row in merged.iterrows()]
    teacher_df = pd.DataFrame(teacher_rows)
    for col in OUTPUT_TEACHER_COLS + ["accumulatable_label", "return_type_label"]:
        merged[col] = teacher_df[col].values if col in teacher_df.columns else ""

    # Preserve existing final_train_weight if present; otherwise create a conservative default.
    if "final_train_weight" not in merged.columns:
        maj = _numeric(merged.get("maj_frac", pd.Series(1.0, index=merged.index)), 1.0)
        merged["final_train_weight"] = maj.clip(0.0, 1.0)

    out_session.mkdir(parents=True, exist_ok=True)
    _ensure_processing_assets(processing_session, out_session, copy_assets=bool(args.copy_assets))
    out_label = out_session / label_csv_name
    merged.to_csv(out_label, index=False)

    counts = merged["accumulatable_label"].fillna("").astype(str).str.lower().value_counts().to_dict()
    reasons = merged["depth_teacher_reason"].fillna("").astype(str).value_counts().head(20).to_dict()
    summary = {
        "session": processing_session.name,
        "source_label_csv": str(src_label),
        "out_label_csv": str(out_label),
        "depth_csv": str(depth_csv),
        "depth_report": str(depth_report),
        "depth_teacher_eligible": eligible,
        "rows": int(len(merged)),
        **merge_stats,
        "accumulatable_counts": counts,
        "teacher_reason_counts": reasons,
        "ghost_score_combined_mean": float(_numeric(merged["ghost_score_combined"], 0.5).mean()) if len(merged) else 0.0,
    }
    (out_session / "ghost_teacher_fusion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _session_dirs(root: Path, session_glob: str, max_sessions: int | None) -> list[Path]:
    sessions = sorted(p for p in root.glob(session_glob) if p.is_dir())
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    return sessions


def cmd_fuse_root(args: argparse.Namespace) -> None:
    processing_root = args.processing_root.resolve()
    depth_root = args.depth_root.resolve()
    out_root = args.out_root.resolve()
    summaries: list[dict[str, Any]] = []
    for src_session in _session_dirs(processing_root, args.session_glob, args.max_sessions):
        try:
            summary = fuse_session(
                processing_session=src_session,
                depth_session=depth_root / src_session.name,
                out_session=out_root / src_session.name,
                label_csv_name=args.label_csv_name,
                args=args,
            )
            summary["status"] = "ok"
            summary["error"] = ""
        except Exception as exc:
            summary = {
                "session": src_session.name,
                "status": "error",
                "error": str(exc),
                "rows": 0,
                "depth_teacher_eligible": False,
            }
        summaries.append(summary)
        print(
            f"[{summary['status']}] {src_session.name}: rows={summary.get('rows', 0)} "
            f"eligible={summary.get('depth_teacher_eligible', False)} "
            f"counts={summary.get('accumulatable_counts', {})} {summary.get('error', '')}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "ghost_teacher_fusion_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    (out_root / "ghost_teacher_fusion_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary: {summary_csv}")


def cmd_fuse_session(args: argparse.Namespace) -> None:
    summary = fuse_session(
        processing_session=args.processing_session.resolve(),
        depth_session=args.depth_session.resolve(),
        out_session=args.out_session.resolve(),
        label_csv_name=args.label_csv_name,
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_report(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(args.root.resolve().glob("*/ghost_teacher_fusion_summary.json")):
        row = _load_json(path)
        row["report_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No ghost_teacher_fusion_summary.json files under {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


def add_fusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--depth-no-threshold", type=float, default=0.20)
    parser.add_argument("--depth-yes-threshold", type=float, default=0.70)
    parser.add_argument("--p-dir-no-threshold", type=float, default=0.20)
    parser.add_argument("--p-dir-yes-threshold", type=float, default=0.30)
    parser.add_argument("--ghost-score-no-threshold", type=float, default=0.70)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-cluster-radius-m", type=float, default=0.54)
    parser.add_argument("--depth-unknown-floor", type=float, default=1.0)
    parser.add_argument("--w-doppler", type=float, default=1.0)
    parser.add_argument("--w-depth", type=float, default=1.0)
    parser.add_argument("--w-static", type=float, default=1.0)
    parser.add_argument("--hard-no-weight", type=float, default=1.0)
    parser.add_argument("--hard-yes-weight", type=float, default=1.0)
    parser.add_argument("--copy-assets", action="store_true", help="Copy required session assets instead of symlinking them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_root = sub.add_parser("fuse-root", help="Fuse all matching sessions under a processing root.")
    p_root.add_argument("--processing-root", type=Path, required=True)
    p_root.add_argument("--depth-root", type=Path, required=True)
    p_root.add_argument("--out-root", type=Path, required=True)
    p_root.add_argument("--session-glob", default="session_*")
    p_root.add_argument("--max-sessions", type=int, default=None)
    p_root.add_argument("--label-csv-name", default="labeled_radar_points_v4.csv")
    add_fusion_args(p_root)
    p_root.set_defaults(func=cmd_fuse_root)

    p_session = sub.add_parser("fuse-session", help="Fuse one processing session with one depth-correspondence session.")
    p_session.add_argument("--processing-session", type=Path, required=True)
    p_session.add_argument("--depth-session", type=Path, required=True)
    p_session.add_argument("--out-session", type=Path, required=True)
    p_session.add_argument("--label-csv-name", default="labeled_radar_points_v4.csv")
    add_fusion_args(p_session)
    p_session.set_defaults(func=cmd_fuse_session)

    p_report = sub.add_parser("report", help="Collect fusion summaries into a CSV.")
    p_report.add_argument("--root", type=Path, required=True)
    p_report.add_argument("--out", type=Path, required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
