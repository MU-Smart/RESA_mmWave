#!/usr/bin/env python3
"""Offline depth-correspondence tools for radar point datasets.

Subcommands:

    annotate-session   Annotate one processing session with depth evidence.
    annotate-root      Annotate multiple sessions under a processing root.
    report             Collect per-session depth correspondence reports.
    overlay            Render QA overlays from an annotated correspondence CSV.

The tool is conservative: missing sync/depth/calibration evidence becomes an
explicit `depth_missing_reason`, not a ghost label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
CANON_DIR = THIS_FILE.parents[1]
if str(CANON_DIR) not in sys.path:
    sys.path.insert(0, str(CANON_DIR))

from perception.depth_correspondence import (  # noqa: E402
    DEPTH_NUMERIC_COLS,
    DepthCorrespondenceConfig,
    DepthFrameCache,
    annotate_semantic_from_seg,
    build_best_video_for_radar_frame,
    build_depth_correspondence_evidence,
    cluster_depth_contradictions,
    compute_depth_correspondence_for_points,
    load_depth_scale,
    make_default_depth_record,
)
from perception.calibration import load_calibration  # noqa: E402


def _find_first(directory: Path, *patterns: str) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _pick_radar_csv(session_dir: Path) -> Path:
    preferred = session_dir / f"{session_dir.name}.csv"
    if preferred.exists():
        return preferred
    for path in sorted(session_dir.glob("session_*.csv")):
        if "synchronized" not in path.name and "labeled" not in path.name:
            return path
    raise FileNotFoundError(f"No raw radar CSV found in {session_dir}")


def _pick_sync_csv(session_dir: Path) -> Path:
    preferred = session_dir / f"synchronized_{session_dir.name}.csv"
    if preferred.exists():
        return preferred
    path = _find_first(session_dir, "synchronized_*.csv")
    if path is None:
        raise FileNotFoundError(f"No synchronized_*.csv found in {session_dir}")
    return path


def _pick_frame_col(df: pd.DataFrame) -> str:
    for col in ("radar_frame_num", "frame_num", "frame_index"):
        if col in df.columns:
            return col
    raise ValueError(f"Radar CSV has no frame column. Columns: {list(df.columns)}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_seg_id_to_label(session_dir: Path) -> dict[str, str]:
    meta = _load_json(session_dir / "seg_meta.json")
    id2label = meta.get("id2label", meta)
    return {str(k): str(v) for k, v in id2label.items()} if isinstance(id2label, dict) else {}


def _pick_color_video(session_dir: Path) -> Path | None:
    preferred = session_dir / f"{session_dir.name}_color.mp4"
    if preferred.exists():
        return preferred
    return _find_first(session_dir, "*_color.mp4")


def _read_timestamps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame_index" not in df.columns:
        df["frame_index"] = np.arange(len(df), dtype=int)
    for col in ("timestamp_ms", "timestamp", "timestamp_us", "timestamp_ns"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").astype(float)
            if col == "timestamp_us":
                vals = vals * 1e-3
            elif col == "timestamp_ns":
                vals = vals * 1e-6
            df["timestamp_ms"] = vals
            return df[["frame_index", "timestamp_ms"]].dropna()
    raise ValueError(f"{path} has no timestamp column")


def _build_color_to_depth_map(session_dir: Path, max_dt_ms: float = 250.0) -> dict[int, int | None]:
    color_ts_path = _find_first(session_dir, "*_color_timestamps.csv", "seg_timestamps.csv")
    depth_ts_path = _find_first(session_dir, "*_depth_timestamps.csv")
    if color_ts_path is None or depth_ts_path is None:
        return {}
    color_ts = _read_timestamps(color_ts_path)
    depth_ts = _read_timestamps(depth_ts_path)
    d_times = depth_ts["timestamp_ms"].to_numpy(dtype=float)
    d_frames = depth_ts["frame_index"].to_numpy(dtype=int)
    order = np.argsort(d_times)
    d_times = d_times[order]
    d_frames = d_frames[order]
    out: dict[int, int | None] = {}
    for _, row in color_ts.iterrows():
        cf = int(row["frame_index"])
        ct = float(row["timestamp_ms"])
        j = int(np.searchsorted(d_times, ct))
        candidates = [k for k in (j - 1, j, j + 1) if 0 <= k < len(d_times)]
        if not candidates:
            out[cf] = None
            continue
        best = min(candidates, key=lambda k: abs(float(d_times[k]) - ct))
        out[cf] = int(d_frames[best]) if abs(float(d_times[best]) - ct) <= max_dt_ms else None
    return out


def _resolve_depth_frame_index(
    session_dir: Path,
    video_frame_index: int,
    color_to_depth: dict[int, int | None],
) -> int | None:
    direct = session_dir / "depth" / f"{int(video_frame_index):06d}.npy"
    if direct.exists():
        return int(video_frame_index)
    mapped = color_to_depth.get(int(video_frame_index))
    return int(mapped) if mapped is not None else None


def _config_from_args(args: argparse.Namespace) -> DepthCorrespondenceConfig:
    return DepthCorrespondenceConfig(
        depth_patch_r=int(args.depth_patch_r),
        semantic_patch_r=int(args.semantic_patch_r),
        use_distortion=bool(args.use_distortion),
        max_time_diff_ms=args.max_time_diff_ms,
        foreground_occlusion_margin_m=float(args.foreground_occlusion_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_frac=float(args.sigma_frac),
        candidate_residual_m=float(args.candidate_residual_m),
        candidate_p_depth_match_max=float(args.candidate_p_depth_match_max),
        candidate_dbscan_eps_m=float(args.candidate_dbscan_eps_m),
        candidate_min_samples=int(args.candidate_min_samples),
    )


def annotate_session(
    session_dir: Path,
    *,
    out_csv: Path,
    report_json: Path,
    radar_csv: Path | None,
    sync_csv: Path | None,
    extrinsics_json: Path | None,
    config: DepthCorrespondenceConfig,
    include_semantics: bool = True,
) -> dict[str, Any]:
    session_dir = session_dir.resolve()
    radar_csv = radar_csv.resolve() if radar_csv is not None else _pick_radar_csv(session_dir)
    sync_csv = sync_csv.resolve() if sync_csv is not None else _pick_sync_csv(session_dir)

    df = pd.read_csv(radar_csv, low_memory=False)
    frame_col = _pick_frame_col(df)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise ValueError(f"{radar_csv} missing required column {col!r}")
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce").fillna(-1).astype(int)
    if len(df) == 0:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        report_json.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        evidence = {
            "session": session_dir.name,
            "session_dir": str(session_dir),
            "radar_csv": str(radar_csv),
            "sync_csv": str(sync_csv) if sync_csv is not None else "",
            "out_csv": str(out_csv),
            "points": 0,
            "ok_points": 0,
            "depth_teacher_eligible": False,
            "failure_reason": "no_points",
        }
        report_json.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
        return evidence

    sync = pd.read_csv(sync_csv, low_memory=False)
    best_map, sync_stats = build_best_video_for_radar_frame(
        sync.to_dict("records"),
        max_time_diff_ms=config.max_time_diff_ms,
    )

    try:
        calibration = load_calibration(session_dir, extrinsics_json=extrinsics_json)
        calibration_error = ""
    except Exception as exc:
        calibration = None
        calibration_error = str(exc)

    depth_dir = session_dir / "depth"
    depth_scale = load_depth_scale(session_dir / "meta_data.json")
    depth_cache = DepthFrameCache(depth_dir, depth_scale) if depth_dir.is_dir() else None
    color_to_depth = _build_color_to_depth_map(session_dir)

    records: list[dict[str, Any]] = [make_default_depth_record(config, "unprocessed") for _ in range(len(df))]
    xyz_all = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    for frame_num, grp in df.groupby(frame_col, sort=False):
        idx = grp.index.to_numpy()
        match = best_map.get(int(frame_num))
        if match is None:
            for row_idx in idx:
                records[int(row_idx)] = make_default_depth_record(config, "no_sync")
            continue
        video_frame_index = int(match["video_frame_index"])
        depth_frame_index = _resolve_depth_frame_index(session_dir, video_frame_index, color_to_depth)
        if depth_frame_index is None or depth_cache is None:
            depth_m = None
            depth_frame_index = -1
        else:
            depth_m = depth_cache.get(depth_frame_index)

        frame_records = compute_depth_correspondence_for_points(
            grp[["x", "y", "z"]].to_numpy(dtype=np.float32),
            depth_m=depth_m,
            video_frame_index=video_frame_index,
            depth_frame_index=int(depth_frame_index),
            match_time_diff_us=match.get("time_diff_us"),
            calibration=calibration,
            config=config,
        )
        for row_idx, rec in zip(idx, frame_records):
            records[int(row_idx)] = rec

    p_dir = None
    if "p_dir_doppler" in df.columns:
        p_dir = pd.to_numeric(df["p_dir_doppler"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    cluster_depth_contradictions(records, xyz_all, config=config, p_dir_doppler=p_dir)

    if include_semantics:
        annotate_semantic_from_seg(
            records,
            seg_dir=session_dir / "seg",
            seg_id_to_label=_load_seg_id_to_label(session_dir),
            config=config,
        )

    for key in sorted(records[0].keys()):
        df[key] = [rec.get(key) for rec in records]
    for col in DEPTH_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    evidence = build_depth_correspondence_evidence(records, sync_stats=sync_stats, config=config)
    evidence.update(
        {
            "session": session_dir.name,
            "session_dir": str(session_dir),
            "radar_csv": str(radar_csv),
            "sync_csv": str(sync_csv),
            "out_csv": str(out_csv),
            "depth_dir": str(depth_dir),
            "depth_scale_m_per_unit": depth_scale,
            "calibration_error": calibration_error,
            "extrinsics_source": calibration.get("extrinsics_source", "") if calibration else "",
            "intrinsics_source": calibration.get("intrinsics_source", "") if calibration else "",
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    report_json.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def _session_dirs(root: Path, session_glob: str, max_sessions: int | None) -> list[Path]:
    sessions = sorted(p for p in root.glob(session_glob) if p.is_dir())
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    return sessions


def cmd_annotate_session(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    session_dir = args.session_dir.resolve()
    out_csv = args.out_csv or (session_dir / "points_with_depth_correspondence.csv")
    report_json = args.report_json or (session_dir / "depth_correspondence_report.json")
    evidence = annotate_session(
        session_dir,
        out_csv=out_csv.resolve(),
        report_json=report_json.resolve(),
        radar_csv=args.radar_csv,
        sync_csv=args.sync_csv,
        extrinsics_json=args.extrinsics_json,
        config=cfg,
        include_semantics=not args.no_semantics,
    )
    print(
        f"{session_dir.name}: points={evidence['points']} ok={evidence['ok_points']} "
        f"eligible={evidence['depth_teacher_eligible']} match_rate={evidence['depth_match_rate']:.3f} "
        f"out={out_csv}"
    )


def cmd_annotate_root(args: argparse.Namespace) -> None:
    cfg = _config_from_args(args)
    root = args.processing_root.resolve()
    out_root = args.out_root.resolve()
    summaries: list[dict[str, Any]] = []
    for session_dir in _session_dirs(root, args.session_glob, args.max_sessions):
        out_dir = out_root / session_dir.name
        try:
            evidence = annotate_session(
                session_dir,
                out_csv=out_dir / "points_with_depth_correspondence.csv",
                report_json=out_dir / "depth_correspondence_report.json",
                radar_csv=None,
                sync_csv=None,
                extrinsics_json=args.extrinsics_json,
                config=cfg,
                include_semantics=not args.no_semantics,
            )
            status = "ok"
            error = ""
        except Exception as exc:
            evidence = {"session": session_dir.name, "points": 0, "depth_teacher_eligible": False}
            status = "error"
            error = str(exc)
        evidence = dict(evidence)
        evidence["status"] = status
        evidence["error"] = error
        summaries.append(evidence)
        print(
            f"[{status}] {session_dir.name}: points={evidence.get('points', 0)} "
            f"eligible={evidence.get('depth_teacher_eligible', False)} {error}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "depth_correspondence_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    (out_root / "depth_correspondence_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"summary: {summary_csv}")


def cmd_report(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(args.root.resolve().glob("*/depth_correspondence_report.json")):
        row = _load_json(path)
        row["report_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No depth_correspondence_report.json files under {args.root}")
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows to {out}")


def _metric_color(row: pd.Series, metric: str) -> tuple[int, int, int]:
    """Return BGR point color for one overlay row."""
    if metric == "residual":
        residual = float(row.get("depth_corr_residual_m", 0.0) or 0.0)
        mag = min(abs(residual) / 1.0, 1.0)
        if residual >= 0.0:
            return (0, int(200 * (1.0 - mag)), int(60 + 195 * mag))
        return (int(60 + 195 * mag), int(200 * (1.0 - mag)), 0)
    if metric == "p_dir_doppler":
        value = float(row.get("p_dir_doppler", 1.0) or 0.0)
        value = max(0.0, min(1.0, value))
        return (0, int(60 + 180 * value), int(240 * (1.0 - value)))
    if metric == "semantic":
        bucket = str(row.get("cam_semantic_bucket", "unknown")).lower()
        if bucket == "human":
            return (255, 0, 255)
        if bucket == "floor":
            return (255, 160, 0)
        if bucket == "structure":
            return (0, 180, 255)
        return (180, 180, 180)
    if metric == "teacher":
        label = str(row.get("depth_teacher_acc_label", row.get("accumulatable_label", "unknown"))).lower()
        if label == "yes":
            return (0, 220, 0)
        if label == "no":
            return (0, 0, 255)
        return (180, 180, 180)

    value = float(row.get("p_depth_match", 1.0) or 0.0)
    value = max(0.0, min(1.0, value))
    if value >= 0.5:
        t = (value - 0.5) / 0.5
        return (0, int(180 + 60 * t), int(220 * (1.0 - t)))
    t = value / 0.5
    return (0, int(220 * t), 255)


def _draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55) -> None:
    import cv2

    x, y = xy
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _load_color_frame(cap: Any, frame_index: int, fallback_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    if cap is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    h, w = fallback_shape
    return np.full((h, w, 3), 35, dtype=np.uint8)


def _depth_vis(session_dir: Path, depth_frame_index: int, fallback_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    h, w = fallback_shape
    path = session_dir / "depth" / f"{int(depth_frame_index):06d}.npy"
    if not path.exists():
        return np.full((h, w, 3), 35, dtype=np.uint8)
    raw = np.load(path).astype(np.float32)
    depth_scale = load_depth_scale(session_dir / "meta_data.json")
    depth_m = raw * float(depth_scale)
    vis = np.clip(depth_m / 5.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def _select_overlay_frames(df: pd.DataFrame, max_frames: int, mode: str) -> list[int]:
    valid = df[pd.to_numeric(df.get("matched_video_frame_index", -1), errors="coerce").fillna(-1).astype(int) >= 0].copy()
    if valid.empty:
        return []
    valid["matched_video_frame_index"] = pd.to_numeric(valid["matched_video_frame_index"], errors="coerce").fillna(-1).astype(int)
    grouped = valid.groupby("matched_video_frame_index", sort=True)
    frames = list(grouped.groups.keys())
    if mode == "first":
        return [int(x) for x in frames[:max_frames]]
    if mode == "even":
        if len(frames) <= max_frames:
            return [int(x) for x in frames]
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        return [int(frames[i]) for i in idx]

    scored: list[tuple[float, int]] = []
    for frame_idx, grp in grouped:
        p_depth = pd.to_numeric(grp.get("p_depth_match", 1.0), errors="coerce").fillna(1.0)
        residual = pd.to_numeric(grp.get("depth_corr_abs_residual_m", 0.0), errors="coerce").fillna(0.0)
        cluster = pd.to_numeric(grp.get("depth_candidate_cluster_id", -1), errors="coerce").fillna(-1)
        score = float((1.0 - p_depth.clip(0, 1)).mean() + min(residual.median(), 3.0) / 3.0)
        score += float((cluster >= 0).mean())
        scored.append((score, int(frame_idx)))
    scored.sort(reverse=True)
    return [frame for _, frame in scored[:max_frames]]


def _render_overlay_frame(
    *,
    color_frame: np.ndarray,
    depth_frame: np.ndarray,
    points: pd.DataFrame,
    metric: str,
    frame_index: int,
    panel_width: int,
    point_radius: int,
) -> np.ndarray:
    import cv2

    h0, w0 = color_frame.shape[:2]
    scale = float(panel_width) / max(float(w0), 1.0)
    panel_height = max(1, int(round(h0 * scale)))
    color = cv2.resize(color_frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(depth_frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

    for _, row in points.iterrows():
        try:
            u = float(row.get("cam_u_depth", np.nan)) * scale
            v = float(row.get("cam_v_depth", np.nan)) * scale
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < panel_width and 0 <= vi < panel_height):
            continue
        color_bgr = _metric_color(row, metric)
        ring = (255, 255, 255) if str(row.get("depth_missing_reason", "")) == "ok" else (40, 40, 40)
        radius = max(2, int(point_radius))
        cv2.circle(color, (ui, vi), radius + 1, ring, 1, cv2.LINE_AA)
        cv2.circle(color, (ui, vi), radius, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(depth, (ui, vi), radius + 1, ring, 1, cv2.LINE_AA)
        cv2.circle(depth, (ui, vi), radius, color_bgr, -1, cv2.LINE_AA)

    p_depth = pd.to_numeric(points.get("p_depth_match", 1.0), errors="coerce").fillna(1.0)
    residual = pd.to_numeric(points.get("depth_corr_abs_residual_m", 0.0), errors="coerce").fillna(0.0)
    ok_rate = float((points.get("depth_missing_reason", "") == "ok").mean()) if len(points) else 0.0
    low_rate = float((p_depth < 0.2).mean()) if len(points) else 0.0
    median_residual = float(residual.median()) if len(points) else 0.0
    cluster_count = int((pd.to_numeric(points.get("depth_candidate_cluster_id", -1), errors="coerce").fillna(-1) >= 0).sum())

    _draw_text(color, f"Color frame={frame_index} metric={metric} points={len(points)}", (10, 24))
    _draw_text(color, f"ok={ok_rate:.2f} low_depth={low_rate:.2f} med_abs_res={median_residual:.2f}m clusters={cluster_count}", (10, 48), 0.45)
    _draw_text(depth, f"Depth frame={frame_index} red=bad green=match", (10, 24))
    _draw_text(depth, "black ring=missing/off-image, white ring=depth ok", (10, 48), 0.45)
    return np.concatenate([color, depth], axis=1)


def cmd_overlay(args: argparse.Namespace) -> None:
    import cv2

    session_dir = args.session_dir.resolve()
    csv_path = args.input_csv.resolve()
    out_dir = args.out_dir.resolve()
    df = pd.read_csv(csv_path, low_memory=False)
    frames = _select_overlay_frames(df, int(args.max_frames), args.frame_selection)
    if not frames:
        raise ValueError(f"No matched frames found in {csv_path}")

    color_video = _pick_color_video(session_dir)
    cap = cv2.VideoCapture(str(color_video)) if color_video is not None else None
    if cap is not None and not cap.isOpened():
        cap.release()
        cap = None

    meta = _load_json(session_dir / "meta_data.json")
    fallback_shape = (int(meta.get("height", 720)), int(meta.get("width", 1280)))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    frame_col = pd.to_numeric(df.get("matched_video_frame_index", -1), errors="coerce").fillna(-1).astype(int)
    for frame_idx in frames:
        points = df[frame_col == int(frame_idx)].copy()
        if len(points) > int(args.max_points_per_frame):
            score = (
                (1.0 - pd.to_numeric(points.get("p_depth_match", 1.0), errors="coerce").fillna(1.0).clip(0, 1))
                + pd.to_numeric(points.get("depth_corr_abs_residual_m", 0.0), errors="coerce").fillna(0.0).clip(0, 3) / 3.0
            )
            points = points.assign(_overlay_score=score).sort_values("_overlay_score", ascending=False).head(int(args.max_points_per_frame))
        color_frame = _load_color_frame(cap, int(frame_idx), fallback_shape)
        depth_idx_series = pd.to_numeric(points.get("matched_depth_frame_index", -1), errors="coerce").fillna(-1).astype(int)
        depth_frame_idx = int(depth_idx_series[depth_idx_series >= 0].mode().iloc[0]) if (depth_idx_series >= 0).any() else int(frame_idx)
        depth_frame = _depth_vis(session_dir, depth_frame_idx, color_frame.shape[:2])
        composed = _render_overlay_frame(
            color_frame=color_frame,
            depth_frame=depth_frame,
            points=points,
            metric=args.metric,
            frame_index=int(frame_idx),
            panel_width=int(args.panel_width),
            point_radius=int(args.point_radius),
        )
        out_path = out_dir / f"{session_dir.name}_frame_{int(frame_idx):06d}_{args.metric}.png"
        cv2.imwrite(str(out_path), composed)
        manifest.append(
            {
                "frame_index": int(frame_idx),
                "depth_frame_index": int(depth_frame_idx),
                "points_rendered": int(len(points)),
                "path": str(out_path),
            }
        )

    if cap is not None:
        cap.release()
    manifest_path = out_dir / "overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(manifest)} overlays to {out_dir}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extrinsics-json", type=Path, default=None)
    parser.add_argument("--depth-patch-r", type=int, default=2)
    parser.add_argument("--semantic-patch-r", type=int, default=1)
    parser.add_argument(
        "--use-distortion",
        action="store_true",
        help="Apply lens distortion during projection. Disabled by default to match the Branch 1 autolabel path.",
    )
    parser.add_argument("--max-time-diff-ms", type=float, default=35.0)
    parser.add_argument("--foreground-occlusion-m", type=float, default=0.20)
    parser.add_argument("--sigma-min-m", type=float, default=0.08)
    parser.add_argument("--sigma-frac", type=float, default=0.04)
    parser.add_argument("--candidate-residual-m", type=float, default=0.45)
    parser.add_argument("--candidate-p-depth-match-max", type=float, default=0.20)
    parser.add_argument("--candidate-dbscan-eps-m", type=float, default=0.54)
    parser.add_argument("--candidate-min-samples", type=int, default=2)
    parser.add_argument("--no-semantics", action="store_true", help="Skip optional OneFormer seg lookup.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_session = sub.add_parser("annotate-session", help="Annotate one session.")
    p_session.add_argument("--session-dir", type=Path, required=True)
    p_session.add_argument("--radar-csv", type=Path, default=None)
    p_session.add_argument("--sync-csv", type=Path, default=None)
    p_session.add_argument("--out-csv", type=Path, default=None)
    p_session.add_argument("--report-json", type=Path, default=None)
    add_common_args(p_session)
    p_session.set_defaults(func=cmd_annotate_session)

    p_root = sub.add_parser("annotate-root", help="Annotate sessions under a processing root.")
    p_root.add_argument("--processing-root", type=Path, required=True)
    p_root.add_argument("--out-root", type=Path, required=True)
    p_root.add_argument("--session-glob", default="session_*")
    p_root.add_argument("--max-sessions", type=int, default=None)
    add_common_args(p_root)
    p_root.set_defaults(func=cmd_annotate_root)

    p_report = sub.add_parser("report", help="Collect per-session reports into a CSV.")
    p_report.add_argument("--root", type=Path, required=True)
    p_report.add_argument("--out", type=Path, required=True)
    p_report.set_defaults(func=cmd_report)

    p_overlay = sub.add_parser("overlay", help="Render QA overlays from an annotated CSV.")
    p_overlay.add_argument("--session-dir", type=Path, required=True)
    p_overlay.add_argument("--input-csv", type=Path, required=True)
    p_overlay.add_argument("--out-dir", type=Path, required=True)
    p_overlay.add_argument(
        "--metric",
        choices=["p_depth_match", "residual", "p_dir_doppler", "semantic", "teacher"],
        default="p_depth_match",
    )
    p_overlay.add_argument("--frame-selection", choices=["worst", "first", "even"], default="worst")
    p_overlay.add_argument("--max-frames", type=int, default=8)
    p_overlay.add_argument("--max-points-per-frame", type=int, default=600)
    p_overlay.add_argument("--panel-width", type=int, default=640)
    p_overlay.add_argument("--point-radius", type=int, default=4)
    p_overlay.set_defaults(func=cmd_overlay)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
