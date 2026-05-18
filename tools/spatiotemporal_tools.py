#!/usr/bin/env python3
"""Spatiotemporal tools for the radar navigation pipeline.

Subcommands:

    fuse-labels          Fuse spatiotemporal motion annotations with OneFormer autolabels.
    compare-aggregation  Compare current scene aggregation with motion-aware aggregation.

Usage:
    python spatiotemporal_tools.py fuse-labels --help
    python spatiotemporal_tools.py compare-aggregation --help
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]

if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

from perception.scene_motion_adapter import (  # noqa: E402
    EgoMotionEstimate,
    FramePacket,
    MotionAwareConfig,
    aggregate_motion_aware_scene,
    annotate_ego_doppler,
    estimate_ego_velocity,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BUCKET_TO_CLASS: dict[str, str] = {
    "wall": "structure",
    "box_like": "structure",
    "door": "structure",
    "staircase": "structure",
    "posting": "structure",
    "obstacle": "structure",
    "pillar": "structure",
    "structure": "structure",
    "floor": "floor",
    "human": "human",
    "person": "human",
}

MIN_VALID_Z = -0.851
MIN_RANGE_M = 0.3
MAX_RANGE_M = 5.0


# =========================================================================
# fuse-labels subcommand
# =========================================================================

MOTION_TO_GHOST: dict[str, float] = {
    "static_world": 0.05,
    "ambiguous_motion": 0.40,
    "dynamic_or_ghost": 0.90,
    "unknown": 0.50,
}

MOTION_TO_STATIC_CONF: dict[str, float] = {
    "static_world": 1.00,
    "ambiguous_motion": 0.50,
    "dynamic_or_ghost": 0.05,
    "unknown": 0.50,
}

PASSTHROUGH_COLS = [
    "video_frame_index", "radar_frame_num", "match_time_diff_us",
    "u", "v", "ade_id", "ade_name", "bucket", "maj_frac",
    "x", "y", "z", "cam_z", "doppler", "snr", "range_bin", "doppler_bin",
]

FUSION_COLS = [
    "ego_expected_doppler_mps",
    "ego_doppler_residual_mps",
    "motion_label",
    "spatiotemporal_weight",
    "temporal_persistence_frames",
    "ghost_score",
    "static_confidence",
    "final_train_weight",
    "ego_available",
    "label_source",
]


def _load_timestamps(session_dir: Path) -> dict[int, int]:
    csv_path = session_dir / f"{session_dir.name}.csv"
    out: dict[int, int] = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fn = int(float(row.get("radar_frame_num", row.get("frame_num", 0))))
            if fn not in out:
                out[fn] = int(float(row.get("timestamp_us", 0)))
    return out


def _load_labeled_frames(
    session_dir: Path,
) -> tuple[list[tuple[int, list[dict[str, Any]]]], list[dict[str, Any]]]:
    csv_path = session_dir / "labeled_radar_points_v4.csv"
    if not csv_path.exists():
        return [], []

    by_frame: dict[int, list[dict[str, Any]]] = {}
    raw_rows: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            point: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    point[k] = float(v)
                except (ValueError, TypeError):
                    point[k] = v

            bucket = str(point.get("bucket", "")).strip().lower()
            pred_class = BUCKET_TO_CLASS.get(bucket)
            if pred_class is None:
                continue
            point["pred_class"] = pred_class

            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            z = float(point.get("z", 0.0))
            rng = float(np.sqrt(x**2 + y**2 + z**2))
            if z <= MIN_VALID_Z or rng < MIN_RANGE_M or rng > MAX_RANGE_M:
                continue
            point["range_m"] = rng

            fn = int(float(point.get("radar_frame_num", point.get("frame_num", 0))))
            point["_frame_num"] = fn
            by_frame.setdefault(fn, []).append(point)
            raw_rows.append(point)

    return sorted(by_frame.items()), raw_rows


def _grid_key(point: dict[str, Any], grid_size: float) -> tuple[str, int, int, int]:
    return (
        str(point.get("pred_class", "")),
        int(np.floor(float(point.get("x", 0.0)) / grid_size)),
        int(np.floor(float(point.get("y", 0.0)) / grid_size)),
        int(np.floor(float(point.get("z", 0.0)) / grid_size)),
    )


def _compute_persistence(
    window: list[tuple[int, list[dict[str, Any]]]],
    grid_size: float,
) -> dict[tuple, int]:
    cell_frames: dict[tuple, set[int]] = {}
    for fn, pts in window:
        for pt in pts:
            cls = str(pt.get("pred_class", ""))
            if cls not in {"structure", "floor"}:
                continue
            label = str(pt.get("motion_label", "unknown"))
            if label == "dynamic_or_ghost":
                continue
            key = _grid_key(pt, grid_size)
            cell_frames.setdefault(key, set()).add(fn)
    return {k: len(v) for k, v in cell_frames.items()}


def _final_weight(
    pred_class: str,
    motion_label: str,
    spatiotemporal_weight: float,
    maj_frac: float,
    ego_available: bool,
) -> float:
    if pred_class == "human" or not ego_available:
        return float(np.clip(float(maj_frac), 0.0, 1.0))
    return float(np.clip(float(spatiotemporal_weight) * float(maj_frac), 0.0, 1.0))


def annotate_session(
    session_dir: Path,
    config: MotionAwareConfig,
    window_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames, raw_rows = _load_labeled_frames(session_dir)
    if not frames:
        return [], {"session": session_dir.name, "labeled_rows": 0, "status": "no_labeled_csv"}

    timestamps = _load_timestamps(session_dir)
    packets = [
        FramePacket(frame_num=fn, timestamp_us=timestamps.get(fn, 0), points=pts)
        for fn, pts in frames
    ]

    ego_by_frame: dict[int, EgoMotionEstimate] = {
        pkt.frame_num: estimate_ego_velocity(pkt.points, config)
        for pkt in packets
    }

    annotated_by_frame: dict[int, list[dict[str, Any]]] = {}
    for pkt in packets:
        ego = ego_by_frame[pkt.frame_num]
        annotated_by_frame[pkt.frame_num] = annotate_ego_doppler(pkt.points, ego, config)

    annotated_frames_list = [(fn, annotated_by_frame[fn]) for fn, _ in frames]

    persistence_by_frame: dict[int, dict[tuple, int]] = {}
    for anchor_idx, (anchor_fn, _) in enumerate(frames):
        w_start = max(0, anchor_idx - window_frames + 1)
        window_slice = annotated_frames_list[w_start: anchor_idx + 1]
        persistence_by_frame[anchor_fn] = _compute_persistence(window_slice, config.grid_size_m)

    annotation_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for fn, pts in annotated_by_frame.items():
        for pt in pts:
            rb = int(float(pt.get("range_bin", -1)))
            db = int(float(pt.get("doppler_bin", -1)))
            annotation_lookup[(fn, rb, db)] = pt

    n_ego_available = 0
    n_dynamic = 0
    n_static = 0
    n_unknown = 0

    for point in raw_rows:
        fn = int(point["_frame_num"])
        rb = int(float(point.get("range_bin", -1)))
        db = int(float(point.get("doppler_bin", -1)))

        ann = annotation_lookup.get((fn, rb, db), point)
        motion_label = str(ann.get("motion_label", "unknown"))
        st_weight = float(ann.get("spatiotemporal_weight", 1.0))
        ego_avail = bool(ego_by_frame.get(fn, EgoMotionEstimate(False, 0, 0, 0, 0, 0, float("inf"), 0)).available)

        key = _grid_key(point, config.grid_size_m)
        persist = persistence_by_frame.get(fn, {}).get(key, 1)

        pred_class = str(point.get("pred_class", ""))
        maj_frac = float(point.get("maj_frac", 1.0))
        fw = _final_weight(pred_class, motion_label, st_weight, maj_frac, ego_avail)

        point["ego_expected_doppler_mps"] = round(float(ann.get("ego_expected_doppler_mps", 0.0)), 4)
        point["ego_doppler_residual_mps"] = round(float(ann.get("ego_doppler_residual_mps", 0.0)), 4)
        point["motion_label"] = motion_label
        point["spatiotemporal_weight"] = round(st_weight, 4)
        point["temporal_persistence_frames"] = int(persist)
        point["ghost_score"] = MOTION_TO_GHOST.get(motion_label, 0.5)
        point["static_confidence"] = MOTION_TO_STATIC_CONF.get(motion_label, 0.5)
        point["final_train_weight"] = round(fw, 4)
        point["ego_available"] = bool(ego_avail)
        point["label_source"] = "camera_oneformer_depth_gated"

        if ego_avail:
            n_ego_available += 1
        if motion_label == "dynamic_or_ghost":
            n_dynamic += 1
        elif motion_label == "static_world":
            n_static += 1
        elif motion_label == "unknown":
            n_unknown += 1

    stats: dict[str, Any] = {
        "session": session_dir.name,
        "labeled_rows": len(raw_rows),
        "frames": len(frames),
        "ego_available_frames": int(sum(1 for e in ego_by_frame.values() if e.available)),
        "n_ego_available_points": n_ego_available,
        "n_static_world_points": n_static,
        "n_dynamic_or_ghost_points": n_dynamic,
        "n_unknown_points": n_unknown,
        "mean_final_train_weight": float(
            np.mean([float(p["final_train_weight"]) for p in raw_rows]) if raw_rows else 0.0
        ),
        "status": "ok",
    }
    return raw_rows, stats


def _output_columns(rows: list[dict[str, Any]]) -> list[str]:
    all_keys: list[str] = []
    for col in PASSTHROUGH_COLS:
        if col not in all_keys:
            all_keys.append(col)
    for col in FUSION_COLS:
        if col not in all_keys:
            all_keys.append(col)
    seen = set(all_keys)
    if rows:
        for k in rows[0]:
            if k not in seen and not k.startswith("_"):
                all_keys.append(k)
                seen.add(k)
    return all_keys


def _write_fused_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = _output_columns(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


def _build_fuse_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "fuse-labels",
        help="Fuse spatiotemporal motion annotations with OneFormer autolabels.",
        description="Fuse spatiotemporal motion annotations with OneFormer autolabels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processing-root", required=True, type=Path,
                   help="Root directory containing session_* subdirectories")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output root (mirrors session structure). Defaults to processing-root if --in-place.")
    p.add_argument("--in-place", action="store_true",
                   help="Write fused CSVs back into the original session directories (safe: uses _fused suffix)")
    p.add_argument("--session-limit", type=int, default=None,
                   help="Process only the first N sessions (for testing)")
    p.add_argument("--window-frames", type=int, default=4,
                   help="Temporal window size for persistence (default: 4)")
    p.add_argument("--session-prefix", default="session_2026-04-28_",
                   help="Only process session directories with this prefix")
    p.add_argument("--output-csv-name", default="labeled_radar_points_v4_fused.csv",
                   help="Filename for per-session output CSV (default: labeled_radar_points_v4_fused.csv)")


def _run_fuse(args: argparse.Namespace) -> None:
    root = Path(args.processing_root).resolve()
    if args.in_place:
        out_root = root
    elif args.output_dir is not None:
        out_root = Path(args.output_dir).resolve()
    else:
        print("ERROR: provide --output-dir or use --in-place", file=sys.stderr)
        sys.exit(1)

    sessions = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(args.session_prefix)
    )
    if args.session_limit:
        sessions = sessions[: args.session_limit]

    if not sessions:
        print(f"No sessions found under {root} with prefix '{args.session_prefix}'", file=sys.stderr)
        sys.exit(1)

    config = MotionAwareConfig()
    all_stats: list[dict[str, Any]] = []

    for i, session_dir in enumerate(sessions):
        print(f"[{i+1}/{len(sessions)}] {session_dir.name} ...", end=" ", flush=True)
        rows, stats = annotate_session(session_dir, config, args.window_frames)
        if rows:
            if args.in_place:
                dest = session_dir / args.output_csv_name
            else:
                dest = out_root / session_dir.name / args.output_csv_name
            _write_fused_csv(dest, rows)
            print(
                f"{stats['labeled_rows']} pts | "
                f"ego_avail_frames={stats['ego_available_frames']}/{stats['frames']} | "
                f"ghost={stats['n_dynamic_or_ghost_points']} | "
                f"mean_weight={stats['mean_final_train_weight']:.3f}"
            )
        else:
            print(f"SKIPPED ({stats['status']})")
        all_stats.append(stats)

    summary = {
        "sessions_processed": len(sessions),
        "sessions_ok": int(sum(1 for s in all_stats if s["status"] == "ok")),
        "total_labeled_rows": int(sum(s.get("labeled_rows", 0) for s in all_stats)),
        "total_ego_available_frames": int(sum(s.get("ego_available_frames", 0) for s in all_stats)),
        "total_dynamic_or_ghost_points": int(sum(s.get("n_dynamic_or_ghost_points", 0) for s in all_stats)),
        "total_static_world_points": int(sum(s.get("n_static_world_points", 0) for s in all_stats)),
        "total_unknown_points": int(sum(s.get("n_unknown_points", 0) for s in all_stats)),
        "mean_final_train_weight": float(
            np.mean([s["mean_final_train_weight"] for s in all_stats if s.get("labeled_rows", 0) > 0])
        ),
        "output_csv_name": args.output_csv_name,
        "window_frames": args.window_frames,
        "sessions": all_stats,
    }

    summary_path = out_root / "fusion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {summary['sessions_ok']}/{summary['sessions_processed']} sessions.")
    print(f"Total labeled rows: {summary['total_labeled_rows']}")
    print(f"Ghost/dynamic points: {summary['total_dynamic_or_ghost_points']}")
    print(f"Mean final_train_weight: {summary['mean_final_train_weight']:.3f}")
    print(f"Summary: {summary_path}")


# =========================================================================
# compare-aggregation subcommand
# =========================================================================

CSV_RENAME = {
    "radar_frame_num": "frame",
    "frame_num": "frame",
    "doppler_mps": "doppler",
    "v": "doppler",
    "power_snr": "snr",
    "bucket": "pred_class",
}

DEFAULT_COMPARE_OUTPUT_DIR = THIS_FILE.parent / "results" / "scene_motion_comparison"


def _load_aggregate_scene():
    path = THIS_FILE.parent / "scene_aggregator.py"
    spec = importlib.util.spec_from_file_location("spatiotemporal_branch_scene_agg", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load aggregator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aggregate_scene


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _timestamp_us(row: dict[str, str]) -> int:
    if row.get("timestamp_us"):
        return int(float(row["timestamp_us"]))
    if row.get("timestamp_ms"):
        return int(float(row["timestamp_ms"])) * 1000
    if row.get("timestamp_ns"):
        return int(float(row["timestamp_ns"])) // 1000
    raise ValueError("timestamp row has no timestamp_us/timestamp_ms/timestamp_ns")


def _load_radar_timestamps(session_dir: Path) -> dict[int, int]:
    path = session_dir / f"{session_dir.name}_radar_timestamps.csv"
    out: dict[int, int] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(float(row.get("frame_index", row.get("frame_num", len(out)))))
            out[frame] = _timestamp_us(row)
    return out


def _point_range(point: dict[str, Any]) -> float:
    if "range_m" in point:
        return float(point["range_m"])
    return float(np.sqrt(float(point.get("x", 0.0)) ** 2 + float(point.get("y", 0.0)) ** 2 + float(point.get("z", 0.0)) ** 2))


def _load_compare_frames(session_dir: Path) -> list[tuple[int, list[dict[str, Any]]]]:
    csv_path = session_dir / "labeled_radar_points_v4.csv"
    if not csv_path.exists():
        csv_path = session_dir / f"{session_dir.name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No labeled or point CSV found in {session_dir}")

    by_frame: dict[int, list[dict[str, Any]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point: dict[str, Any] = {}
            for key, value in row.items():
                out_key = CSV_RENAME.get(key, key)
                try:
                    point[out_key] = float(value)
                except (ValueError, TypeError):
                    point[out_key] = value
            raw_cls = str(point.get("pred_class", point.get("bucket", ""))).strip().lower()
            mapped = BUCKET_TO_CLASS.get(raw_cls)
            if mapped is None:
                continue
            point["pred_class"] = mapped
            point["range_m"] = _point_range(point)
            if (
                float(point.get("z", 0.0)) <= MIN_VALID_Z
                or float(point.get("range_m", 0.0)) < MIN_RANGE_M
                or float(point.get("range_m", 0.0)) > MAX_RANGE_M
            ):
                continue
            frame = int(point.get("frame", point.get("video_frame_index", 0)))
            by_frame.setdefault(frame, []).append(point)
    return sorted(by_frame.items())


def _resolve_sessions(root: Path | None, session_args: list[str], limit: int | None) -> list[Path]:
    sessions: list[Path] = []
    if session_args:
        for item in session_args:
            path = Path(item)
            if not path.is_absolute() and root is not None:
                path = root / path
            sessions.append(path.resolve())
    elif root is not None:
        sessions = sorted(path.resolve() for path in root.iterdir() if path.is_dir() and path.name.startswith("session_"))
    if limit is not None:
        sessions = sessions[: max(0, int(limit))]
    return sessions


def _confidence_rank(value: str) -> int:
    return {"none": 0, "unknown": 0, "low": 1, "weak": 1, "medium": 2, "observed": 2, "strong": 3, "high": 3}.get(str(value), 0)


def _scene_row(
    session: str,
    frame_num: int,
    baseline_scene: dict[str, Any],
    motion_scene: dict[str, Any],
    motion_diag: dict[str, Any],
) -> dict[str, Any]:
    b_block = baseline_scene.get("blocking", {})
    m_block = motion_scene.get("blocking", {})
    b_open = baseline_scene.get("open_directions", {})
    m_open = motion_scene.get("open_directions", {})
    return {
        "session": session,
        "frame_num": int(frame_num),
        "baseline_points": int(baseline_scene.get("total_points", 0)),
        "motion_points": int(motion_scene.get("total_points", 0)),
        "motion_input_points": int(motion_diag.get("input_points", 0)),
        "motion_suppressed_points": int(motion_diag.get("suppressed_points", 0)),
        "motion_persistent_points": int(motion_diag.get("persistent_points", 0)),
        "motion_dynamic_or_ghost_points": int(motion_diag.get("dynamic_or_ghost_points", 0)),
        "baseline_center_effective": float(b_block.get("center_effective_points", b_block.get("center_points", 0.0))),
        "motion_center_effective": float(m_block.get("center_effective_points", m_block.get("center_points", 0.0))),
        "baseline_center_confidence": b_block.get("center_confidence", "none"),
        "motion_center_confidence": m_block.get("center_confidence", "none"),
        "baseline_center_clear": bool(b_open.get("center_clear", True)),
        "motion_center_clear": bool(m_open.get("center_clear", True)),
        "baseline_confidence": baseline_scene.get("confidence", "low"),
        "motion_confidence": motion_scene.get("confidence", "low"),
        "ego_available": bool(motion_diag.get("anchor_ego", {}).get("available", False)),
        "ego_speed_mps": float(motion_diag.get("anchor_ego", {}).get("speed_mps", 0.0)),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def mean(key: str) -> float:
        return float(statistics.fmean(float(row[key]) for row in rows))

    downgraded = sum(
        1 for row in rows
        if _confidence_rank(row["motion_center_confidence"]) < _confidence_rank(row["baseline_center_confidence"])
    )
    upgraded = sum(
        1 for row in rows
        if _confidence_rank(row["motion_center_confidence"]) > _confidence_rank(row["baseline_center_confidence"])
    )
    clear_changed = sum(1 for row in rows if row["baseline_center_clear"] != row["motion_center_clear"])
    suppressed_total = sum(int(row["motion_suppressed_points"]) for row in rows)
    persistent_total = sum(int(row["motion_persistent_points"]) for row in rows)
    dynamic_total = sum(int(row["motion_dynamic_or_ghost_points"]) for row in rows)
    return {
        "frames": int(len(rows)),
        "ego_available_frames": int(sum(1 for row in rows if row["ego_available"])),
        "baseline_mean_points": mean("baseline_points"),
        "motion_mean_points": mean("motion_points"),
        "motion_mean_input_points": mean("motion_input_points"),
        "suppressed_points_total": int(suppressed_total),
        "dynamic_or_ghost_points_total": int(dynamic_total),
        "persistent_points_total": int(persistent_total),
        "mean_center_effective_delta": float(mean("motion_center_effective") - mean("baseline_center_effective")),
        "center_confidence_downgraded_frames": int(downgraded),
        "center_confidence_upgraded_frames": int(upgraded),
        "center_clear_changed_frames": int(clear_changed),
        "baseline_center_confidence_counts": dict(Counter(str(row["baseline_center_confidence"]) for row in rows)),
        "motion_center_confidence_counts": dict(Counter(str(row["motion_center_confidence"]) for row in rows)),
    }


def run_session(session_dir: Path, aggregate_scene, config: MotionAwareConfig, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame_rows = _load_compare_frames(session_dir)
    timestamps = _load_radar_timestamps(session_dir)
    if not frame_rows:
        return [], {"session": session_dir.name, "usable": False, "reason": "no_frames"}

    rows: list[dict[str, Any]] = []
    window: deque[FramePacket] = deque(maxlen=int(config.window_frames))
    for idx, (frame_num, points) in enumerate(frame_rows):
        if idx % max(1, int(args.frame_step)) != 0:
            continue
        if args.max_frames is not None and len(rows) >= int(args.max_frames):
            break
        ts_us = int(timestamps.get(frame_num, frame_num * 100000))
        packet = FramePacket(frame_num=frame_num, timestamp_us=ts_us, points=points)
        window.append(packet)

        baseline_ego = estimate_ego_velocity(points, config)
        baseline_scene = aggregate_scene(
            points,
            window_frames=1,
            ego_velocity_mps=baseline_ego.as_payload() if baseline_ego.available else None,
        )
        motion_scene, motion_window = aggregate_motion_aware_scene(list(window), aggregate_scene, config)
        rows.append(_scene_row(session_dir.name, frame_num, baseline_scene, motion_scene, motion_window.diagnostics))

    summary = _summarize_rows(rows)
    summary.update({
        "session": session_dir.name,
        "usable": True,
        "source_frames": int(len(frame_rows)),
        "evaluated_frames": int(len(rows)),
    })
    return rows, summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_compare_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "compare-aggregation",
        help="Compare current scene aggregation with motion-aware aggregation.",
        description="Compare current scene aggregation with motion-aware aggregation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processing-root", type=Path, default=None)
    p.add_argument("--session", action="append", default=[])
    p.add_argument("--session-limit", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARE_OUTPUT_DIR)
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--window-frames", type=int, default=4)
    p.add_argument("--min-persist-frames", type=int, default=2)
    p.add_argument("--grid-size-m", type=float, default=0.35)
    p.add_argument("--static-residual-mps", type=float, default=0.25)
    p.add_argument("--dynamic-residual-mps", type=float, default=0.55)
    p.add_argument("--max-ego-speed-mps", type=float, default=2.5)
    p.add_argument("--min-static-inlier-fraction", type=float, default=0.25)


def _run_compare(args: argparse.Namespace) -> int:
    if args.processing_root is None and not args.session:
        raise SystemExit("[ERROR] Provide --processing-root or --session")
    root = args.processing_root.resolve() if args.processing_root is not None else None
    sessions = _resolve_sessions(root, args.session, args.session_limit)
    if not sessions:
        raise SystemExit("[ERROR] No sessions resolved")

    aggregate_scene = _load_aggregate_scene()
    config = MotionAwareConfig(
        window_frames=int(args.window_frames),
        static_residual_mps=float(args.static_residual_mps),
        dynamic_residual_mps=float(args.dynamic_residual_mps),
        max_ego_speed_mps=float(args.max_ego_speed_mps),
        min_static_inlier_fraction=float(args.min_static_inlier_fraction),
        grid_size_m=float(args.grid_size_m),
        min_persist_frames=int(args.min_persist_frames),
    )

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for session in sessions:
        print(f"[session] {session.name}", flush=True)
        try:
            rows, summary = run_session(session, aggregate_scene, config, args)
        except Exception as exc:
            rows = []
            summary = {"session": session.name, "usable": False, "reason": str(exc)}
        all_rows.extend(rows)
        summaries[session.name] = summary
        if summary.get("usable"):
            print(
                "[session]"
                f" frames={summary['evaluated_frames']}"
                f" suppressed={summary.get('suppressed_points_total', 0)}"
                f" center_downgraded={summary.get('center_confidence_downgraded_frames', 0)}",
                flush=True,
            )
        else:
            print(f"[session] skipped: {summary.get('reason')}", flush=True)

    aggregate_summary = _summarize_rows(all_rows)
    payload = {
        "method": "scene_motion_integration_comparison",
        "config": asdict(config),
        "sessions": summaries,
        "aggregate": aggregate_summary,
        "interpretation": {
            "suppressed_points_total": "Points removed from the temporal aggregation window by ego-Doppler/persistence gates.",
            "center_confidence_downgraded_frames": "Frames where motion-aware aggregation reduced center blocking confidence versus current-frame aggregation.",
            "persistent_points_total": "Motion-compensated structure/floor points that repeated in at least min_persist_frames grid cells.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "scene_motion_comparison_summary.json"
    rows_path = args.output_dir / "scene_motion_frame_comparison.csv"
    _write_json(summary_path, payload)
    write_rows_csv(rows_path, all_rows)
    print(f"[done] summary={summary_path}", flush=True)
    print(f"[done] frames_csv={rows_path}", flush=True)
    print(
        "[done]"
        f" frames={aggregate_summary.get('frames', 0)}"
        f" suppressed={aggregate_summary.get('suppressed_points_total', 0)}"
        f" center_downgraded={aggregate_summary.get('center_confidence_downgraded_frames', 0)}"
        f" center_changed={aggregate_summary.get('center_clear_changed_frames', 0)}",
        flush=True,
    )
    return 0


# =========================================================================
# CLI dispatch
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _build_fuse_parser(subparsers)
    _build_compare_parser(subparsers)

    args = parser.parse_args()

    if args.command == "fuse-labels":
        _run_fuse(args)
    elif args.command == "compare-aggregation":
        _run_compare(args)


if __name__ == "__main__":
    main()