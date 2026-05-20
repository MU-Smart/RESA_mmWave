#!/usr/bin/env python3
"""
prepare_autolabel.py
====================
Batch preparation script for the camera-derived OneFormer autolabel flow.

For each training session in data/dataTraining/:
  1. Generate radar point-cloud CSV (if missing) via adc_to_pointcloud_v6
  2. Write meta_data.json with 1280x720 camera intrinsics
  3. Build synchronized_{session}.csv from radar timestamps + seg frame count
  4. Write seg_meta.json from ADE20K id2label (HF cache)
  5. Run noah_autolabel_radar_using_synccsv_v4_bestmatch.py from
     LLM_ML/etc/noah_scripts/OneFormer/

This is the camera-segmentation path for generating true projected labels.
It is distinct from LLM_ML/jetson_nav_pipeline/3branch/preprocess_sessions.py,
which writes branch-local pseudo-labels by running the Branch 1 checkpoint.

Run from LLM_ML/ directory:
    python3 prepare_autolabel.py [--dry-run] [--skip-radar-gen] [--skip-autolabel]
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================================
# Paths
# ============================================================================

THIS_DIR     = Path(__file__).resolve().parent
REPO_ROOT    = THIS_DIR.parents[2]
DATA_ROOT    = REPO_ROOT / "data" / "dataTraining"
AUTOLABEL_SCRIPT = (
    THIS_DIR
    / "noah_scripts"
    / "OneFormer"
    / "noah_autolabel_radar_using_synccsv_v4_bestmatch.py"
)
EXTRINSICS_JSON = REPO_ROOT / "config" / "radar_camera_extrinsics.json"
HF_CONFIG    = Path(
    "/home/hullumdr/.cache/huggingface/hub"
    "/models--shi-labs--oneformer_ade20k_swin_tiny"
    "/snapshots/05f2812b1eccf9909b3897777450f8d68148cafc"
    "/config.json"
)


def _first_existing_path(candidates: list[Path], default: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default


def _radar_cfg_path() -> Path:
    env_cfg = os.environ.get("RADAR_CFG")
    candidates = []
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates.extend(
        [
            # Running from LLM_ML/prepare_autolabel.py.
            REPO_ROOT / "config" / "profile_objdet.cfg",
            # Running from /content/work/code/prepare_autolabel.py.
            THIS_DIR / "canon" / "config" / "profile_objdet.cfg",
            # Running from canon/processing/prepare_autolabel.py.
            THIS_DIR.parent / "config" / "profile_objdet.cfg",
            # Local checkout fallback when launched from notebooks.
            Path("/content/work/code/canon/config/profile_objdet.cfg"),
        ]
    )
    return _first_existing_path(candidates, candidates[0])


def _add_adc_module_paths() -> None:
    candidates = [
        # Running from LLM_ML/prepare_autolabel.py.
        REPO_ROOT,
        REPO_ROOT / "perception",
        REPO_ROOT / "models",
        # Running from /content/work/code/prepare_autolabel.py.
        THIS_DIR / "canon" / "perception",
        # Running from canon/processing/prepare_autolabel.py.
        THIS_DIR.parent / "perception",
        THIS_DIR.parents[1] / "etc" if len(THIS_DIR.parents) > 1 else THIS_DIR / "etc",
    ]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))


RADAR_CFG = _radar_cfg_path()

# Camera intrinsics at 1280x720 (training video resolution).
# Calibrated from checkerboard images; dist_coeffs are dimensionless.
CAM_W, CAM_H = 1280, 720
K_1280x720 = [
    [1362.2661031864602, 0.0,               743.6851887988306],
    [0.0,               1359.1988002634798, 230.50777574003672],
    [0.0,               0.0,               1.0               ],
]
DIST_COEFFS = [
    -0.5132266243084922,
    6.088778241160198,
    -0.0036906927493171415,
    0.016100547661819797,
    -35.09708723510832,
]

# ============================================================================
# Helpers — radar timestamps
# ============================================================================

def radar_timestamps_from_manifest(session_dir: Path) -> list[int] | None:
    """Return list of per-frame timestamps (ms) from hybrid_rd/manifest.json, or None."""
    manifest_path = session_dir / "hybrid_rd" / "manifest.json"
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [f["timestamp_ms"] for f in data["frames"]]


def radar_timestamps_from_csv(csv_path: Path) -> list[int]:
    """
    Derive per-frame timestamps (ms) from a radar point-cloud CSV.
    Takes the first timestamp_us seen for each frame_num.
    """
    per_frame: dict[int, int] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        frame_col = None
        ts_col = None
        for row in reader:
            if frame_col is None:
                for c in ("frame_num", "radar_frame_num", "frame"):
                    if c in row:
                        frame_col = c
                        break
                for c in ("timestamp_us",):
                    if c in row:
                        ts_col = c
                        break
                if frame_col is None or ts_col is None:
                    break
            fn = int(row[frame_col])
            if fn not in per_frame:
                per_frame[fn] = int(float(row[ts_col])) // 1000  # us -> ms
    return [per_frame[k] for k in sorted(per_frame)]


def video_timestamps_from_csv(session_dir: Path) -> list[int] | None:
    """Return per-frame camera timestamps (ms) from *_color_timestamps.csv, if present."""
    ts_path = session_dir / f"{session_dir.name}_color_timestamps.csv"
    if not ts_path.exists():
        return None

    per_frame: dict[int, int] = {}
    with ts_path.open(newline="") as f:
        reader = csv.DictReader(f)
        frame_col = None
        ts_col = None
        for row in reader:
            if frame_col is None:
                for c in ("frame_index", "frame_num", "video_frame_index"):
                    if c in row:
                        frame_col = c
                        break
                for c in ("timestamp_ms", "timestamp_us"):
                    if c in row:
                        ts_col = c
                        break
                if frame_col is None or ts_col is None:
                    return None
            frame_idx = int(row[frame_col])
            ts_value = float(row[ts_col])
            per_frame[frame_idx] = int(ts_value // 1000) if ts_col == "timestamp_us" else int(ts_value)
    return [per_frame[k] for k in sorted(per_frame)]


# ============================================================================
# Helpers — radar timestamps from raw bin (fallback, not used if CSV available)
# ============================================================================

def radar_timestamps_from_bin(bin_path: Path) -> list[int]:
    """Parse frame headers from .bin file: each frame is struct <QII> + payload."""
    HEADER_FMT = "<QII"
    HEADER_SZ = struct.calcsize(HEADER_FMT)
    timestamps = []
    with bin_path.open("rb") as f:
        while True:
            hdr = f.read(HEADER_SZ)
            if len(hdr) < HEADER_SZ:
                break
            ts_ms, payload_len, _ = struct.unpack(HEADER_FMT, hdr)
            timestamps.append(ts_ms)
            f.seek(payload_len, 1)
    return timestamps


# ============================================================================
# Step 1 — Generate radar CSV
# ============================================================================

def generate_radar_csv(session_dir: Path, processor, dry_run: bool) -> Path | None:
    """Generate {session}.csv if it doesn't already exist. Returns CSV path."""
    import adc_to_pointcloud_v6 as v6

    preferred = session_dir / f"{session_dir.name}.csv"
    if preferred.exists():
        print(f"  [CSV] already exists: {preferred.name}")
        return preferred

    print(f"  [CSV] generating radar CSV for {session_dir.name} …")
    if dry_run:
        print(f"  [CSV] (dry-run) would call process_session_fast")
        return None

    try:
        ok, n_det, csv_path = v6.process_session_fast(processor, session_dir)
    except Exception as exc:
        print(f"  [CSV] ERROR: {exc}")
        return None

    if not ok:
        print(f"  [CSV] FAILED: {csv_path}")
        return None

    print(f"  [CSV] wrote {n_det} detections -> {Path(csv_path).name}")
    return Path(csv_path)


# ============================================================================
# Step 2 — Write meta_data.json
# ============================================================================

def write_meta_data(session_dir: Path, dry_run: bool) -> None:
    out_path = session_dir / "meta_data.json"
    meta = {
        "session": session_dir.name,
        "width": CAM_W,
        "height": CAM_H,
        "recorder": "prepare_autolabel.py",
        "realsense_calibration": {
            "rgb": {
                "K": K_1280x720,
                "fx": K_1280x720[0][0],
                "fy": K_1280x720[1][1],
                "cx": K_1280x720[0][2],
                "cy": K_1280x720[1][2],
                "dist_coeffs": DIST_COEFFS,
                "width": CAM_W,
                "height": CAM_H,
            }
        },
    }
    if dry_run:
        print(f"  [META] (dry-run) would write meta_data.json")
        return
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [META] wrote meta_data.json (1280x720)")


# ============================================================================
# Step 3 — Build synchronized CSV
# ============================================================================

def build_sync_csv(
    session_dir: Path,
    radar_ts_ms: list[int],
    dry_run: bool,
) -> None:
    """
    For each radar frame, find the nearest video frame by timestamp.
    Video frames are assumed to span the same time window as radar at
    a uniform fps derived from n_seg_files / radar_duration.

    Writes: session_dir / synchronized_{session}.csv
    Columns: video_frame_index, radar_frame_nums, time_diffs_us
    """
    seg_files = sorted((session_dir / "seg").glob("*.npy"))
    n_seg = len(seg_files)
    if n_seg == 0:
        print(f"  [SYNC] no seg files found, skipping")
        return

    video_ts_ms = video_timestamps_from_csv(session_dir)
    if video_ts_ms:
        n_sync = min(n_seg, len(video_ts_ms))
        if n_sync != n_seg or n_sync != len(video_ts_ms):
            print(
                f"  [SYNC] camera timestamp count ({len(video_ts_ms)}) does not match seg count ({n_seg}); "
                f"using first {n_sync} frames"
            )
        video_ts = np.array(video_ts_ms[:n_sync], dtype=np.float64)
    else:
        n_radar = len(radar_ts_ms)
        t0 = radar_ts_ms[0]
        t_end = radar_ts_ms[-1]
        radar_duration_ms = t_end - t0 if t_end > t0 else max(n_radar * 100, 1)
        fps_video = (n_seg - 1) / (radar_duration_ms / 1000.0) if radar_duration_ms > 0 else 30.0
        video_ts = t0 + np.arange(n_seg) * 1000.0 / fps_video  # ms
        n_sync = n_seg

    radar_ts_arr = np.array(radar_ts_ms, dtype=np.float64)

    # Map each radar frame -> nearest video frame
    vid_to_radar: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for j, rt in enumerate(radar_ts_arr):
        diffs = np.abs(video_ts - rt)
        best_i = int(np.argmin(diffs))
        diff_us = int(round(diffs[best_i] * 1000))  # ms -> us
        vid_to_radar[best_i].append((j, diff_us))

    out_path = session_dir / f"synchronized_{session_dir.name}.csv"
    if dry_run:
        matched = sum(1 for i in range(n_sync) if i in vid_to_radar)
        if video_ts_ms:
            print(f"  [SYNC] (dry-run) using real camera timestamps, {matched}/{n_sync} video frames matched")
        else:
            print(f"  [SYNC] (dry-run) fps={fps_video:.2f}, {matched}/{n_sync} video frames matched")
        return

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_frame_index", "radar_frame_nums", "time_diffs_us"])
        for i in range(n_sync):
            entries = vid_to_radar.get(i, [])
            rfns = ";".join(str(j) for j, _ in entries)
            diffs_str = ";".join(str(d) for _, d in entries)
            writer.writerow([i, rfns, diffs_str])

    matched = sum(1 for i in range(n_sync) if i in vid_to_radar)
    if video_ts_ms:
        print(f"  [SYNC] wrote {n_sync} rows, {matched} matched, using real camera timestamps")
    else:
        print(f"  [SYNC] wrote {n_sync} rows, {matched} matched, fps={fps_video:.2f}")


# ============================================================================
# Step 4 — Write seg_meta.json
# ============================================================================

def write_seg_meta(session_dir: Path, id2label: dict, dry_run: bool) -> None:
    out_path = session_dir / "seg_meta.json"
    meta = {
        "model": "shi-labs/oneformer_ade20k_swin_tiny",
        "dataset": "ADE20K",
        "n_classes": len(id2label),
        "id2label": id2label,
    }
    if dry_run:
        print(f"  [SEG_META] (dry-run) would write seg_meta.json ({len(id2label)} classes)")
        return
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  [SEG_META] wrote seg_meta.json ({len(id2label)} classes)")


def load_id2label(data_root: Path) -> dict[str, str]:
    """Load ADE20K labels from HF cache, or reuse OneFormer session metadata."""
    if HF_CONFIG.exists():
        print(f"Loading ADE20K id2label from HF cache: {HF_CONFIG}")
        hf_cfg = json.loads(HF_CONFIG.read_text(encoding="utf-8"))
        return hf_cfg["id2label"]

    for seg_meta in sorted(data_root.glob("session_*/seg_meta.json")):
        try:
            meta = json.loads(seg_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        id2label = meta.get("id2label")
        if id2label:
            print(f"Loading ADE20K id2label from session metadata: {seg_meta}")
            return id2label

    raise FileNotFoundError(
        f"No HF config found at {HF_CONFIG} and no session seg_meta.json found under {data_root}"
    )


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare training sessions for autolabeling.")
    ap.add_argument(
        "--data-root",
        default=None,
        help="Override the session root directory (defaults to LLM_ML/data/dataTraining).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing files.")
    ap.add_argument("--skip-radar-gen", action="store_true", help="Skip radar CSV generation.")
    ap.add_argument("--skip-autolabel", action="store_true", help="Skip the autolabel run.")
    ap.add_argument("--session", default=None, help="Process only this session name (for debugging).")
    ap.add_argument(
        "--session-prefix",
        default="session_",
        help="Process only sessions whose directory name starts with this prefix.",
    )
    ap.add_argument(
        "--assign-mode",
        choices=["all_within_window", "best_per_radar"],
        default="all_within_window",
        help="Radar/video sync assignment mode passed to the autolabeler.",
    )
    ap.add_argument(
        "--use-distortion",
        action="store_true",
        help="Opt in to distortion-aware projection in the autolabeler.",
    )
    args = ap.parse_args()

    dry_run = args.dry_run
    data_root = Path(args.data_root).resolve() if args.data_root else DATA_ROOT

    # Load ADE20K id2label
    id2label = load_id2label(data_root)
    print(f"  Loaded {len(id2label)} classes.")

    # Discover sessions
    sessions = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith(args.session_prefix)
    )
    if args.session:
        sessions = [p for p in sessions if p.name == args.session]
        if not sessions:
            print(f"Session {args.session!r} not found in {data_root}")
            return 1

    print(f"\nFound {len(sessions)} sessions in {data_root}")

    # Build radar processor once (shared across sessions)
    processor = None
    if not args.skip_radar_gen:
        print(f"Building radar processor from {RADAR_CFG} …")
        sys.path.insert(0, str(THIS_DIR))
        _add_adc_module_paths()
        import adc_to_pointcloud_v6 as v6
        if not RADAR_CFG.exists():
            print(f"ERROR: radar cfg not found: {RADAR_CFG}")
            return 1
        processor = v6.build_processor(RADAR_CFG)
        print(f"  Processor ready.")

    failed_sessions: list[str] = []
    skipped_sessions: list[str] = []

    for idx, session_dir in enumerate(sessions):
        name = session_dir.name
        print(f"\n[{idx+1}/{len(sessions)}] {name}")

        seg_dir = session_dir / "seg"
        if not seg_dir.exists() or not any(seg_dir.glob("*.npy")):
            print(f"  [SKIP] no seg/ maps found")
            skipped_sessions.append(name)
            continue

        # Step 1: Generate radar CSV
        csv_path: Path | None = None
        if not args.skip_radar_gen:
            csv_path = generate_radar_csv(session_dir, processor, dry_run)
            if csv_path is None and not dry_run:
                print(f"  [SKIP] radar CSV generation failed")
                failed_sessions.append(name)
                continue
        else:
            preferred = session_dir / f"{name}.csv"
            if preferred.exists():
                csv_path = preferred
            else:
                # Fallback: any non-excluded CSV
                for p in sorted(session_dir.glob("*.csv")):
                    if p.name not in ("labeled_radar_points_v4.csv", "batch_autolabel_summary_v3.csv") \
                            and not p.name.startswith("synchronized_"):
                        csv_path = p
                        break

        # Step 2: meta_data.json
        write_meta_data(session_dir, dry_run)

        # Step 3: synchronized CSV
        radar_ts: list[int] | None = radar_timestamps_from_manifest(session_dir)
        if radar_ts is None:
            # Fall back to CSV timestamps (must have CSV at this point)
            if csv_path is not None and csv_path.exists():
                print(f"  [SYNC] no manifest — reading timestamps from CSV …")
                radar_ts = radar_timestamps_from_csv(csv_path)
            elif not dry_run:
                print(f"  [SYNC] no manifest and no CSV — skipping sync")
                skipped_sessions.append(name)
                continue
            else:
                print(f"  [SYNC] (dry-run) no manifest — would read from CSV")
                radar_ts = []

        if radar_ts:
            build_sync_csv(session_dir, radar_ts, dry_run)
        else:
            print(f"  [SYNC] empty radar timestamps, skipping")

        # Step 4: seg_meta.json
        write_seg_meta(session_dir, id2label, dry_run)

    print(f"\n{'='*60}")
    print(f"Preparation complete.")
    print(f"  OK:      {len(sessions) - len(failed_sessions) - len(skipped_sessions)}")
    print(f"  Skipped: {len(skipped_sessions)}")
    print(f"  Failed:  {len(failed_sessions)}")
    if failed_sessions:
        print(f"  Failed sessions: {failed_sessions}")

    if args.skip_autolabel or dry_run:
        if dry_run:
            print("\n[DRY-RUN] Skipping autolabel step.")
        return 0

    # -------------------------------------------------------------------------
    # Step 5: Run autolabel
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Running autolabel …")
    if not EXTRINSICS_JSON.exists():
        print(f"  [ERROR] extrinsics JSON not found: {EXTRINSICS_JSON}")
        return 1

    cmd = [
        sys.executable,
        str(AUTOLABEL_SCRIPT),
        "--processing_root_dir", str(data_root),
        "--synchronized_root_dir", str(data_root),
        "--extrinsics_json", str(EXTRINSICS_JSON),
        "--assign_mode", args.assign_mode,
    ]
    if args.use_distortion:
        cmd.append("--use_distortion")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(THIS_DIR))
    if result.returncode != 0:
        print(f"  [ERROR] autolabel exited with code {result.returncode}")
        return result.returncode

    print("Autolabel complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
