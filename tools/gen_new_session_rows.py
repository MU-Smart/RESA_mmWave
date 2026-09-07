#!/usr/bin/env python3
"""
Scan a local directory of freshly-captured, unstructured raw sessions and
generate the manifest rows + local->GCS path map needed to fold them into
the canonical raw/sessions/ layout, without touching session_catalog.csv or
raw_session_path_map.csv directly.

Each "group" is a top-level folder under --source-root whose immediate
children are session_YYYY-MM-DD_HH-MM-SS directories (arbitrary nesting
depth to the session folder is fine; this script only requires session
folders to be discoverable via a directory glob per group, matched by name).

Writes (never touches the master manifests):
    manifests/new_sessions/<label>_catalog_rows.csv       (session_catalog.csv format)
    manifests/new_sessions/<label>_path_map_rows.csv       (raw_session_path_map.csv format,
                                                             current_raw_path is a local D:\\ path)
    manifests/new_sessions/<label>_skipped.csv             (corrupted/incomplete sessions excluded)

This script is read-only against the source drive and against the repo's
existing manifests -- it only writes new files under manifests/new_sessions/.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "manifests" / "new_sessions"
BUCKET = "miamioh-resa-data"


def corruption_reason(session_dir: Path) -> str | None:
    meta = session_dir / "meta_data.json"
    bin_path = session_dir / f"{session_dir.name}.bin"
    meta_len = meta.stat().st_size if meta.exists() else -1
    bin_len = bin_path.stat().st_size if bin_path.exists() else -1
    reasons = []
    if meta_len < 0:
        reasons.append("meta_data.json missing")
    elif meta_len == 0:
        reasons.append("meta_data.json is zero bytes")
    if bin_len < 0:
        reasons.append(f"{session_dir.name}.bin missing")
    elif bin_len == 0:
        reasons.append(f"{session_dir.name}.bin is zero bytes")
    return "; ".join(reasons) if reasons else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", required=True, help=r"Local folder to scan, e.g. D:\Data\McVey")
    ap.add_argument("--dataset-id", required=True, help="e.g. ds_mcvey_2026-06-14")
    ap.add_argument("--legacy-name", required=True, help="Source group label, e.g. McVey or imu")
    ap.add_argument("--building", required=True, help="Building name, or 'unknown'")
    ap.add_argument("--label", required=True, help="Output file prefix, e.g. mcvey")
    ap.add_argument("--notes", default="", help="Extra notes appended to every catalog row.")
    args = ap.parse_args()

    source_root = Path(args.source_root)
    session_dirs = sorted(p for p in source_root.rglob("session_*") if p.is_dir())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_path = OUT_DIR / f"{args.label}_catalog_rows.csv"
    map_path = OUT_DIR / f"{args.label}_path_map_rows.csv"
    skipped_path = OUT_DIR / f"{args.label}_skipped.csv"

    catalog_cols = [
        "session_id", "dataset_id", "legacy_dataset_name", "building", "capture_date",
        "raw_path", "processed_path", "oneformer_batch_id", "quality_tier", "label_status",
        "split", "published_dataset_version", "checkpoint_usage", "notes",
    ]
    map_cols = [
        "session_id", "dataset_id", "legacy_dataset_name", "current_raw_path", "target_raw_path",
        "mapping_scope", "migration_status", "notes",
    ]

    kept = 0
    skipped: list[tuple[str, str]] = []
    with catalog_path.open("w", newline="", encoding="utf-8") as cf, map_path.open("w", newline="", encoding="utf-8") as mf:
        cw = csv.DictWriter(cf, fieldnames=catalog_cols)
        cw.writeheader()
        mw = csv.DictWriter(mf, fieldnames=map_cols)
        mw.writeheader()

        for session_dir in session_dirs:
            session_id = session_dir.name
            reason = corruption_reason(session_dir)
            if reason is not None:
                skipped.append((session_id, reason))
                continue
            capture_date = session_id[len("session_"):len("session_") + 10]
            target_raw_path = f"gs://{BUCKET}/CapstoneData/raw/sessions/{args.dataset_id}/{session_id}/"

            cw.writerow({
                "session_id": session_id,
                "dataset_id": args.dataset_id,
                "legacy_dataset_name": args.legacy_name,
                "building": args.building,
                "capture_date": capture_date,
                "raw_path": target_raw_path,
                "processed_path": "",
                "oneformer_batch_id": "",
                "quality_tier": "silver",
                "label_status": "raw",
                "split": "none",
                "published_dataset_version": "",
                "checkpoint_usage": "",
                "notes": args.notes,
            })
            mw.writerow({
                "session_id": session_id,
                "dataset_id": args.dataset_id,
                "legacy_dataset_name": args.legacy_name,
                "current_raw_path": str(session_dir) + "\\",
                "target_raw_path": target_raw_path,
                "mapping_scope": "session_root",
                "migration_status": "planned",
                "notes": "Fresh upload from external drive. Local source path, not GCS.",
            })
            kept += 1

    with skipped_path.open("w", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        sw.writerow(["session_id", "reason"])
        for session_id, reason in skipped:
            sw.writerow([session_id, reason])

    print(f"Scanned {len(session_dirs)} session folder(s) under {source_root}")
    print(f"  kept: {kept} -> {catalog_path.name}, {map_path.name}")
    print(f"  skipped (corrupted): {len(skipped)} -> {skipped_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
