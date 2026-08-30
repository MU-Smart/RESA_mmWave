#!/usr/bin/env python3
"""
Copy raw session roots from their legacy GCS location to the canonical
target location, one session at a time, using manifests/raw_session_path_map.csv
as the single source of truth for source -> target pairs.

Session-level (not dataset-root-level) sync is required: each of the four
legacy dataset root prefixes (raw-data/raw_session_extracts/dataset{1..4}/)
has a stray zero-byte "folder marker" object with the exact same name as the
prefix, which makes `gcloud storage rsync` refuse to resolve the source at
the dataset-root level ("matched more than one URL"). Session-level folders
do not have this marker, so syncing per-session avoids it entirely.

This script is strictly additive: it never passes
--delete-unmatched-destination-objects, so it will not remove anything
already present at a target path (including any stray objects left over
from a previous bad copy attempt). Run verify_raw_copy.py afterward to see
what, if anything, still needs manual cleanup.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP_CSV = ROOT / "manifests" / "raw_session_path_map.csv"
GCLOUD = shutil.which("gcloud")
if GCLOUD is None:
    raise SystemExit("gcloud not found on PATH")


def load_rows(map_csv: Path, dataset_filter: set[str] | None) -> list[dict[str, str]]:
    with map_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if dataset_filter:
        rows = [r for r in rows if r["legacy_dataset_name"] in dataset_filter]
    return rows


def sync_one(row: dict[str, str], dry_run: bool) -> tuple[str, bool, str]:
    session_id = row["session_id"]
    src = row["current_raw_path"]
    dst = row["target_raw_path"]
    cmd = [GCLOUD, "--quiet", "--verbosity=error", "storage", "rsync", "-r"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    msg = (proc.stdout + proc.stderr).strip()
    return session_id, ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Legacy dataset name to copy (e.g. dataset4). Repeatable. Default: all four, smallest first.",
    )
    ap.add_argument("--parallel", type=int, default=6, help="Concurrent rsync processes.")
    ap.add_argument("--dry-run", action="store_true", help="Pass --dry-run through to gcloud storage rsync.")
    ap.add_argument(
        "--map-csv",
        type=Path,
        default=DEFAULT_MAP_CSV,
        help="Path map CSV to read (session_id, legacy_dataset_name, current_raw_path, target_raw_path, ...). "
             "Default: manifests/raw_session_path_map.csv. current_raw_path may be a gs:// URI or a local path.",
    )
    args = ap.parse_args()

    dataset_filter = set(args.dataset) if args.dataset else None
    rows = load_rows(args.map_csv, dataset_filter)
    if not rows:
        print("No matching rows in raw_session_path_map.csv", file=sys.stderr)
        return 1

    # Smallest-dataset-first ordering by default, so a validation batch runs
    # before the larger, slower ones.
    order = {"dataset4": 0, "dataset3": 1, "dataset2": 2, "dataset1": 3}
    rows.sort(key=lambda r: order.get(r["legacy_dataset_name"], 99))

    print(f"Copying {len(rows)} session(s) with {args.parallel} parallel workers"
          f"{' (dry run)' if args.dry_run else ''}.")

    t0 = time.time()
    failures: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(sync_one, row, args.dry_run): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            session_id, ok, msg = fut.result()
            done += 1
            status = "OK" if ok else "FAIL"
            print(f"[{done}/{len(rows)}] {status} {row['legacy_dataset_name']}/{session_id}")
            if not ok:
                failures.append((f"{row['legacy_dataset_name']}/{session_id}", msg))
                print(f"    {msg}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. {len(rows) - len(failures)}/{len(rows)} session(s) succeeded.")
    if failures:
        print(f"{len(failures)} failure(s):")
        for name, msg in failures:
            print(f"  {name}: {msg.splitlines()[0] if msg else '(no output)'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
