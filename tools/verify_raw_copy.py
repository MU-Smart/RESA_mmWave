#!/usr/bin/env python3
"""
Verify that the canonical raw/sessions/ tree is a complete, clean mirror of
each legacy raw-data/raw_session_extracts/<dataset>/ tree, by comparing full
relative-path listings (not just session/object counts).

Writes manifests/raw_copy_verification_<today>.json in the same schema as
the original verification report, but with complete missing/extra path
lists (not just 10-sample previews), so any leftover cleanup after
tools/copy_raw_sessions.py can be scoped exactly instead of guessed at.

This script is read-only: it only runs `gcloud storage ls -r` and never
modifies anything in the bucket.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKET = "miamioh-resa-data"
GCLOUD = shutil.which("gcloud")
if GCLOUD is None:
    raise SystemExit("gcloud not found on PATH")

DATASETS = {
    "ds_legacy_dataset1": "dataset1",
    "ds_legacy_dataset2": "dataset2",
    "ds_legacy_dataset3": "dataset3",
    "ds_legacy_dataset4": "dataset4",
}


def list_relative_paths(prefix: str) -> set[str]:
    """List all object paths under prefix, relative to prefix, excluding
    zero-length "folder marker" objects (name ends with '/')."""
    cmd = [GCLOUD, "--quiet", "--verbosity=error", "storage", "ls", "-r", prefix + "**"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and "matched no objects" not in proc.stderr:
        print(f"WARNING: listing failed for {prefix}: {proc.stderr.strip()}", file=sys.stderr)
    paths = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.endswith(":") or not line.startswith(prefix):
            continue
        rel = line[len(prefix):]
        if not rel or rel.endswith("/"):
            continue  # folder marker
        paths.add(rel)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Legacy dataset name to verify (e.g. dataset3). Repeatable. Default: all four.",
    )
    args = ap.parse_args()
    datasets = DATASETS
    if args.dataset:
        wanted = set(args.dataset)
        datasets = {k: v for k, v in DATASETS.items() if v in wanted}

    result = {
        "verified_at": date.today().isoformat(),
        "summary": {},
        "datasets": {},
    }
    complete = 0
    for dataset_id, legacy_name in datasets.items():
        source_prefix = f"gs://{BUCKET}/CapstoneData/raw-data/raw_session_extracts/{legacy_name}/"
        target_prefix = f"gs://{BUCKET}/CapstoneData/raw/sessions/{dataset_id}/"

        print(f"Listing {source_prefix} ...", file=sys.stderr)
        source_paths = list_relative_paths(source_prefix)
        print(f"Listing {target_prefix} ...", file=sys.stderr)
        target_paths = list_relative_paths(target_prefix)

        missing = sorted(source_paths - target_paths)
        extra = sorted(target_paths - source_paths)
        status = "complete" if not missing and not extra else "incomplete"
        if status == "complete":
            complete += 1

        result["datasets"][dataset_id] = {
            "legacy_dataset_name": legacy_name,
            "source_prefix": source_prefix,
            "target_prefix": target_prefix,
            "source_object_count": len(source_paths),
            "target_object_count": len(target_paths),
            "missing_path_count": len(missing),
            "extra_path_count": len(extra),
            "status": status,
            "missing_paths": missing,
            "extra_paths": extra,
        }
        print(
            f"  {dataset_id}: source={len(source_paths)} target={len(target_paths)} "
            f"missing={len(missing)} extra={len(extra)} -> {status}",
            file=sys.stderr,
        )

    result["summary"] = {
        "dataset_count": len(datasets),
        "complete_dataset_count": complete,
        "incomplete_dataset_count": len(datasets) - complete,
    }
    result["overall_status"] = "complete" if complete == len(datasets) else "incomplete"

    out_path = ROOT / "manifests" / f"raw_copy_verification_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Overall status: {result['overall_status']}")
    return 0 if result["overall_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
