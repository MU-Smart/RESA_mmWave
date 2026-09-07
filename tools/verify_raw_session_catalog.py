#!/usr/bin/env python3

import csv
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path("/home/hullumdr/resa/RESA_mmWave")
CATALOG_PATH = ROOT / "manifests" / "session_catalog.csv"
SESSION_ID_RE = re.compile(r"^session_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")


def main() -> int:
    with CATALOG_PATH.open(newline="") as catalog_file:
        rows = list(csv.DictReader(catalog_file))

    errors = []
    seen = set()
    by_legacy_group = Counter()

    for idx, row in enumerate(rows, start=2):
        session_id = row["session_id"]
        dataset_id = row["dataset_id"]
        legacy_dataset_name = row["legacy_dataset_name"]
        raw_path = row["raw_path"]

        if not SESSION_ID_RE.match(session_id):
            errors.append(f"line {idx}: invalid session_id {session_id}")

        expected_suffix = f"/{session_id}/"
        if not raw_path.endswith(expected_suffix):
            errors.append(
                f"line {idx}: raw_path does not end with {expected_suffix}: {raw_path}"
            )

        # Timestamp-based session names are reused by replicated legacy
        # extracts, so the canonical identity is (dataset_id, session_id).
        key = (dataset_id, session_id)
        if key in seen:
            errors.append(f"line {idx}: duplicate catalog identity {key}")
        seen.add(key)
        by_legacy_group[legacy_dataset_name] += 1

        if not dataset_id:
            errors.append(f"line {idx}: dataset_id is empty")

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"rows={len(rows)}")
    for legacy_dataset_name in sorted(by_legacy_group):
        print(f"{legacy_dataset_name}={by_legacy_group[legacy_dataset_name]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
