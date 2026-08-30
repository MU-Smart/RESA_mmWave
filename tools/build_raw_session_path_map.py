#!/usr/bin/env python3

import csv
from pathlib import Path


ROOT = Path("/home/hullumdr/resa/RESA_mmWave")
CATALOG_PATH = ROOT / "manifests" / "session_catalog.csv"
OUTPUT_PATH = ROOT / "manifests" / "raw_session_path_map.csv"


def build_target_path(dataset_id: str, session_id: str) -> str:
    return f"gs://miamioh-resa-data/CapstoneData/raw/sessions/{dataset_id}/{session_id}/"


def main() -> None:
    with CATALOG_PATH.open(newline="") as catalog_file:
        rows = list(csv.DictReader(catalog_file))

    output_fields = [
        "session_id",
        "dataset_id",
        "legacy_dataset_name",
        "current_raw_path",
        "target_raw_path",
        "mapping_scope",
        "migration_status",
        "notes",
    ]

    with OUTPUT_PATH.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=output_fields)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "session_id": row["session_id"],
                    "dataset_id": row["dataset_id"],
                    "legacy_dataset_name": row["legacy_dataset_name"],
                    "current_raw_path": row["raw_path"],
                    "target_raw_path": build_target_path(
                        row["dataset_id"], row["session_id"]
                    ),
                    "mapping_scope": "session_root",
                    "migration_status": "planned",
                    "notes": (
                        "Exact 1-to-1 raw-session mapping generated from "
                        "session_catalog.csv. Source tree remains authoritative "
                        "until target copy is verified."
                    ),
                }
            )


if __name__ == "__main__":
    main()
