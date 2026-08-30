#!/usr/bin/env python3
"""
Generate the full list of stray/extra objects sitting in the canonical raw
target prefixes that do not exist in the corresponding legacy source tree
(leftovers from the earlier bad copy attempt), plus a ready-to-run cleanup
script.

This script only reads the latest manifests/raw_copy_verification_*.json
and writes text/shell files. It never deletes anything itself -- object
deletion in GCS is something to run deliberately and by hand, not something
this script (or the assistant that wrote it) performs automatically.

Usage:
    python3 tools/gen_extra_cleanup_script.py
    # then review manifests/extra_cleanup/*.txt and cleanup_extras.sh
    # and run cleanup_extras.sh yourself when ready.
"""

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "manifests" / "extra_cleanup"


def latest_report() -> Path:
    candidates = sorted(glob.glob(str(ROOT / "manifests" / "raw_copy_verification_*.json")))
    if not candidates:
        raise SystemExit("No manifests/raw_copy_verification_*.json found. Run verify_raw_copy.py first.")
    return Path(candidates[-1])


def main() -> int:
    report_path = latest_report()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"Using {report_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_list_paths = []
    total_extra = 0

    for dataset_id, info in report["datasets"].items():
        extra_paths = info.get("extra_paths")
        if extra_paths is None:
            print(f"  {dataset_id}: report has no full extra_paths list (old-format report?) - skipping")
            continue
        if not extra_paths:
            continue
        target_prefix = info["target_prefix"]
        list_path = OUT_DIR / f"{dataset_id}_extra_objects.txt"
        with list_path.open("w", encoding="utf-8") as fh:
            for rel in extra_paths:
                fh.write(target_prefix + rel + "\n")
        print(f"  {dataset_id}: wrote {len(extra_paths)} extra object path(s) -> {list_path}")
        all_list_paths.append(list_path)
        total_extra += len(extra_paths)

    if not all_list_paths:
        print("No extra/stray objects found. Nothing to clean up.")
        return 0

    script_path = ROOT / "manifests" / "extra_cleanup" / "cleanup_extras.sh"
    with script_path.open("w", encoding="utf-8") as fh:
        fh.write("#!/usr/bin/env bash\n")
        fh.write("# Deletes the stray/extra objects listed below from GCS.\n")
        fh.write("# Review the .txt file(s) in this directory before running this.\n")
        fh.write("# This is NOT run automatically by anything -- run it yourself when ready.\n")
        fh.write("set -euo pipefail\n\n")
        fh.write('cd "$(dirname "${BASH_SOURCE[0]}")"\n\n')
        for list_path in all_list_paths:
            fh.write(f'echo "Deleting objects listed in {list_path.name} ..."\n')
            fh.write(f'xargs -a "{list_path.name}" -n 100 gcloud --quiet storage rm --\n\n')
    print(f"\nWrote {script_path}")
    print(f"Total extra objects across all datasets: {total_extra}")
    print("\nReview the .txt file(s), then run cleanup_extras.sh yourself when ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
