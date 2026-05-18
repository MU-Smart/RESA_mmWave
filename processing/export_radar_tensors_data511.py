#!/usr/bin/env python3
"""
Generate {session}_radar_tensors.npz for any Branch 1 processing session root so
dataset_tools.py build-patches can extract true beamformed RA patches.
By default this only processes sessions that already have non-empty labels,
then strips temporary rd_cube arrays back out of hybrid_rd sidecars.

Run from repo root:
  source .venv/bin/activate
  python LLM_ML/jetson_nav_pipeline/canon/processing/export_radar_tensors_data511.py \
    --data-root LLM_ML/data/data5-11 --session-prefix session_2026-05-11_
  python LLM_ML/jetson_nav_pipeline/canon/processing/export_radar_tensors_data511.py \
    --data-root LLM_ML/data/data-4-30/Processing --session-prefix session_2026-04-28_
"""
import sys
import json
import argparse
import os
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
CANON_ROOT = THIS_FILE.parents[1]
IO_ROOT = THIS_FILE.parents[4] if len(THIS_FILE.parents) > 4 else CANON_ROOT
for path in (CANON_ROOT, IO_ROOT, IO_ROOT / "LLM_ML" / "jetson_nav_pipeline" / "canon"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from models.hybrid_rd_runtime import export_session_sidecars, RuntimeRDPatchConfig

CFG_PATH = Path("LLM_ML/jetson_nav_pipeline/canon/config/profile_objdet.cfg")
DATA_ROOT = Path(os.environ.get("RADAR_TENSOR_PROCESSING_ROOT", "LLM_ML/data/data5-11"))
SESSION_PREFIX = os.environ.get("RADAR_TENSOR_SESSION_PREFIX", "session_2026-05-11_")
FRAME_NUMBER_OFFSET = 0


def _has_data_rows(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh) > 1
    except OSError:
        return False


def _get_rd_cube_from_sidecar(sidecar_path: Path) -> np.ndarray:
    with np.load(sidecar_path) as d:
        if "rd_cube" not in d:
            raise KeyError(f"rd_cube not in {sidecar_path}")
        return np.asarray(d["rd_cube"], dtype=np.complex64)


def build_radar_tensors_from_sidecars(session_dir: Path) -> None:
    """Write {session}_radar_tensors.npz by reading rd_cube from sidecar frames."""
    frames_dir = session_dir / "hybrid_rd" / "frames"
    manifest_path = session_dir / "hybrid_rd" / "manifest.json"

    if not manifest_path.exists():
        print(f"  [SKIP] No manifest.json in {session_dir.name}")
        return

    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("include_rd_cube", False):
        print(f"  [SKIP] Sidecar manifest has include_rd_cube=False — re-export sidecars first")
        return

    out_path = session_dir / f"{session_dir.name}_radar_tensors.npz"
    if out_path.exists():
        print(f"  [EXISTS] {out_path.name}")
        return

    tensors = {}
    for frame_info in manifest["frames"]:
        frame_index = int(frame_info["frame_index"])
        frame_path = session_dir / "hybrid_rd" / frame_info["path"]
        try:
            cube = _get_rd_cube_from_sidecar(frame_path)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [WARN] frame {frame_index}: {e}")
            continue
        tensors[f"rd_{frame_index}"] = cube

    if not tensors:
        print(f"  [SKIP] No tensors extracted for {session_dir.name}")
        return

    np.savez_compressed(out_path, **tensors)
    print(f"  [OK] wrote {out_path.name} ({len(tensors)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate <session>_radar_tensors.npz for Branch 1 sessions from local "
            "hybrid_rd sidecars. By default, only sessions with non-empty labels are processed."
        )
    )
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--session-prefix", default=SESSION_PREFIX)
    ap.add_argument("--cfg-path", type=Path, default=CFG_PATH)
    ap.add_argument("--label-csv-name", default="labeled_radar_points_v4.csv")
    ap.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also process sessions without a non-empty label CSV.",
    )
    ap.add_argument(
        "--keep-rd-cube-sidecars",
        action="store_true",
        help="Keep rd_cube arrays in hybrid_rd frame sidecars after building the tensor NPZ.",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    sessions = sorted(data_root.glob(f"{args.session_prefix}*"))
    if not sessions:
        raise SystemExit(f"No sessions found at {data_root} with prefix {args.session_prefix}")

    if not args.include_unlabeled:
        sessions = [
            s for s in sessions
            if _has_data_rows(s / args.label_csv_name)
        ]
        if not sessions:
            raise SystemExit(
                f"No labeled sessions found at {data_root} with prefix {args.session_prefix}"
            )

    print(f"Found {len(sessions)} sessions at {data_root} with prefix {args.session_prefix}\n")

    rd_config = RuntimeRDPatchConfig(
        frame_number_offset=FRAME_NUMBER_OFFSET,
        sidecar_subdir="hybrid_rd",
        clutter_mode="off",
        log_scale=True,
        include_rd_cube=True,
    )

    for session_dir in sessions:
        print(f"Processing {session_dir.name}")
        bin_path = session_dir / f"{session_dir.name}.bin"
        if not bin_path.exists():
            print(f"  [SKIP] No .bin file found")
            continue

        # Step 1: re-export sidecar frames with rd_cube
        print(f"  Exporting sidecars with rd_cube ...")
        try:
            export_session_sidecars(
                session_dir,
                cfg_path=args.cfg_path,
                config=rd_config,
                overwrite=True,
            )
            print(f"  [OK] sidecars written with rd_cube=True")
        except Exception as e:
            print(f"  [ERROR] sidecar export: {e}")
            continue

        # Step 2: build _radar_tensors.npz from the sidecar rd_cube keys
        build_radar_tensors_from_sidecars(session_dir)

        if not args.keep_rd_cube_sidecars:
            print(f"  Stripping rd_cube from sidecars ...")
            try:
                export_session_sidecars(
                    session_dir,
                    cfg_path=args.cfg_path,
                    config=RuntimeRDPatchConfig(
                        frame_number_offset=FRAME_NUMBER_OFFSET,
                        sidecar_subdir="hybrid_rd",
                        clutter_mode="off",
                        log_scale=True,
                        include_rd_cube=False,
                    ),
                    overwrite=True,
                )
                print(f"  [OK] sidecars restored with rd_cube=False")
            except Exception as e:
                print(f"  [WARN] sidecar rd_cube strip failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
