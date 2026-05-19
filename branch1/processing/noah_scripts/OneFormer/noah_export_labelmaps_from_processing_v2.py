"""
noah_export_labelmaps_from_processing.py

Stage 1 of the auto-labeling pipeline.

For a given processing session directory, runs OneFormer semantic segmentation
on every (or every Nth) color video frame and saves:
  - seg/<frame_index:06d>.npy   — int32 ADE20K label map (H x W)
  - seg_vis/<frame_index:06d>.png  — optional BGR colorized visualization
  - seg_meta.json               — model name, id2label mapping, run config
  - seg_timestamps.csv          — frame_index, timestamp_ms for processed frames

Usage (large model, every frame):
    python noah_export_labelmaps_from_processing.py
        --processing_session_dir "C:\\...\\Processing\\session_2026-02-26_18-20-44"
        --model large
        --every_n 1

Usage (tiny model, every 3rd frame, save PNG visualizations):
    python noah_export_labelmaps_from_processing.py
        --processing_session_dir "C:\\...\\Processing\\session_2026-02-26_18-20-44"
        --model tiny
        --every_n 3
        --save_png

Notes:
  - Looks for OneFormer model weights in:
      <script_dir>/oneformer_large/   or   <script_dir>/oneformer_tiny/
    Falls back to HuggingFace Hub if local folder not found.
  - Requires: pip install torch transformers Pillow opencv-python pandas tqdm
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


# =============================================================================
# File discovery helpers
# =============================================================================

def find_color_mp4(processing_session_dir: Path) -> Path:
    """Find the color video in the session directory."""
    mp4 = next(processing_session_dir.glob("session_*_color.mp4"), None)
    if not mp4:
        mp4 = next(processing_session_dir.glob("*_color.mp4"), None)
    if not mp4:
        raise FileNotFoundError(f"No *_color.mp4 found in {processing_session_dir}")
    return mp4


def find_color_timestamps_csv(processing_session_dir: Path) -> Path:
    """
    Find the color-frame hardware timestamp CSV.
    Checks for session_*_color_timestamps.csv first, then any frame_timestamps*.csv.
    """
    p = next(processing_session_dir.glob("session_*_color_timestamps.csv"), None)
    if p:
        return p
    p = next(processing_session_dir.glob("frame_timestamps_session_*.csv"), None)
    if p:
        return p
    raise FileNotFoundError(
        f"No color timestamp CSV found in {processing_session_dir}.\n"
        f"Expected: session_*_color_timestamps.csv  or  frame_timestamps_session_*.csv"
    )


def load_frame_indices(ts_csv: Path):
    """
    Read the color timestamp CSV and return:
      frame_indices : list[int]   — sequential frame indices
      ts_map        : dict[int→int]  — frame_index → timestamp_ms
      columns       : list[str]

    Supports:
      session_*_color_timestamps.csv  columns: frame_index, timestamp_ms
      frame_timestamps_session_*.csv  columns: frame_index, timestamp_ms[, filename]
    """
    import pandas as pd

    df = pd.read_csv(ts_csv)

    # The recorder writes 'frame_index'; some older exports used just an index column.
    if "frame_index" not in df.columns:
        # Attempt to create frame_index from row position
        print(f"  [WARN] '{ts_csv.name}' has no 'frame_index' column — using row position.")
        df["frame_index"] = range(len(df))

    ts_col = None
    for candidate in ("timestamp_ms", "timestamp"):
        if candidate in df.columns:
            ts_col = candidate
            break

    frame_indices = df["frame_index"].astype(int).tolist()
    ts_map: dict = {}
    if ts_col:
        ts_map = dict(zip(df["frame_index"].astype(int), df[ts_col].astype(int)))

    return frame_indices, ts_map, df.columns.tolist()


# =============================================================================
# Visualization helpers
# =============================================================================

def id_to_color(idx: int) -> np.ndarray:
    """Deterministic pseudo-random BGR color for ADE label index."""
    x = (idx + 1) * 2654435761
    return np.array([(x & 0xFF), ((x >> 8) & 0xFF), ((x >> 16) & 0xFF)], dtype=np.uint8)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert int32 label map → BGR color image."""
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        out[label_map == cid] = id_to_color(int(cid))
    return out


def is_lfs_pointer(path: Path) -> bool:
    """Return True if the file looks like a Git LFS pointer stub."""
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    return bool(head) and head[0].startswith("version https://git-lfs.github.com/spec/v1")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Export per-frame ADE20K semantic label maps from a processing session."
    )
    ap.add_argument(
        "--processing_session_dir", required=True,
        help="Path to the session folder inside dataNoah/Processing/"
    )
    ap.add_argument(
        "--model", choices=["tiny", "large"], default="tiny",
        help="OneFormer backbone: 'tiny' (faster) or 'large' (more accurate). Default: tiny"
    )
    ap.add_argument(
        "--every_n", type=int, default=1,
        help="Process every Nth frame index. Default: 1 (all frames)"
    )
    ap.add_argument(
        "--max_w", type=int, default=640,
        help="Downscale width for inference speed. Label map is upscaled back to original "
             "resolution. Default: 640. Set 0 to disable."
    )
    ap.add_argument(
        "--save_png", action="store_true",
        help="Also save colorized segmentation PNGs to seg_vis/ for visual inspection."
    )
    args = ap.parse_args()

    processing_dir = Path(args.processing_session_dir).resolve()
    if not processing_dir.is_dir():
        raise NotADirectoryError(f"Processing session dir not found: {processing_dir}")

    print("=" * 70)
    print("noah_export_labelmaps_from_processing.py")
    print("=" * 70)

    # ---- Locate inputs ----
    color_mp4 = find_color_mp4(processing_dir)
    ts_csv    = find_color_timestamps_csv(processing_dir)
    frame_indices, ts_map, cols = load_frame_indices(ts_csv)

    print(f"  Session  : {processing_dir.name}")
    print(f"  Video    : {color_mp4.name}")
    print(f"  Timestamps: {ts_csv.name}  ({len(frame_indices)} entries, cols={cols})")

    # ---- Output dirs ----
    seg_dir = processing_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        (processing_dir / "seg_vis").mkdir(parents=True, exist_ok=True)

    # ---- Resolve model ----
    here = Path(__file__).resolve().parent
    local_name = "oneformer_tiny" if args.model == "tiny" else "oneformer_large"
    local_path = here / local_name
    hf_name    = (
        "shi-labs/oneformer_ade20k_swin_tiny"
        if args.model == "tiny"
        else "shi-labs/oneformer_ade20k_swin_large"
    )
    local_model_bin = local_path / "pytorch_model.bin"
    local_model_uses_lfs_stub = is_lfs_pointer(local_model_bin)
    use_local = local_path.is_dir() and not local_model_uses_lfs_stub
    model_name = str(local_path) if use_local else hf_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model    : {model_name}  [{args.model}]")
    print(f"  Device   : {device}")
    print(f"  every_n  : {args.every_n}")
    print(f"  max_w    : {args.max_w or 'disabled'}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ---- Load model ----
    if local_model_uses_lfs_stub:
        print("  [NOTE] Local model weights are Git LFS stubs; falling back to Hugging Face Hub.")

    processor = OneFormerProcessor.from_pretrained(model_name, local_files_only=use_local)

    # Detect safetensors in local folder
    use_st = True
    if local_path.is_dir():
        st_files = list(local_path.glob("*.safetensors")) + list(local_path.glob("model-*.safetensors"))
        use_st = bool(st_files)

    # Repo-local OneFormer checkpoints predate torch.load(weights_only=True).
    # They are trusted project assets, so allow the legacy full-state load path.
    weights_only = not use_local

    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_name,
        local_files_only=use_local,
        use_safetensors=use_st,
        weights_only=weights_only,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    print(f"  Model loaded.")

    # ---- Write seg_meta.json ----
    id2label = {
        str(k): v
        for k, v in (getattr(model.config, "id2label", {}) or {}).items()
    }
    meta = {
        "model_name":     model_name,
        "model_size":     args.model,
        "id2label":       id2label,
        "every_n":        args.every_n,
        "max_w":          args.max_w,
        "color_video":    color_mp4.name,
        "timestamps_csv": ts_csv.name,
    }
    (processing_dir / "seg_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"  seg_meta.json written ({len(id2label)} ADE classes).")

    # ---- Open video ----
    cap = cv2.VideoCapture(str(color_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {color_mp4}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    processed_rows = []
    n_skipped = 0

    with torch.no_grad():
        pbar = tqdm(
            total=min(len(frame_indices), total_frames) if total_frames > 0 else len(frame_indices),
            desc="Export label maps",
        )

        for i, frame_index in enumerate(frame_indices):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Skip if this frame index not in the every_n pattern
            if frame_index % args.every_n != 0:
                n_skipped += 1
                pbar.update(1)
                continue

            # Skip frames already exported (allows resuming interrupted runs)
            out_npy = seg_dir / f"{frame_index:06d}.npy"
            if out_npy.exists():
                processed_rows.append({
                    "frame_index":  int(frame_index),
                    "timestamp_ms": int(ts_map.get(frame_index, -1)),
                })
                pbar.update(1)
                continue

            h0, w0 = frame_bgr.shape[:2]

            # Optionally downscale for faster inference
            if args.max_w and w0 > args.max_w:
                scale      = args.max_w / float(w0)
                w1, h1     = int(w0 * scale), int(h0 * scale)
                frame_small = cv2.resize(frame_bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame_bgr

            pil_img = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
            inputs  = processor(images=pil_img, task_inputs=["semantic"], return_tensors="pt")
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            if device == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            outputs = model(**inputs)

            # Post-process back to original resolution
            seg = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(h0, w0)]
            )[0]
            seg_np = seg.detach().cpu().numpy().astype(np.int32)

            np.save(out_npy, seg_np)

            if args.save_png:
                vis = colorize_label_map(seg_np)
                cv2.imwrite(
                    str(processing_dir / "seg_vis" / f"{frame_index:06d}.png"), vis
                )

            processed_rows.append({
                "frame_index":  int(frame_index),
                "timestamp_ms": int(ts_map.get(frame_index, -1)),
            })
            pbar.update(1)

        pbar.close()

    cap.release()

    # ---- Save seg_timestamps.csv ----
    import pandas as pd
    pd.DataFrame(processed_rows).to_csv(
        processing_dir / "seg_timestamps.csv", index=False
    )

    n_done = len(processed_rows)
    print(f"\n[DONE] {n_done} frames exported → {seg_dir}")
    print(f"       {n_skipped} frames skipped (every_n={args.every_n})")
    print(f"       seg_meta.json and seg_timestamps.csv written.")
    print(f"\nNext step:")
    print(f"  python noah_autolabel_radar_using_synccsv_v3.py")
    print(f"    --processing_session_dir \"{processing_dir}\"")
    print(f"    --sync_csv <path_to_synchronized_<stem>.csv>")
    print(f"    --radar_csv <path_to_<stem>.csv>")
    print(f"    --meta_json \"{processing_dir / 'meta_data.json'}\"")
    print(f"    --extrinsics_json <path_to_radar_camera_extrinsics.json>")


if __name__ == "__main__":
    main()
