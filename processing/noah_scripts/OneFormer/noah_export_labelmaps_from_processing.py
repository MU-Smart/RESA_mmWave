import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


def find_color_mp4(processing_session_dir: Path) -> Path:
    mp4 = next(processing_session_dir.glob("session_*_color.mp4"), None)
    if not mp4:
        mp4 = next(processing_session_dir.glob("*_color.mp4"), None)
    if not mp4:
        raise FileNotFoundError(f"No *_color.mp4 found in {processing_session_dir}")
    return mp4


def find_color_timestamps_csv(processing_session_dir: Path) -> Path:
    # Prefer the explicit color timestamps if present
    p = next(processing_session_dir.glob("session_*_color_timestamps.csv"), None)
    if p:
        return p
    # Fallback to your frame_timestamps file
    p = next(processing_session_dir.glob("frame_timestamps_session_*.csv"), None)
    if p:
        return p
    raise FileNotFoundError(
        f"No timestamp CSV found in {processing_session_dir}. Expected session_*_color_timestamps.csv or frame_timestamps_session_*.csv"
    )


def load_frame_indices(ts_csv: Path):
    """
    Supports:
      - session_*_color_timestamps.csv (frame_index,timestamp_ms)
      - frame_timestamps_session_*.csv (frame_index,timestamp_ms,filename)   [filename optional]
    Returns: frame_indices(list[int]), ts_map(dict[int->int])
    """
    import pandas as pd

    df = pd.read_csv(ts_csv)
    if "frame_index" not in df.columns:
        raise ValueError(f"{ts_csv} missing column: frame_index")

    # some files may have 'timestamp' vs 'timestamp_ms' — support both
    if "timestamp_ms" in df.columns:
        ts_col = "timestamp_ms"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        # not strictly required for segmentation export, but we store it if present
        ts_col = None

    frame_indices = df["frame_index"].astype(int).tolist()
    ts_map = {}
    if ts_col:
        ts_map = dict(zip(df["frame_index"].astype(int), df[ts_col].astype(int)))
    return frame_indices, ts_map, df.columns.tolist()


def id_to_color(idx: int) -> np.ndarray:
    x = (idx + 1) * 2654435761
    return np.array([(x & 0xFF), ((x >> 8) & 0xFF), ((x >> 16) & 0xFF)], dtype=np.uint8)  # BGR


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        out[label_map == cid] = id_to_color(int(cid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processing_session_dir", required=True, help=".../dataNoah/Processing/session_... folder")
    ap.add_argument("--model", choices=["tiny", "large"], default="tiny")
    ap.add_argument("--every_n", type=int, default=1, help="Process every Nth frame_index")
    ap.add_argument("--max_w", type=int, default=640, help="Downscale width for speed; output label map is resized back to original")
    ap.add_argument("--save_png", action="store_true", help="Save seg_vis/*.png for debugging")
    args = ap.parse_args()

    processing_dir = Path(args.processing_session_dir)
    color_mp4 = find_color_mp4(processing_dir)
    ts_csv = find_color_timestamps_csv(processing_dir)

    frame_indices, ts_map, cols = load_frame_indices(ts_csv)
    print(f"Video: {color_mp4.name}")
    print(f"Timestamps: {ts_csv.name} (cols={cols})")

    seg_dir = processing_dir / "seg"
    seg_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        (processing_dir / "seg_vis").mkdir(parents=True, exist_ok=True)

    # Use your local cache folders in repo/OneFormer if present; otherwise HF name
    model_name = (
        "shi-labs/oneformer_ade20k_swin_tiny"
        if args.model == "tiny"
        else "shi-labs/oneformer_ade20k_swin_large"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = OneFormerProcessor.from_pretrained(model_name)
    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_name,
        use_safetensors=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Write seg metadata
    meta = {
        "model_name": model_name,
        "id2label": {str(k): v for k, v in (getattr(model.config, "id2label", {}) or {}).items()},
        "every_n": args.every_n,
        "max_w": args.max_w,
        "color_video": color_mp4.name,
        "timestamps_csv": ts_csv.name,
    }
    (processing_dir / "seg_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cap = cv2.VideoCapture(str(color_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {color_mp4}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # We assume the mp4 frame order matches frame_index order (as in your recorder)
    # We'll read sequentially and use frame_index from the timestamps list.
    processed_rows = []

    with torch.no_grad():
        pbar = tqdm(total=min(len(frame_indices), total) if total > 0 else len(frame_indices), desc="Export label maps")
        for i, frame_index in enumerate(frame_indices):
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % args.every_n != 0:
                pbar.update(1)
                continue

            h0, w0 = frame_bgr.shape[:2]
            if args.max_w and w0 > args.max_w:
                scale = args.max_w / float(w0)
                w1, h1 = int(w0 * scale), int(h0 * scale)
                frame_small = cv2.resize(frame_bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame_bgr

            pil_img = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
            inputs = processor(images=pil_img, task_inputs=["semantic"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            outputs = model(**inputs)
            seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h0, w0)])[0]
            seg_np = seg.detach().cpu().numpy().astype(np.int32)

            np.save(seg_dir / f"{frame_index:06d}.npy", seg_np)

            if args.save_png:
                vis = colorize_label_map(seg_np)
                cv2.imwrite(str(processing_dir / "seg_vis" / f"{frame_index:06d}.png"), vis)

            processed_rows.append({
                "frame_index": int(frame_index),
                "timestamp_ms": int(ts_map.get(frame_index, -1))
            })

            pbar.update(1)

        pbar.close()
        cap.release()

    # Save which frames got segmentation (for later matching)
    import pandas as pd
    pd.DataFrame(processed_rows).to_csv(processing_dir / "seg_timestamps.csv", index=False)
    print(f"Done. seg/ has {len(processed_rows)} frames. Wrote seg_timestamps.csv")


if __name__ == "__main__":
    main()