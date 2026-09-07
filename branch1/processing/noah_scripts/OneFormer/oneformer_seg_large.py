import os
from pathlib import Path
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


def id_to_color(idx: int) -> np.ndarray:
    x = (idx + 1) * 2654435761
    b = (x & 0xFF)
    g = ((x >> 8) & 0xFF)
    r = ((x >> 16) & 0xFF)
    return np.array([b, g, r], dtype=np.uint8)  # BGR


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        out[label_map == cid] = id_to_color(int(cid))
    return out


def overlay_segmentation(frame_bgr: np.ndarray, seg_color_bgr: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, seg_color_bgr, alpha, 0)


def parse_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def parse_float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def find_color_mp4(session_dir: Path) -> Path:
    mp4 = next(session_dir.glob("*_color.mp4"), None)
    if not mp4:
        raise FileNotFoundError(f"No *_color.mp4 found in: {session_dir}")
    return mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session_dir", type=str, default=None, help="dataNoah/Processing/session_... directory")
    ap.add_argument("--video", type=str, default=None, help="Optional override: explicit path to *_color.mp4")
    ap.add_argument("--out", type=str, default=None, help="Optional override output mp4 path")
    args = ap.parse_args()

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
    else:
        if not args.session_dir:
            raise ValueError("Provide either --session_dir or --video")
        session_dir = Path(args.session_dir)
        video_path = find_color_mp4(session_dir)

    base = video_path.with_suffix("")
    out_path = Path(args.out) if args.out else Path(str(base) + "_seg_large.mp4")

    # ---- Settings ----
    here = Path(__file__).resolve().parent
    local_model = here / "oneformer_large"
    default_model = str(local_model) if local_model.is_dir() else "shi-labs/oneformer_ade20k_swin_large"

    model_name = os.environ.get("ONEFORMER_MODEL", default_model)
    force_local = os.environ.get("ONEFORMER_LOCAL", "0") == "1"
    max_w = parse_int_env("ONEFORMER_MAXW", 640)
    alpha = parse_float_env("ONEFORMER_ALPHA", 0.45)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | torch: {torch.__version__} | cuda?: {torch.cuda.is_available()}")
    print(f"Input : {video_path}")
    print(f"Output: {out_path}")
    print(f"Model : {model_name} (force_local={force_local})")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    processor = OneFormerProcessor.from_pretrained(model_name, local_files_only=force_local)

    # Auto-detect safetensors presence in local folder
    use_st = False
    if Path(model_name).is_dir():
        if (Path(model_name) / "model.safetensors").is_file():
            use_st = True
        else:
            for f in Path(model_name).iterdir():
                if f.name.startswith("model-") and f.name.endswith(".safetensors"):
                    use_st = True
                    break

    model = OneFormerForUniversalSegmentation.from_pretrained(
        model_name,
        local_files_only=force_local,
        use_safetensors=use_st,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video: {width}x{height} @ {fps:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {out_path}")

    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            h0, w0 = frame_bgr.shape[:2]

            if max_w and w0 > max_w:
                scale = max_w / float(w0)
                w1 = int(w0 * scale)
                h1 = int(h0 * scale)
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
            seg_color_bgr = colorize_label_map(seg_np)

            out_frame = overlay_segmentation(frame_bgr, seg_color_bgr, alpha=alpha)
            writer.write(out_frame)

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"Processed {frame_idx} frames.", flush=True)

    cap.release()
    writer.release()
    print(f"\nDone.\nInput : {video_path}\nOutput: {out_path}")


if __name__ == "__main__":
    main()