"""
visualize_radar_on_video_with_seg.py

Visualization tool for the auto-labeling pipeline.

Reads the labeled_radar_points_v3.csv (output of Stage 2) and renders an
annotated video that overlays:
  - The ADE20K segmentation colormap (from seg/*.npy)
  - Each labeled radar point as a colored dot (color = bucket class)
  - A per-frame legend showing bucket counts
  - Frame index HUD

This lets you visually verify that radar points are landing on the correct
semantic regions before training a model on the labels.

Usage:
    python visualize_radar_on_video_with_seg.py
        --processing_session_dir "C:\\...\\Processing\\session_..."
        [--alpha_seg 0.35]
        [--point_radius 4]
        [--max_points_per_frame 3000]
        [--use_seg_vis]
        [--out_mp4 annotated_radar_seg.mp4]

Bucket color legend (BGR):
    human    → red      (0, 0, 255)
    pillar   → magenta  (255, 0, 255)
    door     → yellow   (0, 255, 255)
    wall     → cyan     (255, 255, 0)
    floor    → green    (0, 255, 0)
    box_like → blue     (255, 0, 0)

Dependencies:
    pip install numpy pandas opencv-python
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# Color definitions
# =============================================================================

BUCKET_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "human":    (0,   0,   255),   # red
    "pillar":   (255, 0,   255),   # magenta
    "door":     (0,   255, 255),   # yellow
    "wall":     (255, 255, 0),     # cyan
    "floor":    (0,   255, 0),     # green
    "box_like": (255, 0,   0),     # blue
    "obstacle": (255, 0,   0),     # alias for box_like
}

UNKNOWN_COLOR = (180, 180, 180)    # grey for unmapped buckets


def id_to_color(idx: int) -> np.ndarray:
    """Deterministic pseudo-random BGR color for ADE label id."""
    x = (idx + 1) * 2654435761
    return np.array([(x & 0xFF), ((x >> 8) & 0xFF), ((x >> 16) & 0xFF)], dtype=np.uint8)


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert int32 label map → BGR color image."""
    out = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cid in np.unique(label_map):
        out[label_map == cid] = id_to_color(int(cid))
    return out


# =============================================================================
# File discovery
# =============================================================================

def find_color_mp4(session_dir: Path) -> Path:
    mp4 = next(session_dir.glob("session_*_color.mp4"), None)
    if not mp4:
        mp4 = next(session_dir.glob("*_color.mp4"), None)
    if not mp4:
        raise FileNotFoundError(f"No *_color.mp4 found in {session_dir}")
    return mp4


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.65,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draw text with a dark drop-shadow for legibility on any background."""
    x, y = origin
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_legend(
    img: np.ndarray,
    counts: dict[str, int],
    frame_idx: int,
) -> None:
    """Render frame index and bucket point counts in the top-left corner."""
    y = 28
    draw_text_with_shadow(img, f"frame {frame_idx:06d}", (10, y), scale=0.7)
    y += 26

    total = sum(counts.values())
    if total > 0:
        draw_text_with_shadow(img, f"pts: {total}", (10, y), scale=0.6)
        y += 22

    for bucket, count in sorted(counts.items(), key=lambda kv: -kv[1])[:7]:
        color = BUCKET_COLORS_BGR.get(bucket, UNKNOWN_COLOR)
        draw_text_with_shadow(
            img,
            f"{bucket}: {count}",
            (10, y),
            scale=0.58,
            color=color,
        )
        y += 21


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Overlay labeled radar points + segmentation on color video."
    )
    ap.add_argument("--processing_session_dir", required=True,
                    help="Processing session directory (contains seg/, color video, labeled CSV)")
    ap.add_argument("--out_mp4", default=None,
                    help="Output video path. Default: <session_dir>/annotated_radar_seg.mp4")
    ap.add_argument("--alpha_seg", type=float, default=0.35,
                    help="Segmentation overlay transparency (0=none, 1=opaque). Default: 0.35")
    ap.add_argument("--point_radius", type=int, default=4,
                    help="Radar point circle radius in pixels. Default: 4")
    ap.add_argument("--max_points_per_frame", type=int, default=3000,
                    help="Cap on radar points drawn per frame (for speed). Default: 3000")
    ap.add_argument("--use_seg_vis", action="store_true",
                    help="Load segmentation from seg_vis/*.png instead of seg/*.npy "
                         "(faster but less accurate color).")
    args = ap.parse_args()

    session_dir = Path(args.processing_session_dir).resolve()
    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session_dir}")

    video_path = find_color_mp4(session_dir)
    labeled_csv = session_dir / "labeled_radar_points_v3.csv"
    if not labeled_csv.exists():
        raise FileNotFoundError(f"Missing labeled_radar_points_v3.csv in {session_dir}")

    seg_dir    = session_dir / "seg"
    seg_vis_dir = session_dir / "seg_vis"

    # Validate seg availability
    use_npy = seg_dir.exists() and any(seg_dir.glob("*.npy"))
    use_png = args.use_seg_vis and seg_vis_dir.exists() and any(seg_vis_dir.glob("*.png"))
    if not use_npy and not use_png:
        print(f"[WARN] No segmentation found in seg/ or seg_vis/ — "
              f"rendering radar overlay only (no segmentation background).")

    out_mp4 = Path(args.out_mp4) if args.out_mp4 else (session_dir / "annotated_radar_seg.mp4")

    print("=" * 70)
    print("visualize_radar_on_video_with_seg.py")
    print("=" * 70)
    print(f"  Session  : {session_dir.name}")
    print(f"  Video    : {video_path.name}")
    print(f"  Labels   : {labeled_csv}")
    print(f"  Output   : {out_mp4}")
    print(f"  alpha_seg: {args.alpha_seg}  |  radius: {args.point_radius}  |  "
          f"max_pts/frame: {args.max_points_per_frame}")

    # ---- Load labeled CSV ----
    df = pd.read_csv(labeled_csv)
    required = {"video_frame_index", "u", "v", "bucket"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"labeled_csv is missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    df["video_frame_index"] = df["video_frame_index"].astype(int)
    df["u"]      = df["u"].astype(int)
    df["v"]      = df["v"].astype(int)
    df["bucket"] = df["bucket"].astype(str)

    # Group by frame for O(1) lookup
    grouped: dict[int, pd.DataFrame] = {
        k: g for k, g in df.groupby("video_frame_index")
    }
    print(f"  Loaded {len(df)} labeled points across {len(grouped)} frames.")

    # ---- Open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"  Video    : {W}x{H} @ {fps:.2f} fps  ({total_frames} frames)")

    # ---- Open writer ----
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {out_mp4}")

    frame_idx = 0
    frames_with_radar = 0
    total_pts_drawn   = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- Segmentation overlay ----
        if args.alpha_seg > 0:
            seg_bgr = None

            if use_png:
                p = seg_vis_dir / f"{frame_idx:06d}.png"
                if p.exists():
                    loaded = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    if loaded is not None and loaded.shape[:2] == (H, W):
                        seg_bgr = loaded
            elif use_npy:
                p = seg_dir / f"{frame_idx:06d}.npy"
                if p.exists():
                    label_map = np.load(p).astype(np.int32)
                    seg_bgr   = colorize_label_map(label_map)

            if seg_bgr is not None:
                frame = cv2.addWeighted(
                    frame, 1.0 - args.alpha_seg,
                    seg_bgr, args.alpha_seg,
                    0,
                )

        # ---- Radar point overlay ----
        counts: dict[str, int] = {}
        g = grouped.get(frame_idx)

        if g is not None and len(g) > 0:
            if len(g) > args.max_points_per_frame:
                g = g.sample(args.max_points_per_frame, random_state=0)

            for _, row in g.iterrows():
                u, v = int(row["u"]), int(row["v"])
                if not (0 <= u < W and 0 <= v < H):
                    continue

                bucket = str(row["bucket"])
                counts[bucket] = counts.get(bucket, 0) + 1
                color = BUCKET_COLORS_BGR.get(bucket, UNKNOWN_COLOR)
                cv2.circle(frame, (u, v), args.point_radius, color, -1)
                total_pts_drawn += 1

            if counts:
                frames_with_radar += 1

        # ---- HUD legend ----
        draw_legend(frame, counts, frame_idx)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{total_frames} frames...", end="\r", flush=True)

    cap.release()
    writer.release()

    print(f"\n[DONE] {frame_idx} frames processed.")
    print(f"       {frames_with_radar} frames had radar points.")
    print(f"       {total_pts_drawn} total points drawn.")
    print(f"  Output: {out_mp4}")


if __name__ == "__main__":
    main()
