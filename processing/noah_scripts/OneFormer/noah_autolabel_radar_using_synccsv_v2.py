import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Segmentation metadata
# ----------------------------
def load_seg_meta(processing_session_dir: Path):
    meta_path = processing_session_dir / "seg_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing seg_meta.json: {meta_path} (run labelmap exporter first)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("id2label", {})


# ----------------------------
# Calibration loaders (match your JSON schema)
# ----------------------------
def load_intrinsics_from_meta(meta_json: Path) -> np.ndarray:
    d = json.loads(meta_json.read_text(encoding="utf-8"))
    K = np.array(d["realsense_calibration"]["rgb"]["K"], dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"RGB K must be 3x3; got {K.shape}")
    return K


def load_Tcr_from_extrinsics(extrinsics_json: Path) -> np.ndarray:
    d = json.loads(extrinsics_json.read_text(encoding="utf-8"))
    R = np.array(d["R_radar_to_camera"], dtype=np.float32).reshape(3, 3)
    t = np.array(d["t_radar_to_camera"], dtype=np.float32).reshape(3)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ----------------------------
# Projection + labeling helpers
# ----------------------------
def project_points(points_r: np.ndarray, K: np.ndarray, T_cr: np.ndarray):
    N = points_r.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pr_h = np.concatenate([points_r.astype(np.float32), ones], axis=1)  # Nx4

    pc_h = (T_cr @ pr_h.T).T
    pc = pc_h[:, :3]
    zc = pc[:, 2]

    uv = np.full((N, 2), np.nan, dtype=np.float32)
    valid = zc > 0.05
    if valid.any():
        x, y, z = pc[valid, 0], pc[valid, 1], pc[valid, 2]
        uv[valid, 0] = K[0, 0] * (x / z) + K[0, 2]
        uv[valid, 1] = K[1, 1] * (y / z) + K[1, 2]
    return uv, zc


def majority_label(label_map: np.ndarray, u: int, v: int, r: int = 1):
    H, W = label_map.shape
    u0, u1 = max(0, u - r), min(W, u + r + 1)
    v0, v1 = max(0, v - r), min(H, v + r + 1)
    patch = label_map[v0:v1, u0:u1].reshape(-1)
    vals, counts = np.unique(patch, return_counts=True)
    best = int(vals[np.argmax(counts)])
    frac = float(np.max(counts) / patch.size)
    return best, frac


# ----------------------------
# v2: ignore overhead / non-nav ADE labels BEFORE mapping to buckets
# ----------------------------
IGNORE_ADE_EXACT = {
    "ceiling",
    "light",
    "building",
    "window",
    "signboard, sign",
    "bulletin board",
}

IGNORE_ADE_CONTAINS = [
    "lamp",
    "lighting",
    "pipe",
    "duct",
]


def should_ignore_ade(ade_name: str) -> bool:
    if not ade_name:
        return False
    n = ade_name.lower().strip()
    if n in IGNORE_ADE_EXACT:
        return True
    for sub in IGNORE_ADE_CONTAINS:
        if sub in n:
            return True
    return False


def map_bucket(ade_name: str) -> str:
    n = (ade_name or "").lower()

    # high-priority classes
    if "person" in n:
        return "human"
    if "column" in n or "pillar" in n or "pole" in n:
        return "pillar"
    if "wall" in n:
        return "wall"
    if "floor" in n:
        return "floor"
    if "door" in n:
        return "door"

    # optional: keep stairs as its own class later, for now treat as obstacle
    if "stairway" in n or "staircase" in n or "escalator" in n:
        return "box_like"

    # catch-all obstacle
    return "box_like"


# ----------------------------
# Parsing helpers for your sync CSV (semicolon list)
# ----------------------------
def parse_semicolon_int_list(val):
    if pd.isna(val):
        return []
    s = str(val).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            pass
    return out


# ----------------------------
# Radar detections CSV column picking
# ----------------------------
def pick_radar_columns(df: pd.DataFrame):
    cols = set(df.columns)

    def first_present(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    frame = first_present(["radar_frame_num", "frame_num", "radar_frame", "frame"])
    x = first_present(["x", "x_m", "x_radar", "X"])
    y = first_present(["y", "y_m", "y_radar", "Y"])
    z = first_present(["z", "z_m", "z_radar", "Z"])

    if frame is None:
        raise ValueError(f"Could not find radar frame column in radar_csv. Columns={list(df.columns)}")
    if not (x and y and z):
        raise ValueError(f"Could not find radar xyz columns in radar_csv. Columns={list(df.columns)}")

    doppler = first_present(["doppler", "doppler_mps", "velocity", "vr"])
    snr = first_present(["snr", "SNR", "snr_db"])

    return {"frame": frame, "x": x, "y": y, "z": z, "doppler": doppler, "snr": snr}


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processing_session_dir", required=True)
    ap.add_argument("--sync_csv", required=True)
    ap.add_argument("--radar_csv", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--extrinsics_json", required=True)
    ap.add_argument("--neigh_r", type=int, default=1)
    ap.add_argument("--min_majority", type=float, default=0.60)
    ap.add_argument("--roi_top_frac", type=float, default=0.35, help="Ignore projected points above this fraction of image height")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    processing_dir = Path(args.processing_session_dir)
    seg_dir = processing_dir / "seg"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Missing seg/ in session: {seg_dir} (run labelmap exporter first)")

    id2label = load_seg_meta(processing_dir)
    K = load_intrinsics_from_meta(Path(args.meta_json))
    T_cr = load_Tcr_from_extrinsics(Path(args.extrinsics_json))

    sync = pd.read_csv(args.sync_csv)
    for c in ["video_frame_index", "radar_frame_nums"]:
        if c not in sync.columns:
            raise ValueError(f"sync_csv missing '{c}'. Found columns={list(sync.columns)}")

    radar = pd.read_csv(args.radar_csv)
    rcol = pick_radar_columns(radar)
    radar[rcol["frame"]] = radar[rcol["frame"]].astype(int)
    radar_by_frame = {k: v for k, v in radar.groupby(rcol["frame"])}

    out_rows = []

    for _, srow in tqdm(sync.iterrows(), total=len(sync), desc="Auto-label v2"):
        if "status" in sync.columns and str(srow.get("status")) != "matched":
            continue

        vfi = int(srow["video_frame_index"])
        seg_path = seg_dir / f"{vfi:06d}.npy"
        if not seg_path.exists():
            continue

        label_map = np.load(seg_path).astype(np.int32)
        H, W = label_map.shape
        roi_v_min = int(H * args.roi_top_frac)

        radar_frames = parse_semicolon_int_list(srow["radar_frame_nums"])
        if not radar_frames:
            continue

        for rf in radar_frames:
            grp = radar_by_frame.get(rf)
            if grp is None:
                continue

            pts = grp[[rcol["x"], rcol["y"], rcol["z"]]].to_numpy(dtype=np.float32)
            uv, zc = project_points(pts, K, T_cr)

            for i in range(len(pts)):
                u, v = uv[i]
                if not np.isfinite(u) or not np.isfinite(v):
                    continue
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < W and 0 <= vi < H):
                    continue

                # v2: ignore upper part of image (ceiling/overhead clutter)
                if vi < roi_v_min:
                    continue

                lid, maj = majority_label(label_map, ui, vi, r=args.neigh_r)
                if maj < args.min_majority:
                    continue

                ade_name = id2label.get(str(lid), str(lid))

                # v2: ignore overhead/non-nav ADE classes
                if should_ignore_ade(ade_name):
                    continue

                bucket = map_bucket(ade_name)

                rec = {
                    "video_frame_index": vfi,
                    "radar_frame_num": int(rf),
                    "u": int(ui),
                    "v": int(vi),
                    "ade_id": int(lid),
                    "ade_name": ade_name,
                    "bucket": bucket,
                    "maj_frac": float(maj),
                    "x": float(pts[i, 0]),
                    "y": float(pts[i, 1]),
                    "z": float(pts[i, 2]),
                    "cam_z": float(zc[i]),
                }

                if rcol["doppler"] is not None:
                    rec["doppler"] = float(grp.iloc[i][rcol["doppler"]])
                if rcol["snr"] is not None:
                    rec["snr"] = float(grp.iloc[i][rcol["snr"]])

                out_rows.append(rec)

    out_df = pd.DataFrame(out_rows)
    out_csv = args.out_csv or str(processing_dir / "labeled_radar_points_v2.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} rows={len(out_df)}")


if __name__ == "__main__":
    main()