"""Depth correspondence evidence for radar points.

This module turns the existing depth-gated autolabel idea into reusable soft
evidence.  It is intended for offline teacher-label generation first: missing
camera/depth evidence should become "unknown", not "ghost".
"""

from __future__ import annotations

import json
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch1.calib.calibration import load_calibration, project_radar_to_image


DEPTH_NUMERIC_COLS = [
    "depth_corr_available",
    "depth_corr_in_fov",
    "depth_corr_valid_px",
    "depth_corr_patch_px",
    "depth_corr_radar_z_m",
    "depth_corr_min_m",
    "depth_corr_median_m",
    "depth_corr_selected_m",
    "depth_corr_residual_m",
    "depth_corr_abs_residual_m",
    "depth_corr_sigma_m",
    "depth_corr_quality",
    "p_depth_match",
    "depth_foreground_occluded",
    "depth_candidate_cluster_id",
    "depth_candidate_cluster_size",
    "depth_candidate_cluster_radius_m",
    "depth_candidate_cluster_min_depth_match",
    "depth_candidate_cluster_mean_p_dir_doppler",
]


@dataclass(frozen=True)
class DepthCorrespondenceConfig:
    depth_patch_r: int = 2
    semantic_patch_r: int = 1
    use_distortion: bool = False
    max_time_diff_ms: float | None = 35.0
    foreground_occlusion_margin_m: float = 0.20
    sigma_min_m: float = 0.08
    sigma_frac: float = 0.04
    candidate_residual_m: float = 0.45
    candidate_p_depth_match_max: float = 0.20
    candidate_dbscan_eps_m: float = 0.54
    candidate_min_samples: int = 2
    unknown_p_depth_match: float = 1.0
    projection_valid_rate_min: float = 0.40
    depth_available_rate_min: float = 0.60
    median_residual_abs_max_m: float = 0.20
    residual_mad_max_m: float = 0.35
    time_match_p95_max_ms: float = 35.0


def load_depth_scale(meta_json: Path) -> float:
    """Read RealSense depth scale from a session meta file."""
    try:
        data = json.loads(meta_json.read_text(encoding="utf-8"))
        return float(
            data.get("realsense_calibration", {}).get(
                "depth_scale_m_per_unit",
                0.001,
            )
        )
    except Exception:
        return 0.001


class DepthFrameCache:
    """Small LRU cache for `depth/<frame_index>.npy` arrays in metres."""

    def __init__(self, depth_dir: Path, depth_scale_m_per_unit: float, capacity: int = 8):
        self.depth_dir = Path(depth_dir)
        self.depth_scale_m_per_unit = float(depth_scale_m_per_unit)
        self.capacity = max(1, int(capacity))
        self._cache: dict[int, np.ndarray | None] = {}
        self._order: deque[int] = deque()

    def get(self, frame_index: int) -> np.ndarray | None:
        frame_index = int(frame_index)
        if frame_index in self._cache:
            return self._cache[frame_index]

        path = self.depth_dir / f"{frame_index:06d}.npy"
        if path.exists():
            raw = np.load(path)
            depth_m = raw.astype(np.float32) * self.depth_scale_m_per_unit
        else:
            depth_m = None

        if len(self._order) >= self.capacity:
            old = self._order.popleft()
            self._cache.pop(old, None)
        self._cache[frame_index] = depth_m
        self._order.append(frame_index)
        return depth_m


def sample_depth_patch(depth_m: np.ndarray, u: int, v: int, r: int) -> dict[str, float | int]:
    """Sample valid depth statistics around one projected pixel."""
    h, w = depth_m.shape[:2]
    r = max(0, int(r))
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    patch_px = int(patch.size)
    if valid.size == 0:
        return {
            "valid_px": 0,
            "patch_px": patch_px,
            "min_m": 0.0,
            "median_m": 0.0,
            "quality": 0.0,
        }
    return {
        "valid_px": int(valid.size),
        "patch_px": patch_px,
        "min_m": float(valid.min()),
        "median_m": float(np.median(valid)),
        "quality": float(valid.size / max(patch_px, 1)),
    }


def parse_semicolon_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    out: list[int] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            pass
    return out


def parse_semicolon_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    out: list[float] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def build_best_video_for_radar_frame(
    sync_rows: list[dict[str, Any]],
    max_time_diff_ms: float | None,
) -> tuple[dict[int, dict[str, int | None]], dict[str, int]]:
    """Map each radar frame to its closest synchronized video frame."""
    max_time_diff_us = None
    if max_time_diff_ms is not None:
        max_time_diff_us = int(round(float(max_time_diff_ms) * 1000.0))

    best: dict[int, dict[str, int | None]] = {}
    stats = {
        "candidate_links": 0,
        "links_missing_diff": 0,
        "links_over_time_gate": 0,
        "links_nonbest_duplicate": 0,
        "unique_radar_frames_selected": 0,
    }
    for row in sync_rows:
        if str(row.get("status", "matched")).strip() not in ("", "matched"):
            continue
        try:
            video_frame_index = int(float(row["video_frame_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        radar_frames = parse_semicolon_int_list(row.get("radar_frame_nums"))
        time_diffs = parse_semicolon_float_list(row.get("time_diffs_us"))
        for idx, radar_frame in enumerate(radar_frames):
            stats["candidate_links"] += 1
            diff_us = abs(int(round(time_diffs[idx]))) if idx < len(time_diffs) else None
            if diff_us is None:
                stats["links_missing_diff"] += 1
            if max_time_diff_us is not None and diff_us is not None and diff_us > max_time_diff_us:
                stats["links_over_time_gate"] += 1
                continue
            prev = best.get(radar_frame)
            cand_diff = int(diff_us if diff_us is not None else 10**18)
            if prev is None:
                best[radar_frame] = {
                    "video_frame_index": video_frame_index,
                    "time_diff_us": diff_us,
                }
            else:
                prev_diff = int(prev.get("time_diff_us") if prev.get("time_diff_us") is not None else 10**18)
                if cand_diff < prev_diff:
                    best[radar_frame] = {
                        "video_frame_index": video_frame_index,
                        "time_diff_us": diff_us,
                    }
                stats["links_nonbest_duplicate"] += 1
    stats["unique_radar_frames_selected"] = len(best)
    return best, stats


def _default_record(config: DepthCorrespondenceConfig, reason: str) -> dict[str, Any]:
    rec = {
        "depth_corr_available": 0,
        "depth_corr_in_fov": 0,
        "depth_corr_valid_px": 0,
        "depth_corr_patch_px": 0,
        "depth_corr_radar_z_m": 0.0,
        "depth_corr_min_m": 0.0,
        "depth_corr_median_m": 0.0,
        "depth_corr_selected_m": 0.0,
        "depth_corr_residual_m": 0.0,
        "depth_corr_abs_residual_m": 0.0,
        "depth_corr_sigma_m": float(config.sigma_min_m),
        "depth_corr_quality": 0.0,
        "p_depth_match": float(config.unknown_p_depth_match),
        "depth_foreground_occluded": 0,
        "depth_missing_reason": reason,
        "matched_video_frame_index": -1,
        "matched_depth_frame_index": -1,
        "match_time_diff_us": -1,
        "cam_u_depth": -1.0,
        "cam_v_depth": -1.0,
        "cam_semantic_bucket": "unknown",
        "cam_semantic_conf": 0.0,
    }
    for col in DEPTH_NUMERIC_COLS:
        rec.setdefault(col, 0.0)
    return rec


def make_default_depth_record(config: DepthCorrespondenceConfig, reason: str) -> dict[str, Any]:
    """Public helper for callers that cannot compute correspondence for a point."""
    return _default_record(config, reason)


def _majority_id(label_map: np.ndarray, u: int, v: int, r: int) -> tuple[int, float]:
    h, w = label_map.shape[:2]
    r = max(0, int(r))
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch = label_map[v0:v1, u0:u1].reshape(-1)
    if patch.size == 0:
        return -1, 0.0
    vals, counts = np.unique(patch, return_counts=True)
    best_idx = int(np.argmax(counts))
    return int(vals[best_idx]), float(counts[best_idx] / patch.size)


def semantic_bucket_from_name(name: str) -> str:
    text = str(name).strip().lower()
    if not text:
        return "unknown"
    if "person" in text or "human" in text:
        return "human"
    if any(token in text for token in ("floor", "ground", "road", "sidewalk", "earth")):
        return "floor"
    if any(
        token in text
        for token in (
            "wall",
            "door",
            "pillar",
            "column",
            "cabinet",
            "box",
            "shelf",
            "table",
            "chair",
            "stair",
            "railing",
            "structure",
        )
    ):
        return "structure"
    return "unknown"


def annotate_semantic_from_seg(
    records: list[dict[str, Any]],
    *,
    seg_dir: Path,
    seg_id_to_label: dict[str, str] | None,
    config: DepthCorrespondenceConfig,
) -> None:
    """Optionally attach local camera semantic bucket from OneFormer maps."""
    if not seg_dir.is_dir():
        return
    cache: dict[int, np.ndarray | None] = {}
    id_to_label = seg_id_to_label or {}
    for rec in records:
        vfi = int(rec.get("matched_video_frame_index", -1))
        if vfi < 0:
            continue
        if vfi not in cache:
            path = seg_dir / f"{vfi:06d}.npy"
            cache[vfi] = np.load(path).astype(np.int32) if path.exists() else None
        label_map = cache[vfi]
        if label_map is None:
            continue
        u = int(round(float(rec.get("cam_u_depth", -1.0))))
        v = int(round(float(rec.get("cam_v_depth", -1.0))))
        if not (0 <= u < label_map.shape[1] and 0 <= v < label_map.shape[0]):
            continue
        label_id, conf = _majority_id(label_map, u, v, config.semantic_patch_r)
        label_name = id_to_label.get(str(label_id), str(label_id))
        rec["cam_semantic_bucket"] = semantic_bucket_from_name(label_name)
        rec["cam_semantic_conf"] = round(float(conf), 4)


def compute_depth_correspondence_for_points(
    pts_radar: np.ndarray,
    *,
    depth_m: np.ndarray | None,
    video_frame_index: int,
    depth_frame_index: int,
    match_time_diff_us: int | None,
    calibration: dict[str, Any] | None,
    config: DepthCorrespondenceConfig,
) -> list[dict[str, Any]]:
    """Compute depth correspondence records for one radar frame."""
    n = int(pts_radar.shape[0])
    if n == 0:
        return []
    if calibration is None:
        return [_default_record(config, "bad_calibration") for _ in range(n)]
    if depth_m is None:
        out = [_default_record(config, "no_depth_frame") for _ in range(n)]
        for rec in out:
            rec["matched_video_frame_index"] = int(video_frame_index)
            rec["matched_depth_frame_index"] = int(depth_frame_index)
            rec["match_time_diff_us"] = int(match_time_diff_us) if match_time_diff_us is not None else -1
        return out

    uv, zc, valid = project_radar_to_image(
        pts_radar.astype(np.float32),
        calibration["R"],
        calibration["t"],
        calibration["K"],
        calibration["dist"],
        depth_m.shape[:2],
        use_distortion=bool(config.use_distortion),
    )

    records: list[dict[str, Any]] = []
    for i in range(n):
        rec = _default_record(config, "ok")
        rec["matched_video_frame_index"] = int(video_frame_index)
        rec["matched_depth_frame_index"] = int(depth_frame_index)
        rec["match_time_diff_us"] = int(match_time_diff_us) if match_time_diff_us is not None else -1
        rec["depth_corr_available"] = 1
        rec["depth_corr_radar_z_m"] = round(float(zc[i]) if np.isfinite(zc[i]) else 0.0, 5)
        rec["cam_u_depth"] = round(float(uv[i, 0]) if np.isfinite(uv[i, 0]) else -1.0, 3)
        rec["cam_v_depth"] = round(float(uv[i, 1]) if np.isfinite(uv[i, 1]) else -1.0, 3)

        if not bool(valid[i]):
            rec["depth_missing_reason"] = "off_image" if float(zc[i]) > 0.1 else "behind_camera"
            records.append(rec)
            continue

        rec["depth_corr_in_fov"] = 1
        ui, vi = int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))
        stats = sample_depth_patch(depth_m, ui, vi, config.depth_patch_r)
        rec["depth_corr_valid_px"] = int(stats["valid_px"])
        rec["depth_corr_patch_px"] = int(stats["patch_px"])
        rec["depth_corr_min_m"] = round(float(stats["min_m"]), 5)
        rec["depth_corr_median_m"] = round(float(stats["median_m"]), 5)
        rec["depth_corr_quality"] = round(float(stats["quality"]), 5)

        if int(stats["valid_px"]) <= 0:
            rec["depth_missing_reason"] = "no_valid_depth"
            records.append(rec)
            continue

        radar_z = float(zc[i])
        min_m = float(stats["min_m"])
        median_m = float(stats["median_m"])
        foreground = min_m > 0.0 and min_m < (radar_z - float(config.foreground_occlusion_margin_m))
        selected_m = min_m if foreground else median_m
        residual = radar_z - selected_m
        sigma = max(float(config.sigma_min_m), float(config.sigma_frac) * max(radar_z, 0.0))
        abs_z = min(abs(residual) / max(sigma, 1e-6), 4.0)
        p_depth = float(stats["quality"]) * float(np.exp(-0.5 * abs_z * abs_z))

        rec["depth_foreground_occluded"] = int(foreground)
        rec["depth_corr_selected_m"] = round(selected_m, 5)
        rec["depth_corr_residual_m"] = round(residual, 5)
        rec["depth_corr_abs_residual_m"] = round(abs(residual), 5)
        rec["depth_corr_sigma_m"] = round(sigma, 5)
        rec["p_depth_match"] = round(float(np.clip(p_depth, 0.0, 1.0)), 5)
        records.append(rec)
    return records


def cluster_depth_contradictions(
    records: list[dict[str, Any]],
    xyz: np.ndarray,
    *,
    config: DepthCorrespondenceConfig,
    p_dir_doppler: np.ndarray | None = None,
) -> None:
    """Attach simple DBSCAN-like cluster summaries to depth contradiction candidates."""
    n = len(records)
    if n == 0:
        return
    for rec in records:
        rec["depth_candidate_cluster_id"] = -1
        rec["depth_candidate_cluster_size"] = 0
        rec["depth_candidate_cluster_radius_m"] = 0.0
        rec["depth_candidate_cluster_min_depth_match"] = float(rec.get("p_depth_match", config.unknown_p_depth_match))
        rec["depth_candidate_cluster_mean_p_dir_doppler"] = (
            float(np.nanmean(p_dir_doppler)) if p_dir_doppler is not None and len(p_dir_doppler) else 1.0
        )

    candidate = np.zeros(n, dtype=bool)
    for i, rec in enumerate(records):
        if rec.get("depth_missing_reason") != "ok":
            continue
        residual = float(rec.get("depth_corr_abs_residual_m", 0.0))
        p_match = float(rec.get("p_depth_match", config.unknown_p_depth_match))
        candidate[i] = (
            residual > float(config.candidate_residual_m)
            or p_match < float(config.candidate_p_depth_match_max)
        )

    cand_idx = np.where(candidate)[0]
    if len(cand_idx) == 0:
        return

    eps = max(float(config.candidate_dbscan_eps_m), 1e-6)
    cand_pts = xyz[cand_idx].astype(np.float32)
    diff = cand_pts[:, None, :] - cand_pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    visited = np.zeros(len(cand_idx), dtype=bool)
    cluster_id = 0
    for local_start in range(len(cand_idx)):
        if visited[local_start]:
            continue
        queue = [local_start]
        visited[local_start] = True
        members: list[int] = []
        while queue:
            cur = queue.pop()
            members.append(cur)
            neighbors = np.where(dists[cur] <= eps)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
        if len(members) < int(config.candidate_min_samples):
            continue

        global_members = cand_idx[np.asarray(members, dtype=int)]
        pts = xyz[global_members].astype(np.float32)
        centroid = pts.mean(axis=0)
        radius = float(np.max(np.linalg.norm(pts - centroid, axis=1))) if len(pts) else 0.0
        min_depth = min(float(records[j].get("p_depth_match", 1.0)) for j in global_members)
        if p_dir_doppler is not None and len(p_dir_doppler) == n:
            mean_p_dir = float(np.mean(p_dir_doppler[global_members]))
        else:
            vals = [float(records[j].get("p_dir_doppler", 1.0)) for j in global_members]
            mean_p_dir = float(np.mean(vals)) if vals else 1.0
        for j in global_members:
            records[j]["depth_candidate_cluster_id"] = cluster_id
            records[j]["depth_candidate_cluster_size"] = int(len(global_members))
            records[j]["depth_candidate_cluster_radius_m"] = round(radius, 5)
            records[j]["depth_candidate_cluster_min_depth_match"] = round(min_depth, 5)
            records[j]["depth_candidate_cluster_mean_p_dir_doppler"] = round(mean_p_dir, 5)
        cluster_id += 1


def build_depth_correspondence_evidence(
    records: list[dict[str, Any]],
    *,
    sync_stats: dict[str, int] | None = None,
    config: DepthCorrespondenceConfig | None = None,
) -> dict[str, Any]:
    """Build a compact session/frame summary for QA and gating."""
    cfg = config or DepthCorrespondenceConfig()
    total = len(records)
    if total == 0:
        return {
            "points": 0,
            "depth_teacher_eligible": False,
            "failure_reason": "no_points",
        }

    reasons = Counter(str(r.get("depth_missing_reason", "unknown")) for r in records)
    available = sum(int(r.get("depth_corr_available", 0)) for r in records)
    in_fov = sum(int(r.get("depth_corr_in_fov", 0)) for r in records)
    ok_records = [r for r in records if r.get("depth_missing_reason") == "ok"]
    residuals = np.asarray([float(r.get("depth_corr_residual_m", 0.0)) for r in ok_records], dtype=np.float32)
    abs_residuals = np.abs(residuals)
    median_residual = float(np.median(residuals)) if residuals.size else 0.0
    residual_mad = float(np.median(np.abs(residuals - median_residual))) if residuals.size else 0.0
    p_depth = np.asarray([float(r.get("p_depth_match", cfg.unknown_p_depth_match)) for r in ok_records], dtype=np.float32)
    time_diffs_ms = np.asarray(
        [
            abs(float(r.get("match_time_diff_us", -1))) / 1000.0
            for r in records
            if int(r.get("match_time_diff_us", -1)) >= 0
        ],
        dtype=np.float32,
    )
    p95_ms = float(np.percentile(time_diffs_ms, 95)) if time_diffs_ms.size else 0.0
    projection_valid_rate = float(in_fov / max(total, 1))
    depth_available_rate = float(available / max(total, 1))
    depth_match_rate = float(np.mean(p_depth >= 0.5)) if p_depth.size else 0.0

    gates = {
        "projection_valid_rate": projection_valid_rate >= cfg.projection_valid_rate_min,
        "depth_available_rate": depth_available_rate >= cfg.depth_available_rate_min,
        "median_residual_abs": abs(median_residual) <= cfg.median_residual_abs_max_m,
        "residual_mad": residual_mad <= cfg.residual_mad_max_m,
        "time_match_p95_ms": p95_ms <= cfg.time_match_p95_max_ms,
    }
    eligible = all(gates.values())
    failed = [name for name, ok in gates.items() if not ok]
    return {
        "points": int(total),
        "ok_points": int(len(ok_records)),
        "depth_teacher_eligible": bool(eligible),
        "failure_reason": ",".join(failed),
        "projection_valid_rate": round(projection_valid_rate, 5),
        "depth_available_rate": round(depth_available_rate, 5),
        "depth_match_rate": round(depth_match_rate, 5),
        "foreground_occlusion_rate": round(
            sum(int(r.get("depth_foreground_occluded", 0)) for r in records) / max(total, 1),
            5,
        ),
        "median_depth_residual_m": round(median_residual, 5),
        "median_abs_depth_residual_m": round(float(np.median(abs_residuals)) if abs_residuals.size else 0.0, 5),
        "depth_residual_mad_m": round(residual_mad, 5),
        "mean_p_depth_match": round(float(np.mean(p_depth)) if p_depth.size else float(cfg.unknown_p_depth_match), 5),
        "time_match_p95_ms": round(p95_ms, 3),
        "missing_reason_counts": dict(reasons),
        "sync": sync_stats or {},
    }
