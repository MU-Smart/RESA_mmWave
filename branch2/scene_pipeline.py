#!/usr/bin/env python3
"""Scene pipeline orchestrator — logging, aggregation, decision, and mapping."""

# =========================================================================
# 1. nav_logger — logging infrastructure
# =========================================================================

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional
import queue
import threading

import numpy as np

_NAV_LOG_DIR: str | None = None


def _setup_logging() -> logging.Logger:
    global _NAV_LOG_DIR
    _NAV_LOG_DIR = os.environ.get("NAV_LOG_DIR")
    if _NAV_LOG_DIR is None:
        _NAV_LOG_DIR = str(Path.home() / "nav_logs" / f"nav_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    Path(_NAV_LOG_DIR).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nav")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d  %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        log_file = Path(_NAV_LOG_DIR) / f"nav_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        fh = logging.FileHandler(str(log_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class NavLogger:
    def __init__(self, component: str) -> None:
        self._component = component
        self._logger = logging.getLogger("nav")

    def _fmt(self, msg: str) -> str:
        return f"[{self._component}]  {msg}"

    def info(self, msg: str, **kwargs) -> None:
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.info(self._fmt(msg) + extra)

    def warn(self, msg: str, **kwargs) -> None:
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.warning(self._fmt(msg) + extra)

    def error(self, msg: str, **kwargs) -> None:
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.error(self._fmt(msg) + extra)

    def debug(self, msg: str, **kwargs) -> None:
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.debug(self._fmt(msg) + extra)

    def exception(self, msg: str, **kwargs) -> None:
        extra = "  " + "  ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        self._logger.exception(self._fmt(msg) + extra)

    def warning(self, msg: str, **kwargs) -> None:
        """Alias for warn()."""
        self.warn(msg, **kwargs)

    @property
    def log_dir(self) -> str:
        return str(_NAV_LOG_DIR or "")


_logger_setup_done = False


def get_logger(component: str) -> NavLogger:
    global _logger_setup_done
    if not _logger_setup_done:
        _setup_logging()
        _logger_setup_done = True
    return NavLogger(component)


def get_log_path() -> str:
    return str(_NAV_LOG_DIR or "")


# =========================================================================
# 2. scene_aggregator — point aggregation into structured scene evidence
# =========================================================================

log_cache = {}


def _log(component: str) -> NavLogger:
    if component not in log_cache:
        log_cache[component] = get_logger(component)
    return log_cache[component]


def m_to_ft(meters: float) -> float:
    return float(meters) * 3.28084


def azimuth_sector(x: float) -> str:
    if x < -0.35:
        return "left"
    if x > 0.35:
        return "right"
    return "center"


def range_band(y: float) -> str:
    if y <= 1.2:
        return "immediate"
    if y <= 3.0:
        return "near"
    return "far"


def doppler_motion(doppler: float) -> str:
    if abs(doppler) < 0.15:
        return "stationary"
    if doppler < 0:
        return "approaching"
    return "receding"


def _closest_forward_y(arr: np.ndarray) -> float:
    forward = arr[arr > 0.1]
    return float(np.min(forward)) if len(forward) > 0 else float("inf")


def _blocking_summary(structure_pts: np.ndarray) -> Dict[str, int]:
    if len(structure_pts) == 0:
        return {"center_points": 0, "left_points": 0, "right_points": 0}
    xs = structure_pts[:, 0]
    center_mask = (xs >= -0.35) & (xs <= 0.35)
    left_mask = xs < -0.35
    right_mask = xs > 0.35
    return {
        "center_points": int(center_mask.sum()),
        "left_points": int(left_mask.sum()),
        "right_points": int(right_mask.sum()),
    }


def corridor_open_directions(structure_pts: np.ndarray) -> dict:
    if len(structure_pts) == 0:
        return {"center_clear": True, "left_clear": True, "right_clear": True}
    xs = structure_pts[:, 0]
    sectors = {"center": [], "left": [], "right": []}
    for i in range(len(structure_pts)):
        x = float(xs[i])
        if x < -0.35:
            sectors["left"].append(structure_pts[i])
        elif x > 0.35:
            sectors["right"].append(structure_pts[i])
        else:
            sectors["center"].append(structure_pts[i])
    result = {}
    for sector, pts in sectors.items():
        if not pts:
            result[f"{sector}_clear"] = True
        else:
            min_y = min(p[1] for p in pts)
            result[f"{sector}_clear"] = min_y > 1.5
    return result


def _class_counts(struct_pts: list, floor_pts: list, human_pts: list) -> dict:
    return {"structure": len(struct_pts), "floor": len(floor_pts), "human": len(human_pts)}


def _ego_velocity_components(ego_velocity_mps) -> tuple[float, float, bool]:
    if ego_velocity_mps is None:
        return 0.0, 0.0, False
    if isinstance(ego_velocity_mps, dict):
        return float(ego_velocity_mps.get("vx_mps", 0.0)), float(ego_velocity_mps.get("vy_mps", 0.0)), bool(ego_velocity_mps.get("available", False))
    return 0.0, 0.0, False


def _scene_confidence(total_pts: int, counts: dict) -> str:
    if total_pts >= 60:
        return "high"
    if total_pts >= 30:
        return "medium"
    return "low"


def _base_evidence_channels(counts: dict, total_pts: int) -> dict:
    return {"structure_count": counts.get("structure", 0), "floor_count": counts.get("floor", 0),
            "human_count": counts.get("human", 0), "total_points": total_pts, "scene_confidence": _scene_confidence(total_pts, counts)}


def aggregate_scene(points: list, window_frames: int = 3, ego_velocity_mps=None) -> dict:
    _log("Aggr").debug(f"Aggregating {len(points)} points over {window_frames} frames")
    struct_pts, floor_pts, human_pts = [], [], []
    for pt in points:
        cls = str(pt.get("pred_class", "")).lower()
        if cls == "structure":
            struct_pts.append(pt)
        elif cls == "floor":
            floor_pts.append(pt)
        elif cls == "human":
            human_pts.append(pt)

    struct_np = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0), p.get("doppler", 0.0), p.get("snr", 0.0), p.get("spatiotemporal_weight", 1.0)] for p in struct_pts]) if struct_pts else np.empty((0, 6))
    floor_np = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0), p.get("doppler", 0.0), p.get("snr", 0.0), p.get("spatiotemporal_weight", 1.0)] for p in floor_pts]) if floor_pts else np.empty((0, 6))
    human_np = np.array([[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0), p.get("doppler", 0.0), p.get("snr", 0.0), p.get("spatiotemporal_weight", 1.0)] for p in human_pts]) if human_pts else np.empty((0, 6))

    counts = _class_counts(struct_pts, floor_pts, human_pts)
    total_pts = int(len(points))
    evx, evy, ego_avail = _ego_velocity_components(ego_velocity_mps)

    struct_summary = _aggregate_structure(struct_np)
    floor_summary = _aggregate_floor(floor_np)
    human_summary = _aggregate_humans(human_np, ego_velocity_mps=ego_velocity_mps)

    blocking = _blocking_summary(struct_np)
    open_dirs = corridor_open_directions(struct_np)
    blocking["center_effective_points"] = int(struct_np[(struct_np[:, 0] >= -0.35) & (struct_np[:, 0] <= 0.35)].shape[0])

    scene = {"total_points": total_pts, "window_frames": int(window_frames), "class_counts": counts,
             "blocking": blocking, "open_directions": open_dirs, "ego_velocity_mps": ego_velocity_mps,
             "confidence": _scene_confidence(total_pts, counts),
             "structure": struct_summary, "floor": floor_summary, "humans": human_summary,
             "evidence_channels": {"base": _base_evidence_channels(counts, total_pts)}}
    return scene


def _empty_scene() -> dict:
    return {"total_points": 0, "window_frames": 0, "class_counts": {"structure": 0, "floor": 0, "human": 0},
            "blocking": {"center_points": 0, "left_points": 0, "right_points": 0, "center_effective_points": 0},
            "open_directions": {"center_clear": True, "left_clear": True, "right_clear": True},
            "structure": {"closest_m": float("inf"), "closest_ft": float("inf"), "closest_sector": "none", "centroid": [0, 0, 0]},
            "floor": {"n_points": 0, "mean_z_m": 0.0, "range_m": 0.0},
            "humans": {"count": 0, "detections": [], "closest_m": float("inf")},
            "confidence": "low", "evidence_channels": {"base": {"structure_count": 0, "floor_count": 0, "human_count": 0, "total_points": 0, "scene_confidence": "low"}}}


def _aggregate_structure(pts: np.ndarray) -> dict:
    if pts.shape[0] == 0:
        return {"closest_m": float("inf"), "closest_ft": float("inf"), "closest_sector": "none", "centroid": [0, 0, 0]}
    ys = pts[:, 1]
    closest_idx = int(np.argmin(np.abs(ys)))
    closest_m = float(abs(ys[closest_idx]))
    closest_ft = m_to_ft(closest_m)
    return {"closest_m": round(closest_m, 2), "closest_ft": round(closest_ft, 2),
            "closest_sector": azimuth_sector(float(pts[closest_idx, 0])),
            "centroid": [float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), float(np.mean(pts[:, 2]))]}


def _aggregate_floor(pts: np.ndarray) -> dict:
    if pts.shape[0] == 0:
        return {"n_points": 0, "mean_z_m": 0.0, "range_m": 0.0}
    zs = pts[:, 2]
    return {"n_points": int(pts.shape[0]), "mean_z_m": round(float(np.mean(zs)), 3), "range_m": round(float(np.ptp(zs)), 3)}


def _aggregate_humans(pts: np.ndarray, *, ego_velocity_mps=None) -> dict:
    if pts.shape[0] == 0:
        return {"count": 0, "detections": [], "closest_m": float("inf")}
    detections = []
    for row in pts:
        detections.append({"x": round(float(row[0]), 2), "y": round(float(row[1]), 2), "z": round(float(row[2]), 2),
                           "doppler": round(float(row[3]), 2), "range_m": round(np.linalg.norm([row[0], row[1], row[2]]), 2),
                           "azimuth_sector": azimuth_sector(float(row[0])), "range_band": range_band(float(row[1])),
                           "snr": round(float(row[4]), 1), "motion": doppler_motion(float(row[3]))})
    ys = pts[:, 1]
    closest_m = float(np.min(np.abs(ys[ys > 0]))) if np.any(ys > 0) else float(np.min(np.abs(ys))) if len(ys) > 0 else float("inf")
    return {"count": int(pts.shape[0]), "detections": detections, "closest_m": round(closest_m, 2)}


# =========================================================================
# 3. nav_decision — navigation decision logic
# =========================================================================

def _sorted_by_distance(items: List[dict]) -> List[dict]:
    return sorted(items, key=lambda x: float(x.get("y", float("inf"))))


def _first(items: List[dict], sector: Optional[str] = None) -> Optional[dict]:
    if sector:
        items = [i for i in items if azimuth_sector(float(i.get("x", 0))) == sector]
    items = _sorted_by_distance(items)
    return items[0] if items else None


def _choose_open_side(scene: dict, humans: List[dict], human_empty: bool, center_clear: bool) -> str:
    if not human_empty:
        human = _first(humans)
        if human:
            side = azimuth_sector(float(human.get("x", 0)))
            return "left" if side == "right" else "right"
    if not center_clear:
        for side in ("left", "right"):
            if scene.get("open_directions", {}).get(f"{side}_clear", True):
                return side
    return "center"


def _parse_unet(unet_freespace: dict) -> tuple:
    if not unet_freespace:
        return "none", "unknown", False
    conf = str(unet_freespace.get("free_space_confidence", "none"))
    dirs = {k.replace("_free", ""): bool(v) for k, v in unet_freespace.items() if k.endswith("_free")}
    return conf, *([True] if not dirs else [dirs.get("center", True)])


def _blocking_strength(blocking: dict) -> str:
    center = int(blocking.get("center_points", 0))
    if center >= 30:
        return "strong"
    if center >= 12:
        return "medium"
    if center >= 4:
        return "weak"
    return "none"


def compute_nav_decision(scene: dict) -> Dict[str, Any]:
    blocking = scene.get("blocking", {})
    humans = scene.get("humans", {})
    open_dirs = scene.get("open_directions", {})
    human_detections = humans.get("detections", [])
    human_empty = len(human_detections) == 0
    center_clear = open_dirs.get("center_clear", True)
    center_points = int(blocking.get("center_points", 0))
    blocking_confidence = _blocking_strength(blocking)

    reason = "clear_path"
    urgency = "none"
    primary_action = "continue_straight"
    best_direction = "center"

    if not human_empty:
        nearest = _first(human_detections)
        if nearest:
            dist_m = float(nearest.get("y", float("inf")))
            dist_ft = m_to_ft(dist_m)
            motion = str(nearest.get("motion", "stationary"))
            raw_motion = "approaching" if motion == "approaching" else "stationary"
            rel_motion = motion
            sector = azimuth_sector(float(nearest.get("x", 0)))
            if dist_ft <= 10 and motion == "approaching":
                reason = "approaching_person"
                urgency = "urgent"
                primary_action = "stop"
                best_direction = _choose_open_side(scene, human_detections, human_empty, center_clear)
            elif dist_ft <= 15:
                reason = "person_ahead"
                urgency = "moderate"
                primary_action = "slow_down"
                best_direction = _choose_open_side(scene, human_detections, human_empty, center_clear)
            elif not center_clear:
                reason = "blocked_path"
                urgency = "moderate"
                primary_action = "turn_away"
                best_direction = "left" if open_dirs.get("left_clear", True) else "right"
            else:
                reason = "human_nearby"
                urgency = "low"
                primary_action = "continue_straight"
                best_direction = "center"
            human_decision = {"nearest_human_ft": round(dist_ft, 1), "nearest_human_motion": rel_motion,
                             "nearest_human_raw_motion": raw_motion, "nearest_human_sector": sector,
                             "nearest_human_points": int(nearest.get("count", 0)) if "count" in nearest else 1,
                             "n_humans": int(humans.get("count", 0))}
        else:
            human_decision = {"nearest_human_ft": float("inf"), "nearest_human_motion": "unknown",
                             "nearest_human_raw_motion": "unknown", "nearest_human_sector": "none",
                             "nearest_human_points": 0, "n_humans": 0}
    else:
        human_decision = {"nearest_human_ft": float("inf"), "nearest_human_motion": "none",
                         "nearest_human_raw_motion": "none", "nearest_human_sector": "none",
                         "nearest_human_points": 0, "n_humans": 0}
        if not center_clear:
            reason = "blocked_path"
            urgency = "moderate"
            primary_action = "turn_away"
            best_direction = "left" if open_dirs.get("left_clear", True) else "right"

    decision: Dict[str, Any] = {"primary_action": primary_action, "reason": reason, "urgency": urgency,
                                "best_direction": best_direction, "center_clear": center_clear,
                                "left_clear": open_dirs.get("left_clear", True), "right_clear": open_dirs.get("right_clear", True),
                                "blocking_confidence": blocking_confidence, "center_points": center_points,
                                "confidence": scene.get("confidence", "low"), "scene_confidence": scene.get("confidence", "low"),
                                "total_scene_points": int(scene.get("total_points", 0))}
    decision.update(human_decision)
    return decision


# =========================================================================
# 4. map_builder — occupancy grid and anomaly detection
# =========================================================================

class EgoVelocityEstimator:
    def __init__(self, alpha: float = 0.6, max_speed_mps: float = 2.5, min_static_inlier_fraction: float = 0.25):
        self._alpha = alpha
        self._max_speed = max_speed_mps
        self._min_static_frac = min_static_inlier_fraction
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._available: bool = False
        self._last_update: float = 0.0
        self._n_updates: int = 0

    def estimate(self, points: list[dict[str, Any]], timestamp: float = 0.0, dt_s: float = 0.033) -> None:
        if len(points) < 8:
            self._available = False
            return
        unit_vectors, radial = [], []
        for pt in points:
            cls = str(pt.get("pred_class", "")).lower()
            if cls not in ("structure", "floor"):
                continue
            doppler = float(pt.get("doppler", 0.0))
            if not np.isfinite(doppler) or abs(doppler) > self._max_speed:
                continue
            x, y = float(pt.get("x", 0.0)), float(pt.get("y", 0.0))
            rng = np.hypot(x, y)
            if rng < 0.1:
                continue
            unit_vectors.append([x / rng, y / rng])
            radial.append(-doppler)
        if len(unit_vectors) < 8:
            self._available = False
            return
        u = np.asarray(unit_vectors, dtype=np.float64)
        r = np.asarray(radial, dtype=np.float64)
        solution, *_ = np.linalg.lstsq(u, r, rcond=None)
        vx, vy = float(solution[0]), float(solution[1])
        residuals = np.abs(u @ np.asarray([vx, vy]) - r)
        inliers = residuals <= 0.3
        frac = float(inliers.sum()) / max(len(residuals), 1)
        if frac >= self._min_static_frac and np.hypot(vx, vy) <= self._max_speed:
            if self._n_updates > 0:
                self._vx = self._alpha * vx + (1 - self._alpha) * self._vx
                self._vy = self._alpha * vy + (1 - self._alpha) * self._vy
            else:
                self._vx, self._vy = vx, vy
            self._available = True
            self._n_updates += 1
            self._last_update = timestamp
        else:
            self._available = False

    def get_velocity(self) -> dict[str, Any]:
        return {"available": self._available, "vx_mps": round(self._vx, 3), "vy_mps": round(self._vy, 3),
                "speed_mps": round(np.hypot(self._vx, self._vy), 3), "n_updates": self._n_updates}


class OccupancyGrid:
    def __init__(self, cell_m: float = 0.25, range_m: float = 5.0):
        self._cell_m = cell_m
        self._range_m = range_m
        self._n_cells = int(np.ceil(2 * range_m / cell_m))
        offset = (self._n_cells - 1) / 2.0
        self._grid_structure = np.zeros((self._n_cells, self._n_cells), dtype=np.float32)
        self._grid_free = np.zeros((self._n_cells, self._n_cells), dtype=np.float32)
        self._grid_freespace = np.ones((self._n_cells, self._n_cells), dtype=np.float32)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.round((x / self._cell_m) + (self._n_cells - 1) / 2.0))
        iy = int(np.round((y / self._cell_m) + (self._n_cells - 1) / 2.0))
        return np.clip(ix, 0, self._n_cells - 1), np.clip(iy, 0, self._n_cells - 1)

    def update_structure(self, points: list[dict[str, Any]], decay: float = 0.98) -> None:
        self._grid_structure *= decay
        for pt in points:
            if str(pt.get("pred_class", "")).lower() != "structure":
                continue
            cx, cy = self._cell(float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
            weight = float(pt.get("map_update_weight", 1.0))
            self._grid_structure[cx, cy] = min(self._grid_structure[cx, cy] + weight, 10.0)

    def update_free(self, points: list[dict[str, Any]], decay: float = 0.98) -> None:
        self._grid_free *= decay
        for pt in points:
            if str(pt.get("pred_class", "")).lower() != "floor":
                continue
            cx, cy = self._cell(float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
            weight = float(pt.get("map_update_weight", 1.0))
            self._grid_free[cx, cy] = min(self._grid_free[cx, cy] + weight, 10.0)

    def query_anomalies(self, x: float, y: float, radius_m: float = 0.5) -> dict[str, Any]:
        cx, cy = self._cell(x, y)
        r = max(1, int(radius_m / self._cell_m))
        x_slice = slice(max(0, cx - r), min(self._n_cells, cx + r + 1))
        y_slice = slice(max(0, cy - r), min(self._n_cells, cy + r + 1))
        struct_hist = float(np.sum(self._grid_structure[x_slice, y_slice]))
        free_hist = float(np.sum(self._grid_free[x_slice, y_slice]))
        return {"novel_occupancy": struct_hist < 0.5, "novel_free_space": free_hist < 0.5,
                "struct_history": struct_hist, "free_history": free_hist, "has_history": struct_hist > 0.5 or free_hist > 0.5}


class MapBuilder(threading.Thread):
    MAP_UPDATE_MODE_UNIFORM = "uniform"
    MAP_UPDATE_MODE_EGO_ONLY = "ego_only"
    MAP_UPDATE_MODE_EGO_DIRECTNESS = "ego_directness"
    MAP_UPDATE_MODE_LEARNED_ACC = "learned_acc"

    def __init__(self, cell_m: float = 0.25, range_m: float = 5.0, decay: float = 0.98, session_s: float = 1.5,
                 save_path: str | Path | None = None, save_every: int = 0,
                 map_update_mode: str = "uniform"):
        super().__init__(daemon=True)
        self._cell_m = cell_m
        self._range_m = range_m
        self._decay = decay
        self._session_s = session_s
        self._save_path = Path(save_path) if save_path else None
        self._save_every = save_every
        self._grid = OccupancyGrid(cell_m, range_m)
        self._ego = EgoVelocityEstimator()
        self._push_queue: queue.Queue = queue.Queue()
        self._running = False
        self._map_update_mode = map_update_mode
        self._map_update_stats: dict[str, Any] = {
            "mode": map_update_mode,
            "points_seen": 0,
            "points_updated": 0,
            "mean_weight_by_class": {},
            "suppressed_low_directness": 0,
            "suppressed_low_ego": 0,
            "suppressed_low_acc": 0,
        }

    @property
    def _queue(self) -> queue.Queue:
        """Backward-compat alias for _push_queue, used by branch3_replay_simulation."""
        return self._push_queue

    def compute_map_update_weight(self, point: dict[str, Any], frame_context: dict[str, Any] | None = None) -> float:
        """Compute the map update weight for a single point based on the current mode.

        Modes:
            uniform: weight = 1.0 (current behavior)
            ego_only: weight = pred_conf * p_ego
            ego_directness: weight = pred_conf * p_ego * p_dir_doppler
            learned_acc: weight = pred_conf * p_acc * p_ego * (1-rig_prior) * (1-mirror_score) * class_weight
        """
        mode = self._map_update_mode
        if mode == self.MAP_UPDATE_MODE_UNIFORM:
            return 1.0

        class_conf = float(point.get("pred_conf", 1.0))
        p_ego = float(point.get("p_ego", 1.0))
        p_dir = float(point.get("p_dir_doppler", 1.0))
        p_acc = float(point.get("p_acc", 1.0))
        rig_prior = float(point.get("rig_prior", 0.0))
        mirror_score = float(point.get("mirror_score", 0.0))
        class_weight = {
            "structure": 1.0,
            "floor": 1.0,
            "human": 0.5,
        }.get(str(point.get("pred_class", "")).lower(), 1.0)

        if mode == self.MAP_UPDATE_MODE_EGO_ONLY:
            return float(np.clip(class_conf * p_ego, 0.0, 1.0))

        if mode == self.MAP_UPDATE_MODE_EGO_DIRECTNESS:
            return float(np.clip(class_conf * p_ego * p_dir, 0.0, 1.0))

        if mode == self.MAP_UPDATE_MODE_LEARNED_ACC:
            weight = class_conf * p_acc * p_ego * (1.0 - rig_prior) * (1.0 - mirror_score) * class_weight
            return float(np.clip(weight, 0.0, 1.0))

        return 1.0

    def push_classified_points(self, points: list[dict[str, Any]]) -> None:
        # Attach map_update_weight to each point before queueing
        annotated = []
        for pt in points:
            p = dict(pt)
            p["map_update_weight"] = self.compute_map_update_weight(p)
            annotated.append(p)
        self._push_queue.put(annotated)

    def run(self) -> None:
        self._running = True
        while self._running:
            try:
                pts = self._push_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Update map update stats
                self._map_update_stats["points_seen"] += len(pts)
                weights = [float(p.get("map_update_weight", 1.0)) for p in pts]
                by_class: dict[str, list[float]] = {}
                acc_by_class: dict[str, list[float]] = {}
                for p, w in zip(pts, weights):
                    cls = str(p.get("pred_class", "unknown"))
                    by_class.setdefault(cls, []).append(w)
                    if "p_acc" in p:
                        acc_by_class.setdefault(cls, []).append(float(p.get("p_acc", 1.0)))
                    if w < 0.5 and float(p.get("p_dir_doppler", 1.0)) < 0.5:
                        self._map_update_stats["suppressed_low_directness"] += 1
                    if w < 0.5 and float(p.get("p_ego", 1.0)) < 0.5:
                        self._map_update_stats["suppressed_low_ego"] += 1
                    if w < 0.5 and float(p.get("p_acc", 1.0)) < 0.5:
                        self._map_update_stats["suppressed_low_acc"] += 1
                self._map_update_stats["mean_weight_by_class"] = {
                    cls: round(float(np.mean(wlist)), 4)
                    for cls, wlist in by_class.items()
                }
                self._map_update_stats["mean_p_acc_by_class"] = {
                    cls: round(float(np.mean(vlist)), 4)
                    for cls, vlist in acc_by_class.items()
                }
                self._map_update_stats["points_updated"] += sum(1 for w in weights if w > 0.01)

                self._grid.update_structure(pts)
                self._grid.update_free(pts)
            finally:
                self._push_queue.task_done()

    def stop(self) -> None:
        self._running = False

    def query_anomalies(self, x: float, y: float, radius_m: float = 0.5) -> dict[str, Any]:
        return self._grid.query_anomalies(x, y, radius_m)

    def query_occupancy(self, x: float, y: float, radius_m: float = 0.5) -> dict[str, Any]:
        """Return occupancy status at a query position. Alias/wrapper for query_anomalies
        that also exposes raw grid values for map inspection."""
        cx, cy = self._grid._cell(x, y)
        r = max(1, int(radius_m / self._cell_m))
        x_slice = slice(max(0, cx - r), min(self._grid._n_cells, cx + r + 1))
        y_slice = slice(max(0, cy - r), min(self._grid._n_cells, cy + r + 1))
        struct_val = float(self._grid._grid_structure[cx, cy])
        free_val = float(self._grid._grid_free[cx, cy])
        struct_sum = float(np.sum(self._grid._grid_structure[x_slice, y_slice]))
        free_sum = float(np.sum(self._grid._grid_free[x_slice, y_slice]))
        return {
            "struct_at_cell": struct_val,
            "free_at_cell": free_val,
            "struct_in_radius": struct_sum,
            "free_in_radius": free_sum,
            "has_occupancy": struct_sum > 0.5,
        }

    def get_point_occupancy_scores(self, xyz: list | np.ndarray, radius_m: float = 0.25) -> np.ndarray:
        """Return per-point structure-history scores for use as persist_score.

        Args:
            xyz: (N, 3) array or list of [x, y, z] triples.
            radius_m: radius in meters for the occupancy query.

        Returns:
            (N,) float32 array of structure-history values.
        """
        xyz_arr = np.asarray(xyz, dtype=np.float64)
        if xyz_arr.ndim != 2 or xyz_arr.shape[1] < 2:
            return np.zeros(0, dtype=np.float32)
        scores = np.zeros(len(xyz_arr), dtype=np.float32)
        for i in range(len(xyz_arr)):
            cx, cy = self._grid._cell(float(xyz_arr[i, 0]), float(xyz_arr[i, 1]))
            scores[i] = float(self._grid._grid_structure[cx, cy])
        return scores

    def get_polar_prior(self, shape: tuple | None = None) -> np.ndarray:
        """Return the structure occupancy grid as a polar prior for Branch 3.

        Returns a (H, W) float32 array usable as a U-Net prior channel.
        Default shape is derived from the occupancy grid dimensions.
        """
        n_cells = self._grid._n_cells
        if shape is not None:
            h, w = shape
            # Downsample/upsample from native grid to requested shape
            from PIL import Image
            img = Image.fromarray((self._grid._grid_structure.T * 255).astype(np.uint8))
            img = img.resize((w, h), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32) / 255.0
        return np.asarray(self._grid._grid_structure.T, dtype=np.float32).copy()

    def estimate_ego_velocity(self, points: list[dict[str, Any]], timestamp: float = 0.0, dt_s: float = 0.033) -> dict[str, Any]:
        """Estimate ego velocity from classified points (replay/live compatible).

        Runs the internal ego estimator and returns a dict matching the
        contract expected by navigation_loop.py and branch3_replay_simulation.py.
        """
        self._ego.estimate(points, timestamp, dt_s)
        vel = self._ego.get_velocity()
        vel["source"] = "map_builder"
        return vel

    def get_ego_velocity(self) -> dict[str, Any]:
        return self._ego.get_velocity()

    def update_ego_velocity(self, points: list[dict[str, Any]], timestamp: float = 0.0, dt_s: float = 0.033) -> None:
        self._ego.estimate(points, timestamp, dt_s)


# Explicit re-exports to match original import expectations
__all__ = ["get_logger", "get_log_path", "NavLogger",
           "aggregate_scene", "compute_nav_decision",
           "MapBuilder", "OccupancyGrid", "EgoVelocityEstimator",
           "m_to_ft", "azimuth_sector", "corridor_open_directions"]
