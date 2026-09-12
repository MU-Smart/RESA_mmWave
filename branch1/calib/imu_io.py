#!/usr/bin/env python3
"""Parser and diagnostics for RESA's interleaved D435i IMU CSV export.

CSV schema (confirmed against real sessions in `ds_buildingunknown_2026-07-20`,
see manifests/imu_audit_2026-09-04.md):

    sample_index, timestamp_ms, sensor_type, x, y, z

Gyro and accelerometer rows are interleaved but asynchronous -- they must be
split into independent streams before use and never zipped into fake 6-axis
packets (see docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md section 3, carried
over from the reference document this plan is based on).

Deliberate deviation from the reference document's literal timestamp advice:
it recommends converting to seconds relative to session start before
optimization, to avoid numerically poor optimization on raw epoch values.
This repo's existing spatiotemporal calibration code
(spatiotemporal_calibrate.py) already interpolates radar/color/depth
timestamps in absolute EPOCH MICROSECONDS as int/float64, and float64 keeps
sub-microsecond precision at that magnitude (~1.8e15). Rather than introduce
a second, incompatible timestamp convention just for IMU data, this module
converts the CSV's timestamp_ms (epoch milliseconds) to the same
epoch-microsecond convention used everywhere else in this codebase.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class IMUStream:
    """One IMU sensor stream (gyro or accel), sorted by timestamp."""

    sensor_type: str
    timestamps_us: np.ndarray  # float64, epoch microseconds
    values: np.ndarray  # (N, 3) float64

    def __len__(self) -> int:
        return int(len(self.timestamps_us))

    def value_at(self, timestamp_us: float) -> np.ndarray:
        """Component-wise linear interpolation, clamped to the stream span."""
        if len(self.timestamps_us) == 0:
            return np.zeros(3, dtype=np.float64)
        if len(self.timestamps_us) == 1:
            return self.values[0].copy()
        ts = float(np.clip(timestamp_us, self.timestamps_us[0], self.timestamps_us[-1]))
        return np.asarray(
            [np.interp(ts, self.timestamps_us, self.values[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )

    def rotated(self, rotation: np.ndarray) -> "IMUStream":
        """Copy with every sample rotated by a fixed 3x3 rotation (e.g. the
        factory depth<->color rotation), timestamps unchanged."""
        rotation = np.asarray(rotation, dtype=np.float64)
        return IMUStream(
            sensor_type=self.sensor_type,
            timestamps_us=self.timestamps_us,
            values=(rotation @ self.values.T).T if len(self.values) else self.values,
        )


@dataclass(slots=True)
class IMUStreams:
    gyro: IMUStream
    accel: IMUStream
    n_unknown_sensor_type_rows: int = 0


def load_imu_csv(path: Path) -> IMUStreams:
    gyro_rows: list[tuple[float, float, float, float]] = []
    accel_rows: list[tuple[float, float, float, float]] = []
    n_unknown = 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sensor_type = (row.get("sensor_type") or "").strip().lower()
            # timestamp_ms carries sub-millisecond precision (e.g.
            # 1784587258195.6545) -- never cast to int before converting.
            timestamp_us = float(row["timestamp_ms"]) * 1000.0
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            if sensor_type == "gyro":
                gyro_rows.append((timestamp_us, x, y, z))
            elif sensor_type == "accel":
                accel_rows.append((timestamp_us, x, y, z))
            else:
                n_unknown += 1

    def _to_stream(sensor_type: str, rows: list[tuple[float, float, float, float]]) -> IMUStream:
        if not rows:
            return IMUStream(sensor_type=sensor_type, timestamps_us=np.zeros(0), values=np.zeros((0, 3)))
        rows_sorted = sorted(rows, key=lambda r: r[0])
        ts = np.asarray([r[0] for r in rows_sorted], dtype=np.float64)
        vals = np.asarray([r[1:] for r in rows_sorted], dtype=np.float64)
        return IMUStream(sensor_type=sensor_type, timestamps_us=ts, values=vals)

    return IMUStreams(
        gyro=_to_stream("gyro", gyro_rows),
        accel=_to_stream("accel", accel_rows),
        n_unknown_sensor_type_rows=n_unknown,
    )


def find_imu_csv(session_dir: Path) -> Path | None:
    candidate = session_dir / f"{session_dir.name}_imu.csv"
    return candidate if candidate.exists() else None


def stream_cadence_stats(stream: IMUStream) -> dict[str, Any]:
    if len(stream) < 2:
        return {
            "n_samples": int(len(stream)),
            "median_dt_ms": float("nan"),
            "effective_hz": float("nan"),
            "min_dt_ms": float("nan"),
            "max_dt_ms": float("nan"),
            "monotonic": True,
            "n_nonpositive_dt": 0,
        }
    dt_us = np.diff(stream.timestamps_us)
    dt_ms = dt_us / 1000.0
    median_dt_ms = float(np.median(dt_ms))
    return {
        "n_samples": int(len(stream)),
        "median_dt_ms": median_dt_ms,
        "effective_hz": float(1000.0 / median_dt_ms) if median_dt_ms > 0 else float("nan"),
        "min_dt_ms": float(np.min(dt_ms)),
        "max_dt_ms": float(np.max(dt_ms)),
        "monotonic": bool(np.all(dt_us > 0.0)),
        "n_nonpositive_dt": int(np.sum(dt_us <= 0.0)),
    }


def accel_gravity_check(stream: IMUStream) -> dict[str, Any]:
    if len(stream) == 0:
        return {"mean_magnitude_mps2": float("nan"), "min_magnitude_mps2": float("nan"), "max_magnitude_mps2": float("nan")}
    magnitudes = np.linalg.norm(stream.values, axis=1)
    return {
        "mean_magnitude_mps2": float(np.mean(magnitudes)),
        "min_magnitude_mps2": float(np.min(magnitudes)),
        "max_magnitude_mps2": float(np.max(magnitudes)),
    }


def summarize_imu_streams(streams: IMUStreams) -> dict[str, Any]:
    return {
        "present": True,
        "n_unknown_sensor_type_rows": int(streams.n_unknown_sensor_type_rows),
        "gyro": stream_cadence_stats(streams.gyro),
        "accel": {**stream_cadence_stats(streams.accel), "gravity": accel_gravity_check(streams.accel)},
    }
