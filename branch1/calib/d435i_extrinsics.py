#!/usr/bin/env python3
"""Loader for config/d435i_factory_extrinsics.json.

Applies the correct COLUMN-MAJOR reshape of librealsense's flat rotation
arrays (see that file's "_rotation_convention" note).

Deliberately does NOT read meta_data.json's realsense_calibration
depth_to_color_extrinsics / color_to_depth_extrinsics for anything
precision-sensitive. Byte-for-byte comparison (2026-09-04) showed
meta_data.json's stored rotation is the naive ROW-major reshape of the same
underlying flat array returned by a live pyrealsense2 query -- i.e. it is
the TRANSPOSE of the true rotation, unless the recorder applied the
column-major fix before writing JSON (its source was not available to
confirm either way). The discrepancy is small (~1.5 degrees of effective
rotation error) but real, and this module avoids depending on it. See
docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md open items for the full writeup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_FACTORY_EXTRINSICS_PATH = Path(__file__).resolve().parents[2] / "config" / "d435i_factory_extrinsics.json"


def _reshape_column_major(flat9: list) -> np.ndarray:
    return np.asarray(flat9, dtype=np.float64).reshape(3, 3, order="F")


class D435iFactoryExtrinsics:
    __slots__ = (
        "R_depth_to_accel",
        "t_depth_to_accel",
        "R_depth_to_gyro",
        "t_depth_to_gyro",
        "R_depth_to_color",
        "t_depth_to_color",
        "device_serial",
    )

    def __init__(self, payload: dict) -> None:
        self.R_depth_to_accel = _reshape_column_major(payload["T_depth_to_accel"]["rotation_flat_column_major"])
        self.t_depth_to_accel = np.asarray(payload["T_depth_to_accel"]["translation"], dtype=np.float64)
        self.R_depth_to_gyro = _reshape_column_major(payload["T_depth_to_gyro"]["rotation_flat_column_major"])
        self.t_depth_to_gyro = np.asarray(payload["T_depth_to_gyro"]["translation"], dtype=np.float64)
        self.R_depth_to_color = _reshape_column_major(payload["T_depth_to_color"]["rotation_flat_column_major"])
        self.t_depth_to_color = np.asarray(payload["T_depth_to_color"]["translation"], dtype=np.float64)
        self.device_serial = payload.get("device_serial")


def load_d435i_factory_extrinsics(path: Path | None = None) -> D435iFactoryExtrinsics | None:
    path = path or DEFAULT_FACTORY_EXTRINSICS_PATH
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return D435iFactoryExtrinsics(payload)
