#!/usr/bin/env python3
"""One-time D435i factory extrinsics query, run on hardware with the camera attached.

Not runnable in Colab or on a machine without the physical D435i connected --
this queries live device stream profiles via pyrealsense2, it does not read
anything from a recorded session. Run this once per physical camera unit (or
whenever the unit is swapped) and commit the resulting JSON to
config/d435i_factory_extrinsics.json.

See docs/IMU_SPATIOTEMPORAL_CALIBRATION_PLAN.md section 2.1 / 3 / 6 for why
this transform is treated as a fixed known constant rather than solved for:
the targetless calibration only needs to estimate radar->IMU (T_IR); the
D435i's internal depth<->IMU and depth<->color geometry is already resolved
at the factory to sub-mm/sub-degree accuracy and should not be re-estimated.

Usage (on the Jetson, or any host with the D435i attached and pyrealsense2
installed -- on this project's Jetson that means the `mmwave` conda env, not
the broken apt dist-packages install):

    python3 query_d435i_factory_extrinsics.py > d435i_factory_extrinsics.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pyrealsense2 as rs


def _extrinsics_to_dict(ext) -> dict:
    # librealsense's rs2_extrinsics.rotation is a COLUMN-MAJOR flattened 3x3
    # matrix -- NOT row-major like a naive reshape would assume. Keep the raw
    # flat list here rather than reshaping in this script, so a reshape bug
    # here can't silently transpose the calibration for every downstream
    # consumer. Any consumer must do the column-major reshape explicitly:
    #   R[i][j] = rotation_flat_column_major[j * 3 + i]
    return {
        "rotation_flat_column_major": list(ext.rotation),
        "translation": list(ext.translation),
    }


def query_factory_extrinsics() -> dict:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found -- is the D435i attached?")

    dev = devices[0]
    device_name = dev.get_info(rs.camera_info.name)
    device_serial = dev.get_info(rs.camera_info.serial_number)
    firmware_version = dev.get_info(rs.camera_info.firmware_version)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)
    try:
        depth_profile = profile.get_stream(rs.stream.depth)
        color_profile = profile.get_stream(rs.stream.color)
        accel_profile = profile.get_stream(rs.stream.accel)
        gyro_profile = profile.get_stream(rs.stream.gyro)

        return {
            "device_name": device_name,
            "device_serial": device_serial,
            "firmware_version": firmware_version,
            "queried_at_utc": datetime.now(timezone.utc).isoformat(),
            "T_depth_to_accel": _extrinsics_to_dict(depth_profile.get_extrinsics_to(accel_profile)),
            "T_depth_to_gyro": _extrinsics_to_dict(depth_profile.get_extrinsics_to(gyro_profile)),
            "T_depth_to_color": _extrinsics_to_dict(depth_profile.get_extrinsics_to(color_profile)),
            "T_accel_to_depth": _extrinsics_to_dict(accel_profile.get_extrinsics_to(depth_profile)),
            "T_gyro_to_depth": _extrinsics_to_dict(gyro_profile.get_extrinsics_to(depth_profile)),
            "source": "pyrealsense2 get_extrinsics_to(), queried directly from physical device",
        }
    finally:
        pipeline.stop()


def main() -> int:
    try:
        result = query_factory_extrinsics()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
