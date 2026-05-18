#!/usr/bin/env python3
"""
adc_to_pointcloud_v6.py
=======================
Convert chunked raw ADC IQ sessions into point-cloud CSVs for navigation.

This module is intentionally self-contained so it can be copied onto the Jetson
next to navigation_loop.py. It exports the API expected by that loop:

    build_processor(cfg_path, pfa, min_snr)
    process_session_fast(processor, session_dir) -> (success, detections, csv_path)

The CSV keeps the v3 compatibility columns while adding detection provenance
needed by the hybrid point-cloud + RD-patch branch:

    range_bin, doppler_bin, radar_frame_num, elevation_deg, power, quality

When NAV_PERCEPTION_MODE=pointcloud_rd_patch or ADC_WRITE_RD_SIDECARS=1 is set,
the converter also writes per-frame RD power sidecars under:

    <session>/hybrid_rd/frames/rd_frame_000000.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


_C_M_S = 299_792_458.0
CHUNK_HDR_FMT = "<QII"
CHUNK_HDR_SZ = struct.calcsize(CHUNK_HDR_FMT)
VALID_CLUTTER_MODES = {"off", "mean-subtract", "zero-doppler"}
VALID_WINDOWS = {"hann", "blackman", "rect", "none", "boxcar"}

DEFAULT_PFA = 1e-2
DEFAULT_MIN_SNR_DB = 3.0
DEFAULT_MIN_RANGE_BIN = 18
DEFAULT_FRAME_NUMBER_OFFSET = 1
DEFAULT_RD_SIDECAR_SUBDIR = "hybrid_rd"

MAX_AZIMUTH_DEG = 60.0
MAX_ELEVATION_DEG = 15.0

CSV_HEADER = [
    "timestamp_us",
    "frame_num",
    "x",
    "y",
    "z",
    "v",
    "snr",
    "noise",
    "range_m",
    "doppler_mps",
    "azimuth_deg",
    "power_snr",
    "elevation_deg",
    "doppler",
    "radar_frame_num",
    "power",
    "quality",
    "range_bin",
    "doppler_bin",
    "chunk_index",
    "file_offset",
    "session",
    "clipped_ratio",
]


def _bit_count(value: int) -> int:
    return int(value).bit_count()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    return _safe_int(raw, default)


def _env_float_or_none(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return _safe_float(raw, 0.0)


@dataclass
class RadarConfig:
    """Parsed TI mmWave waveform and raw frame layout."""

    cfg_path: str | None
    device: str
    start_freq_ghz: float
    idle_time_us: float
    adc_start_time_us: float
    ramp_end_time_us: float
    tx_start_time_us: float
    freq_slope_mhz_us: float
    adc_samples: int
    sample_rate_ksps: float
    rx_mask: int
    tx_mask: int
    num_rx: int
    num_tx: int
    chirp_start_idx: int
    chirp_end_idx: int
    num_loops: int
    num_frames_cfg: int
    frame_period_ms: float
    chirp_tx_order: list[int] = field(default_factory=list)
    chirp_tx_masks: list[int] = field(default_factory=list)
    lvds_stream_cfg: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def sample_rate_hz(self) -> float:
        return self.sample_rate_ksps * 1_000.0

    @property
    def slope_hz_s(self) -> float:
        return self.freq_slope_mhz_us * 1.0e12

    @property
    def center_frequency_hz(self) -> float:
        return self.start_freq_ghz * 1.0e9

    @property
    def wavelength_m(self) -> float:
        if self.center_frequency_hz <= 0.0:
            return float("nan")
        return _C_M_S / self.center_frequency_hz

    @property
    def adc_collection_time_us(self) -> float:
        if self.sample_rate_hz <= 0.0:
            return 0.0
        return (self.adc_samples / self.sample_rate_hz) * 1.0e6

    @property
    def useful_bandwidth_hz(self) -> float:
        return self.slope_hz_s * (self.adc_collection_time_us * 1.0e-6)

    @property
    def chirp_count_per_loop(self) -> int:
        return max(0, self.chirp_end_idx - self.chirp_start_idx + 1)

    @property
    def chirps_per_frame(self) -> int:
        return self.chirp_count_per_loop * self.num_loops

    @property
    def chirp_period_us(self) -> float:
        return self.idle_time_us + self.ramp_end_time_us

    @property
    def same_tx_chirp_period_s(self) -> float:
        stride = self.chirp_count_per_loop if self.chirp_count_per_loop > 0 else self.num_tx
        return self.chirp_period_us * 1.0e-6 * max(stride, 1)

    @property
    def bytes_per_frame(self) -> int:
        return self.chirps_per_frame * self.num_rx * self.adc_samples * 4

    @property
    def range_resolution_m(self) -> float:
        if self.useful_bandwidth_hz <= 0.0:
            return float("nan")
        return _C_M_S / (2.0 * self.useful_bandwidth_hz)

    @property
    def max_range_m(self) -> float:
        if self.sample_rate_hz <= 0.0 or self.slope_hz_s <= 0.0:
            return float("nan")
        return self.sample_rate_hz * _C_M_S / (2.0 * self.slope_hz_s)

    @property
    def velocity_resolution_mps(self) -> float:
        if self.wavelength_m <= 0.0 or self.same_tx_chirp_period_s <= 0.0 or self.num_loops <= 0:
            return float("nan")
        total_observation_s = self.num_loops * self.same_tx_chirp_period_s
        return self.wavelength_m / (2.0 * total_observation_s)

    @property
    def max_unambiguous_velocity_mps(self) -> float:
        if self.wavelength_m <= 0.0 or self.same_tx_chirp_period_s <= 0.0:
            return float("nan")
        return self.wavelength_m / (4.0 * self.same_tx_chirp_period_s)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "sample_rate_hz": self.sample_rate_hz,
                "slope_hz_s": self.slope_hz_s,
                "wavelength_m": self.wavelength_m,
                "adc_collection_time_us": self.adc_collection_time_us,
                "useful_bandwidth_hz": self.useful_bandwidth_hz,
                "chirp_count_per_loop": self.chirp_count_per_loop,
                "chirps_per_frame": self.chirps_per_frame,
                "chirp_period_us": self.chirp_period_us,
                "bytes_per_frame": self.bytes_per_frame,
                "range_resolution_m": self.range_resolution_m,
                "max_range_m": self.max_range_m,
                "velocity_resolution_mps": self.velocity_resolution_mps,
                "max_unambiguous_velocity_mps": self.max_unambiguous_velocity_mps,
            }
        )
        return payload


def parse_mmwave_cfg(cfg_path: str | Path) -> RadarConfig:
    """Parse the TI mmWave cfg used for raw DCA capture."""

    path = Path(cfg_path)
    if not path.exists():
        raise FileNotFoundError(f"CFG not found: {path}")

    channel_cfg: list[str] | None = None
    profile_cfg: list[str] | None = None
    frame_cfg: list[str] | None = None
    lvds_stream_cfg: list[int] = []
    chirp_defs: dict[int, int] = {}
    warnings: list[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            command = parts[0].lower()

            if command == "channelcfg":
                channel_cfg = parts
            elif command == "profilecfg":
                profile_cfg = parts
            elif command == "framecfg":
                frame_cfg = parts
            elif command == "chirpcfg":
                if len(parts) >= 9:
                    chirp_start = _safe_int(parts[1])
                    chirp_end = _safe_int(parts[2])
                    tx_mask = _safe_int(parts[8])
                    for chirp_idx in range(chirp_start, chirp_end + 1):
                        chirp_defs[chirp_idx] = tx_mask
            elif command == "lvdsstreamcfg":
                lvds_stream_cfg = [_safe_int(value) for value in parts[1:]]

    if channel_cfg is None:
        raise ValueError(f"channelCfg missing in {path}")
    if profile_cfg is None:
        raise ValueError(f"profileCfg missing in {path}")
    if frame_cfg is None:
        raise ValueError(f"frameCfg missing in {path}")

    rx_mask = _safe_int(channel_cfg[1])
    tx_mask = _safe_int(channel_cfg[2])
    num_rx = _bit_count(rx_mask)
    num_tx = _bit_count(tx_mask)
    if num_rx <= 0 or num_tx <= 0:
        raise ValueError(f"Invalid channelCfg in {path}: rx_mask={rx_mask}, tx_mask={tx_mask}")

    chirp_start_idx = _safe_int(frame_cfg[1])
    chirp_end_idx = _safe_int(frame_cfg[2])
    chirp_tx_masks: list[int] = []
    chirp_tx_order: list[int] = []
    for chirp_idx in range(chirp_start_idx, chirp_end_idx + 1):
        mask = chirp_defs.get(chirp_idx, 0)
        chirp_tx_masks.append(mask)
        if mask <= 0 or _bit_count(mask) != 1:
            warnings.append(
                f"chirpCfg for chirp {chirp_idx} is missing or not one-hot "
                f"(mask={mask}); using positional TX order."
            )
            chirp_tx_order.append(len(chirp_tx_order) % max(num_tx, 1))
        else:
            chirp_tx_order.append(mask.bit_length() - 1)

    return RadarConfig(
        cfg_path=str(path),
        device="AWR1843",
        start_freq_ghz=_safe_float(profile_cfg[2]),
        idle_time_us=_safe_float(profile_cfg[3]),
        adc_start_time_us=_safe_float(profile_cfg[4]),
        ramp_end_time_us=_safe_float(profile_cfg[5]),
        tx_start_time_us=_safe_float(profile_cfg[9]),
        freq_slope_mhz_us=_safe_float(profile_cfg[8]),
        adc_samples=_safe_int(profile_cfg[10]),
        sample_rate_ksps=_safe_float(profile_cfg[11]),
        rx_mask=rx_mask,
        tx_mask=tx_mask,
        num_rx=num_rx,
        num_tx=num_tx,
        chirp_start_idx=chirp_start_idx,
        chirp_end_idx=chirp_end_idx,
        num_loops=_safe_int(frame_cfg[3]),
        num_frames_cfg=_safe_int(frame_cfg[4]),
        frame_period_ms=_safe_float(frame_cfg[5]),
        chirp_tx_order=chirp_tx_order,
        chirp_tx_masks=chirp_tx_masks,
        lvds_stream_cfg=lvds_stream_cfg,
        warnings=warnings,
    )


@dataclass(slots=True)
class RawRadarFrame:
    frame_index: int
    timestamp_ms: int
    capture_aux: int
    file_offset: int
    chunk_index: int
    data: bytes


@dataclass(slots=True)
class FrameDecodeStats:
    clipped_ratio: float
    iq_mean_real: list[float]
    iq_mean_imag: list[float]
    per_rx_power: list[float]


@dataclass(slots=True)
class TdmMimoFrame:
    raw_frame: RawRadarFrame
    iq: np.ndarray
    tx_data: np.ndarray
    stats: FrameDecodeStats


@dataclass(slots=True)
class RangeDopplerProducts:
    range_fft: np.ndarray
    rd_cube: np.ndarray
    virt_array: np.ndarray
    power_map: np.ndarray


@dataclass(slots=True)
class AngleEstimate:
    azimuth_deg: float
    elevation_deg: float
    quality: float


@dataclass(slots=True)
class DetectionRow:
    timestamp_us: int
    frame_num: int
    range_m: float
    doppler_mps: float
    azimuth_deg: float
    elevation_deg: float
    x: float
    y: float
    z: float
    snr: float
    noise: float
    power: float
    quality: float
    doppler_bin: int
    range_bin: int
    chunk_index: int
    file_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProcessingStats:
    session_dir: Path
    bin_path: Path
    csv_path: Path
    frames_total: int = 0
    frames_empty: int = 0
    frame_errors: int = 0
    detections_total: int = 0
    sidecar_dir: Path | None = None
    error: str = ""

    @property
    def success(self) -> bool:
        return self.frames_total > 0 and not self.error


def _looks_like_chunked(first_header: bytes, file_size: int, *, max_chunk_bytes: int) -> bool:
    if len(first_header) != CHUNK_HDR_SZ:
        return False
    try:
        _, chunk_len, _ = struct.unpack(CHUNK_HDR_FMT, first_header)
    except struct.error:
        return False
    return 0 < chunk_len <= max_chunk_bytes and CHUNK_HDR_SZ + chunk_len <= file_size


def iter_raw_radar_frames(
    bin_path: str | Path,
    *,
    cfg: RadarConfig,
    split_multiples: bool = True,
    max_chunk_bytes: int = 16 * 1024 * 1024,
) -> Iterator[RawRadarFrame]:
    """Yield raw frames from chunked recorder output or contiguous raw files."""

    path = Path(bin_path)
    file_size = path.stat().st_size
    frame_bytes = cfg.bytes_per_frame
    frame_index = 0

    with path.open("rb") as handle:
        first_header = handle.read(CHUNK_HDR_SZ)
        handle.seek(0)
        if not _looks_like_chunked(first_header, file_size, max_chunk_bytes=max_chunk_bytes):
            if frame_bytes <= 0 or file_size % frame_bytes != 0:
                raise ValueError(
                    f"{path.name} is neither chunked nor a whole number of "
                    f"{frame_bytes}-byte raw frames"
                )
            while True:
                file_offset = handle.tell()
                data = handle.read(frame_bytes)
                if not data:
                    break
                if len(data) != frame_bytes:
                    raise ValueError(f"Truncated raw frame at offset {file_offset} in {path.name}")
                yield RawRadarFrame(
                    frame_index=frame_index,
                    timestamp_ms=int(round(frame_index * cfg.frame_period_ms)),
                    capture_aux=0,
                    file_offset=int(file_offset),
                    chunk_index=frame_index,
                    data=data,
                )
                frame_index += 1
            return

        chunk_index = 0
        while True:
            file_offset = handle.tell()
            header = handle.read(CHUNK_HDR_SZ)
            if not header:
                break
            if len(header) != CHUNK_HDR_SZ:
                raise ValueError(f"Truncated chunk header at offset {file_offset} in {path.name}")
            timestamp_ms, chunk_len, capture_aux = struct.unpack(CHUNK_HDR_FMT, header)
            if chunk_len <= 0 or chunk_len > max_chunk_bytes:
                raise ValueError(f"Invalid chunk length {chunk_len} at offset {file_offset} in {path.name}")
            data = handle.read(chunk_len)
            if len(data) != chunk_len:
                raise ValueError(f"Truncated chunk payload at offset {file_offset} in {path.name}")

            if chunk_len == frame_bytes:
                yield RawRadarFrame(
                    frame_index=frame_index,
                    timestamp_ms=int(timestamp_ms),
                    capture_aux=int(capture_aux),
                    file_offset=int(file_offset),
                    chunk_index=int(chunk_index),
                    data=data,
                )
                frame_index += 1
            elif split_multiples and frame_bytes > 0 and chunk_len % frame_bytes == 0:
                count = chunk_len // frame_bytes
                for idx in range(count):
                    start = idx * frame_bytes
                    end = start + frame_bytes
                    yield RawRadarFrame(
                        frame_index=frame_index,
                        timestamp_ms=int(timestamp_ms),
                        capture_aux=int(capture_aux),
                        file_offset=int(file_offset + CHUNK_HDR_SZ + start),
                        chunk_index=int(chunk_index),
                        data=data[start:end],
                    )
                    frame_index += 1
            else:
                raise ValueError(
                    f"Chunk {chunk_index} in {path.name} has length {chunk_len}, "
                    f"expected {frame_bytes}"
                )
            chunk_index += 1


def decode_complex_iq(raw_bytes: bytes, cfg: RadarConfig) -> tuple[np.ndarray, FrameDecodeStats]:
    """Decode interleaved int16 IQ into [chirp, rx, sample] complex data."""

    raw = np.frombuffer(raw_bytes, dtype=np.int16)
    expected_i16 = cfg.chirps_per_frame * cfg.num_rx * cfg.adc_samples * 2
    if raw.size != expected_i16:
        raise ValueError(f"Expected {expected_i16} int16 values, got {raw.size}")

    clipped = (raw == np.iinfo(np.int16).max) | (raw == np.iinfo(np.int16).min)
    iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    iq = iq.reshape(cfg.chirps_per_frame, cfg.num_rx, cfg.adc_samples)
    per_rx_power = np.mean(np.abs(iq) ** 2, axis=(0, 2))
    stats = FrameDecodeStats(
        clipped_ratio=float(np.mean(clipped)),
        iq_mean_real=[float(x) for x in np.mean(iq.real, axis=(0, 2))],
        iq_mean_imag=[float(x) for x in np.mean(iq.imag, axis=(0, 2))],
        per_rx_power=[float(x) for x in per_rx_power],
    )
    return iq, stats


def demux_tdm_mimo(iq: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Group chirps into [tx, chirps_per_tx, rx, sample]."""

    chirp_order = cfg.chirp_tx_order or list(range(cfg.chirp_count_per_loop))
    if len(chirp_order) != cfg.chirp_count_per_loop:
        raise ValueError("chirp_tx_order length does not match chirp_count_per_loop")
    if iq.shape[0] != cfg.num_loops * len(chirp_order):
        raise ValueError("Frame chirp count does not match cfg-derived loop layout")

    loop_view = iq.reshape(cfg.num_loops, len(chirp_order), cfg.num_rx, cfg.adc_samples)
    tx_blocks: list[np.ndarray] = []
    expected_chirps_per_tx: int | None = None
    for tx_idx in range(cfg.num_tx):
        positions = [i for i, tx in enumerate(chirp_order) if tx == tx_idx]
        if not positions:
            raise ValueError(f"TX{tx_idx} missing from chirp order {chirp_order}")
        block = loop_view[:, positions, :, :].reshape(-1, cfg.num_rx, cfg.adc_samples)
        expected_chirps_per_tx = expected_chirps_per_tx or block.shape[0]
        if block.shape[0] != expected_chirps_per_tx:
            raise ValueError("Uneven chirp counts per TX are not supported")
        tx_blocks.append(block)
    return np.stack(tx_blocks, axis=0)


def iter_tdm_mimo_frames(bin_path: str | Path, cfg: RadarConfig) -> Iterator[TdmMimoFrame]:
    for raw_frame in iter_raw_radar_frames(bin_path, cfg=cfg, split_multiples=True):
        iq, stats = decode_complex_iq(raw_frame.data, cfg)
        tx_data = demux_tdm_mimo(iq, cfg)
        yield TdmMimoFrame(raw_frame=raw_frame, iq=iq, tx_data=tx_data, stats=stats)


def choose_window(name: str, size: int) -> np.ndarray:
    window = name.strip().lower()
    if window == "hann":
        return np.hanning(size).astype(np.float32)
    if window == "blackman":
        return np.blackman(size).astype(np.float32)
    if window in {"rect", "none", "boxcar"}:
        return np.ones(size, dtype=np.float32)
    raise ValueError(f"Unsupported FFT window: {name}")


def range_doppler_cube(
    tx_data: np.ndarray,
    *,
    cfg: RadarConfig,
    range_window: str = "hann",
    doppler_window: str = "hann",
    clutter_mode: str = "off",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute range FFT and Doppler FFT products."""

    if clutter_mode not in VALID_CLUTTER_MODES:
        raise ValueError(f"Unsupported clutter_mode={clutter_mode!r}")

    data = tx_data - np.mean(tx_data, axis=-1, keepdims=True)
    data = data * choose_window(range_window, cfg.adc_samples)[None, None, None, :]
    range_fft = np.fft.fft(data, n=cfg.adc_samples, axis=-1)

    if clutter_mode == "mean-subtract":
        range_fft = range_fft - np.mean(range_fft, axis=1, keepdims=True)

    rd_cube = range_fft * choose_window(doppler_window, tx_data.shape[1])[None, :, None, None]
    rd_cube = np.fft.fft(rd_cube, n=tx_data.shape[1], axis=1)
    rd_cube = np.fft.fftshift(rd_cube, axes=1)
    if clutter_mode == "zero-doppler":
        rd_cube[:, rd_cube.shape[1] // 2, :, :] = 0
    return range_fft, rd_cube


def build_virtual_array(rd_cube: np.ndarray) -> np.ndarray:
    """Flatten TX/RX into [virtual_antenna, doppler, range]."""

    return rd_cube.transpose(0, 2, 1, 3).reshape(
        rd_cube.shape[0] * rd_cube.shape[2],
        rd_cube.shape[1],
        rd_cube.shape[3],
    )


def build_power_map(virt_array: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(virt_array) ** 2, axis=0)


def compute_range_doppler_products(
    tx_data: np.ndarray,
    *,
    cfg: RadarConfig,
    range_window: str = "hann",
    doppler_window: str = "hann",
    clutter_mode: str = "off",
) -> RangeDopplerProducts:
    range_fft, rd_cube = range_doppler_cube(
        tx_data,
        cfg=cfg,
        range_window=range_window,
        doppler_window=doppler_window,
        clutter_mode=clutter_mode,
    )
    virt_array = build_virtual_array(rd_cube)
    power_map = build_power_map(virt_array)
    return RangeDopplerProducts(
        range_fft=range_fft,
        rd_cube=rd_cube,
        virt_array=virt_array,
        power_map=power_map,
    )


def cfar_2d(
    power_map: np.ndarray,
    *,
    pfa: float,
    min_snr_db: float,
    min_range_bin: int = DEFAULT_MIN_RANGE_BIN,
) -> list[tuple[int, int, float, float, float]]:
    """Apply CA-CFAR over [doppler, range] power."""

    guard_d, guard_r = 2, 4
    train_d, train_r = 4, 8
    pad_d = guard_d + train_d
    pad_r = guard_r + train_r
    n_dopp, n_range = power_map.shape
    n_train = (2 * pad_d + 1) * (2 * pad_r + 1) - (2 * guard_d + 1) * (2 * guard_r + 1)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    wrapped = np.pad(power_map, ((pad_d, pad_d), (0, 0)), mode="wrap")

    detections: list[tuple[int, int, float, float, float]] = []
    for di in range(n_dopp):
        local_d = wrapped[di : di + 2 * pad_d + 1]
        for ri in range(max(min_range_bin, pad_r), n_range - pad_r):
            local = local_d[:, ri - pad_r : ri + pad_r + 1]
            mask = np.ones(local.shape, dtype=bool)
            d0 = pad_d - guard_d
            d1 = pad_d + guard_d + 1
            r0 = pad_r - guard_r
            r1 = pad_r + guard_r + 1
            mask[d0:d1, r0:r1] = False
            training = local[mask]
            noise = float(np.mean(training))
            if noise <= 0.0:
                continue
            power = float(power_map[di, ri])
            threshold = alpha * noise
            if power <= threshold:
                continue
            snr_db = 10.0 * math.log10(max(power / noise, 1e-12))
            if snr_db >= min_snr_db:
                detections.append((di, ri, snr_db, noise, power))
    return detections


def estimate_aoa_fft(virt_array: np.ndarray, doppler_bin: int, range_bin: int, cfg: RadarConfig) -> AngleEstimate:
    """Estimate azimuth and elevation with simple FFT beamforming."""

    if cfg.num_rx < 4 or cfg.num_tx < 3 or virt_array.shape[0] < cfg.num_rx * 3:
        return AngleEstimate(azimuth_deg=0.0, elevation_deg=0.0, quality=0.25)

    steering = virt_array[:, doppler_bin, range_bin]
    tx0 = steering[0 : cfg.num_rx]
    tx1 = steering[cfg.num_rx : 2 * cfg.num_rx]
    tx2 = steering[2 * cfg.num_rx : 3 * cfg.num_rx]
    az_elements = np.concatenate([tx0, tx2])

    fft_size = 64
    az_fft = np.fft.fftshift(np.fft.fft(az_elements, n=fft_size))
    az_power = np.abs(az_fft) ** 2
    peak_bin = int(np.argmax(az_power)) - fft_size // 2
    sin_az = float(np.clip(2.0 * peak_bin / fft_size, -1.0, 1.0))
    azimuth_deg = float(np.degrees(np.arcsin(sin_az)))

    phase_diff = float(np.angle(tx1[0] * np.conj(tx0[0]))) if abs(tx0[0]) > 0 and abs(tx1[0]) > 0 else 0.0
    sin_el = float(np.clip(phase_diff / (4.0 * np.pi), -1.0, 1.0))
    elevation_deg = float(np.degrees(np.arcsin(sin_el)))

    confidence = float(np.max(az_power) / max(np.mean(az_power), 1e-9))
    quality = float(np.clip((confidence - 1.0) / 10.0, 0.0, 1.0))
    return AngleEstimate(
        azimuth_deg=float(np.clip(azimuth_deg, -MAX_AZIMUTH_DEG, MAX_AZIMUTH_DEG)),
        elevation_deg=float(np.clip(elevation_deg, -MAX_ELEVATION_DEG, MAX_ELEVATION_DEG)),
        quality=quality,
    )


def spherical_to_cartesian(range_m: float, azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    az_rad = math.radians(azimuth_deg)
    el_rad = math.radians(elevation_deg)
    cos_el = math.cos(el_rad)
    x = range_m * math.sin(az_rad) * cos_el
    y = range_m * math.cos(az_rad) * cos_el
    z = range_m * math.sin(el_rad)
    return float(x), float(y), float(z)


def detections_to_rows(
    *,
    detections: Iterable[tuple[int, int, float, float, float]],
    virt_array: np.ndarray,
    cfg: RadarConfig,
    frame_index: int,
    timestamp_us: int,
    chunk_index: int,
    file_offset: int,
    enable_aoa: bool = True,
) -> list[DetectionRow]:
    rows: list[DetectionRow] = []
    doppler_center = virt_array.shape[1] // 2
    for doppler_bin, range_bin, snr_db, noise, power in detections:
        range_m = float(range_bin * cfg.range_resolution_m)
        doppler_mps = float((doppler_bin - doppler_center) * cfg.velocity_resolution_mps)
        if enable_aoa:
            angle = estimate_aoa_fft(virt_array, doppler_bin, range_bin, cfg)
        else:
            angle = AngleEstimate(azimuth_deg=0.0, elevation_deg=0.0, quality=0.5)

        x, y, z = spherical_to_cartesian(range_m, angle.azimuth_deg, angle.elevation_deg)
        snr_quality = float(np.clip((snr_db - 3.0) / 15.0, 0.0, 1.0))
        quality = float(np.clip(0.5 * angle.quality + 0.5 * snr_quality, 0.0, 1.0))
        rows.append(
            DetectionRow(
                timestamp_us=int(timestamp_us),
                frame_num=int(frame_index),
                range_m=round(range_m, 5),
                doppler_mps=round(doppler_mps, 5),
                azimuth_deg=round(float(angle.azimuth_deg), 4),
                elevation_deg=round(float(angle.elevation_deg), 4),
                x=round(x, 5),
                y=round(y, 5),
                z=round(z, 5),
                snr=round(float(snr_db), 3),
                noise=round(float(noise), 6),
                power=round(float(power), 6),
                quality=round(quality, 4),
                doppler_bin=int(doppler_bin),
                range_bin=int(range_bin),
                chunk_index=int(chunk_index),
                file_offset=int(file_offset),
            )
        )
    return rows


@dataclass
class RadarProcessor:
    cfg_path: str | Path
    cfg: RadarConfig
    pfa: float = DEFAULT_PFA
    min_snr: float = DEFAULT_MIN_SNR_DB
    min_range_bin: int = DEFAULT_MIN_RANGE_BIN
    clutter_mode: str = "off"
    range_window: str = "hann"
    doppler_window: str = "hann"
    enable_aoa: bool = True
    max_range_m: float | None = None
    max_abs_azimuth_deg: float | None = None
    frame_number_offset: int = DEFAULT_FRAME_NUMBER_OFFSET
    output_name: str | None = None
    write_rd_sidecars: bool = False
    rd_sidecar_subdir: str = DEFAULT_RD_SIDECAR_SUBDIR
    rd_sidecar_clutter_mode: str = "off"
    rd_log_scale: bool = True

    def __post_init__(self) -> None:
        self.clutter_mode = self.clutter_mode.strip().lower()
        self.rd_sidecar_clutter_mode = self.rd_sidecar_clutter_mode.strip().lower()
        self.range_window = self.range_window.strip().lower()
        self.doppler_window = self.doppler_window.strip().lower()
        if self.clutter_mode not in VALID_CLUTTER_MODES:
            raise ValueError(f"Unsupported clutter_mode={self.clutter_mode!r}")
        if self.rd_sidecar_clutter_mode not in VALID_CLUTTER_MODES:
            raise ValueError(f"Unsupported rd_sidecar_clutter_mode={self.rd_sidecar_clutter_mode!r}")
        if self.range_window not in VALID_WINDOWS:
            raise ValueError(f"Unsupported range_window={self.range_window!r}")
        if self.doppler_window not in VALID_WINDOWS:
            raise ValueError(f"Unsupported doppler_window={self.doppler_window!r}")

    def compute_products(self, tx_data: np.ndarray, *, clutter_mode: str | None = None) -> RangeDopplerProducts:
        return compute_range_doppler_products(
            tx_data,
            cfg=self.cfg,
            range_window=self.range_window,
            doppler_window=self.doppler_window,
            clutter_mode=clutter_mode or self.clutter_mode,
        )

    def process_frame(self, frame: TdmMimoFrame) -> tuple[list[DetectionRow], RangeDopplerProducts]:
        products = self.compute_products(frame.tx_data)
        detections = cfar_2d(
            products.power_map,
            pfa=float(self.pfa),
            min_snr_db=float(self.min_snr),
            min_range_bin=int(self.min_range_bin),
        )
        rows = detections_to_rows(
            detections=detections,
            virt_array=products.virt_array,
            cfg=self.cfg,
            frame_index=int(frame.raw_frame.frame_index),
            timestamp_us=int(frame.raw_frame.timestamp_ms) * 1000,
            chunk_index=int(frame.raw_frame.chunk_index),
            file_offset=int(frame.raw_frame.file_offset),
            enable_aoa=bool(self.enable_aoa),
        )
        rows = self._filter_rows(rows)
        return rows, products

    def _filter_rows(self, rows: list[DetectionRow]) -> list[DetectionRow]:
        if self.max_range_m is None and self.max_abs_azimuth_deg is None:
            return rows
        kept: list[DetectionRow] = []
        for row in rows:
            if self.max_range_m is not None and row.range_m > self.max_range_m:
                continue
            if self.max_abs_azimuth_deg is not None and abs(row.azimuth_deg) > self.max_abs_azimuth_deg:
                continue
            kept.append(row)
        return kept


def build_processor(
    cfg_path: str | Path,
    pfa: float = DEFAULT_PFA,
    min_snr: float = DEFAULT_MIN_SNR_DB,
    **kwargs: Any,
) -> RadarProcessor:
    """Build a reusable processor for navigation_loop.py."""

    nav_mode = os.environ.get("NAV_PERCEPTION_MODE", "").strip().lower()
    write_sidecars_default = nav_mode == "pointcloud_rd_patch" or _env_bool("ADC_WRITE_RD_SIDECARS", False)
    frame_offset_default = _env_int(
        "NAV_POINTCLOUD_RD_FRAME_OFFSET",
        _env_int("ADC_FRAME_NUMBER_OFFSET", DEFAULT_FRAME_NUMBER_OFFSET),
    )

    options: dict[str, Any] = {
        "min_range_bin": _env_int("ADC_MIN_RANGE_BIN", DEFAULT_MIN_RANGE_BIN),
        "clutter_mode": os.environ.get("ADC_CLUTTER_MODE", "off"),
        "range_window": os.environ.get("ADC_RANGE_WINDOW", "hann"),
        "doppler_window": os.environ.get("ADC_DOPPLER_WINDOW", "hann"),
        "enable_aoa": not _env_bool("ADC_DISABLE_AOA", False),
        "max_range_m": _env_float_or_none("ADC_MAX_RANGE_M"),
        "max_abs_azimuth_deg": _env_float_or_none("ADC_MAX_ABS_AZIMUTH_DEG"),
        "frame_number_offset": frame_offset_default,
        "output_name": os.environ.get("ADC_OUTPUT_NAME") or None,
        "write_rd_sidecars": write_sidecars_default,
        "rd_sidecar_subdir": os.environ.get("NAV_POINTCLOUD_RD_PATCH_SUBDIR", DEFAULT_RD_SIDECAR_SUBDIR),
        "rd_sidecar_clutter_mode": os.environ.get(
            "NAV_POINTCLOUD_RD_CLUTTER_MODE",
            os.environ.get("ADC_CLUTTER_MODE", "off"),
        ),
        "rd_log_scale": not _env_bool("NAV_POINTCLOUD_RD_LINEAR_POWER", False),
    }
    options.update(kwargs)
    cfg = parse_mmwave_cfg(cfg_path)
    return RadarProcessor(cfg_path=cfg_path, cfg=cfg, pfa=float(pfa), min_snr=float(min_snr), **options)


def find_session_bin(session_dir: str | Path) -> Path:
    session_path = Path(session_dir)
    preferred = session_path / f"{session_path.name}.bin"
    if preferred.exists():
        return preferred
    candidates = sorted(p for p in session_path.glob("*.bin") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .bin file found in {session_path}")
    names = ", ".join(p.name for p in candidates[:5])
    raise FileNotFoundError(f"Multiple .bin files found in {session_path}; expected {preferred.name}. Found: {names}")


def output_csv_path(processor: RadarProcessor, session_dir: str | Path) -> Path:
    session_path = Path(session_dir)
    name = processor.output_name or f"{session_path.name}.csv"
    return session_path / name


def row_to_csv_dict(
    row: DetectionRow,
    *,
    session_name: str,
    frame_number_offset: int,
    clipped_ratio: float,
) -> dict[str, Any]:
    payload = row.as_dict()
    payload["v"] = row.doppler_mps
    payload["doppler"] = row.doppler_mps
    payload["power_snr"] = row.snr
    payload["radar_frame_num"] = int(row.frame_num) + int(frame_number_offset)
    payload["session"] = session_name
    payload["clipped_ratio"] = round(float(clipped_ratio), 6)
    return {key: payload.get(key, "") for key in CSV_HEADER}


def _axis_payload(cfg: RadarConfig, doppler_bins: int, range_bins: int) -> tuple[np.ndarray, np.ndarray]:
    range_axis_m = np.arange(range_bins, dtype=np.float32) * float(cfg.range_resolution_m)
    doppler_axis_mps = (
        (np.arange(doppler_bins, dtype=np.float32) - (doppler_bins // 2))
        * float(cfg.velocity_resolution_mps)
    )
    return range_axis_m, doppler_axis_mps


def _write_rd_sidecar(
    processor: RadarProcessor,
    *,
    session_dir: Path,
    frame: TdmMimoFrame,
    products: RangeDopplerProducts,
    frames_dir: Path,
) -> dict[str, object]:
    frame_index = int(frame.raw_frame.frame_index)
    out_path = frames_dir / f"rd_frame_{frame_index:06d}.npz"

    if processor.rd_sidecar_clutter_mode == processor.clutter_mode:
        rd_power = np.asarray(products.power_map, dtype=np.float32)
    else:
        sidecar_products = processor.compute_products(
            frame.tx_data,
            clutter_mode=processor.rd_sidecar_clutter_mode,
        )
        rd_power = np.asarray(sidecar_products.power_map, dtype=np.float32)

    if processor.rd_log_scale:
        rd_power = np.log1p(rd_power).astype(np.float32)
    doppler_bins, range_bins = rd_power.shape
    range_axis_m, doppler_axis_mps = _axis_payload(processor.cfg, doppler_bins, range_bins)

    np.savez_compressed(
        out_path,
        rd_power=rd_power,
        range_axis_m=range_axis_m,
        doppler_axis_mps=doppler_axis_mps,
        frame_index=np.asarray(frame_index, dtype=np.int32),
        radar_frame_num=np.asarray(frame_index + processor.frame_number_offset, dtype=np.int32),
        timestamp_ms=np.asarray(frame.raw_frame.timestamp_ms, dtype=np.int64),
    )
    return {
        "frame_index": frame_index,
        "radar_frame_num": int(frame_index + processor.frame_number_offset),
        "timestamp_ms": int(frame.raw_frame.timestamp_ms),
        "path": str(out_path.relative_to(session_dir / processor.rd_sidecar_subdir)),
    }


def _write_sidecar_manifest(
    processor: RadarProcessor,
    *,
    session_dir: Path,
    bin_path: Path,
    out_dir: Path,
    frames: list[Mapping[str, object]],
) -> None:
    manifest = {
        "kind": "hybrid_rd_sidecars",
        "writer": "adc_to_pointcloud_v6",
        "session_name": session_dir.name,
        "bin_path": str(bin_path),
        "cfg_path": str(processor.cfg_path),
        "output_subdir": processor.rd_sidecar_subdir,
        "clutter_mode": processor.rd_sidecar_clutter_mode,
        "log_scale": bool(processor.rd_log_scale),
        "frame_number_offset": int(processor.frame_number_offset),
        "range_resolution_m": float(processor.cfg.range_resolution_m),
        "velocity_resolution_mps": float(processor.cfg.velocity_resolution_mps),
        "n_frames": len(frames),
        "n_frames_exported_this_run": len(frames),
        "frames": list(frames),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _count_existing_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def process_session(
    processor: RadarProcessor,
    session_dir: str | Path,
    *,
    force: bool = False,
    quiet: bool = False,
) -> ProcessingStats:
    """Process one session and return detailed conversion stats."""

    session_path = Path(session_dir)
    bin_path = find_session_bin(session_path)
    csv_path = output_csv_path(processor, session_path)
    stats = ProcessingStats(session_dir=session_path, bin_path=bin_path, csv_path=csv_path)

    if csv_path.exists() and not force:
        stats.detections_total = _count_existing_rows(csv_path)
        stats.frames_total = 1 if stats.detections_total >= 0 else 0
        return stats

    sidecar_frames: list[Mapping[str, object]] = []
    sidecar_out_dir: Path | None = None
    sidecar_frames_dir: Path | None = None
    if processor.write_rd_sidecars:
        sidecar_out_dir = session_path / processor.rd_sidecar_subdir
        sidecar_frames_dir = sidecar_out_dir / "frames"
        sidecar_frames_dir.mkdir(parents=True, exist_ok=True)
        stats.sidecar_dir = sidecar_out_dir

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()

        for frame in iter_tdm_mimo_frames(bin_path, processor.cfg):
            try:
                rows, products = processor.process_frame(frame)
                if sidecar_frames_dir is not None:
                    sidecar_frames.append(
                        _write_rd_sidecar(
                            processor,
                            session_dir=session_path,
                            frame=frame,
                            products=products,
                            frames_dir=sidecar_frames_dir,
                        )
                    )
            except Exception as exc:
                stats.frame_errors += 1
                if not quiet:
                    print(f"[WARN] {session_path.name} frame {frame.raw_frame.frame_index}: {exc}", file=sys.stderr)
                continue

            if rows:
                writer.writerows(
                    row_to_csv_dict(
                        row,
                        session_name=session_path.name,
                        frame_number_offset=processor.frame_number_offset,
                        clipped_ratio=frame.stats.clipped_ratio,
                    )
                    for row in rows
                )
                stats.detections_total += len(rows)
            else:
                stats.frames_empty += 1

            stats.frames_total += 1
            if not quiet and stats.frames_total % 50 == 0:
                print(
                    f"  {session_path.name}: {stats.frames_total} frames, "
                    f"{stats.detections_total} detections",
                    flush=True,
                )

    if sidecar_out_dir is not None:
        _write_sidecar_manifest(
            processor,
            session_dir=session_path,
            bin_path=bin_path,
            out_dir=sidecar_out_dir,
            frames=sidecar_frames,
        )

    if stats.frames_total <= 0:
        stats.error = f"No frames decoded from {bin_path}"
    return stats


def process_session_fast(processor: RadarProcessor, session_dir: str | Path) -> tuple[bool, int, str]:
    """Navigation-loop wrapper. Always rewrites the CSV for the fresh session."""

    try:
        stats = process_session(processor, session_dir, force=True, quiet=True)
    except Exception as exc:
        return False, 0, str(exc)
    if not stats.success:
        return False, int(stats.detections_total), stats.error or str(stats.csv_path)
    return True, int(stats.detections_total), str(stats.csv_path)


def _iter_session_dirs(root: Path | None, sessions: list[Path]) -> list[Path]:
    out = [p.resolve() for p in sessions]
    if root is not None:
        out.extend(p.resolve() for p in sorted(root.iterdir()) if p.is_dir())
    deduped: dict[str, Path] = {}
    for path in out:
        deduped[str(path)] = path
    return list(deduped.values())


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cfg", required=True, type=Path, help="TI mmWave cfg used for collection.")
    parser.add_argument("--force", action="store_true", help="Reprocess even if the CSV already exists.")
    parser.add_argument("--pfa", type=float, default=DEFAULT_PFA, help=f"CFAR PFA. Default: {DEFAULT_PFA:g}.")
    parser.add_argument("--min-snr", type=float, default=DEFAULT_MIN_SNR_DB, help="Minimum CFAR SNR in dB.")
    parser.add_argument("--min-range-bin", type=int, default=DEFAULT_MIN_RANGE_BIN, help="Minimum range bin for CFAR.")
    parser.add_argument("--clutter-mode", choices=sorted(VALID_CLUTTER_MODES), default="off")
    parser.add_argument("--range-window", choices=sorted(VALID_WINDOWS), default="hann")
    parser.add_argument("--doppler-window", choices=sorted(VALID_WINDOWS), default="hann")
    parser.add_argument("--no-aoa", action="store_true", help="Disable angle estimation and emit boresight points.")
    parser.add_argument("--max-range-m", type=float, default=None, help="Optional post-CFAR range gate.")
    parser.add_argument("--max-abs-azimuth-deg", type=float, default=None, help="Optional post-CFAR azimuth gate.")
    parser.add_argument(
        "--frame-number-offset",
        type=int,
        default=DEFAULT_FRAME_NUMBER_OFFSET,
        help="radar_frame_num = raw frame_index + offset. Default keeps sidecar frame_index zero-based.",
    )
    parser.add_argument("--output-name", default=None, help="Output CSV filename. Default: <session>.csv.")
    parser.add_argument("--write-rd-sidecars", action="store_true", help="Write hybrid-RD per-frame sidecars.")
    parser.add_argument("--rd-sidecar-subdir", default=DEFAULT_RD_SIDECAR_SUBDIR)
    parser.add_argument("--rd-clutter-mode", choices=sorted(VALID_CLUTTER_MODES), default=None)
    parser.add_argument("--linear-rd-power", action="store_true", help="Store linear RD power instead of log1p(power).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw ADC IQ session bins to navigation point-cloud CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", action="append", type=Path, default=[], help="One session directory.")
    group.add_argument("--root", type=Path, help="Root whose child directories are sessions.")
    _add_common_args(parser)
    args = parser.parse_args(argv)

    processor = build_processor(
        cfg_path=args.cfg,
        pfa=args.pfa,
        min_snr=args.min_snr,
        min_range_bin=args.min_range_bin,
        clutter_mode=args.clutter_mode,
        range_window=args.range_window,
        doppler_window=args.doppler_window,
        enable_aoa=not args.no_aoa,
        max_range_m=args.max_range_m,
        max_abs_azimuth_deg=args.max_abs_azimuth_deg,
        frame_number_offset=args.frame_number_offset,
        output_name=args.output_name,
        write_rd_sidecars=args.write_rd_sidecars,
        rd_sidecar_subdir=args.rd_sidecar_subdir,
        rd_sidecar_clutter_mode=args.rd_clutter_mode or args.clutter_mode,
        rd_log_scale=not args.linear_rd_power,
    )

    sessions = _iter_session_dirs(args.root, args.session)
    if not sessions:
        raise SystemExit("No sessions found.")

    print("adc_to_pointcloud_v6 - Raw ADC -> point cloud")
    print(f"  cfg:              {args.cfg}")
    print(f"  range_res_m:      {processor.cfg.range_resolution_m:.5f}")
    print(f"  velocity_res_mps: {processor.cfg.velocity_resolution_mps:.5f}")
    print(f"  bytes_per_frame:  {processor.cfg.bytes_per_frame}")
    print(f"  sidecars:         {processor.write_rd_sidecars}")

    failed = 0
    total_detections = 0
    for session_dir in sessions:
        print(f"\n[SESSION] {session_dir.name}")
        try:
            stats = process_session(processor, session_dir, force=args.force, quiet=False)
        except Exception as exc:
            failed += 1
            print(f"  [ERROR] {exc}")
            continue
        total_detections += stats.detections_total
        if not stats.success:
            failed += 1
            print(f"  [ERROR] {stats.error}")
            continue
        print(f"  frames:     {stats.frames_total}")
        print(f"  empty:      {stats.frames_empty}")
        print(f"  errors:     {stats.frame_errors}")
        print(f"  detections: {stats.detections_total}")
        print(f"  csv:        {stats.csv_path}")
        if stats.sidecar_dir is not None:
            print(f"  sidecars:   {stats.sidecar_dir}")

    print(f"\nDone. sessions={len(sessions)} failed={failed} detections={total_detections}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
