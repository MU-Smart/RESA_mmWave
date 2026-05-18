"""Runtime helpers for hybrid point-cloud + RD patch inference."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
IO_ROOT = THIS_FILE.parents[2]
for path in (IO_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Lazy imports for local radar decoding modules. These used to come from the
# external CALIBRATION tree; keep this path self-contained for Colab.
_iter_tdm_mimo_frames = None
_parse_mmwave_cfg = None
_compute_range_doppler_products = None


def _ensure_radar_imports():
    global _iter_tdm_mimo_frames, _parse_mmwave_cfg, _compute_range_doppler_products
    if _iter_tdm_mimo_frames is not None:
        return
    try:
        from perception.adc_to_pointcloud_v6 import (
            compute_range_doppler_products as _crdp,
            iter_tdm_mimo_frames as _itf,
            parse_mmwave_cfg as _pmc,
        )
        _iter_tdm_mimo_frames = _itf
        _parse_mmwave_cfg = _pmc
        _compute_range_doppler_products = _crdp
    except ImportError:
        import importlib
        try:
            _iter_tdm_mimo_frames = importlib.import_module(
                "adc_to_pointcloud_v6").iter_tdm_mimo_frames
            _parse_mmwave_cfg = importlib.import_module(
                "adc_to_pointcloud_v6").parse_mmwave_cfg
            _compute_range_doppler_products = importlib.import_module(
                "adc_to_pointcloud_v6").compute_range_doppler_products
        except ImportError:
            raise ImportError(
                "Cannot import local adc_to_pointcloud_v6 radar helpers. Ensure canon/perception "
                "or canon is in PYTHONPATH before exporting RD sidecars."
            )


@dataclass(frozen=True)
class RuntimeRDPatchConfig:
    doppler_bins: int = 17
    range_bins: int = 7
    frame_number_offset: int = 0
    sidecar_subdir: str = "hybrid_rd"
    clutter_mode: str = "off"
    log_scale: bool = True
    include_rd_cube: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.doppler_bins), int(self.range_bins))


@dataclass(frozen=True)
class RuntimeRAPatchConfig:
    azimuth_bins: int = 9
    range_bins: int = 7
    frame_number_offset: int = 0
    source: str = "true_ra"
    az_mode: str = "tx0_only"
    az_fft_size: int = 64
    tensor_filename_suffix: str = "_radar_tensors.npz"
    power_key_prefix: str = "pw_"
    log_scale: bool = True
    normalize_by_frame_max: bool = True
    row_edge_mode: str = "zero_pad"

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.azimuth_bins), int(self.range_bins))


def export_session_sidecars(
    session_dir: str | Path,
    *,
    cfg_path: str | Path,
    config: RuntimeRDPatchConfig,
    overwrite: bool = False,
) -> Path:
    """Export live RD power sidecars for one navigation session if needed."""

    session_path = Path(session_dir).resolve()
    bin_path = session_path / f"{session_path.name}.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Missing radar bin for RD sidecars: {bin_path}")

    out_dir = session_path / config.sidecar_subdir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not config.include_rd_cube or bool(manifest.get("include_rd_cube", False)):
                return out_dir
        except Exception:
            pass

    _ensure_radar_imports()
    cfg = _parse_mmwave_cfg(cfg_path)
    manifest_frames: list[dict[str, object]] = []
    n_exported = 0
    for frame in _iter_tdm_mimo_frames(bin_path, cfg):
        frame_index = int(frame.raw_frame.frame_index)
        out_path = frames_dir / f"rd_frame_{frame_index:06d}.npz"
        if out_path.exists() and not overwrite:
            if not config.include_rd_cube:
                manifest_frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ms": int(frame.raw_frame.timestamp_ms),
                        "path": str(out_path.relative_to(out_dir)),
                    }
                )
                continue
            try:
                with np.load(out_path) as payload:
                    has_cube = "rd_cube" in payload
                if has_cube:
                    manifest_frames.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": int(frame.raw_frame.timestamp_ms),
                            "path": str(out_path.relative_to(out_dir)),
                        }
                    )
                    continue
            except Exception:
                pass
        products = _compute_range_doppler_products(
            frame.tx_data,
            cfg=cfg,
            clutter_mode=config.clutter_mode,
        )
        rd_power = np.asarray(products.power_map, dtype=np.float32)
        if config.log_scale:
            rd_power = np.log1p(rd_power).astype(np.float32)
        doppler_bins, range_bins = rd_power.shape
        range_axis_m = np.arange(range_bins, dtype=np.float32) * float(cfg.range_resolution_m)
        doppler_axis_mps = (
            (np.arange(doppler_bins, dtype=np.float32) - (doppler_bins // 2))
            * float(cfg.velocity_resolution_mps)
        )
        payload = {
            "rd_power": rd_power,
            "range_axis_m": range_axis_m,
            "doppler_axis_mps": doppler_axis_mps,
            "frame_index": np.asarray(frame_index, dtype=np.int32),
            "radar_frame_num": np.asarray(frame_index + config.frame_number_offset, dtype=np.int32),
            "timestamp_ms": np.asarray(frame.raw_frame.timestamp_ms, dtype=np.int64),
        }
        if config.include_rd_cube:
            payload["rd_cube"] = np.asarray(products.rd_cube, dtype=np.complex64)
        np.savez_compressed(out_path, **payload)
        manifest_frames.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": int(frame.raw_frame.timestamp_ms),
                "path": str(out_path.relative_to(out_dir)),
            }
        )
        n_exported += 1

    manifest = {
        "kind": "hybrid_rd_sidecars",
        "session_name": session_path.name,
        "cfg_path": str(cfg_path),
        "output_subdir": config.sidecar_subdir,
        "clutter_mode": config.clutter_mode,
        "log_scale": bool(config.log_scale),
        "frame_number_offset": int(config.frame_number_offset),
        "include_rd_cube": bool(config.include_rd_cube),
        "n_frames": len(manifest_frames),
        "n_frames_exported_this_run": int(n_exported),
        "frames": manifest_frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def _extract_patch(rd_power: np.ndarray, doppler_bin: int, range_bin: int, config: RuntimeRDPatchConfig) -> np.ndarray:
    rd = np.asarray(rd_power, dtype=np.float32)
    if rd.ndim != 2:
        raise ValueError(f"rd_power must have shape [doppler, range], got {rd.shape}")
    n_doppler, n_range = rd.shape
    d_center = int(doppler_bin)
    r_center = int(range_bin)
    if not (0 <= d_center < n_doppler):
        raise IndexError(f"doppler_bin {d_center} outside [0, {n_doppler})")
    if not (0 <= r_center < n_range):
        raise IndexError(f"range_bin {r_center} outside [0, {n_range})")
    d_half = config.doppler_bins // 2
    r_half = config.range_bins // 2
    d_idx = (np.arange(d_center - d_half, d_center + d_half + 1) % n_doppler).astype(np.int64)
    r_idx = np.arange(r_center - r_half, r_center + r_half + 1, dtype=np.int64)
    patch = np.zeros(config.shape, dtype=np.float32)
    valid_r = (r_idx >= 0) & (r_idx < n_range)
    if np.any(valid_r):
        patch[:, valid_r] = rd[np.ix_(d_idx, r_idx[valid_r])]
    return patch


def load_frame_rd_power(
    session_dir: str | Path,
    frame_num: int,
    *,
    config: RuntimeRDPatchConfig,
) -> np.ndarray:
    frame_index = int(frame_num) - int(config.frame_number_offset)
    frame_path = (
        Path(session_dir)
        / config.sidecar_subdir
        / "frames"
        / f"rd_frame_{frame_index:06d}.npz"
    )
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing RD sidecar frame: {frame_path}")
    with np.load(frame_path) as payload:
        return np.asarray(payload["rd_power"], dtype=np.float32)


def _load_session_tensor_npz(
    session_dir: str | Path,
    *,
    config: RuntimeRAPatchConfig,
    session_tensor_cache: dict[Path, object] | None = None,
):
    session_path = Path(session_dir).resolve()
    tensor_path = session_path / f"{session_path.name}{config.tensor_filename_suffix}"
    if session_tensor_cache is not None and tensor_path in session_tensor_cache:
        payload = session_tensor_cache[tensor_path]
        if payload is None:
            raise FileNotFoundError(f"Missing RA tensor sidecar: {tensor_path}")
        return payload
    if not tensor_path.exists():
        if session_tensor_cache is not None:
            session_tensor_cache[tensor_path] = None
        raise FileNotFoundError(f"Missing RA tensor sidecar: {tensor_path}")
    payload = np.load(tensor_path, allow_pickle=False)
    if session_tensor_cache is not None:
        session_tensor_cache[tensor_path] = payload
    return payload


def load_frame_ra_power(
    session_dir: str | Path,
    frame_num: int,
    *,
    config: RuntimeRAPatchConfig,
    session_tensor_cache: dict[Path, object] | None = None,
) -> np.ndarray:
    """Load the per-frame power map used for point-aligned RA patch extraction."""

    frame_index = int(frame_num) - int(config.frame_number_offset)
    key = f"{config.power_key_prefix}{frame_index}"
    payload = _load_session_tensor_npz(
        session_dir,
        config=config,
        session_tensor_cache=session_tensor_cache,
    )
    if key not in payload:
        raise KeyError(f"RA tensor sidecar does not contain '{key}'.")
    return np.asarray(payload[key], dtype=np.float32)


def load_frame_rd_cube(
    session_dir: str | Path,
    frame_num: int,
    *,
    config: RuntimeRAPatchConfig,
    rd_config: RuntimeRDPatchConfig | None = None,
    session_tensor_cache: dict[Path, object] | None = None,
) -> np.ndarray:
    """Load the complex RD cube used for true beamformed RA patch extraction.

    The archived 3branch path stores per-frame cubes in
    ``{session}/{session}_radar_tensors.npz`` under ``rd_<frame_index>``. Live
    sessions may instead have ``rd_cube`` in the hybrid RD frame sidecar when
    sidecar export was requested with ``include_rd_cube=True``.
    """

    frame_index = int(frame_num) - int(config.frame_number_offset)
    key = f"rd_{frame_index}"
    try:
        payload = _load_session_tensor_npz(
            session_dir,
            config=config,
            session_tensor_cache=session_tensor_cache,
        )
        if key in payload:
            return np.asarray(payload[key], dtype=np.complex64)
    except FileNotFoundError:
        pass

    if rd_config is None:
        raise FileNotFoundError(
            f"Missing legacy RD cube tensor for frame {frame_index}; no RD sidecar fallback was provided."
        )

    frame_path = (
        Path(session_dir)
        / rd_config.sidecar_subdir
        / "frames"
        / f"rd_frame_{int(frame_num) - int(rd_config.frame_number_offset):06d}.npz"
    )
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing RD cube sidecar frame: {frame_path}")
    with np.load(frame_path) as payload:
        if "rd_cube" not in payload:
            raise KeyError(
                f"{frame_path} does not contain rd_cube; export RD sidecars with include_rd_cube=True."
            )
        return np.asarray(payload["rd_cube"], dtype=np.complex64)


def _prepare_patch_power_map(
    power_map: np.ndarray,
    *,
    log_scale: bool,
    normalize_by_frame_max: bool,
) -> np.ndarray:
    power = np.asarray(power_map, dtype=np.float32)
    power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
    if log_scale:
        power = np.log1p(np.maximum(power, 0.0)).astype(np.float32)
    if normalize_by_frame_max:
        max_val = float(np.max(power)) if power.size else 0.0
        if max_val > 1e-6:
            power = (power / max_val).astype(np.float32)
    return power


def _extract_patch_from_map(
    power_map: np.ndarray,
    *,
    row_bin: int,
    range_bin: int,
    row_bins: int,
    range_bins: int,
    row_edge_mode: str = "zero_pad",
) -> np.ndarray:
    power = np.asarray(power_map, dtype=np.float32)
    if power.ndim != 2:
        raise ValueError(f"power_map must have shape [rows, range], got {power.shape}")

    n_rows, n_range = power.shape
    row_center = int(row_bin)
    range_center = int(range_bin)
    if not (0 <= row_center < n_rows):
        raise IndexError(f"row_bin {row_center} outside [0, {n_rows})")
    if not (0 <= range_center < n_range):
        raise IndexError(f"range_bin {range_center} outside [0, {n_range})")

    row_half = int(row_bins) // 2
    range_half = int(range_bins) // 2
    patch = np.zeros((int(row_bins), int(range_bins)), dtype=np.float32)
    range_idx = np.arange(range_center - range_half, range_center + range_half + 1, dtype=np.int64)

    if row_edge_mode == "wrap":
        row_idx = (np.arange(row_center - row_half, row_center + row_half + 1) % n_rows).astype(np.int64)
        valid_range = (range_idx >= 0) & (range_idx < n_range)
        if np.any(valid_range):
            patch[:, valid_range] = power[np.ix_(row_idx, range_idx[valid_range])]
        return patch
    if row_edge_mode != "zero_pad":
        raise ValueError(f"Unsupported row_edge_mode: {row_edge_mode}")

    row_start_src = max(0, row_center - row_half)
    row_end_src = min(n_rows, row_center + row_half + 1)
    range_start_src = max(0, range_center - range_half)
    range_end_src = min(n_range, range_center + range_half + 1)
    row_start_dst = row_start_src - (row_center - row_half)
    range_start_dst = range_start_src - (range_center - range_half)
    if row_end_src > row_start_src and range_end_src > range_start_src:
        patch[
            row_start_dst:row_start_dst + (row_end_src - row_start_src),
            range_start_dst:range_start_dst + (range_end_src - range_start_src),
        ] = power[row_start_src:row_end_src, range_start_src:range_end_src]
    return patch


def extract_ra_patch(
    power_map: np.ndarray,
    *,
    row_bin: int,
    range_bin: int,
    config: RuntimeRAPatchConfig | None = None,
) -> np.ndarray:
    cfg = config or RuntimeRAPatchConfig()
    return _extract_patch_from_map(
        power_map,
        row_bin=row_bin,
        range_bin=range_bin,
        row_bins=cfg.azimuth_bins,
        range_bins=cfg.range_bins,
        row_edge_mode=cfg.row_edge_mode,
    )


def extract_true_ra_patches(
    rd_cube: np.ndarray,
    *,
    doppler_bins: np.ndarray,
    range_bins: np.ndarray,
    azimuth_degs: np.ndarray,
    config: RuntimeRAPatchConfig,
) -> np.ndarray:
    """Extract true beamformed local range-azimuth patches from a complex RD cube.

    This is the archived robust RA formulation: anchor each point at its CFAR
    Doppler/range bin, use the complex antenna snapshot, and beamform a 9-bin
    azimuth neighborhood centered on the point's estimated azimuth.
    """

    cube = np.asarray(rd_cube)
    doppler_bins = np.asarray(doppler_bins, dtype=np.int32)
    range_bins = np.asarray(range_bins, dtype=np.int32)
    azimuth_degs = np.asarray(azimuth_degs, dtype=np.float32)
    n_points = int(len(range_bins))
    patches = np.zeros((n_points, int(config.azimuth_bins), int(config.range_bins)), dtype=np.float32)
    if n_points == 0:
        return patches[:, None, :, :]
    if cube.ndim != 4:
        raise ValueError(f"rd_cube must have shape [tx, doppler, rx, range], got {cube.shape}")

    num_tx, n_doppler, num_rx, n_range = cube.shape
    az_mode = str(config.az_mode)
    if az_mode == "tx0_only":
        tx_ids = [0]
    elif az_mode == "tx1_only":
        tx_ids = [1] if num_tx > 1 else [0]
    elif az_mode == "tx0_tx1":
        tx_ids = [0, 1] if num_tx > 1 else [0]
    else:
        raise ValueError(f"Unsupported RA az_mode: {az_mode}")

    positions: list[float] = []
    element_refs: list[tuple[int, int]] = []
    for tx_id in tx_ids:
        if tx_id >= num_tx:
            continue
        for rx_idx in range(num_rx):
            tx_offset = 0.0
            if az_mode == "tx0_tx1":
                tx_offset = 0.0 if tx_id == 0 else 2.0
            positions.append(tx_offset + 0.5 * rx_idx)
            element_refs.append((tx_id, rx_idx))
    if not element_refs:
        return patches[:, None, :, :]

    positions_arr = np.asarray(positions, dtype=np.float32)
    az_half = int(config.azimuth_bins) // 2
    range_half = int(config.range_bins) // 2
    sin_step = 2.0 / max(float(int(config.az_fft_size) - 1), 1.0)
    az_offsets = (np.arange(int(config.azimuth_bins), dtype=np.float32) - az_half) * sin_step

    for i in range(n_points):
        d_bin = int(doppler_bins[i])
        r_center = int(range_bins[i])
        if d_bin < 0 or d_bin >= n_doppler:
            continue

        center_sin = float(np.sin(np.deg2rad(float(azimuth_degs[i]))))
        local_sin = np.clip(center_sin + az_offsets, -1.0, 1.0)
        steering = np.exp(
            1j * 2.0 * np.pi * positions_arr[:, None] * local_sin[None, :]
        ).astype(np.complex64)

        for j in range(int(config.range_bins)):
            r_bin = r_center + (j - range_half)
            if r_bin < 0 or r_bin >= n_range:
                continue
            snapshot = np.empty((len(element_refs),), dtype=np.complex64)
            for k, (tx_id, rx_idx) in enumerate(element_refs):
                snapshot[k] = cube[tx_id, d_bin, rx_idx, r_bin]
            response = np.abs(snapshot @ steering) ** 2
            patches[i, :, j] = response.astype(np.float32)

    patches = _prepare_patch_power_map(
        patches,
        log_scale=bool(config.log_scale),
        normalize_by_frame_max=bool(config.normalize_by_frame_max),
    )
    return patches[:, None, :, :].astype(np.float32)


def extract_ra_patches_for_frame_df(
    frame_df,
    *,
    session_dir: str | Path,
    frame_num: int,
    config: RuntimeRAPatchConfig,
    rd_config: RuntimeRDPatchConfig | None = None,
    session_tensor_cache: dict[Path, object] | None = None,
) -> np.ndarray:
    """Extract point-aligned RA patches for every row in a frame DataFrame."""

    source = str(config.source).strip().lower()
    if source in {"true_ra", "rd_cube", "beamformed"}:
        missing = [c for c in ("range_bin", "doppler_bin", "azimuth_deg") if c not in frame_df.columns]
        if missing:
            raise ValueError(
                "True RA patch inference requires ADC CSV columns: "
                + ", ".join(missing)
            )
        rd_cube = load_frame_rd_cube(
            session_dir,
            frame_num,
            config=config,
            rd_config=rd_config,
            session_tensor_cache=session_tensor_cache,
        )
        return extract_true_ra_patches(
            rd_cube,
            doppler_bins=frame_df["doppler_bin"].to_numpy(),
            range_bins=frame_df["range_bin"].to_numpy(),
            azimuth_degs=frame_df["azimuth_deg"].to_numpy(),
            config=config,
        )

    if source not in {"pw_proxy", "power_map", "proxy"}:
        raise ValueError(f"Unsupported RA patch source: {config.source}")

    missing = [c for c in ("range_bin", "doppler_bin") if c not in frame_df.columns]
    if missing:
        raise ValueError(
            "Proxy RA patch inference requires ADC CSV columns: "
            + ", ".join(missing)
        )
    power_map = load_frame_ra_power(
        session_dir,
        frame_num,
        config=config,
        session_tensor_cache=session_tensor_cache,
    )
    power_map = _prepare_patch_power_map(
        power_map,
        log_scale=config.log_scale,
        normalize_by_frame_max=config.normalize_by_frame_max,
    )
    patches = np.zeros((len(frame_df), 1, config.azimuth_bins, config.range_bins), dtype=np.float32)
    for out_idx, (_, row) in enumerate(frame_df.iterrows()):
        patches[out_idx, 0] = extract_ra_patch(
            power_map,
            row_bin=int(row["doppler_bin"]),
            range_bin=int(row["range_bin"]),
            config=config,
        )
    return patches


def extract_patches_for_frame_df(
    frame_df,
    *,
    session_dir: str | Path,
    frame_num: int,
    config: RuntimeRDPatchConfig,
) -> np.ndarray:
    """Extract patches for all rows in one frame DataFrame."""

    missing = [c for c in ("range_bin", "doppler_bin") if c not in frame_df.columns]
    if missing:
        raise ValueError(
            "Hybrid RD patch inference requires ADC CSV columns: "
            + ", ".join(missing)
            + ". Enable richer range_bin/doppler_bin export in adc_to_pointcloud_v6.py."
        )
    rd_power = load_frame_rd_power(session_dir, frame_num, config=config)
    patches = np.zeros((len(frame_df), 1, config.doppler_bins, config.range_bins), dtype=np.float32)
    for out_idx, (_, row) in enumerate(frame_df.iterrows()):
        patches[out_idx, 0] = _extract_patch(
            rd_power,
            doppler_bin=int(row["doppler_bin"]),
            range_bin=int(row["range_bin"]),
            config=config,
        )
    return patches


# ---------------------------------------------------------------------------
# RD patch scalar feature computation
# ---------------------------------------------------------------------------


def compute_rd_patch_scalars(
    patch: np.ndarray,
    doppler_axis_mps: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute interpretable scalar features from a single RD patch.

    Args:
        patch: (doppler_bins, range_bins) float32 RD power patch.
        doppler_axis_mps: (doppler_bins,) Doppler axis in m/s. If None,
                          uses bin indices.

    Returns:
        Dict with scalar features: rd_energy, rd_peak_ratio, rd_entropy,
        rd_doppler_spread, rd_doppler_skewness, rd_anisotropy,
        rd_ridge_angle, wheel_sideband_score_local.
    """
    patch = np.asarray(patch, dtype=np.float64)
    if patch.ndim != 2:
        raise ValueError(f"patch must be 2D, got shape {patch.shape}")
    n_doppler, n_range = patch.shape

    # rd_energy: total power
    rd_energy = float(np.sum(patch))

    # rd_peak_ratio: ratio of max to mean
    p_mean = float(np.mean(patch))
    p_max = float(np.max(patch))
    rd_peak_ratio = p_max / max(p_mean, 1e-12)

    # rd_entropy: normalized Shannon entropy
    p_norm = patch / max(patch.sum(), 1e-12)
    p_flat = p_norm.flatten()
    rd_entropy = float(-np.sum(p_flat[p_flat > 0] * np.log(p_flat[p_flat > 0] + 1e-12)) / max(np.log(n_doppler * n_range), 1e-12))

    # Doppler marginal
    doppler_marg = np.sum(patch, axis=1)  # (n_doppler,)
    doppler_marg /= max(doppler_marg.sum(), 1e-12)
    if doppler_axis_mps is not None:
        d_axis = np.asarray(doppler_axis_mps, dtype=np.float64)
    else:
        d_axis = np.arange(n_doppler, dtype=np.float64) - n_doppler // 2
    d_mean = float(np.sum(doppler_marg * d_axis))
    d_var = float(np.sum(doppler_marg * (d_axis - d_mean) ** 2))
    rd_doppler_spread = float(np.sqrt(max(d_var, 0.0)))
    d_skew_num = float(np.sum(doppler_marg * (d_axis - d_mean) ** 3))
    d_skew_den = max(rd_doppler_spread ** 3, 1e-12)
    rd_doppler_skewness = d_skew_num / d_skew_den if np.isfinite(d_skew_num) else 0.0

    # Range marginal
    range_marg = np.sum(patch, axis=0)  # (n_range,)
    range_marg /= max(range_marg.sum(), 1e-12)
    r_axis = np.arange(n_range, dtype=np.float64)
    r_mean = float(np.sum(range_marg * r_axis))
    r_var = float(np.sum(range_marg * (r_axis - r_mean) ** 2))
    r_spread = float(np.sqrt(max(r_var, 0.0)))

    # rd_anisotropy: ratio of Doppler spread to range spread
    rd_anisotropy = rd_doppler_spread / max(r_spread, 1e-12)

    # rd_ridge_angle: approximate ridge orientation via gradient
    # Use Sobel-like finite differences
    gy, gx = np.gradient(patch)
    g_mag = np.sqrt(gx ** 2 + gy ** 2)
    g_angle = np.arctan2(gy, gx)
    # Weighted mean angle (circular mean)
    sin_sum = float(np.sum(g_mag * np.sin(g_angle)))
    cos_sum = float(np.sum(g_mag * np.cos(g_angle)))
    rd_ridge_angle = float(np.arctan2(sin_sum, cos_sum)) if (sin_sum != 0 or cos_sum != 0) else 0.0

    # wheel_sideband_score_local: energy in Doppler sidebands relative to center
    # Sidebands = outer 1/3 of Doppler bins on each side
    d_third = max(1, n_doppler // 3)
    center_band = slice(d_third, n_doppler - d_third)
    sideband_energy = float(np.sum(patch[:d_third, :]) + np.sum(patch[-d_third:, :]))
    total_energy = float(np.sum(patch))
    wheel_sideband_score_local = sideband_energy / max(total_energy, 1e-12)

    return {
        "rd_energy": round(rd_energy, 4),
        "rd_peak_ratio": round(rd_peak_ratio, 4),
        "rd_entropy": round(rd_entropy, 4),
        "rd_doppler_spread": round(rd_doppler_spread, 4),
        "rd_doppler_skewness": round(rd_doppler_skewness, 4),
        "rd_anisotropy": round(rd_anisotropy, 4),
        "rd_ridge_angle": round(rd_ridge_angle, 4),
        "wheel_sideband_score_local": round(wheel_sideband_score_local, 4),
    }


def extract_patches_and_scalars_for_frame_df(
    frame_df,
    *,
    session_dir: str | Path,
    frame_num: int,
    config: RuntimeRDPatchConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Extract patches and scalar features for all rows in one frame DataFrame.

    Returns:
        patches: (N, 1, doppler_bins, range_bins) float32 array.
        scalars: list of N dicts with scalar feature keys.
    """
    missing = [c for c in ("range_bin", "doppler_bin") if c not in frame_df.columns]
    if missing:
        raise ValueError(
            "Hybrid RD patch inference requires ADC CSV columns: "
            + ", ".join(missing)
        )
    rd_power = load_frame_rd_power(session_dir, frame_num, config=config)
    # Load doppler axis from sidecar
    frame_index = int(frame_num) - int(config.frame_number_offset)
    frame_path = (
        Path(session_dir)
        / config.sidecar_subdir
        / "frames"
        / f"rd_frame_{frame_index:06d}.npz"
    )
    doppler_axis = None
    if frame_path.exists():
        with np.load(frame_path) as payload:
            if "doppler_axis_mps" in payload:
                doppler_axis = np.asarray(payload["doppler_axis_mps"], dtype=np.float64)

    n_doppler_full = rd_power.shape[0]
    n = len(frame_df)
    patches = np.zeros((n, 1, config.doppler_bins, config.range_bins), dtype=np.float32)
    scalars: list[dict[str, float]] = []
    d_half = config.doppler_bins // 2
    for out_idx, (_, row) in enumerate(frame_df.iterrows()):
        d_center = int(row["doppler_bin"])
        patch = _extract_patch(
            rd_power,
            doppler_bin=d_center,
            range_bin=int(row["range_bin"]),
            config=config,
        )
        patches[out_idx, 0] = patch
        # Slice the doppler axis to the patch window, respecting circular wrap
        patch_doppler_axis = None
        if doppler_axis is not None:
            d_idx = (np.arange(d_center - d_half, d_center + d_half + 1) % n_doppler_full).astype(np.int64)
            patch_doppler_axis = doppler_axis[d_idx]
        scalars.append(compute_rd_patch_scalars(patch, doppler_axis_mps=patch_doppler_axis))
    return patches, scalars


def extract_rdra_patches_and_scalars_for_frame_df(
    frame_df,
    *,
    session_dir: str | Path,
    frame_num: int,
    rd_config: RuntimeRDPatchConfig,
    ra_config: RuntimeRAPatchConfig | None = None,
    session_tensor_cache: dict[Path, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Extract RD patches, RA patches, and RD scalar summaries for one frame."""

    ra_cfg = ra_config or RuntimeRAPatchConfig(frame_number_offset=rd_config.frame_number_offset)
    rd_patches, scalars = extract_patches_and_scalars_for_frame_df(
        frame_df,
        session_dir=session_dir,
        frame_num=frame_num,
        config=rd_config,
    )
    ra_patches = extract_ra_patches_for_frame_df(
        frame_df,
        session_dir=session_dir,
        frame_num=frame_num,
        config=ra_cfg,
        rd_config=rd_config,
        session_tensor_cache=session_tensor_cache,
    )
    return rd_patches, ra_patches, scalars
