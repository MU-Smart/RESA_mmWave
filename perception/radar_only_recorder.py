"""
radar_only_recorder.py
======================
Deployment recorder — radar only, no camera, runs continuously until Ctrl+C.

Collects raw ADC frames from the DCA1000EVM ethernet stream and writes
short rolling .bin session files. Each completed session is placed in a
queue for the processing loop to pick up. After processing, everything
is deleted.

This is a stripped version of dca_realsense_recorder_v8_train.py with
all RealSense / camera / preview / color writer code removed.

Configuration
-------------
Edit the constants below before running on the Jetson:
    OUTPUT_ROOT      — where session folders are written
    MMWAVE_CFG_PATH  — path to profile_objdet.cfg
    SESSION_DURATION_S — how many seconds of radar per session file
                         Shorter = lower latency to inference.
                         Longer  = more temporal context per file.
                         Recommended: 2-5s for navigation.

Usage
-----
    python3 radar_only_recorder.py

Ctrl+C stops cleanly — the current session is finalized before exit.
"""

import csv
import json
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from nav_logger import get_logger
log = get_logger("Recorder")

# =============================================================================
# CONFIGURATION — edit these
# =============================================================================

OUTPUT_ROOT        = "/home/ryan/nav_sessions/"
MMWAVE_CFG_PATH    = "/home/ryan/xwr/profile_objdet.cfg"
SESSION_DURATION_S = 1.5       # seconds of radar per session file

# Real-time bias: keep only a bounded amount of unread raw radar frames.
# If the writer falls behind, older frames are discarded in favor of newer ones.
RAW_FRAME_QUEUE_MAX = 64

# Radar warmup — discard first N frames before recording
# (radar needs time to stabilize after start)
RADAR_WARMUP_FRAMES    = 15
RADAR_WARMUP_TIMEOUT_S = 20.0

# xwr / DCA1000 network config — must match your hardware
XWR_CONFIG = {
    "radar": {
        "device":         "AWR1843",
        "port":           None,
        "frequency":      77.0,
        "idle_time":      7.0,
        "adc_start_time": 3.0,
        "ramp_end_time":  39.0,
        "tx_start_time":  1.0,
        "freq_slope":     100.0,
        "adc_samples":    256,
        "sample_rate":    7200,
        "frame_length":   32,
        "frame_period":   100.0,
    },
    "capture": {
        "sys_ip":        "192.168.33.30",
        "fpga_ip":       "192.168.33.180",
        "socket_buffer": 26214400,
    },
}

BIN_FORMAT_VERSION = 3

# RealSense D435i RGB intrinsics at the recording resolution.
# Calibrated at 1280x720 via opencv_intrinsic.py; K pre-scaled by 0.5 for 640x360.
# dist_coeffs are dimensionless and unchanged by resolution scaling.
# Update these if the camera or recording resolution changes.
CAMERA_RGB_WIDTH  = 640
CAMERA_RGB_HEIGHT = 360
CAMERA_RGB_INTRINSICS = {
    "K": [
        [681.133, 0.0,     371.843],
        [0.0,     679.599, 115.254],
        [0.0,     0.0,     1.0    ],
    ],
    "fx":          681.133,
    "fy":          679.599,
    "cx":          371.843,
    "cy":          115.254,
    "dist_coeffs": [-0.5132266, 6.0887782, -0.0036907, 0.0161005, -35.0970872],
    "width":       CAMERA_RGB_WIDTH,
    "height":      CAMERA_RGB_HEIGHT,
}


# =============================================================================
# Monotonic epoch clock (same as recorder v8)
# =============================================================================

_PERF_EPOCH_NS = time.perf_counter_ns()
_WALL_EPOCH_MS = int(time.time() * 1000)


def mono_epoch_ms() -> int:
    return _WALL_EPOCH_MS + (time.perf_counter_ns() - _PERF_EPOCH_NS) // 1_000_000


# =============================================================================
# System setup
# =============================================================================

def ensure_rmem_max(target: int = 26_214_400) -> None:
    try:
        r = subprocess.run(
            ["sysctl", "-w",
             f"net.core.rmem_max={target}",
             f"net.core.rmem_default={target}"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            print(f"[SYS] rmem_max set to {target:,}")
            return
        print(f"[SYS] sysctl failed: {r.stderr.strip()}")
    except Exception as e:
        print(f"[SYS] Could not set rmem_max: {e}")


def parse_framecfg_period_ms(cfg_path: str) -> float:
    try:
        with open(cfg_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('%', '#')):
                    continue
                if line.lower().startswith("framecfg"):
                    parts = line.split()
                    if len(parts) >= 6:
                        return float(parts[5])
    except Exception as e:
        print(f"[CFG] Could not parse frameCfg period: {e}")
    return 100.0  # default 10Hz


# =============================================================================
# DCA radar reader thread (identical to recorder v8)
# =============================================================================

class DCARadarReader(threading.Thread):

    def __init__(self, xwr_config: dict):
        super().__init__(daemon=True, name="DCARadarReader")
        self.xwr_config   = xwr_config
        self.q            = queue.Queue(maxsize=RAW_FRAME_QUEUE_MAX)
        self._stop_event  = threading.Event()
        self.error        = None
        self.total_frames = 0

    def run(self):
        try:
            import xwr
            log.info("Initializing DCA radar stream...")
            awr = xwr.XWRSystem(**self.xwr_config)
            log.info("Radar stream started.")
            for frame in awr.dstream(numpy=True):
                if self._stop_event.is_set():
                    break
                ts_ms = mono_epoch_ms()
                raw   = frame.tobytes()
                if self.q.full():
                    dropped = 0
                    while self.q.full():
                        try:
                            self.q.get_nowait()
                            dropped += 1
                        except queue.Empty:
                            break
                    if dropped:
                        log.warning("Dropping stale radar frames", dropped=dropped, queue_depth=self.q.qsize())
                try:
                    self.q.put_nowait((ts_ms, raw))
                except queue.Full:
                    log.warning("Skipping newest radar frame after drop attempt", queue_depth=self.q.qsize())
                    continue
                self.total_frames += 1
            awr.stop()
            log.info(f"Radar stream stopped.", total_frames=self.total_frames)
        except ImportError:
            self.error = "xwr not installed. pip install xwr --break-system-packages"
            log.radar_error(self.error)
        except Exception as e:
            self.error = str(e)
            log.radar_error(str(e))

    def stop(self):
        self._stop_event.set()


# =============================================================================
# Session writer — writes one .bin + radar_timestamps.csv per session
# =============================================================================

class SessionWriter:
    """
    Manages writing a single session's .bin and timestamp CSV.
    Call open_session() to start, close_session() to finalize.
    """

    def __init__(self, output_root: str):
        self.output_root   = output_root
        self.session_dir   = None
        self.session_name  = None
        self.bin_f         = None
        self.ts_f          = None
        self.ts_writer     = None
        self.frame_count   = 0
        self.byte_count    = 0
        self.start_ms      = 0

    def open_session(self) -> str:
        """Open a new session. Returns the session directory path."""
        time_str          = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_name = f"session_{time_str}"
        self.session_dir  = os.path.join(self.output_root, self.session_name)
        stem_path         = os.path.join(self.session_dir, self.session_name)

        os.makedirs(self.session_dir, exist_ok=True)

        self.bin_f     = open(stem_path + ".bin", "wb")
        self.ts_f      = open(stem_path + "_radar_timestamps.csv", "w", newline="")
        self.ts_writer = csv.writer(self.ts_f)
        self.ts_writer.writerow(["frame_index", "timestamp_ms"])

        self.frame_count = 0
        self.byte_count  = 0
        self.start_ms    = 0

        return self.session_dir

    def write_frame(self, ts_ms: int, raw: bytes) -> None:
        if self.start_ms == 0:
            self.start_ms = ts_ms

        # Write binary header + raw data (same format as recorder v8)
        self.bin_f.write(struct.pack('<QII', ts_ms, len(raw), 0))
        self.bin_f.write(raw)
        self.ts_writer.writerow([self.frame_count, ts_ms])

        self.frame_count += 1
        self.byte_count  += len(raw)

    def close_session(self) -> dict:
        """Flush and close files. Returns session metadata dict."""
        if self.bin_f:
            self.bin_f.close()
        if self.ts_f:
            self.ts_f.flush()
            self.ts_f.close()

        meta = {
            "session":            self.session_name,
            "bin_format_version": BIN_FORMAT_VERSION,
            "recorder":           "radar_only_recorder.py",
            "radar_mode":         "objdet",
            "radar_cfg_path":     MMWAVE_CFG_PATH,
            "radar": {
                "start_ms":     self.start_ms,
                "total_bytes":  self.byte_count,
                "total_frames": self.frame_count,
            },
            "realsense_calibration": {
                "rgb": CAMERA_RGB_INTRINSICS,
            },
        }

        meta_path = os.path.join(self.session_dir, "meta_data.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return meta


# =============================================================================
# Main continuous recorder loop
# =============================================================================

def record_continuous(ready_queue: queue.Queue, stop_event: threading.Event) -> None:
    """
    Continuously records radar sessions of SESSION_DURATION_S seconds each.
    Completed session directories are pushed to ready_queue for processing.
    Stops cleanly when stop_event is set.

    Args:
        ready_queue:  Queue shared with the processing loop.
                      Each item is a session directory path (str).
        stop_event:   Set by Ctrl+C handler to stop the loop.
    """
    ensure_rmem_max(26_214_400)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    radar_frame_period_ms = parse_framecfg_period_ms(MMWAVE_CFG_PATH)
    radar_hz = 1000.0 / radar_frame_period_ms if radar_frame_period_ms > 0 else 10.0
    expected_frames_per_session = int(SESSION_DURATION_S * radar_hz)

    log.info(
        f"Recorder starting",
        radar_hz=f"{radar_hz:.1f}",
        session_s=SESSION_DURATION_S,
        expected_frames=expected_frames_per_session,
        output=OUTPUT_ROOT,
    )

    # --- Start radar reader thread ---
    radar_reader = DCARadarReader(XWR_CONFIG)
    radar_reader.start()

    if radar_reader.error:
        log.radar_error(f"Radar failed to start: {radar_reader.error}")
        raise SystemExit(f"[ERROR] Radar failed to start: {radar_reader.error}")

    # --- Warmup: drain first N frames ---
    log.info(f"Warming up radar", warmup_frames=RADAR_WARMUP_FRAMES)
    warmup_count = 0
    warmup_start = time.monotonic()

    while warmup_count < RADAR_WARMUP_FRAMES:
        if time.monotonic() - warmup_start > RADAR_WARMUP_TIMEOUT_S:
            log.warning(f"Warmup timeout", frames_received=warmup_count)
            break
        if radar_reader.error:
            log.radar_error(f"Error during warmup: {radar_reader.error}")
            raise SystemExit(f"[ERROR] Radar error during warmup: {radar_reader.error}")
        try:
            radar_reader.q.get_nowait()
            warmup_count += 1
        except queue.Empty:
            time.sleep(0.01)

    log.warmup_done(warmup_count, time.monotonic() - warmup_start)

    writer         = SessionWriter(OUTPUT_ROOT)
    session_start  = time.monotonic()
    last_health_t  = time.monotonic()
    total_sessions = 0

    writer.open_session()
    log.session_start(writer.session_name)

    try:
        while not stop_event.is_set():

            if radar_reader.error:
                log.radar_error(radar_reader.error)
                break

            # Drain all available frames into current session
            while not radar_reader.q.empty():
                try:
                    ts_ms, raw = radar_reader.q.get_nowait()
                    writer.write_frame(ts_ms, raw)
                except queue.Empty:
                    break

            # Health report every 10s
            now = time.monotonic()
            if now - last_health_t >= 10.0:
                log.radar_health(
                    session_name=writer.session_name,
                    frames=writer.frame_count,
                    elapsed_s=now - session_start,
                    queue_depth=radar_reader.q.qsize(),
                )
                last_health_t = now

            # Roll session when duration reached
            if time.monotonic() - session_start >= SESSION_DURATION_S:
                duration = time.monotonic() - session_start
                meta = writer.close_session()
                log.session_end(
                    session_name=writer.session_name,
                    frames=meta["radar"]["total_frames"],
                    byte_count=meta["radar"]["total_bytes"],
                    duration_s=duration,
                )
                session_dir = writer.session_dir
                total_sessions += 1

                while True:
                    try:
                        stale_session = ready_queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        log.warning("Dropping stale completed session", session=stale_session)
                        try:
                            import shutil
                            shutil.rmtree(stale_session, ignore_errors=True)
                        except Exception as e:
                            log.warning("Failed to remove stale session", session=stale_session, error=str(e))

                try:
                    ready_queue.put_nowait(session_dir)
                except queue.Full:
                    log.warning("Ready queue still full; dropping newest session", session=session_dir)
                    try:
                        import shutil
                        shutil.rmtree(session_dir, ignore_errors=True)
                    except Exception as e:
                        log.warning("Failed to remove dropped newest session", session=session_dir, error=str(e))

                writer.open_session()
                session_start = time.monotonic()
                log.session_start(writer.session_name)

            time.sleep(0.005)

    finally:
        if writer.frame_count > 0:
            duration = time.monotonic() - session_start
            meta = writer.close_session()
            log.session_end(
                session_name=writer.session_name,
                frames=meta["radar"]["total_frames"],
                byte_count=meta["radar"]["total_bytes"],
                duration_s=duration,
            )
            while True:
                try:
                    stale_session = ready_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    log.warning("Dropping stale completed session", session=stale_session)
                    try:
                        import shutil
                        shutil.rmtree(stale_session, ignore_errors=True)
                    except Exception as e:
                        log.warning("Failed to remove stale session", session=stale_session, error=str(e))
            try:
                ready_queue.put_nowait(writer.session_dir)
            except queue.Full:
                log.warning("Ready queue still full at shutdown; dropping newest session", session=writer.session_dir)
                try:
                    import shutil
                    shutil.rmtree(writer.session_dir, ignore_errors=True)
                except Exception as e:
                    log.warning("Failed to remove dropped shutdown session", session=writer.session_dir, error=str(e))

        radar_reader.stop()
        radar_reader.join(timeout=5.0)
        log.info(f"Recorder stopped", total_sessions=total_sessions + 1)
