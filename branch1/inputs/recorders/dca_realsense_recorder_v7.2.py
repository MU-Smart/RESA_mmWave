"""
dca_realsense_recorder_v7.py
============================
Single recorder for both calibration and object detection sessions.

Fixes vs v6
-----------
[FIX — PREVIEW DISPLAY NOT WORKING ON UBUNTU/LINUX]
  On Ubuntu/Linux, cv2.imshow fails silently or crashes with a Qt/GTK
  display error under several common conditions:

  1. No display server — script run over plain SSH (no -X forwarding),
     inside tmux/screen on a headless machine, or as a sudo process
     whose environment does not have DISPLAY/WAYLAND_DISPLAY set.
     cv2.imshow then throws:
       "QXcbConnection: Could not connect to display"
     or a segfault, killing the entire recording.

  2. OpenCV headless build — pip may install opencv-python-headless as a
     transitive dependency. System OpenCV (apt install python3-opencv)
     may be compiled without Qt/GTK HighGUI support. In both cases imshow
     raises cv2.error immediately.

  3. cv2.namedWindow must be called before the recording loop on Linux
     with certain backends (GTK2/GTK3). Without a namedWindow call the
     first imshow initializes the display inside the capture loop, which
     can cause a 100-500ms stall on frame 1 that the timestamp domain
     check misinterprets.

  Fix:
    a) Probe display availability at startup with check_display_available()
       before starting any hardware. On Linux this reads $DISPLAY and
       $WAYLAND_DISPLAY. If neither is set, preview is auto-disabled
       regardless of SHOW_PREVIEW.

    b) Probe OpenCV HighGUI support with probe_cv2_display(). Creates a
       1x1 test window, calls waitKey(1), destroys it. If any cv2.error
       is raised, preview is auto-disabled. Uses a 2-second timeout via
       a subprocess so a hanging Qt init does not block the main script.

    c) cv2.namedWindow is called once before the capture loop so the
       window is ready without a first-frame stall.

    d) All imshow/waitKey calls are wrapped in try/except so a runtime
       display loss (e.g. screen-lock, remote session disconnect) never
       aborts the recording — it just disables the preview and logs once.

[CONFIRMED WORKING from v6]
  - 0% radar frame drop (warmup + rmem_max fix)
  - 30.3 fps camera (threaded ColorWriter + DepthWriter + USB 3)
  - _stop_flag naming (no TypeError on join)
  - rmem_max set programmatically before xwr opens socket
  - Split ColorWriter / DepthWriter threads

Usage
-----
  python3 dca_realsense_recorder_v7.py

  Set MMWAVE_CFG_PATH to profile_objdet.cfg for both calibration and
  object detection. adc_to_pointcloud_v2.py handles both use cases.
"""

import csv
import json
import os
import queue
import signal
import shutil
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# FOV overlay for preview window
try:
    from fov_overlay import draw_fov_overlay_simple
    _HAS_FOV_OVERLAY = True
except ImportError:
    _HAS_FOV_OVERLAY = False

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_ROOT          = "./dataNoah/Processing/"
_REPO_ROOT = Path(__file__).resolve().parents[3]

MMWAVE_CFG_PATH      = os.environ.get("MMWAVE_CFG_PATH", str(_REPO_ROOT / "config" / "profile_objdet.cfg"))

# Continuous stop-go capture. Each individual output session records for
# SESSION_RECORDING_DURATION_S, and the recorder keeps starting new sessions
# until OVERALL_RECORDING_DURATION_S is reached or the user stops it.
SESSION_RECORDING_DURATION_S = 15.0
OVERALL_RECORDING_DURATION_S = 300.0

# Backward-compatible alias for older local edits that may import this constant.
RECORDING_DURATION_S = SESSION_RECORDING_DURATION_S

# Audible session markers: short beep at session start, long beep at session stop.
BEEP_ENABLED       = True
BEEP_FREQUENCY_HZ  = 880
SHORT_BEEP_MS      = 120
LONG_BEEP_MS       = 600

# Radar warmup
RADAR_WARMUP_FRAMES    = 15
RADAR_WARMUP_TIMEOUT_S = 20.0

# RealSense
RS_WIDTH      = 1280
RS_HEIGHT     = 720
RS_FPS        = 30

# Preview display.
# Set to True to attempt to show a live camera window.
# On Linux/Ubuntu the script will automatically disable the preview if:
#   - No DISPLAY or WAYLAND_DISPLAY environment variable is set (e.g. plain SSH)
#   - OpenCV was compiled without HighGUI support (headless build)
#   - The display probe fails for any other reason
# The recording is never aborted due to display issues.
SHOW_PREVIEW  = True

# Set False to skip depth visualization video (saves CPU + I/O in DepthWriter)
SAVE_DEPTH_VIS = True

# Writer queue capacities (frames). Sized independently.
COLOR_QUEUE_MAX = 120
DEPTH_QUEUE_MAX = 120

# xwr / DCA1000 configuration
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

# =============================================================================
# Monotonic epoch clock
# =============================================================================

_PERF_EPOCH_NS = time.perf_counter_ns()
_WALL_EPOCH_MS = int(time.time() * 1000)


def mono_epoch_ms() -> int:
    return _WALL_EPOCH_MS + (time.perf_counter_ns() - _PERF_EPOCH_NS) // 1_000_000


def now_ms() -> int:
    return int(time.time() * 1000)


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
            print(f"[SYS] rmem_max/default set to {target:,}")
            return
        print(f"[SYS] sysctl -w failed: {r.stderr.strip()}")
    except Exception as e:
        print(f"[SYS] Could not run sysctl: {e}")
    try:
        current = int(open("/proc/sys/net/core/rmem_max").read().strip())
        status = "OK" if current >= target else f"WARNING: need {target:,}"
        print(f"[SYS] rmem_max={current:,}  {status}")
    except Exception:
        pass


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
        print(f"[CFG] Could not parse frameCfg: {e}")
    return 0.0


def infer_radar_mode(cfg_path: str) -> str:
    try:
        with open(cfg_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(('%', '#')):
                    continue
                if line.lower().startswith("lvdsstream"):
                    parts = line.split()
                    if len(parts) >= 4:
                        return "calib" if int(parts[3]) == 1 else "objdet"
    except Exception:
        pass
    return "calib" if "calib" in os.path.basename(cfg_path).lower() else "objdet"


def emit_beep(duration_ms: int) -> None:
    if not BEEP_ENABLED:
        return
    beep_bin = shutil.which("beep")
    if beep_bin:
        try:
            result = subprocess.run(
                [beep_bin, "-f", str(BEEP_FREQUENCY_HZ), "-l", str(duration_ms)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, duration_ms / 1000.0 + 0.5),
            )
            if result.returncode == 0:
                return
        except Exception:
            pass

    # Portable fallback: terminal bell. Duration cannot be controlled reliably,
    # so the long beep is represented as multiple short bells.
    repeats = 1 if duration_ms <= SHORT_BEEP_MS else 3
    for i in range(repeats):
        sys.stdout.write("\a")
        sys.stdout.flush()
        if i + 1 < repeats:
            time.sleep(0.18)


def beep_session_start() -> None:
    emit_beep(SHORT_BEEP_MS)


def beep_session_stop() -> None:
    emit_beep(LONG_BEEP_MS)


def discard_pending_radar_frames(radar_reader: "DCARadarReader") -> int:
    discarded = 0
    while not radar_reader.q.empty():
        try:
            radar_reader.q.get_nowait()
            discarded += 1
        except queue.Empty:
            break
    return discarded


# =============================================================================
# Display availability probing (Linux/Ubuntu fix)
# =============================================================================

def check_display_available() -> bool:
    """
    Return True if a graphical display server appears to be reachable.

    On Linux, cv2.imshow requires either:
      - X11:     $DISPLAY is set (e.g. ':0', ':1', 'localhost:10.0')
      - Wayland: $WAYLAND_DISPLAY is set (e.g. 'wayland-0')

    If neither variable is present the process has no display and imshow
    will crash with a Qt/GTK connection error.  This is the common case
    when the script runs over plain SSH without -X forwarding.
    """
    display     = os.environ.get("DISPLAY", "")
    wayland     = os.environ.get("WAYLAND_DISPLAY", "")
    has_display = bool(display or wayland)
    if not has_display:
        print("[Preview] No DISPLAY or WAYLAND_DISPLAY — preview auto-disabled.")
    return has_display


def probe_cv2_display(timeout_s: float = 3.0) -> bool:
    """
    Confirm that cv2.imshow actually works by creating a 1x1 test window
    in a subprocess.  Returns True if the probe succeeded.

    Running the probe in a subprocess is essential: on some Ubuntu
    configurations (GTK2/GTK3 backend, missing libGL) the Qt/GTK
    initialisation either hangs for 10+ seconds or segfaults the
    interpreter.  A subprocess isolates that failure so it never
    affects the main recording process.
    """
    probe_code = (
        "import cv2, numpy as np, sys;"
        "img = np.zeros((1,1,3), dtype='uint8');"
        "cv2.namedWindow('_probe', cv2.WINDOW_NORMAL);"
        "cv2.imshow('_probe', img);"
        "cv2.waitKey(1);"
        "cv2.destroyAllWindows();"
        "sys.exit(0)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            timeout=timeout_s,
            capture_output=True,
            text=True,
            env=os.environ,   # forward DISPLAY / WAYLAND_DISPLAY to child
        )
        if result.returncode == 0:
            return True
        stderr = result.stderr.strip()
        print(f"[Preview] cv2 display probe failed (exit {result.returncode}): "
              f"{stderr[:200] if stderr else '(no stderr)'}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[Preview] cv2 display probe timed out after {timeout_s:.0f}s "
              f"(Qt/GTK init hung) — preview disabled.")
        return False
    except Exception as e:
        print(f"[Preview] cv2 display probe error: {e}")
        return False


def resolve_preview(requested: bool) -> bool:
    """
    Determine whether to actually show the preview window.
    Runs display checks only when the user requested SHOW_PREVIEW = True.
    Returns the resolved flag.
    """
    if not requested:
        return False

    # Step 1: environment check (fast, no subprocess)
    if not check_display_available():
        return False

    # Step 2: functional OpenCV HighGUI probe (subprocess, ~1s)
    print("[Preview] Probing cv2 display support...")
    if not probe_cv2_display():
        print("[Preview] OpenCV display unavailable — preview disabled.")
        return False

    print("[Preview] Display OK — live preview enabled.")
    return True


# =============================================================================
# RealSense calibration
# =============================================================================

def read_realsense_calibration(profile, resx, resy, fps) -> dict:
    try:
        cs = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ds = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci = cs.get_intrinsics()
        di = ds.get_intrinsics()
        e_c2d = cs.get_extrinsics_to(ds)
        e_d2c = ds.get_extrinsics_to(cs)
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale())

        def _intr(x):
            return {
                "width": int(x.width), "height": int(x.height),
                "fx": float(x.fx), "fy": float(x.fy),
                "cx": float(x.ppx), "cy": float(x.ppy),
                "K": [[float(x.fx), 0.0, float(x.ppx)],
                      [0.0, float(x.fy), float(x.ppy)],
                      [0.0, 0.0, 1.0]],
                "dist_model": str(x.model),
                "dist_coeffs": [float(c) for c in x.coeffs],
            }

        def _extr(e):
            r = [float(v) for v in e.rotation]
            t = [float(v) for v in e.translation]
            return {
                "R": [[r[0], r[3], r[6]], [r[1], r[4], r[7]], [r[2], r[5], r[8]]],
                "t": t,
            }

        calib = {
            "rgb": _intr(ci), "depth": _intr(di),
            "color_to_depth_extrinsics": _extr(e_c2d),
            "depth_to_color_extrinsics": _extr(e_d2c),
            "depth_scale_m_per_unit": depth_scale,
            "stream_mode": f"{resx}x{resy}@{fps}fps",
        }
        print(f"[RS] Intrinsics: fx={ci.fx:.2f} fy={ci.fy:.2f} "
              f"cx={ci.ppx:.2f} cy={ci.ppy:.2f}  "
              f"depth_scale={depth_scale:.6f}")
        return calib
    except Exception as e:
        print(f"[RS WARN] calibration read failed: {e}")
        return {}


# =============================================================================
# Color writer thread
# Handles: color MP4 encoding + color timestamps CSV
# =============================================================================

class ColorWriter(threading.Thread):
    """
    Writes color video frames and timestamps.
    Queue item: (color_image_ndarray, timestamp_ms, frame_idx)
    """

    def __init__(self, color_mp4_path: str, ts_writer, fps: int,
                 width: int, height: int):
        super().__init__(daemon=True, name="ColorWriter")
        self.color_mp4_path = color_mp4_path
        self.ts_writer      = ts_writer
        self.fps            = fps
        self.width          = width
        self.height         = height
        self.q              = queue.Queue(maxsize=COLOR_QUEUE_MAX)
        self._stop_flag     = threading.Event()
        self.frames_written = 0
        self.dropped_count  = 0

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.color_mp4_path, fourcc, self.fps,
                                 (self.width, self.height))
        while not self._stop_flag.is_set() or not self.q.empty():
            try:
                color_img, ts_ms, frame_idx = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            writer.write(color_img)
            self.ts_writer.writerow([frame_idx, ts_ms])
            self.frames_written += 1
            self.q.task_done()
        writer.release()

    def enqueue(self, color_img, ts_ms: int, frame_idx: int) -> bool:
        try:
            self.q.put_nowait((color_img, ts_ms, frame_idx))
            return True
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 10 == 1:
                print(f"[ColorWriter] Queue full — {self.dropped_count} color frames dropped")
            return False

    def stop(self):
        self._stop_flag.set()


# =============================================================================
# Depth writer thread
# Handles: depth .npy saves + depth vis MP4 + depth timestamps CSV
# =============================================================================

class DepthWriter(threading.Thread):
    """
    Writes depth frames (.npy), optional depth visualization video, and timestamps.
    Queue item: (depth_image_ndarray, timestamp_ms, frame_idx)
    """

    def __init__(self, depth_dir: str, depth_vis_path: str, ts_writer,
                 fps: int, save_vis: bool = True):
        super().__init__(daemon=True, name="DepthWriter")
        self.depth_dir      = depth_dir
        self.depth_vis_path = depth_vis_path
        self.ts_writer      = ts_writer
        self.fps            = fps
        self.save_vis       = save_vis
        self.q              = queue.Queue(maxsize=DEPTH_QUEUE_MAX)
        self._stop_flag     = threading.Event()
        self.vis_writer     = None
        self.frames_written = 0
        self.dropped_count  = 0

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        while not self._stop_flag.is_set() or not self.q.empty():
            try:
                depth_img, ts_ms, frame_idx = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Save raw depth
            np.save(os.path.join(self.depth_dir, f"{frame_idx:06d}.npy"), depth_img)
            self.ts_writer.writerow([frame_idx, ts_ms])

            # Optional colorized visualization video
            if self.save_vis:
                dmap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03),
                    cv2.COLORMAP_JET)
                if self.vis_writer is None:
                    dh, dw = dmap.shape[:2]
                    self.vis_writer = cv2.VideoWriter(
                        self.depth_vis_path, fourcc, self.fps, (dw, dh))
                self.vis_writer.write(dmap)

            self.frames_written += 1
            self.q.task_done()

        if self.vis_writer:
            self.vis_writer.release()

    def enqueue(self, depth_img, ts_ms: int, frame_idx: int) -> bool:
        try:
            self.q.put_nowait((depth_img, ts_ms, frame_idx))
            return True
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 10 == 1:
                print(f"[DepthWriter] Queue full — {self.dropped_count} depth frames dropped")
            return False

    def stop(self):
        self._stop_flag.set()


# =============================================================================
# Radar writer thread
# Offloads all radar disk I/O from the main capture loop so that binary
# writes and CSV rows never block wait_for_frames(). The main loop simply
# enqueues (ts_ms, chunk, in_waiting) tuples and immediately moves on.
# Queue capacity: 200 frames @ 100ms = 20s of buffer — far more than any
# transient spike. If the queue fills the session has a deeper I/O problem.
# =============================================================================

RADAR_QUEUE_MAX = 200

class RadarWriter(threading.Thread):
    """
    Writes radar binary chunks and radar timestamp CSV rows.
    Queue item: (ts_ms: int, chunk: bytes, in_waiting: int)
    """

    def __init__(self, bin_path: str, ts_writer):
        super().__init__(daemon=True, name="RadarWriter")
        self.bin_path       = bin_path
        self.ts_writer      = ts_writer
        self.q              = queue.Queue(maxsize=RADAR_QUEUE_MAX)
        self._stop_flag     = threading.Event()
        self.frames_written = 0
        self.dropped_count  = 0
        self.peak_q_depth   = 0

    def run(self):
        with open(self.bin_path, "wb") as bin_f:
            while not self._stop_flag.is_set() or not self.q.empty():
                try:
                    ts_ms, chunk, in_waiting = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue
                bin_f.write(struct.pack('<QII', ts_ms, len(chunk), in_waiting))
                bin_f.write(chunk)
                self.ts_writer.writerow([self.frames_written, ts_ms])
                self.frames_written += 1
                self.q.task_done()

    def enqueue(self, ts_ms: int, chunk: bytes, in_waiting: int) -> bool:
        qd = self.q.qsize()
        if qd > self.peak_q_depth:
            self.peak_q_depth = qd
        try:
            self.q.put_nowait((ts_ms, chunk, in_waiting))
            return True
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 10 == 1:
                print(f"[RadarWriter] Queue full — {self.dropped_count} radar frames dropped. "
                      f"Check disk write speed.")
            return False

    def stop(self):
        self._stop_flag.set()


# =============================================================================
# DCA radar reader thread
# =============================================================================

class DCARadarReader(threading.Thread):

    def __init__(self, xwr_config: dict):
        super().__init__(daemon=True, name="DCARadarReader")
        self.xwr_config   = xwr_config
        self.q            = queue.Queue()
        self._stop_event  = threading.Event()
        self.error        = None
        self.peak_q_depth = 0
        self.total_frames = 0

    def run(self):
        try:
            import xwr
            print("[DCA] Initializing xwr system...")
            awr = xwr.XWRSystem(**self.xwr_config)
            print("[DCA] Radar stream starting...")
            for frame in awr.dstream(numpy=True):
                if self._stop_event.is_set():
                    break
                ts_ms = mono_epoch_ms()
                raw   = frame.tobytes()
                qd    = self.q.qsize()
                if qd > self.peak_q_depth:
                    self.peak_q_depth = qd
                self.q.put((ts_ms, raw, 0))
                self.total_frames += 1
            awr.stop()
            print(f"[DCA] Stream stopped. Total frames: {self.total_frames}")
        except ImportError:
            self.error = "xwr not installed. pip install xwr --break-system-packages"
            print(f"[DCA ERROR] {self.error}")
        except Exception as e:
            self.error = e
            print(f"[DCA] Thread error: {e}")

    def stop(self):
        self._stop_event.set()


# =============================================================================
# Main recording function
# =============================================================================

def record():
    ensure_rmem_max(26_214_400)

    if not os.path.exists(MMWAVE_CFG_PATH):
        raise SystemExit(f"[ERROR] CFG not found: {MMWAVE_CFG_PATH}")

    radar_mode            = infer_radar_mode(MMWAVE_CFG_PATH)
    radar_frame_period_ms = parse_framecfg_period_ms(MMWAVE_CFG_PATH)
    radar_hz              = 1000.0 / radar_frame_period_ms \
                            if radar_frame_period_ms > 0 else 10.0

    # -------------------------------------------------------------------------
    # Resolve preview display before starting any hardware.
    # On Ubuntu/Linux: auto-disables if no display server or headless OpenCV.
    # -------------------------------------------------------------------------
    show_preview = resolve_preview(SHOW_PREVIEW)

    print("=" * 62)
    print("  DCA1000 + RealSense Recorder v7 (Linux/Jetson)")
    print("=" * 62)
    print(f"[Info]    Mode: {radar_mode.upper()}  "
          f"Session: {SESSION_RECORDING_DURATION_S:.0f}s  "
          f"Overall: {OVERALL_RECORDING_DURATION_S:.0f}s  "
          f"Radar: {radar_hz:.1f} Hz")
    print(f"[Info]    Preview: {'ON' if show_preview else 'OFF'}  "
          f"DepthVis: {'ON' if SAVE_DEPTH_VIS else 'OFF'}  "
          f"Warmup: {RADAR_WARMUP_FRAMES} frames\n")

    # --- Radar warmup ---
    print(f"[WARMUP] Starting radar... waiting for {RADAR_WARMUP_FRAMES} frames...")
    radar_reader = DCARadarReader(XWR_CONFIG)
    radar_reader.start()

    warmup_start = time.monotonic()
    warmup_count = 0
    warmup_done  = False

    while time.monotonic() - warmup_start < RADAR_WARMUP_TIMEOUT_S:
        if radar_reader.error:
            raise SystemExit(f"[ERROR] Radar init failed: {radar_reader.error}")
        while not radar_reader.q.empty():
            try:
                radar_reader.q.get_nowait()
                warmup_count += 1
            except queue.Empty:
                break
        if warmup_count >= RADAR_WARMUP_FRAMES:
            warmup_done = True
            elapsed = time.monotonic() - warmup_start
            print(f"[WARMUP] Radar stable after {warmup_count} frames "
                  f"({elapsed:.1f}s). Starting recording.\n")
            break
        time.sleep(0.05)

    if not warmup_done:
        print(f"[WARMUP] Timeout ({warmup_count} frames). Proceeding anyway.")

    _stop = threading.Event()
    signal.signal(signal.SIGINT,  lambda s, f: _stop.set())
    signal.signal(signal.SIGTERM, lambda s, f: _stop.set())

    # --- RealSense pipeline ---
    pipeline       = rs.pipeline()
    rs_cfg         = rs.config()
    rs_cfg.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)
    rs_cfg.enable_stream(rs.stream.depth, RS_WIDTH, RS_HEIGHT, rs.format.z16,  RS_FPS)
    active_profile = pipeline.start(rs_cfg)
    rs_calibration = read_realsense_calibration(
        active_profile, RS_WIDTH, RS_HEIGHT, RS_FPS)

    # --- Pre-create preview window before the hot loop (avoids first-frame stall) ---
    if show_preview:
        try:
            cv2.namedWindow("Recording — press Q to stop", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Recording — press Q to stop", 640, 360)
        except cv2.error as e:
            print(f"[Preview] namedWindow failed: {e} — preview disabled.")
            show_preview = False

    completed_sessions = []
    preview_error_logged = False
    rs_ts_domain_name = "unknown"
    hw_to_wall_offset_ms = 0.0
    first_frame_seen = False

    def run_one_session(session_index: int, duration_s: float):
        nonlocal show_preview, preview_error_logged
        nonlocal rs_ts_domain_name, hw_to_wall_offset_ms, first_frame_seen

        time_str     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_name = f"session_{time_str}"
        session_dir  = os.path.join(OUTPUT_ROOT, session_name)
        suffix = 2
        while os.path.exists(session_dir):
            session_name = f"session_{time_str}_{suffix:02d}"
            session_dir  = os.path.join(OUTPUT_ROOT, session_name)
            suffix += 1
        depth_dir = os.path.join(session_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        stem_path = os.path.join(session_dir, session_name)

        discarded = discard_pending_radar_frames(radar_reader)
        if discarded:
            print(f"[DCA] Discarded {discarded} inter-session radar frames before start.")

        radar_ts_f = open(stem_path + "_radar_timestamps.csv", "w", newline="")
        color_ts_f = open(stem_path + "_color_timestamps.csv", "w", newline="")
        depth_ts_f = open(stem_path + "_depth_timestamps.csv", "w", newline="")

        radar_ts_writer = csv.writer(radar_ts_f)
        color_ts_writer = csv.writer(color_ts_f)
        depth_ts_writer = csv.writer(depth_ts_f)
        radar_ts_writer.writerow(["frame_index", "timestamp_ms"])
        color_ts_writer.writerow(["frame_index", "timestamp_ms"])
        depth_ts_writer.writerow(["frame_index", "timestamp_ms"])

        radar_writer = RadarWriter(stem_path + ".bin", radar_ts_writer)
        color_writer = ColorWriter(
            color_mp4_path = stem_path + "_color.mp4",
            ts_writer      = color_ts_writer,
            fps            = RS_FPS,
            width          = RS_WIDTH,
            height         = RS_HEIGHT,
        )
        depth_writer = DepthWriter(
            depth_dir       = depth_dir,
            depth_vis_path  = stem_path + "_depth_vis.mp4",
            ts_writer       = depth_ts_writer,
            fps             = RS_FPS,
            save_vis        = SAVE_DEPTH_VIS,
        )

        radar_writer.start()
        color_writer.start()
        depth_writer.start()

        total_radar_bytes  = 0
        total_radar_frames = 0
        color_frame_idx    = 0
        depth_frame_idx    = 0
        color_timestamps   = []
        depth_timestamps   = []
        radar_start_ms     = 0
        radar_end_ms       = 0
        video_start_ms     = 0
        dup_frames_skipped = 0
        radar_health_ok    = False

        recording_start   = time.monotonic()
        last_report_time  = recording_start
        last_report_bytes = 0
        stopped_by_user   = False

        print("\n" + "=" * 62)
        print(f"  Session {session_index} Start: {session_name}")
        print("=" * 62)
        print(f"[INFO] Recording this session for {duration_s:.1f}s"
              + (" — press Q to stop all sessions" if show_preview else ""))
        beep_session_start()

        try:
            while not _stop.is_set():
                while not radar_reader.q.empty():
                    try:
                        ts_ms, chunk, in_waiting = radar_reader.q.get_nowait()
                        total_radar_bytes += len(chunk)
                        if radar_start_ms == 0:
                            radar_start_ms = ts_ms
                            print(f"[DCA] First recorded frame at {ts_ms} ms")
                        radar_writer.enqueue(ts_ms, chunk, in_waiting)
                        total_radar_frames += 1
                        radar_end_ms = ts_ms
                    except queue.Empty:
                        break

                if radar_reader.error:
                    print(f"[DCA] Fatal error: {radar_reader.error}")
                    _stop.set()
                    break

                now_mono = time.monotonic()
                if now_mono - last_report_time >= 5.0:
                    elapsed = now_mono - last_report_time
                    bps = (total_radar_bytes - last_report_bytes) / elapsed
                    first_rep = last_report_bytes == 0

                    if bps < 500:
                        status = "WARNING: near-zero"
                    elif bps < 5_000:
                        status = "LOW"
                    else:
                        radar_health_ok = True
                        status = "OK"

                    t_elapsed = now_mono - recording_start
                    print(f"[Health s{session_index} t={t_elapsed:.0f}s] "
                          f"radar={bps/1024:.1f} KB/s  "
                          f"r_frames={total_radar_frames}  "
                          f"color={color_frame_idx}  "
                          f"rq={radar_writer.q.qsize()}  "
                          f"cq={color_writer.q.qsize()}  "
                          f"dq={depth_writer.q.qsize()}  "
                          f"{status}")

                    if first_rep and bps < 500:
                        print("[DCA] No radar data after 5s. Check ethernet + rmem_max.")

                    last_report_time  = now_mono
                    last_report_bytes = total_radar_bytes

                try:
                    frames      = pipeline.wait_for_frames(timeout_ms=5000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                except Exception as e:
                    print(f"[RS] Frame error: {e} — retrying...")
                    continue

                if not color_frame:
                    continue

                if not first_frame_seen:
                    domain = color_frame.get_frame_timestamp_domain()
                    rs_ts_domain_name = str(domain)
                    is_global = domain == rs.timestamp_domain.global_time
                    print(f"[RS] Timestamp domain: {rs_ts_domain_name}  "
                          f"({'OK' if is_global else 'WARNING'})")
                    hw_to_wall_offset_ms = color_frame.get_timestamp() - now_ms()
                    print(f"[RS] hw_to_wall_offset_ms = {hw_to_wall_offset_ms:+.1f} ms")
                    first_frame_seen = True

                frame_ts_ms = int(color_frame.get_timestamp())

                if color_timestamps and frame_ts_ms == color_timestamps[-1]:
                    dup_frames_skipped += 1
                    continue

                color_img = np.asanyarray(color_frame.get_data()).copy()
                depth_img = np.asanyarray(depth_frame.get_data()).copy() \
                            if depth_frame else None
                depth_ts  = int(depth_frame.get_timestamp()) \
                            if depth_frame else frame_ts_ms

                color_timestamps.append(frame_ts_ms)
                if video_start_ms == 0:
                    video_start_ms = frame_ts_ms

                color_writer.enqueue(color_img, frame_ts_ms, color_frame_idx)
                color_frame_idx += 1

                if depth_img is not None:
                    depth_writer.enqueue(depth_img, depth_ts, depth_frame_idx)
                    depth_timestamps.append(depth_ts)
                    depth_frame_idx += 1

                if show_preview:
                    try:
                        preview_img = color_img.copy()
                        if _HAS_FOV_OVERLAY:
                            draw_fov_overlay_simple(preview_img, RS_WIDTH, RS_HEIGHT)
                        cv2.imshow("Recording — press Q to stop", preview_img)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                            print("\n[Info] Q pressed — stopping all sessions.")
                            stopped_by_user = True
                            _stop.set()
                            break
                    except cv2.error as e:
                        if not preview_error_logged:
                            print(f"[Preview] Display error: {e} — preview disabled.")
                            preview_error_logged = True
                        show_preview = False
                        try:
                            cv2.destroyAllWindows()
                        except Exception:
                            pass

                if time.monotonic() - recording_start >= duration_s:
                    print(f"\n[Info] Session {session_index} duration reached.")
                    break

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback; traceback.print_exc()
            _stop.set()

        finally:
            recording_end = time.monotonic()
            beep_session_stop()

            radar_q = radar_writer.q.qsize()
            color_q = color_writer.q.qsize()
            depth_q = depth_writer.q.qsize()
            if radar_q + color_q + depth_q > 0:
                print(f"[Info] Flushing writers: "
                      f"{radar_q} radar + {color_q} color + {depth_q} depth frames...")
            radar_writer.stop()
            color_writer.stop()
            depth_writer.stop()
            radar_writer.join(timeout=60.0)
            color_writer.join(timeout=60.0)
            depth_writer.join(timeout=120.0)

            if radar_writer.dropped_count > 0:
                print(f"[WARN] RadarWriter dropped {radar_writer.dropped_count} frames "
                      f"(disk too slow). Peak queue depth: {radar_writer.peak_q_depth}")

            radar_ts_f.close()
            color_ts_f.close()
            depth_ts_f.close()

            if len(color_timestamps) >= 2:
                import statistics
                deltas = [color_timestamps[i+1] - color_timestamps[i]
                          for i in range(len(color_timestamps)-1)
                          if color_timestamps[i+1] > color_timestamps[i]]
                median_dt = statistics.median(deltas) if deltas else 33.3
                actual_fps = round(1000.0 / median_dt, 1) if median_dt > 0 else RS_FPS
            else:
                actual_fps = RS_FPS

            actual_duration_s = max(0.0, recording_end - recording_start)
            meta = {
                "session":               session_name,
                "session_index":         session_index,
                "bin_format_version":    BIN_FORMAT_VERSION,
                "recorder":              "dca_realsense_recorder_v7.py",
                "radar_mode":            radar_mode,
                "radar_cfg_path":        MMWAVE_CFG_PATH,
                "radar_frame_period_ms": radar_frame_period_ms,
                "configured_session_duration_s": duration_s,
                "configured_overall_duration_s": OVERALL_RECORDING_DURATION_S,
                "actual_session_duration_s": round(actual_duration_s, 3),
                "stopped_by_user":        stopped_by_user,
                "hw_to_wall_offset_ms":  round(hw_to_wall_offset_ms, 3),
                "rs_timestamp_domain":   rs_ts_domain_name,
                "realsense_calibration": rs_calibration,
                "xwr_config":            XWR_CONFIG,
                "radar_warmup_frames_discarded": warmup_count,
                "inter_session_radar_frames_discarded": discarded,
                "radar": {
                    "start_ms":     radar_start_ms,
                    "end_ms":       radar_end_ms,
                    "total_bytes":  total_radar_bytes,
                    "total_frames": total_radar_frames,
                    "health_ok":    radar_health_ok,
                },
                "video": {
                    "start_ms":   video_start_ms,
                    "end_ms":     color_timestamps[-1] if color_timestamps else 0,
                    "fps":        RS_FPS,
                    "actual_fps": actual_fps,
                    "width":      RS_WIDTH,
                    "height":     RS_HEIGHT,
                },
                "depth": {
                    "start_ms":   depth_timestamps[0]  if depth_timestamps else 0,
                    "end_ms":     depth_timestamps[-1] if depth_timestamps else 0,
                    "num_frames": depth_frame_idx,
                    "format":     "uint16_mm",
                    "storage":    "per_frame_npy",
                },
            }
            meta_path = os.path.join(session_dir, "meta_data.json")
            with open(meta_path, "w") as mf:
                json.dump(meta, mf, indent=2)

            expected_radar = int(duration_s * radar_hz)
            drop_pct = max(0.0, (1 - total_radar_frames / expected_radar) * 100) \
                       if expected_radar > 0 else 0.0
            color_drops = color_writer.dropped_count
            depth_drops = depth_writer.dropped_count

            print("\n" + "=" * 62)
            print(f"  Session {session_index} Complete")
            print("=" * 62)
            print(f"  Session:            {session_name}")
            print(f"  Mode:               {radar_mode.upper()}")
            print(f"  Radar frames:       {total_radar_frames}  "
                  f"(expected ~{expected_radar},  drop {drop_pct:.1f}%)")
            print(f"  Radar written:      {radar_writer.frames_written}  "
                  f"(dropped: {radar_writer.dropped_count}  "
                  f"peak_q: {radar_writer.peak_q_depth})")
            print(f"  Radar data:         {total_radar_bytes/1024:.0f} KB")
            print(f"  Color frames:       {color_frame_idx}  "
                  f"(dups: {dup_frames_skipped},  fps: {actual_fps}  "
                  f"drops: {color_drops})")
            print(f"  Color written:      {color_writer.frames_written}")
            print(f"  Depth frames:       {depth_frame_idx}  "
                  f"(drops: {depth_drops})")
            print(f"  Depth written:      {depth_writer.frames_written}")
            print(f"  Warmup discarded:   {warmup_count} radar frames")
            print(f"  Output:             {session_dir}")

            all_good = (drop_pct <= 5 and color_drops == 0 and depth_drops == 0)
            if all_good:
                print("\n[OK] Clean session — no drops on any channel.")
            else:
                if drop_pct > 5:
                    print(f"\n[WARN] Radar: {drop_pct:.1f}% frame loss.")
                if color_drops > 0:
                    print(f"[WARN] Color writer dropped {color_drops} frames.")
                if depth_drops > 0:
                    print(f"[WARN] Depth writer dropped {depth_drops} frames.")
                if color_drops > 0 or depth_drops > 0:
                    print("  Try: set SAVE_DEPTH_VIS = False  (removes one encode pass)")
                    print("  Or:  reduce RS_WIDTH / RS_HEIGHT")

            completed_sessions.append(session_dir)
            return actual_duration_s

    session_index = 0
    recorded_duration_s = 0.0
    print(f"[INFO] Starting stop-go sessions for up to "
          f"{OVERALL_RECORDING_DURATION_S:.0f}s of recorded data.\n")

    try:
        while not _stop.is_set():
            remaining = OVERALL_RECORDING_DURATION_S - recorded_duration_s
            if remaining <= 0:
                print("[Info] Overall duration reached — stopping.")
                break
            session_index += 1
            actual_duration_s = run_one_session(
                session_index,
                min(SESSION_RECORDING_DURATION_S, remaining),
            )
            recorded_duration_s += min(actual_duration_s, remaining)
    finally:
        radar_reader.stop()
        radar_reader.join(timeout=5.0)
        pipeline.stop()
        if show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("\n" + "=" * 62)
    print("  Stop-Go Recording Complete")
    print("=" * 62)
    print(f"  Sessions written:   {len(completed_sessions)}")
    print(f"  Overall target:     {OVERALL_RECORDING_DURATION_S:.0f}s")
    print(f"  Recorded duration:  {recorded_duration_s:.1f}s")
    print(f"  Output root:        {OUTPUT_ROOT}")
    if completed_sessions:
        print("\n  Next steps:")
        print("    python temporal_align_dca_v2.py")
        pfa_arg = " --pfa 1e-2 --min-snr 3" if radar_mode == "calib" else ""
        for session_dir in completed_sessions:
            print(f"    python adc_to_pointcloud_v2.py "
                  f"--session {session_dir}{pfa_arg}")
        if radar_mode == "calib":
            print("    python extract_calibration_targets_v2.py")


if __name__ == "__main__":
    record()
