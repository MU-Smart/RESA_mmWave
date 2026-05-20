#!/usr/bin/env python3
"""Guidance orchestrator — fast guidance, LLM guidance, and TTS output."""

# =========================================================================
# 1. fast_guidance — rule-based fallback guidance
# =========================================================================

import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_pipeline import get_logger, m_to_ft, azimuth_sector

log = get_logger("Guidance")


class FastGuidanceEngine:
    """Rule-based guidance that fires when the LLM is unavailable or too slow."""

    def __init__(self, min_interval_s: float = 3.0):
        self._min_interval = min_interval_s
        self._last_guidance: float = 0.0
        self._last_text: str = ""

    def get_guidance(self, scene: dict) -> str:
        now = time.monotonic()
        if now - self._last_guidance < self._min_interval:
            return self._last_text

        decision = scene.get("decision", {})
        action = str(decision.get("primary_action", "continue_straight"))
        reason = str(decision.get("reason", "clear_path"))
        direction = str(decision.get("best_direction", "center"))
        urgency = str(decision.get("urgency", "none"))

        if action == "stop":
            text = "Stop. Person ahead."
        elif action == "slow_down":
            text = "Slow down. Person nearby."
        elif action == "turn_away":
            text = f"Turn {direction}. Path blocked."
        elif reason == "blocked_path":
            text = f"Path blocked. Move {direction}."
        else:
            text = "Path clear. Continue forward."

        self._last_guidance = now
        self._last_text = text
        return text


# =========================================================================
# 2. llm_guidance — LLM-based guidance engine
# =========================================================================

def _coerce_payload(payload: dict) -> Tuple[dict, dict]:
    scene = payload.get("scene", {}) or {}
    decision = payload.get("decision", {}) or {}
    return scene, decision


def build_prompt(payload: dict) -> tuple[str, str]:
    scene, decision = _coerce_payload(payload)
    blocking = scene.get("blocking", {})
    humans = scene.get("humans", {})
    open_dirs = scene.get("open_directions", {})
    counts = scene.get("class_counts", {})
    action = str(decision.get("primary_action", "continue_straight"))
    reason = str(decision.get("reason", "clear_path"))
    direction = str(decision.get("best_direction", "center"))
    urgency = str(decision.get("urgency", "none"))

    system_prompt = (
        "You are a navigation assistant for a visually impaired person using a radar-guided mobility aid. "
        "Provide brief, clear spoken guidance (1-2 sentences). "
        "Use natural language. Prioritize safety. "
        "Do not mention radar, point clouds, or technical details."
    )

    user_prompt = (
        f"Current situation: {counts.get('structure', 0)} structure points, "
        f"{counts.get('floor', 0)} floor points, {counts.get('human', 0)} human detections. "
        f"Center clear: {open_dirs.get('center_clear', True)}. "
        f"Left clear: {open_dirs.get('left_clear', True)}. "
        f"Right clear: {open_dirs.get('right_clear', True)}. "
        f"Action: {action}. Reason: {reason}. Direction: {direction}. Urgency: {urgency}."
    )
    return system_prompt, user_prompt


def build_scene_description(payload: dict) -> str:
    scene, decision = _coerce_payload(payload)
    blocking = scene.get("blocking", {})
    humans = scene.get("humans", {})
    open_dirs = scene.get("open_directions", {})
    counts = scene.get("class_counts", {})
    action = str(decision.get("primary_action", "continue_straight"))
    reason = str(decision.get("reason", "clear_path"))
    direction = str(decision.get("best_direction", "center"))
    urgency = str(decision.get("urgency", "none"))

    if action == "stop":
        return "Stop. Person detected ahead."
    if action == "slow_down":
        return "Slow down. Person nearby."
    if action == "turn_away":
        return f"Turn {direction}. Path is blocked."
    if reason == "blocked_path":
        return f"Path blocked. Move {direction}."
    return "Path clear. Continue forward."


class GuidanceEngine:
    def __init__(self, model: str = "llama3.2:3b", min_interval_s: float = 8.0,
                 ollama_host: str = "http://127.0.0.1:11434", change_threshold: bool = False):
        self._model = model
        self._min_interval = min_interval_s
        self._ollama_host = ollama_host
        self._change_threshold = change_threshold
        self._last_guidance: float = 0.0
        self._last_text: str = ""
        self._last_scene_hash: int = 0

    def get_guidance(self, scene: dict) -> str:
        now = time.monotonic()
        if now - self._last_guidance < self._min_interval:
            return self._last_text

        payload = {"scene": scene, "decision": scene.get("decision", {})}
        system_prompt, user_prompt = build_prompt(payload)

        try:
            import requests
            response = requests.post(
                f"{self._ollama_host}/api/generate",
                json={"model": self._model, "system": system_prompt, "prompt": user_prompt, "stream": False},
                timeout=15.0,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text:
                    self._last_guidance = now
                    self._last_text = text
                    return text
        except Exception as e:
            log.warning(f"LLM request failed: {e}")

        return ""


# =========================================================================
# 3. tts_output — Piper TTS engine
# =========================================================================

class TTSEngine:
    def __init__(self, model: str = "", backend: str = "auto"):
        self._model = Path(model) if model else None
        self._backend = backend
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def speak_async(self, text: str) -> None:
        if not text:
            return
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def _speak(self, text: str) -> None:
        if self._model is None or not self._model.exists():
            log.info(f"[TTS] {text}")
            return
        try:
            proc = subprocess.Popen(
                [str(self._model), "--output-raw", "--model", str(self._model)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            if proc.stdin:
                proc.stdin.write(text.encode("utf-8"))
                proc.stdin.close()
        except Exception as e:
            log.warning(f"TTS failed: {e}")

    def stop(self) -> None:
        with self._lock:
            if self._process:
                self._process.terminate()
                self._process = None


# Explicit re-exports
__all__ = ["FastGuidanceEngine", "GuidanceEngine", "TTSEngine",
           "build_prompt", "build_scene_description"]
