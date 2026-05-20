"""Compatibility wrapper for legacy ``import nav_logger`` callers."""

from branch2.scene_pipeline import NavLogger, get_log_path, get_logger

__all__ = ["NavLogger", "get_log_path", "get_logger"]

