"""Shared session catalog sync and selection helpers for notebook drivers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_BUCKET = "miamioh-resa-data"
DEFAULT_GCS_ROOT = f"gs://{DEFAULT_BUCKET}/CapstoneData"
DEFAULT_CATALOG_URI = f"{DEFAULT_GCS_ROOT}/curated/manifests/session_catalog.csv"
COLUMN_ORDER = [
    "session_id",
    "dataset_id",
    "legacy_dataset_name",
    "building",
    "capture_date",
    "raw_path",
    "processed_path",
    "oneformer_batch_id",
    "quality_tier",
    "label_status",
    "split",
    "published_dataset_version",
    "checkpoint_usage",
    "notes",
]


def _run_gcloud_cp(src: str | Path, dst: str | Path) -> None:
    cmd = ["gcloud", "storage", "cp", str(src), str(dst)]
    subprocess.run(cmd, check=True, text=True)


def sync_catalog_from_gcs(
    local_path: str | Path,
    gcs_uri: str = DEFAULT_CATALOG_URI,
) -> Path:
    """Download the live session catalog from GCS to a local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _run_gcloud_cp(gcs_uri, local_path)
    return local_path


def sync_catalog_to_gcs(
    local_path: str | Path,
    gcs_uri: str = DEFAULT_CATALOG_URI,
) -> Path:
    """Upload a local session catalog snapshot back to the live GCS path."""
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Catalog not found: {local_path}")
    _run_gcloud_cp(local_path, gcs_uri)
    return local_path


def _empty_catalog() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMN_ORDER)


def load_catalog(path: str | Path) -> pd.DataFrame:
    """Load a session catalog, preserving a stable column order."""
    path = Path(path)
    if not path.exists():
        return _empty_catalog()
    df = pd.read_csv(path, low_memory=False)
    for column in COLUMN_ORDER:
        if column not in df.columns:
            df[column] = ""
    return df[COLUMN_ORDER + [c for c in df.columns if c not in COLUMN_ORDER]].copy()


def _normalize_values(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(str(v) for v in values if str(v))


def select_sessions(
    df: pd.DataFrame,
    quality_tier: Iterable[str] = ("gold",),
    split_exclude: Iterable[str] = (),
    dataset_ids: Iterable[str] | None = None,
    *,
    label_status: Iterable[str] | None = None,
    require_processed_path: bool = True,
) -> pd.DataFrame:
    """Select catalog rows for a notebook stage.

    Downstream consumers normally leave ``require_processed_path`` enabled, so
    they receive only usable curated records.  Intake code can select raw rows
    by setting it to ``False`` and filtering on ``label_status``.
    """
    out = df.copy()
    tiers = set(_normalize_values(quality_tier))
    if tiers:
        out = out[out["quality_tier"].astype(str).isin(tiers)]
    splits = set(_normalize_values(split_exclude))
    if splits:
        out = out[~out["split"].astype(str).isin(splits)]
    datasets = set(_normalize_values(dataset_ids))
    if datasets:
        out = out[out["dataset_id"].astype(str).isin(datasets)]
    statuses = set(_normalize_values(label_status))
    if statuses:
        out = out[out["label_status"].astype(str).isin(statuses)]
    if require_processed_path:
        out = out[out["processed_path"].fillna("").astype(str).str.len() > 0]
    return out.sort_values(["dataset_id", "session_id"]).reset_index(drop=True)


def admit_session(
    df: pd.DataFrame,
    session_id: str,
    quality_tier: str,
    label_status: str,
    processed_path: str,
    notes: str = "",
    *,
    dataset_id: str | None = None,
) -> pd.DataFrame:
    """Update or insert one catalog row after a curation decision.

    ``session_id`` is not globally unique in the legacy extracts: the same
    capture-name can occur in multiple legacy dataset roots.  Callers that
    process catalog-backed data must therefore supply ``dataset_id``.
    """
    out = df.copy()
    if out.empty:
        out = _empty_catalog()
    mask = out["session_id"].astype(str) == str(session_id)
    if dataset_id is not None:
        mask &= out["dataset_id"].astype(str) == str(dataset_id)
    matches = int(mask.sum())
    if matches > 1:
        raise ValueError(
            "Catalog admission is ambiguous; identify the row with dataset_id. "
            f"session_id={session_id!r} dataset_id={dataset_id!r} matches={matches}"
        )
    if matches == 1:
        out.loc[mask, "quality_tier"] = str(quality_tier)
        out.loc[mask, "label_status"] = str(label_status)
        out.loc[mask, "processed_path"] = str(processed_path)
        if notes:
            out.loc[mask, "notes"] = str(notes)
    else:
        row = {column: "" for column in COLUMN_ORDER}
        row["session_id"] = str(session_id)
        if dataset_id is not None:
            row["dataset_id"] = str(dataset_id)
        row["quality_tier"] = str(quality_tier)
        row["label_status"] = str(label_status)
        row["processed_path"] = str(processed_path)
        row["notes"] = str(notes)
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out[COLUMN_ORDER + [c for c in out.columns if c not in COLUMN_ORDER]].copy()
