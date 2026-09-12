"""Curated Branch 1 ghost-subset rolling retraining driver."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tools.session_selector import load_catalog, select_sessions, sync_catalog_from_gcs

try:
    import pandas as pd
except ModuleNotFoundError:  # Allows --help before Colab dependency setup.
    pd = None  # type: ignore[assignment]


DEFAULT_DATASET_GROUPS = ("auto",)
DEFAULT_EXTERNAL_VAL_GROUP = "external_val"
DEFAULT_SESSION_PREFIX = "session_"
DEFAULT_BUCKET = "miamioh-resa-data"
DEFAULT_PROJECT = "fluent-webbing-496616-u8"
DEFAULT_TRAIN_QUALITY_TIERS = ("gold",)
DEFAULT_EXTERNAL_VAL_QUALITY_TIERS = ("gold", "silver")
DEFAULT_TRAIN_SPLITS = ("train",)
DEFAULT_EXTERNAL_VAL_SPLITS = ("external_val", "holdout", "val", "test")
GHOST_LABEL_CSV_NAME = "labeled_radar_points_v4_fused.csv"


@dataclass
class Config:
    bucket: str
    project: str
    gcs_mount_point: Path
    work_root: Path
    code_root: Path
    session_batch_size: int
    session_download_batch_size: int
    gcloud_process_count: int
    gcloud_thread_count: int
    train_epochs: int
    train_epochs_per_session_batch: int
    train_windows_per_batch: int
    train_num_workers: int
    train_points_per_frame: int
    train_emb_dims: int
    train_temporal_layers: int
    train_temporal_heads: int
    train_gate_hidden: int
    train_rd_patch_embed_dim: int
    train_k: int
    train_n_kernel_points: int
    train_pin_memory: bool
    ghost_negative_ratio: float
    ghost_p_dir_max: float
    ghost_ego_abs_z_min: float
    ghost_p_ego_min: float
    ghost_direct_p_dir_min: float
    ghost_depth_no_threshold: float
    ghost_depth_yes_threshold: float
    ghost_min_positive_rows: int
    ghost_min_negative_rows: int
    ghost_random_seed: int
    validation_session_count: int
    dataset_groups: tuple[str, ...]
    external_val_group: str
    train_quality_tiers: tuple[str, ...]
    external_val_quality_tiers: tuple[str, ...]
    train_splits: tuple[str, ...]
    external_val_splits: tuple[str, ...]
    train_dataset_ids: tuple[str, ...]
    external_val_dataset_ids: tuple[str, ...]
    session_prefix: str
    debug: bool
    enable_ghost_teacher: bool
    copy_ghost_assets: bool
    force_rebuild_external_val: bool
    compact_train_output: bool
    dry_run: bool

    @property
    def gcs_root_uri(self) -> str:
        return f"gs://{self.bucket}/CapstoneData"

    @property
    def gcs_code_src(self) -> str:
        return f"{self.gcs_root_uri}/code_v2/RESA_mmWave"

    @property
    def gcs_catalog_uri(self) -> str:
        return f"{self.gcs_root_uri}/curated/manifests/session_catalog.csv"

    @property
    def gcs_processed_session_root(self) -> str:
        return f"{self.gcs_root_uri}/curated/sessions"

    @property
    def gcs_final_output_root(self) -> str:
        return f"{self.gcs_root_uri}/published/branch1/checkpoints"

    @property
    def drive_root(self) -> Path:
        return self.gcs_mount_point / "CapstoneData"


@dataclass
class Paths:
    root: Path
    code: Path
    branch1: Path
    py: Path
    dataset_tools: Path
    fourclass_tools: Path
    spatiotemporal_tools: Path
    depth_correspondence_tools: Path
    ghost_teacher_fusion: Path
    train_script: Path
    extrinsics_json: Path
    cfg_path: Path
    rolling_root: Path
    rolling_processing_root: Path
    rolling_dataset_root: Path
    rolling_val_root: Path
    train_out: Path

    @property
    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        additions = [
            str(self.root),
            str(self.code),
        ]
        env["PYTHONPATH"] = ":".join(additions + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return env


def print_header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def require_pandas():
    if pd is None:
        raise ModuleNotFoundError("pandas is required for this stage. Run the notebook dependency setup cell first.")
    return pd


def run(
    cmd: Iterable[object],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    argv = [str(x) for x in cmd]
    print("\n$ " + " ".join(shlex.quote(x) for x in argv), flush=True)
    print("cwd:", cwd, flush=True)
    if dry_run:
        return None
    return subprocess.run(argv, cwd=str(cwd), env=env, check=True, text=True)


def run_capture(
    cmd: Iterable[object],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    allow_empty: bool = False,
) -> str:
    argv = [str(x) for x in cmd]
    if dry_run:
        return ""
    proc = subprocess.run(argv, cwd=str(cwd), env=env, text=True, capture_output=True)
    if proc.returncode != 0 and not allow_empty:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, argv, output=proc.stdout, stderr=proc.stderr)
    return (proc.stdout or "") + (proc.stderr or "")


def _filter_gcloud_noise(text: str) -> str:
    """Remove verbose per-object gcloud progress/copy chatter while preserving useful errors."""
    keep: list[str] = []
    noisy_prefixes = (
        "Copying ",
        "Removing ",
        "WARNING: Component check failed. Could not verify SDK install path.",
        "Average throughput:",
    )
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped and set(stripped) <= {"."}:
            continue
        if any(stripped.startswith(prefix) for prefix in noisy_prefixes):
            continue
        keep.append(line)
    return "".join(keep)


def run_gcloud_quiet(
    cmd: Iterable[object],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    label: str = "gcloud transfer",
) -> subprocess.CompletedProcess[str] | None:
    argv = [str(x) for x in cmd]
    compact = " ".join(shlex.quote(x) for x in argv[:4])
    print(f"$ {label}: {compact} ...", flush=True)
    print("cwd:", cwd, flush=True)
    if dry_run:
        return None
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        filtered = _filter_gcloud_noise((proc.stdout or "") + (proc.stderr or ""))
        if filtered.strip():
            print(filtered, file=sys.stderr, flush=True)
        else:
            print("gcloud command failed, but emitted only suppressed progress output.", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(proc.returncode, argv, output=proc.stdout, stderr=proc.stderr)
    return proc


def gcloud_storage_rsync(
    src: object,
    dst: object,
    *,
    cwd: Path,
    env: dict[str, str] | None,
    dry_run: bool,
    delete_unmatched: bool = False,
    exclude: str | None = None,
) -> None:
    cmd: list[object] = ["gcloud", "--quiet", "--verbosity=error", "storage", "rsync", src, dst, "--recursive"]
    if delete_unmatched:
        cmd.append("--delete-unmatched-destination-objects")
    if exclude:
        cmd.extend(["--exclude", exclude])
    run_gcloud_quiet(cmd, cwd=cwd, env=env, dry_run=dry_run, label=f"gcloud rsync {src} -> {dst}")


def gcloud_storage_cp(
    sources: object | Iterable[object],
    dst: object,
    *,
    cwd: Path,
    env: dict[str, str] | None,
    dry_run: bool,
    recursive: bool = False,
) -> None:
    if isinstance(sources, (str, Path)):
        source_args = [sources]
    else:
        source_args = list(sources)
    if not source_args:
        return
    cmd: list[object] = ["gcloud", "--quiet", "--verbosity=error", "storage", "cp"]
    if recursive:
        cmd.append("--recursive")
    cmd.extend(source_args)
    cmd.append(dst)
    run_gcloud_quiet(
        cmd,
        cwd=cwd,
        env=env,
        dry_run=dry_run,
        label=f"gcloud cp {len(source_args)} source(s) -> {dst}",
    )


def gcloud_storage_ls(
    pattern: str,
    *,
    cwd: Path,
    env: dict[str, str] | None,
    dry_run: bool,
    allow_empty: bool = True,
) -> list[str]:
    if dry_run:
        return []
    proc = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        if allow_empty:
            if proc.stderr.strip():
                print(proc.stderr.strip(), flush=True)
            return []
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def first_existing(label: str, *paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Missing " + label + ". Checked:\n  " + "\n  ".join(str(p) for p in paths))


def configure_gcloud(cfg: Config) -> None:
    env = os.environ.copy()
    run(["gcloud", "config", "set", "project", cfg.project], cwd=cfg.work_root, env=env, dry_run=cfg.dry_run)
    run(["gcloud", "config", "set", "storage/process_count", cfg.gcloud_process_count], cwd=cfg.work_root, env=env, dry_run=cfg.dry_run)
    run(["gcloud", "config", "set", "storage/thread_count", cfg.gcloud_thread_count], cwd=cfg.work_root, env=env, dry_run=cfg.dry_run)


def sync_code_from_gcs(cfg: Config) -> None:
    print_header("Sync Code From GCS")
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    cfg.code_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    exclude = r"LLM_ML/data/.*|data/.*|colab_outputs/.*|__pycache__/.*|\.ipynb_checkpoints/.*"
    gcloud_storage_rsync(
        cfg.gcs_code_src,
        cfg.code_root,
        cwd=cfg.work_root,
        env=env,
        dry_run=cfg.dry_run,
        delete_unmatched=True,
        exclude=exclude,
    )


def _safe_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip()).strip("_")
    return safe or "dataset"


def _dataset_suffix(groups: tuple[str, ...] | list[str]) -> str:
    return "_".join(_safe_name(group) for group in groups)


def resolve_paths(cfg: Config) -> Paths:
    # code_root is synced from CapstoneData/code_v2/RESA_mmWave and is the RESA_mmWave
    # repo root: branch1/, branch2/, branch3/, config/ live directly under it.
    code = cfg.code_root
    branch1 = code / "branch1"
    config_dir = code / "config"

    dataset_tools = branch1 / "tools" / "dataset_tools.py"
    fourclass_tools = branch1 / "tools" / "branch1_fourclass_tools.py"
    spatiotemporal_tools = branch1 / "tools" / "spatiotemporal_tools.py"
    depth_correspondence_tools = branch1 / "tools" / "depth_correspondence_tools.py"
    # Moved from tools/ to processing/ in the refactor.
    ghost_teacher_fusion = branch1 / "processing" / "ghost_teacher_fusion.py"
    train_script = branch1 / "training" / "train_hybrid_rd_patch_kpconv.py"
    rolling_root = cfg.work_root / "_rolling_branch1_rdra_batches"
    return Paths(
        root=cfg.work_root,
        code=code,
        branch1=branch1,
        py=Path(sys.executable),
        dataset_tools=dataset_tools,
        fourclass_tools=fourclass_tools,
        spatiotemporal_tools=spatiotemporal_tools,
        depth_correspondence_tools=depth_correspondence_tools,
        ghost_teacher_fusion=ghost_teacher_fusion,
        train_script=train_script,
        extrinsics_json=config_dir / "radar_camera_extrinsics.json",
        cfg_path=config_dir / "profile_objdet.cfg",
        rolling_root=rolling_root,
        rolling_processing_root=rolling_root / "processing",
        rolling_dataset_root=rolling_root / "datasets",
        rolling_val_root=rolling_root / "external_validation",
        train_out=cfg.work_root / "_train_output" / f"kpconv_branch1_ghost_validated_binary_{_dataset_suffix(cfg.dataset_groups)}_ext_{_safe_name(cfg.external_val_group)}_v1",
    )


def require_paths(paths: Paths) -> None:
    required = [
        paths.code,
        paths.branch1,
        paths.dataset_tools,
        paths.fourclass_tools,
        paths.spatiotemporal_tools,
        paths.depth_correspondence_tools,
        paths.ghost_teacher_fusion,
        paths.train_script,
        paths.cfg_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(str(path) for path in missing))


def gcs_mount_path_to_uri(cfg: Config, path: Path) -> str:
    text = str(path)
    prefix = str(cfg.gcs_mount_point).rstrip("/") + "/"
    if text.startswith(prefix):
        return "gs://" + text[len(prefix):]
    return text


def chunks(items: list, n: int):
    for i in range(0, len(items), n):
        yield i // n, items[i : i + n]


def chunk_items(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def dataset_processing_root(paths: Paths, group: str) -> Path:
    return paths.rolling_processing_root / group.strip("/")


def _group_config(cfg: Config, paths: Paths, group: str) -> tuple[str, str, Path]:
    group = str(group).strip().strip("/")
    if not group:
        raise ValueError("Dataset group cannot be empty.")
    return group, cfg.session_prefix, dataset_processing_root(paths, group)


def _session_name_from_gcs_uri(cfg: Config, uri: str, group: str, prefix: str) -> str | None:
    base = f"{cfg.gcs_processed_session_root.rstrip('/')}/{group.strip('/')}/"
    if not uri.startswith(base):
        return None
    rel = uri[len(base):].strip("/")
    if not rel:
        return None
    name = rel.split("/", 1)[0].rstrip("/")
    if prefix and not name.startswith(prefix):
        return None
    if name.endswith((".tgz", ".tar.gz", ".tar")):
        return None
    return name


def _is_archive_uri(uri: str) -> bool:
    lowered = str(uri).lower()
    return lowered.endswith(".tar.gz") or lowered.endswith(".tgz") or lowered.endswith(".tar")


def _is_compressed_group(group: str) -> bool:
    group = str(group).strip().strip("/").lower()
    return group.endswith("_compressed") or group.endswith("_archives") or group.endswith("_tarballs")


def _strip_archive_suffix(name: str) -> str:
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _session_name_from_archive_uri(uri: str, prefix: str) -> str | None:
    leaf = str(uri).rstrip("/").rsplit("/", 1)[-1]
    name = _strip_archive_suffix(leaf)
    if prefix and not name.startswith(prefix):
        return None
    return name or None


def _list_compressed_session_archives(cfg: Config, paths: Paths, group: str, prefix: str | None = None) -> list[tuple[str, str]]:
    group = group.strip("/")
    prefix = cfg.session_prefix if prefix is None else prefix
    base = f"{cfg.gcs_processed_session_root.rstrip('/')}/{group}"
    patterns = [
        f"{base}/{prefix}*.tar.gz",
        f"{base}/{prefix}*.tgz",
        f"{base}/{prefix}*.tar",
        f"{base}/**/{prefix}*.tar.gz",
        f"{base}/**/{prefix}*.tgz",
        f"{base}/**/{prefix}*.tar",
    ]
    archives: dict[str, str] = {}
    for pattern in patterns:
        matches = gcloud_storage_ls(
            pattern,
            cwd=paths.root,
            env=paths.env,
            dry_run=cfg.dry_run,
            allow_empty=True,
        )
        for uri in matches:
            uri = uri.strip()
            if not _is_archive_uri(uri):
                continue
            name = _session_name_from_archive_uri(uri, prefix)
            if not name:
                continue
            archives.setdefault(name, uri)
    out = sorted(archives.items(), key=lambda item: item[0])
    print(f"{group} compressed session archive(s) matching {prefix!r}: {len(out)}", flush=True)
    if out:
        print("  first:", out[:3], flush=True)
        print("  last :", out[-3:], flush=True)
    else:
        print(f"  looked under: {base}", flush=True)
    return out


def _safe_extract_tar(archive_path: Path, dst: Path, expected_session_name: str) -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    extract_tmp = dst / "_extract_tmp" / expected_session_name
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:*") as tf:
        base = extract_tmp.resolve()
        members = tf.getmembers()
        for member in members:
            target = (extract_tmp / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe archive path in {archive_path}: {member.name}")
        tf.extractall(extract_tmp)

    candidate = extract_tmp / expected_session_name
    if candidate.exists() and candidate.is_dir():
        src_session = candidate
    else:
        session_dirs = sorted(p for p in extract_tmp.rglob(f"{expected_session_name}") if p.is_dir())
        if session_dirs:
            src_session = session_dirs[0]
        else:
            # Some archives contain session files at the archive root rather than a top-level session folder.
            src_session = extract_tmp

    final_session = dst / expected_session_name
    if final_session.exists():
        shutil.rmtree(final_session)
    if src_session == extract_tmp:
        final_session.mkdir(parents=True, exist_ok=True)
        for item in extract_tmp.iterdir():
            if item.name == expected_session_name:
                continue
            shutil.move(str(item), str(final_session / item.name))
    else:
        shutil.move(str(src_session), str(final_session))
    shutil.rmtree(dst / "_extract_tmp", ignore_errors=True)
    return final_session


def list_expanded_session_names(cfg: Config, paths: Paths, group: str, prefix: str | None = None) -> list[str]:
    group = group.strip("/")
    prefix = cfg.session_prefix if prefix is None else prefix
    base = f"{cfg.gcs_processed_session_root.rstrip('/')}/{group}"
    pattern = f"{base}/{prefix}*" if prefix else f"{base}/*"
    matches = gcloud_storage_ls(
        pattern,
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
        allow_empty=True,
    )
    sessions = sorted({
        name
        for uri in matches
        for name in [_session_name_from_gcs_uri(cfg, uri, group, prefix)]
        if name
    })
    if not sessions:
        deep_pattern = f"{base}/{prefix}*/**" if prefix else f"{base}/**"
        matches = gcloud_storage_ls(
            deep_pattern,
            cwd=paths.root,
            env=paths.env,
            dry_run=cfg.dry_run,
            allow_empty=True,
        )
        sessions = sorted({
            name
            for uri in matches
            for name in [_session_name_from_gcs_uri(cfg, uri, group, prefix)]
            if name
        })
    print(f"{group} expanded sessions matching {prefix!r}: {len(sessions)}", flush=True)
    if sessions:
        print("  first:", sessions[:3], flush=True)
        print("  last :", sessions[-3:], flush=True)
    else:
        print(f"  looked under: {base}", flush=True)
        print(f"  pattern     : {pattern}", flush=True)
    return sessions


def load_live_catalog(cfg: Config, paths: Paths) -> pd.DataFrame:
    local_catalog = paths.root / "_session_catalog" / "session_catalog.csv"
    local_catalog.parent.mkdir(parents=True, exist_ok=True)
    sync_catalog_from_gcs(local_catalog, cfg.gcs_catalog_uri)
    return load_catalog(local_catalog)


def _normalize_csv_values(values: tuple[str, ...] | list[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = values
    return tuple(str(x).strip() for x in raw if str(x).strip())


def _catalog_records(
    cfg: Config,
    paths: Paths,
    *,
    quality_tiers: tuple[str, ...],
    include_splits: tuple[str, ...],
    dataset_ids: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    catalog = load_live_catalog(cfg, paths)
    rows = select_sessions(catalog, quality_tier=quality_tiers, split_exclude=("none", ""))
    include = set(_normalize_csv_values(include_splits))
    if include:
        rows = rows[rows["split"].astype(str).isin(include)]
    dataset_filter = set(_normalize_csv_values(dataset_ids))
    if dataset_filter:
        rows = rows[rows["dataset_id"].astype(str).isin(dataset_filter)]
    rows = rows.sort_values(["dataset_id", "session_id"]).reset_index(drop=True)
    if rows.empty:
        return []
    return [
        (str(row.dataset_id), str(row.session_id), str(row.processed_path))
        for row in rows.itertuples(index=False)
    ]


def validation_session_records(cfg: Config, paths: Paths) -> list[tuple[str, str, str]]:
    return _catalog_records(
        cfg,
        paths,
        quality_tiers=cfg.external_val_quality_tiers,
        include_splits=cfg.external_val_splits,
        dataset_ids=cfg.external_val_dataset_ids,
    )


def session_records(cfg: Config, paths: Paths) -> list[tuple[str, str, str]]:
    records = _catalog_records(
        cfg,
        paths,
        quality_tiers=cfg.train_quality_tiers,
        include_splits=cfg.train_splits,
        dataset_ids=cfg.train_dataset_ids,
    )
    if not records:
        raise RuntimeError(
            "No curated ghost-training sessions matched the current catalog filters. "
            f"catalog={cfg.gcs_catalog_uri} splits={cfg.train_splits} quality={cfg.train_quality_tiers}"
        )
    if cfg.dataset_groups == ("auto",):
        cfg.dataset_groups = tuple(dict.fromkeys(group for group, _name, _uri in records))
    return records


def clear_rolling_training_dirs(paths: Paths) -> None:
    for path in [paths.rolling_processing_root, paths.rolling_dataset_root]:
        if path.exists():
            shutil.rmtree(path)
    paths.rolling_processing_root.mkdir(parents=True, exist_ok=True)
    paths.rolling_dataset_root.mkdir(parents=True, exist_ok=True)


def _download_records_to_roots(
    cfg: Config,
    paths: Paths,
    records: list[tuple[str, str, str]],
    group_to_root: dict[str, Path],
) -> dict[str, int]:
    counts: dict[str, int] = {group: 0 for group in cfg.dataset_groups}
    counts.setdefault(cfg.external_val_group, 0)
    by_group: dict[str, list[tuple[str, str]]] = {group: [] for group in counts}
    for group, name, uri in records:
        by_group.setdefault(group, []).append((name, uri))
        counts.setdefault(group, 0)

    for group, items in by_group.items():
        if not items:
            continue
        dst = group_to_root.get(group, dataset_processing_root(paths, group))
        dst.mkdir(parents=True, exist_ok=True)
        expected = [name for name, _ in items]
        archive_items = [(name, uri) for name, uri in items if _is_archive_uri(uri)]
        expanded_items = [(name, uri) for name, uri in items if not _is_archive_uri(uri)]

        if archive_items:
            archive_dir = dst / "_downloaded_archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {len(archive_items)} compressed {group} session archive(s) -> {archive_dir}", flush=True)
            print("  first archive URIs:", [uri for _, uri in archive_items[:3]], flush=True)
            for batch_idx, batch in enumerate(chunk_items(archive_items, cfg.session_download_batch_size), start=1):
                print(f"  archive transfer batch {batch_idx}: {len(batch)} archive(s)", flush=True)
                gcloud_storage_cp(
                    [uri for _, uri in batch],
                    archive_dir,
                    cwd=paths.root,
                    env=paths.env,
                    dry_run=cfg.dry_run,
                    recursive=False,
                )
            if not cfg.dry_run:
                for name, uri in archive_items:
                    archive_name = uri.rstrip("/").rsplit("/", 1)[-1]
                    local_archive = archive_dir / archive_name
                    if not local_archive.exists():
                        matches = sorted(archive_dir.rglob(archive_name))
                        if matches:
                            local_archive = matches[0]
                    if not local_archive.exists():
                        raise RuntimeError(f"Missing downloaded archive for {name}: {uri} -> {archive_dir}")
                    final_session = _safe_extract_tar(local_archive, dst, name)
                    print(f"  extracted {local_archive.name} -> {final_session}", flush=True)
                shutil.rmtree(archive_dir, ignore_errors=True)

        if expanded_items:
            print(f"Downloading {len(expanded_items)} expanded {group} session folder(s) -> {dst}", flush=True)
            print("  first URIs:", [uri for _, uri in expanded_items[:3]], flush=True)
            for batch_idx, batch in enumerate(chunk_items(expanded_items, cfg.session_download_batch_size), start=1):
                print(f"  folder transfer batch {batch_idx}: {len(batch)} folder(s)", flush=True)
                gcloud_storage_cp(
                    [uri for _, uri in batch],
                    dst,
                    cwd=paths.root,
                    env=paths.env,
                    dry_run=cfg.dry_run,
                    recursive=True,
                )

        if not cfg.dry_run:
            local = {p.name for p in dst.iterdir() if p.is_dir() and not p.name.startswith("_")}
            missing = sorted(set(expected) - local)
            print(f"  local session directories after download/extract: {len(local)}", flush=True)
            if local:
                print("  local sample:", sorted(local)[:5], flush=True)
            if missing:
                raise RuntimeError(f"Missing downloaded/extracted {group} sessions under {dst}: {missing[:10]}")
        counts[group] += len(items)
    return counts


def download_session_records(cfg: Config, paths: Paths, records: list[tuple[str, str, str]]) -> dict[str, int]:
    return _download_records_to_roots(
        cfg,
        paths,
        records,
        {group: dataset_processing_root(paths, group) for group in cfg.dataset_groups},
    )


def debug_preflight(cfg: Config, paths: Paths) -> None:
    print_header("Driver Preflight")
    print("bucket:", cfg.bucket, flush=True)
    print("project:", cfg.project, flush=True)
    print("catalog:", cfg.gcs_catalog_uri, flush=True)
    print("processed session root:", cfg.gcs_processed_session_root, flush=True)
    print("train quality tiers:", list(cfg.train_quality_tiers), flush=True)
    print("train splits:", list(cfg.train_splits), flush=True)
    print("external val quality tiers:", list(cfg.external_val_quality_tiers), flush=True)
    print("external val splits:", list(cfg.external_val_splits), flush=True)
    print("dataset ids:", list(cfg.dataset_groups), flush=True)
    print("work root:", paths.root, flush=True)
    print("code root:", paths.code, flush=True)
    print("training script:", paths.train_script, flush=True)
    print("train output:", paths.train_out, flush=True)
    try:
        version = run_capture(["gcloud", "--version"], cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
        print("gcloud version:\n" + "\n".join(version.splitlines()[:4]), flush=True)
    except Exception as exc:
        print(f"WARNING: could not read gcloud version: {type(exc).__name__}: {exc}", flush=True)


def command_help_text(paths: Paths, cmd: list[object], cfg: Config) -> str:
    return run_capture(cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)


def append_supported_flag(cmd: list[object], help_text: str, flag: str, *values: object) -> None:
    if flag in help_text:
        cmd.append(flag)
        cmd.extend(values)


def append_supported_bool_flag(cmd: list[object], help_text: str, flag: str, enabled: bool) -> None:
    if enabled and flag in help_text:
        cmd.append(flag)


def append_training_resource_args(cmd: list[object], paths: Paths, cfg: Config) -> None:
    """Append A100/heavy-training flags only if the current training script supports them.

    This avoids argparse failures when the repository training script is older/newer
    than the notebook. Unsupported knobs are printed in debug mode and skipped.
    """
    help_text = command_help_text(paths, [paths.py, paths.train_script, "--help"], cfg)
    requested: list[tuple[str, object]] = [
        ("--num-workers", cfg.train_num_workers),
        ("--points-per-frame", cfg.train_points_per_frame),
        ("--emb-dims", cfg.train_emb_dims),
        ("--temporal-layers", cfg.train_temporal_layers),
        ("--temporal-heads", cfg.train_temporal_heads),
        ("--gate-hidden", cfg.train_gate_hidden),
        ("--rd-patch-embed-dim", cfg.train_rd_patch_embed_dim),
        ("--k", cfg.train_k),
        ("--n-kernel-points", cfg.train_n_kernel_points),
    ]
    supported: list[str] = []
    unsupported: list[str] = []
    for flag, value in requested:
        try:
            skip_value = value is None or int(value) <= 0
        except Exception:
            skip_value = value is None or str(value).strip() == ""
        if skip_value:
            continue
        if flag in help_text:
            cmd.extend([flag, str(value)])
            supported.append(f"{flag}={value}")
        else:
            unsupported.append(flag)

    if cfg.train_pin_memory and "--pin-memory" in help_text:
        cmd.append("--pin-memory")
        supported.append("--pin-memory")
    elif cfg.train_pin_memory:
        unsupported.append("--pin-memory")

    if cfg.debug:
        print("Training resource/capacity flags passed:", supported or "none", flush=True)
        if unsupported:
            print("Training resource/capacity flags unsupported by current training script:", unsupported, flush=True)


def build_patches_cmd(cfg: Config, paths: Paths, points_csv: Path, processing_root: Path, patch_dir: Path, prefix: str) -> list[object]:
    cmd: list[object] = [
        paths.py,
        paths.dataset_tools,
        "build-patches",
        "--point-csv",
        points_csv,
        "--sidecar-root",
        processing_root,
        "--output-dir",
        patch_dir,
        "--frame-number-offset",
        "0",
        "--drop-invalid",
        "--require-all-valid",
        "--allowed-session-prefix",
        prefix,
        "--extract-rd-scalars",
    ]
    help_text = command_help_text(paths, [paths.py, paths.dataset_tools, "build-patches", "--help"], cfg)
    for flag, value in [
        ("--ra-patch-source", "true_ra"),
        ("--ra-source", "true_ra"),
        ("--ra-patch-az-mode", "tx0_only"),
        ("--ra-patch-az-fft-size", "64"),
        ("--ra-patch-row-edge-mode", "zero_pad"),
    ]:
        append_supported_flag(cmd, help_text, flag, value)
    for flag in ["--ra-patch-log-scale", "--ra-patch-normalize-by-frame-max"]:
        append_supported_flag(cmd, help_text, flag)
    return cmd


def assert_ra_patch_dataset(csv_path: Path, context: str, cfg: Config) -> None:
    if cfg.dry_run:
        return
    required = {"ra_patch_shard", "ra_patch_index", "ra_patch_valid"}
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"{context} is RD-only; missing RA columns: {missing}")
        rows = 0
        valid = 0
        for row in reader:
            rows += 1
            valid += int(str(row.get("ra_patch_valid", "")).strip().lower() in {"1", "true", "t", "yes", "y"})
    if rows == 0 or valid == 0:
        raise RuntimeError(f"{context} has no valid RA patches: rows={rows}, valid_ra={valid}")
    print(f"{context}: valid RA patches {valid}/{rows}", flush=True)


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in fh) > 1


def _csv_row_count(path: Path, *, max_scan: int | None = None) -> int:
    if not path.exists() or not path.is_file():
        return 0
    rows = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for rows, _ in enumerate(fh, start=1):
            if max_scan is not None and rows >= max_scan:
                break
    return rows


def _session_quality_reasons(
    session_dir: Path,
    label_csv_name: str,
    *,
    require_radar_tensors: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if not session_dir.exists() or not session_dir.is_dir():
        return ["missing_session_dir"]

    label_csv = session_dir / label_csv_name
    if not label_csv.exists():
        reasons.append(f"missing_{label_csv_name}")
    else:
        try:
            if _csv_row_count(label_csv, max_scan=3) <= 1:
                reasons.append(f"empty_{label_csv_name}")
        except Exception as exc:
            reasons.append(f"unreadable_{label_csv_name}:{type(exc).__name__}")

    if require_radar_tensors:
        has_tensor = any(session_dir.glob("*radar_tensors*.npz")) or any(session_dir.glob("radar_tensors*.npz"))
        if not has_tensor:
            reasons.append("missing_radar_tensors_npz")

    return reasons


def usable_session_dirs(
    root: Path,
    prefix: str,
    label_csv_name: str,
    *,
    require_radar_tensors: bool = True,
) -> tuple[list[Path], list[tuple[Path, list[str]]]]:
    root = Path(root)
    sessions = sorted(p for p in root.glob(f"{prefix}*") if p.is_dir()) if root.exists() else []
    good: list[Path] = []
    bad: list[tuple[Path, list[str]]] = []
    for session_dir in sessions:
        reasons = _session_quality_reasons(
            session_dir,
            label_csv_name,
            require_radar_tensors=require_radar_tensors,
        )
        if reasons:
            bad.append((session_dir, reasons))
        else:
            good.append(session_dir)
    return good, bad


def quarantine_bad_sessions(root: Path, bad: list[tuple[Path, list[str]]], context: str) -> None:
    if not bad:
        return
    root = Path(root)
    quarantine_root = root / "_skipped_bad_sessions" / _safe_name(context)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    print(f"Skipping/quarantining {len(bad)} bad session(s) for {context}.", flush=True)
    for session_dir, reasons in bad:
        target = quarantine_root / session_dir.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if session_dir.exists():
            shutil.move(str(session_dir), str(target))
            (target / "skip_reason.txt").write_text("\n".join(reasons) + "\n", encoding="utf-8")
        print(f"  skipped {session_dir.name}: {', '.join(reasons)}", flush=True)


def ensure_usable_label_root(
    root: Path,
    prefix: str,
    label_csv_name: str,
    *,
    context: str,
    require_radar_tensors: bool = True,
    min_sessions: int = 1,
) -> list[Path]:
    good, bad = usable_session_dirs(
        root,
        prefix,
        label_csv_name,
        require_radar_tensors=require_radar_tensors,
    )
    print(
        f"Session QA for {context}: good={len(good)} bad={len(bad)} "
        f"root={root} label={label_csv_name}",
        flush=True,
    )
    if bad:
        quarantine_bad_sessions(root, bad, context)
        good, _ = usable_session_dirs(
            root,
            prefix,
            label_csv_name,
            require_radar_tensors=require_radar_tensors,
        )
    if len(good) < min_sessions:
        raise RuntimeError(
            f"No usable sessions remain for {context}. "
            f"root={root}, label={label_csv_name}, require_radar_tensors={require_radar_tensors}"
        )
    print("  usable sample:", [p.name for p in good[:5]], flush=True)
    return good


def build_points_command(
    paths: Paths,
    label_root: Path,
    points_dir: Path,
    holdout_last: int,
    prefix: str,
    label_csv_name: str,
) -> list[object]:
    return [
        paths.py,
        paths.dataset_tools,
        "build-points",
        "--processing-root",
        label_root,
        "--output-dir",
        points_dir,
        "--holdout-last",
        int(holdout_last),
        "--session-prefix",
        prefix,
        "--labeled-csv-name",
        label_csv_name,
    ]


def run_build_points_with_recovery(
    cfg: Config,
    paths: Paths,
    *,
    label_root: Path,
    points_dir: Path,
    holdout_last: int,
    prefix: str,
    label_csv_name: str,
    dataset_tag: str,
) -> None:
    ensure_usable_label_root(
        label_root,
        prefix,
        label_csv_name,
        context=f"{dataset_tag} before build-points",
        require_radar_tensors=False,
    )
    cmd = build_points_command(paths, label_root, points_dir, holdout_last, prefix, label_csv_name)
    try:
        run(cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    except subprocess.CalledProcessError:
        print_header(f"build-points failed for {dataset_tag}; rechecking sessions and retrying once")
        good, bad = usable_session_dirs(
            label_root,
            prefix,
            label_csv_name,
            require_radar_tensors=False,
        )
        if bad:
            quarantine_bad_sessions(label_root, bad, f"{dataset_tag}_build_points_retry")
        if not good and not bad:
            raise
        good_after, _ = usable_session_dirs(
            label_root,
            prefix,
            label_csv_name,
            require_radar_tensors=False,
        )
        if not good_after:
            raise RuntimeError(f"No usable sessions left after build-points recovery for {dataset_tag}.")
        if points_dir.exists():
            shutil.rmtree(points_dir)
        points_dir.mkdir(parents=True, exist_ok=True)
        run(cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)


def _dataset_has_ghost_inputs(dataset_root: Path) -> bool:
    csv_path = dataset_root / "points_with_rd_patch_index.csv"
    if not csv_path.exists():
        return False
    header = set(_csv_header(csv_path))
    return bool({"ghost_score_combined", "ghost_score", "depth_teacher_acc_label", "accumulatable_label"} & header)




def debug_dataset_csv(dataset_root: Path, label: str) -> None:
    csv_path = dataset_root / "points_with_rd_patch_index.csv"
    print_header(f"Dataset Debug: {label}")
    print("dataset root:", dataset_root, flush=True)
    print("csv:", csv_path, flush=True)
    if not csv_path.exists():
        print("CSV missing.", flush=True)
        return
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"Could not read CSV: {type(exc).__name__}: {exc}", flush=True)
        return
    print("rows:", len(df), flush=True)
    print("columns:", len(df.columns), flush=True)
    print("first columns:", list(df.columns[:30]), flush=True)
    for col in ["split", "bucket_4class", "rd_patch_shard", "ra_patch_shard", "point_label_weight"]:
        if col in df.columns:
            try:
                print(f"{col} counts/sample:", df[col].value_counts(dropna=False).head(12).to_dict(), flush=True)
            except Exception:
                print(f"{col}: present", flush=True)
        else:
            print(f"MISSING column: {col}", flush=True)
    for patch_col in ["rd_patch_shard", "ra_patch_shard"]:
        if patch_col in df.columns and len(df):
            sample = str(df[patch_col].dropna().astype(str).iloc[0]) if df[patch_col].notna().any() else ""
            if sample:
                sample_path = dataset_root / sample
                print(f"sample {patch_col}: {sample} exists={sample_path.exists()}", flush=True)




class GhostSubsetEmpty(RuntimeError):
    """Raised when a train batch contains no usable validated ghost subset."""


def _series(frame: pd.DataFrame, col: str, default) -> pd.Series:
    if col in frame.columns:
        return frame[col]
    return pd.Series(default, index=frame.index)


def _str_series(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    return _series(frame, col, default).fillna(default).astype(str).str.strip().str.lower()


def _num_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series(frame, col, default), errors="coerce").replace([float("inf"), float("-inf")], pd.NA).fillna(default).astype(float)


def _bool_series(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    raw = _series(frame, col, default)
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    return raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _derived_ego_abs_z(df: pd.DataFrame) -> pd.Series:
    if "ego_residual_abs_z" in df.columns:
        return _num_series(df, "ego_residual_abs_z", 0.0)
    if "ego_doppler_residual_mps" not in df.columns:
        return pd.Series(0.0, index=df.index)
    residual = _num_series(df, "ego_doppler_residual_mps", 0.0).abs()
    # Cheap robust fallback if preprocessing did not already write ego_residual_abs_z.
    mad = float((residual - residual.median()).abs().median())
    sigma = max(1.4826 * mad, 0.05)
    return (residual / sigma).clip(lower=0.0, upper=10.0)


def validated_ghost_subset_masks(df: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    bucket4 = _str_series(df, "bucket_4class", "")
    accum = _str_series(df, "accumulatable_label", "")
    return_type = _str_series(df, "return_type_label", "")
    depth_acc = _str_series(df, "depth_teacher_acc_label", "")
    depth_reason = _str_series(df, "depth_teacher_reason", "")
    depth_source = _str_series(df, "depth_teacher_source", "")
    depth_eligible = _bool_series(df, "depth_teacher_eligible", False)

    p_depth = _num_series(df, "p_depth_match", 1.0)
    cluster_size = _num_series(df, "depth_candidate_cluster_size", 0.0)
    cluster_radius = _num_series(df, "depth_candidate_cluster_radius_m", 999.0)
    p_dir = _num_series(df, "p_dir_doppler", 1.0)
    p_ego = _num_series(df, "p_ego", 1.0)
    ego_abs_z = _derived_ego_abs_z(df)
    ego_inlier = _num_series(df, "ego_inlier_flag", 1.0)

    ghost_candidate = (
        bucket4.eq("ghost_return")
        | accum.eq("no")
        | depth_acc.eq("no")
        | return_type.isin({"wall_multipath", "floor_bounce", "depth_occluded", "multipath", "ghost", "unknown_ghost"})
    )

    depth_reason_good = depth_reason.str.contains("depth", regex=False) | depth_source.str.contains("depth", regex=False)
    depth_cluster_good = (cluster_size >= 2.0) & (cluster_radius <= 0.75)
    depth_mismatch = (p_depth <= float(cfg.ghost_depth_no_threshold)) & depth_cluster_good
    depth_validated_ghost = ghost_candidate & depth_eligible & (depth_acc.eq("no") | depth_mismatch) & depth_reason_good

    directness_evidence = (
        (p_dir <= float(cfg.ghost_p_dir_max))
        | (ego_abs_z >= float(cfg.ghost_ego_abs_z_min))
        | (ego_inlier <= 0.0)
    )
    ego_available = p_ego >= float(cfg.ghost_p_ego_min)
    directness_validated_ghost = ghost_candidate & ego_available & directness_evidence

    positive = depth_validated_ghost | directness_validated_ghost

    depth_yes = depth_eligible & (depth_acc.eq("yes") | (p_depth >= float(cfg.ghost_depth_yes_threshold)))
    directness_yes = (p_dir >= float(cfg.ghost_direct_p_dir_min)) & ego_available & (ego_abs_z < float(cfg.ghost_ego_abs_z_min))
    accum_yes = accum.eq("yes") | return_type.eq("direct")
    negative = (~ghost_candidate) & (depth_yes | directness_yes | accum_yes)

    both = depth_validated_ghost & directness_validated_ghost
    source = pd.Series("", index=df.index, dtype=object)
    source.loc[depth_validated_ghost] = "depth_validated_correspondence"
    source.loc[directness_validated_ghost] = "directness_ego_motion_gated"
    source.loc[both] = "depth_and_directness"
    source.loc[negative] = "validated_direct_negative"

    return {
        "positive": positive.fillna(False),
        "negative": negative.fillna(False),
        "depth_validated_ghost": depth_validated_ghost.fillna(False),
        "directness_validated_ghost": directness_validated_ghost.fillna(False),
        "source": source,
    }


def _cap_negative_ratio(df: pd.DataFrame, cfg: Config, *, label: str) -> pd.DataFrame:
    if cfg.ghost_negative_ratio <= 0:
        return df
    out_parts = []
    for split_name, split_df in df.groupby("split", dropna=False, sort=False):
        pos = split_df[split_df["ghost_binary_label"] == "ghost_return"]
        neg = split_df[split_df["ghost_binary_label"] == "direct_return"]
        max_neg = int(max(len(pos) * float(cfg.ghost_negative_ratio), cfg.ghost_min_negative_rows))
        if len(neg) > max_neg:
            neg = neg.sample(n=max_neg, random_state=int(cfg.ghost_random_seed))
        out_parts.append(pd.concat([pos, neg], ignore_index=True))
    out = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
    print(
        f"{label}: after negative-ratio cap rows={len(out)} "
        f"counts={out['ghost_binary_label'].value_counts().to_dict()}",
        flush=True,
    )
    return out


def create_ghost_validated_subset_dataset(
    cfg: Config,
    paths: Paths,
    source_root: Path,
    label: str,
    *,
    require_both_classes: bool,
) -> Path:
    source_root = Path(source_root)
    src_csv = source_root / "points_with_rd_patch_index.csv"
    if not src_csv.exists():
        raise FileNotFoundError(f"Missing source dataset CSV for ghost subset: {src_csv}")

    df = pd.read_csv(src_csv, low_memory=False)
    masks = validated_ghost_subset_masks(df, cfg)
    pos = df[masks["positive"]].copy()
    neg = df[masks["negative"]].copy()
    pos["ghost_binary_label"] = "ghost_return"
    neg["ghost_binary_label"] = "direct_return"
    pos["ghost_validation_source"] = masks["source"].loc[pos.index].values
    neg["ghost_validation_source"] = masks["source"].loc[neg.index].values

    keep = pd.concat([pos, neg], ignore_index=True)
    if keep.empty or len(pos) < int(cfg.ghost_min_positive_rows):
        msg = (
            f"GHOST_SUBSET_EMPTY: {label} has too few validated ghost positives. "
            f"positive={len(pos)} negative={len(neg)} source_rows={len(df)}"
        )
        if require_both_classes:
            raise GhostSubsetEmpty(msg)
        raise RuntimeError(msg)
    if require_both_classes and len(neg) < int(cfg.ghost_min_negative_rows):
        raise GhostSubsetEmpty(
            f"GHOST_SUBSET_EMPTY: {label} has too few validated direct negatives. "
            f"positive={len(pos)} negative={len(neg)} source_rows={len(df)}"
        )

    keep = _cap_negative_ratio(keep, cfg, label=label)

    subset_root = source_root.parent / f"{source_root.name}_ghost_validated_subset"
    if subset_root.exists() or subset_root.is_symlink():
        if subset_root.is_dir() and not subset_root.is_symlink():
            shutil.rmtree(subset_root)
        else:
            subset_root.unlink()
    subset_root.mkdir(parents=True, exist_ok=True)

    # Symlink patch directories so rd_patch_shard / ra_patch_shard relative paths remain valid.
    for child in source_root.iterdir():
        if child.name.startswith("patches"):
            dst = subset_root / child.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(child.resolve(), dst, target_is_directory=True)

    out_csv = subset_root / "points_with_rd_patch_index.csv"
    keep.to_csv(out_csv, index=False)

    subset_counts = {
        "label": label,
        "source_root": str(source_root),
        "source_csv": str(src_csv),
        "output_csv": str(out_csv),
        "label_col": "ghost_binary_label",
        "bucket_order": ["direct_return", "ghost_return"],
        "source_rows": int(len(df)),
        "rows": int(len(keep)),
        "positive_rows": int((keep["ghost_binary_label"] == "ghost_return").sum()),
        "negative_rows": int((keep["ghost_binary_label"] == "direct_return").sum()),
        "validation_source_counts": {str(k): int(v) for k, v in keep["ghost_validation_source"].value_counts().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in keep["ghost_binary_label"].value_counts().to_dict().items()},
        "split_counts": {str(k): int(v) for k, v in keep.get("split", pd.Series([], dtype=str)).value_counts().to_dict().items()},
        "thresholds": {
            "ghost_p_dir_max": float(cfg.ghost_p_dir_max),
            "ghost_ego_abs_z_min": float(cfg.ghost_ego_abs_z_min),
            "ghost_p_ego_min": float(cfg.ghost_p_ego_min),
            "ghost_direct_p_dir_min": float(cfg.ghost_direct_p_dir_min),
            "ghost_depth_no_threshold": float(cfg.ghost_depth_no_threshold),
            "ghost_depth_yes_threshold": float(cfg.ghost_depth_yes_threshold),
            "ghost_negative_ratio": float(cfg.ghost_negative_ratio),
        },
    }
    (subset_root / "manifest.json").write_text(json.dumps(subset_counts, indent=2, default=str), encoding="utf-8")
    print_header(f"Ghost-validated subset: {label}")
    print(json.dumps(subset_counts, indent=2, default=str), flush=True)

    run(
        [
            paths.py,
            paths.dataset_tools,
            "validate",
            "--point-csv",
            out_csv,
            "--dataset-root",
            subset_root,
            "--session-prefix",
            "session_",
            "--allow-train-only",
        ],
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )
    return subset_root

def summarize_label_file(root: Path, prefix: str, label_csv_name: str) -> None:
    rows = 0
    cols_seen: set[str] = set()
    accum_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for session_dir in sorted(root.glob(f"{prefix}*")):
        csv_path = session_dir / label_csv_name
        if not _csv_has_data_rows(csv_path):
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        rows += len(df)
        cols_seen.update(df.columns)
        if "accumulatable_label" in df.columns:
            for key, value in df["accumulatable_label"].astype(str).str.lower().value_counts().to_dict().items():
                accum_counts[key] = accum_counts.get(key, 0) + int(value)
        if "depth_teacher_reason" in df.columns:
            for key, value in df["depth_teacher_reason"].astype(str).value_counts().head(10).to_dict().items():
                reason_counts[key] = reason_counts.get(key, 0) + int(value)
    ghost_cols = sorted(c for c in cols_seen if c in {
        "ghost_score",
        "ghost_score_combined",
        "static_confidence",
        "ego_doppler_residual_mps",
        "depth_teacher_acc_label",
        "depth_teacher_weight",
        "accumulatable_label",
        "return_type_label",
    })
    print(f"ghost label source: root={root} label={label_csv_name} rows={rows}", flush=True)
    print("ghost/evidence columns:", ghost_cols, flush=True)
    print("accumulatable counts:", accum_counts, flush=True)
    print("depth teacher reasons:", reason_counts, flush=True)


def materialize_ghost_teacher_processing_root(
    cfg: Config,
    paths: Paths,
    processing_root: Path,
    prefix: str,
    dataset_tag: str,
) -> tuple[Path, str]:
    processing_root = Path(processing_root)
    if not cfg.enable_ghost_teacher:
        ensure_usable_label_root(
            processing_root,
            prefix,
            "labeled_radar_points_v4.csv",
            context=f"{dataset_tag} raw labels",
            require_radar_tensors=True,
        )
        return processing_root, "labeled_radar_points_v4.csv"

    sessions = sorted(p for p in processing_root.glob(f"{prefix}*") if p.is_dir())
    if not sessions:
        raise RuntimeError(f"No sessions found for ghost-teacher labeling: {processing_root} {prefix}")

    print(f"Ghost-teacher stage for {dataset_tag}: {len(sessions)} session(s)", flush=True)
    run(
        [
            paths.py,
            paths.spatiotemporal_tools,
            "fuse-labels",
            "--processing-root",
            processing_root,
            "--in-place",
            "--session-prefix",
            prefix,
            "--output-csv-name",
            GHOST_LABEL_CSV_NAME,
        ],
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )

    summarize_label_file(processing_root, prefix, GHOST_LABEL_CSV_NAME)
    ensure_usable_label_root(
        processing_root,
        prefix,
        GHOST_LABEL_CSV_NAME,
        context=f"{dataset_tag} spatiotemporal labels",
        require_radar_tensors=True,
    )

    ghost_stage_root = paths.rolling_dataset_root / f"ghost_teacher_{dataset_tag}"
    depth_root = ghost_stage_root / "depth_correspondence"
    fused_root = ghost_stage_root / "processing"
    if depth_root.exists():
        shutil.rmtree(depth_root)
    if fused_root.exists():
        shutil.rmtree(fused_root)

    depth_cmd: list[object] = [
        paths.py,
        paths.depth_correspondence_tools,
        "annotate-root",
        "--processing-root",
        processing_root,
        "--out-root",
        depth_root,
        "--session-glob",
        f"{prefix}*",
        "--max-time-diff-ms",
        "35",
        "--foreground-occlusion-m",
        "0.20",
        "--depth-patch-r",
        "2",
    ]
    if paths.extrinsics_json.exists():
        depth_cmd.extend(["--extrinsics-json", paths.extrinsics_json])

    try:
        run(depth_cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: depth correspondence failed for {dataset_tag}; "
            f"falling back to spatiotemporal fused labels. error={exc}",
            flush=True,
        )
        return processing_root, GHOST_LABEL_CSV_NAME

    depth_summary_csv = depth_root / "depth_correspondence_summary.csv"
    if depth_summary_csv.exists():
        depth_summary = pd.read_csv(depth_summary_csv)
        keep = [
            c
            for c in [
                "session",
                "status",
                "depth_teacher_eligible",
                "failure_reason",
                "projection_valid_rate",
                "depth_available_rate",
                "depth_match_rate",
                "ok_points",
                "points",
            ]
            if c in depth_summary.columns
        ]
        print()
        print("Depth teacher QA summary:", flush=True)
        print(depth_summary[keep].to_string(index=False), flush=True)
        if "depth_teacher_eligible" in depth_summary.columns:
            eligible = depth_summary["depth_teacher_eligible"].astype(str).str.lower().isin({"true", "1", "yes", "y"}).sum()
            print(f"Depth-teacher eligible sessions: {int(eligible)}/{len(depth_summary)}", flush=True)

    fusion_cmd: list[object] = [
        paths.py,
        paths.ghost_teacher_fusion,
        "fuse-root",
        "--processing-root",
        processing_root,
        "--depth-root",
        depth_root,
        "--out-root",
        fused_root,
        "--session-glob",
        f"{prefix}*",
        "--label-csv-name",
        GHOST_LABEL_CSV_NAME,
    ]
    if cfg.copy_ghost_assets:
        fusion_cmd.append("--copy-assets")

    try:
        run(fusion_cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    except subprocess.CalledProcessError as exc:
        print(
            f"WARNING: ghost_teacher_fusion failed for {dataset_tag}; "
            f"falling back to spatiotemporal fused labels. error={exc}",
            flush=True,
        )
        return processing_root, GHOST_LABEL_CSV_NAME

    summarize_label_file(fused_root, prefix, GHOST_LABEL_CSV_NAME)
    fused_good, fused_bad = usable_session_dirs(
        fused_root,
        prefix,
        GHOST_LABEL_CSV_NAME,
        require_radar_tensors=True,
    )
    print(
        f"Ghost-depth fusion usable sessions for {dataset_tag}: "
        f"good={len(fused_good)} bad={len(fused_bad)}",
        flush=True,
    )
    if fused_bad:
        quarantine_bad_sessions(fused_root, fused_bad, f"{dataset_tag}_ghost_depth_fusion")
        fused_good, _ = usable_session_dirs(
            fused_root,
            prefix,
            GHOST_LABEL_CSV_NAME,
            require_radar_tensors=True,
        )

    if fused_good:
        print(f"Using ghost-depth fused root for {dataset_tag}: {fused_root}", flush=True)
        return fused_root, GHOST_LABEL_CSV_NAME

    print(
        f"WARNING: ghost_teacher_fusion produced zero usable session folders for {dataset_tag}. "
        f"This usually means depth correspondence was ineligible for the selected sessions. "
        f"Falling back to the in-place spatiotemporal fused labels at {processing_root}.",
        flush=True,
    )
    ensure_usable_label_root(
        processing_root,
        prefix,
        GHOST_LABEL_CSV_NAME,
        context=f"{dataset_tag} fallback spatiotemporal labels",
        require_radar_tensors=True,
    )
    return processing_root, GHOST_LABEL_CSV_NAME


def build_source_dataset(
    cfg: Config,
    paths: Paths,
    tag: str,
    processing_root: Path,
    prefix: str,
    holdout_last: int,
    *,
    dataset_tag: str | None = None,
    skip_validate: bool = False,
) -> Path:
    dataset_tag = dataset_tag or tag
    label_root, label_csv_name = materialize_ghost_teacher_processing_root(cfg, paths, processing_root, prefix, dataset_tag)
    ensure_usable_label_root(
        label_root,
        prefix,
        label_csv_name,
        context=f"{dataset_tag} final label root",
        require_radar_tensors=True,
    )

    points_dir = paths.rolling_dataset_root / f"branch1_{dataset_tag}_directness_true_ra_v1" / "points"
    patch_dir = paths.rolling_dataset_root / f"branch1_{dataset_tag}_directness_true_ra_v1" / "rdra_patch_dataset"
    four_dir = paths.rolling_dataset_root / f"branch1_4class_depth_directness_{dataset_tag}_true_ra_v1" / "rdra_patch_dataset"
    points_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)
    four_dir.mkdir(parents=True, exist_ok=True)

    run_build_points_with_recovery(
        cfg,
        paths,
        label_root=label_root,
        points_dir=points_dir,
        holdout_last=holdout_last,
        prefix=prefix,
        label_csv_name=label_csv_name,
        dataset_tag=dataset_tag,
    )
    run(
        build_patches_cmd(cfg, paths, points_dir / "points.csv", label_root, patch_dir, prefix),
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )
    assert_ra_patch_dataset(patch_dir / "points_with_rd_patch_index.csv", f"{dataset_tag} patch dataset", cfg)
    run(
        [
            paths.py,
            paths.fourclass_tools,
            "label-dataset",
            "--source-csv",
            patch_dir / "points_with_rd_patch_index.csv",
            "--source-root",
            patch_dir,
            "--output-dir",
            four_dir,
            "--allowed-session-prefix",
            prefix,
            "--depth-teacher-csv",
            points_dir / "points.csv",
            "--use-accumulatable-no",
        ],
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )
    assert_ra_patch_dataset(four_dir / "points_with_rd_patch_index.csv", f"{dataset_tag} four-class dataset", cfg)
    if not skip_validate:
        cmd: list[object] = [
            paths.py,
            paths.dataset_tools,
            "validate",
            "--point-csv",
            four_dir / "points_with_rd_patch_index.csv",
            "--dataset-root",
            four_dir,
            "--session-prefix",
            prefix,
        ]
        if holdout_last == 0:
            cmd.append("--allow-train-only")
        run(cmd, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    return four_dir


def force_dataset_split(dataset_root: Path, split: str) -> None:
    csv_path = dataset_root / "points_with_rd_patch_index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df["split"] = split
    df.to_csv(csv_path, index=False)
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        manifest["split_rewritten_to"] = split
        manifest["split_counts"] = {split: int(len(df))}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_external_validation_dataset(cfg: Config, paths: Paths) -> Path:
    val_root = paths.rolling_val_root / "rdra_patch_dataset"
    val_csv = val_root / "points_with_rd_patch_index.csv"
    if val_csv.exists() and _dataset_has_ghost_inputs(val_root) and not cfg.force_rebuild_external_val:
        print("Reusing ghost-aware curated external validation dataset:", val_csv, flush=True)
        force_dataset_split(val_root, "val")
        debug_dataset_csv(val_root, "external validation reuse")
        return create_ghost_validated_subset_dataset(
            cfg,
            paths,
            val_root,
            "external validation reuse",
            require_both_classes=True,
        )
    if paths.rolling_val_root.exists():
        shutil.rmtree(paths.rolling_val_root)

    records = validation_session_records(cfg, paths)
    if not records:
        raise RuntimeError(
            "Rolling training requires at least one curated session in the configured external-validation splits. "
            f"splits={cfg.external_val_splits}"
        )

    dataset_ids = sorted({group for group, _name, _uri in records})
    group_to_root = {dataset_id: paths.rolling_val_root / "processing" / dataset_id for dataset_id in dataset_ids}
    for root in group_to_root.values():
        root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(records)} curated external-validation session archive(s).", flush=True)
    _download_records_to_roots(cfg, paths, records, group_to_root)

    old_dataset_root = paths.rolling_dataset_root
    paths.rolling_dataset_root = paths.rolling_val_root / "datasets"
    try:
        source_dirs: list[tuple[str, Path]] = []
        for dataset_id in dataset_ids:
            processing_root = group_to_root[dataset_id]
            if not processing_root.exists():
                continue
            source_dirs.append(
                (
                    f"ext_{_safe_name(dataset_id)}",
                    build_source_dataset(
                        cfg,
                        paths,
                        dataset_id,
                        processing_root,
                        cfg.session_prefix,
                        0,
                        dataset_tag=f"external_{_safe_name(dataset_id)}_val",
                        skip_validate=True,
                    ),
                )
            )
        combined_source = build_combined_batch_dataset(cfg, paths, 0, source_dirs)
        debug_dataset_csv(combined_source, "external validation combined")
        if val_root.exists():
            shutil.rmtree(val_root)
        shutil.move(str(combined_source), str(val_root))
        debug_dataset_csv(val_root, "external validation built")
        subset_root = create_ghost_validated_subset_dataset(
            cfg,
            paths,
            val_root,
            "external validation built",
            require_both_classes=True,
        )
    finally:
        paths.rolling_dataset_root = old_dataset_root
        for root in group_to_root.values():
            if root.exists():
                shutil.rmtree(root)
    print("Built ghost-aware curated external validation dataset:", val_root, flush=True)
    print("Built ghost-validated external subset:", subset_root, flush=True)
    return subset_root


def build_combined_batch_dataset(cfg: Config, paths: Paths, batch_id: int, source_dirs: list[tuple[str, Path]]) -> Path:
    combined_root = paths.rolling_dataset_root / f"batch_{batch_id:04d}" / "combined_branch1_4class_depth_directness_true_ra_v1"
    combined_rd_root = combined_root / "rdra_patch_dataset"
    combined_rd_root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    manifest_sources: dict[str, str] = {}
    for tag, root_dir in source_dirs:
        csv_path = root_dir / "points_with_rd_patch_index.csv"
        patches_dir = root_dir / "patches"
        link_path = combined_rd_root / f"patches_{tag}"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        os.symlink(patches_dir, link_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df["rd_patch_shard"] = df["rd_patch_shard"].astype(str).str.replace("patches/", f"patches_{tag}/", regex=False)
        if "ra_patch_shard" in df.columns:
            df["ra_patch_shard"] = df["ra_patch_shard"].astype(str).str.replace("patches/", f"patches_{tag}/", regex=False)
        df["split"] = "train"
        frames.append(df)
        manifest_sources[tag] = str(csv_path)
    if not frames:
        raise RuntimeError(f"Batch {batch_id} did not produce any source datasets.")

    all_cols: list[str] = []
    for df in frames:
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)
    for idx, df in enumerate(frames):
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0 if col == "point_label_weight" else ""
        frames[idx] = df[all_cols]

    combined = pd.concat(frames, ignore_index=True)
    combined["point_label_weight"] = pd.to_numeric(
        combined.get("point_label_weight", 1.0),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    combined_csv = combined_rd_root / "points_with_rd_patch_index.csv"
    combined.to_csv(combined_csv, index=False)
    (combined_rd_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "rolling_batch_branch1_4class_depth_directness_true_ra_colab",
                "batch_id": int(batch_id),
                "label_col": "bucket_4class",
                "sample_weight_col": "point_label_weight",
                "sources": manifest_sources,
                "output_csv": "points_with_rd_patch_index.csv",
                "n_rows": int(len(combined)),
                "split_counts": {k: int(v) for k, v in combined["split"].value_counts().to_dict().items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run(
        [
            paths.py,
            paths.dataset_tools,
            "validate",
            "--point-csv",
            combined_csv,
            "--dataset-root",
            combined_rd_root,
            "--session-prefix",
            "session_2026-",
            "--allow-train-only",
        ],
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )
    print("Batch dataset:", combined_csv, flush=True)
    print(combined["bucket_4class"].value_counts(), flush=True)
    return create_ghost_validated_subset_dataset(
        cfg,
        paths,
        combined_rd_root,
        f"training batch {batch_id}",
        require_both_classes=True,
    )


def maybe_restore_train_checkpoint(cfg: Config, paths: Paths) -> None:
    if (paths.train_out / "last_model.pt").exists():
        return
    paths.train_out.mkdir(parents=True, exist_ok=True)
    src = f"{cfg.gcs_final_output_root}/{paths.train_out.name}"
    print(f"Restoring checkpoint with gcloud storage: {src} -> {paths.train_out}", flush=True)
    try:
        gcloud_storage_rsync(src, paths.train_out, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    except subprocess.CalledProcessError:
        print("No prior checkpoint restored. Starting from scratch.", flush=True)


def sync_train_checkpoint(cfg: Config, paths: Paths) -> None:
    dst = f"{cfg.gcs_final_output_root}/{paths.train_out.name}"
    print(f"Syncing checkpoint with gcloud storage: {paths.train_out} -> {dst}", flush=True)
    gcloud_storage_rsync(paths.train_out, dst, cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)


def stream_training(cmd: list[object], *, paths: Paths, cfg: Config) -> None:
    argv = [str(x) for x in cmd]
    print("\n$ " + " ".join(shlex.quote(x) for x in argv), flush=True)
    if cfg.dry_run:
        return

    log_dir = paths.train_out / "debug_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_command_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print("Full command log:", log_path, flush=True)

    proc = subprocess.Popen(
        argv,
        cwd=str(paths.root),
        env=paths.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    keep = (
        "epoch", "bal_acc", "accuracy", "acc", "loss", "val", "f1", "precision", "recall",
        "error", "exception", "traceback", "warning", "dataset", "rows", "class", "shape",
    )
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("$ " + " ".join(shlex.quote(x) for x in argv) + "\n")
        log_fh.write(f"cwd: {paths.root}\n")
        log_fh.flush()
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            tail.append(line.rstrip("\n"))
            if len(tail) > 200:
                tail = tail[-200:]
            if (not cfg.compact_train_output) or any(token in line.lower() for token in keep):
                print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print_header("Training Command Failed")
        print("return code:", proc.returncode, flush=True)
        print("full log:", log_path, flush=True)
        print("last 200 output lines:", flush=True)
        for line in tail:
            print(line, flush=True)
        raise subprocess.CalledProcessError(proc.returncode, argv)


def train_one_batch(cfg: Config, paths: Paths, batch_id: int, records: list[tuple[str, str, str]], external_val_root: Path) -> None:
    clear_rolling_training_dirs(paths)
    counts = download_session_records(cfg, paths, records)
    print(f"Rolling batch {batch_id}: downloaded {counts}", flush=True)
    source_dirs: list[tuple[str, Path]] = []
    for group in cfg.dataset_groups:
        if counts.get(group, 0) <= 0:
            continue
        safe_group = _safe_name(group)
        processing_root = dataset_processing_root(paths, group)
        source_dirs.append(
            (
                f"b{batch_id:04d}_{safe_group}",
                build_source_dataset(
                    cfg,
                    paths,
                    group,
                    processing_root,
                    cfg.session_prefix,
                    0,
                    dataset_tag=f"b{batch_id:04d}_{safe_group}",
                    skip_validate=True,
                ),
            )
        )
    if not source_dirs:
        raise RuntimeError(f"Batch {batch_id} downloaded no usable source datasets. records={records[:5]!r}")

    try:
        batch_root = build_combined_batch_dataset(cfg, paths, batch_id, source_dirs)
    except GhostSubsetEmpty as exc:
        print(f"[SKIP] Batch {batch_id}: {exc}", flush=True)
        clear_rolling_training_dirs(paths)
        return
    debug_dataset_csv(batch_root, f"ghost-validated training batch {batch_id}")
    debug_dataset_csv(external_val_root, "external validation")

    cmd: list[object] = [
        paths.py,
        paths.train_script,
        "--dataset",
        batch_root / "points_with_rd_patch_index.csv",
        "--dataset-root",
        batch_root,
        "--external-val-dataset",
        external_val_root / "points_with_rd_patch_index.csv",
        "--external-val-root",
        external_val_root,
        "--out-dir",
        paths.train_out,
        "--label-col",
        "ghost_binary_label",
        "--bucket-order",
        "direct_return,ghost_return",
        "--feature-mode",
        "directness_soft_structural",
        "--sample-weight-col",
        "point_label_weight",
        "--loss-kind",
        "focal",
        "--focal-gamma",
        "1.5",
        "--balanced-window-sampler",
        "--window-sampler-power",
        "0.5",
        "--window-sampler-max-weight",
        "20",
        "--best-metric",
        "external_val_bal_acc",
        "--windows-per-batch",
        cfg.train_windows_per_batch,
        "--epochs",
        cfg.train_epochs_per_session_batch,
        "--lr",
        "3e-4",
        "--weight-decay",
        "1e-4",
        "--device",
        "auto",
    ]
    append_training_resource_args(cmd, paths, cfg)
    last_ckpt = paths.train_out / "last_model.pt"
    if last_ckpt.exists():
        cmd.extend(["--resume-checkpoint", last_ckpt, "--feature-stats-checkpoint", last_ckpt])
    stream_training(cmd, paths=paths, cfg=cfg)
    paths.train_out.mkdir(parents=True, exist_ok=True)
    with (paths.train_out / "batch_progress.log").open("a", encoding="utf-8") as fh:
        fh.write(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] completed batch={batch_id} "
            f"records={len(records)} counts={counts} epochs_per_batch={cfg.train_epochs_per_session_batch}\n"
        )
    sync_train_checkpoint(cfg, paths)
    clear_rolling_training_dirs(paths)


def run_rolling_training(cfg: Config, paths: Paths, *, max_batches: int | None = None) -> None:
    print_header("Rolling Training")
    debug_preflight(cfg, paths)
    records = session_records(cfg, paths)
    batches = list(chunks(records, cfg.session_batch_size))
    if max_batches is not None:
        batches = batches[:max_batches]
    print(
        f"Curated ghost rolling training: {len(records)} train sessions in {len(batches)} batches of up to {cfg.session_batch_size}",
        flush=True,
    )
    print("training record sample:", records[:5], flush=True)
    maybe_restore_train_checkpoint(cfg, paths)
    external_val_root = build_external_validation_dataset(cfg, paths)
    for pass_idx in range(cfg.train_epochs):
        print(f"\n===== streaming pass {pass_idx + 1}/{cfg.train_epochs} =====", flush=True)
        for batch_idx, batch_records in batches:
            global_batch_id = pass_idx * len(batches) + batch_idx + 1
            print(f"--- Batch {global_batch_id} ---", flush=True)
            train_one_batch(cfg, paths, global_batch_id, batch_records, external_val_root)
    print("\nRolling training complete. Latest model output:", paths.train_out, flush=True)


def rewrite_patch_paths(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    out = df.copy()
    out["rd_patch_shard"] = out["rd_patch_shard"].astype(str).str.replace("patches/", f"patches_{tag}/", regex=False)
    if "ra_patch_shard" in out.columns:
        out["ra_patch_shard"] = out["ra_patch_shard"].astype(str).str.replace("patches/", f"patches_{tag}/", regex=False)
    return out


def build_full_gcs_dataset(cfg: Config, paths: Paths) -> Path:
    print_header("Build Full GCS Dataset")
    full_root = cfg.drive_root / "colab_outputs" / "full_branch1_rdra_dataset_v3" / "rdra_patch_dataset"
    if full_root.exists():
        raise FileExistsError(f"Dataset already exists at {full_root}; delete it first to rebuild.")
    full_root.mkdir(parents=True, exist_ok=True)
    all_records = session_records(cfg, paths)
    batches = list(chunks(all_records, cfg.session_batch_size))
    staging_root = cfg.work_root / "_gcs_build_staging"
    local_ds_root = cfg.work_root / "_gcs_build_ds"
    all_frames: list[pd.DataFrame] = []

    for batch_idx, batch in batches:
        print(f"\n--- Processing full-dataset batch {batch_idx + 1}/{len(batches)} ---", flush=True)
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(local_ds_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        local_ds_root.mkdir(parents=True, exist_ok=True)
        counts = _download_records_to_roots(
            cfg,
            paths,
            batch,
            {group: staging_root for group in cfg.dataset_groups},
        )

        old_dataset_root = paths.rolling_dataset_root
        paths.rolling_dataset_root = local_ds_root
        try:
            batch_tags: list[tuple[str, Path]] = []
            for group in cfg.dataset_groups:
                if not counts.get(group, 0):
                    continue
                tag = f"b{batch_idx:04d}_{_safe_name(group)}"
                ds_dir = build_source_dataset(cfg, paths, tag, staging_root, cfg.session_prefix, 0, dataset_tag=tag, skip_validate=True)
                batch_tags.append((tag, ds_dir))
            for tag, ds_dir in batch_tags:
                csv_path = ds_dir / "points_with_rd_patch_index.csv"
                patches_dir = ds_dir / "patches"
                if not csv_path.exists() or not patches_dir.exists():
                    raise FileNotFoundError(f"Missing outputs for {tag}: {csv_path} / {patches_dir}")
                gcs_patches_target = full_root / f"patches_{tag}"
                gcloud_storage_rsync(
                    patches_dir,
                    gcs_mount_path_to_uri(cfg, gcs_patches_target),
                    cwd=paths.root,
                    env=paths.env,
                    dry_run=cfg.dry_run,
                    delete_unmatched=True,
                )
                df = rewrite_patch_paths(pd.read_csv(csv_path, low_memory=False), tag)
                df["split"] = "train"
                all_frames.append(df)
        finally:
            paths.rolling_dataset_root = old_dataset_root
            shutil.rmtree(local_ds_root, ignore_errors=True)
            shutil.rmtree(staging_root, ignore_errors=True)

    if not all_frames:
        raise RuntimeError("No data produced from full dataset batches.")

    all_cols: list[str] = []
    for df in all_frames:
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)
    for idx, df in enumerate(all_frames):
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0.0 if col == "point_label_weight" else ""
        all_frames[idx] = df[all_cols]
    combined = pd.concat(all_frames, ignore_index=True)
    combined["point_label_weight"] = pd.to_numeric(combined.get("point_label_weight", 1.0), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    combined_local_csv = cfg.work_root / "points_with_rd_patch_index.csv"
    combined.to_csv(combined_local_csv, index=False)
    gcloud_storage_cp(
        combined_local_csv,
        gcs_mount_path_to_uri(cfg, full_root / "points_with_rd_patch_index.csv"),
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )
    combined_local_csv.unlink(missing_ok=True)
    print("Full GCS dataset build complete:", full_root, flush=True)
    return full_root


def run_full_gcs_training(cfg: Config, paths: Paths) -> None:
    print_header("Full GCS Training")
    full_root = cfg.drive_root / "colab_outputs" / "full_branch1_rdra_dataset" / "rdra_patch_dataset"
    if not full_root.exists():
        raise FileNotFoundError(f"Dataset not found at {full_root}")
    external_val_root = build_external_validation_dataset(cfg, paths)
    out_dir = paths.root / "_train_output" / "kpconv_branch1_4class_direct_gcs_v1"
    cmd: list[object] = [
        paths.py,
        paths.train_script,
        "--dataset",
        full_root / "points_with_rd_patch_index.csv",
        "--dataset-root",
        full_root,
        "--external-val-dataset",
        external_val_root / "points_with_rd_patch_index.csv",
        "--external-val-root",
        external_val_root,
        "--out-dir",
        out_dir,
        "--label-col",
        "bucket_4class",
        "--bucket-order",
        "structure,floor,human_candidate,ghost_return",
        "--feature-mode",
        "directness_soft_structural",
        "--sample-weight-col",
        "point_label_weight",
        "--loss-kind",
        "focal",
        "--focal-gamma",
        "1.5",
        "--balanced-window-sampler",
        "--window-sampler-power",
        "0.5",
        "--window-sampler-max-weight",
        "20",
        "--best-metric",
        "external_val_bal_acc",
        "--windows-per-batch",
        cfg.train_windows_per_batch,
        "--epochs",
        cfg.train_epochs,
        "--lr",
        "3e-4",
        "--weight-decay",
        "1e-4",
        "--device",
        "auto",
    ]
    append_training_resource_args(cmd, paths, cfg)
    last_ckpt = out_dir / "last_model.pt"
    if last_ckpt.exists():
        cmd.extend(["--resume-checkpoint", last_ckpt, "--feature-stats-checkpoint", last_ckpt])
    stream_training(cmd, paths=paths, cfg=cfg)
    backup = cfg.drive_root / "colab_outputs" / "final_branch1_rdra_retrain" / "models" / out_dir.name
    gcloud_storage_rsync(
        out_dir,
        gcs_mount_path_to_uri(cfg, backup),
        cwd=paths.root,
        env=paths.env,
        dry_run=cfg.dry_run,
    )


def smoke(cfg: Config, paths: Paths) -> None:
    print_header("Smoke")
    require_paths(paths)
    print("GCS root:", cfg.gcs_root_uri, flush=True)
    print("Code root:", paths.code, flush=True)
    print("BRANCH1:", paths.branch1, flush=True)
    print("TRAIN_OUT:", paths.train_out, flush=True)
    print("DATASET_IDS:", list(cfg.dataset_groups), flush=True)
    print("TRAIN_SPLITS:", list(cfg.train_splits), flush=True)
    print("EXTERNAL_VAL_SPLITS:", list(cfg.external_val_splits), flush=True)
    print("SESSION_PREFIX:", repr(cfg.session_prefix), flush=True)
    print("A100 TRAIN WINDOWS PER BATCH:", cfg.train_windows_per_batch, flush=True)
    print("A100 TRAIN NUM WORKERS:", cfg.train_num_workers, flush=True)
    print("A100 TRAIN POINTS PER FRAME:", cfg.train_points_per_frame, flush=True)
    print("A100 TRAIN EMB DIMS:", cfg.train_emb_dims, flush=True)
    print("A100 TRAIN TEMPORAL LAYERS/HEADS:", cfg.train_temporal_layers, cfg.train_temporal_heads, flush=True)
    print("GHOST VALIDATED SUBSET:", {
        "negative_ratio": cfg.ghost_negative_ratio,
        "p_dir_max": cfg.ghost_p_dir_max,
        "ego_abs_z_min": cfg.ghost_ego_abs_z_min,
        "p_ego_min": cfg.ghost_p_ego_min,
        "direct_p_dir_min": cfg.ghost_direct_p_dir_min,
        "depth_no_threshold": cfg.ghost_depth_no_threshold,
        "depth_yes_threshold": cfg.ghost_depth_yes_threshold,
    }, flush=True)
    run([paths.py, paths.dataset_tools, "build-points", "--help"], cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    run([paths.py, paths.dataset_tools, "validate", "--help"], cwd=paths.root, env=paths.env, dry_run=cfg.dry_run)
    records = session_records(cfg, paths)
    val_records = validation_session_records(cfg, paths)
    print("training session records:", len(records), flush=True)
    print("validation session records:", len(val_records), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stages",
        nargs="*",
        default=["rolling"],
        choices=["setup-code", "smoke", "rolling", "build-full-gcs", "train-full-gcs"],
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--gcs-mount-point", type=Path, default=Path("/content/gcs"))
    parser.add_argument("--work-root", type=Path, default=Path("/content/work"))
    parser.add_argument("--code-root", type=Path, default=Path("/content/work/code"))
    parser.add_argument("--session-batch-size", type=int, default=30)
    parser.add_argument("--session-download-batch-size", type=int, default=8)
    parser.add_argument("--gcloud-process-count", type=int, default=4)
    parser.add_argument("--gcloud-thread-count", type=int, default=32)
    parser.add_argument("--train-epochs", type=int, default=60)
    parser.add_argument("--train-epochs-per-session-batch", type=int, default=1)
    parser.add_argument("--train-windows-per-batch", type=int, default=256)
    parser.add_argument("--train-num-workers", type=int, default=8)
    parser.add_argument("--train-points-per-frame", type=int, default=128)
    parser.add_argument("--train-emb-dims", type=int, default=512)
    parser.add_argument("--train-temporal-layers", type=int, default=4)
    parser.add_argument("--train-temporal-heads", type=int, default=8)
    parser.add_argument("--train-gate-hidden", type=int, default=256)
    parser.add_argument("--train-rd-patch-embed-dim", type=int, default=64)
    parser.add_argument("--train-k", type=int, default=32)
    parser.add_argument("--train-n-kernel-points", type=int, default=31)
    parser.add_argument("--train-pin-memory", action="store_true")
    parser.add_argument("--ghost-negative-ratio", type=float, default=3.0)
    parser.add_argument("--ghost-p-dir-max", type=float, default=0.20)
    parser.add_argument("--ghost-ego-abs-z-min", type=float, default=2.0)
    parser.add_argument("--ghost-p-ego-min", type=float, default=0.50)
    parser.add_argument("--ghost-direct-p-dir-min", type=float, default=0.70)
    parser.add_argument("--ghost-depth-no-threshold", type=float, default=0.20)
    parser.add_argument("--ghost-depth-yes-threshold", type=float, default=0.70)
    parser.add_argument("--ghost-min-positive-rows", type=int, default=1)
    parser.add_argument("--ghost-min-negative-rows", type=int, default=1)
    parser.add_argument("--ghost-random-seed", type=int, default=42)
    parser.add_argument("--validation-session-count", type=int, default=-1)
    parser.add_argument("--dataset-groups", default=",".join(DEFAULT_DATASET_GROUPS))
    parser.add_argument("--external-val-group", default=DEFAULT_EXTERNAL_VAL_GROUP)
    parser.add_argument("--session-prefix", default=DEFAULT_SESSION_PREFIX)
    parser.add_argument("--train-quality-tiers", default=",".join(DEFAULT_TRAIN_QUALITY_TIERS))
    parser.add_argument("--external-val-quality-tiers", default=",".join(DEFAULT_EXTERNAL_VAL_QUALITY_TIERS))
    parser.add_argument("--train-splits", default=",".join(DEFAULT_TRAIN_SPLITS))
    parser.add_argument("--external-val-splits", default=",".join(DEFAULT_EXTERNAL_VAL_SPLITS))
    parser.add_argument("--train-dataset-ids", default="")
    parser.add_argument("--external-val-dataset-ids", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--only", default=None, help=argparse.SUPPRESS)  # accepted for old notebook cells; ignored.
    parser.add_argument("--disable-ghost-teacher", action="store_true")
    parser.add_argument("--copy-ghost-assets", action="store_true")
    parser.add_argument("--force-rebuild-external-val", action="store_true")
    parser.add_argument("--compact-train-output", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None, help="Debug limit for rolling training.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    dataset_groups = tuple(
        dict.fromkeys(
            group.strip().strip("/")
            for group in str(args.dataset_groups).split(",")
            if group.strip().strip("/")
        )
    )
    if not dataset_groups:
        raise RuntimeError("--dataset-groups must contain at least one folder name, e.g. dataset2,dataset4")

    external_val_group = str(args.external_val_group or DEFAULT_EXTERNAL_VAL_GROUP).strip().strip("/")
    if not external_val_group:
        raise RuntimeError("--external-val-group cannot be empty")

    train_dataset_ids = tuple(
        dict.fromkeys(
            dataset_id.strip()
            for dataset_id in str(args.train_dataset_ids).split(",")
            if dataset_id.strip()
        )
    )
    external_val_dataset_ids = tuple(
        dict.fromkeys(
            dataset_id.strip()
            for dataset_id in str(args.external_val_dataset_ids).split(",")
            if dataset_id.strip()
        )
    )
    if dataset_groups == ("auto",) and train_dataset_ids:
        dataset_groups = train_dataset_ids

    return Config(
        bucket=args.bucket,
        project=args.project,
        gcs_mount_point=args.gcs_mount_point,
        work_root=args.work_root,
        code_root=args.code_root,
        session_batch_size=args.session_batch_size,
        session_download_batch_size=args.session_download_batch_size,
        gcloud_process_count=args.gcloud_process_count,
        gcloud_thread_count=args.gcloud_thread_count,
        train_epochs=args.train_epochs,
        train_epochs_per_session_batch=args.train_epochs_per_session_batch,
        train_windows_per_batch=args.train_windows_per_batch,
        train_num_workers=args.train_num_workers,
        train_points_per_frame=args.train_points_per_frame,
        train_emb_dims=args.train_emb_dims,
        train_temporal_layers=args.train_temporal_layers,
        train_temporal_heads=args.train_temporal_heads,
        train_gate_hidden=args.train_gate_hidden,
        train_rd_patch_embed_dim=args.train_rd_patch_embed_dim,
        train_k=args.train_k,
        train_n_kernel_points=args.train_n_kernel_points,
        train_pin_memory=bool(args.train_pin_memory),
        ghost_negative_ratio=float(args.ghost_negative_ratio),
        ghost_p_dir_max=float(args.ghost_p_dir_max),
        ghost_ego_abs_z_min=float(args.ghost_ego_abs_z_min),
        ghost_p_ego_min=float(args.ghost_p_ego_min),
        ghost_direct_p_dir_min=float(args.ghost_direct_p_dir_min),
        ghost_depth_no_threshold=float(args.ghost_depth_no_threshold),
        ghost_depth_yes_threshold=float(args.ghost_depth_yes_threshold),
        ghost_min_positive_rows=int(args.ghost_min_positive_rows),
        ghost_min_negative_rows=int(args.ghost_min_negative_rows),
        ghost_random_seed=int(args.ghost_random_seed),
        validation_session_count=args.validation_session_count,
        dataset_groups=dataset_groups,
        external_val_group=external_val_group,
        train_quality_tiers=tuple(x.strip() for x in str(args.train_quality_tiers).split(",") if x.strip()),
        external_val_quality_tiers=tuple(x.strip() for x in str(args.external_val_quality_tiers).split(",") if x.strip()),
        train_splits=tuple(x.strip() for x in str(args.train_splits).split(",") if x.strip()),
        external_val_splits=tuple(x.strip() for x in str(args.external_val_splits).split(",") if x.strip()),
        train_dataset_ids=train_dataset_ids,
        external_val_dataset_ids=external_val_dataset_ids,
        session_prefix=str(args.session_prefix),
        debug=bool(args.debug),
        enable_ghost_teacher=not args.disable_ghost_teacher,
        copy_ghost_assets=args.copy_ghost_assets,
        force_rebuild_external_val=args.force_rebuild_external_val,
        compact_train_output=args.compact_train_output,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    cfg.work_root.mkdir(parents=True, exist_ok=True)
    configure_gcloud(cfg)
    if "setup-code" in args.stages:
        sync_code_from_gcs(cfg)
    paths = resolve_paths(cfg)
    require_paths(paths)
    for stage in args.stages:
        if stage == "setup-code":
            continue
        if stage == "smoke":
            smoke(cfg, paths)
        elif stage == "rolling":
            require_pandas()
            run_rolling_training(cfg, paths, max_batches=args.max_batches)
        elif stage == "build-full-gcs":
            require_pandas()
            build_full_gcs_dataset(cfg, paths)
        elif stage == "train-full-gcs":
            require_pandas()
            run_full_gcs_training(cfg, paths)
        else:
            raise ValueError(stage)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print_header("FATAL DRIVER ERROR")
        print(f"{type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
