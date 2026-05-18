#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hullumdr/prof/io"
CANON_DIR="$ROOT/LLM_ML/jetson_nav_pipeline/canon"
VENV_PY="$ROOT/.venv/bin/python"

MAY11_PROCESSING_ROOT="${MAY11_PROCESSING_ROOT:-${PROCESSING_ROOT:-$ROOT/LLM_ML/data/data5-11}}"
APR28_PROCESSING_ROOT="${APR28_PROCESSING_ROOT:-$ROOT/LLM_ML/data/data-4-30/Processing}"
MAY11_PREFIX="${MAY11_PREFIX:-session_2026-05-11_}"
APR28_PREFIX="${APR28_PREFIX:-session_2026-04-28_}"

MAY11_HOLDOUT_LAST="${MAY11_HOLDOUT_LAST:-5}"
APR28_HOLDOUT_LAST="${APR28_HOLDOUT_LAST:-5}"

ONEFORMER_MODEL="${ONEFORMER_MODEL:-tiny}"
ONEFORMER_EVERY_N="${ONEFORMER_EVERY_N:-1}"
SKIP_ONEFORMER="${SKIP_ONEFORMER:-0}"
ONEFORMER_EXPORT="$CANON_DIR/processing/noah_scripts/OneFormer/noah_export_labelmaps_from_processing_v2.py"
ONEFORMER_AUTOLABEL="$CANON_DIR/processing/noah_scripts/OneFormer/noah_autolabel_radar_using_synccsv_v4_depth_gated.py"
PREPARE_AUTOLABEL="$CANON_DIR/processing/prepare_autolabel.py"
MAY11_RADAR_TENSOR_EXPORT="$CANON_DIR/processing/export_radar_tensors_data511.py"
DATASET_TOOLS="$CANON_DIR/tools/dataset_tools.py"
FOURCLASS_TOOLS="$CANON_DIR/tools/branch1_fourclass_tools.py"
TRAIN_SCRIPT="$CANON_DIR/training/train_hybrid_rd_patch_kpconv.py"
EXTRINSICS_JSON="$CANON_DIR/config/radar_camera_extrinsics.json"

MAY11_POINTS_DIR="$ROOT/LLM_ML/data/branch1_may11_directness_v1/points"
MAY11_PATCH_DIR="$ROOT/LLM_ML/data/branch1_may11_directness_v1/rd_patch_dataset"
MAY11_FOURCLASS_DIR="$ROOT/LLM_ML/data/branch1_4class_depth_directness_may11_v1"

APR28_POINTS_DIR="$ROOT/LLM_ML/data/branch1_apr28_directness_v1/points"
APR28_PATCH_DIR="$ROOT/LLM_ML/data/branch1_apr28_directness_v1/rd_patch_dataset"
APR28_FOURCLASS_DIR="$ROOT/LLM_ML/data/branch1_4class_depth_directness_apr28_v1"

COMBINED_ROOT="${COMBINED_ROOT:-$ROOT/LLM_ML/data/branch1_4class_depth_directness_may11_apr28_v1}"
COMBINED_RD_ROOT="$COMBINED_ROOT/rd_patch_dataset"
COMBINED_CSV="$COMBINED_RD_ROOT/points_with_rd_patch_index.csv"
TRAIN_OUT_DIR="${TRAIN_OUT_DIR:-$CANON_DIR/models/kpconv_may11_apr28_directness_v1}"

TRAIN_FEATURE_MODE="${TRAIN_FEATURE_MODE:-directness_soft_structural}"
TRAIN_LABEL_COL="${TRAIN_LABEL_COL:-bucket_4class}"
TRAIN_BUCKET_ORDER="${TRAIN_BUCKET_ORDER:-structure,floor,human_candidate,ghost_return}"
TRAIN_SAMPLE_WEIGHT_COL="${TRAIN_SAMPLE_WEIGHT_COL:-point_label_weight}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-60}"
TRAIN_LR="${TRAIN_LR:-3e-4}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-1e-4}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
RUN_APR28_AUTOLABEL="${RUN_APR28_AUTOLABEL:-0}"
SKIP_RADAR_TENSOR_EXPORT="${SKIP_RADAR_TENSOR_EXPORT:-0}"

export PYTHONPATH="$ROOT:$CANON_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: missing venv python: $VENV_PY" >&2
  exit 1
fi

if [[ ! -d "$MAY11_PROCESSING_ROOT" ]]; then
  echo "ERROR: missing May 11 processing root: $MAY11_PROCESSING_ROOT" >&2
  exit 1
fi

if [[ ! -d "$APR28_PROCESSING_ROOT" ]]; then
  echo "ERROR: missing Apr 28 processing root: $APR28_PROCESSING_ROOT" >&2
  exit 1
fi

if [[ ! -f "$ONEFORMER_EXPORT" ]]; then
  echo "ERROR: missing OneFormer export script: $ONEFORMER_EXPORT" >&2
  exit 1
fi

if [[ ! -f "$ONEFORMER_AUTOLABEL" ]]; then
  echo "ERROR: missing OneFormer autolabel script: $ONEFORMER_AUTOLABEL" >&2
  exit 1
fi

if [[ ! -f "$EXTRINSICS_JSON" ]]; then
  echo "ERROR: missing extrinsics JSON: $EXTRINSICS_JSON" >&2
  exit 1
fi

mkdir -p "$MAY11_POINTS_DIR" "$MAY11_PATCH_DIR" "$MAY11_FOURCLASS_DIR"
mkdir -p "$APR28_POINTS_DIR" "$APR28_PATCH_DIR" "$APR28_FOURCLASS_DIR"
mkdir -p "$COMBINED_RD_ROOT" "$TRAIN_OUT_DIR"

count_sessions() {
  local root="$1"
  local prefix="$2"
  find "$root" -maxdepth 1 -type d -name "${prefix}*" | wc -l | tr -d ' '
}

is_truthy() {
  [[ "${1:-0}" =~ ^(1|true|yes|on)$ ]]
}

count_training_ready_sessions() {
  local root="$1"
  local prefix="$2"
  "$VENV_PY" - "$root" "$prefix" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]
ready = 0
for session_dir in sorted(root.glob(f"{prefix}*")):
    if not session_dir.is_dir():
        continue
    name = session_dir.name
    label_csv = session_dir / "labeled_radar_points_v4.csv"
    try:
        has_labels = sum(1 for _ in label_csv.open("r", encoding="utf-8", errors="replace")) > 1
    except OSError:
        has_labels = False
    required = [
        session_dir / f"{name}.csv",
        label_csv,
        session_dir / f"{name}_radar_tensors.npz",
        session_dir / "hybrid_rd" / "manifest.json",
    ]
    if has_labels and all(p.exists() for p in required):
        ready += 1
print(ready)
PY
}

count_labeled_missing_radar_tensors() {
  local root="$1"
  local prefix="$2"
  "$VENV_PY" - "$root" "$prefix" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]
missing = 0
for session_dir in sorted(root.glob(f"{prefix}*")):
    if not session_dir.is_dir():
        continue
    label_csv = session_dir / "labeled_radar_points_v4.csv"
    try:
        has_labels = sum(1 for _ in label_csv.open("r", encoding="utf-8", errors="replace")) > 1
    except OSError:
        has_labels = False
    if has_labels and not (session_dir / f"{session_dir.name}_radar_tensors.npz").exists():
        missing += 1
print(missing)
PY
}

run_oneformer_batch() {
  if is_truthy "$SKIP_ONEFORMER"; then
    echo "== OneFormer export skipped (SKIP_ONEFORMER=$SKIP_ONEFORMER) =="
    return
  fi

  local done=0
  local skipped=0
  local failed=0
  mapfile -t sessions < <(find "$MAY11_PROCESSING_ROOT" -maxdepth 1 -type d -name "${MAY11_PREFIX}*" | sort)
  echo "== OneFormer export =="
  echo "Processing root: $MAY11_PROCESSING_ROOT"
  echo "Session prefix : $MAY11_PREFIX"
  echo "Sessions found : ${#sessions[@]}"

  for session_dir in "${sessions[@]}"; do
    local session_name
    session_name="$(basename "$session_dir")"
    if [[ -f "$session_dir/seg_timestamps.csv" ]]; then
      echo "[SKIP] $session_name (seg_timestamps.csv present)"
      skipped=$((skipped + 1))
      continue
    fi

    echo "[RUN ] $session_name"
    if "$VENV_PY" "$ONEFORMER_EXPORT" \
      --processing_session_dir "$session_dir" \
      --model "$ONEFORMER_MODEL" \
      --every_n "$ONEFORMER_EVERY_N"; then
      done=$((done + 1))
      echo "[ OK ] $session_name"
    else
      failed=$((failed + 1))
      echo "[ERR ] $session_name"
    fi
  done

  echo "OneFormer complete: done=$done skipped=$skipped failed=$failed"
  if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: OneFormer export failed for at least one session." >&2
    exit 1
  fi
}

run_autolabel_subset() {
  local processing_root="$1"
  local prefix="$2"

  echo "== Sync/meta generation: $prefix =="
  "$VENV_PY" "$PREPARE_AUTOLABEL" \
    --data-root "$processing_root" \
    --session-prefix "$prefix" \
    --skip-radar-gen \
    --skip-autolabel

  echo "== Depth-gated autolabel: $prefix =="
  "$VENV_PY" "$ONEFORMER_AUTOLABEL" \
    --processing_root_dir "$processing_root" \
    --synchronized_root_dir "$processing_root" \
    --extrinsics_json "$EXTRINSICS_JSON" \
    --session_glob "${prefix}*" \
    --assign_mode best_per_radar \
    --max_time_diff_ms 35 \
    --depth_occlusion_m 0.20 \
    --depth_patch_r 2
}

ensure_training_inputs() {
  local prefix="$1"
  local holdout_last="$2"
  local processing_root="$3"
  local ready

  ready="$(count_training_ready_sessions "$processing_root" "$prefix")"
  if [[ "$ready" -gt "$holdout_last" ]]; then
    echo "== Training input check: $prefix ready sessions=$ready =="
    return
  fi

  if [[ "$processing_root" == "$MAY11_PROCESSING_ROOT" ]]; then
    local missing_tensors
    missing_tensors="$(count_labeled_missing_radar_tensors "$processing_root" "$prefix")"
    if [[ "$missing_tensors" -gt 0 ]]; then
      if is_truthy "$SKIP_RADAR_TENSOR_EXPORT"; then
        echo "ERROR: $prefix has $missing_tensors labeled sessions missing *_radar_tensors.npz." >&2
        echo "       Clear SKIP_RADAR_TENSOR_EXPORT or run $MAY11_RADAR_TENSOR_EXPORT first." >&2
        exit 1
      fi
      if [[ ! -f "$MAY11_RADAR_TENSOR_EXPORT" ]]; then
        echo "ERROR: missing May 11 radar tensor exporter: $MAY11_RADAR_TENSOR_EXPORT" >&2
        exit 1
      fi
      echo "== Export May 11 radar tensors for labeled sessions =="
      "$VENV_PY" "$MAY11_RADAR_TENSOR_EXPORT" \
        --data-root "$processing_root" \
        --session-prefix "$prefix" \
        --cfg-path "$CANON_DIR/config/profile_objdet.cfg"
      ready="$(count_training_ready_sessions "$processing_root" "$prefix")"
    fi
  fi

  if [[ "$ready" -le "$holdout_last" ]]; then
    echo "ERROR: Need more than $holdout_last training-ready sessions for $prefix, found $ready." >&2
    echo "       A ready session needs raw CSV, non-empty labeled_radar_points_v4.csv," >&2
    echo "       <session>_radar_tensors.npz, and hybrid_rd/manifest.json." >&2
    exit 1
  fi
}

build_subset() {
  local prefix="$1"
  local holdout_last="$2"
  local points_dir="$3"
  local patch_dir="$4"
  local fourclass_dir="$5"
  local processing_root="$6"

  echo "== Build points: $prefix =="
  "$VENV_PY" "$DATASET_TOOLS" build-points \
    --processing-root "$processing_root" \
    --output-dir "$points_dir" \
    --holdout-last "$holdout_last" \
    --session-prefix "$prefix"

  echo "== Build patches: $prefix =="
  "$VENV_PY" "$DATASET_TOOLS" build-patches \
    --point-csv "$points_dir/points.csv" \
    --sidecar-root "$processing_root" \
    --output-dir "$patch_dir" \
    --frame-number-offset 0 \
    --drop-invalid \
    --require-all-valid \
    --allowed-session-prefix "$prefix" \
    --extract-rd-scalars

  echo "== Materialize four-class labels: $prefix =="
  "$VENV_PY" "$FOURCLASS_TOOLS" label-dataset \
    --source-csv "$patch_dir/points_with_rd_patch_index.csv" \
    --source-root "$patch_dir" \
    --output-dir "$fourclass_dir" \
    --allowed-session-prefix "$prefix" \
    --depth-teacher-csv "$points_dir/points.csv"

  echo "== Validate subset: $prefix =="
  "$VENV_PY" "$DATASET_TOOLS" validate \
    --point-csv "$fourclass_dir/points_with_rd_patch_index.csv" \
    --dataset-root "$fourclass_dir" \
    --session-prefix "$prefix"
}

combine_datasets() {
  local may11_csv="$MAY11_FOURCLASS_DIR/points_with_rd_patch_index.csv"
  local apr28_csv="$APR28_FOURCLASS_DIR/points_with_rd_patch_index.csv"
  local may11_patches="$MAY11_FOURCLASS_DIR/patches"
  local apr28_patches="$APR28_FOURCLASS_DIR/patches"

  echo "== Combine May 11 + Apr 28 =="
  mkdir -p "$COMBINED_RD_ROOT"
  ln -sfn "$may11_patches" "$COMBINED_RD_ROOT/patches_may11"
  ln -sfn "$apr28_patches" "$COMBINED_RD_ROOT/patches_apr28"

  "$VENV_PY" - "$may11_csv" "$apr28_csv" "$COMBINED_CSV" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

may11_csv = Path(sys.argv[1])
apr28_csv = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
out_root = out_csv.parent

may11 = pd.read_csv(may11_csv, low_memory=False)
apr28 = pd.read_csv(apr28_csv, low_memory=False)

for frame, shard_prefix in ((may11, "patches_may11"), (apr28, "patches_apr28")):
    frame["rd_patch_shard"] = frame["rd_patch_shard"].astype(str).str.replace("patches/", f"{shard_prefix}/", regex=False)
    if "ra_patch_shard" in frame.columns:
        frame["ra_patch_shard"] = frame["ra_patch_shard"].astype(str).str.replace("patches/", f"{shard_prefix}/", regex=False)

may11["split"] = "train"

all_cols = list(may11.columns)
for col in apr28.columns:
    if col not in all_cols:
        all_cols.append(col)

for col in all_cols:
    if col not in may11.columns:
        may11[col] = 0.0 if col == "point_label_weight" else ""
    if col not in apr28.columns:
        apr28[col] = 0.0 if col == "point_label_weight" else ""

combined = pd.concat([may11[all_cols], apr28[all_cols]], ignore_index=True)
if "point_label_weight" not in combined.columns:
    combined["point_label_weight"] = 1.0
else:
    combined["point_label_weight"] = pd.to_numeric(combined["point_label_weight"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

if "final_train_weight" in combined.columns:
    combined["final_train_weight"] = pd.to_numeric(combined["final_train_weight"], errors="coerce").fillna(1.0).clip(0.0, 1.0)

out_root.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_csv, index=False)

manifest = {
    "kind": "combined_branch1_4class_rd_patch_dataset",
    "label_col": "bucket_4class",
    "sample_weight_col": "point_label_weight",
    "sources": {
        "may11": str(may11_csv.resolve()),
        "apr28": str(apr28_csv.resolve()),
    },
    "patch_roots": {
        "patches_may11": str((may11_csv.parent / "patches").resolve()),
        "patches_apr28": str((apr28_csv.parent / "patches").resolve()),
    },
    "output_csv": str(out_csv.relative_to(out_root)),
    "n_rows": int(len(combined)),
    "n_may11_rows": int(len(may11)),
    "n_apr28_rows": int(len(apr28)),
    "split_counts": {k: int(v) for k, v in combined["split"].value_counts().to_dict().items()},
    "class_counts_4class": {
        split: combined[combined["split"] == split]["bucket_4class"].value_counts().to_dict()
        for split in sorted(combined["split"].dropna().astype(str).unique())
    },
}
(out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({
    "combined_csv": str(out_csv),
    "n_rows": int(len(combined)),
    "split_counts": manifest["split_counts"],
}, indent=2))
PY

  echo "== Validate combined dataset =="
  "$VENV_PY" "$DATASET_TOOLS" validate \
    --point-csv "$COMBINED_CSV" \
    --dataset-root "$COMBINED_RD_ROOT" \
    --session-prefix "session_2026-"
}

train_model() {
  echo "== Train Branch 1 model =="
  "$VENV_PY" "$TRAIN_SCRIPT" \
    --dataset "$COMBINED_CSV" \
    --dataset-root "$COMBINED_RD_ROOT" \
    --out-dir "$TRAIN_OUT_DIR" \
    --label-col "$TRAIN_LABEL_COL" \
    --bucket-order "$TRAIN_BUCKET_ORDER" \
    --feature-mode "$TRAIN_FEATURE_MODE" \
    --sample-weight-col "$TRAIN_SAMPLE_WEIGHT_COL" \
    --best-metric val_bal_acc \
    --epochs "$TRAIN_EPOCHS" \
    --lr "$TRAIN_LR" \
    --weight-decay "$TRAIN_WEIGHT_DECAY" \
    --device "$TRAIN_DEVICE"
}

echo "============================================================"
echo "Branch 1 overnight batch"
echo "May 11 root     : $MAY11_PROCESSING_ROOT"
echo "Apr 28 root     : $APR28_PROCESSING_ROOT"
echo "May 11 sessions : $(count_sessions "$MAY11_PROCESSING_ROOT" "$MAY11_PREFIX")"
echo "Apr 28 sessions : $(count_sessions "$APR28_PROCESSING_ROOT" "$APR28_PREFIX")"
echo "Skip OneFormer  : $SKIP_ONEFORMER"
echo "Apr28 autolabel : $RUN_APR28_AUTOLABEL"
echo "============================================================"

run_oneformer_batch
run_autolabel_subset "$MAY11_PROCESSING_ROOT" "$MAY11_PREFIX"
if is_truthy "$RUN_APR28_AUTOLABEL"; then
  run_autolabel_subset "$APR28_PROCESSING_ROOT" "$APR28_PREFIX"
fi
ensure_training_inputs "$MAY11_PREFIX" "$MAY11_HOLDOUT_LAST" "$MAY11_PROCESSING_ROOT"
ensure_training_inputs "$APR28_PREFIX" "$APR28_HOLDOUT_LAST" "$APR28_PROCESSING_ROOT"
build_subset "$MAY11_PREFIX" "$MAY11_HOLDOUT_LAST" "$MAY11_POINTS_DIR" "$MAY11_PATCH_DIR" "$MAY11_FOURCLASS_DIR" "$MAY11_PROCESSING_ROOT"
build_subset "$APR28_PREFIX" "$APR28_HOLDOUT_LAST" "$APR28_POINTS_DIR" "$APR28_PATCH_DIR" "$APR28_FOURCLASS_DIR" "$APR28_PROCESSING_ROOT"
combine_datasets
train_model

echo "Done."
