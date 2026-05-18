#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/hullumdr/prof/io"
CANON_DIR="$ROOT/LLM_ML/jetson_nav_pipeline/canon"
VENV_PY="$ROOT/.venv/bin/python"

APR28_PROCESSING_ROOT="${APR28_PROCESSING_ROOT:-${SOURCE_PROCESSING_ROOT:-$ROOT/LLM_ML/data/data-4-30/Processing}}"
APR22_DATASET_ROOT="${APR22_DATASET_ROOT:-$ROOT/LLM_ML/data/apr22_oneformer/rd_patch_dataset}"
APR22_REVIEW_CSV="${APR22_REVIEW_CSV:-$CANON_DIR/hnm_debug_apr22_directness_v1/mining_candidates/auto_labels.csv}"
SEED_MODEL="${SEED_MODEL:-$CANON_DIR/models/3branch_best_model.pt}"

HOLDOUT_LAST="${HOLDOUT_LAST:-5}"
MIN_SCORE="${MIN_SCORE:-2.5}"
LIMIT="${LIMIT:-1000}"
DEBUG_STRIDE="${DEBUG_STRIDE:-5}"
FEATURE_MODE="${FEATURE_MODE:-directness_soft_structural}"
SAMPLE_WEIGHT_COL="${SAMPLE_WEIGHT_COL:-final_train_weight}"
ACC_LOSS_WEIGHT="${ACC_LOSS_WEIGHT:-0.25}"
EPOCHS="${EPOCHS:-60}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
MAP_UPDATE_MODE="${MAP_UPDATE_MODE:-ego_directness}"

export PYTHONPATH="$ROOT:$CANON_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

if [[ ! -f "$SEED_MODEL" ]]; then
  echo "ERROR: seed checkpoint not found: $SEED_MODEL" >&2
  exit 1
fi

if [[ ! -f "$APR22_DATASET_ROOT/points_with_rd_patch_index.csv" ]]; then
  echo "ERROR: Apr 22 combined source dataset not found: $APR22_DATASET_ROOT/points_with_rd_patch_index.csv" >&2
  exit 1
fi

if [[ ! -d "$APR22_DATASET_ROOT/patches" ]]; then
  echo "ERROR: Apr 22 patch directory not found: $APR22_DATASET_ROOT/patches" >&2
  exit 1
fi

if [[ ! -f "$APR22_REVIEW_CSV" ]]; then
  echo "ERROR: Apr 22 review CSV not found: $APR22_REVIEW_CSV" >&2
  exit 1
fi

build_combined_dataset() {
  local iter="$1"
  local apr28_csv="$2"
  local apr28_patches_root="$3"

  local combined_root="$ROOT/LLM_ML/data/combined_step8_v${iter}"
  local combined_rd_root="$combined_root/rd_patch_dataset"
  local combined_csv="$combined_rd_root/points_with_rd_patch_index.csv"
  local apr28_link="$combined_rd_root/patches_apr28"
  local apr22_link="$combined_rd_root/patches_apr22"

  mkdir -p "$combined_rd_root"
  ln -sfn "$apr28_patches_root/patches" "$apr28_link"
  ln -sfn "$APR22_DATASET_ROOT/patches" "$apr22_link"

  "$VENV_PY" - "$apr28_csv" "$APR22_DATASET_ROOT/points_with_rd_patch_index.csv" "$combined_csv" "$APR22_REVIEW_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

import pandas as pd

apr28_csv = Path(sys.argv[1])
apr22_csv = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
apr22_review_csv = Path(sys.argv[4])
out_root = out_csv.parent

apr28 = pd.read_csv(apr28_csv, low_memory=False)
apr22 = pd.read_csv(apr22_csv, low_memory=False)

if "rd_patch_shard" not in apr28.columns or "rd_patch_shard" not in apr22.columns:
    raise SystemExit("Both source datasets must contain rd_patch_shard.")

apr28 = apr28.copy()
apr22 = apr22.copy()
apr28["rd_patch_shard"] = apr28["rd_patch_shard"].astype(str).str.replace("patches/", "patches_apr28/", regex=False)
apr22["rd_patch_shard"] = apr22["rd_patch_shard"].astype(str).str.replace("patches/", "patches_apr22/", regex=False)
apr22["split"] = "train"

review_cols = ["accumulatable_label", "return_type_label", "review_confidence", "review_source"]
for frame_df in (apr28, apr22):
    for col in review_cols:
        if col not in frame_df.columns:
            frame_df[col] = ""

apr22_review_records = 0
apr22_review_hits = 0
if apr22_review_csv.exists():
    with apr22_review_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reviews = {}
        for row in reader:
            if row.get("auto_label_status") != "ok":
                continue
            acc = str(row.get("accumulatable", "") or "").strip().lower()
            return_type = str(row.get("return_type", "") or "").strip().lower()
            if acc not in {"yes", "no", "uncertain"} and not return_type:
                continue
            try:
                key = (str(row.get("session", "")).strip(), int(row.get("frame_num") or 0))
            except ValueError:
                continue
            reviews[key] = {
                "accumulatable_label": acc if acc in {"yes", "no", "uncertain"} else "",
                "return_type_label": return_type,
                "review_confidence": str(row.get("confidence", "") or ""),
                "review_source": "hnm_apr22_autolabel",
            }
        apr22_review_records = len(reviews)

    if reviews:
        keys = list(zip(apr22["session"].astype(str), pd.to_numeric(apr22["radar_frame_num"], errors="coerce").fillna(-1).astype(int)))
        mask = [key in reviews for key in keys]
        if any(mask):
            for col in review_cols:
                apr22.loc[mask, col] = [reviews[key][col] for key, keep in zip(keys, mask) if keep]
            apr22_review_hits = int(sum(mask))

all_cols = list(apr28.columns)
for col in apr22.columns:
    if col not in all_cols:
        all_cols.append(col)

for col in all_cols:
    if col not in apr28.columns:
        apr28[col] = ""
    if col not in apr22.columns:
        apr22[col] = ""

combined = pd.concat([apr28[all_cols], apr22[all_cols]], ignore_index=True)
if "final_train_weight" not in combined.columns:
    combined["final_train_weight"] = 1.0
else:
    combined["final_train_weight"] = (
        pd.to_numeric(combined["final_train_weight"], errors="coerce")
        .fillna(1.0)
        .clip(lower=0.0, upper=1.0)
    )
out_root.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_csv, index=False)

manifest = {
    "kind": "combined_step8_rd_patch_dataset",
    "sources": {
        "apr28": str(apr28_csv.resolve()),
        "apr22": str(apr22_csv.resolve()),
    },
    "output_csv": str(out_csv.relative_to(out_root)),
    "n_rows": int(len(combined)),
    "n_apr28_rows": int(len(apr28)),
    "n_apr22_rows": int(len(apr22)),
    "apr22_review_records": int(apr22_review_records),
    "apr22_review_hits": int(apr22_review_hits),
    "split_counts": {k: int(v) for k, v in combined["split"].value_counts().to_dict().items()},
    "class_counts": {k: int(v) for k, v in combined["bucket_3class"].value_counts().to_dict().items()},
    "patch_roots": {
        "patches_apr28": str((apr28_csv.parent / "patches").resolve()),
        "patches_apr22": str((Path(sys.argv[2]).parent / "patches").resolve()),
    },
}
(out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({
    "combined_csv": str(out_csv),
    "n_rows": int(len(combined)),
    "apr22_review_records": int(apr22_review_records),
    "apr22_review_hits": int(apr22_review_hits),
    "split_counts": manifest["split_counts"],
}, indent=2), file=sys.stderr)
PY

  printf '%s\n' "$combined_csv"
}

run_pass() {
  local iter="$1"
  local model_pt="$2"

  local replay_out="$CANON_DIR/step8_iter${iter}_replay"
  local candidates_dir="$replay_out/mining_candidates"
  local candidates_csv="$candidates_dir/hard_negative_candidates.csv"
  local review_csv="$candidates_dir/auto_labels.csv"
  local points_out="$ROOT/LLM_ML/data/step8_v${iter}/points"
  local patches_out="$ROOT/LLM_ML/data/step8_v${iter}/rd_patch_dataset"
  local combined_csv
  local combined_root="$ROOT/LLM_ML/data/combined_step8_v${iter}"
  local combined_rd_root="$combined_root/rd_patch_dataset"
  local model_out="$CANON_DIR/models/kpconv_step8_acc_v${iter}"
  local replay_map_mode="$MAP_UPDATE_MODE"

  mkdir -p "$candidates_dir" "$points_out" "$patches_out" "$model_out"

  if [[ "$iter" -gt 1 && "$replay_map_mode" == "ego_directness" ]]; then
    replay_map_mode="learned_acc"
  fi

  if [[ -s "$review_csv" && -s "$candidates_csv" ]]; then
    echo "== Pass ${iter}: reusing existing replay candidates and auto labels =="
    echo "   candidates: $candidates_csv"
    echo "   reviews:    $review_csv"
  else
    echo "== Pass ${iter}: replay =="
    "$VENV_PY" "$CANON_DIR/branch3_replay_simulation.py" \
      --input-mode model \
      --model-pt "$model_pt" \
      --sessions-dir "$APR28_PROCESSING_ROOT" \
      --session-selection sequential \
      --max-sessions 0 \
      --output "$replay_out" \
      --ui-style integrated \
      --debug-frame-snapshots \
      --debug-frame-snapshot-stride "$DEBUG_STRIDE" \
      --map-update-mode "$replay_map_mode" \
      --fps 10 \
      --width 1280 \
      --height 720

    echo "== Pass ${iter}: mine candidates =="
    "$VENV_PY" "$CANON_DIR/tools/hnm_tools.py" mine-candidates \
      --debug-root "$replay_out" \
      --output-dir "$candidates_dir" \
      --min-score "$MIN_SCORE" \
      --limit "$LIMIT"

    if [[ -s "$review_csv" ]]; then
      echo "== Pass ${iter}: auto labels already exist =="
    else
      echo "== Pass ${iter}: autolabel candidates =="
      "$VENV_PY" "$CANON_DIR/tools/hnm_autolabel.py" \
        --candidates "$candidates_csv" \
        --output "$review_csv" \
        --skip-existing
    fi
  fi

  echo "== Pass ${iter}: apply corrections in place =="
  "$VENV_PY" "$CANON_DIR/tools/hnm_tools.py" apply-corrections \
    --review-csv "$review_csv" \
    --processing-root "$APR28_PROCESSING_ROOT" \
    --output-root "$APR28_PROCESSING_ROOT" \
    --label-column bucket

  echo "== Pass ${iter}: build points =="
  "$VENV_PY" "$CANON_DIR/tools/dataset_tools.py" build-points \
    --processing-root "$APR28_PROCESSING_ROOT" \
    --output-dir "$points_out" \
    --holdout-last "$HOLDOUT_LAST"

  echo "== Pass ${iter}: build patches =="
  "$VENV_PY" "$CANON_DIR/tools/dataset_tools.py" build-patches \
    --point-csv "$points_out/apr28_points.csv" \
    --sidecar-root "$APR28_PROCESSING_ROOT" \
    --output-dir "$patches_out" \
    --frame-number-offset 0 \
    --extract-rd-scalars

  echo "== Pass ${iter}: build combined Apr 28 + Apr 22 dataset =="
  combined_csv="$(build_combined_dataset "$iter" "$patches_out/points_with_rd_patch_index.csv" "$patches_out")"

  echo "== Pass ${iter}: train Step 8 model =="
  "$VENV_PY" "$CANON_DIR/training/train_hybrid_rd_patch_kpconv.py" \
    --dataset "$combined_csv" \
    --dataset-root "$combined_rd_root" \
    --out-dir "$model_out" \
    --feature-mode "$FEATURE_MODE" \
    --sample-weight-col "$SAMPLE_WEIGHT_COL" \
    --has-acc-head \
    --acc-label-col accumulatable_label \
    --acc-loss-weight "$ACC_LOSS_WEIGHT" \
    --epochs "$EPOCHS" \
    --device "$TRAIN_DEVICE"

  echo "Pass ${iter} complete: $model_out/best_model.pt"
  CURRENT_MODEL="$model_out/best_model.pt"
}

cd "$CANON_DIR"

CURRENT_MODEL="$SEED_MODEL"
run_pass 1 "$CURRENT_MODEL"
run_pass 2 "$CURRENT_MODEL"
run_pass 3 "$CURRENT_MODEL"

echo "Bootstrap complete. Final checkpoint: $CURRENT_MODEL"
