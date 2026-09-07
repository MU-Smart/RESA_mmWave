#!/usr/bin/env bash
# run_oneformer_data430.sh
#
# Runs Stage 1 (OneFormer segmentation export) on every session in
# LLM_ML/data/data-4-30/Processing/
#
# Prerequisites:
#   - CUDA GPU available
#   - pip install torch transformers Pillow opencv-python pandas tqdm
#   - Run from OUTSIDE the Claude Code sandbox (requires GPU passthrough)
#
# Resumable: sessions with a complete seg_timestamps.csv are skipped.
#
# Usage:
#   bash run_oneformer_data430.sh [--save-png]
#
# Options:
#   --save-png   Also write colorized seg_vis/*.png (slower, bigger disk)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="$SCRIPT_DIR/noah_export_labelmaps_from_processing_v2.py"
PROCESSING_ROOT="/home/hullumdr/prof/io/LLM_ML/data/data-4-30/Processing"

SAVE_PNG=""
for arg in "$@"; do
  [[ "$arg" == "--save-png" ]] && SAVE_PNG="--save_png"
done

echo "========================================================"
echo "OneFormer batch export — data-4-30"
echo "Processing root : $PROCESSING_ROOT"
echo "Model           : tiny  (large weights are LFS stubs)"
echo "every_n         : 1  (all frames)"
echo "save_png        : ${SAVE_PNG:-no}"
echo "========================================================"

mapfile -t SESSIONS < <(find "$PROCESSING_ROOT" -maxdepth 1 -type d -name "session_*" | sort)
echo "Found ${#SESSIONS[@]} sessions."
echo ""

DONE=0
SKIPPED=0
FAILED=0

for SESSION_DIR in "${SESSIONS[@]}"; do
  SESSION_NAME="$(basename "$SESSION_DIR")"

  # Skip if seg_timestamps.csv already exists (completed run)
  if [[ -f "$SESSION_DIR/seg_timestamps.csv" ]]; then
    N_SEG=$(find "$SESSION_DIR/seg" -name "*.npy" 2>/dev/null | wc -l)
    echo "[SKIP] $SESSION_NAME  (seg_timestamps.csv present, $N_SEG seg files)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  echo "------------------------------------------------------------"
  echo "[RUN ] $SESSION_NAME"
  echo "------------------------------------------------------------"

  if python3 "$EXPORT_SCRIPT" \
      --processing_session_dir "$SESSION_DIR" \
      --model tiny \
      --every_n 1 \
      $SAVE_PNG; then
    DONE=$((DONE + 1))
    echo "[OK  ] $SESSION_NAME"
  else
    FAILED=$((FAILED + 1))
    echo "[ERR ] $SESSION_NAME — export script returned non-zero exit code"
  fi

  echo ""
done

echo "========================================================"
echo "BATCH COMPLETE"
echo "  Exported : $DONE"
echo "  Skipped  : $SKIPPED  (already done)"
echo "  Failed   : $FAILED"
echo ""
echo "Next step: run prepare_autolabel.py or call the depth-gated"
echo "autolabeler directly:"
echo "  python3 noah_autolabel_radar_using_synccsv_v4_depth_gated.py \\"
echo "    --processing_root_dir $PROCESSING_ROOT \\"
echo "    --synchronized_root_dir $PROCESSING_ROOT \\"
echo "    --assign_mode best_per_radar --max_time_diff_ms 35"
echo "========================================================"
