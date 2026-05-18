#!/bin/bash
source /home/hullumdr/prof/io/.venv/bin/activate
export PYTHONPATH=/home/hullumdr/prof/io:/home/hullumdr/prof/io/LLM_ML/jetson_nav_pipeline/canon
cd /home/hullumdr/prof/io/LLM_ML/jetson_nav_pipeline/canon
python3 branch3_replay_simulation.py \
  --input-mode model \
  --model-pt /home/hullumdr/prof/io/LLM_ML/jetson_nav_pipeline/canon/models/3branch_best_model.pt \
  --sessions-dir /home/hullumdr/prof/io/LLM_ML/data/data-4-30/Processing \
  --session-selection sequential \
  --output /home/hullumdr/prof/io/LLM_ML/jetson_nav_pipeline/canon/hnm_debug_apr28_directness_v1 \
  --debug-frame-snapshots \
  --debug-frame-snapshot-stride 5 \
  --max-sessions 0 \
  --map-update-mode ego_directness \
  --fps 10 \
  --width 1280 \
  --height 720
