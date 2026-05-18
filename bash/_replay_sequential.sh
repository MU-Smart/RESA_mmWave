#!/bin/bash
# Poll until Apr 28 process finishes, then run Apr 22
APR28_PID=$(cat /tmp/replay_apr28_video.pid 2>/dev/null)
if [ -n "$APR28_PID" ]; then
    echo "Polling for Apr 28 replay (PID=$APR28_PID) to finish..."
    while kill -0 "$APR28_PID" 2>/dev/null; do
        sleep 30
    done
    echo "Apr 28 done. Starting Apr 22..."
fi
exec bash /home/hullumdr/prof/io/LLM_ML/jetson_nav_pipeline/canon/_replay_apr22_video.sh
