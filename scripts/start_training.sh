#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_training.sh  —  Reliable background training launcher for CropOSGNN
#
# Usage (from /workspace/cropos):
#   bash scripts/start_training.sh
#
# Starts training in a tmux session (installs tmux if needed) with nohup
# fallback, so the job survives SSH disconnections and RunPod connection drops.
#
# Monitor:
#   tmux attach -t cropos-train        # live output (Ctrl+B D to detach)
#   tail -f /tmp/cropos_train.log      # from any new connection
#   cat /tmp/cropos_train.pid          # get PID to check/kill
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="/workspace/cropos"
LOG="/tmp/cropos_train.log"
PID_FILE="/tmp/cropos_train.pid"
SESSION="cropos-train"

cd "$REPO_DIR"

# ── sanity checks ─────────────────────────────────────────────────────────────
if [ ! -f "src/training/train.py" ]; then
    echo "ERROR: run this script from /workspace/cropos (can't find src/training/train.py)"
    exit 1
fi

if [ ! -f "data/raw/era5_thailand.parquet" ]; then
    echo "ERROR: data/raw/era5_thailand.parquet not found"
    echo "       Run the data download workflow first, or copy parquets to data/raw/"
    exit 1
fi

# ── kill any existing training run ────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing training run (PID $OLD_PID)..."
        kill -9 "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
fi

# Also kill any lingering train processes
pkill -f "src.training.train" 2>/dev/null || true
sleep 1

# ── install tmux if missing ───────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    echo "Installing tmux..."
    apt-get install -y tmux -q 2>/dev/null || \
    conda install -y tmux -q 2>/dev/null || \
    echo "WARNING: tmux install failed — falling back to nohup"
fi

TRAIN_CMD="cd $REPO_DIR && python -u -m src.training.train --local data/raw 2>&1 | tee $LOG; echo 'Training finished (exit \$?)' >> $LOG"

# ── start training ────────────────────────────────────────────────────────────
if command -v tmux &>/dev/null; then
    # Kill old session if exists
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.5

    # Start new detached session
    tmux new-session -d -s "$SESSION" -x 220 -y 50
    tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter

    # Get PID of the python process inside tmux
    sleep 2
    TRAIN_PID=$(pgrep -f "src.training.train" | head -1 || echo "unknown")
    echo "$TRAIN_PID" > "$PID_FILE"

    echo ""
    echo "✓ Training started in tmux session '$SESSION' (PID: $TRAIN_PID)"
    echo ""
    echo "  Monitor (attach):  tmux attach -t $SESSION"
    echo "  Detach safely:     Ctrl+B then D"
    echo "  Follow log:        tail -f $LOG"
    echo "  Kill:              kill \$(cat $PID_FILE)"
    echo ""
else
    # Fallback: nohup
    echo "WARNING: tmux not available — using nohup fallback"
    nohup bash -c "$TRAIN_CMD" &
    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"

    echo ""
    echo "✓ Training started with nohup (PID: $TRAIN_PID)"
    echo ""
    echo "  Follow log:  tail -f $LOG"
    echo "  Kill:        kill \$(cat $PID_FILE)"
    echo ""
fi

# ── show live output for 30s so user can confirm startup ─────────────────────
echo "--- Watching startup (30s) — safe to Ctrl+C to detach ---"
sleep 2
timeout 30 tail -f "$LOG" 2>/dev/null || true
echo ""
echo "--- Training is running in background. Reconnect any time. ---"
