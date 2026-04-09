#!/bin/bash
# Run 8 parallel Optuna workers, one per GPU.
# Usage: bash scripts/run_optuna_parallel.sh [CONFIG]
# Default config: configs/tuning/40mio_hp_tuning_c.yaml

CONFIG="${1:-configs/tuning/40mio_hp_tuning_c.yaml}"
STUDY_DB=$(grep -oP 'storage:\s*\K.*' "$CONFIG" | tr -d ' ')
LOG_DIR="logs_optuna"

mkdir -p optuna_studies "$LOG_DIR"

# Clean stale DB if requested
if [ "$2" = "--fresh" ]; then
    echo "Removing old study DB: $STUDY_DB"
    rm -f "$STUDY_DB"
fi

for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i uv run python main.py --stage tune --config "$CONFIG" \
        > "$LOG_DIR/gpu$i.log" 2>&1 &
    echo "Launched worker on GPU $i (PID $!)"
done

echo ""
echo "All 8 workers launched."
echo "  Monitor:  tail -f $LOG_DIR/gpu*.log"
echo "  Kill all: pkill -f 'stage tune'"
echo ""
wait
echo "All workers finished."
