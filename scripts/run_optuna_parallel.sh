#!/bin/bash
# Run 8 parallel Optuna workers, one per GPU.
# Usage: bash scripts/run_optuna_parallel.sh [--fresh] [CONFIG]
# Default config: configs/tuning/40mio_hp_tuning_c.yaml

FRESH=0
CONFIG="configs/tuning/40mio_hp_tuning_c.yaml"
LOG_DIR="logs_optuna"

for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        FRESH=1
    else
        CONFIG="$arg"
    fi
done

mkdir -p optuna_studies "$LOG_DIR"

if [ "$FRESH" = "1" ]; then
    echo "Removing old study DB..."
    rm -f optuna_studies/parrotllm-40mio-hp-c.db
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
