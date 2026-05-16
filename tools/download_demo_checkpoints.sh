#!/usr/bin/env bash
# Download both demo checkpoints for the ParrotLLM chat UI.
#
# Drops two .pt files into runs/demo/ so src/chat/app.py can offer them
# as named quick-load buttons ("Cyril & Christof" and "Gian & Tilman").
#
#   - cyril_christof.pt  : ParrotLLM submission (sft-christof branch),
#                          hosted as a GitHub Release asset on the fork
#                          because GitHub blocks LFS uploads from public
#                          forks.
#   - gian_tilman.pt     : PikoGPT_ParrotLabs submission from the other
#                          half of the original team (PR #13 on the
#                          upstream leaderboard), shipped as a regular
#                          ~80 MB blob in their submission branch.
#
# Run from the repo root.

set -euo pipefail

DEST_DIR="runs/demo"
mkdir -p "$DEST_DIR"

CYRIL_CHRISTOF_URL="https://github.com/steinerchristof/PikoGPT_Leaderboard/releases/download/parrotllm-v7/final_step_0001966_epoch_01_valloss_2p4231.pt"
CYRIL_CHRISTOF_DEST="$DEST_DIR/cyril_christof.pt"

GIAN_TILMAN_URL="https://raw.githubusercontent.com/TilmanHaferbeck/PikoGPT_Leaderboard/parrotllm_submission/Submissions/PikoGPT_ParrotLabs/runs/dpo_v9_submit_fp16.pt"
GIAN_TILMAN_DEST="$DEST_DIR/gian_tilman.pt"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "skip   $dest (already present, $(du -h "$dest" | cut -f1))"
        return
    fi
    echo "fetch  $url"
    echo "  ->   $dest"
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest"
    echo "  size $(du -h "$dest" | cut -f1)"
}

download "$CYRIL_CHRISTOF_URL" "$CYRIL_CHRISTOF_DEST"
download "$GIAN_TILMAN_URL"   "$GIAN_TILMAN_DEST"

echo
echo "Done. Launch the demo with:"
echo "  uv run python main.py --stage chat --config configs/chat/chat_demo.yaml"
