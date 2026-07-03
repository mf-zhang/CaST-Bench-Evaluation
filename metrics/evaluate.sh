#!/bin/bash
# CaST-Bench evaluation — single model
# Usage: edit the variables below, then run:  bash evaluate.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# ── Configure these variables ──────────────────────────────────────────────────
# Defaults run the bundled Claude Sonnet 4.6 example (80 QAs, no dataset download
# needed). Point GT_PATH / PRED_PATH at your own files to evaluate another model.
MODEL_NAME="claude-sonnet-4-6"

# Path to GT castbench_hf.jsonl (download the full file from the CaST-Bench dataset release)
GT_PATH="$SCRIPT_DIR/example/castbench_hf.jsonl"

# Path to the predictions .jsonl file (one prediction per line)
PRED_PATH="$SCRIPT_DIR/../inference/predictions/sonnet-4.6_predictions.jsonl"

# Index range (inclusive start, exclusive end). Used to evaluate a subset of QAs.
START_INDEX=0
END_INDEX=99999

# Output directory
OUT_DIR="$SCRIPT_DIR/outputs/${MODEL_NAME}"
# ──────────────────────────────────────────────────────────────────────────────

python "$SCRIPT_DIR/src/evaluate_benchmark.py" \
  --gt "$GT_PATH" \
  --pred "$PRED_PATH" \
  --model_name "$MODEL_NAME" \
  --eps_overlap 2 --tau_t 0.5 --tau_st 0.1 \
  --use_coverage_aware_score false \
  --start_index "$START_INDEX" \
  --end_index "$END_INDEX" \
  --out_dir "$OUT_DIR"
