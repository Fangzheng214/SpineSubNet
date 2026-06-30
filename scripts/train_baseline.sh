#!/bin/bash

# Baseline Model Training Script (Grayscale)
#
# Usage:
#   bash scripts/train_baseline.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/configs/baseline_grayscale.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Baseline Training (Grayscale)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"
python train.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Baseline training completed successfully!"
    echo "Results: experiments/baseline_grayscale/"
    echo "=========================================="
else
    echo ""
    echo "Baseline training failed"
    exit 1
fi
