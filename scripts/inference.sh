#!/bin/bash

# Vertebra Subregion Segmentation Inference Script
#
# This script performs batch inference using the grayscale baseline model
#
# Usage:
#   bash scripts/inference.sh                    # Use default config
#   bash scripts/inference.sh custom_config.yaml # Use custom config

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
CONFIG_FILE=${1:-"${PROJECT_ROOT}/configs/inference.yaml"}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "=========================================="
    echo "✗ Error: Config file not found"
    echo "=========================================="
    echo "Expected: $CONFIG_FILE"
    echo ""
    echo "Available inference configs:"
    ls -1 "${PROJECT_ROOT}/configs"/inference*.yaml 2>/dev/null | sed 's/^/  /' || echo "  (no inference config files found)"
    echo ""
    echo "Usage: bash scripts/inference.sh [config_file]"
    echo "Example: bash scripts/inference.sh configs/inference.yaml"
    echo "=========================================="
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Vertebra Subregion Segmentation Inference"
echo "=========================================="
echo "Config:       ${CONFIG_FILE}"
echo "Project Root: ${PROJECT_ROOT}"
echo "=========================================="
echo ""

# Run inference
cd "$PROJECT_ROOT"
python tools/inference.py --config "$CONFIG_FILE"

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Inference completed successfully!"
    echo "=========================================="
    echo "Check output directory for results"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Inference failed"
    echo "=========================================="
    echo "Common issues:"
    echo "  - Model checkpoint path incorrect or not found"
    echo "  - Input directory empty or path incorrect"
    echo "  - Model architecture mismatch"
    echo "  - Insufficient GPU memory"
    echo "=========================================="
    exit 1
fi

