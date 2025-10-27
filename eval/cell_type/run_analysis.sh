#!/bin/bash
# Comprehensive Analysis Pipeline for Cell Type Evaluation Results
#
# This script performs:
# 1. Cell type name standardization
# 2. Comprehensive metrics calculation
# 3. Confusion matrix visualization
#
# Usage:
#   bash run_analysis.sh [RESULTS_DIR] [DATASET] [OUTPUT_DIR]

set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl
cd ~/cell-o1/eval/cell_type

# ============================================
# Configuration
# ============================================

# Default paths (modify as needed)
RESULTS_DIR="${1:-/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/eval_results}"
DATASET="${2:-A013}"
OUTPUT_DIR="${3:-/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/analysis_results}"

echo "============================================"
echo "Cell Type Evaluation Analysis Pipeline"
echo "============================================"
echo "Results Directory: $RESULTS_DIR"
echo "Dataset:           $DATASET"
echo "Output Directory:  $OUTPUT_DIR"
echo "============================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================
# Step 1: Calculate Comprehensive Metrics
# ============================================

echo "[Step 1/3] Calculating comprehensive metrics with cell type standardization..."
python comprehensive_metrics.py \
    --results_dir "$RESULTS_DIR" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Metrics calculation complete!"
else
    echo "[ERROR] Metrics calculation failed!"
    exit 1
fi

# ============================================
# Step 2: Generate Visualizations
# ============================================

echo "[Step 2/3] Generating confusion matrix heatmaps and visualizations..."
python confusion_matrix_heatmap.py \
    --metrics_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/visualizations"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Visualization generation complete!"
else
    echo "[ERROR] Visualization generation failed!"
    exit 1
fi

# ============================================
# Step 3: Generate Simplified Confusion Matrices
# ============================================

echo "[Step 3/3] Generating improved clean confusion matrices for openended variants..."

# Process openended variants with improved simplified confusion matrices
for variant in batch_openended singlecell_openended; do
    metrics_file="$OUTPUT_DIR/${variant}_comprehensive_metrics.json"
    if [ -f "$metrics_file" ]; then
        echo "  Processing $variant..."
        python improved_simplified_cm.py \
            --metrics_file "$metrics_file" \
            --output_dir "$OUTPUT_DIR/visualizations" \
            --strategy top_k \
            --top_k 30 \
            --percentile 95 \
            --dataset "$DATASET"
    else
        echo "  [SKIP] Metrics file not found: $metrics_file"
    fi
done

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Simplified confusion matrix generation complete!"
else
    echo "[WARNING] Some simplified matrices may have failed"
fi

