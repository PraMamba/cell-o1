#!/bin/bash

set -eu

cd ~/cell-o1

echo "========================================================"
echo "📊 Cell Type Prediction Performance Visualization"
echo "========================================================"

# === Input/Output Configuration ===
# Usage: ./run_visualize.sh [judged_results_path] [output_dir] [plots]
# Example: ./run_visualize.sh /path/to/results.json ./output all
JUDGED_RESULTS_PATH="${1:-/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/A013/llm_judge/celltype_judged_results.json}"
OUTPUT_DIR="${2:-/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/A013/eda_visualizations}"
PLOTS="${3:-all}"  # Options: all, score_dist, semantic_breakdown, celltype_perf, heatmap, sankey, different_errors, ambiguous_answers, dataset_perf

mkdir -p "$OUTPUT_DIR"

# === Log Setup ===
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/visualization_${DATE_SUFFIX}.log"

echo "📋 Configuration:"
echo "  Judged Results Path: $JUDGED_RESULTS_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Plots to Generate: $PLOTS"
echo "  Log File: $LOG_FILE"
echo "========================================================"

# === Environment Setup ===
# Source conda if available
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate cell-o1 || echo "Warning: conda environment 'cell-o1' not found, continuing without activation"
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cell-o1 || echo "Warning: conda environment 'cell-o1' not found, continuing without activation"
fi

# === Validation ===
if [ ! -f "$JUDGED_RESULTS_PATH" ]; then
    echo "❌ Error: Judged results file not found: $JUDGED_RESULTS_PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "./eda/visualize_performance.py" ]; then
    echo "❌ Error: Visualization script not found: ./eda/visualize_performance.py"
    exit 1
fi

echo "🔥 Starting visualization generation..."
echo "   Real-time log monitoring: tail -f ${LOG_FILE}"
echo ""

# === Run Visualization ===
# Convert space-separated plots to array format for Python script
IFS=' ' read -ra PLOTS_ARRAY <<< "$PLOTS"

python ./eda/visualize_performance.py \
    --judged_results_path "$JUDGED_RESULTS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --plots "${PLOTS_ARRAY[@]}" \
    >> "${LOG_FILE}" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Visualization generation completed successfully!"
    echo "📂 Results saved to: $OUTPUT_DIR"
    echo "📄 Log file: $LOG_FILE"
    
    # List generated files
    echo ""
    echo "📊 Generated visualization files:"
    ls -lh "${OUTPUT_DIR}"/*.png "${OUTPUT_DIR}"/*.html 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  No visualization files found"
else
    echo ""
    echo "❌ Error: Visualization generation failed (exit code: $EXIT_CODE)"
    echo "📄 Check log file for details: $LOG_FILE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================================"

