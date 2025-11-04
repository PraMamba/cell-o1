#!/bin/bash
set -eu

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl
cd ~/cell-o1/data

# ------------------------- Configurable Paths -------------------------

DATASET_ID="D099"

# Path to your input raw .h5ad files
RAW_H5AD_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/${DATASET_ID}/processed_data"

# TASK_TYPE: batch_constrained, batch_openended, single_constrained, single_openended
TASK_TYPE="${1:-single_openended}"

# Parse task type
if [[ "$TASK_TYPE" == "batch_constrained" ]]; then
    QA_MODE="batch"
    SINGLECELL_MODE="constrained"
elif [[ "$TASK_TYPE" == "batch_openended" ]]; then
    QA_MODE="batch"
    SINGLECELL_MODE="openended"
elif [[ "$TASK_TYPE" == "single_constrained" ]]; then
    QA_MODE="single"
    SINGLECELL_MODE="constrained"
elif [[ "$TASK_TYPE" == "single_openended" ]]; then
    QA_MODE="single"
    SINGLECELL_MODE="openended"
else
    echo "[ERROR] Invalid TASK_TYPE: $TASK_TYPE"
    echo "        Must be one of: batch_constrained, batch_openended, single_constrained, single_openended"
    exit 1
fi

# Base directory for intermediate outputs
BASE_WORK_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/${DATASET_ID}/${TASK_TYPE}"

# Directory to save single-cell level JSON output (Step 1)
CELL_JSON_DIR="${BASE_WORK_DIR}/cell_metadata"

# Directory to save QA-formatted batch data (Step 2)
QA_DIR="${BASE_WORK_DIR}/qa"


# Directory to save final LLM-ready data (Step 3)
# Use the same base directory as intermediate outputs
FINAL_OUTPUT_DIR="${BASE_WORK_DIR}"


echo "=========================================="
echo "Pipeline Configuration:"
echo "  Dataset ID: $DATASET_ID"
echo "  QA Mode: $QA_MODE"
if [ "$QA_MODE" = "single" ]; then
    echo "  Single-cell Mode: $SINGLECELL_MODE"
fi
echo "  Base Work Directory: $BASE_WORK_DIR"
echo "  Cell Metadata Dir: $CELL_JSON_DIR"

echo "  Final Output Directory: $FINAL_OUTPUT_DIR"
echo "=========================================="
echo ""

# ------------------------- Step 1: Extract context from h5ad -------------------------
python build_context.py \
    --input_dir $RAW_H5AD_DIR \
    --output_dir $CELL_JSON_DIR

# ------------------------- Step 2: Build QA pairs -------------------------
if [ "$QA_MODE" = "single" ]; then
    echo "[INFO] Generating single-cell QA pairs (mode: $SINGLECELL_MODE)..."
    python build_singlecell_qa.py \
        --input_dir $CELL_JSON_DIR \
        --output_dir $QA_DIR \
        --mode $SINGLECELL_MODE
elif [ "$QA_MODE" = "batch" ]; then
    echo "[INFO] Generating batch-level QA pairs..."
    python match_qa.py \
        --input_dir $CELL_JSON_DIR \
        --output_dir $QA_DIR
else
    echo "[ERROR] Invalid QA_MODE: $QA_MODE. Must be 'batch' or 'single'"
    exit 1
fi

# ------------------------- Step 3: Convert QA to JSONL conversation format -------------------------
echo "[INFO] Converting QA pairs to JSONL conversation format..."

# Find the QA JSON file
QA_JSON_FILE=$(find $QA_DIR -name "*_qa.json" -type f | head -1)

if [ -z "$QA_JSON_FILE" ]; then
    echo "[ERROR] No QA JSON file found in $QA_DIR"
    exit 1
fi

# Determine output JSONL filename
QA_BASENAME=$(basename "$QA_JSON_FILE" .json)
JSONL_OUTPUT_FILE="${QA_DIR}/${QA_BASENAME}_conversations.jsonl"

python convert_qa_to_jsonl.py \
    --input_file "$QA_JSON_FILE" \
    --output_file "$JSONL_OUTPUT_FILE" \
    --mode "$TASK_TYPE"

echo "[INFO] JSONL conversation file saved to: $JSONL_OUTPUT_FILE"

