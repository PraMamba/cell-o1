#!/bin/bash
set -e  # Exit if any command fails
set -x  # Print each command

# ------------------------- Configurable Paths -------------------------

# Path to your input raw .h5ad files
RAW_H5AD_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013"

# Directory to save single-cell level JSON output (Step 1)
CELL_JSON_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/cell_metadata"

# Directory to save QA-formatted batch data (Step 2)
BATCH_QA_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/batch_qa"

# Directory to save final LLM-ready data (Step 3)
FINAL_OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/final_llm_input"

# Maximum number of test samples for stratified split (Step 3)
MAX_TEST_SAMPLES=1100

# ------------------------- Step 1: Extract context from h5ad -------------------------
python build_context.py \
    --input_dir "$RAW_H5AD_DIR" \
    --output_dir "$CELL_JSON_DIR"

# ------------------------- Step 2: Build batch-level QA -------------------------
python match_qa.py \
    --input_dir "$CELL_JSON_DIR" \
    --output_dir "$BATCH_QA_DIR"

# ------------------------- Step 3: Convert to LLM format with test split -------------------------
python split_train_test.py \
    --input_dir "$BATCH_QA_DIR" \
    --output_dir "$FINAL_OUTPUT_DIR" \
    --max_test_samples $MAX_TEST_SAMPLES

echo "[SUCCESS] Pipeline completed. Final training and test data saved in: $FINAL_OUTPUT_DIR"
