#!/bin/bash
set -eu

export CUDA_VISIBLE_DEVICES=6

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl

# ========================= Configuration =========================

# Task type selection
# Options: batch_constrained | batch_openended | singlecell_constrained | singlecell_openended | all
TASK_TYPE="batch_constrained"

# Base configuration
INPUT_QA_FILE="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/batch_qa/A013_processed_sampled_w_cell2sentence_qa.json"
BASE_OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell-O1/Cell_Type/A013/eval_results"
MODEL_NAME="ncbi/Cell-o1"
DEVICE="cuda"
BATCH_SIZE=32
MAX_NEW_TOKENS=2048

# Set output directory based on task type
if [ "$TASK_TYPE" = "all" ]; then
    OUTPUT_DIR="$BASE_OUTPUT_DIR"
else
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$TASK_TYPE"
fi

# ========================= Run Evaluation =========================

echo "============================================================"
echo "[INFO] Evaluation Configuration"
echo "============================================================"
echo "[INFO] Task Type:        $TASK_TYPE"
echo "[INFO] Input file:       $INPUT_QA_FILE"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Model:            $MODEL_NAME"
echo "[INFO] Batch size:       $BATCH_SIZE"
echo "[INFO] Max new tokens:   $MAX_NEW_TOKENS"
echo "[INFO] Device:           $DEVICE"
echo "============================================================"
echo ""

cd /home/scbjtfy/cell-o1/eval/cell_type

# Function to run evaluation
run_evaluation() {
    local task=$1
    local script=$2
    local batch_size=$3
    local max_tokens=$4
    local output_dir="$BASE_OUTPUT_DIR/$task"
    
    echo ""
    echo "[INFO] Running $task evaluation..."
    python "$script" \
        --input_file "$INPUT_QA_FILE" \
        --output_dir "$output_dir" \
        --model_name "$MODEL_NAME" \
        --max_new_tokens $max_tokens \
        --batch_size $batch_size \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $task evaluation completed."
    else
        echo "[ERROR] $task evaluation failed."
        return 1
    fi
}

# Execute based on task type
case "$TASK_TYPE" in
    batch_constrained)
        run_evaluation "batch_constrained" "batch_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    batch_openended)
        run_evaluation "batch_openended" "batch_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    singlecell_constrained)
        run_evaluation "singlecell_constrained" "singlecell_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    singlecell_openended)
        run_evaluation "singlecell_openended" "singlecell_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    all)
        echo "[INFO] Running all evaluation tasks..."
        run_evaluation "batch_constrained" "batch_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "batch_openended" "batch_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "singlecell_constrained" "singlecell_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "singlecell_openended" "singlecell_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
esac

echo "============================================================"
echo "[SUCCESS] All evaluations completed!"
