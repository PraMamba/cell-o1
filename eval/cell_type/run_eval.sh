#!/bin/bash
set -eu

export CUDA_VISIBLE_DEVICES=6

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl

cd ~/cell-o1/eval/cell_type

# ========================= Configuration =========================

# Task type selection
# Options: batch_constrained | batch_openended | single_constrained | single_openended | all
TASK_TYPE="single_openended"

# Base configuration
DATASET_ID="D096"
BASE_DATA_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/${DATASET_ID}"
BASE_OUTPUT_DIR="${BASE_DATA_DIR}/${TASK_TYPE}/eval_results"
MODEL_NAME="ncbi/Cell-o1"
DEVICE="cuda"
BATCH_SIZE=32
MAX_NEW_TOKENS=2048

OUTPUT_DIR="${BASE_OUTPUT_DIR}/"
INPUT_QA_FILE="${BASE_DATA_DIR}/${TASK_TYPE}/qa/${DATASET_ID}_subset_processed_w_cell2sentence_qa.json"

# ========================= Run Evaluation =========================

echo "============================================================"
echo "[INFO] Evaluation Configuration"
echo "============================================================"
echo "[INFO] Task Type:        $TASK_TYPE"
if [ "$TASK_TYPE" != "all" ]; then
    echo "[INFO] Input file:       $INPUT_QA_FILE"
fi
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Model:            $MODEL_NAME"
echo "[INFO] Batch size:       $BATCH_SIZE"
echo "[INFO] Max new tokens:   $MAX_NEW_TOKENS"
echo "[INFO] Device:           $DEVICE"
echo "============================================================"

# Function to run evaluation
run_evaluation() {
    local task=$1
    local script=$2
    local batch_size=$3
    local max_tokens=$4
    local output_dir="$BASE_OUTPUT_DIR/$task"
    local input_file="${BASE_DATA_DIR}/${task}/qa/${DATASET_ID}_subset_processed_w_cell2sentence_qa.json"
    
    echo ""
    echo "[INFO] Running $task evaluation..."
    echo "[INFO] Input file: $input_file"
    python "$script" \
        --input_file $input_file \
        --output_dir $output_dir \
        --model_name $MODEL_NAME \
        --max_new_tokens $max_tokens \
        --batch_size $batch_size \
        --device $DEVICE
    
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
    single_constrained)
        run_evaluation "single_constrained" "singlecell_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    single_openended)
        run_evaluation "single_openended" "singlecell_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
    all)
        echo "[INFO] Running all evaluation tasks..."
        run_evaluation "batch_constrained" "batch_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "batch_openended" "batch_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "single_constrained" "singlecell_constrained_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        run_evaluation "single_openended" "singlecell_openended_eval.py" $BATCH_SIZE $MAX_NEW_TOKENS
        ;;
esac

echo "============================================================"
echo "[SUCCESS] All evaluations completed!"
