#!/bin/bash
set -eu

export CUDA_VISIBLE_DEVICES=6,7

source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vLLM

cd ~/cell-o1/eval/cell_type

# ========================= Configuration =========================

# Task type selection
# Options: batch_constrained | batch_openended | single_constrained | single_openended
TASK_TYPE="single_openended"

# Base configuration
DATASET_ID="D095"
BASE_DATA_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell-O1/${DATASET_ID}"
BASE_OUTPUT_DIR="${BASE_DATA_DIR}/${TASK_TYPE}/eval_results_vllm"
MODEL_NAME="ncbi/Cell-o1"

# vLLM configuration
BATCH_SIZE=256
MAX_NEW_TOKENS=2048
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.9
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=-1

# Input/Output paths
OUTPUT_DIR="${BASE_OUTPUT_DIR}/"
INPUT_JSONL_FILE="${BASE_DATA_DIR}/${TASK_TYPE}/qa/${DATASET_ID}_subset_processed_w_cell2sentence_qa_conversations.jsonl"

# ========================= Run Evaluation =========================

echo "============================================================"
echo "[INFO] vLLM Evaluation Configuration"
echo "============================================================"
echo "[INFO] Task Type:              $TASK_TYPE"
echo "[INFO] Input file:             $INPUT_JSONL_FILE"
echo "[INFO] Output directory:       $OUTPUT_DIR"
echo "[INFO] Model:                  $MODEL_NAME"
echo "[INFO] Batch size:             $BATCH_SIZE"
echo "[INFO] Max new tokens:         $MAX_NEW_TOKENS"
echo "[INFO] Tensor parallel size:   $TENSOR_PARALLEL_SIZE"
echo "[INFO] GPU memory util:        $GPU_MEMORY_UTILIZATION"
echo "[INFO] Temperature:            $TEMPERATURE"
echo "============================================================"

# Check if input file exists
if [ ! -f "$INPUT_JSONL_FILE" ]; then
    echo "[ERROR] Input JSONL file not found: $INPUT_JSONL_FILE"
    exit 1
fi

# Run vLLM evaluation
echo ""
echo "[INFO] Running ${TASK_TYPE} evaluation with vLLM..."

python singlecell_openended_eval_vllm.py \
    --input_file "$INPUT_JSONL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "[SUCCESS] vLLM evaluation completed!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "[ERROR] vLLM evaluation failed!"
    echo "============================================================"
    exit 1
fi
