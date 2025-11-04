#!/bin/bash
set -eu

cd ~/cell-o1

echo "========================================================"
echo "🎯 Cell Type LLM Judge - DeepSeek API Evaluation"
echo "========================================================"

# === Input/Output Configuration ===
PREDICTIONS_PATH="${1:-/gpfs/Mamba/Project/Single_Cell/Evaluate/MetaQA_SingleCell_Test_Diverse_NEW-RVQ/External/Qwen2.5-7B-Instruct_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretrain=TOP-500_GeneNames_Sample-0.1_MetaQA=Cell_Type_NEW-RVQ_V1/checkpoint-72000/inference-vllm_20251102_17-dataset_id=D094_lung_subset-task_type=cell_type_identification/inference_results.json}"
OUTPUT_DIR="${2:-/gpfs/Mamba/Project/Single_Cell/Evaluate/MetaQA_SingleCell_Test_Diverse_NEW-RVQ/External/Qwen2.5-7B-Instruct_Manual_Resize_Block-8_Size-32_Num-256_ExplicitTokens_Continued-Pretrain=TOP-500_GeneNames_Sample-0.1_MetaQA=Cell_Type_NEW-RVQ_V1/checkpoint-72000/inference-vllm_20251102_17-dataset_id=D094_lung_subset-task_type=cell_type_identification/llm_judge}"
DATASET_ID="${3:-D094}"

mkdir -p "$OUTPUT_DIR"

# === Sampling Configuration ===
MAX_SAMPLES=-1  # Set to -1 to evaluate all samples
RANDOM_SEED=42

# === Performance Configuration ===
BATCH_SIZE=512
MAX_CONCURRENT=16
DELAY_BETWEEN_BATCHES=1.0

# === LLM API Configuration ===
LLM_MODEL="deepseek-chat"
LLM_API_KEY="${DEEPSEEK_API_KEY:}"
BASE_URL="https://api.deepseek.com"

echo "📋 Configuration:"
echo "  Predictions Path: $PREDICTIONS_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Concurrent: $MAX_CONCURRENT"
echo "  Delay Between Batches: ${DELAY_BETWEEN_BATCHES}s"
echo "  LLM Model: $LLM_MODEL"
echo "  Base URL: $BASE_URL"
echo "  Random Seed: $RANDOM_SEED"
echo "========================================================"

echo "🔥 Starting Cell Type LLM judgment..."

# Export API key
export DEEPSEEK_API_KEY="$LLM_API_KEY"

# Run LLM judge
python ./llm_judge/celltype_llm_judge.py \
    --predictions_path "$PREDICTIONS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --random_seed $RANDOM_SEED \
    --batch_size $BATCH_SIZE \
    --max_concurrent $MAX_CONCURRENT \
    --delay_between_batches $DELAY_BETWEEN_BATCHES \
    --llm_model "$LLM_MODEL" \
    --llm_api_key "$LLM_API_KEY" \
    --base_url "$BASE_URL"

echo "✅ Cell Type LLM judgment completed!"
echo "📊 Computing classification metrics..."

# Define paths
JUDGED_RESULTS="$OUTPUT_DIR/celltype_judged_results.json"

# === Metrics Configuration ===
BINARY_THRESHOLD="${3:-0.7}"  # Default threshold: 0.7, can be overridden via 3rd argument

# Check if judged results exist and compute metrics
if [ -f "$JUDGED_RESULTS" ]; then
    echo ""
    echo "📊 Computing classification metrics..."
    echo "  Threshold: $BINARY_THRESHOLD"
    
    python ./llm_judge/compute_metrics.py \
        --judged_results_path "$JUDGED_RESULTS" \
        --output_path "$OUTPUT_DIR" \
        --binary_threshold $BINARY_THRESHOLD
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Classification metrics computed!"
    else
        echo ""
        echo "⚠️  Warning: Metrics computation failed (judgment results still available)"
    fi
    
    # Generate semantic confusion matrix
    echo ""
    echo "📊 Generating semantic confusion matrix..."
    echo "  Threshold: $BINARY_THRESHOLD"
    
    python ./llm_judge/semantic_confusion_matrix.py \
        --judged_results_path "$JUDGED_RESULTS" \
        --output_path "$OUTPUT_DIR" \
        --threshold $BINARY_THRESHOLD \
        --dataset $DATASET_ID
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Semantic confusion matrix generated!"
    else
        echo ""
        echo "⚠️  Warning: Confusion matrix generation failed"
    fi
else
    echo ""
    echo "⚠️  Warning: Judged results not found, skipping metrics computation"
fi

echo ""
echo "📂 Results saved to: $OUTPUT_DIR"

