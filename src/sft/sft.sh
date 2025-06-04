#!/bin/bash

# ========== Configuration ==========
LLM_NAME="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="./sft_train.json"
CACHE_DIR="../huggingface/hub"
SAVE_DIR="./saved_models"
R=256
LR=5e-5
N_EPOCHS=1
BATCH_SIZE=6
ACCUM_STEPS=16
EVAL_RATIO=0.1
MAX_SEQ_LENGTH=4096
DEVICE="auto"



# ========== Step 1: SFT Training ==========
echo "Running SFT training..."
python sft_trainer.py \
    --llm_name "$LLM_NAME" \
    --data_path "$DATA_PATH" \
    --cache_dir "$CACHE_DIR" \
    --r "$R" \
    --lr "$LR" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --accumulation_steps "$ACCUM_STEPS" \
    --eval_ratio "$EVAL_RATIO" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --save_dir "$SAVE_DIR"


# ========== Step 2: Merge LoRA ==========
# Determine checkpoint path (change according to actual best checkpoint number)
CHECKPOINT="${SAVE_DIR}/checkpoint-${N_EPOCHS}"  # adjust this if different
MERGE_OUTPUT="${SAVE_DIR}/merged_model"

echo "Merging LoRA weights from checkpoint: $CHECKPOINT"
python merge_lora.py \
    --base_model_path "$LLM_NAME" \
    --lora_ckpt_path "$CHECKPOINT" \
    --output_dir "$MERGE_OUTPUT" \
    --device "$DEVICE"

echo "All done. Merged model saved to $MERGE_OUTPUT"
