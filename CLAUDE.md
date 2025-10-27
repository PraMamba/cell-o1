# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cell-o1 is a research project for training LLMs to solve single-cell reasoning puzzles using reinforcement learning. The model performs batch-level cell type annotation where each cell in a batch must be assigned a unique cell type from a shared candidate set. The project uses supervised fine-tuning (SFT) on expert traces followed by reinforcement learning (GRPO or PPO) with batch-level rewards.

## Repository Structure

```
cell-o1/
├── data/                   # Data preprocessing pipeline
├── src/
│   ├── sft/               # Supervised fine-tuning with LoRA
│   └── verl/              # RL training framework (fork of volcengine/verl)
│       ├── examples/
│       │   ├── data_preprocess/     # Convert JSON to Parquet for RL
│       │   ├── grpo_trainer/        # GRPO training scripts
│       │   └── ppo_trainer/         # PPO training scripts
│       ├── scripts/                 # Inference and conversion utilities
│       └── verl/
│           └── workers/
│               └── reward_function/ # Custom reward functions for RL
├── eval/                  # Evaluation scripts
└── processed_data/        # Expected location for preprocessed datasets
```

## Common Commands

### Installation

```bash
conda create -n cello1 python=3.9
conda activate cello1

# Install PyTorch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install "https://files.pythonhosted.org/packages/6e/75/b424aebc9f2fc5db319d5df5fff62fa19254c8ef974c254588d48c480df2/pyairports-2.1.1-py3-none-any.whl"
pip install "numpy<2.0" "outlines==0.0.45"
pip install vllm==0.6.3  # or 0.5.4, 0.4.2, 0.3.1
pip install ray transformers==4.47.0 trl==0.17.0

# Install verl package
cd src/verl
pip install -e .

# Install additional packages
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib
```

### Data Preprocessing

```bash
# From raw .h5ad files (edit paths in run_pipeline.sh first)
cd data
bash run_pipeline.sh

# Or download preprocessed data from HuggingFace:
# datasets.load_dataset("ncbi/CellPuzzles")
```

### Supervised Fine-Tuning

```bash
cd src/sft
bash sft.sh
```

This runs SFT with LoRA adapters on `processed_data/sft_train.json` and automatically merges the LoRA weights back into the base model.

### Reinforcement Learning

```bash
# 1. Convert JSON to Parquet format
cd src/verl
python examples/data_preprocess/cello1.py \
    --train_file ../../processed_data/grpo_train.json \
    --local_dir ./parquet_output

# 2a. Run GRPO training
bash examples/grpo_trainer/run_cello1_grpo.sh

# 2b. OR run PPO training
bash examples/ppo_trainer/run_cello1_ppo.sh

# 3. Convert FSDP checkpoint to HuggingFace format
python scripts/convert_fsdp_to_hf.py \
    /path/to/fsdp_checkpoint/actor \
    /path/to/base_model \
    /path/to/output_hf_model
```

### Inference

```bash
# Single-shard inference
cd src/verl
CUDA_VISIBLE_DEVICES=0 python scripts/test.py \
    --n 1 --i 0 \
    --llm_name /path/to/huggingface_model \
    --folder prediction_output \
    --dataset /path/to/test_data.json

# Multi-shard inference (run multiple commands in parallel with different --i values)
```

### Evaluation

```bash
cd eval/cell_type
python zero_shot_eval.py --input_dir prediction_output/
```

## Key Architecture Details

### Data Format

**Training data** consists of batch-level cell annotation problems:
- Each example contains N cells from the same donor
- Each cell has top expressed genes and context (donor info, disease state, etc.)
- N candidate cell types must be matched 1-to-1 with the N cells
- Models must assign unique types (no duplicates allowed)

**Model output format** must follow:
```
<think>
[reasoning steps]
</think>
<answer>
type1 | type2 | type3 | ...
</answer>
```

### Reward Function

The custom reward function is in `src/verl/verl/workers/reward_function/compute_score.py`. It:
- Validates the `<think>...</think>` and `<answer>...</answer>` format
- Extracts predictions from the answer tags
- Checks for uniqueness (no duplicate cell types)
- Computes partial accuracy (cell-level correctness) and exact match (batch-level)
- Returns -1 for invalid/malformed predictions

### VERL Framework

This project uses **verl** (Volcano Engine Reinforcement Learning), a flexible RL training library for LLMs. Key concepts:
- **Actor**: The policy model being trained
- **Rollout**: Generation phase using vLLM for fast inference
- **Ref**: Reference model for KL divergence computation
- **Critic**: Value function (PPO only, not used in GRPO)
- **FSDP**: Fully Sharded Data Parallel for distributed training
- **3D-HybridEngine**: Efficient resharding between training and generation

Configuration is done via command-line arguments to `verl.trainer.main_ppo` using Hydra-style syntax.

### Multi-shard Decoding

The `scripts/test.py` supports splitting inference across multiple processes:
- `--n`: Total number of shards
- `--i`: Shard index (0-based)
- Run multiple instances with different `--i` values on different GPUs

### Important File Paths

Before running training scripts, you must edit these paths:
- `src/sft/sft.sh`: Set `DATA_PATH`, `LLM_NAME`, `CACHE_DIR`
- `src/verl/examples/grpo_trainer/run_cello1_grpo.sh`: Set `TRAIN_DATA`, `VAL_DATA`, `SFT_MERGED_CKPT`, `REWARD_FN`
- `src/verl/examples/ppo_trainer/run_cello1_ppo.sh`: Set `TRAIN_DATA`, `VAL_DATA`, `MERGED_CKPT`, `REWARD_FN`
- `data/run_pipeline.sh`: Set `RAW_H5AD_DIR` and output directories

## Training Pipeline Flow

1. **Data Preparation**: Raw `.h5ad` → cell metadata → batch QA → train/test JSON
2. **SFT**: Fine-tune base model (e.g., Qwen2.5-7B) on expert reasoning traces with LoRA → merge weights
3. **RL Data Prep**: Convert JSON to Parquet format for efficient loading
4. **RL Training**: Use merged SFT checkpoint with GRPO/PPO and custom reward function
5. **Checkpoint Conversion**: FSDP format → HuggingFace format
6. **Inference**: Generate predictions on test set
7. **Evaluation**: Compute metrics (partial accuracy, exact match, format legitimacy, uniqueness)

## Environment Variables

- `VLLM_ATTENTION_BACKEND=XFORMERS`: Use efficient attention backend for vLLM
- `HF_HOME` or `HUGGINGFACE_HUB_CACHE`: HuggingFace cache directory
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility for processes
