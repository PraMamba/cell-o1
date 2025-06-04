<h1 align="center">
🤔 Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning
</h1>

<p align="center">
  <a href="https://www.arxiv.org/abs/2506.02911" target="_blank"><img src="https://img.shields.io/badge/arXiv-2506.02911-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/ncbi-nlp/cell-o1"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/ncbi/CellPuzzles"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a>
  <a href="https://huggingface.co/ncbi/Cell-o1"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
</p>

<br>

---
## Installation

```
conda create -n cello1 python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray
pip install transformers==4.47.0
pip install trl==0.17.0

# verl
cd src/verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib
```

## 📦 Directory Structure

```
.
├── data/                # Data preprocessing pipeline
│   ├── build_context.py       # Step 1: Extract cell-level metadata from .h5ad
│   ├── match_qa.py          # Step 2: Group cells into batch-level QA format
│   ├── split_train_test.py      # Step 3: Format for LLM input, stratified split
│   └── run_pipeline.sh      # One-click runner for the entire pipeline
├── sft/                 # Supervised fine-tuning code
│   ├── sft_trainer.py
│   ├── merge_lora.py
│   └── sft.sh
├── verl/                # GRPO framework and preprocessing
│   └── examples/
│       ├── data_preprocess/cello1.py
│       ├── grpo_trainer/run_cello1_grpo.sh
│       └── ppo_trainer/run_cello1_ppo.sh
└── README.md            # This file
```

---

## 🚀 Quick Start: Data Preprocessing (3 Steps)

### Step 0: Configure paths

Edit the top section of `run_pipeline.sh`:

```bash
# Path to input raw h5ad files
RAW_H5AD_DIR="path/to/h5ad_dir"

# Output directories
CELL_JSON_DIR="path/to/output_cell_metadata"
BATCH_QA_DIR="path/to/output_batch_qa"
FINAL_OUTPUT_DIR="path/to/final_llm_input"
```

### Step 1–3: Run the full pipeline

```bash
cd data
bash run_pipeline.sh
```

This will:

1. Convert raw `.h5ad` to cell-level JSON with context and top genes.
2. Group cells into QA batches (each batch = N cells from a donor).
3. Convert QA into LLM-friendly input and split into `train` / `test`.

---

## ⚙️ Script Usage (Standalone)

```bash
# Step 1: Extract cell context
python data/build_context.py --input_dir path/to/h5ad --output_dir path/to/json

# Step 2: Generate batch QA
python data/match_qa.py --input_dir path/to/json --output_dir path/to/qa

# Step 3: Convert to LLM format
python data/split_train_test.py --input_dir path/to/qa --output_dir path/to/final --max_test_samples 1100
```

---

## 📦 Final Output Format

```
final_llm_input/
├── train/
│   ├── *.json                # Each sample contains {"system_msg", "user_msg", "answer"}
│   ├── train_data.json       # Merged training set
│   └── train_raw_data.json   # Original QA format (for RL)
├── test/
│   ├── *.json
│   └── test_data.json
```

---

## 🔁 SFT + GRPO Pipeline

### Step 1: Supervised Fine-Tuning (SFT)

Fine-tune a base model using distilled reasoning data.

```bash
cd sft
bash sft.sh
```

This runs:

- `sft_trainer.py` to train using LoRA
- `merge_lora.py` to merge LoRA weights into the base model

---

### Step 2: Preprocess for GRPO

Convert SFT-labeled QA batches into Parquet format.

```bash
cd verl

python examples/data_preprocess/cello1.py  --train_file /path/to/sft_train.json  --local_dir /path/to/output_parquet
```

---

### Step 3: Run GRPO Training

Train the policy using batch-level rewards and GRPO.

```bash
bash examples/grpo_trainer/run_cello1_grpo.sh
```

Before running, update:
- `train_files` / `val_files` (to `.parquet` files)
- `model.path` (pointing to merged SFT checkpoint)

---

### 🔄 Optional: Run PPO Instead

You can switch to PPO training with:

```bash
bash examples/ppo_trainer/run_cello1_ppo.sh
```

---

