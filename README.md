<h1 align="center">
  <img src="assets/cello1.png" alt="Cell-o1 Logo" width="60" style="vertical-align: middle; margin-right: 10px;" />
  Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning
</h1>


<p align="center">
  <a href="https://www.arxiv.org/abs/2506.02911" target="_blank"><img src="https://img.shields.io/badge/arXiv-2506.02911-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/ncbi-nlp/cell-o1"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/datasets/ncbi/CellPuzzles"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Dataset"></a>
  <a href="https://huggingface.co/ncbi/Cell-o1"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a>
</p>

<br>


<h2 id="1">📌 Overview</h2>

Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. 
To mimic this expert behavior, we introduce ***CellPuzzles***—a benchmark requiring unique cell-type assignments across cell batches. Existing LLMs struggle with this task, with the best baseline (OpenAI's o1) achieving only 19.0% batch accuracy. To address this, we present ***Cell-o1***, a reasoning-enhanced language model trained via SFT on distilled expert traces, followed by RL with batch-level rewards. ***Cell-o1*** outperforms all baselines on both cell-level and batch-level metrics, and exhibits emergent behaviors such as self-reflection and curriculum reasoning, offering insights into its interpretability and generalization.


<p align="center">
  <img src="assets/overview.png" alt="CellPuzzles Overview" width="95%">
</p>


<h2 id="2">🧰 Installation</h2>

```
conda create -n cello1 python=3.9
conda activate cello1

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


<h2 id="3">🚀 Quick Start: Train with Preprocessed Data</h2>
We provide preprocessed training and test data so you can get started immediately with model fine-tuning and reinforcement learning.

<h3 id="3-1">📦 Step 1: Download Preprocessed Data</h3>

You can load the dataset using the 🤗 `datasets` library:

```python
from datasets import load_dataset
import json

# Load all splits
dataset = load_dataset("ncbi/CellPuzzles")

# Access each split
reasoning_data = dataset["reasoning"]   # For SFT
train_data = dataset["train"]           # For GRPO training
test_data = dataset["test"]             # For evaluation

# Save each split to JSON
os.makedirs("processed_data", exist_ok=True)

with open("processed_data/sft_train.json", "w") as f:
    json.dump(reasoning_data, f, indent=2)

with open("processed_data/grpo_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("processed_data/test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)
```

- `reasoning`: Expert-like reasoning traces distilled from o1, used to cold start the model via SFT.

- `train`: Raw QA-style data used for RL (GRPO), containing both user prompts and gold answers.

- `test`: Held-out data for evaluation, formatted similarly to `train`.


<h3 id="3-2">🧠 Step 2: Supervised Fine-Tuning (SFT)</h3>

Use the `reasoning` split to cold start the model with expert-like reasoning traces.

```bash
cd sft
bash sft.sh
```



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

### Run the full pipeline

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

<h2 id="5-1"> 📚 References</h2>
If you use our repository, please cite the following related paper:

```
@article{fang2025cello1,
  title={Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning},
  author={Fang, Yin and Jin, Qiao and Xiong, Guangzhi and Jin, Bowen and Zhong, Xianrui and Ouyang, Siru and Zhang, Aidong and Han, Jiawei and Lu, Zhiyong},
  journal={arXiv preprint arXiv:2506.02911},
  year={2025}
}
```

<h2 id="5-2"> 🫱🏻‍🫲 Acknowledgements</h2>

We appreciate [verl](https://github.com/volcengine/verl), [TinyZero](https://github.com/Jiayi-Pan/TinyZero), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReCall](https://github.com/Agent-RL/ReCall), and many other related works for their open-source contributions. The logo of the model is automatically generated by [GPT-4o](https://openai.com/index/hello-gpt-4o/).

