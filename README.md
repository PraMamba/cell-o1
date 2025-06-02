



## Step 1: Supervised Fine-Tuning (SFT)

Fine-tune a base model using distilled reasoning data.

cd sft

```bash
bash sft/sft.sh

This runs:
- sft_trainer.py to train with LoRA
- merge_lora.py to merge LoRA weights into the base model


## Step 2: Preprocess for GRPO

Convert cell-type annotation batches into parquet format.

cd verl

```python
python examples/data_preprocess/cello1.py \
    --train_file /path/to/sft_train.json \
    --val_file /path/to/sft_val.json \
    --test_file /path/to/sft_test.json \
    --local_dir /path/to/output

## Step 3: Run GRPO Training

Train the policy using GRPO.
```bash
bash examples/grpo_trainer/run_cello1_grpo.sh

Before running, edit the script to configure:
- train_files / val_files paths (to preprocessed .parquet)
- model.path (to merged SFT checkpoint)

## Optional: Run PPO Instead

You can also run PPO as an alternative:
```bash
bash verl/examples/ppo_trainer/run_cello1_ppo.sh

