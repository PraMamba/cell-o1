# SFTTrainer: Supervised Fine‑Tuning script using TRL + PEFT (LoRA)
"""
Train a causal‑LM with LoRA adapters using TRL's SFTTrainer.
"""

import os
import json
import random
import argparse
from typing import List, Union

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import (
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------
# Core training routine
# ------------------------------------------------

def sft_training(
    model: torch.nn.Module,
    tokenizer,
    data_path: str,
    save_dir: str,
    r: int = 256,
    lr: float = 5e-5,
    n_epochs: int = 3,
    batch_size: int = 4,
    accumulation_steps: int = 16,
    eval_ratio: float = 0.1,
    max_seq_length: int = 4096,
) -> None:
    """Main entry for supervised fine‑tuning with LoRA adapters."""

    os.makedirs(save_dir, exist_ok=True)

    # Load dataset -----------------------------------------------------------
    df = pd.DataFrame(json.load(open(data_path)))
    df["messages"] = df.apply(
        lambda x: [
            {"role": "system", "content": x["system_msg"]},
            {"role": "user", "content": x["user_msg"]},
            {"role": "assistant", "content": x["assistant_msg"]},
        ],
        axis=1,
    )
    df["index"] = df.index

    # Split train/eval --------------------------------------------------------
    if eval_ratio == 0:
        eval_indices = []
    else:
        all_indices = df["index"].tolist()
        random.seed(0)
        random.shuffle(all_indices)
        eval_n = round(len(all_indices) * eval_ratio)
        eval_indices = all_indices[:eval_n]

    train_indices = [idx for idx in df["index"] if idx not in eval_indices]

    train_dataset = Dataset.from_list(
        df[df["index"].isin(train_indices)][["messages"]].to_dict("records")
    )
    print("Number of training sample:", len(train_dataset))

    if not eval_indices:
        eval_dataset = None
        eval_strategy = "no"
    else:
        eval_dataset = Dataset.from_list(
            df[df["index"].isin(eval_indices)][["messages"]].to_dict("records")
        )
        eval_strategy = "epoch"
        print("Number of evaluation sample:", len(eval_dataset))

    # Build response template ids -------------------------------------------
    def longest_common_prefix(s1, s2):
        prefix = []
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix.append(c1)
            else:
                break
        return prefix

    example1 = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
        tokenize=True,
    )
    example2 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ],
        tokenize=True,
    )
    example3 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Placeholder."},
        ],
        tokenize=True,
    )

    response_template_ids = longest_common_prefix(example2, example3)[
        len(longest_common_prefix(example1, example2)) :
    ]
    print("Response template:", tokenizer.decode(response_template_ids))

    # Data collator ----------------------------------------------------------
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # PEFT / LoRA config -----------------------------------------------------
    peft_config = LoraConfig(
        use_dora=True,
        task_type="CAUSAL_LM",
        lora_alpha=r * 2,
        lora_dropout=0.05,
        r=r,
        bias="none",
        target_modules="all-linear",
    )

    # Training arguments -----------------------------------------------------
    training_args = SFTConfig(
        output_dir=save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=n_epochs,
        learning_rate=lr,
        logging_steps=1,
        eval_strategy=eval_strategy,
        max_seq_length=max_seq_length,
        save_strategy="epoch",
        save_only_model=False,
    )

    # Trainer ---------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=peft_config,
    )

    trainer.train()


# ------------------------------------------------
# CLI entry point
# ------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", default="./train.json")
    parser.add_argument("--cache_dir", default="../huggingface/hub")
    parser.add_argument("--r", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.llm_name,
        device_map="auto",
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.llm_name,
        cache_dir=args.cache_dir,
    )

    sft_training(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        save_dir=args.save_dir,
        r=args.r,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        eval_ratio=args.eval_ratio,
        max_seq_length=args.max_seq_length,
    )
