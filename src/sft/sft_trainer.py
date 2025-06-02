# SFTTrainer: Supervised Fine-Tuning script using TRL and PEFT (LoRA)
import os
import json
import random
import argparse
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def sft_training(
    model,
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
):
    os.makedirs(save_dir, exist_ok=True)

    # Load JSON dataset into a pandas DataFrame
    df = pd.DataFrame(json.load(open(data_path)))
    
    # Construct 'messages' in OpenAI format
    df["messages"] = df.apply(
        lambda x: [
            {"role": "system", "content": x["system_msg"]},
            {"role": "user", "content": x["user_msg"]},
            {"role": "assistant", "content": x["assistant_msg"]}
        ],
        axis=1
    )
    df["index"] = df.index

    # Split data into training and evaluation sets
    all_indices = df["index"].tolist()
    if eval_ratio > 0:
        random.seed(0)
        random.shuffle(all_indices)
        eval_count = round(len(all_indices) * eval_ratio)
        eval_indices = all_indices[:eval_count]
        eval_dataset = Dataset.from_list(
            df[df["index"].isin(eval_indices)][["messages"]].to_dict("records")
        )
        eval_strategy = "epoch"
        print("Number of evaluation samples:", len(eval_dataset))
    else:
        eval_indices = []
        eval_dataset = None
        eval_strategy = "no"

    train_indices = [i for i in all_indices if i not in eval_indices]
    train_dataset = Dataset.from_list(
        df[df["index"].isin(train_indices)][["messages"]].to_dict("records")
    )
    print("Number of training samples:", len(train_dataset))

    # Identify token span of assistant response in chat template
    def longest_common_prefix(s1, s2):
        for i, (a, b) in enumerate(zip(s1, s2)):
            if a != b:
                return s1[:i]
        return s1

    example1 = tokenizer.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": ""}
    ], tokenize=True)

    example2 = tokenizer.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""}
    ], tokenize=True)

    example3 = tokenizer.apply_chat_template([
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Placeholder."}
    ], tokenize=True)

    prefix = longest_common_prefix(example1, example2)
    full_prefix = longest_common_prefix(example2, example3)
    response_template_ids = full_prefix[len(prefix):]
    print("Response template prefix:", tokenizer.decode(response_template_ids))

    # Set tokenizer padding config
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_template_ids,
        tokenizer=tokenizer,
        mlm=False
    )

    # LoRA config
    peft_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        use_dora=True
    )

    # Training configuration
    training_args = SFTConfig(
        output_dir=save_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=n_epochs,
        learning_rate=lr,
        logging_steps=1,
        eval_strategy=eval_strategy,
        save_strategy="epoch",
        save_only_model=False,
        max_seq_length=max_seq_length
    )

    # Initialize trainer
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/sft_data.json")
    parser.add_argument("--cache_dir", type=str, default="cache/huggingface")
    parser.add_argument("--r", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args()

    save_dir = os.path.join(
        args.data_path.replace("data", "trained_models").replace(".csv", "").replace(".json", ""),
        args.llm_name.replace('/', '_'),
        "sft",
        f"{args.max_seq_length}_{args.r}_{args.lr}_{args.n_epochs}_{args.batch_size}_{args.accumulation_steps}_{args.eval_ratio}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        device_map="auto",
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_name,
        cache_dir=args.cache_dir
    )

    sft_training(
        model=model,
        tokenizer=tokenizer,
        data_path=args.data_path,
        save_dir=save_dir,
        r=args.r,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        eval_ratio=args.eval_ratio,
        max_seq_length=args.max_seq_length
    )
