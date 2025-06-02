"""
Merge LoRA weights into base model and save the merged model.

Usage:
    python merge_lora.py \
        --base_model_path /path/to/base_model \
        --lora_ckpt_path /path/to/lora/checkpoint \
        --output_dir /path/to/save/merged_model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(base_model_path, lora_ckpt_path, output_dir, device="cpu"):
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map=device)

    print(f"Loading LoRA weights from: {lora_ckpt_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_ckpt_path, device_map=device)

    print("Merging LoRA weights into base model...")
    merged_model = lora_model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_dir)

    print("Merge complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model (e.g., llama or qwen)")
    parser.add_argument("--lora_ckpt_path", type=str, required=True, help="Path to the LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load models on (e.g., 'cpu', 'cuda')")
    args = parser.parse_args()

    merge_lora_weights(
        base_model_path=args.base_model_path,
        lora_ckpt_path=args.lora_ckpt_path,
        output_dir=args.output_dir,
        device=args.device
    )
