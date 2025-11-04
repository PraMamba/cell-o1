"""
Convert QA JSON format to JSONL conversation format for LLM training.

Input format:
{
  "selected_cells": ["cell_id"],
  "question": "...",
  "answer": "...",
  "group": "...",
  "cell_index": "..."
}

Output format (JSONL):
{
  "selected_cells": ["cell_id"],
  "group": "...",
  "cell_index": "...",
  "conversations": [
    {"from": "system", "value": "system_msg"},
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "answer"}
  ]
}

Usage:
    python convert_qa_to_jsonl.py \
        --input_file /path/to/input.json \
        --output_file /path/to/output.jsonl \
        --mode single_openended
"""

import json
import argparse
from pathlib import Path


SYSTEM_MESSAGES = {
    "single_openended": (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given information about a single cell, including the top expressed genes in descending order. "
        "Using the gene expression data and donor information, determine the correct cell type for this cell. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single cell type name."
    ),
    "single_constrained": (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given information about a single cell, including the top expressed genes in descending order. "
        "You will also receive a list of candidate cell types to choose from. "
        "Using the gene expression data and donor information, determine the correct cell type for this cell from the candidate list. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single cell type name from the candidate list."
    ),
    "batch_openended": (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given a batch of N cells from the same donor, where each cell represents a unique cell type. "
        "For each cell, the top expressed genes are provided in descending order of expression. "
        "Using both the gene expression data and donor information, determine the correct cell type for each cell. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single string listing the assigned cell types in order, separated by ' | '."
    ),
    "batch_constrained": (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given a batch of N cells from the same donor, where each cell represents a unique cell type. "
        "For each cell, the top expressed genes are provided in descending order of expression. "
        "Using both the gene expression data and donor information, determine the correct cell type for each cell. "
        "You will also receive a list of N candidate cell types, and each candidate must be assigned to exactly one cell. "
        "Ensure that you consider all cells and candidate types together, rather than annotating each cell individually. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single string listing the assigned cell types in order, separated by ' | '."
    )
}


def convert_qa_to_conversations(qa_item: dict, system_msg: str) -> dict:
    """Convert a single QA item to conversation format."""
    conversation_item = {
        "selected_cells": qa_item.get("selected_cells", []),
        "group": qa_item.get("group", ""),
        "cell_index": qa_item.get("cell_index", ""),
        "conversations": [
            {
                "from": "system",
                "value": system_msg
            },
            {
                "from": "human",
                "value": qa_item["question"]
            },
            {
                "from": "gpt",
                "value": qa_item["answer"]
            }
        ]
    }
    return conversation_item


def main():
    parser = argparse.ArgumentParser(description="Convert QA JSON to JSONL conversation format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input QA JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single_openended", "single_constrained", "batch_openended", "batch_constrained"],
        help="Task mode to determine system message"
    )

    args = parser.parse_args()

    # Get system message for the mode
    system_msg = SYSTEM_MESSAGES[args.mode]

    # Load input JSON
    print(f"[INFO] Loading QA data from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    print(f"[INFO] Loaded {len(qa_data)} QA pairs")

    # Convert to conversation format
    conversations = []
    for qa_item in qa_data:
        conv_item = convert_qa_to_conversations(qa_item, system_msg)
        conversations.append(conv_item)

    # Save as JSONL
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Writing JSONL to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"[SUCCESS] Converted {len(conversations)} items to JSONL format")
    print(f"[INFO] Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()
