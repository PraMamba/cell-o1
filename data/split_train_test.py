"""
Step 3: Convert QA-format JSONs into LLM-ready format with system/user/answer fields,
and split into stratified train/test sets.

Usage:
    python split_train_test.py \
        --input_dir path/to/qa_jsons \
        --output_dir path/to/final_data \
        --max_test_samples 1100
"""

import os
import json
import random
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Prepare LLM input data from QA JSONs with stratified test sampling.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input folder containing *_qa.json files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output folder")
    parser.add_argument("--max_test_samples", type=int, default=1100, help="Maximum number of test samples")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    test_output_dir = os.path.join(output_dir, "test")
    train_output_dir = os.path.join(output_dir, "train")
    test_merged_file = os.path.join(test_output_dir, "test_data.json")
    train_merged_file = os.path.join(train_output_dir, "train_data.json")
    train_raw_merged_file = os.path.join(train_output_dir, "train_raw_data.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(train_output_dir, exist_ok=True)

    system_msg = (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given a batch of N cells from the same donor, where each cell represents a unique cell type. "
        "For each cell, the top expressed genes are provided in descending order of expression. "
        "Using both the gene expression data and donor information, determine the correct cell type for each cell. "
        "You will also receive a list of N candidate cell types, and each candidate must be assigned to exactly one cell. "
        "Ensure that you consider all cells and candidate types together, rather than annotating each cell individually. "
        "You must output exactly \"<think>...</think>\\n<answer>...</answer>.\" "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single string listing the assigned cell types in order, separated by ' | '."
    )
    

    # First pass: gather all items by answer length
    cell_count_items = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith("_qa.json"):
                continue
            path = os.path.join(root, fname)
            try:
                data = json.load(open(path))
                for idx, item in enumerate(data):
                    count = len(item.get("answer", "").split("|"))
                    cell_count_items[count].append((path, idx))
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")

    # Compute test quotas per category
    total_available = sum(len(v) for v in cell_count_items.values())
    max_test = args.max_test_samples
    test_quotas = {}

    if total_available <= max_test:
        for k, v in cell_count_items.items():
            test_quotas[k] = max(1, int(len(v) * 0.1))
    else:
        min_per_k = 5
        remaining = max_test - len(cell_count_items) * min_per_k
        for k in cell_count_items:
            test_quotas[k] = min(min_per_k, len(cell_count_items[k]))
        if remaining > 0:
            total_rest = total_available - sum(test_quotas.values())
            if total_rest > 0:
                for k, v in cell_count_items.items():
                    if len(v) > test_quotas[k]:
                        extra = int((len(v) - test_quotas[k]) / total_rest * remaining)
                        test_quotas[k] += extra
    total_test_quota = sum(test_quotas.values())
    if total_test_quota > max_test:
        scale = max_test / total_test_quota
        for k in test_quotas:
            test_quotas[k] = max(1, int(test_quotas[k] * scale))

    # Sample test items
    random.seed(42)
    test_file_indices = defaultdict(set)
    for k, items in cell_count_items.items():
        sample = random.sample(items, min(len(items), test_quotas[k]))
        for path, idx in sample:
            test_file_indices[path].add(idx)

    # Second pass: split & save
    all_test_data, all_train_data, all_train_raw_data = [], [], []
    test_len_dist, train_len_dist = defaultdict(int), defaultdict(int)
    files_processed, items_processed = 0, 0

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith("_qa.json"):
                continue
            path = os.path.join(root, fname)
            rel_path = os.path.relpath(root, input_dir)
            test_file = os.path.join(test_output_dir, rel_path, fname.replace("_qa.json", "_test.json"))
            train_file = os.path.join(train_output_dir, rel_path, fname.replace("_qa.json", "_train.json"))
            train_raw_file = os.path.join(train_output_dir, rel_path, fname.replace("_qa.json", "_train_raw.json"))
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            os.makedirs(os.path.dirname(train_file), exist_ok=True)

            data = json.load(open(path))
            test_data, train_data, train_raw = [], [], []
            for i, item in enumerate(data):
                count = len(item.get("answer", "").split("|"))
                new_item = {
                    "system_msg": system_msg,
                    "user_msg": item.get("question", ""),
                    "assistant_msg": item.get("answer", "")
                }
                if i in test_file_indices[path]:
                    test_data.append(new_item)
                    all_test_data.append(new_item)
                    test_len_dist[count] += 1
                else:
                    train_data.append(new_item)
                    all_train_data.append(new_item)
                    train_raw.append(item)
                    all_train_raw_data.append(item)
                    train_len_dist[count] += 1

            json.dump(test_data, open(test_file, "w"), indent=2, ensure_ascii=False)
            json.dump(train_data, open(train_file, "w"), indent=2, ensure_ascii=False)
            json.dump(train_raw, open(train_raw_file, "w"), indent=2, ensure_ascii=False)

            files_processed += 1
            items_processed += len(data)

    json.dump(all_test_data, open(test_merged_file, "w"), indent=2, ensure_ascii=False)
    json.dump(all_train_data, open(train_merged_file, "w"), indent=2, ensure_ascii=False)
    json.dump(all_train_raw_data, open(train_raw_merged_file, "w"), indent=2, ensure_ascii=False)

    print(f"\nProcessed {files_processed} files with {items_processed} items.")
    print(f"Merged test samples: {len(all_test_data)} → {test_merged_file}")
    print(f"Merged train samples: {len(all_train_data)} → {train_merged_file}")
    print(f"Raw train samples:    {len(all_train_raw_data)} → {train_raw_merged_file}")

if __name__ == "__main__":
    main()
