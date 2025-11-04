"""
Step 3: Convert QA-format JSONs into LLM-ready format with system/user/answer fields,
and split into stratified train/test sets.

Usage:
    # Normal mode: Split data into train and test sets
    python split_train_test.py \
        --input_dir path/to/qa_jsons \
        --output_dir path/to/final_data \
        --max_test_samples 1100
    
    # Test-only mode: Put all data into test set (no train set)
    python split_train_test.py \
        --input_dir path/to/qa_jsons \
        --output_dir path/to/final_data \
        --test_only
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
    parser.add_argument("--test_only", action="store_true", help="Put all data into test set without splitting (no train set)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    test_output_dir = os.path.join(output_dir, "test")
    train_output_dir = os.path.join(output_dir, "train")
    # For evaluation: keep original QA format (question, answer, group, etc.)
    test_qa_merged_file = os.path.join(test_output_dir, "test_data.json")
    # For LLM training: convert to system_msg, user_msg, assistant_msg format
    test_llm_merged_file = os.path.join(test_output_dir, "test_llm_data.json")
    if not args.test_only:
        train_merged_file = os.path.join(train_output_dir, "train_data.json")
        train_raw_merged_file = os.path.join(train_output_dir, "train_raw_data.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    if not args.test_only:
        os.makedirs(train_output_dir, exist_ok=True)
    
    if args.test_only:
        print("[INFO] Test-only mode: All data will be placed in test set (no train set)")

    # Detect QA type (batch vs single-cell) from first file
    is_singlecell = False
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith("_qa.json"):
                continue
            path = os.path.join(root, fname)
            try:
                data = json.load(open(path))
                if data:
                    # Check if answer contains '|' separator (batch-level) or not (single-cell)
                    first_answer = data[0].get("answer", "")
                    if "|" not in first_answer:
                        is_singlecell = True
                    break
            except Exception as e:
                print(f"[WARN] Skipping {path}: {e}")
        if is_singlecell:
            break
    
    # Set system message based on QA type
    if is_singlecell:
        system_msg = (
            "You are an expert assistant specialized in cell type annotation. "
            "You will be given information about a single cell, including the top expressed genes in descending order. "
            "Using the gene expression data and donor information, determine the correct cell type for this cell. "
            "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
            "The final answer should be a single cell type name."
        )
        print("[INFO] Detected single-cell QA format")
    else:
        system_msg = (
            "You are an expert assistant specialized in cell type annotation. "
            "You will be given a batch of N cells from the same donor, where each cell represents a unique cell type. "
            "For each cell, the top expressed genes are provided in descending order of expression. "
            "Using both the gene expression data and donor information, determine the correct cell type for each cell. "
            "You will also receive a list of N candidate cell types, and each candidate must be assigned to exactly one cell. "
            "Ensure that you consider all cells and candidate types together, rather than annotating each cell individually. "
            "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
            "The final answer should be a single string listing the assigned cell types in order, separated by ' | '."
        )
        print("[INFO] Detected batch-level QA format")

    # If test_only mode, skip splitting logic and process all data as test set
    if args.test_only:
        test_file_indices = None  # Will be ignored in processing loop
    else:
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
                        # For single-cell, answer is single type (count = 1)
                        # For batch-level, answer is "type1 | type2 | ..." (count = number of types)
                        answer = item.get("answer", "")
                        if "|" in answer:
                            count = len(answer.split("|"))
                        else:
                            count = 1 if answer.strip() else 0
                        if count > 0:
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
    # For evaluation: original QA format (question, answer, group, etc.)
    all_test_qa_data, all_train_qa_data = [], []
    # For LLM training: converted format (system_msg, user_msg, assistant_msg)
    all_test_llm_data, all_train_llm_data = [], []
    all_train_raw_data = []
    test_len_dist, train_len_dist = defaultdict(int), defaultdict(int)
    files_processed, items_processed = 0, 0

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.endswith("_qa.json"):
                continue
            path = os.path.join(root, fname)
            rel_path = os.path.relpath(root, input_dir)
            test_file = os.path.join(test_output_dir, rel_path, fname.replace("_qa.json", "_test.json"))
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            
            train_file = None
            train_raw_file = None
            if not args.test_only:
                train_file = os.path.join(train_output_dir, rel_path, fname.replace("_qa.json", "_train.json"))
                train_raw_file = os.path.join(train_output_dir, rel_path, fname.replace("_qa.json", "_train_raw.json"))
                os.makedirs(os.path.dirname(train_file), exist_ok=True)

            data = json.load(open(path))
            test_qa_data, train_qa_data = [], []
            test_llm_data, train_llm_data = [], []
            train_raw = []
            for i, item in enumerate(data):
                # For single-cell, answer is single type (count = 1)
                # For batch-level, answer is "type1 | type2 | ..." (count = number of types)
                answer = item.get("answer", "")
                if "|" in answer:
                    count = len(answer.split("|"))
                else:
                    count = 1 if answer.strip() else 0
                
                # Original QA format for evaluation (preserve all fields)
                qa_item = {}
                # Preserve question and answer (required)
                if "question" in item:
                    qa_item["question"] = item["question"]
                if "answer" in item:
                    qa_item["answer"] = item["answer"]
                # Preserve optional fields if they exist
                if "group" in item and item["group"]:
                    qa_item["group"] = item["group"]
                if "selected_cells" in item:
                    qa_item["selected_cells"] = item["selected_cells"]
                if "cell_index" in item and item["cell_index"]:
                    qa_item["cell_index"] = item["cell_index"]
                # Preserve other fields that might be needed (e.g., for batch decomposition)
                for key in ["original_batch_index", "cell_index_in_batch", "total_cells_in_batch"]:
                    if key in item:
                        qa_item[key] = item[key]
                
                # LLM training format
                llm_item = {
                    "system_msg": system_msg,
                    "user_msg": item.get("question", ""),
                    "assistant_msg": item.get("answer", "")
                }
                
                # In test_only mode, all items go to test set
                # Otherwise, use test_file_indices to split
                if args.test_only or (test_file_indices is not None and i in test_file_indices.get(path, set())):
                    test_qa_data.append(qa_item)
                    test_llm_data.append(llm_item)
                    all_test_qa_data.append(qa_item)
                    all_test_llm_data.append(llm_item)
                    test_len_dist[count] += 1
                else:
                    train_qa_data.append(qa_item)
                    train_llm_data.append(llm_item)
                    train_raw.append(item)
                    all_train_qa_data.append(qa_item)
                    all_train_llm_data.append(llm_item)
                    all_train_raw_data.append(item)
                    train_len_dist[count] += 1

            # Save per-file outputs (optional, for debugging)
            # Only save QA format for evaluation (test_qa_data), not LLM format per file
            if test_qa_data:
                json.dump(test_qa_data, open(test_file, "w"), indent=2, ensure_ascii=False)
            if not args.test_only:
                if train_qa_data:
                    json.dump(train_qa_data, open(train_file, "w"), indent=2, ensure_ascii=False)
                json.dump(train_raw, open(train_raw_file, "w"), indent=2, ensure_ascii=False)

            files_processed += 1
            items_processed += len(data)

    # Save merged files
    # For evaluation: original QA format (question, answer, group, etc.)
    json.dump(all_test_qa_data, open(test_qa_merged_file, "w"), indent=2, ensure_ascii=False)
    # For LLM training: converted format (system_msg, user_msg, assistant_msg)
    json.dump(all_test_llm_data, open(test_llm_merged_file, "w"), indent=2, ensure_ascii=False)
    
    if not args.test_only:
        json.dump(all_train_qa_data, open(train_merged_file, "w"), indent=2, ensure_ascii=False)
        json.dump(all_train_raw_data, open(train_raw_merged_file, "w"), indent=2, ensure_ascii=False)

    print(f"\nProcessed {files_processed} files with {items_processed} items.")
    print(f"[INFO] For evaluation (original QA format):")
    print(f"  Merged test samples: {len(all_test_qa_data)} → {test_qa_merged_file}")
    print(f"[INFO] For LLM training (converted format):")
    print(f"  Merged test samples: {len(all_test_llm_data)} → {test_llm_merged_file}")
    
    if args.test_only:
        print(f"[INFO] Test-only mode: All {len(all_test_qa_data)} samples are in test set (no train set)")
    else:
        print(f"[INFO] Train samples (original QA format): {len(all_train_qa_data)} → {train_merged_file}")
        print(f"[INFO] Raw train samples: {len(all_train_raw_data)} → {train_raw_merged_file}")

if __name__ == "__main__":
    main()
