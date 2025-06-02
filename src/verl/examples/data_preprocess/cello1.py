"""
Preprocess a cell-type annotation dataset into Parquet format for reinforcement learning tasks.
"""

import os
import argparse
import datasets

from verl.utils.hdfs_io import copy, makedirs


def get_final_answer(answer_str):
    # Optionally postprocess raw answer here
    return answer_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default='~/data/mydata', help="Path to save local parquet files")
    parser.add_argument('--hdfs_dir', type=str, default=None, help="Optional HDFS path to sync data")
    parser.add_argument('--train_file', type=str, required=True, help="Path to training JSON file")
    parser.add_argument('--val_file', type=str, required=True, help="Path to validation JSON file")
    parser.add_argument('--test_file', type=str, required=True, help="Path to test JSON file")

    args = parser.parse_args()

    data_files = {
        "train": args.train_file,
        "val": args.val_file,
        "test": args.test_file,
    }

    # Load dataset from JSON
    dataset = datasets.load_dataset("json", data_files=data_files)

    # Instruction template for model prompting
    instruction = (
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

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")

            return {
                "data_source": "cell_data",
                "prompt": [{
                    "role": "user",
                    "content": f"{question_raw} {instruction}"
                }],
                "ability": "cell_type_annotation",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": get_final_answer(answer_raw)
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": answer_raw,
                }
            }
        return process_fn

    # Map preprocessing function to each split
    dataset["train"] = dataset["train"].map(make_map_fn("train"), with_indices=True)
    dataset["val"] = dataset["val"].map(make_map_fn("val"), with_indices=True)
    dataset["test"] = dataset["test"].map(make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_dir, exist_ok=True)

    dataset["train"].to_parquet(os.path.join(args.local_dir, "train.parquet"))
    dataset["val"].to_parquet(os.path.join(args.local_dir, "val.parquet"))
    dataset["test"].to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # Optionally copy to HDFS
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
