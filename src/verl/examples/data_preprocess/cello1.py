"""
Preprocess a cell‑type annotation dataset into Parquet format for reinforcement
learning tasks.

Changes vs. original version
----------------------------
* Only the training JSON file is required; the first **8** examples are copied
  into a held‑out *test* split.
* `--test_file` argument removed (or kept optional but ignored if not given).
"""

import os
import argparse
import datasets

from verl.utils.hdfs_io import copy, makedirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_final_answer(answer_str: str) -> str:
    """Post‑process raw ground‑truth answer if needed (placeholder)."""
    return answer_str


def make_map_fn(split: str, instruction: str):
    """Return a mapping function that adds the prompt/reward schema."""

    def process_fn(example, idx):
        question_raw = example["user_msg"]
        answer_raw = example["assistant_msg"]
        
        return {
            "data_source": "cell_data",
            "prompt": [
                {
                    "role": "user",
                    "content": f"{question_raw} {instruction}",
                }
            ],
            "ability": "cell_type_annotation",
            "reward_model": {
                "style": "rule",
                "ground_truth": get_final_answer(answer_raw),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question_raw,
                "answer": answer_raw,
            },
        }

    return process_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        type=str,
        default="~/data/mydata",
        help="Directory to save Parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS path to sync data.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training JSON file.",
    )
    # `--test_file` is optional/ignored; kept for backward compatibility
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="(Ignored) – test split will be derived from the first 8 samples.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load dataset – only train file is needed
    # ---------------------------------------------------------------------
    dataset = datasets.load_dataset("json", data_files={"train": args.train_file})

    # Instruction template ------------------------------------------------
    instruction = (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given a batch of N cells from the same donor, where each "
        "cell represents a unique cell type. For each cell, the top expressed "
        "genes are provided in descending order of expression. Using both the "
        "gene expression data and donor information, determine the correct cell "
        "type for each cell. You will also receive a list of N candidate cell "
        "types, and each candidate must be assigned to exactly one cell. Ensure "
        "that you consider all cells and candidate types together, rather than "
        "annotating each cell individually. Include your detailed reasoning within "
        "<think> and </think> tags, and provide your final answer within <answer> "
        "and </answer> tags. The final answer should be a single string listing "
        "the assigned cell types in order, separated by ' | '."
    )

    # Apply mapping -------------------------------------------------------
    dataset["train"] = dataset["train"].map(
        make_map_fn("train", instruction),
        with_indices=True,
    )

    # Derive test split: first 8 examples ---------------------------------
    dataset["test"] = dataset["train"].select(range(min(8, len(dataset["train"]))))
    dataset["test"] = dataset["test"].map(make_map_fn("test", instruction), with_indices=True)

    # Save to Parquet ------------------------------------------------------
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    dataset["train"].to_parquet(os.path.join(local_dir, "train.parquet"))
    dataset["test"].to_parquet(os.path.join(local_dir, "test.parquet"))

    # Optionally sync to HDFS --------------------------------------------
    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
