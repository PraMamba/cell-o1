"""
Step 2: Convert single-cell records into grouped Q&A format for batch-level annotation.

Usage:
    python match_qa.py \
        --input_dir path/to/single_cell_json \
        --output_dir path/to/save_batch_qa_json
"""

import os
import json
import random
import argparse
from collections import defaultdict
import concurrent.futures

MAX_CELLS_PER_QA = 15
TARGET_PER_N = 40
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def load_and_group(json_file_path):
    """Group records by donor and batch (if available)."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, rec in enumerate(data):
        rec.setdefault("cell_index", str(i))

    donor_groups = defaultdict(list)
    for rec in data:
        donor = rec.get("donor_id")
        if not donor:
            continue
        batch = rec.get("batch_id")
        key = (donor, batch) if batch else donor
        donor_groups[key].append(rec)

    return donor_groups

def sample_pair(records, num_cells):
    types = list({r["cell_type"] for r in records if r.get("cell_type")})
    chosen_types = random.sample(types, num_cells)
    chosen_records = [
        random.choice([r for r in records if r["cell_type"] == t])
        for t in chosen_types
    ]
    return chosen_records, chosen_types

def build_one_qa(chosen_records, chosen_types, group_key):
    marker = "Top expressed genes are"
    context = chosen_records[0].get("context", "")
    if marker in context:
        context = context.split(marker)[0]

    cell_lines = [
        f"Cell {i}: {', '.join(rec.get('top_expressed_genes', []))}"
        for i, rec in enumerate(chosen_records, start=1)
    ]
    question = (
        f"Context: {context}\n\n"
        + "\n".join(cell_lines)
        + "\n\nMatch the cells above to one of the following cell types:\n"
        + "\n".join(sorted(chosen_types))
    )
    answer = " | ".join(chosen_types)

    return {
        "selected_cells": [rec["cell_index"] for rec in chosen_records],
        "question": question,
        "answer": answer,
        "group": str(group_key),
    }

def generate_pairs_for_file(in_path, out_path):
    donor_groups = load_and_group(in_path)
    remaining = {n: TARGET_PER_N for n in range(3, MAX_CELLS_PER_QA + 1)}
    qa_pairs = []

    made_progress = True
    while any(v > 0 for v in remaining.values()) and made_progress:
        made_progress = False
        for gkey, records in donor_groups.items():
            types = list({r["cell_type"] for r in records if r.get("cell_type")})
            max_n = min(len(types), MAX_CELLS_PER_QA)
            if max_n < 8:
                continue
            valid_ns = [n for n in range(8, max_n + 1) if remaining[n] > 0]
            if not valid_ns:
                continue
            n = random.choice(valid_ns)
            recs, types = sample_pair(records, n)
            qa_pairs.append(build_one_qa(recs, types, gkey))
            remaining[n] -= 1
            made_progress = True

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"[INFO] {os.path.basename(in_path)} → {len(qa_pairs)} QA pairs")
    return len(qa_pairs)

def main():
    parser = argparse.ArgumentParser(description="Generate batch-level QA pairs from single-cell metadata.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing single-cell JSONs")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save batch QA JSONs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if fname.endswith(".json"):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(root, args.input_dir)
                out_dir = os.path.join(args.output_dir, rel_path)
                out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_qa.json")
                tasks.append((in_path, out_path))

    total = 0
    with concurrent.futures.ProcessPoolExecutor() as ex:
        futures = [ex.submit(generate_pairs_for_file, i, o) for i, o in tasks]
        for fut in concurrent.futures.as_completed(futures):
            total += fut.result()

    print(f"\n[SUMMARY] Total QA pairs generated: {total}")

if __name__ == "__main__":
    main()
