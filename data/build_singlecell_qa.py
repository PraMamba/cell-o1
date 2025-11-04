"""
Step 2b: Generate single-cell level QA pairs directly from single-cell JSON records.

This script creates one QA pair per cell, supporting both:
- Open-ended: No candidate list provided
- Constrained: Candidate list from same donor/batch group

Usage:
    python build_singlecell_qa.py \
        --input_dir path/to/single_cell_json \
        --output_dir path/to/save_singlecell_qa_json \
        --mode openended  # or "constrained"
"""

import os
import json
import argparse
import random
from collections import defaultdict
import concurrent.futures

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

    return donor_groups, data

def build_openended_qa(cell_record):
    """Build open-ended single-cell QA (no candidate list)."""
    marker = "Top expressed genes are"
    context = cell_record.get("context", "")
    if marker in context:
        context = context.split(marker)[0].strip()
    
    top_genes = cell_record.get("top_expressed_genes", [])
    genes_str = ", ".join(top_genes)
    
    cell_line = f"Cell: {genes_str}"
    
    question = (
        f"Context: {context}\n\n"
        + cell_line
        + "\n\nDetermine the correct cell type for the cell above based on the gene expression pattern and donor context."
    )
    
    answer = cell_record.get("cell_type", "")
    
    return {
        "selected_cells": [cell_record["cell_index"]],
        "question": question,
        "answer": answer,
        "group": str((cell_record.get("donor_id"), cell_record.get("batch_id")) if cell_record.get("batch_id") else cell_record.get("donor_id")),
        "cell_index": cell_record["cell_index"]
    }

def build_constrained_qa(cell_record, candidate_types):
    """Build constrained single-cell QA (with candidate list)."""
    marker = "Top expressed genes are"
    context = cell_record.get("context", "")
    if marker in context:
        context = context.split(marker)[0].strip()
    
    top_genes = cell_record.get("top_expressed_genes", [])
    genes_str = ", ".join(top_genes)
    
    cell_line = f"Cell: {genes_str}"
    
    # Sort candidate types for consistency
    sorted_candidates = sorted(candidate_types)
    
    question = (
        f"Context: {context}\n\n"
        + cell_line
        + "\n\nMatch the cell above to one of the following cell types:\n"
        + "\n".join(sorted_candidates)
    )
    
    answer = cell_record.get("cell_type", "")
    
    return {
        "selected_cells": [cell_record["cell_index"]],
        "question": question,
        "answer": answer,
        "group": str((cell_record.get("donor_id"), cell_record.get("batch_id")) if cell_record.get("batch_id") else cell_record.get("donor_id")),
        "cell_index": cell_record["cell_index"]
    }

def generate_singlecell_qas_for_file(in_path, out_path, mode="openended"):
    """
    Generate single-cell QA pairs from single-cell JSON records.
    
    Args:
        in_path: Path to input single-cell JSON file
        out_path: Path to output QA JSON file
        mode: "openended" or "constrained"
    """
    donor_groups, all_records = load_and_group(in_path)
    
    qa_pairs = []
    cells_without_type = 0
    cells_without_donor = 0
    
    if mode == "openended":
        # Simple: one QA pair per cell, no candidate list
        for rec in all_records:
            if not rec.get("donor_id"):
                cells_without_donor += 1
                continue
            
            cell_type = rec.get("cell_type", "").strip()
            if not cell_type:
                cells_without_type += 1
                continue
            
            qa_pair = build_openended_qa(rec)
            qa_pairs.append(qa_pair)
    
    elif mode == "constrained":
        # One QA pair per cell, with candidate list from same group
        for rec in all_records:
            if not rec.get("donor_id"):
                cells_without_donor += 1
                continue
            
            cell_type = rec.get("cell_type", "").strip()
            if not cell_type:
                cells_without_type += 1
                continue
            
            # Get candidate types from same donor/batch group
            donor = rec.get("donor_id")
            batch = rec.get("batch_id")
            group_key = (donor, batch) if batch else donor
            
            group_records = donor_groups.get(group_key, [])
            candidate_types = list(set(
                r.get("cell_type", "").strip() 
                for r in group_records 
                if r.get("cell_type", "").strip()
            ))
            
            # Need at least 2 types in group (including the current cell)
            if len(candidate_types) < 2:
                continue
            
            qa_pair = build_constrained_qa(rec, candidate_types)
            qa_pairs.append(qa_pair)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'openended' or 'constrained'")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"[INFO] {os.path.basename(in_path)} → {len(qa_pairs)} single-cell QA pairs")
    print(f"       Mode: {mode}, Total cells: {len(all_records)}")
    print(f"       Skipped (no donor): {cells_without_donor}, Skipped (no cell_type): {cells_without_type}")
    return len(qa_pairs)

def main():
    parser = argparse.ArgumentParser(
        description="Generate single-cell level QA pairs from single-cell metadata JSONs."
    )
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing single-cell JSONs (from build_context.py)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save single-cell QA JSONs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="openended",
        choices=["openended", "constrained"],
        help="QA mode: 'openended' (no candidate list) or 'constrained' (with candidate list from same group)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if fname.endswith(".json") and not fname.endswith("_qa.json"):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(root, args.input_dir)
                out_dir = os.path.join(args.output_dir, rel_path)
                out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_qa.json")
                tasks.append((in_path, out_path))

    total = 0
    with concurrent.futures.ProcessPoolExecutor() as ex:
        futures = [
            ex.submit(generate_singlecell_qas_for_file, i, o, args.mode) 
            for i, o in tasks
        ]
        for fut in concurrent.futures.as_completed(futures):
            total += fut.result()

    print(f"\n[SUMMARY] Total single-cell QA pairs generated: {total}")
    print(f"[SUMMARY] Mode: {args.mode}")

if __name__ == "__main__":
    main()

