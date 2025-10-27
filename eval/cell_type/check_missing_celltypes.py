"""
Check Missing Cell Types in Mapping

This script scans all prediction JSON files to find cell types that are not
covered by the celltype_mapping.py mappings.

Usage:
    python check_missing_celltypes.py \
        --results_dir /path/to/eval_results \
        --dataset A013
"""

import os
import json
import argparse
from typing import Set, List, Dict
from collections import defaultdict

from celltype_mapping import (
    CELLTYPE_MAPPING_A013,
    CELLTYPE_MAPPING_D099,
    standardize_celltype
)


def load_all_celltypes_from_predictions(results_dir: str) -> Set[str]:
    """
    Load all unique cell types from all prediction files.

    Args:
        results_dir: Directory containing evaluation results

    Returns:
        Set of all unique cell type names
    """
    all_celltypes = set()

    # Scan all subdirectories
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') and 'predictions' in file:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract cell types from predictions
                    for item in data:
                        # Ground truth
                        gt = item.get('ground_truth', '')
                        if '|' in gt:
                            # Batch format
                            for celltype in gt.split('|'):
                                celltype = celltype.strip()
                                if celltype:
                                    all_celltypes.add(celltype)
                        elif gt.strip():
                            # Single cell format
                            all_celltypes.add(gt.strip())

                        # Predictions
                        pred = item.get('predicted_answer', '')
                        if '|' in pred:
                            # Batch format
                            for celltype in pred.split('|'):
                                celltype = celltype.strip()
                                if celltype:
                                    all_celltypes.add(celltype)
                        elif pred.strip():
                            # Single cell format
                            all_celltypes.add(pred.strip())

                    print(f"[INFO] Processed: {file_path}")

                except Exception as e:
                    print(f"[WARNING] Failed to process {file_path}: {e}")

    return all_celltypes


def get_mapped_celltypes(dataset: str) -> Set[str]:
    """
    Get all cell types that are covered by the mapping.

    Args:
        dataset: Dataset identifier ("A013" or "D099")

    Returns:
        Set of all mapped cell type names (case-insensitive)
    """
    if dataset == "A013":
        mapping = CELLTYPE_MAPPING_A013
    elif dataset == "D099":
        mapping = CELLTYPE_MAPPING_D099
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    mapped_names = set()
    for entry in mapping:
        # Add raw name
        mapped_names.add(entry['raw_name'].lower())
        # Add standardized name
        mapped_names.add(entry['standardized_name'].lower())
        # Add synonyms if exists
        if 'synonyms' in entry:
            for synonym in entry['synonyms']:
                mapped_names.add(synonym.lower())

    return mapped_names


def find_missing_celltypes(
    all_celltypes: Set[str],
    mapped_celltypes: Set[str]
) -> List[str]:
    """
    Find cell types that are not in the mapping.

    Args:
        all_celltypes: All cell types found in predictions
        mapped_celltypes: All cell types covered by mapping

    Returns:
        List of missing cell type names
    """
    missing = []

    for celltype in all_celltypes:
        if celltype.lower() not in mapped_celltypes:
            missing.append(celltype)

    return sorted(missing)


def analyze_celltype_frequency(results_dir: str) -> Dict[str, int]:
    """
    Analyze the frequency of each cell type in the dataset.

    Args:
        results_dir: Directory containing evaluation results

    Returns:
        Dictionary mapping cell types to their frequencies
    """
    celltype_counts = defaultdict(int)

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json') and 'predictions' in file:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for item in data:
                        # Ground truth only
                        gt = item.get('ground_truth', '')
                        if '|' in gt:
                            for celltype in gt.split('|'):
                                celltype = celltype.strip()
                                if celltype:
                                    celltype_counts[celltype] += 1
                        elif gt.strip():
                            celltype_counts[gt.strip()] += 1

                except Exception as e:
                    pass

    return dict(celltype_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing cell types in mapping"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="A013",
        choices=["A013", "D099"],
        help="Dataset identifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for missing cell types (optional)"
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("CHECKING FOR MISSING CELL TYPES IN MAPPING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Results Directory: {args.results_dir}")
    print()

    # Load all cell types from predictions
    print("[STEP 1] Scanning prediction files...")
    all_celltypes = load_all_celltypes_from_predictions(args.results_dir)
    print(f"[INFO] Found {len(all_celltypes)} unique cell type names")
    print()

    # Get mapped cell types
    print("[STEP 2] Loading mapping...")
    mapped_celltypes = get_mapped_celltypes(args.dataset)
    print(f"[INFO] Mapping covers {len(mapped_celltypes)} cell type variations")
    print()

    # Find missing cell types
    print("[STEP 3] Identifying missing cell types...")
    missing_celltypes = find_missing_celltypes(all_celltypes, mapped_celltypes)

    if not missing_celltypes:
        print("[SUCCESS] ✓ All cell types are covered by the mapping!")
    else:
        print(f"[WARNING] ✗ Found {len(missing_celltypes)} cell type(s) not in mapping:")
        print()

        # Analyze frequency
        celltype_freq = analyze_celltype_frequency(args.results_dir)

        print(f"{'Cell Type':<60} {'Frequency':>10}")
        print("-" * 80)
        for celltype in missing_celltypes:
            freq = celltype_freq.get(celltype, 0)
            print(f"{celltype:<60} {freq:>10}")

        # Generate suggested mapping entries
        print()
        print("[STEP 4] Suggested mapping entries to add:")
        print("-" * 80)
        print()

        for celltype in missing_celltypes:
            freq = celltype_freq.get(celltype, 0)
            # Try to standardize (will return as-is if not found)
            std_name = standardize_celltype(celltype, dataset=args.dataset)

            print(f"    {{")
            print(f"        \"raw_name\": \"{celltype}\",")
            print(f"        \"standardized_name\": \"{std_name}\",")
            print(f"        \"broad_category\": \"TODO\",  # Frequency: {freq}")
            print(f"        \"synonyms\": []")
            print(f"    }},")

        # Save to file if requested
        if args.output:
            output_data = {
                "dataset": args.dataset,
                "total_missing": len(missing_celltypes),
                "missing_celltypes": [
                    {
                        "name": celltype,
                        "frequency": celltype_freq.get(celltype, 0),
                        "suggested_standardized_name": standardize_celltype(celltype, dataset=args.dataset)
                    }
                    for celltype in missing_celltypes
                ]
            }

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print()
            print(f"[INFO] Missing cell types saved to: {args.output}")

    print()
    print(f"{'='*80}")
    print()


if __name__ == "__main__":
    main()
