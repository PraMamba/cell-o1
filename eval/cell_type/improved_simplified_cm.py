"""
Improved Simplified Confusion Matrix

This version addresses three key issues:
1. Cleans up format errors at the source
2. Removes meta-labels ([Format Error], [Other]) from visualization
3. Supports dynamic threshold based on GT label distribution

Usage:
    python improved_simplified_cm.py \
        --metrics_file /path/to/metrics.json \
        --output_dir /path/to/output \
        --strategy filter_freq_clean \
        --top_k 20 \
        --dataset A013
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from celltype_mapping import get_broad_category, standardize_celltype
from improved_answer_extraction import is_format_error, clean_answer


def filter_predictions_top_k(
    y_true: List[str],
    y_pred: List[str],
    top_k: int = 20,
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Keep only top-K most frequent cell types based on GT distribution.

    This strategy:
    - Always keeps all ground truth classes
    - Keeps top-K most frequent predictions (by count)
    - Discards format errors completely (not shown in matrix)
    - Discards rare predictions (not shown as separate category)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        top_k: Number of prediction types to keep
        dataset: Dataset identifier

    Returns:
        Tuple of (filtered_y_true, filtered_y_pred, stats)
    """
    # Get all GT classes
    gt_classes = set(y_true)

    # Count prediction frequencies (excluding format errors)
    valid_preds = []
    format_error_count = 0
    empty_count = 0

    for pred in y_pred:
        # Check if original prediction had format errors (for statistics)
        had_format_error = pred and is_format_error(pred)

        # Clean the prediction
        cleaned_pred = clean_answer(pred) if pred else ""

        # Skip empty predictions
        if not cleaned_pred or cleaned_pred in ['-', '']:
            empty_count += 1
            continue

        # Count format errors that were cleaned
        if had_format_error:
            format_error_count += 1

        # Skip predictions that are still invalid after cleaning
        if is_format_error(cleaned_pred):
            continue

        valid_preds.append(cleaned_pred)

    pred_counter = Counter(valid_preds)

    # Get top-K most frequent predictions
    top_k_preds = set([pred for pred, count in pred_counter.most_common(top_k)])

    # Always include all GT classes
    keep_classes = gt_classes | top_k_preds

    # Filter both y_true and y_pred
    filtered_y_true = []
    filtered_y_pred = []

    for gt, pred in zip(y_true, y_pred):
        # Clean prediction
        cleaned_pred = clean_answer(pred) if pred else ""

        # Skip if prediction is invalid (don't include in matrix at all)
        if not cleaned_pred or cleaned_pred in ['-', '']:
            continue
        if is_format_error(cleaned_pred):
            continue

        # Only include if prediction is in keep_classes
        if cleaned_pred in keep_classes:
            filtered_y_true.append(gt)
            filtered_y_pred.append(cleaned_pred)

    stats = {
        "total_samples": len(y_true),
        "valid_samples": len(filtered_y_true),
        "format_errors": format_error_count,
        "empty_predictions": empty_count,
        "original_unique_preds": len(set(valid_preds)),
        "kept_unique_preds": len(keep_classes),
        "top_k": top_k,
        "gt_classes": len(gt_classes)
    }

    return filtered_y_true, filtered_y_pred, stats


def filter_predictions_percentile(
    y_true: List[str],
    y_pred: List[str],
    percentile: float = 95.0,
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Keep predictions that cover top percentile% of samples.

    This automatically determines how many cell types to keep based on
    the distribution of predictions.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        percentile: Percentage of samples to cover (default: 95%)
        dataset: Dataset identifier

    Returns:
        Tuple of (filtered_y_true, filtered_y_pred, stats)
    """
    gt_classes = set(y_true)

    # Clean and count predictions
    valid_preds = []
    format_error_count = 0
    empty_count = 0

    for pred in y_pred:
        # Check if original prediction had format errors (for statistics)
        had_format_error = pred and is_format_error(pred)

        # Clean the prediction
        cleaned_pred = clean_answer(pred) if pred else ""

        # Skip empty predictions and dashes
        if not cleaned_pred or cleaned_pred in ['-', '']:
            empty_count += 1
            continue

        # Count format errors that were cleaned
        if had_format_error:
            format_error_count += 1

        # Skip predictions that are still invalid after cleaning
        if is_format_error(cleaned_pred):
            continue

        valid_preds.append(cleaned_pred)

    pred_counter = Counter(valid_preds)

    # Calculate cumulative coverage
    total_valid = sum(pred_counter.values())
    target_count = total_valid * (percentile / 100.0)

    # Sort by frequency and accumulate until we reach target
    sorted_preds = sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)
    cumsum = 0
    keep_preds = set()

    for pred, count in sorted_preds:
        keep_preds.add(pred)
        cumsum += count
        if cumsum >= target_count:
            break

    # Always include all GT classes
    keep_classes = gt_classes | keep_preds

    # Filter
    filtered_y_true = []
    filtered_y_pred = []

    for gt, pred in zip(y_true, y_pred):
        cleaned_pred = clean_answer(pred) if pred else ""

        # Skip invalid predictions
        if not cleaned_pred or cleaned_pred in ['-', '']:
            continue
        if is_format_error(cleaned_pred):
            continue

        if cleaned_pred in keep_classes:
            filtered_y_true.append(gt)
            filtered_y_pred.append(cleaned_pred)

    stats = {
        "total_samples": len(y_true),
        "valid_samples": len(filtered_y_true),
        "format_errors": format_error_count,
        "empty_predictions": empty_count,
        "percentile": percentile,
        "original_unique_preds": len(pred_counter),
        "kept_unique_preds": len(keep_classes),
        "coverage": len(filtered_y_true) / total_valid * 100
    }

    return filtered_y_true, filtered_y_pred, stats


def aggregate_to_broad_category_clean(
    y_true: List[str],
    y_pred: List[str],
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Aggregate to broad categories, excluding format errors.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset: Dataset identifier

    Returns:
        Tuple of (aggregated_y_true, aggregated_y_pred, stats)
    """
    aggregated_y_true = []
    aggregated_y_pred = []
    format_error_count = 0
    empty_count = 0
    unmapped_count = 0

    for gt, pred in zip(y_true, y_pred):
        # Map GT to broad category
        gt_category = get_broad_category(gt, dataset=dataset)
        if not gt_category:
            gt_category = "Other"

        # Check if original prediction had format errors (for statistics)
        had_format_error = pred and is_format_error(pred)

        # Clean and check prediction
        cleaned_pred = clean_answer(pred) if pred else ""

        # Skip empty predictions and dashes
        if not cleaned_pred or cleaned_pred in ['-', '']:
            empty_count += 1
            continue

        # Count format errors that were cleaned
        if had_format_error:
            format_error_count += 1

        # Skip predictions that are still invalid after cleaning
        if is_format_error(cleaned_pred):
            continue

        # Map prediction to broad category
        pred_category = get_broad_category(cleaned_pred, dataset=dataset)
        if not pred_category:
            pred_category = "Other"
            unmapped_count += 1

        aggregated_y_true.append(gt_category)
        aggregated_y_pred.append(pred_category)

    stats = {
        "original_samples": len(y_true),
        "valid_samples": len(aggregated_y_true),
        "format_errors": format_error_count,
        "empty_predictions": empty_count,
        "unmapped_predictions": unmapped_count,
        "unique_categories": len(set(aggregated_y_true + aggregated_y_pred))
    }

    return aggregated_y_true, aggregated_y_pred, stats


def create_clean_confusion_matrix(
    metrics_file: str,
    output_dir: str,
    strategy: str = "filter_freq_clean",
    top_k: int = 20,
    percentile: float = 95.0,
    dataset: str = "A013"
):
    """
    Create clean confusion matrix without meta-labels.

    Args:
        metrics_file: Path to comprehensive metrics JSON
        output_dir: Output directory
        strategy: Simplification strategy
        top_k: Number of top predictions to keep
        percentile: Percentile for coverage
        dataset: Dataset identifier
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    variant = metrics.get('task_variant', 'unknown')

    # Reconstruct flat lists from confusion matrix
    confusion_matrix = np.array(metrics['confusion_matrix'])
    class_labels = metrics['class_labels']

    y_true_flat = []
    y_pred_flat = []
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            count = int(confusion_matrix[i][j])
            y_true_flat.extend([true_label] * count)
            y_pred_flat.extend([pred_label] * count)

    print(f"\n[INFO] Processing {variant}")
    print(f"[INFO] Original: {len(class_labels)} classes, {len(y_true_flat)} samples")
    print(f"[INFO] Strategy: {strategy}")

    # Apply strategy
    if strategy == "filter_freq_clean" or strategy == "top_k":
        y_true_clean, y_pred_clean, stats = filter_predictions_top_k(
            y_true_flat, y_pred_flat, top_k=top_k, dataset=dataset
        )
        title_suffix = f"(Top-{top_k})"

    elif strategy == "percentile":
        y_true_clean, y_pred_clean, stats = filter_predictions_percentile(
            y_true_flat, y_pred_flat, percentile=percentile, dataset=dataset
        )
        title_suffix = f"({percentile}% Coverage)"

    elif strategy == "broad_category_clean":
        y_true_clean, y_pred_clean, stats = aggregate_to_broad_category_clean(
            y_true_flat, y_pred_flat, dataset=dataset
        )
        title_suffix = "(Broad Categories)"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Print statistics
    print(f"\n[INFO] Cleaning statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get unique labels
    unique_labels = sorted(list(set(y_true_clean + y_pred_clean)))
    print(f"\n[INFO] Final: {len(unique_labels)} classes, {len(y_true_clean)} samples")

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm = sklearn_cm(y_true_clean, y_pred_clean, labels=unique_labels)

    # Remove empty rows and columns (where sum == 0)
    # This happens when a GT class has no samples predicted to any kept class
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)

    # Find non-empty rows and columns
    non_empty_rows = row_sums > 0
    non_empty_cols = col_sums > 0

    # Count how many will be removed
    removed_rows = (~non_empty_rows).sum()
    removed_cols = (~non_empty_cols).sum()

    if removed_rows > 0 or removed_cols > 0:
        print(f"\n[INFO] Removing empty rows/cols:")
        print(f"  Empty rows (GT classes with no predictions in kept set): {removed_rows}")
        print(f"  Empty cols (Predictions with no samples): {removed_cols}")

        if removed_rows > 0:
            removed_labels = [unique_labels[i] for i in range(len(unique_labels)) if not non_empty_rows[i]]
            print(f"  Removed GT classes: {removed_labels[:5]}{'...' if len(removed_labels) > 5 else ''}")

    # Filter confusion matrix and labels
    cm = cm[non_empty_rows][:, non_empty_cols]

    # Update labels separately for rows (GT) and columns (predictions)
    unique_labels_rows = [label for i, label in enumerate(unique_labels) if non_empty_rows[i]]
    unique_labels_cols = [label for i, label in enumerate(unique_labels) if non_empty_cols[i]]

    print(f"[INFO] After removing empty rows/cols:")
    print(f"  Rows (GT classes): {len(unique_labels_rows)}")
    print(f"  Cols (Predictions): {len(unique_labels_cols)}")

    # For display purposes, use row labels (since they represent actual GT classes)
    unique_labels = unique_labels_rows

    # Determine figure size
    num_labels = len(unique_labels)
    if num_labels <= 15:
        figsize = (14, 12)
        fontsize = 10
        annot = True
    elif num_labels <= 30:
        figsize = (18, 16)
        fontsize = 12
        annot = True
    else:
        figsize = (22, 20)
        fontsize = 12
        annot = True  # Too many labels, skip annotations

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annot,
        fmt='d',
        cmap='Reds',
        xticklabels=unique_labels_cols,
        yticklabels=unique_labels_rows,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='lightgray'
    )

    title = f"Confusion Matrix - {variant.replace('_', ' ').title()} ({dataset})\n{title_suffix}"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()

    # Save
    strategy_name = strategy.replace('_clean', '')
    output_file = os.path.join(output_dir, f"{variant}_clean_confusion_matrix_{strategy_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Saved: {output_file}")
    plt.close()

    # Save stats
    stats_file = os.path.join(output_dir, f"{variant}_clean_stats_{strategy_name}.json")
    stats_output = {
        "variant": variant,
        "strategy": strategy,
        "statistics": stats,
        "final_classes_rows": len(unique_labels_rows),
        "final_classes_cols": len(unique_labels_cols),
        "final_samples": len(y_true_clean),
        "removed_empty_rows": int(removed_rows),
        "removed_empty_cols": int(removed_cols),
        "unique_labels_rows": unique_labels_rows,
        "unique_labels_cols": unique_labels_cols
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create clean simplified confusion matrices"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        required=True,
        help="Path to comprehensive metrics JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="top_k",
        choices=["top_k", "percentile", "broad_category_clean", "all"],
        help="Simplification strategy"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top predictions to keep (for top_k strategy)"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Coverage percentile (for percentile strategy)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="A013",
        choices=["A013", "D099"],
        help="Dataset identifier"
    )

    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics_file}")

    if args.strategy == "all":
        # Run all strategies
        for strategy in ["top_k", "percentile", "broad_category_clean"]:
            print(f"\n{'='*80}")
            print(f"Running strategy: {strategy}")
            print(f"{'='*80}")
            create_clean_confusion_matrix(
                args.metrics_file,
                args.output_dir,
                strategy=strategy,
                top_k=args.top_k,
                percentile=args.percentile,
                dataset=args.dataset
            )
    else:
        create_clean_confusion_matrix(
            args.metrics_file,
            args.output_dir,
            strategy=args.strategy,
            top_k=args.top_k,
            percentile=args.percentile,
            dataset=args.dataset
        )

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
