"""
Simplified Confusion Matrix for Better Readability

This module provides functions to create more readable confusion matrices
by reducing the number of labels while minimizing information loss.

Strategies:
1. Filter low-frequency predictions
2. Aggregate to broad categories
3. Show only ground truth classes
4. Separate format errors

Usage:
    python simplified_confusion_matrix.py \
        --metrics_file /path/to/comprehensive_metrics.json \
        --output_dir /path/to/output \
        --strategy filter_freq \
        --threshold 5
"""

import os
import json
import argparse
from typing import Dict, List, Tuple, Set
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from celltype_mapping import get_broad_category


def is_format_error(prediction: str) -> bool:
    """
    Check if a prediction contains format errors.

    Args:
        prediction: Cell type prediction string

    Returns:
        True if contains format errors
    """
    error_patterns = [
        '</think>',
        '<think>',
        '</answer>',
        '<answer>',
        'Final answer:',
        'Cell 1',
        'Cell 2',
        'Cell 3',
        'Cell 4',
        'Cell 5'
    ]

    for pattern in error_patterns:
        if pattern in prediction:
            return True

    return False


def filter_low_frequency_predictions(
    y_true: List[str],
    y_pred: List[str],
    threshold: int = 5,
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Filter out low-frequency predictions and group them as 'Other'.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        threshold: Minimum frequency to keep a prediction
        dataset: Dataset identifier

    Returns:
        Tuple of (filtered_y_true, filtered_y_pred, stats)
    """
    # Count prediction frequencies
    pred_counter = Counter(y_pred)

    # Separate format errors
    format_errors = []
    valid_preds = []
    for pred in y_pred:
        if is_format_error(pred):
            format_errors.append(pred)
        else:
            valid_preds.append(pred)

    # Get frequent predictions
    frequent_preds = {pred for pred, count in pred_counter.items()
                     if count >= threshold and not is_format_error(pred)}

    # Always include all ground truth classes
    gt_classes = set(y_true)
    keep_classes = frequent_preds | gt_classes

    # Filter predictions
    filtered_y_pred = []
    for pred in y_pred:
        if is_format_error(pred):
            filtered_y_pred.append("[Format Error]")
        elif pred in keep_classes:
            filtered_y_pred.append(pred)
        else:
            filtered_y_pred.append("[Other Predictions]")

    stats = {
        "total_unique_predictions": len(pred_counter),
        "kept_predictions": len(keep_classes),
        "format_errors_count": len(format_errors),
        "other_predictions_count": sum(1 for p in filtered_y_pred if p == "[Other Predictions]"),
        "threshold": threshold
    }

    return y_true, filtered_y_pred, stats


def aggregate_to_broad_category(
    y_true: List[str],
    y_pred: List[str],
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Aggregate cell types to broad categories.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset: Dataset identifier

    Returns:
        Tuple of (aggregated_y_true, aggregated_y_pred, stats)
    """
    aggregated_y_true = []
    aggregated_y_pred = []

    for gt in y_true:
        category = get_broad_category(gt, dataset=dataset)
        if category:
            aggregated_y_true.append(category)
        else:
            aggregated_y_true.append("[Unknown]")

    for pred in y_pred:
        if is_format_error(pred):
            aggregated_y_pred.append("[Format Error]")
        else:
            category = get_broad_category(pred, dataset=dataset)
            if category:
                aggregated_y_pred.append(category)
            else:
                aggregated_y_pred.append("[Other]")

    stats = {
        "original_gt_classes": len(set(y_true)),
        "aggregated_gt_classes": len(set(aggregated_y_true)),
        "original_pred_classes": len(set(y_pred)),
        "aggregated_pred_classes": len(set(aggregated_y_pred))
    }

    return aggregated_y_true, aggregated_y_pred, stats


def filter_to_gt_classes_only(
    y_true: List[str],
    y_pred: List[str],
    dataset: str = "A013"
) -> Tuple[List[str], List[str], Dict]:
    """
    Show only ground truth classes on prediction axis.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        dataset: Dataset identifier

    Returns:
        Tuple of (y_true, filtered_y_pred, stats)
    """
    gt_classes = set(y_true)

    filtered_y_pred = []
    for pred in y_pred:
        if is_format_error(pred):
            filtered_y_pred.append("[Format Error]")
        elif pred in gt_classes:
            filtered_y_pred.append(pred)
        else:
            filtered_y_pred.append("[Other Predictions]")

    stats = {
        "gt_classes": len(gt_classes),
        "total_pred_classes": len(set(y_pred)),
        "other_count": sum(1 for p in filtered_y_pred if p == "[Other Predictions]"),
        "format_error_count": sum(1 for p in filtered_y_pred if p == "[Format Error]")
    }

    return y_true, filtered_y_pred, stats


def create_simplified_confusion_matrix(
    metrics_file: str,
    output_dir: str,
    strategy: str = "filter_freq",
    threshold: int = 5,
    dataset: str = "A013"
):
    """
    Create simplified confusion matrix with reduced labels.

    Args:
        metrics_file: Path to comprehensive metrics JSON
        output_dir: Output directory
        strategy: Simplification strategy
        threshold: Frequency threshold (for filter_freq strategy)
        dataset: Dataset identifier
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    variant = metrics.get('task_variant', 'unknown')
    y_true_flat = []
    y_pred_flat = []

    # Reconstruct predictions from confusion matrix
    confusion_matrix = np.array(metrics['confusion_matrix'])
    class_labels = metrics['class_labels']

    # Flatten confusion matrix back to lists
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            count = int(confusion_matrix[i][j])
            y_true_flat.extend([true_label] * count)
            y_pred_flat.extend([pred_label] * count)

    print(f"\n[INFO] Processing {variant}")
    print(f"[INFO] Original labels: {len(class_labels)} unique classes")
    print(f"[INFO] Total samples: {len(y_true_flat)}")
    print(f"[INFO] Strategy: {strategy}")

    # Apply simplification strategy
    if strategy == "filter_freq":
        y_true_simple, y_pred_simple, stats = filter_low_frequency_predictions(
            y_true_flat, y_pred_flat, threshold=threshold, dataset=dataset
        )
        title_suffix = f"(Freq ≥ {threshold})"

    elif strategy == "broad_category":
        y_true_simple, y_pred_simple, stats = aggregate_to_broad_category(
            y_true_flat, y_pred_flat, dataset=dataset
        )
        title_suffix = "(Broad Categories)"

    elif strategy == "gt_only":
        y_true_simple, y_pred_simple, stats = filter_to_gt_classes_only(
            y_true_flat, y_pred_flat, dataset=dataset
        )
        title_suffix = "(GT Classes Only)"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Print statistics
    print(f"\n[INFO] Simplification statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get unique labels after simplification
    unique_labels = sorted(list(set(y_true_simple + y_pred_simple)))
    print(f"\n[INFO] Simplified labels: {len(unique_labels)} unique classes")

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix as sklearn_cm
    cm = sklearn_cm(y_true_simple, y_pred_simple, labels=unique_labels)

    # Determine figure size based on number of labels
    num_labels = len(unique_labels)
    if num_labels <= 15:
        figsize = (12, 10)
        fontsize = 10
    elif num_labels <= 30:
        figsize = (16, 14)
        fontsize = 8
    else:
        figsize = (20, 18)
        fontsize = 6

    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar_kws={'label': 'Count'}
    )

    plt.title(f"Simplified Confusion Matrix - {variant.replace('_', ' ').title()} {title_suffix}",
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, f"{variant}_simplified_confusion_matrix_{strategy}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Saved simplified confusion matrix to: {output_file}")
    plt.close()

    # Save statistics
    stats_file = os.path.join(output_dir, f"{variant}_simplification_stats_{strategy}.json")
    stats_output = {
        "variant": variant,
        "strategy": strategy,
        "original_classes": len(class_labels),
        "simplified_classes": len(unique_labels),
        "total_samples": len(y_true_flat),
        "statistics": stats,
        "unique_labels": unique_labels
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved statistics to: {stats_file}")


def process_all_strategies(
    metrics_file: str,
    output_dir: str,
    dataset: str = "A013",
    threshold: int = 5
):
    """
    Process all simplification strategies for comparison.

    Args:
        metrics_file: Path to comprehensive metrics JSON
        output_dir: Output directory
        dataset: Dataset identifier
        threshold: Frequency threshold
    """
    strategies = [
        ("filter_freq", threshold),
        ("broad_category", None),
        ("gt_only", None)
    ]

    for strategy, thresh in strategies:
        print(f"\n{'='*80}")
        print(f"Processing strategy: {strategy}")
        print(f"{'='*80}")

        if strategy == "filter_freq" and thresh is not None:
            create_simplified_confusion_matrix(
                metrics_file, output_dir, strategy=strategy,
                threshold=thresh, dataset=dataset
            )
        else:
            create_simplified_confusion_matrix(
                metrics_file, output_dir, strategy=strategy, dataset=dataset
            )


def main():
    parser = argparse.ArgumentParser(
        description="Create simplified confusion matrices for better readability"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        required=True,
        help="Path to comprehensive metrics JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for simplified matrices"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["filter_freq", "broad_category", "gt_only", "all"],
        help="Simplification strategy"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Frequency threshold for filter_freq strategy"
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
        process_all_strategies(
            args.metrics_file,
            args.output_dir,
            dataset=args.dataset,
            threshold=args.threshold
        )
    else:
        create_simplified_confusion_matrix(
            args.metrics_file,
            args.output_dir,
            strategy=args.strategy,
            threshold=args.threshold,
            dataset=args.dataset
        )

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
