"""
Comprehensive Metrics Calculation with Cell Type Standardization

This script calculates comprehensive evaluation metrics including:
- Accuracy
- Macro Precision/Recall/F1
- Weighted Precision/Recall/F1
- Confusion Matrix

It applies cell type name standardization before metric calculation.

Usage:
    python comprehensive_metrics.py \
        --results_dir /path/to/eval_results \
        --dataset A013 \
        --output_dir /path/to/output
"""

import os
import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from celltype_mapping import standardize_celltype
from batch_level_metrics import calculate_batch_metrics as calc_batch_level_metrics


def load_predictions(pred_file: str) -> List[Dict]:
    """Load prediction results from JSON file."""
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def standardize_prediction_results(
    predictions: List[Dict],
    dataset: str = "A013",
    task_variant: str = "batch"
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    Standardize cell type names in prediction results.

    Args:
        predictions: List of prediction dictionaries
        dataset: Dataset identifier ("A013" or "D099")
        task_variant: "batch" or "singlecell"

    Returns:
        Tuple of (standardized_predictions, standardized_ground_truths, detailed_results)
    """
    std_predictions = []
    std_ground_truths = []
    detailed_results = []

    for idx, pred_item in enumerate(predictions):
        pred_answer = pred_item.get("predicted_answer", "")
        gt_answer = pred_item.get("ground_truth", "")

        # For batch variants, answers are separated by "|"
        if task_variant == "batch" or "|" in gt_answer:
            pred_types = [t.strip() for t in pred_answer.split("|") if t.strip()]
            gt_types = [t.strip() for t in gt_answer.split("|") if t.strip()]

            # Standardize each cell type
            batch_predictions = []
            batch_ground_truths = []
            batch_std_predictions = []
            batch_std_ground_truths = []

            for pred_type, gt_type in zip(pred_types, gt_types):
                std_pred = standardize_celltype(pred_type, dataset=dataset)
                std_gt = standardize_celltype(gt_type, dataset=dataset)
                std_predictions.append(std_pred)
                std_ground_truths.append(std_gt)

                batch_predictions.append(pred_type)
                batch_ground_truths.append(gt_type)
                batch_std_predictions.append(std_pred)
                batch_std_ground_truths.append(std_gt)

            detailed_results.append({
                "sample_index": idx,
                "type": "batch",
                "original_prediction": pred_types,
                "original_ground_truth": gt_types,
                "standardized_prediction": batch_std_predictions,
                "standardized_ground_truth": batch_std_ground_truths
            })

        # For single-cell variants, each answer is a single cell type
        else:
            std_pred = standardize_celltype(pred_answer, dataset=dataset)
            std_gt = standardize_celltype(gt_answer, dataset=dataset)
            std_predictions.append(std_pred)
            std_ground_truths.append(std_gt)

            detailed_results.append({
                "sample_index": idx,
                "type": "singlecell",
                "original_prediction": pred_answer,
                "original_ground_truth": gt_answer,
                "standardized_prediction": std_pred,
                "standardized_ground_truth": std_gt
            })

    return std_predictions, std_ground_truths, detailed_results


def calculate_comprehensive_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: List of ground truth labels (standardized)
        y_pred: List of predicted labels (standardized)

    Returns:
        Dictionary containing all metrics
    """
    # Get unique labels
    all_labels = sorted(list(set(y_true + y_pred)))

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, f1 for macro and weighted averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=all_labels, zero_division=0
    )

    # Build per-class metrics dictionary
    per_class_metrics = {}
    for i, label in enumerate(all_labels):
        per_class_metrics[label] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i])
        }

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=all_labels)

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "total_samples": len(y_true),
        "num_classes": len(all_labels),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "class_labels": all_labels
    }

    return metrics


def process_evaluation_results(
    results_dir: str,
    dataset: str = "A013",
    output_dir: str = None
) -> Dict[str, Dict]:
    """
    Process all evaluation results in the directory.

    Args:
        results_dir: Directory containing evaluation results
        dataset: Dataset identifier
        output_dir: Output directory for processed metrics

    Returns:
        Dictionary mapping task variants to their metrics
    """
    if output_dir is None:
        output_dir = results_dir

    os.makedirs(output_dir, exist_ok=True)

    task_variants = [
        "batch_constrained",
        "batch_openended",
        "singlecell_constrained",
        "singlecell_openended"
    ]

    all_metrics = {}

    for variant in task_variants:
        variant_dir = os.path.join(results_dir, variant)
        if not os.path.exists(variant_dir):
            print(f"[WARNING] Directory not found: {variant_dir}")
            continue

        # Find prediction file
        pred_files = [f for f in os.listdir(variant_dir) if f.startswith(f"{variant}_predictions_") and f.endswith(".json")]
        if not pred_files:
            print(f"[WARNING] No prediction file found in {variant_dir}")
            continue

        pred_file = os.path.join(variant_dir, pred_files[0])
        print(f"\n[INFO] Processing {variant}...")
        print(f"[INFO] Loading: {pred_file}")

        # Load predictions
        predictions = load_predictions(pred_file)

        # Determine task type (batch vs singlecell)
        task_type = "singlecell" if "singlecell" in variant else "batch"

        # Standardize cell types
        std_predictions, std_ground_truths, detailed_results = standardize_prediction_results(
            predictions, dataset=dataset, task_variant=task_type
        )

        print(f"[INFO] Total samples after standardization: {len(std_ground_truths)}")

        # Save standardized results
        standardized_output_file = os.path.join(output_dir, f"{variant}_standardized_results.json")
        with open(standardized_output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved standardized results to: {standardized_output_file}")

        # Calculate batch-level metrics (cell-level and batch-level accuracy)
        # For batch variants, we need to reconstruct batch-level predictions
        if task_type == "batch":
            # Reconstruct batch predictions from detailed_results
            batch_predictions = []
            batch_ground_truths = []
            for item in detailed_results:
                # Join standardized cell types back into pipe-separated format
                pred_str = " | ".join(item["standardized_prediction"])
                gt_str = " | ".join(item["standardized_ground_truth"])
                batch_predictions.append(pred_str)
                batch_ground_truths.append(gt_str)

            batch_metrics = calc_batch_level_metrics(
                batch_predictions,
                batch_ground_truths,
                task_variant=variant
            )
        else:
            # For single-cell variants, use flattened predictions
            batch_metrics = calc_batch_level_metrics(
                std_predictions,
                std_ground_truths,
                task_variant=variant
            )

        # Calculate sklearn metrics (confusion matrix, per-class metrics, etc.)
        metrics = calculate_comprehensive_metrics(std_ground_truths, std_predictions)
        metrics["task_variant"] = variant
        metrics["dataset"] = dataset

        # Add batch-level metrics to the main metrics dict
        metrics["cell_level_accuracy"] = batch_metrics["cell_level_accuracy"]
        metrics["batch_level_accuracy"] = batch_metrics["batch_level_accuracy"]
        if "num_batches" in batch_metrics:
            metrics["num_batches"] = batch_metrics["num_batches"]
            metrics["exact_match_batches"] = batch_metrics["exact_match_batches"]
        if "num_samples" in batch_metrics:
            metrics["num_samples"] = batch_metrics["num_samples"]

        # Print summary
        print(f"[INFO] Metrics for {variant}:")
        if "num_batches" in metrics:
            print(f"  Num Batches:         {metrics['num_batches']}")
            print(f"  Total Cells:         {metrics['total_samples']}")
        else:
            print(f"  Num Samples:         {metrics.get('num_samples', metrics['total_samples'])}")
        print(f"  Cell-level Accuracy: {metrics['cell_level_accuracy']:.4f}")
        print(f"  Batch-level Accuracy:{metrics['batch_level_accuracy']:.4f}")
        print(f"  Sklearn Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Macro Precision:     {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:        {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
        print(f"  Weighted Precision:  {metrics['weighted_precision']:.4f}")
        print(f"  Weighted Recall:     {metrics['weighted_recall']:.4f}")
        print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")

        # Save metrics
        metrics_output_file = os.path.join(output_dir, f"{variant}_comprehensive_metrics.json")
        with open(metrics_output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved metrics to: {metrics_output_file}")

        all_metrics[variant] = metrics

    # Create summary comparison
    if all_metrics:
        summary = create_summary_comparison(all_metrics)
        summary_file = os.path.join(output_dir, "metrics_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Summary saved to: {summary_file}")

        # Print comparison table
        print_comparison_table(summary)

    return all_metrics


def create_summary_comparison(all_metrics: Dict[str, Dict]) -> Dict:
    """Create a summary comparison of all task variants."""
    summary = {
        "dataset": all_metrics[list(all_metrics.keys())[0]].get("dataset", "Unknown"),
        "variants": {}
    }

    metric_names = [
        "cell_level_accuracy",
        "batch_level_accuracy",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1"
    ]

    for variant, metrics in all_metrics.items():
        summary["variants"][variant] = {
            metric: metrics.get(metric, 0.0)
            for metric in metric_names
        }
        summary["variants"][variant]["total_samples"] = metrics.get("total_samples", 0)
        summary["variants"][variant]["num_classes"] = metrics.get("num_classes", 0)
        if "num_batches" in metrics:
            summary["variants"][variant]["num_batches"] = metrics.get("num_batches", 0)
        if "num_samples" in metrics:
            summary["variants"][variant]["num_samples"] = metrics.get("num_samples", 0)

    return summary


def print_comparison_table(summary: Dict):
    """Print a formatted comparison table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*120)

    variants = list(summary["variants"].keys())
    metric_names = [
        "cell_level_accuracy",
        "batch_level_accuracy",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1"
    ]

    # Print header
    header = f"{'Metric':<30}"
    for variant in variants:
        header += f"{variant:<25}"
    print(header)
    print("-"*120)

    # Print each metric
    for metric in metric_names:
        row = f"{metric:<30}"
        for variant in variants:
            value = summary["variants"][variant].get(metric, 0.0)
            row += f"{value:<25.4f}"
        print(row)

    print("-"*120)

    # Print sample/batch counts
    row = f"{'Total Samples/Cells':<30}"
    for variant in variants:
        count = summary["variants"][variant].get("total_samples", 0)
        row += f"{count:<25}"
    print(row)

    row = f"{'Num Batches/Samples':<30}"
    for variant in variants:
        if "num_batches" in summary["variants"][variant]:
            count = summary["variants"][variant].get("num_batches", 0)
        else:
            count = summary["variants"][variant].get("num_samples", 0)
        row += f"{count:<25}"
    print(row)

    print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate comprehensive metrics with cell type standardization"
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
        help="Dataset identifier for cell type mapping"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed metrics (default: same as results_dir)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    print(f"\n[INFO] Processing evaluation results from: {args.results_dir}")
    print(f"[INFO] Dataset: {args.dataset}")

    all_metrics = process_evaluation_results(
        results_dir=args.results_dir,
        dataset=args.dataset,
        output_dir=args.output_dir
    )

    print(f"\n[SUCCESS] Processing complete!")


if __name__ == "__main__":
    main()
