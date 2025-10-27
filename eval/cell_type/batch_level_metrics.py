"""
Batch-level and Cell-level Accuracy Calculation

This module implements the two key accuracy metrics for batch-level cell type annotation:

1. Cell-level Accuracy (CellAcc): Average proportion of correctly predicted cells within batches
   Formula: CellAcc = (1/|D|) * Σ(1/N_j * Σ 1(ŷ_i^(j) = y_i^(j)))

2. Batch-level Accuracy (BatchAcc): Proportion of batches where all cells are correctly predicted
   Formula: BatchAcc = (1/|D|) * Σ Π 1(ŷ_i^(j) = y_i^(j)))

Usage:
    from batch_level_metrics import calculate_batch_metrics

    metrics = calculate_batch_metrics(predictions, ground_truths, task_variant="batch")
"""

from typing import List, Dict, Tuple
import re


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    answer = re.sub(r'\s+', ' ', answer.strip())
    answer = re.sub(r'\s*\|\s*', ' | ', answer)
    return answer


def calculate_cell_level_accuracy_batch(predictions: List[str], ground_truths: List[str]) -> Tuple[float, Dict]:
    """
    Calculate Cell-level Accuracy for batch task variants.

    Formula: CellAcc = (1/|D|) * Σ_{j=1}^{|D|} (1/N_j * Σ_{i=1}^{N_j} 1(ŷ_i^(j) = y_i^(j)))

    This calculates the average accuracy per batch, where each batch's accuracy is the
    proportion of correctly predicted cells in that batch.

    Args:
        predictions: List of batch predictions (pipe-separated cell types)
        ground_truths: List of batch ground truths (pipe-separated cell types)

    Returns:
        Tuple of (cell_level_accuracy, detailed_stats)
    """
    num_batches = len(predictions)
    batch_accuracies = []
    total_cells_correct = 0
    total_cells = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        pred_types = [t.strip() for t in pred_norm.split('|') if t.strip()]
        gt_types = [t.strip() for t in gt_norm.split('|') if t.strip()]

        # Handle length mismatch
        if len(pred_types) != len(gt_types):
            # Batch accuracy is 0 if length doesn't match
            batch_accuracies.append(0.0)
            total_cells += len(gt_types)
            continue

        # Calculate accuracy for this batch
        correct_in_batch = 0
        for p, g in zip(pred_types, gt_types):
            if p.lower() == g.lower():
                correct_in_batch += 1
                total_cells_correct += 1
            total_cells += 1

        batch_accuracy = correct_in_batch / len(gt_types) if len(gt_types) > 0 else 0.0
        batch_accuracies.append(batch_accuracy)

    # Average over batches (formula: 1/|D| * Σ batch_accuracy)
    cell_level_accuracy = sum(batch_accuracies) / num_batches if num_batches > 0 else 0.0

    stats = {
        "num_batches": num_batches,
        "total_cells": total_cells,
        "total_cells_correct": total_cells_correct,
        "batch_accuracies": batch_accuracies,
        "min_batch_accuracy": min(batch_accuracies) if batch_accuracies else 0.0,
        "max_batch_accuracy": max(batch_accuracies) if batch_accuracies else 0.0
    }

    return cell_level_accuracy, stats


def calculate_batch_level_accuracy(predictions: List[str], ground_truths: List[str]) -> Tuple[float, Dict]:
    """
    Calculate Batch-level Accuracy for batch task variants.

    Formula: BatchAcc = (1/|D|) * Σ_{j=1}^{|D|} Π_{i=1}^{N_j} 1(ŷ_i^(j) = y_i^(j))

    This calculates the proportion of batches where ALL cells are correctly predicted.
    A batch is only considered correct if every single cell type in it is correct.

    Args:
        predictions: List of batch predictions (pipe-separated cell types)
        ground_truths: List of batch ground truths (pipe-separated cell types)

    Returns:
        Tuple of (batch_level_accuracy, detailed_stats)
    """
    num_batches = len(predictions)
    exact_match_count = 0
    batch_correct_flags = []

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        pred_types = [t.strip() for t in pred_norm.split('|') if t.strip()]
        gt_types = [t.strip() for t in gt_norm.split('|') if t.strip()]

        # Check if length matches
        if len(pred_types) != len(gt_types):
            batch_correct_flags.append(False)
            continue

        # Check if all cells are correct (product of indicators)
        all_correct = all(
            p.lower() == g.lower()
            for p, g in zip(pred_types, gt_types)
        )

        if all_correct:
            exact_match_count += 1
            batch_correct_flags.append(True)
        else:
            batch_correct_flags.append(False)

    batch_level_accuracy = exact_match_count / num_batches if num_batches > 0 else 0.0

    stats = {
        "num_batches": num_batches,
        "exact_match_count": exact_match_count,
        "batch_correct_flags": batch_correct_flags
    }

    return batch_level_accuracy, stats


def calculate_cell_level_accuracy_singlecell(predictions: List[str], ground_truths: List[str]) -> Tuple[float, Dict]:
    """
    Calculate Cell-level Accuracy for single-cell task variants.

    For single-cell tasks, each sample is a single cell, so cell-level accuracy
    equals the proportion of correctly predicted cells.

    Args:
        predictions: List of single-cell predictions
        ground_truths: List of single-cell ground truths

    Returns:
        Tuple of (cell_level_accuracy, detailed_stats)
    """
    num_cells = len(predictions)
    correct_count = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        if pred_norm.lower() == gt_norm.lower():
            correct_count += 1

    cell_level_accuracy = correct_count / num_cells if num_cells > 0 else 0.0

    stats = {
        "num_cells": num_cells,
        "correct_count": correct_count
    }

    return cell_level_accuracy, stats


def calculate_batch_metrics(
    predictions: List[str],
    ground_truths: List[str],
    task_variant: str = "batch"
) -> Dict:
    """
    Calculate comprehensive batch-level metrics.

    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        task_variant: "batch" or "singlecell"

    Returns:
        Dictionary containing cell-level and batch-level accuracy metrics
    """
    if task_variant == "batch" or task_variant.startswith("batch_"):
        # Batch task variants
        cell_acc, cell_stats = calculate_cell_level_accuracy_batch(predictions, ground_truths)
        batch_acc, batch_stats = calculate_batch_level_accuracy(predictions, ground_truths)

        metrics = {
            "task_variant": task_variant,
            "cell_level_accuracy": cell_acc,
            "batch_level_accuracy": batch_acc,
            "num_batches": cell_stats["num_batches"],
            "total_cells": cell_stats["total_cells"],
            "total_cells_correct": cell_stats["total_cells_correct"],
            "exact_match_batches": batch_stats["exact_match_count"],
            # Additional statistics
            "min_batch_cell_accuracy": cell_stats["min_batch_accuracy"],
            "max_batch_cell_accuracy": cell_stats["max_batch_accuracy"],
            # Legacy fields for compatibility
            "exact_match_count": batch_stats["exact_match_count"],
            "exact_match_accuracy": batch_acc,
            "cell_level_correct": cell_stats["total_cells_correct"],
            "cell_level_total": cell_stats["total_cells"]
        }

    else:
        # Single-cell task variants
        cell_acc, cell_stats = calculate_cell_level_accuracy_singlecell(predictions, ground_truths)

        # For single-cell tasks, batch-level accuracy equals cell-level accuracy
        # since each "batch" contains only one cell
        metrics = {
            "task_variant": task_variant,
            "cell_level_accuracy": cell_acc,
            "batch_level_accuracy": cell_acc,  # Same as cell-level for single-cell
            "num_samples": cell_stats["num_cells"],
            "total_cells_correct": cell_stats["correct_count"],
            "total_cells": cell_stats["num_cells"],
            # Legacy fields for compatibility
            "exact_match_count": cell_stats["correct_count"],
            "exact_match_accuracy": cell_acc,
            "cell_level_correct": cell_stats["correct_count"],
            "cell_level_total": cell_stats["num_cells"]
        }

    return metrics


def print_batch_metrics(metrics: Dict, title: str = "EVALUATION METRICS"):
    """
    Print batch-level metrics in a formatted way.

    Args:
        metrics: Dictionary containing metrics
        title: Title for the metrics display
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(f"Task Variant:            {metrics.get('task_variant', 'Unknown')}")
    print()

    if "batch" in metrics.get('task_variant', ''):
        print(f"Number of Batches:       {metrics.get('num_batches', 0)}")
        print(f"Total Cells:             {metrics.get('total_cells', 0)}")
        print()
        print(f"Cell-level Accuracy:     {metrics.get('cell_level_accuracy', 0):.2%}")
        print(f"  (Avg accuracy per batch)")
        print(f"  - Cells Correct:       {metrics.get('total_cells_correct', 0)}/{metrics.get('total_cells', 0)}")
        print(f"  - Min Batch Acc:       {metrics.get('min_batch_cell_accuracy', 0):.2%}")
        print(f"  - Max Batch Acc:       {metrics.get('max_batch_cell_accuracy', 0):.2%}")
        print()
        print(f"Batch-level Accuracy:    {metrics.get('batch_level_accuracy', 0):.2%}")
        print(f"  (All cells correct in batch)")
        print(f"  - Exact Match Batches: {metrics.get('exact_match_batches', 0)}/{metrics.get('num_batches', 0)}")
    else:
        print(f"Number of Samples:       {metrics.get('num_samples', 0)}")
        print()
        print(f"Cell-level Accuracy:     {metrics.get('cell_level_accuracy', 0):.2%}")
        print(f"  - Cells Correct:       {metrics.get('total_cells_correct', 0)}/{metrics.get('total_cells', 0)}")
        print()
        print(f"Note: For single-cell tasks, batch-level accuracy equals cell-level accuracy")

    print("="*70)


if __name__ == "__main__":
    # Example usage
    print("Example 1: Batch task variant")
    batch_preds = [
        "CD4+ T cell | CD8+ T cell | NK cell",
        "B cell | Monocyte",
        "CD4+ T cell | NK cell"
    ]
    batch_gts = [
        "CD4+ T cell | CD8+ T cell | NK cell",
        "B cell | Monocyte",
        "CD4+ T cell | CD8+ T cell"  # Last cell is wrong
    ]

    batch_metrics = calculate_batch_metrics(batch_preds, batch_gts, task_variant="batch_constrained")
    print_batch_metrics(batch_metrics, "BATCH TASK VARIANT EXAMPLE")

    print("\n" + "="*70)
    print()

    print("Example 2: Single-cell task variant")
    sc_preds = ["CD4+ T cell", "NK cell", "B cell", "Monocyte"]
    sc_gts = ["CD4+ T cell", "CD8+ T cell", "B cell", "Monocyte"]  # 2nd is wrong

    sc_metrics = calculate_batch_metrics(sc_preds, sc_gts, task_variant="singlecell_constrained")
    print_batch_metrics(sc_metrics, "SINGLE-CELL TASK VARIANT EXAMPLE")
