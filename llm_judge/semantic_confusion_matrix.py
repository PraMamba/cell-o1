#!/usr/bin/env python3
"""
Semantic Confusion Matrix Heatmap
==================================

Creates a confusion matrix heatmap where predictions are "semantically corrected"
based on LLM Judge scores.

The key idea:
- If score >= threshold: Prediction is semantically correct → replace with ground truth label
- If score < threshold: Prediction is semantically wrong → keep original prediction

This creates a confusion matrix showing:
- Diagonal: Semantically correct predictions (mapped to GT)
- Off-diagonal: Semantically wrong predictions (original errors)

Usage:
    python semantic_confusion_matrix.py \
        --judged_results_path /path/to/celltype_judged_results.json \
        --output_path /path/to/output \
        --threshold 0.7 \
        --dataset A013
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_judged_results(results_path: str) -> List[Dict]:
    """Load LLM judged results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} judged results from {results_path}")
    return data


def semantically_correct_predictions(
    judged_results: List[Dict],
    threshold: float = 0.7
) -> Tuple[List[str], List[str], Dict]:
    """
    Generate semantically corrected predictions based on LLM Judge scores.
    
    Args:
        judged_results: List of judged results, each containing:
            - ground_truth: True label
            - predicted_answer: Original prediction
            - llm_judgment.score: LLM Judge score (0.0-1.0)
        threshold: Score threshold for considering prediction as semantically correct
    
    Returns:
        Tuple of (y_true, y_pred_corrected, stats)
    """
    y_true = []
    y_pred_raw = []
    y_pred_corrected = []
    judge_scores = []
    
    stats = {
        "total_samples": 0,
        "semantically_correct": 0,
        "semantically_wrong": 0,
        "missing_judgment": 0,
        "missing_gt": 0,
        "missing_pred": 0
    }
    
    for result in judged_results:
        # Extract data
        ground_truth = result.get("ground_truth", "")
        predicted_answer = result.get("predicted_answer", "")
        
        # Check for missing data
        if not ground_truth:
            stats["missing_gt"] += 1
            continue
        if not predicted_answer:
            stats["missing_pred"] += 1
            continue
        
        # Get LLM Judge score
        if "llm_judgment" not in result:
            stats["missing_judgment"] += 1
            continue
        
        score = result["llm_judgment"].get("score", 0.0)
        
        # Store original data
        y_true.append(ground_truth)
        y_pred_raw.append(predicted_answer)
        judge_scores.append(score)
        
        stats["total_samples"] += 1
        
        # Semantic correction logic
        if score >= threshold:
            # Semantically correct → replace with ground truth
            y_pred_corrected.append(ground_truth)
            stats["semantically_correct"] += 1
        else:
            # Semantically wrong → keep original prediction
            y_pred_corrected.append(predicted_answer)
            stats["semantically_wrong"] += 1
    
    logging.info(f"Semantic correction statistics:")
    logging.info(f"  Total samples: {stats['total_samples']}")
    logging.info(f"  Semantically correct (score >= {threshold}): {stats['semantically_correct']}")
    logging.info(f"  Semantically wrong (score < {threshold}): {stats['semantically_wrong']}")
    if stats['missing_judgment'] > 0:
        logging.warning(f"  Missing judgment: {stats['missing_judgment']}")
    if stats['missing_gt'] > 0:
        logging.warning(f"  Missing ground truth: {stats['missing_gt']}")
    if stats['missing_pred'] > 0:
        logging.warning(f"  Missing prediction: {stats['missing_pred']}")
    
    return y_true, y_pred_corrected, stats


def create_semantic_confusion_matrix(
    judged_results_path: str,
    output_dir: str,
    threshold: float = 0.7,
    dataset: str = "A013",
    top_k: int = None,
    normalize: bool = False,
    figsize: tuple = None
):
    """
    Create semantic confusion matrix heatmap.
    
    Args:
        judged_results_path: Path to judged results JSON file
        output_dir: Output directory for visualization
        threshold: Score threshold for semantic correctness (default: 0.7)
        dataset: Dataset identifier (for title)
        top_k: Only show top-K most frequent labels (None = show all)
        normalize: Whether to normalize confusion matrix (rows sum to 1)
        figsize: Figure size tuple (auto-determined if None)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load judged results
    logging.info(f"Loading judged results from: {judged_results_path}")
    judged_results = load_judged_results(judged_results_path)
    
    # Generate semantically corrected predictions
    y_true, y_pred_corrected, stats = semantically_correct_predictions(
        judged_results, threshold=threshold
    )
    
    if len(y_true) == 0:
        logging.error("No valid samples found!")
        return
    
    # Get all unique labels
    all_labels = sorted(list(set(y_true + y_pred_corrected)))
    logging.info(f"Total unique labels: {len(all_labels)}")
    
    # Optionally filter to top-K most frequent labels
    if top_k is not None and top_k > 0:
        # Count frequency (prioritize GT labels)
        label_counts = Counter(y_true + y_pred_corrected)
        top_labels = [label for label, count in label_counts.most_common(top_k)]
        
        # Always include all GT labels
        gt_labels = set(y_true)
        keep_labels = set(top_labels) | gt_labels
        
        # Filter data
        filtered_y_true = []
        filtered_y_pred_corrected = []
        for gt, pred in zip(y_true, y_pred_corrected):
            if gt in keep_labels and pred in keep_labels:
                filtered_y_true.append(gt)
                filtered_y_pred_corrected.append(pred)
        
        y_true = filtered_y_true
        y_pred_corrected = filtered_y_pred_corrected
        all_labels = sorted(list(keep_labels))
        
        logging.info(f"After filtering: {len(all_labels)} labels, {len(y_true)} samples")
    
    # Compute confusion matrix
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred_corrected,
        labels=all_labels
    )
    
    # Normalize if requested
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        cm_plot = cm_normalized
        fmt = '.2f'
        vmax = 1.0
        cbar_label = 'Normalized Count'
    else:
        cm_plot = cm
        fmt = 'd'
        vmax = None
        cbar_label = 'Count'
    
    # Determine figure size
    num_labels = len(all_labels)
    if figsize is None:
        if num_labels <= 15:
            figsize = (14, 12)
            fontsize = 10
        elif num_labels <= 30:
            figsize = (18, 16)
            fontsize = 9
        else:
            figsize = (22, 20)
            fontsize = 8
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap='Reds',
        xticklabels=all_labels,
        yticklabels=all_labels,
        cbar_kws={'label': cbar_label},
        linewidths=0.5,
        linecolor='lightgray',
        vmin=0,
        vmax=vmax
    )
    
    # Customize plot
    title = f"Semantic Confusion Matrix (Threshold={threshold})\n{dataset}"
    if top_k:
        title += f" | Top-{top_k} Labels"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth Label', fontsize=12)
    plt.xlabel('Semantically Corrected Prediction', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    
    # Save figure
    output_filename = f"semantic_confusion_matrix_threshold_{threshold}"
    if top_k:
        output_filename += f"_top{top_k}"
    if normalize:
        output_filename += "_normalized"
    output_file = os.path.join(output_dir, f"{output_filename}.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logging.info(f"Saved confusion matrix to: {output_file}")
    plt.close()
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"semantic_confusion_matrix_stats_threshold_{threshold}.json")
    stats_output = {
        "threshold": threshold,
        "dataset": dataset,
        "statistics": stats,
        "total_labels": len(all_labels),
        "total_samples": len(y_true),
        "top_k": top_k,
        "normalize": normalize,
        "labels": all_labels
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved statistics to: {stats_file}")
    
    # Print summary
    diagonal_sum = np.trace(cm)
    total_sum = cm.sum()
    accuracy = diagonal_sum / total_sum if total_sum > 0 else 0.0
    
    logging.info(f"\nConfusion Matrix Summary:")
    logging.info(f"  Total predictions: {total_sum}")
    logging.info(f"  Diagonal (correct): {diagonal_sum}")
    logging.info(f"  Semantic accuracy: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Create semantic confusion matrix heatmap from LLM Judge results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default threshold (0.7)
  python semantic_confusion_matrix.py \\
      --judged_results_path results.json \\
      --output_path /path/to/output
  
  # Custom threshold
  python semantic_confusion_matrix.py \\
      --judged_results_path results.json \\
      --output_path /path/to/output \\
      --threshold 0.75
  
  # Show only top-20 labels
  python semantic_confusion_matrix.py \\
      --judged_results_path results.json \\
      --output_path /path/to/output \\
      --threshold 0.7 \\
      --top_k 20
  
  # Normalized confusion matrix
  python semantic_confusion_matrix.py \\
      --judged_results_path results.json \\
      --output_path /path/to/output \\
      --threshold 0.7 \\
      --normalize
        """
    )
    
    parser.add_argument(
        "--judged_results_path",
        type=str,
        required=True,
        help="Path to LLM judged results JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for visualization"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Score threshold for semantic correctness (default: 0.7)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="A013",
        help="Dataset identifier (for title)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only show top-K most frequent labels (None = show all)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize confusion matrix (rows sum to 1)"
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size (width, height) in inches"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        logging.error(f"Invalid threshold: {args.threshold}. Must be between 0.0 and 1.0")
        return
    
    # Convert figsize tuple if provided
    figsize = tuple(args.figsize) if args.figsize else None
    
    logging.info("="*70)
    logging.info("SEMANTIC CONFUSION MATRIX HEATMAP")
    logging.info("="*70)
    logging.info(f"Judged results: {args.judged_results_path}")
    logging.info(f"Output directory: {args.output_path}")
    logging.info(f"Threshold: {args.threshold}")
    logging.info(f"Dataset: {args.dataset}")
    if args.top_k:
        logging.info(f"Top-K: {args.top_k}")
    logging.info(f"Normalize: {args.normalize}")
    
    create_semantic_confusion_matrix(
        judged_results_path=args.judged_results_path,
        output_dir=args.output_path,
        threshold=args.threshold,
        dataset=args.dataset,
        top_k=args.top_k,
        normalize=args.normalize,
        figsize=figsize
    )
    
    logging.info("\n" + "="*70)
    logging.info("COMPLETE")
    logging.info("="*70)


if __name__ == "__main__":
    main()

