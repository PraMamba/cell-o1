"""
Confusion Matrix Heatmap Visualization

This script creates confusion matrix heatmaps for evaluation results.
It uses standardized cell type names for accurate comparison.

Usage:
    python confusion_matrix_heatmap.py \
        --metrics_file /path/to/comprehensive_metrics.json \
        --output_dir /path/to/output \
        --variant batch_constrained
"""

import os
import json
import argparse
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    title: str = "Confusion Matrix",
    output_file: str = None,
    figsize: tuple = (12, 10),
    normalize: bool = False
):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        confusion_matrix: Confusion matrix array
        class_labels: List of class labels
        title: Plot title
        output_file: Path to save the figure
        figsize: Figure size (width, height)
        normalize: Whether to normalize the confusion matrix
    """
    cm = np.array(confusion_matrix)

    if normalize:
        # Normalize by row (true labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_norm = cm
        fmt = 'd'
        vmax = None

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=fmt,
        cmap='Reds',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        vmin=0,
        vmax=vmax
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved confusion matrix to: {output_file}")

    plt.close()


def plot_per_class_metrics(
    per_class_metrics: Dict[str, Dict],
    title: str = "Per-Class Metrics",
    output_file: str = None,
    figsize: tuple = (14, 8)
):
    """
    Plot per-class precision, recall, and F1 scores.

    Args:
        per_class_metrics: Dictionary of per-class metrics
        title: Plot title
        output_file: Path to save the figure
        figsize: Figure size
    """
    # Extract data
    classes = list(per_class_metrics.keys())
    precision = [per_class_metrics[c]['precision'] for c in classes]
    recall = [per_class_metrics[c]['recall'] for c in classes]
    f1 = [per_class_metrics[c]['f1'] for c in classes]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(classes))
    width = 0.25

    # Plot bars
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1', alpha=0.8)

    # Customize plot
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved per-class metrics to: {output_file}")

    plt.close()


def create_all_visualizations(
    metrics_file: str,
    output_dir: str,
    variant: str = None
):
    """
    Create all visualizations for a metrics file.

    Args:
        metrics_file: Path to comprehensive metrics JSON
        output_dir: Output directory for visualizations
        variant: Task variant name (for file naming)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load metrics
    print(f"[INFO] Loading metrics from: {metrics_file}")
    metrics = load_metrics(metrics_file)

    if variant is None:
        variant = metrics.get('task_variant', 'unknown')

    # Get dataset name
    dataset = metrics.get('dataset', 'Unknown')

    # Extract data
    confusion_matrix = metrics.get('confusion_matrix', [])
    class_labels = metrics.get('class_labels', [])
    per_class_metrics = metrics.get('per_class_metrics', {})

    if not confusion_matrix or not class_labels:
        print("[ERROR] Confusion matrix or class labels not found in metrics file")
        return

    # Create confusion matrix (raw counts)
    cm_file = os.path.join(output_dir, f"{variant}_confusion_matrix.png")
    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_labels=class_labels,
        title=f"Confusion Matrix - {variant.replace('_', ' ').title()} ({dataset})",
        output_file=cm_file,
        normalize=False
    )

    # Create per-class metrics plot
    if per_class_metrics:
        metrics_file_out = os.path.join(output_dir, f"{variant}_per_class_metrics.png")
        plot_per_class_metrics(
            per_class_metrics=per_class_metrics,
            title=f"Per-Class Metrics - {variant.replace('_', ' ').title()} ({dataset})",
            output_file=metrics_file_out
        )

    print(f"[SUCCESS] All visualizations created for {variant}")


def process_all_variants(metrics_dir: str, output_dir: str):
    """
    Process all task variants in a directory.

    Args:
        metrics_dir: Directory containing comprehensive metrics files
        output_dir: Output directory for visualizations
    """
    task_variants = [
        "batch_constrained",
        "batch_openended",
        "singlecell_constrained",
        "singlecell_openended"
    ]

    for variant in task_variants:
        metrics_file = os.path.join(metrics_dir, f"{variant}_comprehensive_metrics.json")
        if os.path.exists(metrics_file):
            print(f"\n[INFO] Processing {variant}...")
            create_all_visualizations(
                metrics_file=metrics_file,
                output_dir=output_dir,
                variant=variant
            )
        else:
            print(f"[WARNING] Metrics file not found: {metrics_file}")


def plot_metrics_comparison(
    summary_file: str,
    output_dir: str,
    figsize: tuple = (14, 8)
):
    """
    Plot a comparison of metrics across all task variants.

    Args:
        summary_file: Path to metrics_summary.json
        output_dir: Output directory
        figsize: Figure size
    """
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Get dataset name from summary
    dataset = summary.get('dataset', 'Unknown')

    variants = list(summary['variants'].keys())
    metric_names = [
        'accuracy',
        'macro_precision',
        'macro_recall',
        'macro_f1',
        'weighted_precision',
        'weighted_recall',
        'weighted_f1'
    ]

    # Prepare data
    data = {metric: [] for metric in metric_names}
    for variant in variants:
        for metric in metric_names:
            data[metric].append(summary['variants'][variant].get(metric, 0))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Plot groups of metrics
    metric_groups = [
        (['accuracy'], 'Accuracy'),
        (['macro_precision', 'macro_recall', 'macro_f1'], 'Macro Metrics'),
        (['weighted_precision', 'weighted_recall', 'weighted_f1'], 'Weighted Metrics'),
    ]

    x = np.arange(len(variants))
    width = 0.25

    for idx, (metrics_group, group_title) in enumerate(metric_groups):
        ax = axes[idx]

        for i, metric in enumerate(metrics_group):
            offset = (i - len(metrics_group)/2 + 0.5) * width
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title(), alpha=0.8)

        ax.set_xlabel('Task Variant', fontsize=10)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(group_title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)

    # Remove unused subplot
    fig.delaxes(axes[3])

    plt.suptitle(f'Metrics Comparison Across Task Variants ({dataset})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved metrics comparison to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create confusion matrix heatmaps and visualizations"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default=None,
        help="Path to comprehensive metrics JSON file (for single variant)"
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=None,
        help="Directory containing all comprehensive metrics files (for all variants)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Task variant name (auto-detected if not provided)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.metrics_file:
        # Process single metrics file
        if not os.path.exists(args.metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {args.metrics_file}")

        create_all_visualizations(
            metrics_file=args.metrics_file,
            output_dir=args.output_dir,
            variant=args.variant
        )

    elif args.metrics_dir:
        # Process all variants
        if not os.path.exists(args.metrics_dir):
            raise FileNotFoundError(f"Metrics directory not found: {args.metrics_dir}")

        process_all_variants(
            metrics_dir=args.metrics_dir,
            output_dir=args.output_dir
        )

        # Create comparison plot if summary exists
        summary_file = os.path.join(args.metrics_dir, 'metrics_summary.json')
        if os.path.exists(summary_file):
            plot_metrics_comparison(
                summary_file=summary_file,
                output_dir=args.output_dir
            )

    else:
        raise ValueError("Either --metrics_file or --metrics_dir must be provided")

    print("\n[SUCCESS] All visualizations created!")


if __name__ == "__main__":
    main()
