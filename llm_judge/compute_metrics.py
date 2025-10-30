#!/usr/bin/env python3
"""
Compute Classification Metrics from LLM Judge Results
======================================================

Computes binary classification metrics (accuracy, precision, recall, F1)
from LLM judge results based on score threshold.
"""

import json
import argparse
import logging
import os
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MetricsComputer:
    """Compute binary classification metrics from LLM judge results."""
    
    def __init__(self, binary_threshold: float = 0.7):
        """
        Initialize MetricsComputer.
        
        Args:
            binary_threshold: Score threshold for binary classification (default: 0.7).
                             Predictions with score >= threshold are considered correct.
        """
        self.binary_threshold = binary_threshold
    
    def load_judged_results(self, results_path: str) -> List[Dict]:
        """Load judged results from JSON file."""
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} judged results from {results_path}")
        return data
    
    def extract_binary_labels(self, results: List[Dict], threshold: float = 0.7) -> tuple:
        """
        Extract binary labels (correct/wrong) based on score threshold.
        
        Returns:
            y_true: All 1s (since we're evaluating predictions against ground truth)
            y_pred: 1 if score >= threshold, 0 otherwise
        """
        y_true = []
        y_pred = []
        
        for result in results:
            if "llm_judgment" not in result:
                continue
            
            score = result["llm_judgment"].get("score", 0.0)
            
            # Ground truth is always "correct" (1)
            y_true.append(1)
            
            # Prediction is "correct" (1) if score >= threshold
            y_pred.append(1 if score >= threshold else 0)
        
        return np.array(y_true), np.array(y_pred)
    
    def compute_binary_metrics(self, results: List[Dict], threshold: float = 0.7) -> Dict:
        """
        Compute binary classification metrics (correct vs wrong).
        
        Args:
            results: Judged results
            threshold: Score threshold for considering prediction as correct
        
        Returns:
            Dictionary of metrics
        """
        y_true, y_pred = self.extract_binary_labels(results, threshold)
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # For binary classification
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Count predictions
        correct_count = np.sum(y_pred == 1)
        wrong_count = np.sum(y_pred == 0)
        total = len(y_pred)
        
        metrics = {
            "threshold": threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "correct_count": int(correct_count),
            "wrong_count": int(wrong_count),
            "correct_rate": float(correct_count / total) if total > 0 else 0.0,
            "total_samples": int(total)
        }
        
        return metrics
    
    def compute_all_metrics(self, results: List[Dict]) -> Dict:
        """
        Compute binary classification metrics for the specified threshold.
        
        Args:
            results: Judged results
        
        Returns:
            Dictionary containing computed metrics
        """
        all_metrics = {}
        
        # Compute metrics with specified threshold
        logging.info(f"Computing binary metrics with threshold={self.binary_threshold}...")
        binary_metrics = self.compute_binary_metrics(results, self.binary_threshold)
        all_metrics["binary_metrics"] = binary_metrics
        logging.info(f"  Threshold={self.binary_threshold}: "
                    f"accuracy={binary_metrics['accuracy']:.3f}, "
                    f"precision={binary_metrics['precision']:.3f}, "
                    f"recall={binary_metrics['recall']:.3f}, "
                    f"f1={binary_metrics['f1']:.3f}, "
                    f"correct_rate={binary_metrics['correct_rate']:.3f}")
        
        # Score distribution statistics
        logging.info("Computing score distribution statistics...")
        scores = [r["llm_judgment"]["score"] for r in results if "llm_judgment" in r]
        all_metrics["score_statistics"] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "q25": float(np.percentile(scores, 25)),
            "q75": float(np.percentile(scores, 75))
        }
        
        return all_metrics
    
    def print_metrics_summary(self, metrics: Dict):
        """Print a summary of computed metrics."""
        print("\n" + "="*70)
        print("CLASSIFICATION METRICS SUMMARY")
        print("="*70)
        
        # Binary metrics
        print("\n### Binary Classification Metrics (Correct vs Wrong)")
        print("-" * 70)
        metrics_dict = metrics["binary_metrics"]
        print(f"\nThreshold >= {metrics_dict['threshold']}:")
        print(f"  Accuracy:  {metrics_dict['accuracy']:.4f}")
        print(f"  Precision: {metrics_dict['precision']:.4f}")
        print(f"  Recall:    {metrics_dict['recall']:.4f}")
        print(f"  F1 Score:  {metrics_dict['f1']:.4f}")
        print(f"  Correct:   {metrics_dict['correct_count']}/{metrics_dict['total_samples']} "
              f"({metrics_dict['correct_rate']:.2%})")
        
        # Score statistics
        print("\n### Score Distribution Statistics")
        print("-" * 70)
        score_stats = metrics["score_statistics"]
        print(f"  Mean:   {score_stats['mean']:.4f}")
        print(f"  Std:    {score_stats['std']:.4f}")
        print(f"  Min:    {score_stats['min']:.4f}")
        print(f"  Max:    {score_stats['max']:.4f}")
        print(f"  Median: {score_stats['median']:.4f}")
        print(f"  Q25:    {score_stats['q25']:.4f}")
        print(f"  Q75:    {score_stats['q75']:.4f}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Compute binary classification metrics from LLM judge results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default threshold (0.7), output to directory (auto-named)
  python compute_metrics.py --judged_results_path results.json --output_path /path/to/output/
  
  # Use custom threshold (0.75), output to directory
  python compute_metrics.py --judged_results_path results.json --output_path /path/to/output/ --binary_threshold 0.75
  
  # Use custom threshold, specify output file path
  python compute_metrics.py --judged_results_path results.json --output_path metrics.json --binary_threshold 0.75
        """
    )
    parser.add_argument("--judged_results_path", type=str, required=True,
                       help="Path to judged results JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save computed metrics (directory or file path). "
                            "If directory, file will be named classification_metrics_{threshold}.json")
    parser.add_argument("--binary_threshold", type=float, default=0.7,
                       help="Score threshold for binary classification (default: 0.7). "
                            "Predictions with score >= threshold are considered correct.")
    
    args = parser.parse_args()
    
    # Validate threshold
    if not (0.0 <= args.binary_threshold <= 1.0):
        logging.error(f"Invalid threshold: {args.binary_threshold}. Must be between 0.0 and 1.0")
        return
    
    # Determine output file path
    if os.path.isdir(args.output_path) or args.output_path.endswith('/'):
        # If output_path is a directory, generate filename with threshold
        output_dir = args.output_path.rstrip('/')
        threshold_str = str(args.binary_threshold)
        output_file = os.path.join(output_dir, f"classification_metrics-binary_threshold={threshold_str}.json")
    else:
        # Use provided path as-is
        output_file = args.output_path
        output_dir = os.path.dirname(output_file)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logging.info("="*70)
    logging.info("COMPUTING CLASSIFICATION METRICS")
    logging.info("="*70)
    logging.info(f"Binary threshold: {args.binary_threshold}")
    logging.info(f"Output file: {output_file}")
    
    # Initialize computer with custom threshold
    computer = MetricsComputer(binary_threshold=args.binary_threshold)
    
    # Load judged results
    results = computer.load_judged_results(args.judged_results_path)
    
    # Compute all metrics
    metrics = computer.compute_all_metrics(results)
    
    # Add threshold info to metrics
    metrics["config"] = {
        "binary_threshold": args.binary_threshold
    }
    
    # Save metrics
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\nMetrics saved to: {output_file}")
    
    # Print summary
    computer.print_metrics_summary(metrics)


if __name__ == "__main__":
    main()
