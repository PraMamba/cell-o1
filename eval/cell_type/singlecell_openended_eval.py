"""
Single-cell + Open-ended Evaluation Script

This variant tests:
- Each batch is decomposed into N independent single-cell tasks
- Each task shows only ONE cell's gene expression
- NO candidate list is provided (open-ended generation)
- Models must freely generate cell type names

This is the most challenging variant, lacking both batch context and candidate constraints.

Usage:
    python singlecell_openended_eval.py \
        --input_file /path/to/qa_pairs.json \
        --output_dir /path/to/save/results \
        --model_name ncbi/Cell-o1 \
        --max_new_tokens 1000
"""

import os
import json
import argparse
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from celltype_standardizer import CellTypeStandardizer, extract_dataset_id_from_path, save_unmapped_report


def load_qa_data(input_file: str) -> List[Dict]:
    """Load QA pairs from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} QA pairs from {input_file}")
    return data


def prepare_messages(qa_item: Dict, system_msg: str) -> List[Dict]:
    """Convert a single-cell QA item to chat-style messages."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": qa_item["question"]}
    ]
    return messages


def extract_answer_from_response(response_text: str) -> str:
    """Extract the final answer from model's response."""
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # For single-cell, there should be no '|' separator
        # If there is, take the first item
        if '|' in answer:
            answer = answer.split('|')[0].strip()
        return answer

    lines = response_text.strip().split('\n')
    for line in reversed(lines):
        if line.strip() and len(line) < 200:
            return line.strip()

    return ""


def calculate_metrics(predictions: List[str], ground_truths: List[str], decomposed_data: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics for single-cell setting.
    Also compute batch-level metrics by aggregating single-cell predictions.
    """
    total_cells = len(predictions)
    cell_level_correct = 0

    # Cell-level accuracy
    for pred, gt in zip(predictions, ground_truths):
        if pred.lower().strip() == gt.lower().strip():
            cell_level_correct += 1

    # Batch-level metrics: group predictions by group field
    batch_groups = {}
    for i, (pred, gt, item) in enumerate(zip(predictions, ground_truths, decomposed_data)):
        batch_idx = item.get("group", f"batch_{i}")  # Use group field if available
        if batch_idx not in batch_groups:
            batch_groups[batch_idx] = {
                "predictions": [],
                "ground_truths": []
            }
        batch_groups[batch_idx]["predictions"].append(pred)
        batch_groups[batch_idx]["ground_truths"].append(gt)

    # Calculate batch-level exact match
    batch_exact_match = 0
    total_batches = len(batch_groups)

    for batch_data in batch_groups.values():
        preds = batch_data["predictions"]
        gts = batch_data["ground_truths"]
        if all(p.lower().strip() == g.lower().strip() for p, g in zip(preds, gts)):
            batch_exact_match += 1

    metrics = {
        "task_variant": "singlecell_openended",
        "total_cells": total_cells,
        "total_batches": total_batches,
        "cell_level_correct": cell_level_correct,
        "cell_level_accuracy": cell_level_correct / total_cells if total_cells > 0 else 0.0,
        "batch_exact_match_count": batch_exact_match,
        "batch_exact_match_accuracy": batch_exact_match / total_batches if total_batches > 0 else 0.0
    }

    return metrics


def run_evaluation(
    input_file: str,
    output_dir: str,
    model_name: str = "ncbi/Cell-o1",
    max_new_tokens: int = 1000,
    batch_size: int = 8,
    device: str = "auto"
):
    """Run single-cell + open-ended evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    qa_data = load_qa_data(input_file)
    
    # Initialize cell type standardizer
    standardizer = CellTypeStandardizer()
    
    # Extract dataset ID from input file path
    dataset_id = extract_dataset_id_from_path(input_file)

    # Data should already be in single-cell format (processed by pipeline)
    all_singlecell_qas = qa_data
    print(f"[INFO] Loaded {len(all_singlecell_qas)} single-cell tasks (already processed by pipeline)")

    # System message for single-cell + open-ended setting
    system_msg = (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given information about a single cell, including the top expressed genes in descending order. "
        "Using the gene expression data and donor information, determine the correct cell type for this cell. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single cell type name."
    )

    print(f"\n[INFO] Task Variant: Single-cell + Open-ended")
    print(f"[INFO] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device
    )

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=device, batch_size=batch_size)

    print(f"[INFO] Starting evaluation on {len(all_singlecell_qas)} single-cell tasks with batch_size={batch_size}...")

    results = []
    predictions = []
    ground_truths = []
    unmapped_records = []  # Track unmapped cell types

    # Prepare all messages at once for batch processing
    all_messages = [prepare_messages(qa_item, system_msg) for qa_item in all_singlecell_qas]

    # Process in batches
    try:
        batch_responses = []
        for i in tqdm(range(0, len(all_messages), batch_size), desc="Evaluating"):
            batch_msgs = all_messages[i:i + batch_size]
            
            try:
                responses = generator(
                    batch_msgs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False
                )
                batch_responses.extend(responses)
            except Exception as e:
                print(f"\n[ERROR] Failed to process batch starting at index {i}: {e}")
                # Add error placeholders for this batch
                for _ in range(len(batch_msgs)):
                    batch_responses.append([{"generated_text": f"ERROR: {str(e)}"}])

        # Process results
        for idx, (qa_item, response) in enumerate(zip(all_singlecell_qas, batch_responses)):
            try:
                if isinstance(response, list) and len(response) > 0:
                    assistant_reply = response[0]["generated_text"]
                else:
                    assistant_reply = str(response)

                predicted_answer_raw = extract_answer_from_response(assistant_reply)
                ground_truth_raw = qa_item["answer"]
                
                # Standardize cell type names (single cell)
                ground_truth_std, gt_is_mapped = standardizer.standardize_single_celltype(ground_truth_raw)
                predicted_answer_std, pred_is_mapped = standardizer.standardize_single_celltype(predicted_answer_raw)
                
                # Track unmapped types
                if not gt_is_mapped and ground_truth_raw:
                    unmapped_records.append({
                        "index": idx,
                        "source": "ground_truth",
                        "original_type": ground_truth_raw,
                        "full_answer": ground_truth_raw
                    })
                if not pred_is_mapped and predicted_answer_raw:
                    unmapped_records.append({
                        "index": idx,
                        "source": "predicted_answer",
                        "original_type": predicted_answer_raw,
                        "full_answer": predicted_answer_raw
                    })

                result_item = {
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "index": idx,
                    "task_type": "cell type",
                    "task_variant": "singlecell_openended",
                    "question": qa_item["question"],
                    "ground_truth": ground_truth_std,
                    "predicted_answer": predicted_answer_std,
                    "full_response": assistant_reply,
                    "group": qa_item.get("group", "")
                }
                results.append(result_item)
                predictions.append(predicted_answer_std)
                ground_truths.append(ground_truth_std)

            except Exception as e:
                print(f"\n[ERROR] Failed to process result for sample {idx}: {e}")
                ground_truth_raw = qa_item["answer"]
                ground_truth_std, gt_is_mapped = standardizer.standardize_single_celltype(ground_truth_raw)
                
                if not gt_is_mapped and ground_truth_raw:
                    unmapped_records.append({
                        "index": idx,
                        "source": "ground_truth",
                        "original_type": ground_truth_raw,
                        "full_answer": ground_truth_raw
                    })
                
                result_item = {
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "index": idx,
                    "task_type": "cell type",
                    "task_variant": "singlecell_openended",
                    "question": qa_item["question"],
                    "ground_truth": ground_truth_std,
                    "predicted_answer": "",
                    "full_response": f"ERROR: {str(e)}",
                    "group": qa_item.get("group", "")
                }
                results.append(result_item)
                predictions.append("")
                ground_truths.append(ground_truth_std)

    except Exception as e:
        print(f"\n[ERROR] Critical error during batch processing: {e}")
        # Fallback to single-item processing
        for idx, qa_item in enumerate(tqdm(all_singlecell_qas, desc="Fallback Evaluating")):
            messages = prepare_messages(qa_item, system_msg)
            try:
                response = generator(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False
                )
                if isinstance(response, list) and len(response) > 0:
                    assistant_reply = response[0]["generated_text"]
                else:
                    assistant_reply = str(response)
                predicted_answer_raw = extract_answer_from_response(assistant_reply)
                ground_truth_raw = qa_item["answer"]
            except Exception as e2:
                print(f"\n[ERROR] Failed to process sample {idx}: {e2}")
                assistant_reply = f"ERROR: {str(e2)}"
                predicted_answer_raw = ""
                ground_truth_raw = qa_item["answer"]
            
            # Standardize cell type names (single cell)
            ground_truth_std, gt_is_mapped = standardizer.standardize_single_celltype(ground_truth_raw)
            predicted_answer_std, pred_is_mapped = standardizer.standardize_single_celltype(predicted_answer_raw)
            
            # Track unmapped types
            if not gt_is_mapped and ground_truth_raw:
                unmapped_records.append({
                    "index": idx,
                    "source": "ground_truth",
                    "original_type": ground_truth_raw,
                    "full_answer": ground_truth_raw
                })
            if not pred_is_mapped and predicted_answer_raw:
                unmapped_records.append({
                    "index": idx,
                    "source": "predicted_answer",
                    "original_type": predicted_answer_raw,
                    "full_answer": predicted_answer_raw
                })

            result_item = {
                "model_name": model_name,
                "dataset_id": dataset_id,
                "index": idx,
                "task_type": "cell type",
                "task_variant": "singlecell_openended",
                "question": qa_item["question"],
                "ground_truth": ground_truth_std,
                "predicted_answer": predicted_answer_std,
                "full_response": assistant_reply,
                "group": qa_item.get("group", "")
            }
            results.append(result_item)
            predictions.append(predicted_answer_std)
            ground_truths.append(ground_truth_std)

    metrics = calculate_metrics(predictions, ground_truths, all_singlecell_qas)

    print("\n" + "="*60)
    print("SINGLE-CELL + OPEN-ENDED EVALUATION")
    print("="*60)
    print(f"Total Single-cell Tasks: {metrics['total_cells']}")
    print(f"Total Batches:           {metrics['total_batches']}")
    print(f"Cell-Level Correct:      {metrics['cell_level_correct']}/{metrics['total_cells']}")
    print(f"Cell-Level Accuracy:     {metrics['cell_level_accuracy']:.2%}")
    print(f"Batch Exact Match:       {metrics['batch_exact_match_count']}/{metrics['total_batches']}")
    print(f"Batch Exact Match Acc:   {metrics['batch_exact_match_accuracy']:.2%}")
    print("="*60)

    results_file = os.path.join(output_dir, f"singlecell_openended_predictions_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Predictions saved to: {results_file}")

    metrics_file = os.path.join(output_dir, f"singlecell_openended_metrics_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_file}")
    
    # Save unmapped cell types report
    save_unmapped_report(unmapped_records, output_dir, "singlecell_openended", timestamp)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Single-cell + Open-ended evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input QA JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="ncbi/Cell-o1", help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    run_evaluation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
