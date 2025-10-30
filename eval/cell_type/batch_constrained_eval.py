"""
Batch + Constrained Evaluation Script (Baseline)

This is the original setting where:
- Models receive the full batch of N cells
- A list of N candidate cell types is provided
- Models must assign each candidate to exactly one cell

This serves as the baseline for comparison with other task design variants.

Usage:
    python batch_constrained_eval.py \
        --input_file /path/to/qa_pairs.json \
        --output_dir /path/to/save/results \
        --model_name ncbi/Cell-o1 \
        --max_new_tokens 2000
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
    """
    Convert a QA item to chat-style messages.
    This is the standard batch + constrained setting (baseline).
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": qa_item["question"]}
    ]
    return messages


def extract_answer_from_response(response_text: str) -> str:
    """Extract the final answer from model's response."""
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    lines = response_text.strip().split('\n')
    for line in reversed(lines):
        if '|' in line and len(line) < 500:
            return line.strip()

    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    answer = re.sub(r'\s+', ' ', answer.strip())
    answer = re.sub(r'\s*\|\s*', ' | ', answer)
    return answer


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate evaluation metrics."""
    total = len(predictions)
    exact_match = 0
    cell_level_correct = 0
    total_cells = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_answer(pred)
        gt_norm = normalize_answer(gt)

        if pred_norm.lower() == gt_norm.lower():
            exact_match += 1

        pred_types = [t.strip() for t in pred_norm.split('|')]
        gt_types = [t.strip() for t in gt_norm.split('|')]

        if len(pred_types) == len(gt_types):
            for p, g in zip(pred_types, gt_types):
                total_cells += 1
                if p.lower() == g.lower():
                    cell_level_correct += 1

    metrics = {
        "task_variant": "batch_constrained",
        "total_samples": total,
        "exact_match_count": exact_match,
        "exact_match_accuracy": exact_match / total if total > 0 else 0.0,
        "cell_level_correct": cell_level_correct,
        "cell_level_total": total_cells,
        "cell_level_accuracy": cell_level_correct / total_cells if total_cells > 0 else 0.0
    }

    return metrics


def run_evaluation(
    input_file: str,
    output_dir: str,
    model_name: str = "ncbi/Cell-o1",
    max_new_tokens: int = 2000,
    batch_size: int = 4,
    device: str = "auto"
):
    """Run batch + constrained evaluation (baseline)."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    qa_data = load_qa_data(input_file)
    
    # Initialize cell type standardizer
    standardizer = CellTypeStandardizer()
    
    # Extract dataset ID from input file path
    dataset_id = extract_dataset_id_from_path(input_file)

    # System message for batch + constrained setting
    system_msg = (
        "You are an expert assistant specialized in cell type annotation. "
        "You will be given a batch of N cells from the same donor, where each cell represents a unique cell type. "
        "For each cell, the top expressed genes are provided in descending order of expression. "
        "Using both the gene expression data and donor information, determine the correct cell type for each cell. "
        "You will also receive a list of N candidate cell types, and each candidate must be assigned to exactly one cell. "
        "Ensure that you consider all cells and candidate types together, rather than annotating each cell individually. "
        "Include your detailed reasoning within <think> and </think> tags, and provide your final answer within <answer> and </answer> tags. "
        "The final answer should be a single string listing the assigned cell types in order, separated by ' | '."
    )

    print(f"\n[INFO] Task Variant: Batch + Constrained (Baseline)")
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

    print(f"[INFO] Starting evaluation on {len(qa_data)} samples with batch_size={batch_size}...")

    results = []
    predictions = []
    ground_truths = []
    unmapped_records = []  # Track unmapped cell types

    # Prepare all messages at once for batch processing
    all_messages = [prepare_messages(qa_item, system_msg) for qa_item in qa_data]

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
        for idx, (qa_item, response) in enumerate(zip(qa_data, batch_responses)):
            try:
                if isinstance(response, list) and len(response) > 0:
                    assistant_reply = response[0]["generated_text"]
                else:
                    assistant_reply = str(response)

                predicted_answer_raw = extract_answer_from_response(assistant_reply)
                ground_truth_raw = qa_item["answer"]
                
                # Standardize cell type names
                ground_truth_std, gt_unmapped = standardizer.standardize_batch_celltype(ground_truth_raw)
                predicted_answer_std, pred_unmapped = standardizer.standardize_batch_celltype(predicted_answer_raw)
                
                # Track unmapped types
                for unmapped_type in gt_unmapped:
                    unmapped_records.append({
                        "index": idx,
                        "source": "ground_truth",
                        "original_type": unmapped_type,
                        "full_answer": ground_truth_raw
                    })
                for unmapped_type in pred_unmapped:
                    unmapped_records.append({
                        "index": idx,
                        "source": "predicted_answer",
                        "original_type": unmapped_type,
                        "full_answer": predicted_answer_raw
                    })

                result_item = {
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "index": idx,
                    "task_type": "cell type",
                    "task_variant": "batch_constrained",
                    "question": qa_item["question"],
                    "ground_truth": ground_truth_std,
                    "predicted_answer": predicted_answer_std,
                    "full_response": assistant_reply,
                    "group": qa_item.get("group", "unknown")
                }
                results.append(result_item)
                predictions.append(predicted_answer_std)
                ground_truths.append(ground_truth_std)

            except Exception as e:
                print(f"\n[ERROR] Failed to process result for sample {idx}: {e}")
                ground_truth_raw = qa_item["answer"]
                ground_truth_std, gt_unmapped = standardizer.standardize_batch_celltype(ground_truth_raw)
                
                for unmapped_type in gt_unmapped:
                    unmapped_records.append({
                        "index": idx,
                        "source": "ground_truth",
                        "original_type": unmapped_type,
                        "full_answer": ground_truth_raw
                    })
                
                result_item = {
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "index": idx,
                    "task_type": "cell type",
                    "task_variant": "batch_constrained",
                    "question": qa_item["question"],
                    "ground_truth": ground_truth_std,
                    "predicted_answer": "",
                    "full_response": f"ERROR: {str(e)}",
                    "group": qa_item.get("group", "unknown")
                }
                results.append(result_item)
                predictions.append("")
                ground_truths.append(ground_truth_std)

    except Exception as e:
        print(f"\n[ERROR] Critical error during batch processing: {e}")
        # Fallback to single-item processing
        for idx, qa_item in enumerate(tqdm(qa_data, desc="Fallback Evaluating")):
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
            
            # Standardize cell type names
            ground_truth_std, gt_unmapped = standardizer.standardize_batch_celltype(ground_truth_raw)
            predicted_answer_std, pred_unmapped = standardizer.standardize_batch_celltype(predicted_answer_raw)
            
            # Track unmapped types
            for unmapped_type in gt_unmapped:
                unmapped_records.append({
                    "index": idx,
                    "source": "ground_truth",
                    "original_type": unmapped_type,
                    "full_answer": ground_truth_raw
                })
            for unmapped_type in pred_unmapped:
                unmapped_records.append({
                    "index": idx,
                    "source": "predicted_answer",
                    "original_type": unmapped_type,
                    "full_answer": predicted_answer_raw
                })

            result_item = {
                "model_name": model_name,
                "dataset_id": dataset_id,
                "index": idx,
                "task_type": "cell type",
                "task_variant": "batch_constrained",
                "question": qa_item["question"],
                "ground_truth": ground_truth_std,
                "predicted_answer": predicted_answer_std,
                "full_response": assistant_reply,
                "group": qa_item.get("group", "unknown")
            }
            results.append(result_item)
            predictions.append(predicted_answer_std)
            ground_truths.append(ground_truth_std)

    metrics = calculate_metrics(predictions, ground_truths)

    print("\n" + "="*60)
    print("BATCH + CONSTRAINED EVALUATION (BASELINE)")
    print("="*60)
    print(f"Total Samples:           {metrics['total_samples']}")
    print(f"Exact Match Count:       {metrics['exact_match_count']}")
    print(f"Exact Match Accuracy:    {metrics['exact_match_accuracy']:.2%}")
    print(f"Cell-Level Correct:      {metrics['cell_level_correct']}/{metrics['cell_level_total']}")
    print(f"Cell-Level Accuracy:     {metrics['cell_level_accuracy']:.2%}")
    print("="*60)

    results_file = os.path.join(output_dir, f"batch_constrained_predictions_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Predictions saved to: {results_file}")

    metrics_file = os.path.join(output_dir, f"batch_constrained_metrics_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_file}")
    
    # Save unmapped cell types report
    save_unmapped_report(unmapped_records, output_dir, "batch_constrained", timestamp)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Batch + Constrained evaluation (baseline)")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input QA JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="ncbi/Cell-o1", help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
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
