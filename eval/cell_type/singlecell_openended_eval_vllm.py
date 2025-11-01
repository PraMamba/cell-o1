"""
Single-cell + Open-ended Evaluation Script with vLLM acceleration

This variant tests:
- Each batch is decomposed into N independent single-cell tasks
- Each task shows only ONE cell's gene expression
- NO candidate list is provided (open-ended generation)
- Models must freely generate cell type names

Uses vLLM for efficient batch inference.

Usage:
    python singlecell_openended_eval_vllm.py \
        --input_file /path/to/conversations.jsonl \
        --output_dir /path/to/save/results \
        --model_name ncbi/Cell-o1 \
        --max_new_tokens 1000 \
        --batch_size 256 \
        --tensor_parallel_size 2
"""

import os
import json
import argparse
import re
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import logging

from vllm import LLM, SamplingParams

from celltype_standardizer import CellTypeStandardizer, extract_dataset_id_from_path, save_unmapped_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_conversation_data(input_file: str) -> List[Dict]:
    """Load conversation data from JSONL file."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"[INFO] Loaded {len(data)} conversation items from {input_file}")
    return data


def prepare_prompts_from_conversations(conversation_items: List[Dict], tokenizer) -> List[str]:
    """
    Convert conversation items to text prompts using chat template.

    Expected format:
    {
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}  # This is the ground truth, exclude from prompt
      ]
    }
    """
    prompts = []

    for item in conversation_items:
        conversations = item.get('conversations', [])

        # Convert 'from' field to 'role' for chat template compatibility
        chat_messages = []
        for conv in conversations:
            if conv.get('from') == 'system':
                chat_messages.append({"role": "system", "content": conv.get('value', '')})
            elif conv.get('from') == 'human':
                chat_messages.append({"role": "user", "content": conv.get('value', '')})
            # Skip 'gpt' messages as those are ground truth answers

        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            try:
                prompt = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(str(prompt) if prompt is not None else "")
            except Exception as e:
                logging.warning(f"Chat template failed: {e}, using simple format")
                # Fallback to simple format
                user_content = chat_messages[-1]["content"] if len(chat_messages) > 0 else ""
                system_content = chat_messages[0]["content"] if len(chat_messages) > 1 else ""
                prompts.append(f"{system_content}\n\nUser: {user_content}\n\nAssistant:")
        else:
            # No chat template, use simple format
            user_content = ""
            system_content = ""
            for msg in chat_messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            prompts.append(f"{system_content}\n\nUser: {user_content}\n\nAssistant:")

    return prompts


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


def extract_ground_truth_from_conversation(conversations: List[Dict]) -> str:
    """Extract ground truth answer from conversations."""
    for conv in conversations:
        if conv.get('from') == 'gpt':
            return conv.get('value', '')
    return ""


def extract_question_from_conversation(conversations: List[Dict]) -> str:
    """Extract question from conversations."""
    for conv in conversations:
        if conv.get('from') == 'human':
            return conv.get('value', '')
    return ""


def calculate_metrics(predictions: List[str], ground_truths: List[str], conversation_data: List[Dict]) -> Dict:
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
    for i, (pred, gt, item) in enumerate(zip(predictions, ground_truths, conversation_data)):
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
    batch_size: int = 256,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1
):
    """Run single-cell + open-ended evaluation with vLLM."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conversation_data = load_conversation_data(input_file)

    # Initialize cell type standardizer
    standardizer = CellTypeStandardizer()

    # Extract dataset ID from input file path
    dataset_id = extract_dataset_id_from_path(input_file)

    print(f"\n[INFO] Task Variant: Single-cell + Open-ended (vLLM)")
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Tensor parallel size: {tensor_parallel_size}")
    print(f"[INFO] GPU memory utilization: {gpu_memory_utilization}")

    # Initialize vLLM engine
    logging.info("Initializing vLLM engine...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()

    # Set tokenizer properties
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None,
    )

    logging.info(f"Sampling parameters: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_new_tokens}")

    print(f"[INFO] Starting evaluation on {len(conversation_data)} single-cell tasks with batch_size={batch_size}...")

    # Prepare all prompts
    all_prompts = prepare_prompts_from_conversations(conversation_data, tokenizer)

    results = []
    predictions = []
    ground_truths = []
    unmapped_records = []  # Track unmapped cell types

    # Batch inference with vLLM
    logging.info("Starting vLLM batch inference...")

    # Generate in batches to show progress
    all_outputs = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="vLLM Inference"):
        batch_prompts = all_prompts[i:i + batch_size]
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(outputs)
        except Exception as e:
            logging.error(f"Error in batch {i//batch_size}: {e}")
            # Add empty outputs for failed batch
            all_outputs.extend([None] * len(batch_prompts))

    # Process results
    for idx, (item, output) in enumerate(zip(conversation_data, all_outputs)):
        try:
            if output is not None:
                assistant_reply = output.outputs[0].text.strip()
            else:
                assistant_reply = "ERROR: Generation failed"

            predicted_answer_raw = extract_answer_from_response(assistant_reply)
            ground_truth_raw = extract_ground_truth_from_conversation(item.get('conversations', []))
            question = extract_question_from_conversation(item.get('conversations', []))

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
                "question": question,
                "ground_truth": ground_truth_std,
                "predicted_answer": predicted_answer_std,
                "full_response": assistant_reply,
                "group": item.get("group", "")
            }
            results.append(result_item)
            predictions.append(predicted_answer_std)
            ground_truths.append(ground_truth_std)

        except Exception as e:
            print(f"\n[ERROR] Failed to process result for sample {idx}: {e}")
            ground_truth_raw = extract_ground_truth_from_conversation(item.get('conversations', []))
            ground_truth_std, gt_is_mapped = standardizer.standardize_single_celltype(ground_truth_raw)
            question = extract_question_from_conversation(item.get('conversations', []))

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
                "question": question,
                "ground_truth": ground_truth_std,
                "predicted_answer": "",
                "full_response": f"ERROR: {str(e)}",
                "group": item.get("group", "")
            }
            results.append(result_item)
            predictions.append("")
            ground_truths.append(ground_truth_std)

    metrics = calculate_metrics(predictions, ground_truths, conversation_data)

    print("\n" + "="*60)
    print("SINGLE-CELL + OPEN-ENDED EVALUATION (vLLM)")
    print("="*60)
    print(f"Total Single-cell Tasks: {metrics['total_cells']}")
    print(f"Total Batches:           {metrics['total_batches']}")
    print(f"Cell-Level Correct:      {metrics['cell_level_correct']}/{metrics['total_cells']}")
    print(f"Cell-Level Accuracy:     {metrics['cell_level_accuracy']:.2%}")
    print(f"Batch Exact Match:       {metrics['batch_exact_match_count']}/{metrics['total_batches']}")
    print(f"Batch Exact Match Acc:   {metrics['batch_exact_match_accuracy']:.2%}")
    print("="*60)

    results_file = os.path.join(output_dir, f"singlecell_openended_predictions_vllm_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Predictions saved to: {results_file}")

    metrics_file = os.path.join(output_dir, f"singlecell_openended_metrics_vllm_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_file}")

    # Save unmapped cell types report
    save_unmapped_report(unmapped_records, output_dir, "singlecell_openended_vllm", timestamp)

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="Single-cell + Open-ended evaluation with vLLM")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input conversation JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--model_name", type=str, default="ncbi/Cell-o1", help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for multi-GPU")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    run_evaluation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
