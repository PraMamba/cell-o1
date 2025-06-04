import os
import json
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import re
import math
import numpy as np

def extract_candidate_and_reasoning_simple(prediction):
    lines = prediction.splitlines()
    candidate = ""
    for line in lines:
        if "<answer>" in line:
            candidate = line.split("<answer>", 1)[1].strip()
            if candidate.endswith("</answer>"):
                candidate = candidate.replace("</answer>", "").strip()
            break
    if not candidate:
        for line in lines:
            if "|" in line:
                candidate = line.strip()
                break
    reasoning = ""
    if "<think>" in prediction and "</think>" in prediction:
        try:
            start = prediction.index("<think>") + len("<think>")
            end = prediction.index("</think>", start)
            reasoning = prediction[start:end].strip()
        except ValueError:
            reasoning = ""
    return candidate, reasoning

def is_legitimate_format(prediction, candidate, gold):
    # Check for exactly one occurrence of each tag
    if (prediction.count('<think>') != 1 or
        prediction.count('</think>') != 1 or
        prediction.count('<answer>') != 1 or
        prediction.count('</answer>') != 1):
        return False
    
    # Ensure the entire string is just <think>...</think><answer>...</answer>
    # with no extra text outside the tags
    pattern = re.compile(r'^<think>.*?</think>\n<answer>.*?</answer>$', re.DOTALL)
    if not pattern.match(prediction):
        return False
    
    # Check for '|' in candidate
    if '|' not in candidate:
        return False
    
    return True


tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

def process_record(record):
    """
    Process a single record to compute evaluation metrics.
    Returns:
        (partial_acc, exact_acc, correct, total, legitimate_flag, uniqueness, num_items, pred_length)
    """
    if not isinstance(record, dict):
        try:
            record = json.loads(record)
        except Exception:
            return 0, 0, 0, 0, 0, 0, 0, 0

    gold = record.get("answer", "")
    prediction = record.get("prediction", "")

    # Split ground-truth into individual labels
    gold_items = [s.strip() for s in gold.split("|") if s.strip()]
    total_count = len(gold_items)
    if total_count == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    # Extract candidate answer and reasoning
    candidate, reasoning = extract_candidate_and_reasoning_simple(prediction)
    record["short_answer"] = candidate
    record["reasoning"] = reasoning

    # Compute token length of the prediction
    prediction_length = len(tokenizer.encode(prediction))

    # Check formatting validity
    legitimate = is_legitimate_format(prediction, candidate)

    if not legitimate:
        # Format is invalid; still count in total but score 0
        return 0, 0, 0, total_count, 0, 0, total_count, prediction_length

    candidate_items = [s.strip() for s in candidate.split("|") if s.strip()]
    correct_count = sum(
        1 for c, g in zip(candidate_items, gold_items)
        if c.lower() == g.lower()
    )

    partial_acc = correct_count / total_count
    exact_acc = 1 if correct_count == total_count else 0
    uniqueness = (
        len(set(candidate_items)) / len(candidate_items)
        if candidate_items else 0
    )

    return (
        partial_acc,
        exact_acc,
        correct_count,
        total_count,
        1,
        uniqueness,
        total_count,
        prediction_length,
    )

def process_record(record):
    """
    Process a single record to compute evaluation metrics.
    Returns:
        (partial_acc, exact_acc, correct, total, legitimate_flag, uniqueness, num_items, pred_length)
    """
    if not isinstance(record, dict):
        try:
            record = json.loads(record)
        except Exception:
            return 0, 0, 0, 0, 0, 0, 0, 0

    gold = record.get("answer", "")
    prediction = record.get("prediction", "")

    # Split ground-truth into individual labels
    gold_items = [s.strip() for s in gold.split("|") if s.strip()]
    total_count = len(gold_items)
    if total_count == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    # Extract candidate answer and reasoning
    candidate, reasoning = extract_candidate_and_reasoning_simple(prediction)
    record["short_answer"] = candidate
    record["reasoning"] = reasoning

    # Compute token length of the prediction
    prediction_length = len(tokenizer.encode(prediction))

    # Check formatting validity
    legitimate = is_legitimate_format(prediction, candidate, gold)

    if not legitimate:
        # Format is invalid; still count in total but score 0
        return 0, 0, 0, total_count, 0, 0, total_count, prediction_length

    candidate_items = [s.strip() for s in candidate.split("|") if s.strip()]
    correct_count = sum(
        1 for c, g in zip(candidate_items, gold_items)
        if c.lower() == g.lower()
    )

    partial_acc = correct_count / total_count
    exact_acc = 1 if correct_count == total_count else 0
    uniqueness = (
        len(set(candidate_items)) / len(candidate_items)
        if candidate_items else 0
    )

    return (
        partial_acc,
        exact_acc,
        correct_count,
        total_count,
        1,
        uniqueness,
        total_count,
        prediction_length,
    )



def main():
    parser = argparse.ArgumentParser(
        description="Update records and evaluate predictions using simple line-based extraction."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input JSON files (each file is a dict or a list of dicts).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output JSON file for updated records.")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Maximum number of worker threads for parallel processing.")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_file = args.output_file
    max_workers = args.max_workers

    updated_records = []
    partial_accuracies = []
    exact_match_accuracies = []
    legitimate_flags = []
    uniqueness_scores = []
    
    total_correct = 0
    total_items = 0
    total_exact = 0
    total_records = 0
    
    length_stats = defaultdict(lambda: {"count": 0, "partial_correct": 0, "partial_total": 0, "exact_matches": 0})
    
    # Store total prediction lengths for average calculation
    total_prediction_length = 0
    total_predictions = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error: {filepath} is not a valid JSON file. Skipped.")
                    continue

            if isinstance(data, dict):
                records = [data]
            elif isinstance(data, list):
                records = data
            else:
                print(f"Warning: {filepath} did not contain a dict or list. Skipped.")
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_record, record) for record in records]
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"Processing {filename}", unit="record"):
                    partial_acc, exact_acc, correct, total, legitimate, uniqueness, num_unique, prediction_length = future.result()
                    partial_accuracies.append(partial_acc)
                    exact_match_accuracies.append(exact_acc)
                    legitimate_flags.append(legitimate)
                    uniqueness_scores.append(uniqueness)
                    
                    length_stats[num_unique]["count"] += 1
                    length_stats[num_unique]["partial_correct"] += correct
                    length_stats[num_unique]["partial_total"] += total
                    length_stats[num_unique]["exact_matches"] += exact_acc
                    
                    total_correct += correct
                    total_items += total
                    total_exact += exact_acc
                    total_records += 1
                    
                    # Update total prediction length
                    total_prediction_length += prediction_length
                    total_predictions += 1

            if isinstance(data, dict):
                updated_records.append(data)
            else:
                updated_records.extend(records)

    overall_partial_accuracy = total_correct / total_items if total_items > 0 else 0
    overall_exact_match_accuracy = total_exact / total_records if total_records > 0 else 0
    legitimate_ratio = sum(legitimate_flags) / total_records if total_records > 0 else 0
    average_uniqueness = sum(uniqueness_scores) / total_records if total_records > 0 else 0

    average_prediction_length = total_prediction_length / total_predictions if total_predictions > 0 else 0


    # ---------- 1. std of proportion ----------
    batch_std_prop = math.sqrt(
        overall_exact_match_accuracy * (1 - overall_exact_match_accuracy) / total_records
    ) if total_records > 0 else 0

    # ---------- 2) cell‑level (partial) ----------
    cell_std_sample = np.std(partial_accuracies, ddof=1) if len(partial_accuracies) > 1 else 0

    format_std_prop = math.sqrt(
        legitimate_ratio * (1 - legitimate_ratio) / total_records
    ) if total_records > 0 else 0
    
    uniqueness_std_sample = np.std(uniqueness_scores, ddof=1) if len(uniqueness_scores) > 1 else 0

    print("Overall Partial Accuracy (cell‑level): {:.4f} ± {:.4f} (sample std across batches)".format(
        overall_partial_accuracy, cell_std_sample))
    print("Overall Exact‑Match Accuracy (batch‑level): {:.4f} ± {:.4f} (std of proportion)".format(
        overall_exact_match_accuracy, batch_std_prop))

    print("Legitimate Format Ratio: {:.4f} ± {:.4f} (std of proportion)".format(
        legitimate_ratio, format_std_prop))
    print("Average Uniqueness: {:.4f} ± {:.4f} (sample std)".format(
        average_uniqueness, uniqueness_std_sample))
    
    print("Overall Partial Accuracy: {:.4f}".format(overall_partial_accuracy))
    print("Overall Exact Match Accuracy: {:.4f}".format(overall_exact_match_accuracy))
    print("Legitimate Format Ratio: {:.4f}".format(legitimate_ratio))
    print("Average Uniqueness: {:.4f}".format(average_uniqueness))
    print("Average Prediction Length: {:.4f}".format(average_prediction_length))
    
    print("\nAccuracy Statistics by Candidate Length:")
    print("+-----------------+-------------+------------------+---------------------+")
    print("| Candidate Length| Sample Count| Partial Accuracy | Exact Match Accuracy|")
    print("+-----------------+-------------+------------------+---------------------+")

    for length in sorted(length_stats.keys()):
        stats = length_stats[length]
        partial_acc = stats["partial_correct"] / stats["partial_total"] if stats["partial_total"] > 0 else 0
        exact_acc = stats["exact_matches"] / stats["count"] if stats["count"] > 0 else 0
        print(f"| {length:^15} | {stats['count']:^11} | {partial_acc:^16.4f} | {exact_acc:^19.4f} |")

    print("+-----------------+-------------+------------------+---------------------+")
    
    print("\nPer-record Partial Accuracy Distribution (Counter):", Counter(partial_accuracies))
    print("Per-record Exact Match Distribution (Counter):", Counter(exact_match_accuracies))
    print("Legitimate Format Distribution (Counter):", Counter(legitimate_flags))
    print("Uniqueness Distribution (Counter):", Counter(round(u, 2) for u in uniqueness_scores))

    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(updated_records, out_f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
