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
    if (prediction.count('<think>') != 1 or
        prediction.count('</think>') != 1 or
        prediction.count('<answer>') != 1 or
        prediction.count('</answer>') != 1):
        return False
    pattern = re.compile(r'^<think>.*?</think>\n<answer>.*?</answer>$', re.DOTALL)
    if not pattern.match(prediction):
        return False
    if '|' not in candidate:
        return False
    return True

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

def process_record(record):
    if not isinstance(record, dict):
        try:
            record = json.loads(record)
        except Exception:
            return 0, 0, 0, 0, 0, 0, 0, 0

    gold = record.get("answer", "")
    prediction = record.get("prediction", "")
    gold_items = [s.strip() for s in gold.split("|") if s.strip()]
    total_count = len(gold_items)
    if total_count == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    candidate, _ = extract_candidate_and_reasoning_simple(prediction)
    prediction_length = len(tokenizer.encode(prediction))
    legitimate = is_legitimate_format(prediction, candidate, gold)

    if not legitimate:
        return 0, 0, 0, total_count, 0, 0, total_count, prediction_length

    candidate_items = [s.strip() for s in candidate.split("|") if s.strip()]
    correct_count = sum(1 for c, g in zip(candidate_items, gold_items) if c.lower() == g.lower())

    partial_acc = correct_count / total_count
    exact_acc = 1 if correct_count == total_count else 0
    uniqueness = len(set(candidate_items)) / len(candidate_items) if candidate_items else 0

    return partial_acc, exact_acc, correct_count, total_count, 1, uniqueness, total_count, prediction_length

def main():
    parser = argparse.ArgumentParser(description="Evaluate Cell-o1 predictions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .json prediction files.")
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()

    input_dir = args.input_dir
    max_workers = args.max_workers

    partial_accuracies, exact_match_accuracies, legitimate_flags, uniqueness_scores = [], [], [], []
    total_correct = total_items = total_exact = total_records = total_prediction_length = total_predictions = 0
    length_stats = defaultdict(lambda: {"count": 0, "partial_correct": 0, "partial_total": 0, "exact_matches": 0})

    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[!] Skipping invalid JSON: {fname}")
                continue

        records = data if isinstance(data, list) else [data]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_record, r) for r in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc=fname):
                p_acc, e_acc, correct, total, legit, uniq, n_items, pred_len = future.result()
                partial_accuracies.append(p_acc)
                exact_match_accuracies.append(e_acc)
                legitimate_flags.append(legit)
                uniqueness_scores.append(uniq)
                total_correct += correct
                total_items += total
                total_exact += e_acc
                total_records += 1
                total_prediction_length += pred_len
                total_predictions += 1

                length_stats[n_items]["count"] += 1
                length_stats[n_items]["partial_correct"] += correct
                length_stats[n_items]["partial_total"] += total
                length_stats[n_items]["exact_matches"] += e_acc

    # Summary
    print("\n=== Evaluation Summary ===")
    overall_partial = total_correct / total_items if total_items else 0
    overall_exact = total_exact / total_records if total_records else 0
    legit_ratio = sum(legitimate_flags) / total_records if total_records else 0
    avg_uniqueness = sum(uniqueness_scores) / total_records if total_records else 0
    avg_pred_len = total_prediction_length / total_predictions if total_predictions else 0
    std_exact = math.sqrt(overall_exact * (1 - overall_exact) / total_records) if total_records else 0
    std_partial = np.std(partial_accuracies, ddof=1) if len(partial_accuracies) > 1 else 0
    std_legit = math.sqrt(legit_ratio * (1 - legit_ratio) / total_records) if total_records else 0
    std_unique = np.std(uniqueness_scores, ddof=1) if len(uniqueness_scores) > 1 else 0

    print(f"Partial Accuracy (cell-level): {overall_partial:.4f} ± {std_partial:.4f}")
    print(f"Exact Match (batch-level):     {overall_exact:.4f} ± {std_exact:.4f}")
    print(f"Legitimate Format Ratio:       {legit_ratio:.4f} ± {std_legit:.4f}")
    print(f"Average Uniqueness:            {avg_uniqueness:.4f} ± {std_unique:.4f}")
    print(f"Average Prediction Length:     {avg_pred_len:.1f}")

    print("\n+-----------------+-------------+------------------+---------------------+")
    print("| Candidate Length| Sample Count| Partial Accuracy | Exact Match Accuracy|")
    print("+-----------------+-------------+------------------+---------------------+")
    for length in sorted(length_stats):
        stats = length_stats[length]
        p = stats["partial_correct"] / stats["partial_total"] if stats["partial_total"] else 0
        e = stats["exact_matches"] / stats["count"] if stats["count"] else 0
        print(f"| {length:^15} | {stats['count']:^11} | {p:^16.4f} | {e:^19.4f} |")
    print("+-----------------+-------------+------------------+---------------------+")

    print("\n[Distribution]")
    print("Partial Accuracies:", Counter(partial_accuracies))
    print("Exact Matches:", Counter(exact_match_accuracies))
    print("Legitimate Format:", Counter(legitimate_flags))
    print("Uniqueness:", Counter(round(u, 2) for u in uniqueness_scores))

if __name__ == "__main__":
    main()
