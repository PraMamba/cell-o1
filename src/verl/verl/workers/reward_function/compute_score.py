import re

def extract_solution(prediction):
    """
    Extract candidate answer and reasoning from the prediction using a simple, line‐based approach.
    
    Candidate Extraction:
      - Split prediction into lines.
      - If any line contains "<answer>", candidate is the text after "<answer>" in that line
        (with any trailing "</answer>" removed).
      - Otherwise, if any line contains the "|" delimiter, candidate is set to that entire line.
      - Otherwise, candidate is empty.
    """
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

    return candidate


def is_legitimate_format(prediction, candidate):
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if "cell_data" not in data_source:
        return -1

    candidate = extract_solution(solution_str)
    
    if candidate is None or not is_legitimate_format(solution_str, candidate):
        return -1
    
    gold_items = [s.strip() for s in ground_truth.split("|") if s.strip()]
    candidate_items = [s.strip() for s in candidate.split("|") if s.strip()]

    if not gold_items or len(gold_items) != len(candidate_items):
        return -1
    
    if len(set(candidate_items)) != len(candidate_items):
        return -1
    
    total_count = len(gold_items)
    if total_count == 0:
        return -1

    correct_count = 0
    for i in range(total_count):
        if i < len(candidate_items) and candidate_items[i].lower() == gold_items[i].lower():
            correct_count += 1

    partial_accuracy = correct_count / total_count
    exact_match_accuracy = 1 if correct_count == total_count else 0
    
    return (partial_accuracy + exact_match_accuracy) / 2
    # return exact_match_accuracy