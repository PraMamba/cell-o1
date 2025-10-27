"""
Improved Answer Extraction Module

This module provides robust answer extraction from model outputs,
handling various format errors and edge cases.
"""

import re
from typing import Optional


def extract_answer_from_response(response_text: str) -> str:
    """
    Extract answer from model response with robust error handling.

    This function tries multiple strategies to extract the answer:
    1. Extract from <answer>...</answer> tags (preferred)
    2. Remove <think>...</think> blocks and extract remaining content
    3. Fall back to heuristics for pipe-separated cell types

    Args:
        response_text: Raw model response

    Returns:
        Extracted answer (pipe-separated cell types) or empty string
    """
    if not response_text or not isinstance(response_text, str):
        return ""

    # Remove any leading/trailing whitespace
    response_text = response_text.strip()

    # Strategy 1: Extract from <answer>...</answer> tags (standard format)
    answer_match = re.search(
        r'<answer>\s*(.*?)\s*</answer>',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        answer = answer_match.group(1).strip()
        # Clean up the answer
        answer = clean_answer(answer)
        if answer:
            return answer

    # Strategy 2: Remove <think> blocks and extract clean content
    # Remove everything between <think> and </think>
    cleaned_text = re.sub(
        r'<think>.*?</think>',
        '',
        response_text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove any remaining <think> or </think> tags
    cleaned_text = re.sub(r'</?think>', '', cleaned_text, flags=re.IGNORECASE)

    # Remove "Final answer:" prefix if present
    cleaned_text = re.sub(r'Final\s+answer\s*:\s*', '', cleaned_text, flags=re.IGNORECASE)

    # Look for pipe-separated cell types in the cleaned text
    lines = cleaned_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue

        # Skip lines that look like XML tags or error messages
        if re.match(r'^<[^>]+>$', line):
            continue
        if 'error' in line.lower() and len(line) < 50:
            continue

        # Look for pipe-separated format (cell type | cell type | ...)
        if '|' in line and len(line) < 1000:
            answer = clean_answer(line)
            if answer and not is_format_error(answer):
                return answer

    # Strategy 3: Look for single cell type (for single-cell tasks)
    # Check the last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 200 and not re.match(r'^<[^>]+>$', line):
            # Check if it looks like a cell type name
            if is_valid_celltype_format(line):
                answer = clean_answer(line)
                if answer and not is_format_error(answer):
                    return answer

    # If all strategies fail, return empty string
    return ""


def clean_answer(answer: str) -> str:
    """
    Clean up extracted answer.

    Args:
        answer: Raw extracted answer

    Returns:
        Cleaned answer
    """
    if not answer:
        return ""

    # Remove any remaining XML tags
    answer = re.sub(r'<[^>]+>', '', answer)

    # Remove "Final answer:" prefix
    answer = re.sub(r'Final\s+answer\s*:\s*', '', answer, flags=re.IGNORECASE)

    # Normalize whitespace around pipes
    answer = re.sub(r'\s*\|\s*', ' | ', answer)

    # Remove leading/trailing whitespace
    answer = answer.strip()

    # Remove trailing punctuation like periods
    answer = re.sub(r'[.。]+$', '', answer)

    return answer


def is_format_error(text: str) -> bool:
    """
    Check if text contains format errors.

    Args:
        text: Text to check

    Returns:
        True if contains format errors
    """
    error_patterns = [
        r'</?think>',
        r'</?answer>',
        r'Final\s+answer\s*:',
        r'^Cell\s+\d+$',
        r'^\s*$',  # Empty or whitespace only
    ]

    for pattern in error_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def is_valid_celltype_format(text: str) -> bool:
    """
    Check if text looks like a valid cell type name.

    Args:
        text: Text to check

    Returns:
        True if looks like a valid cell type
    """
    # Should not be too short or too long
    if len(text) < 2 or len(text) > 150:
        return False

    # Should not contain special characters that indicate errors
    invalid_chars = ['<', '>', '{', '}', '[', ']']
    if any(char in text for char in invalid_chars):
        return False

    # Should contain some alphabetic characters
    if not re.search(r'[a-zA-Z]', text):
        return False

    # Common cell type patterns
    cell_type_patterns = [
        r'\bcell\b',
        r'\bcyte\b',
        r'\bblast\b',
        r'CD\d+',
        r'alpha',
        r'beta',
        r'gamma',
        r'delta',
        r'\bT\b',
        r'\bB\b',
        r'\bNK\b',
    ]

    # If it matches any common pattern, it's probably valid
    for pattern in cell_type_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Otherwise, if it's not too short and doesn't have errors, accept it
    return len(text) > 5


def batch_extract_answers(responses: list) -> list:
    """
    Extract answers from multiple responses.

    Args:
        responses: List of response texts

    Returns:
        List of extracted answers
    """
    return [extract_answer_from_response(resp) for resp in responses]


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Normal case
        ("<think>reasoning here</think>\n<answer>CD4+ T cell | B cell | NK cell</answer>",
         "CD4+ T cell | B cell | NK cell"),

        # Format error case 1
        ("<think>reasoning</think>\nFinal answer: CD4+ T cell</think>",
         "CD4+ T cell"),

        # Format error case 2
        ("CD4+ T cell | B cell | NK cell\n</think>",
         "CD4+ T cell | B cell | NK cell"),

        # Single cell type
        ("<think>reasoning</think>\n<answer>Naive CD4+ T cell</answer>",
         "Naive CD4+ T cell"),

        # Missing tags
        ("CD14+ monocyte | NK cell",
         "CD14+ monocyte | NK cell"),

        # Empty
        ("", ""),

        # Only think block
        ("<think>some reasoning</think>", ""),
    ]

    print("Testing improved answer extraction:")
    print("=" * 80)

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = extract_answer_from_response(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\nTest {i}: {status}")
        print(f"  Input: {input_text[:60]}...")
        print(f"  Expected: '{expected}'")
        print(f"  Got: '{result}'")

    print("\n" + "=" * 80)
