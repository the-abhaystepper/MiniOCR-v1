"""OCR evaluation metrics: CER, WER, Exact Match."""

import math
from typing import List


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def character_error_rate(prediction: str, target: str) -> float:
    """Character Error Rate (CER).

    CER = edit_distance(prediction, target) / len(target)
    """
    if len(target) == 0:
        return 0.0 if len(prediction) == 0 else float('inf')
    return edit_distance(prediction, target) / len(target)


def word_error_rate(prediction: str, target: str) -> float:
    """Word Error Rate (WER).

    WER = edit_distance(words_prediction, words_target) / num_words_target
    """
    pred_words = prediction.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else float('inf')
    return edit_distance(prediction, target) / len(target_words)


def exact_match(prediction: str, target: str) -> bool:
    """Check if prediction exactly matches target (after stripping whitespace)."""
    return prediction.strip() == target.strip()


def batch_metrics(
    predictions: List[str],
    targets: List[str],
) -> dict:
    """Compute metrics for a batch of predictions."""
    assert len(predictions) == len(targets), "Predictions and targets must have same length"

    cer_values = [character_error_rate(p, t) for p, t in zip(predictions, targets)]
    wer_values = [word_error_rate(p, t) for p, t in zip(predictions, targets)]
    em_values = [1.0 if exact_match(p, t) else 0.0 for p, t in zip(predictions, targets)]

    avg_cer = sum(cer_values) / len(cer_values)
    avg_wer = sum(wer_values) / len(wer_values)
    avg_em = sum(em_values) / len(em_values)

    return {
        "CER": avg_cer,
        "WER": avg_wer,
        "ExactMatch": avg_em,
    }
