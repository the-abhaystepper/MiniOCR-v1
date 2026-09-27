"""OCR evaluation metrics: CER, WER, Exact Match."""


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            current_row.append(min(
                previous_row[j + 1] + 1,      # insertion
                current_row[j] + 1,            # deletion
                previous_row[j] + (c1 != c2),  # substitution
            ))
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(prediction: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(prediction) == 0 else float('inf')
    return edit_distance(prediction, target) / len(target)


def word_error_rate(prediction: str, target: str) -> float:
    pred_words = prediction.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else float('inf')
    return edit_distance(prediction, target) / len(target_words)


def exact_match(prediction: str, target: str) -> bool:
    return prediction.strip() == target.strip()


def batch_metrics(predictions: list[str], targets: list[str]) -> dict:
    """Compute average CER, WER, and ExactMatch over a batch."""
    assert len(predictions) == len(targets)
    cer_values = [character_error_rate(p, t) for p, t in zip(predictions, targets)]
    wer_values = [word_error_rate(p, t) for p, t in zip(predictions, targets)]
    em_values = [1.0 if exact_match(p, t) else 0.0 for p, t in zip(predictions, targets)]
    return {
        "CER": sum(cer_values) / len(cer_values),
        "WER": sum(wer_values) / len(wer_values),
        "ExactMatch": sum(em_values) / len(em_values),
    }
