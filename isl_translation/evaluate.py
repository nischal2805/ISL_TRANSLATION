"""
ISL Translation System - Evaluation Module
===========================================
Metrics computation and evaluation utilities.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


def compute_char_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute character-level accuracy.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Character accuracy (0.0 to 1.0)
    """
    total_chars = 0
    correct_chars = 0
    
    for pred, target in zip(predictions, targets):
        # Normalize
        pred = pred.lower().strip()
        target = target.lower().strip()
        
        # Count matching characters (considering length)
        min_len = min(len(pred), len(target))
        max_len = max(len(pred), len(target))
        
        if max_len == 0:
            continue
        
        matches = sum(p == t for p, t in zip(pred, target))
        
        total_chars += max_len
        correct_chars += matches
    
    return correct_chars / max(total_chars, 1)


def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = total characters in target
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        CER (0.0 to 1.0+, lower is better)
    """
    total_distance = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        pred = pred.lower().strip()
        target = target.lower().strip()
        
        distance = compute_edit_distance(pred, target)
        total_distance += distance
        total_chars += len(target)
    
    return total_distance / max(total_chars, 1)


def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Word Error Rate (WER) using TRUE word-level edit distance.
    
    WER = (S + D + I) / N (at word level)
    where S = word substitutions, D = word deletions, I = word insertions
    N = total words in target
    
    IMPORTANT: This computes edit distance on word tokens, NOT characters.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        WER (0.0 to 1.0+, lower is better)
    """
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.lower().strip().split()
        target_words = target.lower().strip().split()
        
        # Use TRUE word-level edit distance (not character-level!)
        distance = word_edit_distance(pred_words, target_words)
        
        total_distance += distance
        total_words += len(target_words)
    
    return total_distance / max(total_words, 1)


def word_edit_distance(words1: List[str], words2: List[str]) -> int:
    """Compute edit distance at word level."""
    if len(words1) < len(words2):
        return word_edit_distance(words2, words1)
    
    if len(words2) == 0:
        return len(words1)
    
    previous_row = range(len(words2) + 1)
    
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_bleu(predictions: List[str], targets: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score (simplified).
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        max_n: Maximum n-gram order
        
    Returns:
        BLEU score (0.0 to 1.0)
    """
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().strip().split()
        return Counter([tuple(words[i:i+n]) for i in range(len(words) - n + 1)])
    
    precisions = []
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_pred = 0
        
        for pred, target in zip(predictions, targets):
            pred_ngrams = get_ngrams(pred, n)
            target_ngrams = get_ngrams(target, n)
            
            # Clipped counts
            for ngram, count in pred_ngrams.items():
                total_matches += min(count, target_ngrams.get(ngram, 0))
                total_pred += count
        
        precision = total_matches / max(total_pred, 1)
        precisions.append(precision)
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precision = sum(np.log(p) for p in precisions) / len(precisions)
        bleu = np.exp(log_precision)
    else:
        bleu = 0.0
    
    # Brevity penalty (simplified)
    pred_lens = sum(len(p.split()) for p in predictions)
    target_lens = sum(len(t.split()) for t in targets)
    
    if pred_lens < target_lens:
        bp = np.exp(1 - target_lens / max(pred_lens, 1))
        bleu *= bp
    
    return bleu


def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Exact match rate (0.0 to 1.0)
    """
    matches = sum(
        pred.lower().strip() == target.lower().strip()
        for pred, target in zip(predictions, targets)
    )
    return matches / max(len(predictions), 1)


def compute_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        
    Returns:
        Dictionary of metrics
    """
    return {
        'char_accuracy': compute_char_accuracy(predictions, targets),
        'cer': compute_cer(predictions, targets),
        'wer': compute_wer(predictions, targets),
        'bleu': compute_bleu(predictions, targets),
        'exact_match': compute_exact_match(predictions, targets)
    }


def print_evaluation_report(
    predictions: List[str],
    targets: List[str],
    sample_size: int = 5
):
    """
    Print detailed evaluation report.
    
    Args:
        predictions: List of predicted strings
        targets: List of target strings
        sample_size: Number of samples to show
    """
    metrics = compute_metrics(predictions, targets)
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    print("\n📊 Metrics:")
    print(f"  Character Accuracy: {metrics['char_accuracy']:.2%}")
    print(f"  Character Error Rate (CER): {metrics['cer']:.2%}")
    print(f"  Word Error Rate (WER): {metrics['wer']:.2%}")
    print(f"  BLEU Score: {metrics['bleu']:.4f}")
    print(f"  Exact Match: {metrics['exact_match']:.2%}")
    
    print(f"\n📝 Sample Predictions ({sample_size}):")
    print("-" * 60)
    
    # Show random samples
    indices = np.random.choice(len(predictions), min(sample_size, len(predictions)), replace=False)
    
    for idx in indices:
        pred = predictions[idx]
        target = targets[idx]
        match = "✓" if pred.lower().strip() == target.lower().strip() else "✗"
        
        print(f"  Target:     '{target}'")
        print(f"  Prediction: '{pred}' {match}")
        print()
    
    print("=" * 60)
    
    return metrics


class MetricTracker:
    """Track metrics over training."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'char_accuracy': [],
            'cer': [],
            'wer': [],
            'bleu': []
        }
    
    def update(self, metrics: Dict[str, float], prefix: str = ''):
        """Add metrics to history."""
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key in self.history:
                self.history[full_key].append(value)
    
    def get_best(self, metric: str, mode: str = 'min') -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        values = self.history.get(metric, [])
        if not values:
            return None, -1
        
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], best_idx + 1


if __name__ == "__main__":
    # Test metrics
    print("Evaluation Module Test")
    print("=" * 50)
    
    predictions = [
        "hello world",
        "this is a test",
        "sign language translation",
        "good morning",
        "thank you"
    ]
    
    targets = [
        "hello world",
        "this is a tset",  # typo
        "sign language transltion",  # missing 'a'
        "good afternoon",  # different word
        "thank you very much"  # extra words
    ]
    
    print_evaluation_report(predictions, targets, sample_size=5)
