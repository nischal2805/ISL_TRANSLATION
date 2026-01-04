"""
Metrics for ISL Translation Training
====================================
Includes accuracy, BLEU, WER, and visualization utilities.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
from pathlib import Path


class TranslationMetrics:
    """Comprehensive metrics for sign language translation."""
    
    def __init__(self, tokenizer, ignore_ids: List[int] = None):
        self.tokenizer = tokenizer
        self.ignore_ids = ignore_ids or [0, 101, 102]  # PAD, CLS, SEP for BERT tokenizer
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators."""
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_sequences = 0
        self.correct_sequences = 0
        self.total_edit_distance = 0
        self.total_reference_length = 0
        self.predictions = []
        self.references = []
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor):
        """
        Update metrics with a batch.
        
        Args:
            logits: (B, T, V) model output logits
            targets: (B, T) target token ids
            target_lengths: (B,) actual target lengths
        """
        predictions = logits.argmax(dim=-1)  # (B, T)
        B = targets.size(0)
        
        for i in range(B):
            length = target_lengths[i].item()
            pred = predictions[i, :length]
            tgt = targets[i, :length]
            
            # Token-level accuracy (excluding special tokens)
            mask = ~torch.isin(tgt, torch.tensor(self.ignore_ids, device=tgt.device))
            valid_pred = pred[mask]
            valid_tgt = tgt[mask]
            
            if len(valid_tgt) > 0:
                self.total_tokens += len(valid_tgt)
                self.correct_tokens += (valid_pred == valid_tgt).sum().item()
            
            # Sequence-level accuracy
            self.total_sequences += 1
            if torch.equal(pred, tgt):
                self.correct_sequences += 1
            
            # Store for BLEU/WER calculation
            pred_text = self.tokenizer.decode(pred.cpu().tolist(), skip_special_tokens=True)
            tgt_text = self.tokenizer.decode(tgt.cpu().tolist(), skip_special_tokens=True)
            self.predictions.append(pred_text)
            self.references.append(tgt_text)
            
            # Edit distance for WER
            pred_words = pred_text.split()
            ref_words = tgt_text.split()
            self.total_edit_distance += self._levenshtein(pred_words, ref_words)
            self.total_reference_length += len(ref_words)
    
    def _levenshtein(self, s1: List[str], s2: List[str]) -> int:
        """Calculate Levenshtein distance between two word sequences."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
    
    def _calculate_bleu(self, n: int = 4) -> float:
        """Calculate corpus-level BLEU score."""
        if not self.predictions:
            return 0.0
        
        def get_ngrams(words, n):
            if len(words) < n:
                return []
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        precisions = []
        for i in range(1, n + 1):
            matches = 0
            total = 0
            for pred, ref in zip(self.predictions, self.references):
                pred_words = pred.split()
                ref_words = ref.split()
                pred_ngrams = get_ngrams(pred_words, i)
                ref_ngrams = get_ngrams(ref_words, i)
                
                ref_counts = defaultdict(int)
                for ng in ref_ngrams:
                    ref_counts[ng] += 1
                
                for ng in pred_ngrams:
                    if ref_counts[ng] > 0:
                        matches += 1
                        ref_counts[ng] -= 1
                total += len(pred_ngrams)
            
            precisions.append(matches / max(total, 1))
        
        # Brevity penalty
        pred_len = sum(len(p.split()) for p in self.predictions)
        ref_len = sum(len(r.split()) for r in self.references)
        bp = np.exp(1 - ref_len / max(pred_len, 1)) if pred_len < ref_len else 1.0
        
        # Geometric mean of precisions
        if min(precisions) > 0:
            bleu = bp * np.exp(np.mean(np.log(precisions)))
        else:
            bleu = 0.0
        return bleu * 100  # Return as percentage
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        token_accuracy = self.correct_tokens / max(self.total_tokens, 1) * 100
        sequence_accuracy = self.correct_sequences / max(self.total_sequences, 1) * 100
        wer = self.total_edit_distance / max(self.total_reference_length, 1) * 100
        bleu = self._calculate_bleu()
        
        return {
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy,
            'wer': wer,
            'bleu': bleu,
            'total_tokens': self.total_tokens,
            'total_sequences': self.total_sequences
        }


class MetricsLogger:
    """Logger for training metrics with visualization."""
    
    def __init__(self, log_dir: str, experiment_name: str = "isl_translation"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Metric history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ce_loss': [], 'val_ce_loss': [],
            'train_ctc_loss': [], 'val_ctc_loss': [],
            'train_token_acc': [], 'val_token_acc': [],
            'train_seq_acc': [], 'val_seq_acc': [],
            'val_wer': [], 'val_bleu': [],
            'learning_rate': [], 'epoch': []
        }
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log metrics for an epoch."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_metrics.get('loss', 0))
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['train_ce_loss'].append(train_metrics.get('ce_loss', 0))
        self.history['val_ce_loss'].append(val_metrics.get('ce_loss', 0))
        self.history['train_ctc_loss'].append(train_metrics.get('ctc_loss', 0))
        self.history['val_ctc_loss'].append(val_metrics.get('ctc_loss', 0))
        self.history['train_token_acc'].append(train_metrics.get('token_accuracy', 0))
        self.history['val_token_acc'].append(val_metrics.get('token_accuracy', 0))
        self.history['train_seq_acc'].append(train_metrics.get('sequence_accuracy', 0))
        self.history['val_seq_acc'].append(val_metrics.get('sequence_accuracy', 0))
        self.history['val_wer'].append(val_metrics.get('wer', 0))
        self.history['val_bleu'].append(val_metrics.get('bleu', 0))
        self.history['learning_rate'].append(lr)
    
    def plot_training_curves(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate comprehensive training visualization."""
        epochs = self.history['epoch']
        if len(epochs) < 2:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.experiment_name} Training Progress', fontsize=14, fontweight='bold')
        
        # 1. Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. CE vs CTC Loss
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_ce_loss'], 'b-', label='Train CE', linewidth=2)
        ax.plot(epochs, self.history['val_ce_loss'], 'b--', label='Val CE', linewidth=2)
        ax.plot(epochs, self.history['train_ctc_loss'], 'g-', label='Train CTC', linewidth=2)
        ax.plot(epochs, self.history['val_ctc_loss'], 'g--', label='Val CTC', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('CE vs CTC Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Token Accuracy
        ax = axes[0, 2]
        ax.plot(epochs, self.history['train_token_acc'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_token_acc'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Token Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 4. Sequence Accuracy
        ax = axes[1, 0]
        ax.plot(epochs, self.history['train_seq_acc'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_seq_acc'], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Sequence Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # 5. WER and BLEU
        ax = axes[1, 1]
        ax2 = ax.twinx()
        line1, = ax.plot(epochs, self.history['val_wer'], 'r-', label='WER ↓', linewidth=2)
        line2, = ax2.plot(epochs, self.history['val_bleu'], 'g-', label='BLEU ↑', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('WER (%)', color='r')
        ax2.set_ylabel('BLEU Score', color='g')
        ax.set_title('WER & BLEU Score')
        ax.legend(handles=[line1, line2], loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 6. Learning Rate
        ax = axes[1, 2]
        ax.plot(epochs, self.history['learning_rate'], 'purple', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.log_dir / f'{self.experiment_name}_training_curves.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Training curves saved to: {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, predictions: List[str], references: List[str], 
                               top_k: int = 20, save: bool = True) -> Optional[plt.Figure]:
        """Plot most common prediction errors."""
        from collections import Counter
        
        errors = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            for pw, rw in zip(pred_words, ref_words):
                if pw != rw:
                    errors.append((rw, pw))
        
        if not errors:
            return None
        
        error_counts = Counter(errors).most_common(top_k)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        labels = [f"{ref}→{pred}" for (ref, pred), _ in error_counts]
        counts = [c for _, c in error_counts]
        
        bars = ax.barh(labels, counts, color='coral')
        ax.set_xlabel('Count')
        ax.set_title(f'Top {top_k} Word-Level Errors')
        ax.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   str(count), va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = self.log_dir / f'{self.experiment_name}_errors.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Error analysis saved to: {save_path}")
        
        return fig
    
    def save_metrics_csv(self):
        """Save metrics history to CSV."""
        import csv
        save_path = self.log_dir / f'{self.experiment_name}_metrics.csv'
        
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            for i in range(len(self.history['epoch'])):
                row = {k: v[i] for k, v in self.history.items()}
                writer.writerow(row)
        
        print(f"📄 Metrics saved to: {save_path}")
    
    def print_summary(self, epoch: int):
        """Print formatted summary of current epoch."""
        idx = -1  # Last entry
        print("\n" + "="*70)
        print(f"📈 EPOCH {epoch} SUMMARY")
        print("="*70)
        print(f"{'Metric':<25} {'Train':>15} {'Validation':>15}")
        print("-"*70)
        print(f"{'Loss':<25} {self.history['train_loss'][idx]:>15.4f} {self.history['val_loss'][idx]:>15.4f}")
        print(f"{'CE Loss':<25} {self.history['train_ce_loss'][idx]:>15.4f} {self.history['val_ce_loss'][idx]:>15.4f}")
        print(f"{'CTC Loss':<25} {self.history['train_ctc_loss'][idx]:>15.4f} {self.history['val_ctc_loss'][idx]:>15.4f}")
        print(f"{'Token Accuracy (%)':<25} {self.history['train_token_acc'][idx]:>15.2f} {self.history['val_token_acc'][idx]:>15.2f}")
        print(f"{'Sequence Accuracy (%)':<25} {self.history['train_seq_acc'][idx]:>15.2f} {self.history['val_seq_acc'][idx]:>15.2f}")
        print(f"{'WER (%)':<25} {'-':>15} {self.history['val_wer'][idx]:>15.2f}")
        print(f"{'BLEU':<25} {'-':>15} {self.history['val_bleu'][idx]:>15.2f}")
        print(f"{'Learning Rate':<25} {self.history['learning_rate'][idx]:>15.2e}")
        print("="*70 + "\n")


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k token accuracy."""
    topk_preds = logits.topk(k, dim=-1).indices  # (B, T, k)
    targets_expanded = targets.unsqueeze(-1).expand_as(topk_preds)  # (B, T, k)
    correct = (topk_preds == targets_expanded).any(dim=-1)  # (B, T)
    mask = targets != 0  # Ignore padding
    return (correct & mask).sum().float() / mask.sum().float() * 100
