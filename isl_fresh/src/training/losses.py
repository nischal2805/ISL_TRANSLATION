"""Training losses for ISL Translation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """Combined CTC + Cross-Entropy loss."""
    
    def __init__(self, vocab_size: int, pad_id: int = 0, ctc_weight: float = 0.3, label_smoothing: float = 0.1):
        super().__init__()
        self.ctc_weight = ctc_weight
        self.ce_weight = 1.0 - ctc_weight
        self.pad_id = pad_id
        
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    
    def forward(self, outputs, targets, target_lengths, encoder_lengths):
        """
        Args:
            outputs: dict with 'logits' and 'ctc_logits'
            targets: (B, T) target token ids
            target_lengths: (B,) actual target lengths
            encoder_lengths: (B,) encoder output lengths
        """
        # Cross-entropy loss
        logits = outputs['logits']
        B, T, V = logits.shape
        ce_loss = self.ce_loss(logits.view(-1, V), targets.view(-1))
        
        # CTC loss
        ctc_loss = torch.tensor(0.0, device=logits.device)
        if 'ctc_logits' in outputs and self.ctc_weight > 0:
            ctc_logits = outputs['ctc_logits']  # (B, T', V)
            ctc_logits = ctc_logits.transpose(0, 1)  # (T', B, V)
            ctc_logits = F.log_softmax(ctc_logits, dim=-1)
            
            # CTC expects targets without padding
            ctc_loss = self.ctc_loss(
                ctc_logits,
                targets,
                encoder_lengths,
                target_lengths
            )
        
        total_loss = self.ce_weight * ce_loss + self.ctc_weight * ctc_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss,
            'ctc_loss': ctc_loss
        }
