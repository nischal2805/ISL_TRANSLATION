"""
ISL Translation System - Training Script
=========================================
Hybrid CTC + CrossEntropy training with NaN-safe loss computation.

Key NaN safety features:
1. CTC length validation: encoder_length >= target_length required
2. Per-sample CTC loss with invalid sample filtering
3. Label smoothing with clamped log_softmax
4. Batch skipping on NaN/Inf detection
5. Gradient clipping before backward pass
6. Loss fallback when CTC fails
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    training_config, model_config, data_config, vocab_config,
    set_gpu_mode, DEVICE
)
from vocab import Vocabulary
from dataset import create_dataloaders, AugmentedDataset, collate_fn
from model import create_model, ISLTranslationModel
from evaluate import compute_metrics, compute_wer


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('ISLTraining')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# NaN-Safe Loss Functions
# ============================================================================

class NaNSafeLabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss with NaN-safe log_softmax.
    
    Prevents -inf in log probabilities by clamping logits before softmax.
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L, V) decoder logits
            targets: (B, L) target token IDs
            
        Returns:
            Scalar loss
        """
        B, L, V = logits.shape
        
        # Clamp logits to prevent extreme values
        logits = logits.clamp(min=-100, max=100)
        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-100)
        
        # Flatten for computation
        log_probs = log_probs.view(-1, V)  # (B*L, V)
        targets = targets.view(-1)  # (B*L,)
        
        # Create mask for valid positions
        mask = (targets != self.ignore_index).float()
        
        # NLL loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        
        # Smoothed loss
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combined loss
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        
        return loss


class NaNSafeHybridLoss(nn.Module):
    """
    Hybrid CTC + CrossEntropy loss with comprehensive NaN safety.
    
    Key safety features:
    1. Validates CTC length constraint: encoder_length >= target_length
    2. Uses per-sample CTC loss to filter invalid samples
    3. Falls back to CE-only loss when CTC fails
    4. Clamps all log probabilities
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        blank_id: int = 0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Use reduction='none' for per-sample filtering
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id, 
            reduction='none',  # Per-sample loss
            zero_infinity=True  # Map inf to zero
        )
        
        self.ce_loss = NaNSafeLabelSmoothingLoss(
            smoothing=label_smoothing,
            ignore_index=pad_id
        )
    
    def forward(
        self,
        ctc_logits: torch.Tensor,
        decoder_logits: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_weight: float = 0.3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss with NaN safety.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss values + diagnostics
        """
        device = ctc_logits.device
        batch_size = ctc_logits.size(0)
        
        # ============================================================
        # CTC Loss with length validation
        # ============================================================
        
        # CTC format: (T, B, V)
        ctc_input = ctc_logits.permute(1, 0, 2)
        
        # Prepare CTC targets (remove SOS token)
        ctc_targets = targets[:, 1:]  # Remove SOS
        ctc_target_lengths = target_lengths - 2  # Subtract SOS and EOS
        ctc_target_lengths = ctc_target_lengths.clamp(min=1)
        
        # CRITICAL: CTC requires encoder_length >= target_length
        # Filter samples that violate this constraint
        valid_ctc_mask = encoder_lengths >= ctc_target_lengths
        num_valid_ctc = valid_ctc_mask.sum().item()
        
        ctc_l = torch.tensor(0.0, device=device)
        ctc_valid = False
        
        if num_valid_ctc > 0:
            # Compute per-sample CTC loss
            try:
                per_sample_ctc = self.ctc_loss(
                    ctc_input,
                    ctc_targets,
                    encoder_lengths,
                    ctc_target_lengths
                )
                
                # Replace inf/nan with 0 in invalid samples
                per_sample_ctc = torch.where(
                    valid_ctc_mask & torch.isfinite(per_sample_ctc),
                    per_sample_ctc,
                    torch.zeros_like(per_sample_ctc)
                )
                
                # Average over valid samples only
                if valid_ctc_mask.sum() > 0:
                    ctc_l = per_sample_ctc.sum() / valid_ctc_mask.sum().clamp(min=1)
                    ctc_valid = torch.isfinite(ctc_l).item()
                
            except RuntimeError as e:
                # CTC can throw on edge cases
                ctc_valid = False
        
        # ============================================================
        # CrossEntropy Loss (always computed)
        # ============================================================
        
        ce_l = self.ce_loss(decoder_logits, targets)
        ce_valid = torch.isfinite(ce_l).item()
        
        # ============================================================
        # Combine losses with fallback
        # ============================================================
        
        if ctc_valid and ce_valid:
            # Normal case: both losses valid
            total_loss = ctc_weight * ctc_l + (1 - ctc_weight) * ce_l
        elif ce_valid:
            # Fallback: use CE only
            total_loss = ce_l
            ctc_weight = 0.0  # For logging
        else:
            # Both failed - this should be skipped
            total_loss = torch.tensor(float('nan'), device=device)
        
        loss_dict = {
            'total_loss': total_loss.item() if torch.isfinite(total_loss) else float('nan'),
            'ctc_loss': ctc_l.item() if torch.isfinite(ctc_l) else 0.0,
            'ce_loss': ce_l.item() if torch.isfinite(ce_l) else 0.0,
            'ctc_valid_ratio': num_valid_ctc / batch_size,
            'ctc_weight_used': ctc_weight
        }
        
        return total_loss, loss_dict


# ============================================================================
# Training Utilities
# ============================================================================

class ScheduledValue:
    """Linearly scheduled value (e.g., for CTC weight, teacher forcing)."""
    
    def __init__(self, start: float, end: float, num_steps: int):
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.step = 0
    
    def get(self) -> float:
        progress = min(self.step / max(self.num_steps, 1), 1.0)
        return self.start + (self.end - self.start) * progress
    
    def update(self):
        self.step += 1


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    path: str,
    extra_info: Optional[Dict] = None
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> int:
    """Load training checkpoint. Returns epoch number."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch']


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        return False
    if torch.isinf(tensor).any():
        return False
    return True


# ============================================================================
# NaN-Safe Trainer
# ============================================================================

class NaNSafeTrainer:
    """
    Trainer with comprehensive NaN/Inf handling.
    
    Features:
    1. Input validation before forward pass
    2. Loss validation after forward pass  
    3. Gradient clipping before backward
    4. Batch skipping on failure
    5. Gradient accumulation with overflow detection
    """
    
    def __init__(
        self,
        model: ISLTranslationModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        vocab: Vocabulary,
        config: dict,
        log_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        self.log_dir = log_dir
        
        # Setup logging
        self.logger = setup_logging(log_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboard'))
        
        # Device
        self.device = DEVICE
        self.model = self.model.to(self.device)
        
        # Loss function
        self.loss_fn = NaNSafeHybridLoss(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            blank_id=vocab.pad_id,  # Use PAD as CTC blank
            label_smoothing=0.1
        )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            eps=1e-8
        )
        
        # Schedulers - Warmup + Cosine Annealing using LambdaLR for proper sync
        total_steps = len(train_loader) * training_config.num_epochs
        warmup_steps = len(train_loader) * training_config.warmup_epochs
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = training_config.learning_rate
        self.min_lr = training_config.min_lr
        
        # Combined warmup + cosine schedule using LambdaLR (Bug #6 fix)
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return (step + 1) / warmup_steps
            else:
                # Cosine annealing after warmup
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return self.min_lr / self.base_lr + (1 - self.min_lr / self.base_lr) * cosine_decay
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.scaler = GradScaler(enabled=training_config.use_amp)
        
        # Scheduled values
        self.ctc_weight_scheduler = ScheduledValue(
            start=0.5,  # Start with equal weight
            end=0.2,    # End with more emphasis on decoder
            num_steps=total_steps
        )
        
        self.tf_ratio_scheduler = ScheduledValue(
            start=1.0,  # Full teacher forcing at start
            end=0.7,    # Reduce to 70% by end
            num_steps=total_steps
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.early_stopping_patience
        )
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.nan_batches = 0
        self.total_batches = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with NaN safety."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'ctc_loss': 0.0,
            'ce_loss': 0.0
        }
        num_batches = 0
        skipped_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            self.total_batches += 1
            
            # ============================================================
            # 1. Unpack batch and move to device
            # ============================================================
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # ============================================================
            # 2. Input validation
            # ============================================================
            if not check_tensor_health(features, "features"):
                self.logger.warning(f"Batch {batch_idx}: NaN/Inf in input features, skipping")
                skipped_batches += 1
                self.nan_batches += 1
                continue
            
            # ============================================================
            # 3. Forward pass with mixed precision
            # ============================================================
            self.optimizer.zero_grad()
            
            try:
                with autocast(enabled=training_config.use_amp):
                    # Get current scheduled values
                    ctc_weight = self.ctc_weight_scheduler.get()
                    tf_ratio = self.tf_ratio_scheduler.get()
                    
                    # Forward pass
                    outputs = self.model(
                        features,
                        feature_lengths,
                        targets,
                        teacher_forcing_ratio=tf_ratio
                    )
                    
                    ctc_logits = outputs['ctc_logits']
                    decoder_logits = outputs['decoder_logits']
                    encoder_lengths = outputs['encoder_lengths']
                    
                    # Compute loss
                    loss, loss_dict = self.loss_fn(
                        ctc_logits=ctc_logits,
                        decoder_logits=decoder_logits,
                        targets=targets,
                        encoder_lengths=encoder_lengths,
                        target_lengths=target_lengths,
                        ctc_weight=ctc_weight
                    )
            
            except RuntimeError as e:
                self.logger.warning(f"Batch {batch_idx}: Forward pass error: {e}")
                skipped_batches += 1
                self.nan_batches += 1
                continue
            
            # ============================================================
            # 4. Loss validation
            # ============================================================
            if not torch.isfinite(loss):
                self.logger.warning(
                    f"Batch {batch_idx}: NaN/Inf loss detected "
                    f"(CTC valid: {loss_dict.get('ctc_valid_ratio', 0):.2%}), skipping"
                )
                skipped_batches += 1
                self.nan_batches += 1
                self.optimizer.zero_grad()
                continue
            
            # ============================================================
            # 5. Backward pass with gradient scaling
            # ============================================================
            self.scaler.scale(loss).backward()
            
            # Gradient clipping BEFORE unscale
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=training_config.max_grad_norm
            )
            
            # Check gradient health
            if not torch.isfinite(grad_norm):
                self.logger.warning(f"Batch {batch_idx}: Non-finite gradient norm, skipping update")
                self.optimizer.zero_grad()
                skipped_batches += 1
                continue
            
            # ============================================================
            # 6. Optimizer step
            # ============================================================
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Learning rate scheduling - LambdaLR handles warmup + cosine internally
            self.scheduler.step()
            
            # Update schedulers
            self.ctc_weight_scheduler.update()
            self.tf_ratio_scheduler.update()
            self.global_step += 1
            
            # ============================================================
            # 7. Logging
            # ============================================================
            epoch_losses['total_loss'] += loss_dict['total_loss']
            epoch_losses['ctc_loss'] += loss_dict['ctc_loss']
            epoch_losses['ce_loss'] += loss_dict['ce_loss']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'ctc': f"{loss_dict['ctc_loss']:.4f}",
                'ce': f"{loss_dict['ce_loss']:.4f}",
                'skip': skipped_batches
            })
            
            # TensorBoard logging
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/total_loss', loss_dict['total_loss'], self.global_step)
                self.writer.add_scalar('train/ctc_loss', loss_dict['ctc_loss'], self.global_step)
                self.writer.add_scalar('train/ce_loss', loss_dict['ce_loss'], self.global_step)
                self.writer.add_scalar('train/ctc_weight', ctc_weight, self.global_step)
                self.writer.add_scalar('train/tf_ratio', tf_ratio, self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                self.writer.add_scalar('train/ctc_valid_ratio', loss_dict.get('ctc_valid_ratio', 0), self.global_step)
        
        # Compute epoch averages
        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        
        epoch_losses['skipped_batches'] = skipped_batches
        epoch_losses['nan_rate'] = self.nan_batches / max(self.total_batches, 1)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'ctc_loss': 0.0,
            'ce_loss': 0.0
        }
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Skip batches with NaN input
            if not check_tensor_health(features, "features"):
                continue
            
            # Forward pass with teacher forcing=1.0 for consistent loss computation
            outputs = self.model(
                features,
                feature_lengths,
                targets,
                teacher_forcing_ratio=1.0  # Use teacher forcing for loss computation
            )
            
            ctc_logits = outputs['ctc_logits']
            decoder_logits = outputs['decoder_logits']
            encoder_lengths = outputs['encoder_lengths']
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                ctc_logits=ctc_logits,
                decoder_logits=decoder_logits,
                targets=targets,
                encoder_lengths=encoder_lengths,
                target_lengths=target_lengths,
                ctc_weight=0.3
            )
            
            if torch.isfinite(loss):
                val_losses['total_loss'] += loss_dict['total_loss']
                val_losses['ctc_loss'] += loss_dict['ctc_loss']
                val_losses['ce_loss'] += loss_dict['ce_loss']
                num_batches += 1
            
            # Use proper greedy decoding for WER computation (Bug #4 fix)
            # This gives a true measure of model's autoregressive generation ability
            predicted_ids, _ = self.model.decode(
                features, 
                feature_lengths, 
                max_len=targets.size(1)
            )
            
            for i in range(predicted_ids.size(0)):
                pred_tokens = predicted_ids[i].tolist()
                ref_tokens = targets[i].tolist()
                
                # Decode to text
                pred_text = self.vocab.decode(pred_tokens)
                ref_text = self.vocab.decode(ref_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
        
        # Compute averages
        if num_batches > 0:
            for key in val_losses:
                val_losses[key] /= num_batches
        
        # Compute WER - compute_wer expects lists of strings
        if len(all_predictions) > 0:
            val_losses['wer'] = compute_wer(all_predictions, all_references)
        else:
            val_losses['wer'] = 1.0
        
        return val_losses
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from and os.path.exists(resume_from):
            start_epoch = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler
            )
            self.logger.info(f"Resumed from epoch {start_epoch}")
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Training
            train_losses = self.train_epoch(epoch + 1)
            
            # Validation
            val_losses = self.validate(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_losses['total_loss']:.4f} | "
                f"Val Loss: {val_losses['total_loss']:.4f} | "
                f"Val WER: {val_losses['wer']:.4f} | "
                f"Skipped: {train_losses['skipped_batches']} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard
            self.writer.add_scalar('val/total_loss', val_losses['total_loss'], epoch + 1)
            self.writer.add_scalar('val/wer', val_losses['wer'], epoch + 1)
            self.writer.add_scalar('train/nan_rate', train_losses['nan_rate'], epoch + 1)
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_losses['total_loss'],
                    os.path.join(self.log_dir, 'best_model.pt'),
                    extra_info={'wer': val_losses['wer']}
                )
                self.logger.info(f"Saved best model with val_loss={val_losses['total_loss']:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % training_config.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_losses['total_loss'],
                    os.path.join(self.log_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                )
            
            # Early stopping
            if self.early_stopping(val_losses['total_loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Final save
        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            num_epochs, val_losses['total_loss'],
            os.path.join(self.log_dir, 'final_model.pt')
        )
        
        self.writer.close()
        self.logger.info("Training completed!")
        self.logger.info(f"Total NaN batches: {self.nan_batches}/{self.total_batches} ({self.nan_batches/max(self.total_batches,1):.2%})")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ISL Translation Model')
    parser.add_argument('--gpu-mode', type=str, choices=['small', 'large'],
                        default='small', help='GPU mode (small=RTX4060, large=A100)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default='runs',
                        help='Directory for logs and checkpoints')
    args = parser.parse_args()
    
    # Set GPU mode
    set_gpu_mode(args.gpu_mode)
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'{args.gpu_mode}_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    config_dict = {
        'gpu_mode': args.gpu_mode,
        'model_config': {
            'input_dim': model_config.input_dim,
            'd_model': model_config.d_model,
            'num_conformer_blocks': model_config.num_conformer_blocks,
            'num_heads': model_config.num_heads,
            'dropout': model_config.dropout
        },
        'training_config': {
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'num_epochs': training_config.num_epochs,
            'use_amp': training_config.use_amp
        }
    }
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Load vocabulary
    vocab = Vocabulary()
    print(f"Vocabulary size: {vocab.size}")
    
    # Create dataloaders
    metadata_path = os.path.join(data_config.processed_dir, 'metadata.csv')
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=metadata_path,
        features_dir=data_config.processed_dir,
        vocab=vocab,
        batch_size=training_config.batch_size,
        num_workers=4,  # Use fixed value or add to config
        use_augmentation=True  # Enable data augmentation for training
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    model = create_model()  # Uses model_config by default
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and train
    trainer = NaNSafeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        config=config_dict,
        log_dir=log_dir
    )
    
    trainer.train(
        num_epochs=training_config.num_epochs,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
