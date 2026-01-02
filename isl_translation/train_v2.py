"""
Training Script for ISL Translation V2
======================================
Hybrid CTC-Attention training with BPE tokenizer.
Optimized for A100 GPU (40GB VRAM).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v2 import ModelConfig, create_model_v2, HybridCTCAttentionLoss
from dataset_v2 import create_dataloaders_v2
from tokenizer import BPETokenizer


class TrainerV2:
    """
    Trainer for ISL Translation V2.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpoint management
    - TensorBoard logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        tokenizer: BPETokenizer,
        train_loader,
        val_loader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.98)
        )
        
        # Scheduler - Noam scheduler (warmup then decay)
        warmup_steps = config.get('warmup_steps', 4000)
        self.scheduler = self._create_noam_scheduler(warmup_steps)
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.use_amp = config.get('use_amp', True)
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints_v2'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_checkpoints = config.get('keep_n_checkpoints', 5)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs_v2'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 10)
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
    
    def _create_noam_scheduler(self, warmup_steps: int):
        """Create Noam learning rate scheduler."""
        def lr_lambda(step):
            step = max(step, 1)
            return min(step ** -0.5, step * warmup_steps ** -1.5)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_ctc_loss = 0
        total_ce_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(features, feature_lengths, targets, target_lengths)
                    losses = self.loss_fn(
                        outputs['ctc_log_probs'],
                        outputs['decoder_logits'],
                        outputs['encoder_lengths'],
                        targets,
                        target_lengths
                    )
                    loss = losses['loss'] / self.grad_accum_steps
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(features, feature_lengths, targets, target_lengths)
                losses = self.loss_fn(
                    outputs['ctc_log_probs'],
                    outputs['decoder_logits'],
                    outputs['encoder_lengths'],
                    targets,
                    target_lengths
                )
                loss = losses['loss'] / self.grad_accum_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            total_loss += losses['loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            num_batches += 1
            
            # Log every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/lr', lr, self.global_step)
        
        return {
            'loss': total_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_ctc_loss = 0
        total_ce_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            outputs = self.model(features, feature_lengths, targets, target_lengths)
            losses = self.loss_fn(
                outputs['ctc_log_probs'],
                outputs['decoder_logits'],
                outputs['encoder_lengths'],
                targets,
                target_lengths
            )
            
            total_loss += losses['loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches
        }
    
    @torch.no_grad()
    def sample_predictions(self, num_samples: int = 3):
        """Generate sample predictions for monitoring."""
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        features = batch['features'][:num_samples].to(self.device)
        feature_lengths = batch['feature_lengths'][:num_samples].to(self.device)
        texts = batch['texts'][:num_samples]
        
        # Decode with attention
        predictions, _ = self.model.decode_attention(features, feature_lengths, beam_size=1)
        
        print("\n" + "=" * 50)
        print("Sample Predictions:")
        print("=" * 50)
        
        for i in range(min(num_samples, len(texts))):
            pred_ids = predictions[i].cpu().tolist()
            pred_text = self.tokenizer.decode(pred_ids)
            
            print(f"\nTarget:     {texts[i]}")
            print(f"Prediction: {pred_text}")
        
        print("=" * 50 + "\n")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.keep_n_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            old_checkpoint.unlink()
            print(f"Removed old checkpoint: {old_checkpoint}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}")
        print("=" * 60)
        
        start_epoch = self.epoch
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(CTC: {train_metrics['ctc_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                  f"(CTC: {val_metrics['ctc_loss']:.4f}, CE: {val_metrics['ce_loss']:.4f})")
            print(f"  Time: {epoch_time:.1f}s")
            
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/ctc_loss', val_metrics['ctc_loss'], epoch)
            self.writer.add_scalar('val/ce_loss', val_metrics['ce_loss'], epoch)
            
            # Sample predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.sample_predictions()
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train ISL Translation V2')
    
    # Data paths - Server paths for A100 GPU
    parser.add_argument('--train-dir', type=str, 
                       default='/media/rvcse22/CSERV/kortex_sem5/data/train/train',
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str,
                       default='/media/rvcse22/CSERV/kortex_sem5/data/val',
                       help='Validation data directory')
    parser.add_argument('--test-dir', type=str,
                       default='/media/rvcse22/CSERV/kortex_sem5/data/test/test',
                       help='Test data directory')
    parser.add_argument('--tokenizer-dir', type=str,
                       default='/media/rvcse22/CSERV/kortex_sem5/nischal/isl_translation/tokenizer_model',
                       help='Directory with trained tokenizer')
    parser.add_argument('--annotations', type=str,
                       default='/media/rvcse22/CSERV/kortex_sem5/data/iSign_v1.1.csv',
                       help='Annotations CSV')
    
    # Model config
    parser.add_argument('--vocab-size', type=int, default=2000)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--encoder-layers', type=int, default=4)
    parser.add_argument('--decoder-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--ctc-weight', type=float, default=0.3)
    
    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-steps', type=int, default=4000)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=15)
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_v2')
    parser.add_argument('--log-dir', type=str, default='logs_v2')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load or train tokenizer
    tokenizer_model_path = os.path.join(args.tokenizer_dir, 'bpe_tokenizer.model')
    
    if os.path.exists(tokenizer_model_path):
        print(f"\nLoading tokenizer from {args.tokenizer_dir}")
        tokenizer = BPETokenizer(model_path=tokenizer_model_path, vocab_size=args.vocab_size)
    else:
        print(f"\nTraining tokenizer...")
        from tokenizer import train_tokenizer_on_dataset
        tokenizer = train_tokenizer_on_dataset(
            args.annotations,
            args.tokenizer_dir,
            vocab_size=args.vocab_size
        )
    
    print(f"Tokenizer vocab size: {tokenizer.size}")
    
    # Create model config
    model_config = ModelConfig(
        vocab_size=tokenizer.size,
        d_model=args.d_model,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        ctc_weight=args.ctc_weight
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model_v2(model_config)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create loss function
    loss_fn = HybridCTCAttentionLoss(
        vocab_size=model_config.vocab_size,
        pad_id=model_config.pad_id,
        blank_id=model_config.blank_id,
        ctc_weight=model_config.ctc_weight,
        label_smoothing=model_config.label_smoothing
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    from dataset_v2 import create_simple_dataloaders
    import pandas as pd
    
    # Load annotations for text labels
    df = pd.read_csv(args.annotations)
    uid_to_text = dict(zip(df['uid'].astype(str), df['text']))
    
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        uid_to_text=uid_to_text,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Training config
    train_config = {
        'lr': args.lr,
        'warmup_steps': args.warmup_steps,
        'gradient_accumulation': args.gradient_accumulation,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'keep_n_checkpoints': 5,
        'patience': args.patience
    }
    
    # Create trainer
    trainer = TrainerV2(
        model=model,
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(args.epochs)
    
    # Save tokenizer with model
    tokenizer.save(args.checkpoint_dir)
    print(f"\nTokenizer saved to {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
