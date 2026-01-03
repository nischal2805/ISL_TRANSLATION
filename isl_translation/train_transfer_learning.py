"""
Training Script for ISL Translation with Transfer Learning
==========================================================

Features:
1. Pretrained VideoMAE encoder (frozen initially, then fine-tuned)
2. Hybrid CTC-Attention loss
3. Mixed precision training
4. Learning rate scheduling with warmup
5. Knowledge distillation ready
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_transfer_learning import ISLConfig, ISLTranslationModel, ISLLoss
from dataset_consolidated import ISLDatasetConsolidated, collate_fn_consolidated


class TransferLearningTrainer:
    """
    Trainer with transfer learning strategy:
    1. Epochs 0-5: Freeze encoder, train decoder only
    2. Epochs 5+: Unfreeze encoder, fine-tune with lower LR
    """
    
    def __init__(
        self,
        model: ISLTranslationModel,
        loss_fn: ISLLoss,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
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
        
        # Optimizer with parameter groups (different LR for encoder/decoder)
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config['encoder_lr']},  # Lower LR for pretrained
            {'params': decoder_params, 'lr': config['decoder_lr']}   # Higher LR for decoder
        ], weight_decay=config.get('weight_decay', 0.01))
        
        # Scheduler
        warmup_steps = config.get('warmup_steps', 2000)
        self.scheduler = self._create_scheduler(warmup_steps)
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.use_amp = config.get('use_amp', True)
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation', 1)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints_transfer'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.log_dir = Path(config.get('log_dir', 'logs_transfer'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
    def _create_scheduler(self, warmup_steps: int):
        """Warmup + cosine decay."""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + torch.cos(torch.tensor(
                (step - warmup_steps) / (self.config['max_steps'] - warmup_steps) * 3.14159
            )).item())
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        # Notify model of epoch (for encoder unfreezing)
        self.model.on_epoch_start(self.epoch)
        
        total_loss = 0
        total_ce_loss = 0
        total_ctc_loss = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            # Forward with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(features, feature_lengths, targets, target_lengths)
                    losses = self.loss_fn(
                        outputs, targets, target_lengths, 
                        outputs['encoder_lengths']
                    )
                    loss = losses['loss'] / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(features, feature_lengths, targets, target_lengths)
                losses = self.loss_fn(
                    outputs, targets, target_lengths,
                    outputs['encoder_lengths']
                )
                loss = losses['loss'] / self.grad_accum_steps
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track losses
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            num_batches += 1
            
            # Log every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                lr_enc = self.optimizer.param_groups[0]['lr']
                lr_dec = self.optimizer.param_groups[1]['lr']
                
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: enc={lr_enc:.2e}, dec={lr_dec:.2e}")
                
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
        
        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        
        total_loss = 0
        total_ce_loss = 0
        total_ctc_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            outputs = self.model(features, feature_lengths, targets, target_lengths)
            losses = self.loss_fn(
                outputs, targets, target_lengths,
                outputs['encoder_lengths']
            )
            
            total_loss += losses['loss'].item()
            total_ce_loss += losses['ce_loss'].item()
            total_ctc_loss += losses['ctc_loss'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'ctc_loss': total_ctc_loss / num_batches
        }
    
    @torch.no_grad()
    def sample_predictions(self, num_samples: int = 3):
        """Generate sample predictions."""
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        features = batch['features'][:num_samples].to(self.device)
        feature_lengths = batch['feature_lengths'][:num_samples].to(self.device)
        texts = batch['texts'][:num_samples]
        
        predictions = self.model.translate(features, feature_lengths)
        
        print("\n" + "="*50)
        print("Sample Predictions:")
        print("="*50)
        
        for i in range(num_samples):
            pred_ids = predictions[i].cpu().tolist()
            # Remove special tokens and decode
            pred_ids = [t for t in pred_ids if t not in [0, 1, 2, 3]]
            pred_text = self.tokenizer.decode(pred_ids)
            
            print(f"\nTarget:     {texts[i]}")
            print(f"Prediction: {pred_text}")
        
        print("="*50 + "\n")
    
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
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model!")
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print("="*60)
        print("TRANSFER LEARNING TRAINING")
        print("="*60)
        
        params = self.model.count_parameters()
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,}")
        print(f"Encoder frozen: {self.model.encoder._frozen}")
        print(f"Unfreeze at epoch: {self.model.config.freeze_encoder_epochs}")
        print("="*60)
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-"*40)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, CTC: {train_metrics['ctc_loss']:.4f})")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} "
                  f"(CE: {val_metrics['ce_loss']:.4f}, CTC: {val_metrics['ctc_loss']:.4f})")
            
            # Log to tensorboard
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            
            # Sample predictions every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.sample_predictions()
            
            # Check for improvement
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train ISL Translation with Transfer Learning')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory with train/val/test .npy files')
    parser.add_argument('--tokenizer', type=str, default='facebook/mbart-large-cc25',
                       help='HuggingFace tokenizer')
    
    # Model
    parser.add_argument('--encoder', type=str, default='MCG-NJU/videomae-base',
                       help='Pretrained encoder')
    parser.add_argument('--freeze-epochs', type=int, default=5,
                       help='Epochs to freeze encoder')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--decoder-lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_transfer')
    parser.add_argument('--log-dir', type=str, default='logs_transfer')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create model config
    config = ISLConfig(
        encoder_name=args.encoder,
        freeze_encoder_epochs=args.freeze_epochs,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id or 1,
        bos_id=tokenizer.bos_token_id or 2,
        eos_id=tokenizer.eos_token_id or 3
    )
    
    # Create model
    print("\nCreating model with transfer learning...")
    model = ISLTranslationModel(config)
    
    params = model.count_parameters()
    print(f"Parameters: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")
    
    # Create loss
    loss_fn = ISLLoss(config)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = ISLDatasetConsolidated(args.data_dir, tokenizer, 'train')
    val_dataset = ISLDatasetConsolidated(args.data_dir, tokenizer, 'val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn_consolidated,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn_consolidated,
        pin_memory=True
    )
    
    # Calculate max steps
    steps_per_epoch = len(train_loader) // args.gradient_accumulation
    max_steps = steps_per_epoch * args.epochs
    
    # Training config
    train_config = {
        'encoder_lr': args.encoder_lr,
        'decoder_lr': args.decoder_lr,
        'warmup_steps': args.warmup_steps,
        'max_steps': max_steps,
        'gradient_accumulation': args.gradient_accumulation,
        'max_grad_norm': 1.0,
        'use_amp': True,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir
    }
    
    # Create trainer
    trainer = TransferLearningTrainer(
        model=model,
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
