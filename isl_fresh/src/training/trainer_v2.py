"""
Production Training Script for ISL Translation
==============================================
Optimized for A100 GPU with comprehensive metrics and visualization.
"""

import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import ISLDataset, collate_fn
from src.models.translator import ISLTranslator
from src.training.losses import HybridLoss
from src.training.metrics import TranslationMetrics, MetricsLogger, compute_topk_accuracy


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine decay scheduler."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / max(1, self.warmup_steps) for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


class Trainer:
    """Production trainer with comprehensive metrics and A100 optimization."""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model.to(self.device)
        
        # Separate LR for encoder/decoder (lower LR for pretrained encoder)
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        ctc_params = list(model.ctc_head.parameters()) if hasattr(model, 'ctc_head') else []
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-5)},
            {'params': decoder_params + ctc_params, 'lr': config.get('decoder_lr', 3e-4)}
        ], weight_decay=config.get('weight_decay', 0.01), betas=(0.9, 0.98))
        
        # Calculate total steps for scheduler
        steps_per_epoch = len(train_loader) // config.get('gradient_accumulation', 1)
        total_steps = steps_per_epoch * config.get('num_epochs', 50)
        
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, 
            config.get('warmup_steps', 2000), 
            total_steps,
            min_lr=config.get('min_lr', 1e-7)
        )
        
        self.loss_fn = HybridLoss(
            model.vocab_size, 
            model.pad_id, 
            config.get('ctc_weight', 0.3), 
            config.get('label_smoothing', 0.1)
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.use_amp = config.get('use_amp', True)
        self.grad_accum = config.get('gradient_accumulation', 1)
        
        # Early stopping
        self.best_loss = float('inf')
        self.best_bleu = 0.0
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(config.get('log_dir', 'logs')) / timestamp
        self.writer = SummaryWriter(log_dir)
        
        # Metrics
        self.train_metrics = TranslationMetrics(tokenizer)
        self.val_metrics = TranslationMetrics(tokenizer)
        self.metrics_logger = MetricsLogger(log_dir, experiment_name=config.get('experiment_name', 'isl_fresh'))
        
        self.epoch = 0
        self.global_step = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch with metrics tracking."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss, total_ce, total_ctc = 0, 0, 0
        num_batches = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            features = batch['features'].to(self.device, non_blocking=True)
            feature_lengths = batch['feature_lengths'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            target_lengths = batch['target_lengths'].to(self.device, non_blocking=True)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(features, feature_lengths, targets, target_lengths)
                    losses = self.loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
                    loss = losses['loss'] / self.grad_accum
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(features, feature_lengths, targets, target_lengths)
                losses = self.loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
                loss = losses['loss'] / self.grad_accum
                loss.backward()
            
            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(outputs['logits'].detach(), targets, target_lengths)
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('max_grad_norm', 1.0)
                )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log to TensorBoard every 100 steps
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train/loss', losses['loss'].item(), self.global_step)
                    self.writer.add_scalar('train/ce_loss', losses['ce_loss'].item(), self.global_step)
                    self.writer.add_scalar('train/ctc_loss', losses['ctc_loss'].item(), self.global_step)
                    self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[1]['lr'], self.global_step)
            
            # Accumulate losses
            total_loss += losses['loss'].item()
            total_ce += losses['ce_loss'].item()
            total_ctc += losses['ctc_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[1]['lr']:.2e}"
            })
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics.update({
            'loss': total_loss / num_batches,
            'ce_loss': total_ce / num_batches,
            'ctc_loss': total_ctc / num_batches
        })
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate with comprehensive metrics."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss, total_ce, total_ctc = 0, 0, 0
        total_topk_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc="Validating")
        for batch in pbar:
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            outputs = self.model(features, feature_lengths, targets, target_lengths)
            losses = self.loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
            
            # Update metrics
            self.val_metrics.update(outputs['logits'], targets, target_lengths)
            
            # Top-5 accuracy
            topk_acc = compute_topk_accuracy(outputs['logits'], targets, k=5)
            total_topk_acc += topk_acc.item()
            
            total_loss += losses['loss'].item()
            total_ce += losses['ce_loss'].item()
            total_ctc += losses['ctc_loss'].item()
            num_batches += 1
        
        # Compute all metrics
        metrics = self.val_metrics.compute()
        metrics.update({
            'loss': total_loss / num_batches,
            'ce_loss': total_ce / num_batches,
            'ctc_loss': total_ctc / num_batches,
            'top5_accuracy': total_topk_acc / num_batches
        })
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', metrics['loss'], self.epoch)
        self.writer.add_scalar('val/token_accuracy', metrics['token_accuracy'], self.epoch)
        self.writer.add_scalar('val/sequence_accuracy', metrics['sequence_accuracy'], self.epoch)
        self.writer.add_scalar('val/wer', metrics['wer'], self.epoch)
        self.writer.add_scalar('val/bleu', metrics['bleu'], self.epoch)
        self.writer.add_scalar('val/top5_accuracy', metrics['top5_accuracy'], self.epoch)
        
        return metrics
    
    @torch.no_grad()
    def sample_predictions(self, num_samples: int = 5):
        """Generate and display sample predictions."""
        self.model.eval()
        batch = next(iter(self.val_loader))
        
        features = batch['features'][:num_samples].to(self.device)
        feature_lengths = batch['feature_lengths'][:num_samples].to(self.device)
        texts = batch['texts'][:num_samples]
        
        # Generate predictions using greedy decoding
        output_ids = self.model.translate(features, feature_lengths, max_len=50)
        
        print("\n" + "="*70)
        print("🔮 SAMPLE PREDICTIONS")
        print("="*70)
        
        for i in range(min(num_samples, len(texts))):
            pred_text = self.tokenizer.decode(output_ids[i].cpu().tolist(), skip_special_tokens=True)
            target_text = texts[i]
            
            print(f"\n[Sample {i+1}]")
            print(f"  Target: {target_text[:80]}{'...' if len(target_text) > 80 else ''}")
            print(f"  Pred:   {pred_text[:80]}{'...' if len(pred_text) > 80 else ''}")
            
            # Show match indicator
            match = "✅" if pred_text.strip().lower() == target_text.strip().lower() else "❌"
            print(f"  Match:  {match}")
        
        print("="*70 + "\n")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_best_bleu: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save epoch checkpoint
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        
        # Save best model by loss
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model_loss.pt')
            print("  💾 Best model (loss) saved!")
        
        # Save best model by BLEU
        if is_best_bleu:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model_bleu.pt')
            print("  💾 Best model (BLEU) saved!")
        
        # Keep only last 3 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()
    
    def train(self, num_epochs: int, resume_from: str = None):
        """Main training loop."""
        # Resume from checkpoint if provided
        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.epoch = ckpt['epoch']
            self.best_loss = ckpt.get('best_loss', float('inf'))
            self.best_bleu = ckpt.get('best_bleu', 0.0)
            if self.scaler and 'scaler_state_dict' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            print(f"✅ Resumed from epoch {self.epoch}")
        
        # Print training info
        self._print_training_info()
        
        # Training loop
        for epoch in range(self.epoch + 1, num_epochs + 1):
            self.epoch = epoch
            
            # Callback for encoder unfreezing
            if hasattr(self.model, 'on_epoch_start'):
                self.model.on_epoch_start(epoch)
            
            # Train and validate
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[1]['lr']
            
            # Log to metrics logger
            self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)
            self.metrics_logger.print_summary(epoch)
            
            # Sample predictions every 5 epochs
            if epoch % 5 == 0:
                self.sample_predictions()
            
            # Check for best model
            is_best_loss = val_metrics['loss'] < self.best_loss
            is_best_bleu = val_metrics['bleu'] > self.best_bleu
            
            if is_best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if is_best_bleu:
                self.best_bleu = val_metrics['bleu']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best_loss, is_best_bleu)
            
            # Generate plots every 10 epochs
            if epoch % 10 == 0:
                self.metrics_logger.plot_training_curves()
                self.metrics_logger.save_metrics_csv()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch}!")
                print(f"   No improvement for {self.patience} epochs.")
                break
        
        # Final summary
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"Best Validation Loss: {self.best_loss:.4f}")
        print(f"Best BLEU Score: {self.best_bleu:.2f}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        # Save final plots and metrics
        self.metrics_logger.plot_training_curves()
        self.metrics_logger.save_metrics_csv()
        self.writer.close()
    
    def _print_training_info(self):
        """Print comprehensive training configuration."""
        print("\n" + "="*70)
        print("🚀 ISL TRANSLATION TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        params = self.model.count_parameters()
        print(f"\nModel Parameters:")
        print(f"  Total: {params['total']/1e6:.1f}M")
        print(f"  Trainable: {params['trainable']/1e6:.1f}M")
        print(f"  Encoder frozen: {self.model.encoder._frozen}")
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {self.config.get('num_epochs', 50)}")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Gradient accumulation: {self.grad_accum}")
        print(f"  Effective batch size: {self.train_loader.batch_size * self.grad_accum}")
        print(f"  Encoder LR: {self.config.get('encoder_lr', 1e-5):.2e}")
        print(f"  Decoder LR: {self.config.get('decoder_lr', 3e-4):.2e}")
        print(f"  Warmup steps: {self.config.get('warmup_steps', 2000)}")
        print(f"  Mixed precision (AMP): {self.use_amp}")
        print(f"  Early stopping patience: {self.patience}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train ISL Translation Model")
    parser.add_argument('--data-dir', type=str, required=True, help='Path to preprocessed data')
    parser.add_argument('--pretrained', type=str, default='MCG-NJU/videomae-base', help='Pretrained model')
    parser.add_argument('--freeze-epochs', type=int, default=10, help='Epochs to keep encoder frozen')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--encoder-lr', type=float, default=1e-5, help='Encoder learning rate')
    parser.add_argument('--decoder-lr', type=float, default=3e-4, help='Decoder learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--gradient-accumulation', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--ctc-weight', type=float, default=0.3, help='CTC loss weight')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--experiment-name', type=str, default='isl_fresh', help='Experiment name')
    args = parser.parse_args()
    
    # Setup tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Setup datasets
    train_ds = ISLDataset(args.data_dir, 'train', tokenizer)
    val_ds = ISLDataset(args.data_dir, 'val', tokenizer)
    
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=True
    )
    
    # Setup model
    model = ISLTranslator(
        input_dim=603,
        hidden_dim=256,
        vocab_size=tokenizer.vocab_size,
        pretrained=args.pretrained,
        freeze_epochs=args.freeze_epochs
    )
    
    # Training config
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'encoder_lr': args.encoder_lr,
        'decoder_lr': args.decoder_lr,
        'warmup_steps': args.warmup_steps,
        'num_epochs': args.epochs,
        'gradient_accumulation': args.gradient_accumulation,
        'ctc_weight': args.ctc_weight,
        'patience': args.patience,
        'use_amp': True,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'min_lr': 1e-7,
        'experiment_name': args.experiment_name
    }
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    trainer.train(args.epochs, args.resume)


if __name__ == '__main__':
    main()
