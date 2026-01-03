"""
Production Training Script for ISL Translation
==============================================
Optimized for A100 GPU with all best practices.
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
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model.to(self.device)
        
        # Separate LR for encoder/decoder
        encoder_params = list(model.encoder.parameters())
        decoder_params = list(model.decoder.parameters())
        ctc_params = list(model.ctc_head.parameters()) if hasattr(model, 'ctc_head') else []
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': config.get('encoder_lr', 1e-5)},
            {'params': decoder_params + ctc_params, 'lr': config.get('decoder_lr', 3e-4)}
        ], weight_decay=config.get('weight_decay', 0.01), betas=(0.9, 0.98))
        
        steps_per_epoch = len(train_loader) // config.get('gradient_accumulation', 1)
        total_steps = steps_per_epoch * config.get('num_epochs', 50)
        
        self.scheduler = CosineWarmupScheduler(self.optimizer, config.get('warmup_steps', 2000), total_steps)
        self.loss_fn = HybridLoss(model.vocab_size, model.pad_id, config.get('ctc_weight', 0.3), config.get('label_smoothing', 0.1))
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        self.use_amp = config.get('use_amp', True)
        self.grad_accum = config.get('gradient_accumulation', 1)
        self.best_loss = float('inf')
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(config.get('log_dir', 'logs')) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        self.epoch = 0
        self.global_step = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_ce, total_ctc, num_batches = 0, 0, 0, 0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device, non_blocking=True)
            feature_lengths = batch['feature_lengths'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)
            target_lengths = batch['target_lengths'].to(self.device, non_blocking=True)
            
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
            
            if (batch_idx + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('train/loss', losses['loss'].item(), self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[1]['lr'], self.global_step)
            
            total_loss += losses['loss'].item()
            total_ce += losses['ce_loss'].item()
            total_ctc += losses['ctc_loss'].item()
            num_batches += 1
            pbar.set_postfix({'loss': f"{losses['loss'].item():.4f}", 'lr': f"{self.optimizer.param_groups[1]['lr']:.2e}"})
        
        return {'loss': total_loss/num_batches, 'ce_loss': total_ce/num_batches, 'ctc_loss': total_ctc/num_batches}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, total_ce, total_ctc, num_batches = 0, 0, 0, 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            features = batch['features'].to(self.device)
            feature_lengths = batch['feature_lengths'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            outputs = self.model(features, feature_lengths, targets, target_lengths)
            losses = self.loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
            total_loss += losses['loss'].item()
            total_ce += losses['ce_loss'].item()
            total_ctc += losses['ctc_loss'].item()
            num_batches += 1
        return {'loss': total_loss/num_batches, 'ce_loss': total_ce/num_batches, 'ctc_loss': total_ctc/num_batches}
    
    @torch.no_grad()
    def sample_predictions(self, num_samples=3):
        self.model.eval()
        batch = next(iter(self.val_loader))
        features = batch['features'][:num_samples].to(self.device)
        feature_lengths = batch['feature_lengths'][:num_samples].to(self.device)
        texts = batch['texts'][:num_samples]
        output_ids = self.model.translate(features, feature_lengths, max_len=50)
        print("\n" + "-"*50 + "\nSample Predictions:")
        for i in range(num_samples):
            pred = self.tokenizer.decode(output_ids[i].cpu().tolist(), skip_special_tokens=True)
            print(f"  Target: {texts[i][:60]}...")
            print(f"  Pred:   {pred[:60]}...")
        print("-"*50 + "\n")
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {'epoch': epoch, 'global_step': self.global_step, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.scheduler.state_dict(),
                      'best_loss': self.best_loss, 'config': self.config}
        if self.scaler: checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        if is_best: torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt'); print("  💾 Best model saved!")
        for old in sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))[:-3]: old.unlink()
    
    def train(self, num_epochs, resume_from=None):
        if resume_from:
            ckpt = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.epoch = ckpt['epoch']
            self.best_loss = ckpt['best_loss']
            print(f"Resumed from epoch {self.epoch}")
        
        print("="*60 + f"\nISL TRANSLATION TRAINING\nDevice: {self.device}")
        if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
        params = self.model.count_parameters()
        print(f"Params: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")
        print(f"Encoder frozen: {self.model.encoder._frozen}\n" + "="*60)
        
        for epoch in range(self.epoch + 1, num_epochs + 1):
            self.epoch = epoch
            if hasattr(self.model, 'on_epoch_start'): self.model.on_epoch_start(epoch)
            
            train_m = self.train_epoch(epoch)
            val_m = self.validate()
            self.writer.add_scalar('val/loss', val_m['loss'], epoch)
            
            print(f"Epoch {epoch}: Train={train_m['loss']:.4f}, Val={val_m['loss']:.4f}")
            if epoch % 5 == 0: self.sample_predictions()
            
            is_best = val_m['loss'] < self.best_loss
            if is_best: self.best_loss = val_m['loss']; self.patience_counter = 0
            else: self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= self.patience:
                print(f"⚠️ Early stopping at epoch {epoch}")
                break
        
        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default='MCG-NJU/videomae-base')
    parser.add_argument('--freeze-epochs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--decoder-lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--gradient-accumulation', type=int, default=2)
    parser.add_argument('--ctc-weight', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_ds = ISLDataset(args.data_dir, 'train', tokenizer)
    val_ds = ISLDataset(args.data_dir, 'val', tokenizer)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    model = ISLTranslator(input_dim=603, hidden_dim=256, vocab_size=tokenizer.vocab_size, pretrained=args.pretrained, freeze_epochs=args.freeze_epochs)
    
    config = {'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'encoder_lr': args.encoder_lr, 'decoder_lr': args.decoder_lr,
              'warmup_steps': args.warmup_steps, 'num_epochs': args.epochs, 'gradient_accumulation': args.gradient_accumulation,
              'ctc_weight': args.ctc_weight, 'patience': args.patience, 'use_amp': True, 'checkpoint_dir': args.checkpoint_dir,
              'log_dir': 'logs', 'max_grad_norm': 1.0, 'weight_decay': 0.01, 'label_smoothing': 0.1}
    
    Trainer(model, train_loader, val_loader, tokenizer, config).train(args.epochs, args.resume)


if __name__ == '__main__':
    main()
