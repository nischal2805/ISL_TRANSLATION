"""
Quick local training on RTX 4060 (8GB VRAM)
Run 3 epochs to verify everything works before A100 training.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import time

from model_v2 import ISLTranslationModelV2, ModelConfig, HybridCTCAttentionLoss
from tokenizer import BPETokenizer


class ISLDataset(Dataset):
    """Simple dataset for local training."""
    
    def __init__(self, data_dir: Path, csv_path: Path, tokenizer: BPETokenizer, max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        
        # Load CSV
        df = pd.read_csv(csv_path)
        # Filter out NaN texts
        df = df.dropna(subset=['text'])
        df = df[df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
        uid_to_text = dict(zip(df['uid'], df['text']))
        
        # Find matching files
        self.samples = []
        files = list(self.data_dir.glob('*.npy'))
        
        for f in files:
            uid = f.stem
            if uid in uid_to_text:
                self.samples.append((f, uid_to_text[uid]))
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"Dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, text = self.samples[idx]
        
        # Load features
        features = np.load(file_path)
        features = torch.from_numpy(features).float()
        
        # Tokenize text
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return features, tokens, len(features), len(tokens)


def collate_fn(batch):
    """Collate function with padding."""
    features, tokens, feat_lens, tok_lens = zip(*batch)
    
    # Pad features
    max_feat_len = max(feat_lens)
    batch_features = torch.zeros(len(features), max_feat_len, features[0].shape[-1])
    for i, (f, l) in enumerate(zip(features, feat_lens)):
        batch_features[i, :l] = f
    
    # Pad tokens
    max_tok_len = max(tok_lens)
    batch_tokens = torch.zeros(len(tokens), max_tok_len, dtype=torch.long)
    for i, (t, l) in enumerate(zip(tokens, tok_lens)):
        batch_tokens[i, :l] = t
    
    return (
        batch_features,
        torch.tensor(feat_lens),
        batch_tokens,
        torch.tensor(tok_lens)
    )


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ctc = 0
    total_ce = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_features, feat_lens, batch_tokens, tok_lens in pbar:
        batch_features = batch_features.to(device)
        feat_lens = feat_lens.to(device)
        batch_tokens = batch_tokens.to(device)
        tok_lens = tok_lens.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            outputs = model(batch_features, feat_lens, batch_tokens, tok_lens)
            loss_dict = criterion(
                ctc_log_probs=outputs['ctc_log_probs'],
                decoder_logits=outputs['decoder_logits'],
                encoder_lengths=outputs['encoder_lengths'],
                targets=batch_tokens,
                target_lengths=tok_lens
            )
            loss = loss_dict['loss']
        
        # Check for NaN
        if torch.isnan(loss):
            print("WARNING: NaN loss, skipping batch")
            continue
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_ctc += loss_dict['ctc_loss'].item()
        total_ce += loss_dict['ce_loss'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'ctc': f"{loss_dict['ctc_loss'].item():.3f}",
            'ce': f"{loss_dict['ce_loss'].item():.3f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'ctc_loss': total_ctc / num_batches,
        'ce_loss': total_ce / num_batches
    }


@torch.no_grad()
def validate(model, dataloader, criterion, tokenizer, device):
    """Validate and show sample predictions."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    sample_predictions = []
    
    for batch_features, feat_lens, batch_tokens, tok_lens in dataloader:
        batch_features = batch_features.to(device)
        feat_lens = feat_lens.to(device)
        batch_tokens = batch_tokens.to(device)
        tok_lens = tok_lens.to(device)
        
        with autocast():
            outputs = model(batch_features, feat_lens, batch_tokens, tok_lens)
            loss_dict = criterion(
                ctc_log_probs=outputs['ctc_log_probs'],
                decoder_logits=outputs['decoder_logits'],
                encoder_lengths=outputs['encoder_lengths'],
                targets=batch_tokens,
                target_lengths=tok_lens
            )
        
        total_loss += loss_dict['loss'].item()
        num_batches += 1
        
        # Get sample predictions (first batch only)
        if len(sample_predictions) < 3:
            ctc_out = model.decode_ctc(batch_features[:1], feat_lens[:1])
            pred_text = tokenizer.decode_ctc(ctc_out[0].tolist())
            target_text = tokenizer.decode(batch_tokens[0, 1:tok_lens[0]-1].tolist())  # Remove BOS/EOS
            sample_predictions.append((pred_text[:80], target_text[:80]))
    
    return {
        'loss': total_loss / num_batches,
        'predictions': sample_predictions
    }


def main():
    print("=" * 60)
    print("ISL Translation - Local Training (RTX 4060)")
    print("=" * 60)
    
    # Config for RTX 4060 (8GB VRAM)
    BATCH_SIZE = 4  # Small batch for 8GB VRAM
    GRAD_ACCUM = 4  # Effective batch size = 16
    NUM_EPOCHS = 3
    MAX_SAMPLES = 5000  # Use subset for quick test
    LR = 1e-4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Paths
    data_dir = Path(r"E:\5thsem el\APPROACH 2\preprocessed_v2")
    csv_path = Path(r"E:\5thsem el\kortex_5th_sem\data\iSign_v1.1.csv")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer_model/bpe_tokenizer.model')
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = ISLDataset(data_dir, csv_path, tokenizer, max_samples=MAX_SAMPLES)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating model...")
    config = ModelConfig(
        input_dim=612,
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        encoder_heads=8,
        decoder_heads=8,
        dropout=0.1
    )
    model = ISLTranslationModelV2(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params/1e6:.1f}M")
    
    # Loss, optimizer, scaler
    criterion = HybridCTCAttentionLoss(
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id,
        pad_id=tokenizer.pad_id,
        ctc_weight=0.3
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "=" * 60)
    print(f"Training for {NUM_EPOCHS} epochs")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*20} Epoch {epoch+1}/{NUM_EPOCHS} {'='*20}")
        
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, tokenizer, device)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (CTC: {train_metrics['ctc_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Sample predictions
        print(f"\n  Sample Predictions:")
        for i, (pred, target) in enumerate(val_metrics['predictions'][:2]):
            print(f"    [{i+1}] Pred:   '{pred}...'")
            print(f"        Target: '{target}...'")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config.__dict__
            }, 'checkpoint_best.pt')
            print(f"\n  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    # Final save
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['loss'],
        'config': config.__dict__
    }, 'checkpoint_final.pt')
    
    print("\n" + "=" * 60)
    print("✅ LOCAL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Val Loss: {val_metrics['loss']:.4f}")
    print(f"Best Val Loss:  {best_val_loss:.4f}")
    print("\nCheckpoints saved:")
    print("  - checkpoint_best.pt")
    print("  - checkpoint_final.pt")
    print("\nReady for full training on A100!")


if __name__ == '__main__':
    main()
