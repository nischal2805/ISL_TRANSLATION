"""
Training Script for Video-Based ISL Translation
===============================================
Uses actual video frames with VideoMAE encoder.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.video_dataset import create_video_dataloaders
from src.models.translator import ISLTranslator
from src.training.trainer_v2 import Trainer
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train ISL Translation with Video Frames")
    
    # Data
    parser.add_argument('--video-dir', type=str, default=r'E:\iSign-videos_v1.1',
                        help='Directory containing .mp4 video files')
    parser.add_argument('--csv-path', type=str, default=r'E:\5thsem el\APPROACH 2\iSign_v1.1.csv',
                        help='CSV file with video metadata')
    
    # Model
    parser.add_argument('--pretrained', type=str, default='MCG-NJU/videomae-base',
                        help='Pretrained VideoMAE model')
    parser.add_argument('--freeze-epochs', type=int, default=10,
                        help='Epochs to keep encoder frozen')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames per video clip')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Decoder hidden dimension')
    parser.add_argument('--decoder-layers', type=int, default=4,
                        help='Number of decoder layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (smaller for video data)')
    parser.add_argument('--encoder-lr', type=float, default=1e-5,
                        help='Encoder learning rate (lower for pretrained)')
    parser.add_argument('--decoder-lr', type=float, default=3e-4,
                        help='Decoder learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2000,
                        help='Warmup steps for LR scheduler')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--ctc-weight', type=float, default=0.3,
                        help='CTC loss weight')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # System
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_video',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs_video',
                        help='Log directory')
    parser.add_argument('--experiment-name', type=str, default='videomae_transfer',
                        help='Experiment name for logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("VIDEO-BASED ISL TRANSLATION TRAINING")
    print("="*70)
    print(f"Video directory: {args.video_dir}")
    print(f"CSV path: {args.csv_path}")
    print(f"Pretrained model: {args.pretrained}")
    print(f"Num frames: {args.num_frames}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Freeze encoder epochs: {args.freeze_epochs}")
    print("="*70)
    
    # Load tokenizer (using BERT tokenizer)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create video dataloaders
    print("\nCreating video dataloaders...")
    train_loader, val_loader, test_loader = create_video_dataloaders(
        video_dir=args.video_dir,
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating VideoMAE-based model...")
    model = ISLTranslator(
        hidden_dim=args.hidden_dim,
        vocab_size=tokenizer.vocab_size,
        decoder_layers=args.decoder_layers,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1,
        pad_id=tokenizer.pad_token_id,
        use_ctc=True,
        pretrained=args.pretrained,
        freeze_epochs=args.freeze_epochs,
        num_frames=args.num_frames
    )
    
    params = model.count_parameters()
    print(f"✓ Total parameters: {params['total']:,} ({params['total']/1e6:.1f}M)")
    print(f"✓ Trainable parameters: {params['trainable']:,} ({params['trainable']/1e6:.1f}M)")
    print(f"✓ Encoder frozen: {model.encoder._frozen}")
    
    # Training configuration
    config = {
        'device': 'cuda',
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
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
    
    # Start training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    trainer.train(args.epochs, args.resume)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    print(f"Best BLEU score: {trainer.best_bleu:.2f}")
    print("="*70)


if __name__ == '__main__':
    main()
