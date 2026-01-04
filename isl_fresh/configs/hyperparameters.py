"""
Advanced Hyperparameter Configuration
======================================
Comprehensive hyperparameter tuning for ISL Translation with VideoMAE.
Includes LR schedulers, optimizer configs, regularization, and training strategies.
"""

import math


# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================

MODEL_CONFIG = {
    # Encoder (VideoMAE)
    'pretrained_encoder': 'MCG-NJU/videomae-base',
    'encoder_output_dim': 256,
    'num_frames': 16,
    'image_size': (224, 224),
    
    # Decoder (Transformer)
    'hidden_dim': 256,
    'decoder_layers': 6,  # Deeper for better translation
    'num_heads': 8,       # More attention heads
    'ff_dim': 1024,       # 4x hidden_dim (standard)
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'activation_dropout': 0.1,
    
    # Vocabulary
    'vocab_size': 30522,  # BERT tokenizer
    'max_text_len': 128,
    'pad_id': 0,
    'bos_id': 101,  # [CLS] for BERT
    'eos_id': 102,  # [SEP] for BERT
}


# =============================================================================
# OPTIMIZER HYPERPARAMETERS
# =============================================================================

OPTIMIZER_CONFIG = {
    # AdamW parameters
    'optimizer': 'adamw',
    'encoder_lr': 1e-5,      # Lower LR for pretrained encoder
    'decoder_lr': 3e-4,      # Higher LR for randomly initialized decoder
    'weight_decay': 0.01,    # L2 regularization
    'betas': (0.9, 0.98),    # Adam beta1, beta2 (beta2=0.98 for transformers)
    'eps': 1e-8,             # Adam epsilon
    'amsgrad': False,        # AMSGrad variant
    
    # Gradient clipping
    'max_grad_norm': 1.0,
    
    # Label smoothing
    'label_smoothing': 0.1,  # Prevents overconfidence
}


# =============================================================================
# LEARNING RATE SCHEDULER HYPERPARAMETERS
# =============================================================================

SCHEDULER_CONFIG = {
    # Scheduler type
    'scheduler': 'cosine_warmup',  # Options: 'cosine_warmup', 'linear_warmup', 'polynomial'
    
    # Warmup
    'warmup_steps': 2000,     # Linear warmup steps
    'warmup_ratio': 0.1,      # Alternative: warmup as ratio of total steps
    
    # Cosine decay
    'min_lr': 1e-7,           # Minimum learning rate
    'max_lr_encoder': 1e-5,   # Max LR for encoder
    'max_lr_decoder': 3e-4,   # Max LR for decoder
    
    # Polynomial decay (if using polynomial scheduler)
    'power': 1.0,             # Polynomial power (1.0 = linear)
    
    # Step decay (if using step scheduler)
    'lr_decay_rate': 0.95,    # Multiply LR by this
    'lr_decay_steps': 1000,   # Decay every N steps
}


# =============================================================================
# LOSS FUNCTION HYPERPARAMETERS
# =============================================================================

LOSS_CONFIG = {
    # Hybrid loss weights
    'ce_weight': 0.7,         # Cross-entropy weight
    'ctc_weight': 0.3,        # CTC weight (for streaming/alignment)
    
    # CTC parameters
    'ctc_blank_id': 0,        # Blank token for CTC
    'ctc_zero_infinity': True, # Handle infinite losses
    
    # Label smoothing
    'label_smoothing': 0.1,
    
    # Focal loss (optional - for class imbalance)
    'use_focal_loss': False,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
}


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

TRAINING_CONFIG = {
    # Training setup
    'num_epochs': 100,
    'batch_size': 16,          # Per GPU
    'gradient_accumulation': 2, # Effective batch = 16 * 2 = 32
    'effective_batch_size': 32,
    
    # Mixed precision
    'use_amp': True,           # Automatic mixed precision (FP16)
    'amp_opt_level': 'O1',     # Apex optimization level
    
    # Transfer learning strategy
    'freeze_encoder_epochs': 10,  # Freeze encoder for first N epochs
    'encoder_unfreeze_lr': 5e-6,  # LR after unfreezing (lower than initial)
    
    # Early stopping
    'patience': 15,            # Stop if no improvement for N epochs
    'min_delta': 1e-4,         # Minimum improvement to count
    'early_stop_metric': 'loss', # 'loss' or 'bleu'
    
    # Checkpointing
    'save_every': 5,           # Save checkpoint every N epochs
    'keep_best_k': 3,          # Keep top-K checkpoints
    'checkpoint_metric': 'bleu', # Metric for best checkpoint
    
    # Validation
    'val_every': 1,            # Validate every N epochs
    'val_steps': None,         # Validate every N steps (None = epoch-based)
    
    # Logging
    'log_every': 100,          # Log metrics every N steps
    'sample_every': 5,         # Sample predictions every N epochs
}


# =============================================================================
# DATA AUGMENTATION HYPERPARAMETERS
# =============================================================================

AUGMENTATION_CONFIG = {
    # Video augmentation
    'random_crop': True,
    'crop_size': (224, 224),
    'horizontal_flip': 0.5,    # Probability
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05,
    },
    'random_rotation': 10,     # Degrees
    
    # Temporal augmentation
    'temporal_crop': True,
    'min_frames': 12,
    'max_frames': 20,
    
    # Text augmentation
    'token_dropout': 0.1,      # Randomly drop tokens
    'token_masking': 0.15,     # Mask tokens (BERT-style)
}


# =============================================================================
# REGULARIZATION HYPERPARAMETERS
# =============================================================================

REGULARIZATION_CONFIG = {
    # Dropout
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'activation_dropout': 0.1,
    'encoder_dropout': 0.0,    # VideoMAE already has dropout
    
    # Weight decay
    'weight_decay': 0.01,
    'weight_decay_encoder': 0.01,
    'weight_decay_decoder': 0.01,
    
    # Gradient clipping
    'max_grad_norm': 1.0,
    
    # Stochastic depth (for transformers)
    'drop_path_rate': 0.1,
    
    # Layer dropout
    'layer_dropout': 0.0,      # Randomly skip layers
}


# =============================================================================
# ADVANCED SCHEDULER: Cosine Warmup with Restarts
# =============================================================================

class CosineWarmupRestartsScheduler:
    """
    Cosine annealing with warmup and periodic restarts (SGDR).
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Linear warmup duration
        cycle_steps: Steps per cosine cycle
        min_lr: Minimum learning rate
        restart_mult: Multiply cycle length by this after each restart
        gamma: Decay max_lr by this after each restart
    """
    
    def __init__(self, optimizer, warmup_steps, cycle_steps, min_lr=1e-7, 
                 restart_mult=2.0, gamma=0.95):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.restart_mult = restart_mult
        self.gamma = gamma
        
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
        self.current_cycle = 0
        self.cycle_start_step = 0
        self.max_lrs = self.base_lrs.copy()
    
    def step(self):
        self.current_step += 1
        
        # Warmup phase
        if self.current_step < self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.base_lrs[i] * scale
            return
        
        # Cosine annealing phase
        steps_in_cycle = self.current_step - self.cycle_start_step - self.warmup_steps
        current_cycle_steps = self.cycle_steps * (self.restart_mult ** self.current_cycle)
        
        # Check for restart
        if steps_in_cycle >= current_cycle_steps:
            self.current_cycle += 1
            self.cycle_start_step = self.current_step
            steps_in_cycle = 0
            # Decay max LR
            self.max_lrs = [lr * self.gamma for lr in self.max_lrs]
        
        # Cosine schedule
        progress = steps_in_cycle / current_cycle_steps
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.min_lr + (self.max_lrs[i] - self.min_lr) * \
                       0.5 * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# =============================================================================
# GPU/HARDWARE SPECIFIC HYPERPARAMETERS
# =============================================================================

A100_CONFIG = {
    # GPU utilization
    'batch_size': 16,          # A100 40GB can handle this
    'gradient_accumulation': 2,
    'num_workers': 8,          # Data loading workers
    'pin_memory': True,
    'prefetch_factor': 2,
    
    # Mixed precision
    'use_amp': True,
    'amp_backend': 'native',   # 'native' or 'apex'
    
    # Distributed training (if multi-GPU)
    'distributed': False,
    'world_size': 1,
    'local_rank': 0,
    
    # CUDA optimizations
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
}


# =============================================================================
# HYPERPARAMETER SEARCH SPACE (for tuning)
# =============================================================================

SEARCH_SPACE = {
    'learning_rate': [1e-5, 3e-5, 5e-5, 1e-4, 3e-4],
    'batch_size': [8, 16, 32],
    'hidden_dim': [256, 384, 512],
    'decoder_layers': [4, 6, 8],
    'num_heads': [4, 8, 16],
    'dropout': [0.05, 0.1, 0.15, 0.2],
    'weight_decay': [0.0, 0.01, 0.05, 0.1],
    'warmup_steps': [500, 1000, 2000, 4000],
    'label_smoothing': [0.0, 0.05, 0.1, 0.15],
    'ctc_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
}


# =============================================================================
# COMPLETE CONFIGURATION (combines all)
# =============================================================================

COMPLETE_CONFIG = {
    **MODEL_CONFIG,
    **OPTIMIZER_CONFIG,
    **SCHEDULER_CONFIG,
    **LOSS_CONFIG,
    **TRAINING_CONFIG,
    **REGULARIZATION_CONFIG,
    **A100_CONFIG,
}


def get_config(preset='default'):
    """
    Get configuration preset.
    
    Args:
        preset: 'default', 'fast' (quick training), 'production' (best quality)
    
    Returns:
        config dict
    """
    if preset == 'fast':
        config = COMPLETE_CONFIG.copy()
        config.update({
            'num_epochs': 20,
            'decoder_layers': 4,
            'batch_size': 32,
            'gradient_accumulation': 1,
            'warmup_steps': 500,
        })
        return config
    
    elif preset == 'production':
        config = COMPLETE_CONFIG.copy()
        config.update({
            'num_epochs': 150,
            'decoder_layers': 8,
            'num_heads': 16,
            'hidden_dim': 384,
            'batch_size': 12,
            'gradient_accumulation': 3,
            'warmup_steps': 4000,
            'patience': 20,
        })
        return config
    
    else:  # default
        return COMPLETE_CONFIG.copy()


if __name__ == '__main__':
    # Print configuration summary
    print("="*70)
    print("HYPERPARAMETER CONFIGURATION SUMMARY")
    print("="*70)
    
    config = get_config('default')
    
    print(f"\n📦 MODEL:")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Decoder layers: {config['decoder_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  Dropout: {config['dropout']}")
    
    print(f"\n⚙️ OPTIMIZER:")
    print(f"  Encoder LR: {config['encoder_lr']}")
    print(f"  Decoder LR: {config['decoder_lr']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Betas: {config['betas']}")
    
    print(f"\n📈 SCHEDULER:")
    print(f"  Type: {config['scheduler']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    print(f"  Min LR: {config['min_lr']}")
    
    print(f"\n🎯 TRAINING:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Grad accumulation: {config['gradient_accumulation']}")
    print(f"  Effective batch: {config['effective_batch_size']}")
    print(f"  Mixed precision: {config['use_amp']}")
    
    print(f"\n🛡️ REGULARIZATION:")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Max grad norm: {config['max_grad_norm']}")
    
    print(f"\n⏹️ EARLY STOPPING:")
    print(f"  Patience: {config['patience']}")
    print(f"  Metric: {config['early_stop_metric']}")
