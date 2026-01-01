"""
ISL Translation System - Utility Functions
===========================================
Helper functions for training, inference, and deployment.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from config import model_config


# ============================================================================
# Random Seed
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Logging
# ============================================================================

def setup_logging(
    log_dir: str = './logs',
    log_level: int = logging.INFO,
    log_file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Whether to save logs to file
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('isl_translation')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'training_{timestamp}.log')
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Model Utilities
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters by component.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'components': component_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = None,
    opset_version: int = 14
):
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (batch, time, features)
        opset_version: ONNX opset version
    """
    if input_shape is None:
        input_shape = (1, 100, model_config.input_dim)
    
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    dummy_lengths = torch.tensor([input_shape[1]])
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_lengths),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['features', 'lengths'],
        output_names=['output_ids', 'output_probs'],
        dynamic_axes={
            'features': {0: 'batch_size', 1: 'sequence_length'},
            'lengths': {0: 'batch_size'},
            'output_ids': {0: 'batch_size', 1: 'output_length'},
            'output_probs': {0: 'batch_size', 1: 'output_length'}
        }
    )
    
    print(f"Model exported to {output_path}")


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Quantize model to INT8 for mobile deployment.
    
    Args:
        model: PyTorch model
        calibration_data: Optional calibration data for static quantization
        
    Returns:
        Quantized model
    """
    model.eval()
    
    # Dynamic quantization (simpler, works well for RNN/GRU)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU, nn.Conv1d},
        dtype=torch.qint8
    )
    
    return quantized_model


# ============================================================================
# Data Utilities
# ============================================================================

def split_data_by_signer(
    data: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data ensuring no signer overlap between splits.
    
    Args:
        data: List of data items with 'signer_id' key
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        train, val, test splits
    """
    np.random.seed(seed)
    
    # Group by signer
    signer_data = {}
    for item in data:
        signer_id = item.get('signer_id', 'unknown')
        if signer_id not in signer_data:
            signer_data[signer_id] = []
        signer_data[signer_id].append(item)
    
    # Shuffle signers
    signers = list(signer_data.keys())
    np.random.shuffle(signers)
    
    # Split signers
    n_signers = len(signers)
    n_train = int(n_signers * train_ratio)
    n_val = int(n_signers * val_ratio)
    
    train_signers = signers[:n_train]
    val_signers = signers[n_train:n_train + n_val]
    test_signers = signers[n_train + n_val:]
    
    # Collect data
    train_data = [item for s in train_signers for item in signer_data[s]]
    val_data = [item for s in val_signers for item in signer_data[s]]
    test_data = [item for s in test_signers for item in signer_data[s]]
    
    return train_data, val_data, test_data


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    
    Args:
        labels: List of class labels
        num_classes: Total number of classes
        
    Returns:
        Tensor of class weights
    """
    counts = np.bincount(labels, minlength=num_classes)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Training Utilities
# ============================================================================

class AverageMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.01
):
    """
    Create cosine learning rate schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles
        min_lr_ratio: Minimum LR as ratio of initial LR
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        
        cosine_decay = 0.5 * (1 + np.cos(np.pi * num_cycles * 2.0 * progress))
        
        return max(min_lr_ratio, cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Inference Utilities
# ============================================================================

def beam_search_decode(
    model: nn.Module,
    encoder_output: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 100,
    sos_id: int = 1,
    eos_id: int = 2,
    length_penalty: float = 0.6
) -> Tuple[List[int], float]:
    """
    Beam search decoding.
    
    Args:
        model: Model with decoder
        encoder_output: (1, T, d_model) encoder output
        beam_size: Beam size
        max_len: Maximum output length
        sos_id: Start token ID
        eos_id: End token ID
        length_penalty: Length normalization penalty
        
    Returns:
        best_sequence: List of token IDs
        best_score: Log probability score
    """
    device = encoder_output.device
    
    # Initialize beams
    beams = [([], 0.0, None)]  # (sequence, score, hidden_state)
    
    for step in range(max_len):
        all_candidates = []
        
        for seq, score, hidden in beams:
            if len(seq) > 0 and seq[-1] == eos_id:
                all_candidates.append((seq, score, hidden))
                continue
            
            # Get next token probabilities
            if len(seq) == 0:
                input_token = torch.tensor([[sos_id]], device=device)
            else:
                input_token = torch.tensor([[seq[-1]]], device=device)
            
            # This is simplified - actual implementation would need
            # proper hidden state handling for GRU decoder
            with torch.no_grad():
                # Would call decoder step here
                pass
            
            # For each possible next token
            # (simplified - would iterate over top-k tokens)
            for token_id in range(model.vocab_size):
                new_seq = seq + [token_id]
                new_score = score  # Would add log probability
                all_candidates.append((new_seq, new_score, hidden))
        
        # Select top beams
        all_candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
        beams = all_candidates[:beam_size]
        
        # Check if all beams ended
        if all(seq[-1] == eos_id for seq, _, _ in beams if len(seq) > 0):
            break
    
    best_seq, best_score, _ = beams[0]
    return best_seq, best_score


# ============================================================================
# File Utilities
# ============================================================================

def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Get path to latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('*.pt'))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return str(checkpoints[0])


# ============================================================================
# Visualization Utilities
# ============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: str = 'training_curves.png'
):
    """
    Plot training curves.
    
    Args:
        history: Dictionary with metric histories
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train')
        ax.plot(history['val_loss'], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend()
    
    # Character Accuracy
    if 'char_accuracy' in history:
        ax = axes[0, 1]
        ax.plot(history['char_accuracy'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Character Accuracy')
    
    # WER
    if 'wer' in history:
        ax = axes[1, 0]
        ax.plot(history['wer'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('WER')
        ax.set_title('Word Error Rate')
    
    # Learning Rate
    if 'lr' in history:
        ax = axes[1, 1]
        ax.plot(history['lr'])
        ax.set_xlabel('Step')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Training curves saved to {output_path}")


if __name__ == "__main__":
    # Test utilities
    print("Utility Functions Test")
    print("=" * 50)
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter: avg={meter.avg}, count={meter.count}")
    
    # Test logging
    logger = setup_logging(log_file=False)
    logger.info("Test log message")
