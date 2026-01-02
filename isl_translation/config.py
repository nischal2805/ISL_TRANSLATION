"""
ISL Translation System - Configuration
=======================================
Central configuration file for all hyperparameters and settings.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Dataset paths - GPU SERVER PATHS
    dataset_root: str = "/media/rvcse22/CSERV/kortex_sem5/ramita"
    videos_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/videos"
    annotations_file: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/iSign_v1.1.csv"
    csv_path: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/iSign_v1.1.csv"  # Alias
    
    # Preprocessed data paths
    preprocessed_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final"
    processed_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final"  # Alias
    train_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/train"
    val_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/val"
    test_dir: str = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/test"
    
    # Data splits (70/15/15)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Max samples (None = use all, set by GPU mode)
    max_samples: Optional[int] = None
    
    # Sequence constraints
    max_src_len: int = 500  # Maximum video frames
    max_tgt_len: int = 100  # Maximum text length
    min_src_len: int = 4    # Minimum frames for CTC
    
    # Preprocessing
    fps: int = 30
    gaussian_sigma: float = 1.0  # Smoothing parameter
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class LandmarkConfig:
    """MediaPipe landmark configuration."""
    # Pose landmarks to extract (shoulders + elbows)
    pose_indices: List[int] = field(default_factory=lambda: [11, 12, 13, 14])
    
    # Hand landmarks (all 21 per hand)
    hand_indices: List[int] = field(default_factory=lambda: list(range(21)))
    
    # Total landmarks: 4 pose + 21 left hand + 21 right hand = 46
    num_landmarks: int = 46
    
    # Dimensions
    num_coords: int = 3  # x, y, z
    position_dims: int = 138  # 46 * 3
    feature_dims: int = 414   # 138 * 3 (pos + vel + acc)
    
    # MediaPipe settings
    static_image_mode: bool = False
    model_complexity: int = 0  # Lite model for speed
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Input/Output dimensions
    input_dim: int = 414      # 138 position * 3 (pos + vel + acc)
    d_model: int = 256        # Model dimension (used by create_model)
    hidden_dim: int = 256     # Main hidden dimension (same as d_model)
    embedding_dim: int = 256  # Decoder embedding dimension
    vocab_size: int = 35      # CORRECTED: 5 special + 26 letters + 4 punctuation
    
    # Encoder
    num_cnn_blocks: int = 2
    cnn_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    num_conformer_blocks: int = 2
    num_heads: int = 4        # Attention heads (used by create_model)
    num_attention_heads: int = 4
    ff_expansion: int = 4     # Feed-forward expansion (used by create_model)
    conformer_ff_expansion: int = 4
    conv_kernel_size: int = 31  # Conformer conv kernel (used by create_model)
    conformer_conv_kernel: int = 31
    subsample_factor: int = 2  # Temporal subsampling
    
    # Decoder (GRU only - no Transformer)
    num_decoder_layers: int = 2
    decoder_hidden_dim: int = 256
    
    # Dropout rates (REDUCED for 127K dataset)
    dropout: float = 0.3      # Main dropout (used by create_model)
    input_dropout: float = 0.2
    cnn_dropout: float = 0.1
    encoder_dropout: float = 0.3  # Reduced from 0.4
    decoder_dropout: float = 0.25  # Reduced from 0.3
    
    # Positional encoding
    max_seq_len: int = 1000


@dataclass
class TrainingConfig:
    """Training configuration."""
    # GPU mode switch
    # "small" = RTX 4060 (8GB), 40% dataset, 20 epochs
    # "large" = A100 GPU, 100% dataset, full training
    gpu_mode: str = "small"  # Options: "small", "large"
    
    # Epochs based on GPU mode
    epochs_small: int = 20
    epochs_large: int = 60
    
    # Dataset subset based on GPU mode
    subset_ratio_small: float = 0.4  # 40% for small GPU
    subset_ratio_large: float = 1.0  # 100% for large GPU
    
    # Batch size (adjusted for GPU memory)
    batch_size_small: int = 8   # RTX 4060
    batch_size_large: int = 32  # A100
    
    # Gradient accumulation
    gradient_accumulation_small: int = 3  # Effective: 8*3=24
    gradient_accumulation_large: int = 1  # Effective: 32*1=32
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 5e-5  # REDUCED from 1e-4
    betas: tuple = (0.9, 0.999)
    
    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    scheduler_t0: int = 10
    scheduler_tmult: int = 2
    
    # Loss weights
    ctc_weight_start: float = 0.3
    ctc_weight_end: float = 0.1
    ctc_decay_epochs: int = 30
    label_smoothing: float = 0.1
    
    # Teacher forcing
    tf_ratio_start: float = 0.9
    tf_ratio_end: float = 0.2
    tf_decay_epochs: int = 15
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Gradient checkpointing (saves memory)
    use_gradient_checkpointing: bool = True
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    
    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 50
    
    # Validation
    val_every_n_epochs: int = 1
    num_val_samples_to_print: int = 5
    
    @property
    def epochs(self) -> int:
        return self.epochs_small if self.gpu_mode == "small" else self.epochs_large
    
    @property
    def num_epochs(self) -> int:
        """Alias for epochs (used by train.py)."""
        return self.epochs
    
    @num_epochs.setter
    def num_epochs(self, value: int):
        """Allow setting num_epochs directly."""
        if self.gpu_mode == "small":
            self.epochs_small = value
        else:
            self.epochs_large = value
    
    @property
    def subset_ratio(self) -> float:
        return self.subset_ratio_small if self.gpu_mode == "small" else self.subset_ratio_large
    
    @property
    def batch_size(self) -> int:
        return self.batch_size_small if self.gpu_mode == "small" else self.batch_size_large
    
    @property
    def gradient_accumulation(self) -> int:
        return self.gradient_accumulation_small if self.gpu_mode == "small" else self.gradient_accumulation_large
    
    @property
    def early_stopping_patience(self) -> int:
        """Alias for patience."""
        return self.patience
    
    @property
    def save_interval(self) -> int:
        """Alias for save_every_n_epochs."""
        return self.save_every_n_epochs
    
    @property
    def gradient_checkpointing(self) -> bool:
        """Alias for use_gradient_checkpointing."""
        return self.use_gradient_checkpointing
    
    @property 
    def teacher_forcing_start(self) -> float:
        """Alias for tf_ratio_start."""
        return self.tf_ratio_start
    
    @property
    def teacher_forcing_end(self) -> float:
        """Alias for tf_ratio_end."""
        return self.tf_ratio_end


@dataclass
class VocabConfig:
    """Vocabulary configuration."""
    # Special tokens
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    space_token: str = " "
    
    # Token IDs
    pad_id: int = 0   # Also CTC blank
    sos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    space_id: int = 4
    
    # Characters (a-z: 5-30)
    char_offset: int = 5
    
    # Punctuation (31-34)
    punct_map: dict = field(default_factory=lambda: {
        '.': 31,
        ',': 32,
        '!': 33,
        '?': 34
    })
    
    # Total vocab size
    vocab_size: int = 35


@dataclass 
class InferenceConfig:
    """Inference configuration."""
    # Decoding
    ctc_beam_width: int = 10
    decoder_max_len: int = 100
    
    # Real-time settings
    buffer_size: int = 90  # 3 seconds at 30fps
    predict_every_n_frames: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Device constant
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global configuration instances
data_config = DataConfig()
landmark_config = LandmarkConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
vocab_config = VocabConfig()
inference_config = InferenceConfig()


def get_config():
    """Get all configuration as a dictionary."""
    return {
        "data": data_config,
        "landmark": landmark_config,
        "model": model_config,
        "training": training_config,
        "vocab": vocab_config,
        "inference": inference_config
    }


def set_gpu_mode(mode: str):
    """
    Switch between small and large GPU modes.
    
    Args:
        mode: "small" for RTX 4060 (40% data, 20 epochs)
              "large" for A100 (100% data, 60 epochs)
    """
    assert mode in ["small", "large"], f"Invalid mode: {mode}"
    training_config.gpu_mode = mode
    print(f"GPU mode set to: {mode}")
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - Dataset subset: {training_config.subset_ratio * 100}%")
    print(f"  - Gradient accumulation: {training_config.gradient_accumulation}")


if __name__ == "__main__":
    # Test configuration
    print("ISL Translation Configuration")
    print("=" * 50)
    
    print("\n[Small GPU Mode - RTX 4060]")
    set_gpu_mode("small")
    
    print("\n[Large GPU Mode - A100]")
    set_gpu_mode("large")
    
    print("\n[Model Config]")
    print(f"  Input dim: {model_config.input_dim}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Encoder dropout: {model_config.encoder_dropout}")
