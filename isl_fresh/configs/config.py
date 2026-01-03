"""
Configuration for ISL Translation Pipeline
==========================================

This defines all the settings for:
1. Feature extraction (landmarks)
2. Model architecture
3. Training hyperparameters
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class LandmarkConfig:
    """Settings for MediaPipe landmark extraction."""
    
    # Which landmarks to extract (x, y, z for each)
    use_pose: bool = True
    use_hands: bool = True
    use_face: bool = True
    
    # Pose: use upper body only (more relevant for signing)
    # Indices: nose(0), shoulders(11,12), elbows(13,14), wrists(15,16), 
    #          pinky(17,18), index(19,20), thumb(21,22), hips(23,24)
    pose_indices: List[int] = field(default_factory=lambda: [
        0,  # nose
        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists
        17, 18,  # pinky
        19, 20,  # index
        21, 22,  # thumb
        23, 24   # hips
    ])  # 15 points x 3 = 45 dims
    
    # Hands: all 21 landmarks per hand
    hand_landmarks: int = 21  # 21 x 3 x 2 = 126 dims
    
    # Face: key expression points (mouth, eyebrows)
    # Indices for mouth corners, lips, eyebrows
    face_indices: List[int] = field(default_factory=lambda: [
        # Mouth
        61, 291,  # mouth corners
        0, 17,    # upper/lower lip center
        # Eyebrows
        70, 300,  # left/right eyebrow
        # Eyes
        33, 263,  # left/right eye outer
        # Nose
        1, 4      # nose bridge, tip
    ])  # 10 points x 3 = 30 dims
    
    # Total: 45 + 126 + 30 = 201 dimensions per frame
    
    @property
    def feature_dim(self) -> int:
        """Calculate total feature dimensions."""
        dim = 0
        if self.use_pose:
            dim += len(self.pose_indices) * 3
        if self.use_hands:
            dim += self.hand_landmarks * 3 * 2  # both hands
        if self.use_face:
            dim += len(self.face_indices) * 3
        return dim


@dataclass
class DataConfig:
    """Settings for data processing."""
    
    # Paths
    video_dir: str = r"E:\iSign-videos_v1.1"
    csv_path: str = r"E:\5thsem el\APPROACH 2\iSign_v1.1.csv"
    output_dir: str = r"E:\5thsem el\APPROACH 2\isl_fresh\data"
    
    # Processing
    max_seq_len: int = 300  # Max frames per video
    min_seq_len: int = 10   # Min frames (skip very short)
    sample_rate: int = 1    # Sample every N frames (1 = all frames)
    
    # Split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Normalization
    normalize: bool = True
    
    # Data augmentation
    augment: bool = True
    augment_noise_std: float = 0.01
    augment_scale_range: tuple = (0.9, 1.1)
    augment_time_stretch: tuple = (0.9, 1.1)


@dataclass 
class ModelConfig:
    """Settings for model architecture."""
    
    # Input
    input_dim: int = 201  # From LandmarkConfig
    
    # Encoder (processes sign sequences)
    encoder_hidden: int = 256
    encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_ff_dim: int = 512
    encoder_dropout: float = 0.1
    
    # Decoder (generates text)
    decoder_hidden: int = 256
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1
    
    # Output
    vocab_size: int = 8000  # BPE vocabulary size
    max_output_len: int = 100
    
    # Special tokens
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    
    # CTC for alignment
    use_ctc: bool = True
    ctc_weight: float = 0.3


@dataclass
class TrainingConfig:
    """Settings for training."""
    
    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_epochs: int = 100
    
    # Gradient
    gradient_clip: float = 1.0
    gradient_accumulation: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = r"E:\5thsem el\APPROACH 2\isl_fresh\checkpoints"
    save_every: int = 5  # Save every N epochs
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Logging
    log_dir: str = r"E:\5thsem el\APPROACH 2\isl_fresh\logs"
    log_every: int = 100  # Log every N steps
    
    # Validation
    val_every: int = 1  # Validate every N epochs


@dataclass
class Config:
    """Master configuration combining all settings."""
    
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Ensure model input_dim matches landmark feature_dim
        self.model.input_dim = self.landmarks.feature_dim
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            landmarks=LandmarkConfig(**data.get('landmarks', {})),
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {}))
        )
    
    def save_yaml(self, path: str):
        """Save config to YAML file."""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    # Print default config
    config = Config()
    print(f"Feature dimension: {config.landmarks.feature_dim}")
    print(f"Model input dim: {config.model.input_dim}")
    print(f"Encoder: {config.model.encoder_layers} layers, {config.model.encoder_hidden} hidden")
    print(f"Decoder: {config.model.decoder_layers} layers, {config.model.decoder_hidden} hidden")
