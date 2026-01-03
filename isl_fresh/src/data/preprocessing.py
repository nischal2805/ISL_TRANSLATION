"""
Landmark Preprocessing
======================

Preprocesses raw landmarks for model input:
1. Normalization (center and scale)
2. Temporal features (velocity, acceleration)
3. Padding/truncation to fixed length
4. Data augmentation
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingStats:
    """Statistics for normalization (computed from training data)."""
    mean: np.ndarray
    std: np.ndarray
    
    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> "PreprocessingStats":
        data = np.load(path)
        return cls(mean=data['mean'], std=data['std'])


def normalize_landmarks(
    landmarks: np.ndarray,
    stats: Optional[PreprocessingStats] = None
) -> Tuple[np.ndarray, PreprocessingStats]:
    """
    Normalize landmarks to zero mean and unit variance.
    
    Args:
        landmarks: Shape (seq_len, feature_dim) or (batch, seq_len, feature_dim)
        stats: Pre-computed statistics (for test data)
    
    Returns:
        Normalized landmarks and statistics used
    """
    if stats is None:
        # Compute statistics
        if landmarks.ndim == 2:
            mean = landmarks.mean(axis=0)
            std = landmarks.std(axis=0) + 1e-8
        else:
            # Batch: compute across all samples and timesteps
            mean = landmarks.reshape(-1, landmarks.shape[-1]).mean(axis=0)
            std = landmarks.reshape(-1, landmarks.shape[-1]).std(axis=0) + 1e-8
        stats = PreprocessingStats(mean=mean, std=std)
    
    normalized = (landmarks - stats.mean) / stats.std
    return normalized, stats


def center_pose(landmarks: np.ndarray, shoulder_indices: Tuple[int, int] = (3, 6)) -> np.ndarray:
    """
    Center landmarks relative to shoulder midpoint.
    
    For sign language, centering on shoulders makes the model
    invariant to the signer's position in the frame.
    
    Args:
        landmarks: Shape (seq_len, feature_dim)
        shoulder_indices: Indices of left and right shoulder (in x,y,z triplets)
    
    Returns:
        Centered landmarks
    """
    # Get shoulder positions (assuming x,y,z format)
    left_idx = shoulder_indices[0] * 3
    right_idx = shoulder_indices[1] * 3
    
    # Calculate shoulder midpoint for each frame
    left_shoulder = landmarks[:, left_idx:left_idx+3]
    right_shoulder = landmarks[:, right_idx:right_idx+3]
    center = (left_shoulder + right_shoulder) / 2
    
    # Subtract center from all landmarks
    # Need to tile center for all landmark groups
    centered = landmarks.copy()
    feature_dim = landmarks.shape[1]
    
    for i in range(0, feature_dim, 3):
        centered[:, i:i+3] -= center
    
    return centered


def add_temporal_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Add velocity and acceleration features.
    
    These capture the motion dynamics which are crucial for
    understanding sign language.
    
    Args:
        landmarks: Shape (seq_len, feature_dim)
    
    Returns:
        Landmarks with velocity and acceleration appended
        Shape: (seq_len, feature_dim * 3)
    """
    # Velocity: difference between consecutive frames
    velocity = np.zeros_like(landmarks)
    velocity[1:] = landmarks[1:] - landmarks[:-1]
    
    # Acceleration: difference of velocity
    acceleration = np.zeros_like(landmarks)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Concatenate
    return np.concatenate([landmarks, velocity, acceleration], axis=-1)


def pad_or_truncate(
    landmarks: np.ndarray,
    target_length: int,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Pad or truncate sequence to target length.
    
    Args:
        landmarks: Shape (seq_len, feature_dim)
        target_length: Desired sequence length
        pad_value: Value to use for padding
    
    Returns:
        Padded/truncated landmarks and actual length
    """
    actual_length = len(landmarks)
    
    if actual_length >= target_length:
        # Truncate
        return landmarks[:target_length], target_length
    
    # Pad
    padding = np.full(
        (target_length - actual_length, landmarks.shape[1]),
        pad_value,
        dtype=landmarks.dtype
    )
    padded = np.concatenate([landmarks, padding], axis=0)
    return padded, actual_length


def augment_landmarks(
    landmarks: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    """
    Apply data augmentation to landmarks.
    
    Augmentations:
    1. Gaussian noise
    2. Random scaling
    3. Time stretching (interpolation)
    
    Args:
        landmarks: Shape (seq_len, feature_dim)
        noise_std: Standard deviation of Gaussian noise
        scale_range: Range for random scaling
        time_stretch_range: Range for time stretching
    
    Returns:
        Augmented landmarks
    """
    augmented = landmarks.copy()
    
    # 1. Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, landmarks.shape)
        augmented = augmented + noise
    
    # 2. Random scaling
    scale = np.random.uniform(*scale_range)
    augmented = augmented * scale
    
    # 3. Time stretching
    stretch = np.random.uniform(*time_stretch_range)
    if stretch != 1.0:
        original_len = len(augmented)
        new_len = int(original_len * stretch)
        
        if new_len > 0:
            # Linear interpolation
            old_indices = np.linspace(0, original_len - 1, new_len)
            new_landmarks = np.zeros((new_len, augmented.shape[1]))
            
            for i, idx in enumerate(old_indices):
                low = int(idx)
                high = min(low + 1, original_len - 1)
                weight = idx - low
                new_landmarks[i] = (1 - weight) * augmented[low] + weight * augmented[high]
            
            augmented = new_landmarks
    
    return augmented.astype(np.float32)


def preprocess_landmarks(
    landmarks: np.ndarray,
    max_length: int = 300,
    stats: Optional[PreprocessingStats] = None,
    add_temporal: bool = True,
    center: bool = True,
    augment: bool = False,
    augment_params: Optional[dict] = None
) -> Tuple[np.ndarray, int, PreprocessingStats]:
    """
    Full preprocessing pipeline for landmarks.
    
    Pipeline:
    1. Center pose (optional)
    2. Add temporal features (optional)
    3. Normalize
    4. Augment (optional, training only)
    5. Pad/truncate
    
    Args:
        landmarks: Raw landmarks (seq_len, feature_dim)
        max_length: Maximum sequence length
        stats: Normalization statistics (compute if None)
        add_temporal: Whether to add velocity/acceleration
        center: Whether to center pose
        augment: Whether to apply augmentation
        augment_params: Augmentation parameters
    
    Returns:
        Processed landmarks, actual length, statistics
    """
    processed = landmarks.copy()
    
    # 1. Center pose
    if center:
        processed = center_pose(processed)
    
    # 2. Add temporal features
    if add_temporal:
        processed = add_temporal_features(processed)
    
    # 3. Normalize
    processed, stats = normalize_landmarks(processed, stats)
    
    # 4. Augment (before padding)
    if augment:
        params = augment_params or {}
        processed = augment_landmarks(processed, **params)
    
    # 5. Pad/truncate
    processed, actual_length = pad_or_truncate(processed, max_length)
    
    return processed.astype(np.float32), actual_length, stats


def test_preprocessing():
    """Test preprocessing functions."""
    print("Testing preprocessing...")
    
    # Create dummy landmarks (100 frames, 201 features)
    landmarks = np.random.randn(100, 201).astype(np.float32)
    
    # Test centering
    centered = center_pose(landmarks)
    print(f"Centered shape: {centered.shape}")
    
    # Test temporal features
    temporal = add_temporal_features(landmarks)
    print(f"With temporal features: {temporal.shape}")  # Should be (100, 603)
    
    # Test normalization
    normalized, stats = normalize_landmarks(landmarks)
    print(f"Normalized mean: {normalized.mean():.4f}, std: {normalized.std():.4f}")
    
    # Test padding
    padded, length = pad_or_truncate(landmarks, 150)
    print(f"Padded shape: {padded.shape}, actual length: {length}")
    
    truncated, length = pad_or_truncate(landmarks, 50)
    print(f"Truncated shape: {truncated.shape}, actual length: {length}")
    
    # Test augmentation
    augmented = augment_landmarks(landmarks)
    print(f"Augmented shape: {augmented.shape}")
    
    # Test full pipeline
    processed, length, stats = preprocess_landmarks(
        landmarks, max_length=150, add_temporal=True, center=True
    )
    print(f"Full pipeline output: {processed.shape}, length: {length}")
    
    print("All preprocessing tests PASSED!")


if __name__ == "__main__":
    test_preprocessing()
