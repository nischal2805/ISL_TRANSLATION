"""
ISL Translation System - Dataset Module
========================================
PyTorch Dataset and DataLoader utilities.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import data_config, training_config
from vocab import Vocabulary


class ISLDataset(Dataset):
    """Dataset for ISL Translation."""
    
    def __init__(
        self,
        metadata_path: str,
        features_dir: str,
        vocab: Vocabulary,
        split: str = 'train',
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            metadata_path: Path to metadata.csv
            features_dir: Directory containing preprocessed .npy files
            vocab: Vocabulary instance
            split: One of 'train', 'val', 'test'
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.vocab = vocab
        self.features_dir = features_dir
        self.split = split
        self.max_src_len = max_src_len or data_config.max_src_len
        self.max_tgt_len = max_tgt_len or data_config.max_tgt_len
        
        # Load metadata
        metadata = pd.read_csv(metadata_path)
        self.data = metadata[metadata['split'] == split].reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - features: (T, 414) tensor
                - target_ids: (L,) tensor with token IDs
                - text: Original text string
                - video_id: Video identifier
        """
        row = self.data.iloc[idx]
        
        # Load preprocessed features
        features_path = row['path']
        features = np.load(features_path)
        
        # Truncate if needed
        if features.shape[0] > self.max_src_len:
            features = features[:self.max_src_len]
        
        # Encode text
        text = row['text']
        target_ids = self.vocab.encode(text)
        
        # Truncate target if needed
        if len(target_ids) > self.max_tgt_len:
            target_ids = target_ids[:self.max_tgt_len - 1] + [self.vocab.eos_id]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'text': text,
            'video_id': row['video_id']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Pads sequences to same length within batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors with:
            - features: (B, T_max, 414)
            - feature_lengths: (B,) - actual lengths before padding
            - targets: (B, L_max)
            - target_lengths: (B,) - actual lengths before padding
            - texts: List of original texts
            - video_ids: List of video identifiers
    """
    # Get batch size
    batch_size = len(batch)
    
    # Find max lengths
    max_src_len = max(sample['features'].shape[0] for sample in batch)
    max_tgt_len = max(sample['target_ids'].shape[0] for sample in batch)
    
    # Get feature dimension
    feat_dim = batch[0]['features'].shape[1]
    
    # Initialize padded tensors
    features = torch.zeros(batch_size, max_src_len, feat_dim, dtype=torch.float32)
    targets = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    texts = []
    video_ids = []
    
    # Fill tensors
    for i, sample in enumerate(batch):
        src_len = sample['features'].shape[0]
        tgt_len = sample['target_ids'].shape[0]
        
        features[i, :src_len] = sample['features']
        targets[i, :tgt_len] = sample['target_ids']
        
        feature_lengths[i] = src_len
        target_lengths[i] = tgt_len
        
        texts.append(sample['text'])
        video_ids.append(sample['video_id'])
    
    return {
        'features': features,
        'feature_lengths': feature_lengths,
        'targets': targets,
        'target_lengths': target_lengths,
        'texts': texts,
        'video_ids': video_ids
    }


def create_dataloaders(
    metadata_path: str,
    features_dir: str,
    vocab: Vocabulary,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        metadata_path: Path to metadata.csv
        features_dir: Directory containing preprocessed features
        vocab: Vocabulary instance
        batch_size: Batch size (uses config default if None)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or training_config.batch_size
    
    # Create datasets
    train_dataset = ISLDataset(
        metadata_path, features_dir, vocab, split='train'
    )
    val_dataset = ISLDataset(
        metadata_path, features_dir, vocab, split='val'
    )
    test_dataset = ISLDataset(
        metadata_path, features_dir, vocab, split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


class StreamingDataset(Dataset):
    """
    Streaming dataset that loads features on-the-fly.
    Use when dataset doesn't fit in memory.
    """
    
    def __init__(
        self,
        video_list: List[Dict],
        vocab: Vocabulary,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None
    ):
        """
        Initialize streaming dataset.
        
        Args:
            video_list: List of dicts with 'path' and 'text' keys
            vocab: Vocabulary instance
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.video_list = video_list
        self.vocab = vocab
        self.max_src_len = max_src_len or data_config.max_src_len
        self.max_tgt_len = max_tgt_len or data_config.max_tgt_len
    
    def __len__(self) -> int:
        return len(self.video_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.video_list[idx]
        
        # Load features
        features = np.load(item['path'])
        
        if features.shape[0] > self.max_src_len:
            features = features[:self.max_src_len]
        
        # Encode text
        target_ids = self.vocab.encode(item['text'])
        
        if len(target_ids) > self.max_tgt_len:
            target_ids = target_ids[:self.max_tgt_len - 1] + [self.vocab.eos_id]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'text': item['text'],
            'video_id': item.get('video_id', f'sample_{idx}')
        }


# Augmentation functions
def augment_features(
    features: np.ndarray,
    noise_std: float = 0.02,
    time_warp_prob: float = 0.3,
    dropout_prob: float = 0.1
) -> np.ndarray:
    """
    Apply data augmentation to features.
    
    Args:
        features: (T, 414) feature array
        noise_std: Standard deviation of Gaussian noise
        time_warp_prob: Probability of time warping
        dropout_prob: Probability of dropping frames
        
    Returns:
        Augmented features
    """
    augmented = features.copy()
    
    # Add Gaussian noise
    if noise_std > 0:
        noise = np.random.randn(*augmented.shape) * noise_std
        augmented = augmented + noise
    
    # Time warping (stretch/compress)
    if np.random.random() < time_warp_prob:
        T = augmented.shape[0]
        scale = np.random.uniform(0.8, 1.2)
        new_T = max(1, int(T * scale))
        
        indices = np.linspace(0, T - 1, new_T)
        indices = np.clip(indices, 0, T - 1).astype(int)
        augmented = augmented[indices]
    
    # Frame dropout
    if dropout_prob > 0:
        mask = np.random.random(augmented.shape[0]) > dropout_prob
        if mask.sum() > 0:
            augmented = augmented[mask]
    
    return augmented.astype(np.float32)


class AugmentedDataset(ISLDataset):
    """Dataset with on-the-fly augmentation for training."""
    
    def __init__(
        self,
        metadata_path: str,
        features_dir: str,
        vocab: Vocabulary,
        split: str = 'train',
        augment: bool = True,
        noise_std: float = 0.02,
        time_warp_prob: float = 0.3,
        dropout_prob: float = 0.1
    ):
        super().__init__(metadata_path, features_dir, vocab, split)
        
        self.augment = augment and (split == 'train')
        self.noise_std = noise_std
        self.time_warp_prob = time_warp_prob
        self.dropout_prob = dropout_prob
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        
        if self.augment:
            features = sample['features'].numpy()
            features = augment_features(
                features,
                noise_std=self.noise_std,
                time_warp_prob=self.time_warp_prob,
                dropout_prob=self.dropout_prob
            )
            sample['features'] = torch.tensor(features, dtype=torch.float32)
        
        return sample


if __name__ == "__main__":
    # Test dataset utilities
    print("ISL Dataset Test")
    print("=" * 50)
    
    # Test collate function with dummy data
    print("\nTesting collate function...")
    
    vocab = Vocabulary()
    
    dummy_batch = [
        {
            'features': torch.randn(100, 414),
            'target_ids': torch.tensor(vocab.encode("hello")),
            'text': 'hello',
            'video_id': 'test_1'
        },
        {
            'features': torch.randn(80, 414),
            'target_ids': torch.tensor(vocab.encode("world")),
            'text': 'world',
            'video_id': 'test_2'
        },
        {
            'features': torch.randn(120, 414),
            'target_ids': torch.tensor(vocab.encode("test")),
            'text': 'test',
            'video_id': 'test_3'
        }
    ]
    
    batch = collate_fn(dummy_batch)
    
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")
    print(f"Feature lengths: {batch['feature_lengths']}")
    print(f"Target lengths: {batch['target_lengths']}")
    
    # Test augmentation
    print("\nTesting augmentation...")
    dummy_features = np.random.randn(100, 414).astype(np.float32)
    augmented = augment_features(dummy_features)
    print(f"Original shape: {dummy_features.shape}")
    print(f"Augmented shape: {augmented.shape}")
