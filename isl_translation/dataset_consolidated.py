"""
Dataset for ISL Translation - CONSOLIDATED FORMAT
=================================================
Loads from single .npy files (train_features.npy, etc.)
Much faster loading than individual files!
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path


class ISLDatasetConsolidated(Dataset):
    """
    Dataset for consolidated .npy format.
    
    Expected files in data_dir:
    - {split}_features.npy: (N, max_T, 540) padded features
    - {split}_lengths.npy: (N,) actual sequence lengths
    - video_metadata.csv: video_id, text, num_frames, split
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        split: str = 'train',
        max_src_len: int = 500,
        max_tgt_len: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load features (memory-mapped for efficiency)
        features_path = self.data_dir / f'{split}_features.npy'
        lengths_path = self.data_dir / f'{split}_lengths.npy'
        
        print(f"Loading {split} features from {features_path}...")
        self.features = np.load(features_path, mmap_mode='r')  # Memory-mapped!
        self.lengths = np.load(lengths_path)
        
        # Load metadata
        meta_path = self.data_dir / 'video_metadata.csv'
        metadata = pd.read_csv(meta_path)
        self.metadata = metadata[metadata['split'] == split].reset_index(drop=True)
        
        print(f"  Loaded {len(self)} {split} samples")
        print(f"  Features shape: {self.features.shape}")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get actual length (not padded length)
        length = min(self.lengths[idx], self.max_src_len)
        
        # Get features (only up to actual length)
        features = self.features[idx, :length].copy()  # .copy() because mmap_mode='r'
        
        # Get text and tokenize
        text = self.metadata.iloc[idx]['text']
        token_ids = self.tokenizer.encode(
            text, add_bos=True, add_eos=True, max_length=self.max_tgt_len
        )
        
        return {
            'features': torch.FloatTensor(features),
            'feature_length': length,
            'targets': torch.LongTensor(token_ids),
            'target_length': len(token_ids),
            'text': text
        }


def collate_fn_consolidated(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    
    max_feat_len = max(item['feature_length'] for item in batch)
    max_tgt_len = max(item['target_length'] for item in batch)
    
    batch_size = len(batch)
    feat_dim = batch[0]['features'].shape[1]
    
    features = torch.zeros(batch_size, max_feat_len, feat_dim)
    feature_lengths = torch.zeros(batch_size, dtype=torch.long)
    targets = torch.zeros(batch_size, max_tgt_len, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    texts = []
    
    for i, item in enumerate(batch):
        feat_len = item['feature_length']
        tgt_len = item['target_length']
        
        features[i, :feat_len] = item['features']
        feature_lengths[i] = feat_len
        targets[i, :tgt_len] = item['targets']
        target_lengths[i] = tgt_len
        texts.append(item['text'])
    
    return {
        'features': features,
        'feature_lengths': feature_lengths,
        'targets': targets,
        'target_lengths': target_lengths,
        'texts': texts
    }


def create_dataloaders_consolidated(
    data_dir: str,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    max_src_len: int = 500,
    max_tgt_len: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from consolidated .npy files."""
    
    train_dataset = ISLDatasetConsolidated(
        data_dir, tokenizer, 'train', max_src_len, max_tgt_len
    )
    val_dataset = ISLDatasetConsolidated(
        data_dir, tokenizer, 'val', max_src_len, max_tgt_len
    )
    test_dataset = ISLDatasetConsolidated(
        data_dir, tokenizer, 'test', max_src_len, max_tgt_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_consolidated,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_consolidated,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_consolidated,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# For backward compatibility with train_v2.py
def create_dataloaders_v2(
    data_dir: str,
    metadata_path: str,  # Ignored, uses video_metadata.csv in data_dir
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    max_src_len: int = 500,
    max_tgt_len: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrapper for compatibility."""
    return create_dataloaders_consolidated(
        data_dir, tokenizer, batch_size, num_workers, max_src_len, max_tgt_len
    )
