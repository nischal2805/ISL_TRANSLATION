"""
Dataset for ISL Translation V2 with BPE Tokenizer
=================================================
"""

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
import pandas as pd


class ISLDatasetV2(Dataset):
    """
    Dataset for ISL Translation V2.
    
    Uses BPE tokenizer and preprocessed landmarks with velocity/acceleration.
    Supports both split-based (train/val/test folders) and flat directory structure.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        tokenizer,
        split: str = 'train',
        max_src_len: int = 500,
        max_tgt_len: int = 100
    ):
        """
        Args:
            data_dir: Directory with .npy feature files
            metadata_path: Path to metadata.csv
            tokenizer: BPETokenizer instance
            split: 'train', 'val', or 'test'
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.split = split
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Load metadata
        metadata = pd.read_csv(metadata_path)
        self.samples = metadata[metadata['split'] == split].reset_index(drop=True)
        
        # Check if split folders exist or flat structure
        self.use_split_folders = (self.data_dir / split).exists()
        
        print(f"Loaded {len(self.samples)} {split} samples")
        print(f"  Data structure: {'split folders' if self.use_split_folders else 'flat directory'}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]
        
        # Load features - check both split folder and flat structure
        if self.use_split_folders:
            feature_path = self.data_dir / self.split / f"{row['video_id']}.npy"
        else:
            feature_path = self.data_dir / f"{row['video_id']}.npy"
        
        features = np.load(feature_path)
        
        # Truncate if needed
        if features.shape[0] > self.max_src_len:
            features = features[:self.max_src_len]
        
        # Tokenize text with BPE
        text = row['text']
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_length=self.max_tgt_len)
        
        return {
            'features': torch.FloatTensor(features),
            'feature_length': features.shape[0],
            'targets': torch.LongTensor(token_ids),
            'target_length': len(token_ids),
            'text': text
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    
    # Get max lengths
    max_feat_len = max(item['feature_length'] for item in batch)
    max_tgt_len = max(item['target_length'] for item in batch)
    
    batch_size = len(batch)
    feat_dim = batch[0]['features'].shape[1]
    
    # Initialize padded tensors
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


def create_dataloaders_v2(
    data_dir: str,
    metadata_path: str,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    max_src_len: int = 500,
    max_tgt_len: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = ISLDatasetV2(
        data_dir, metadata_path, tokenizer, 'train',
        max_src_len, max_tgt_len
    )
    val_dataset = ISLDatasetV2(
        data_dir, metadata_path, tokenizer, 'val',
        max_src_len, max_tgt_len
    )
    test_dataset = ISLDatasetV2(
        data_dir, metadata_path, tokenizer, 'test',
        max_src_len, max_tgt_len
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class SimpleISLDataset(Dataset):
    """Simple dataset that loads from separate train/val/test folders without metadata."""
    
    def __init__(self, data_dir: str, uid_to_text: dict, tokenizer, max_src_len=500, max_tgt_len=100):
        self.data_dir = Path(data_dir)
        self.uid_to_text = uid_to_text
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Get all .npy files that have corresponding valid text
        self.files = []
        for f in self.data_dir.glob('*.npy'):
            if f.stem in uid_to_text:
                text = uid_to_text[f.stem]
                # Skip if text is NaN or not a string
                if isinstance(text, str) and len(text.strip()) > 0:
                    self.files.append(f)
        
        print(f"Loaded {len(self.files)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        features = np.load(f)
        
        # Truncate if needed
        if features.shape[0] > self.max_src_len:
            features = features[:self.max_src_len]
        
        # Get text and tokenize
        text = str(self.uid_to_text[f.stem])  # Ensure string
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True, max_length=self.max_tgt_len)
        
        return {
            'features': torch.FloatTensor(features),
            'feature_length': features.shape[0],
            'targets': torch.LongTensor(token_ids),
            'target_length': len(token_ids),
            'text': text
        }


def create_simple_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    uid_to_text: dict,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    max_src_len: int = 500,
    max_tgt_len: int = 100
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders from separate train/val/test directories."""
    
    train_dataset = SimpleISLDataset(train_dir, uid_to_text, tokenizer, max_src_len, max_tgt_len)
    val_dataset = SimpleISLDataset(val_dir, uid_to_text, tokenizer, max_src_len, max_tgt_len)
    test_dataset = SimpleISLDataset(test_dir, uid_to_text, tokenizer, max_src_len, max_tgt_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

