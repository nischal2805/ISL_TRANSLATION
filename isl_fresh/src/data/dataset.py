"""Dataset for ISL Translation"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class ISLDataset(Dataset):
    """PyTorch Dataset for ISL landmarks."""
    
    def __init__(self, data_dir: str, split: str, tokenizer, max_seq_len: int = 300, max_text_len: int = 100):
        self.data_dir = os.path.join(data_dir, split)
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.tokenizer = tokenizer
        
        # Load metadata
        meta_path = os.path.join(data_dir, 'metadata.csv')
        df = pd.read_csv(meta_path)
        self.samples = df[df['split'] == split].reset_index(drop=True)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict:
        row = self.samples.iloc[idx]
        
        # Load landmarks
        landmarks = np.load(os.path.join(self.data_dir, f"{row['video_id']}.npy"))
        
        # Use raw landmarks only (603 dims) - temporal features cause dimension mismatch
        # TODO: Add temporal features in a separate feature path once dimension handling is fixed
        features = landmarks
        
        # Pad/truncate
        seq_len = min(len(features), self.max_seq_len)
        if len(features) < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - len(features), features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        else:
            features = features[:self.max_seq_len]
        
        # Tokenize text
        tokens = self.tokenizer.encode(row['text'])
        tokens = [self.tokenizer.bos_token_id] + tokens[:self.max_text_len-2] + [self.tokenizer.eos_token_id]
        text_len = len(tokens)
        tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_text_len - len(tokens))
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'feature_len': seq_len,
            'targets': torch.tensor(tokens, dtype=torch.long),
            'target_len': text_len,
            'text': row['text']
        }


def collate_fn(batch):
    """Collate batch."""
    return {
        'features': torch.stack([x['features'] for x in batch]),
        'feature_lengths': torch.tensor([x['feature_len'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'target_lengths': torch.tensor([x['target_len'] for x in batch]),
        'texts': [x['text'] for x in batch]
    }
