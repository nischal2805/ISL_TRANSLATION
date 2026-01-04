"""Video Dataset for ISL Translation with VideoMAE"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
from transformers import VideoMAEImageProcessor


class ISLVideoDataset(Dataset):
    """
    PyTorch Dataset for ISL videos (for VideoMAE encoder).
    
    Loads actual video frames instead of landmarks.
    """
    
    def __init__(
        self, 
        video_dir: str,
        csv_path: str,
        split: str,
        tokenizer,
        num_frames: int = 16,  # VideoMAE expects 16 frames
        max_text_len: int = 100,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            video_dir: Path to directory containing .mp4 files
            csv_path: Path to CSV with video_id, text, split columns
            split: 'train', 'val', or 'test'
            tokenizer: Text tokenizer
            num_frames: Number of frames to sample from each video
            max_text_len: Maximum text sequence length
            image_size: Frame resize dimension (H, W)
        """
        self.video_dir = video_dir
        self.split = split
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.max_text_len = max_text_len
        self.image_size = image_size
        
        # Load metadata
        df = pd.read_csv(csv_path)
        
        # Filter by split if split column exists
        if 'split' in df.columns:
            self.samples = df[df['split'] == split].reset_index(drop=True)
        else:
            # Split manually if no split column
            # Use deterministic split based on video_id hash
            df['_hash'] = df['uid'].apply(lambda x: hash(x) % 100)
            if split == 'train':
                self.samples = df[df['_hash'] < 80].reset_index(drop=True)
            elif split == 'val':
                self.samples = df[(df['_hash'] >= 80) & (df['_hash'] < 90)].reset_index(drop=True)
            else:  # test
                self.samples = df[df['_hash'] >= 90].reset_index(drop=True)
        
        # Initialize VideoMAE processor
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        
        print(f"ISLVideoDataset [{split}]: {len(self.samples)} videos")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path: str) -> Optional[np.ndarray]:
        """
        Load and sample frames from video.
        
        Returns:
            frames: (num_frames, H, W, 3) numpy array or None if error
        """
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            # If video is shorter, repeat last frame
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        else:
            # Uniformly sample frames
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, self.image_size)
                frames.append(frame)
            else:
                # If frame read fails, duplicate last valid frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # Fallback: black frame
                    frames.append(np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8))
        
        cap.release()
        
        return np.stack(frames)  # (num_frames, H, W, 3)
    
    def __getitem__(self, idx) -> Dict:
        row = self.samples.iloc[idx]
        
        # Get video path
        video_id = row['uid']
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        
        # Load video frames
        frames = self._load_video_frames(video_path)
        
        if frames is None:
            # Fallback: create black video
            frames = np.zeros((self.num_frames, self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        
        # Process frames using VideoMAE processor
        # Converts to tensor and normalizes
        pixel_values = self.processor(list(frames), return_tensors="pt")['pixel_values']
        pixel_values = pixel_values.squeeze(0)  # (T, C, H, W) - processor output format
        
        # Tokenize text
        text = row['text']
        if pd.isna(text) or text is None or str(text).strip() == '':
            text = 'unknown'  # Fallback for empty text
        
        try:
            tokens = self.tokenizer.encode(str(text), add_special_tokens=False)
            if tokens is None or len(tokens) == 0:
                tokens = [100]  # [UNK] token ID for BERT
        except Exception:
            tokens = [100]  # [UNK] token ID for BERT
        
        # BERT token IDs: [CLS]=101, [SEP]=102, [PAD]=0, [UNK]=100
        bos_id = 101 if self.tokenizer.bos_token_id is None else self.tokenizer.bos_token_id
        eos_id = 102 if self.tokenizer.eos_token_id is None else self.tokenizer.eos_token_id
        pad_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        
        tokens = [bos_id] + tokens[:self.max_text_len-2] + [eos_id]
        text_len = len(tokens)
        tokens = tokens + [pad_id] * (self.max_text_len - len(tokens))
        
        # Ensure tokens is valid list of integers
        if tokens is None or not isinstance(tokens, list):
            tokens = [bos_id, eos_id] + [pad_id] * (self.max_text_len - 2)
            text_len = 2
        
        return {
            'video_frames': pixel_values,  # (T, C, H, W)
            'feature_len': self.num_frames,
            'targets': torch.tensor(tokens, dtype=torch.long),
            'target_len': text_len,
            'text': text,
            'video_id': video_id
        }


def collate_video_fn(batch):
    """Collate function for video batches."""
    return {
        'video_frames': torch.stack([x['video_frames'] for x in batch]),  # (B, T, C, H, W)
        'feature_lengths': torch.tensor([x['feature_len'] for x in batch]),
        'targets': torch.stack([x['targets'] for x in batch]),
        'target_lengths': torch.tensor([x['target_len'] for x in batch]),
        'texts': [x['text'] for x in batch],
        'video_ids': [x['video_id'] for x in batch]
    }


def create_video_dataloaders(
    video_dir: str,
    csv_path: str,
    tokenizer,
    batch_size: int = 8,  # Smaller batch for video data
    num_workers: int = 4,
    num_frames: int = 16
):
    """
    Create dataloaders for video-based training.
    
    Args:
        video_dir: Directory with .mp4 files
        csv_path: CSV with metadata
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Frames per video clip
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = ISLVideoDataset(video_dir, csv_path, 'train', tokenizer, num_frames)
    val_dataset = ISLVideoDataset(video_dir, csv_path, 'val', tokenizer, num_frames)
    test_dataset = ISLVideoDataset(video_dir, csv_path, 'test', tokenizer, num_frames)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_video_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_video_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
