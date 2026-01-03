"""
Generate metadata.csv from existing preprocessed .npy files.
Run this script on your remote server where the preprocessed data is stored.

This script will:
1. Scan directories for .npy files
2. Match them with annotations from iSign_v1.1.csv
3. Generate metadata.csv required for training
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def generate_metadata(
    preprocessed_dir: str,
    annotations_file: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate metadata.csv from preprocessed .npy files and annotations.
    
    Args:
        preprocessed_dir: Root directory containing preprocessed .npy files
                         Can have subdirectories like train/, val/, test/
                         Or all files in one directory
        annotations_file: Path to iSign_v1.1.csv with 'uid' and 'text' columns
        output_path: Where to save metadata.csv (default: preprocessed_dir/metadata.csv)
    
    Returns:
        DataFrame with metadata
    """
    print("=" * 60)
    print("Generating metadata.csv")
    print("=" * 60)
    
    # Load annotations
    print(f"\nLoading annotations from: {annotations_file}")
    annotations = pd.read_csv(annotations_file)
    
    # Check for uid or video_id column
    id_column = None
    for col in ['uid', 'video_id', 'id', 'filename']:
        if col in annotations.columns:
            id_column = col
            break
    
    if id_column is None:
        print("ERROR: Could not find video ID column in annotations.")
        print(f"Available columns: {list(annotations.columns)}")
        return None
    
    print(f"Using '{id_column}' as video ID column")
    print(f"Total annotations: {len(annotations)}")
    
    # Create lookup dictionary: video_id -> text
    id_to_text = dict(zip(annotations[id_column].astype(str), annotations['text']))
    
    # Find all .npy files
    print(f"\nScanning for .npy files in: {preprocessed_dir}")
    
    npy_files = []
    preprocessed_path = Path(preprocessed_dir)
    
    # Check for split subdirectories
    split_dirs = ['train', 'val', 'test']
    has_split_dirs = any((preprocessed_path / split).exists() for split in split_dirs)
    
    if has_split_dirs:
        print("Found split directories (train/val/test)")
        for split in split_dirs:
            split_path = preprocessed_path / split
            if split_path.exists():
                for npy_file in split_path.glob("*.npy"):
                    npy_files.append((npy_file, split))
    else:
        print("No split directories found, will assign splits automatically")
        for npy_file in preprocessed_path.rglob("*.npy"):
            npy_files.append((npy_file, None))
    
    print(f"Found {len(npy_files)} .npy files")
    
    if len(npy_files) == 0:
        print("ERROR: No .npy files found!")
        return None
    
    # Process each file
    metadata_list = []
    missing_text = 0
    
    for npy_path, split in npy_files:
        # Extract video_id from filename
        # Filename format: _z80Fp9SjhE--6.npy -> video_id is _z80Fp9SjhE
        filename_stem = npy_path.stem
        if '--' in filename_stem:
            video_id = filename_stem.rsplit('--', 1)[0]
        else:
            video_id = filename_stem
        
        # Get text from annotations
        text = id_to_text.get(video_id, None)
        
        if text is None:
            missing_text += 1
            continue
        
        # Get feature length
        try:
            features = np.load(str(npy_path))
            length = features.shape[0]
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            continue
        
        metadata_list.append({
            'video_id': video_id,
            'text': text,
            'split': split,  # May be None if no split dirs
            'length': length,
            'path': str(npy_path)
        })
    
    print(f"\nMatched {len(metadata_list)} files with annotations")
    if missing_text > 0:
        print(f"Warning: {missing_text} files had no matching annotation")
    
    # Create DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    
    # Assign splits if not already assigned
    if not has_split_dirs or metadata_df['split'].isna().any():
        print("\nAssigning train/val/test splits (70/15/15)...")
        n_total = len(metadata_df)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        # Shuffle
        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Assign splits
        metadata_df.loc[:n_train-1, 'split'] = 'train'
        metadata_df.loc[n_train:n_train+n_val-1, 'split'] = 'val'
        metadata_df.loc[n_train+n_val:, 'split'] = 'test'
    
    # Print statistics
    print("\n" + "=" * 40)
    print("Dataset Statistics:")
    print("=" * 40)
    for split in ['train', 'val', 'test']:
        count = len(metadata_df[metadata_df['split'] == split])
        print(f"  {split}: {count} samples")
    print(f"  Total: {len(metadata_df)} samples")
    
    # Save metadata
    if output_path is None:
        output_path = os.path.join(preprocessed_dir, 'metadata.csv')
    
    metadata_df.to_csv(output_path, index=False)
    print(f"\nMetadata saved to: {output_path}")
    
    return metadata_df


def main():
    """Main entry point with example usage."""
    
    # ============================================================
    # GPU SERVER PATHS
    # ============================================================
    
    # Path to your preprocessed .npy files
    PREPROCESSED_DIR = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final"
    
    # Path to your annotations CSV (iSign_v1.1.csv)
    ANNOTATIONS_FILE = "/media/rvcse22/CSERV/kortex_sem5/ramita/iSign_v1.1.csv"
    
    # Where to save metadata.csv (default: inside PREPROCESSED_DIR)
    OUTPUT_PATH = None  # Will save as PREPROCESSED_DIR/metadata.csv
    
    # ============================================================
    
    # Check if paths exist
    if not os.path.exists(PREPROCESSED_DIR):
        print(f"ERROR: Preprocessed directory not found: {PREPROCESSED_DIR}")
        print("\nPlease update the paths in this script:")
        print("  PREPROCESSED_DIR = path to your .npy files")
        print("  ANNOTATIONS_FILE = path to iSign_v1.1.csv")
        return
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"ERROR: Annotations file not found: {ANNOTATIONS_FILE}")
        return
    
    # Generate metadata
    metadata = generate_metadata(
        preprocessed_dir=PREPROCESSED_DIR,
        annotations_file=ANNOTATIONS_FILE,
        output_path=OUTPUT_PATH
    )
    
    if metadata is not None:
        print("\n" + "=" * 60)
        print("SUCCESS! metadata.csv generated.")
        print("You can now run training with:")
        print("  python run_training.py")
        print("=" * 60)


if __name__ == "__main__":
    main()