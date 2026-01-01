"""
Generate metadata.csv for ISL Translation Training
===================================================
Creates the metadata file required by the training pipeline.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


def main():
    # ============ EDIT THESE PATHS ============
    project_root = Path(__file__).parent.resolve()
    
    annotations_path = project_root / "data/iSign_v1.1.csv"  # Path to annotations CSV
    landmarks_dir = project_root / "data/landmarks/train/op"    # Directory with .npy files (already extracted)
    output_dir = project_root / "data/landmarks"             # Output directory
    
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    seed = 42
    copy_files = True  # Set to True to copy files to val/test dirs
    # ==========================================
    
    # Also check if annotations is in root
    if not annotations_path.exists():
        alt_path = project_root / "iSign_v1.1.csv"
        if alt_path.exists():
            annotations_path = alt_path
            print(f"Using annotations from: {annotations_path}")
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {annotations_path}")
    df = pd.read_csv(annotations_path)
    print(f"  Total annotations: {len(df)}")
    
    # Get available landmark files
    if not landmarks_dir.exists():
        print(f"ERROR: Landmarks directory not found: {landmarks_dir}")
        print(f"Make sure your .npy files are in: {landmarks_dir}")
        return
    
    npy_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} landmark files in {landmarks_dir}")
    
    if len(npy_files) == 0:
        print("ERROR: No .npy files found!")
        print(f"Please copy your extracted .npy files to: {landmarks_dir}")
        return
    
    # Match annotations with landmark files
    metadata = []
    missing = 0
    too_short = 0
    
    for npy_file in npy_files:
        video_id = npy_file.replace('.npy', '')
        
        # Find matching annotation
        row = df[df['uid'] == video_id]
        if len(row) == 0:
            missing += 1
            continue
        
        # Load and check length
        npy_path = landmarks_dir / npy_file
        data = np.load(npy_path)
        
        if data.shape[0] < 4:  # min_src_len
            too_short += 1
            continue
        
        metadata.append({
            'video_id': video_id,
            'text': row.iloc[0]['text'],
            'length': data.shape[0],
            'path': str(npy_path.resolve())
        })
    
    print(f"Matched: {len(metadata)}, Missing annotation: {missing}, Too short: {too_short}")
    
    if len(metadata) == 0:
        print("ERROR: No valid samples found!")
        return
    
    # Create DataFrame
    meta_df = pd.DataFrame(metadata)
    
    # Split into train/val/test
    np.random.seed(seed)
    
    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        meta_df, 
        train_size=train_ratio,
        random_state=seed
    )
    
    # Second split: val vs test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=seed
    )
    
    # Add split column
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    # Move files to split directories and update paths
    print("\nOrganizing files into split directories...")
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = output_dir / split_name
        new_paths = []
        
        for idx, row in split_df.iterrows():
            src_path = Path(row['path'])
            dst_path = split_dir / src_path.name
            
            # Copy file if not already in correct location
            if src_path.parent.resolve() != split_dir.resolve():
                if src_path.exists() and copy_files:
                    shutil.copy2(src_path, dst_path)
                    new_paths.append(str(dst_path.resolve()))
                else:
                    new_paths.append(str(src_path.resolve()))
            else:
                new_paths.append(str(dst_path.resolve()))
        
        split_df['path'] = new_paths
    
    # Combine and save
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    metadata_path = output_dir / 'metadata.csv'
    final_df.to_csv(metadata_path, index=False)
    
    print(f"\n✅ Metadata saved to: {metadata_path}")
    print(f"\nDataset Statistics:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Total: {len(final_df)} samples")
    
    # Print sample
    print(f"\nSample entries:")
    print(final_df.head())


if __name__ == "__main__":
    main()
