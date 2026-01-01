"""
Fast Preprocessing for Existing Landmarks
==========================================
Adds velocity and acceleration features to already-extracted landmarks.
Processes 127K files in ~30 minutes using multiprocessing.
"""

import os
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from tqdm import tqdm
import time


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Input: Your existing extracted landmarks (204 dims)
    input_dir = Path(r"E:\5thsem el\kortex_5th_sem\extracted_landmarks_v2")
    
    # Output: Preprocessed with velocity/acceleration (612 dims)
    output_dir = Path(r"E:\5thsem el\APPROACH 2\preprocessed_v2")
    
    # Annotations
    csv_path = Path(r"E:\5thsem el\kortex_5th_sem\data\iSign_v1.1.csv")
    
    # Feature dimensions
    raw_dim = 204  # Your extracted landmarks dimension
    output_dim = 612  # 204 * 3 (pos + vel + acc)
    
    # Preprocessing
    smoothing_sigma = 1.0
    
    # Workers (use physical cores, not threads)
    num_workers = 6  # Optimal for Ryzen 9 7940HS


# ============================================================================
# Preprocessing Functions
# ============================================================================

def compute_velocity(features: np.ndarray) -> np.ndarray:
    """Compute velocity (first derivative)."""
    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]
    return velocity


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """Compute acceleration (second derivative)."""
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def smooth_features(features: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing."""
    if sigma <= 0 or features.shape[0] < 3:
        return features
    return gaussian_filter1d(features, sigma=sigma, axis=0, mode='nearest')


def preprocess_single_file(args) -> dict:
    """Process a single .npy file - add velocity and acceleration."""
    input_path, output_path, video_id = args
    
    try:
        # Load raw landmarks
        landmarks = np.load(input_path).astype(np.float32)
        
        if landmarks.shape[0] == 0:
            return {'video_id': video_id, 'status': 'empty', 'length': 0}
        
        # Handle very short sequences
        if landmarks.shape[0] < 3:
            pad_length = 3 - landmarks.shape[0]
            landmarks = np.pad(landmarks, ((0, pad_length), (0, 0)), mode='edge')
        
        # Apply smoothing
        smoothed = smooth_features(landmarks, sigma=Config.smoothing_sigma)
        
        # Compute velocity
        velocity = compute_velocity(smoothed)
        velocity = smooth_features(velocity, sigma=Config.smoothing_sigma)
        
        # Compute acceleration
        acceleration = compute_acceleration(velocity)
        acceleration = smooth_features(acceleration, sigma=Config.smoothing_sigma)
        
        # Concatenate: (T, 204) -> (T, 612)
        features = np.concatenate([smoothed, velocity, acceleration], axis=1)
        
        # Save
        np.save(output_path, features.astype(np.float32))
        
        return {
            'video_id': video_id,
            'status': 'success',
            'length': features.shape[0],
            'path': str(output_path)
        }
        
    except Exception as e:
        return {'video_id': video_id, 'status': 'error', 'error': str(e)}


def create_metadata(output_dir: Path, csv_path: Path, results: list) -> pd.DataFrame:
    """Create metadata CSV with train/val/test splits."""
    
    # Load annotations
    annotations = pd.read_csv(csv_path)
    
    # Create mapping from results
    processed = {r['video_id']: r for r in results if r['status'] == 'success'}
    
    # Filter to only processed files
    metadata = []
    for _, row in annotations.iterrows():
        video_id = row['uid']
        if video_id in processed:
            metadata.append({
                'video_id': video_id,
                'text': row['text'],
                'length': processed[video_id]['length']
            })
    
    df = pd.DataFrame(metadata)
    
    # Random split: 70% train, 15% val, 15% test
    np.random.seed(42)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    df['split'] = 'test'
    df.iloc[indices[:train_end], df.columns.get_loc('split')] = 'train'
    df.iloc[indices[train_end:val_end], df.columns.get_loc('split')] = 'val'
    
    # Save metadata
    metadata_path = output_dir / 'metadata.csv'
    df.to_csv(metadata_path, index=False)
    
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val: {(df['split'] == 'val').sum()}")
    print(f"  Test: {(df['split'] == 'test').sum()}")
    
    return df


def main():
    """Main preprocessing pipeline."""
    config = Config()
    
    print("=" * 60)
    print("FAST PREPROCESSING - Add Velocity/Acceleration")
    print("=" * 60)
    print(f"Input:  {config.input_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Workers: {config.num_workers}")
    print(f"Feature dim: {config.raw_dim} -> {config.output_dim}")
    print("=" * 60)
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all input files
    input_files = list(config.input_dir.glob("*.npy"))
    print(f"\nFound {len(input_files)} input files")
    
    # Prepare tasks (skip already processed)
    tasks = []
    skipped = 0
    for input_path in input_files:
        video_id = input_path.stem
        output_path = config.output_dir / f"{video_id}.npy"
        
        if output_path.exists():
            skipped += 1
            continue
        
        tasks.append((input_path, output_path, video_id))
    
    print(f"Files to process: {len(tasks)} (skipping {skipped} already done)")
    
    if len(tasks) == 0:
        print("\n✅ All files already preprocessed!")
        
        # Still create metadata if needed
        metadata_path = config.output_dir / 'metadata.csv'
        if not metadata_path.exists():
            print("\nCreating metadata...")
            # Load all results
            results = []
            for f in config.output_dir.glob("*.npy"):
                data = np.load(f)
                results.append({
                    'video_id': f.stem,
                    'status': 'success',
                    'length': data.shape[0]
                })
            create_metadata(config.output_dir, config.csv_path, results)
        return
    
    # Process with multiprocessing
    start_time = time.time()
    
    print(f"\nProcessing {len(tasks)} files with {config.num_workers} workers...")
    
    results = []
    with Pool(config.num_workers) as pool:
        for result in tqdm(pool.imap(preprocess_single_file, tasks, chunksize=100),
                          total=len(tasks), desc="Preprocessing"):
            results.append(result)
    
    # Add skipped files to results
    for f in config.output_dir.glob("*.npy"):
        video_id = f.stem
        if video_id not in [r['video_id'] for r in results]:
            data = np.load(f)
            results.append({
                'video_id': video_id,
                'status': 'success',
                'length': data.shape[0]
            })
    
    elapsed = time.time() - start_time
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')
    empty = sum(1 for r in results if r['status'] == 'empty')
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Success: {success}")
    print(f"Empty: {empty}")
    print(f"Errors: {errors}")
    print(f"Rate: {len(tasks)/elapsed:.1f} files/sec")
    
    # Create metadata
    print("\nCreating metadata with train/val/test splits...")
    create_metadata(config.output_dir, config.csv_path, results)
    
    # Verify output
    sample = np.load(list(config.output_dir.glob("*.npy"))[0])
    print(f"\nOutput feature dimension: {sample.shape[1]} (expected {config.output_dim})")
    print("=" * 60)


if __name__ == '__main__':
    main()
