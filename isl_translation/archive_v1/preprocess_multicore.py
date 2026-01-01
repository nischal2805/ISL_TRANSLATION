"""
Multicore Preprocessing for ISL Translation System
===================================================
Uses multiprocessing to extract landmarks from videos in parallel.
Optimized for AMD Ryzen 9 7940HS (8 cores, 16 threads).
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import data_config, landmark_config, set_gpu_mode, training_config


# ============================================================================
# Worker Functions (must be at module level for multiprocessing)
# ============================================================================

def process_single_video(args: Tuple) -> Optional[Dict]:
    """
    Process a single video - worker function for multiprocessing.
    
    Args:
        args: Tuple of (video_id, text, split, videos_dir, output_dir)
        
    Returns:
        Metadata dict or None if failed
    """
    video_id, text, split, videos_dir, output_dir = args
    
    try:
        # Initialize MediaPipe for this worker
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,  # Lite for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        
        # Find video file
        video_path = os.path.join(videos_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(videos_dir, video_id)
            if not os.path.exists(video_path):
                return None
        
        # Extract landmarks
        landmarks = extract_landmarks(video_path, holistic)
        holistic.close()
        
        if landmarks is None:
            return None
        
        # Process features
        features = process_features(landmarks)
        
        # Validate length
        if features.shape[0] < 4:  # min_src_len
            return None
        
        if features.shape[0] > 500:  # max_src_len
            features = features[:500]
        
        # Save features
        output_path = os.path.join(output_dir, split, f"{video_id}.npy")
        np.save(output_path, features)
        
        return {
            'video_id': video_id,
            'text': text,
            'split': split,
            'length': features.shape[0],
            'path': output_path
        }
        
    except Exception as e:
        return None


def extract_landmarks(video_path: str, holistic) -> Optional[np.ndarray]:
    """Extract 46 landmarks from video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    pose_indices = [11, 12, 13, 14]  # Shoulders + elbows
    landmarks_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        frame_landmarks = []
        
        # Pose (4 points)
        if results.pose_landmarks:
            for idx in pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                frame_landmarks.append([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0]] * 4)
        
        # Left hand (21 points)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.append([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        # Right hand (21 points)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.append([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        landmarks_list.append(frame_landmarks)
    
    cap.release()
    
    if len(landmarks_list) == 0:
        return None
    
    return np.array(landmarks_list, dtype=np.float32)


def process_features(landmarks: np.ndarray) -> np.ndarray:
    """Process landmarks into 414-dim features (pos + vel + acc)."""
    T, num_landmarks, coords = landmarks.shape
    
    # Normalize to [-1, 1] based on pose reference
    landmarks = normalize_landmarks(landmarks)
    
    # Flatten: (T, 46, 3) -> (T, 138)
    positions = landmarks.reshape(T, -1)
    
    # Compute velocity and acceleration
    velocity = np.gradient(positions, axis=0)
    acceleration = np.gradient(velocity, axis=0)
    
    # Smooth with Gaussian filter
    sigma = 1.0
    positions = gaussian_filter1d(positions, sigma=sigma, axis=0)
    velocity = gaussian_filter1d(velocity, sigma=sigma, axis=0)
    acceleration = gaussian_filter1d(acceleration, sigma=sigma, axis=0)
    
    # Concatenate: (T, 414)
    features = np.concatenate([positions, velocity, acceleration], axis=1)
    
    return features.astype(np.float32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks using shoulder-based reference frame."""
    T = landmarks.shape[0]
    normalized = landmarks.copy()
    
    for t in range(T):
        # Reference: midpoint between shoulders (indices 0, 1)
        left_shoulder = landmarks[t, 0, :2]
        right_shoulder = landmarks[t, 1, :2]
        
        if np.allclose(left_shoulder, 0) and np.allclose(right_shoulder, 0):
            continue
        
        center = (left_shoulder + right_shoulder) / 2
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_dist < 1e-6:
            shoulder_dist = 1.0
        
        # Normalize XY
        normalized[t, :, :2] = (landmarks[t, :, :2] - center) / shoulder_dist
        # Z is already normalized by MediaPipe
    
    return normalized


def random_split(df: pd.DataFrame, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Random train/val/test split."""
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    return {
        'train': df.iloc[indices[:train_end]],
        'val': df.iloc[indices[train_end:val_end]],
        'test': df.iloc[indices[val_end:]]
    }


# ============================================================================
# Main Preprocessing Function
# ============================================================================

def preprocess_multicore(
    videos_dir: str,
    annotations_file: str,
    output_dir: str,
    subset_ratio: float = 1.0,
    num_workers: int = None,
    seed: int = 42
):
    """
    Multicore preprocessing of video dataset.
    
    Args:
        videos_dir: Directory containing .mp4 video files
        annotations_file: CSV with 'uid' and 'text' columns
        output_dir: Output directory for .npy feature files
        subset_ratio: Fraction of data to use (0.0-1.0)
        num_workers: Number of parallel workers (default: CPU count - 2)
        seed: Random seed
    """
    start_time = time.time()
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)  # Leave 2 cores free
    
    print("=" * 60)
    print("ISL Translation - Multicore Preprocessing")
    print("=" * 60)
    print(f"Videos directory: {videos_dir}")
    print(f"Annotations file: {annotations_file}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Subset ratio: {subset_ratio * 100:.0f}%")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Load annotations
    print(f"\nLoading annotations...")
    annotations = pd.read_csv(annotations_file)
    print(f"Total samples: {len(annotations)}")
    
    # Subsample if needed
    if subset_ratio < 1.0:
        np.random.seed(seed)
        n_samples = int(len(annotations) * subset_ratio)
        indices = np.random.choice(len(annotations), n_samples, replace=False)
        annotations = annotations.iloc[indices].reset_index(drop=True)
        print(f"Using {subset_ratio*100:.0f}% subset: {len(annotations)} samples")
    
    # Split data
    splits = random_split(annotations, seed)
    
    # Prepare work items
    work_items = []
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            video_id = row.get('uid', row.get('video_id'))
            text = row['text']
            work_items.append((video_id, text, split_name, videos_dir, output_dir))
    
    print(f"\nTotal work items: {len(work_items)}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val: {len(splits['val'])}")
    print(f"  Test: {len(splits['test'])}")
    
    # Process with multiprocessing
    print(f"\nProcessing with {num_workers} workers...")
    
    successful = 0
    failed = 0
    metadata_list = []
    
    with Pool(num_workers) as pool:
        # Use imap for progress tracking
        results = pool.imap(process_single_video, work_items, chunksize=10)
        
        for i, result in enumerate(results):
            if result is not None:
                successful += 1
                metadata_list.append(result)
            else:
                failed += 1
            
            # Progress update every 1000 videos
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(work_items) - i - 1) / rate
                print(f"  Processed {i+1}/{len(work_items)} | "
                      f"Success: {successful} | Failed: {failed} | "
                      f"Rate: {rate:.1f}/s | ETA: {remaining/60:.1f} min")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 60)
    
    return metadata_df


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Configuration
    VIDEOS_DIR = r"E:\iSign-videos_v1.1"
    ANNOTATIONS_FILE = r"E:\5thsem el\APPROACH 2\iSign_v1.1.csv"
    OUTPUT_DIR = r"E:\5thsem el\APPROACH 2\preprocessed_data"
    
    # GPU mode for subset ratio
    GPU_MODE = "small"  # "small" = 40%, "large" = 100%
    
    set_gpu_mode(GPU_MODE)
    SUBSET_RATIO = training_config.subset_ratio
    
    # Number of workers (Ryzen 9 7940HS has 8 cores / 16 threads)
    # Use 12 workers to leave some headroom
    NUM_WORKERS = 12
    
    # Run preprocessing
    preprocess_multicore(
        videos_dir=VIDEOS_DIR,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR,
        subset_ratio=SUBSET_RATIO,
        num_workers=NUM_WORKERS,
        seed=42
    )
