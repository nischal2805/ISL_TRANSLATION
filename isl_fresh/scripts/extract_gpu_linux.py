"""
GPU-Accelerated Extraction Pipeline for A100
=============================================
Uses MediaPipe GPU backend on Linux for massive speedup.
Estimated: 127K videos in ~2-3 hours on A100!
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2

# MediaPipe GPU setup for Linux
os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Enable GPU
import mediapipe as mp

# Configuration - GPU Server paths
VIDEO_DIR = "/media/rvcse22/CSERV/kortex_sem5/videos/iSign-videos_v1.1"
CSV_PATH = "/media/rvcse22/CSERV/kortex_sem5/nischal/training_by_surya/ISL_TRANSLATION/iSign_v1.1.csv"
OUTPUT_DIR = "/media/rvcse22/CSERV/kortex_sem5/nischal/training_by_surya/isl_fresh/data"

MAX_FRAMES = 200          # Frames per video
FRAME_SKIP = 2            # Process every 2nd frame (still good quality)
BATCH_SIZE = 32           # Process multiple frames in memory

# Landmark indices
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
FACE_INDICES = [61, 291, 0, 17, 70, 300, 33, 263, 1, 4]


def create_holistic_gpu():
    """Create MediaPipe Holistic with GPU acceleration."""
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,          # Can use full model with GPU
        enable_segmentation=False,   # Disable for speed
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks_from_frame(results):
    """Extract 201-dim features from MediaPipe results."""
    features = []
    
    # Pose (15 points x 3 = 45)
    if results.pose_landmarks:
        for idx in POSE_INDICES:
            lm = results.pose_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 45)
    
    # Left hand (21 x 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand (21 x 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Face (10 x 3 = 30)
    if results.face_landmarks:
        for idx in FACE_INDICES:
            lm = results.face_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 30)
    
    return features


def extract_single_video(video_path, holistic):
    """Extract landmarks from a single video using GPU."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(fps / 15))  # Target ~15fps
    
    landmarks_list = []
    frame_idx = 0
    
    while len(landmarks_list) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with GPU-accelerated MediaPipe
        results = holistic.process(rgb)
        
        # Extract features
        features = extract_landmarks_from_frame(results)
        landmarks_list.append(features)
    
    cap.release()
    
    if len(landmarks_list) < 5:
        return None
    
    return np.array(landmarks_list, dtype=np.float32)


def process_batch(batch_args, holistic):
    """Process a batch of videos."""
    results = []
    for video_id, text, video_path in batch_args:
        try:
            landmarks = extract_single_video(video_path, holistic)
            if landmarks is not None:
                results.append((video_id, landmarks, text, True))
            else:
                results.append((video_id, None, text, False))
        except Exception as e:
            results.append((video_id, None, text, False))
    return results


def main():
    print("=" * 60)
    print("ISL Landmark Extraction - A100 GPU Accelerated")
    print("=" * 60)
    
    # Check GPU
    print("\nChecking GPU availability...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                               capture_output=True, text=True)
        print(f"GPU: {result.stdout.strip()}")
    except:
        print("Warning: Could not detect GPU with nvidia-smi")
    
    print(f"\nConfiguration:")
    print(f"  Video dir: {VIDEO_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Max frames: {MAX_FRAMES}")
    print(f"  Frame skip: ~{FRAME_SKIP}x")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"\nTotal samples in CSV: {len(df)}")
    
    # Prepare video list
    args_list = []
    for _, row in df.iterrows():
        video_id = row['uid']
        text = row['text']
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            args_list.append((video_id, text, video_path))
    
    print(f"Videos found: {len(args_list)}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Check for already processed (resume support)
    existing = set()
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.endswith('.npy'):
                    existing.add(f.replace('.npy', ''))
    
    if existing:
        print(f"Found {len(existing)} already processed, resuming...")
        args_list = [a for a in args_list if a[0] not in existing]
        print(f"Remaining: {len(args_list)}")
    
    if not args_list:
        print("All videos already processed!")
        return
    
    # Create GPU-accelerated holistic model
    print("\nInitializing MediaPipe with GPU...")
    holistic = create_holistic_gpu()
    
    # Process videos
    print(f"\nProcessing {len(args_list)} videos...")
    
    results = []
    failed = 0
    
    pbar = tqdm(args_list, desc="Extracting", unit="video")
    for video_id, text, video_path in pbar:
        try:
            landmarks = extract_single_video(video_path, holistic)
            if landmarks is not None:
                results.append((video_id, landmarks, text))
            else:
                failed += 1
        except Exception as e:
            failed += 1
        
        pbar.set_postfix({'success': len(results), 'failed': failed})
    
    holistic.close()
    
    print(f"\n\nSuccessfully processed: {len(results)}")
    print(f"Failed: {failed}")
    
    if not results:
        print("No results to save!")
        return
    
    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(results))
    results = [results[i] for i in indices]
    
    n = len(results)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    splits = {
        'train': results[:train_end],
        'val': results[train_end:val_end],
        'test': results[val_end:]
    }
    
    # Save
    metadata = []
    for split_name, split_data in splits.items():
        print(f"\nSaving {split_name}: {len(split_data)} samples")
        for video_id, landmarks, text in tqdm(split_data, desc=split_name):
            save_path = os.path.join(OUTPUT_DIR, split_name, f"{video_id}.npy")
            np.save(save_path, landmarks)
            metadata.append({
                'video_id': video_id,
                'text': text,
                'split': split_name,
                'frames': len(landmarks)
            })
    
    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)
    
    print(f"\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Feature dim: 201 (pose:45 + hands:126 + face:30)")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
