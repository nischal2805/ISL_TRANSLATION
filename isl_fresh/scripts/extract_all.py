"""
Full Extraction Pipeline - Multiprocessing
==========================================
Extracts landmarks from all videos using all CPU cores - 2
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import cv2
import mediapipe as mp

# Configuration
VIDEO_DIR = r"E:\iSign-videos_v1.1"
CSV_PATH = r"E:\5thsem el\APPROACH 2\iSign_v1.1.csv"
OUTPUT_DIR = r"E:\5thsem el\APPROACH 2\isl_fresh\data"
MAX_FRAMES = 300
NUM_WORKERS = max(1, cpu_count() - 2)

# Landmark indices
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
FACE_INDICES = [61, 291, 0, 17, 70, 300, 33, 263, 1, 4]


def extract_single_video(args):
    """Extract landmarks from one video. Returns (video_id, landmarks, text, success)."""
    video_id, text, video_path = args
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_id, None, text, False)
        
        mp_holistic = mp.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        landmarks_list = []
        frame_count = 0
        
        while frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Extract features
            features = []
            
            # Pose (15 points x 3)
            if results.pose_landmarks:
                for idx in POSE_INDICES:
                    lm = results.pose_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 45)
            
            # Left hand (21 x 3)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Right hand (21 x 3)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Face (10 x 3)
            if results.face_landmarks:
                for idx in FACE_INDICES:
                    lm = results.face_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 30)
            
            landmarks_list.append(features)
            frame_count += 1
        
        cap.release()
        holistic.close()
        
        if len(landmarks_list) < 5:  # Skip very short videos
            return (video_id, None, text, False)
        
        return (video_id, np.array(landmarks_list, dtype=np.float32), text, True)
    
    except Exception as e:
        return (video_id, None, text, False)


def main():
    print(f"=== ISL Landmark Extraction ===")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Video dir: {VIDEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print(f"Total samples in CSV: {len(df)}")
    
    # Prepare args
    args_list = []
    for _, row in df.iterrows():
        video_id = row['uid']
        text = row['text']
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            args_list.append((video_id, text, video_path))
    
    print(f"Videos found: {len(args_list)}")
    
    # Create output dirs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Process with multiprocessing
    print(f"\nExtracting landmarks...")
    
    results = []
    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(extract_single_video, args_list), total=len(args_list)):
            if result[3]:  # success
                results.append(result)
    
    print(f"\nSuccessfully processed: {len(results)}/{len(args_list)}")
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(results)
    
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
        for video_id, landmarks, text, _ in tqdm(split_data, desc=split_name):
            # Save landmarks
            save_path = os.path.join(OUTPUT_DIR, split_name, f"{video_id}.npy")
            np.save(save_path, landmarks)
            metadata.append({'video_id': video_id, 'text': text, 'split': split_name, 'frames': len(landmarks)})
    
    # Save metadata
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)
    
    print(f"\n=== DONE ===")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"Feature dim: 201 (pose:45 + hands:126 + face:30)")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
