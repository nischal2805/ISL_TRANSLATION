"""
Extract video features and save as SINGLE consolidated .npy file
================================================================
For easy SCP transfer to server: ONE file instead of 127K files!

Output:
- video_features_all.npy: (N, max_T, 540) all features padded
- video_metadata.csv: video_id, text, length, split
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mproc
from typing import Dict, List, Tuple
from scipy.ndimage import gaussian_filter1d
import warnings
import gc
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    VIDEO_DIR = Path("E:/iSign-videos_v1.1")
    CSV_PATH = Path("E:/5thsem el/APPROACH 2/iSign_v1.1.csv")
    OUTPUT_DIR = Path("E:/5thsem el/APPROACH 2/video_features")
    
    # Video processing
    TARGET_FPS = 10  # Downsample to 10fps
    MAX_FRAMES = 150  # Max 15 seconds at 10fps
    
    # Pose extraction
    POSE_CONFIDENCE = 0.5
    
    # Feature dimensions
    RAW_DIM = 180  # hands(126) + pose(39) + face(15)
    OUTPUT_DIM = 540  # 180 * 3 (pos + vel + acc)
    
    # Preprocessing
    SMOOTHING_SIGMA = 1.0
    
    # Multiprocessing
    NUM_WORKERS = 12  # Adjust based on CPU cores
    
    # Split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15


# ============================================================================
# Preprocessing Functions
# ============================================================================

def compute_velocity(features: np.ndarray) -> np.ndarray:
    """Compute velocity (first derivative)."""
    velocity = np.zeros_like(features)
    if features.shape[0] > 1:
        velocity[1:] = features[1:] - features[:-1]
    return velocity


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """Compute acceleration (second derivative)."""
    acceleration = np.zeros_like(velocity)
    if velocity.shape[0] > 1:
        acceleration[1:] = velocity[1:] - velocity[:-1]
    return acceleration


def smooth_features(features: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing."""
    if sigma <= 0 or features.shape[0] < 3:
        return features
    return gaussian_filter1d(features, sigma=sigma, axis=0, mode='nearest')


# ============================================================================
# Pose Extractor
# ============================================================================

class PoseExtractor:
    """Extract MediaPipe pose landmarks."""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=Config.POSE_CONFIDENCE,
            min_tracking_confidence=Config.POSE_CONFIDENCE
        )
    
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract landmarks from a single frame.
        
        Returns:
            landmarks: (180,) array
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        
        landmarks = []
        
        # Left hand (21 * 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand (21 * 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Pose - upper body (13 * 3 = 39)
        if results.pose_landmarks:
            upper_body_indices = list(range(13))
            for idx in upper_body_indices:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 39)
        
        # Face key points (5 * 3 = 15)
        if results.face_landmarks:
            key_face_indices = [1, 33, 263, 61, 291]
            for idx in key_face_indices:
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 15)
        
        return np.array(landmarks, dtype=np.float32)
    
    def __del__(self):
        self.holistic.close()


# ============================================================================
# Video Processor
# ============================================================================

def process_single_video(args: Tuple[str, str, Path]) -> Dict:
    """Process one video file."""
    video_id, text, video_path = args
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'status': 'error', 'reason': 'cannot_open', 'video_id': video_id}
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps == 0 or total_frames == 0:
            cap.release()
            return {'status': 'error', 'reason': 'invalid_video', 'video_id': video_id}
        
        frame_skip = max(1, int(original_fps / Config.TARGET_FPS))
        
        pose_extractor = PoseExtractor()
        landmarks_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                lm = pose_extractor.extract(frame)
                if lm is not None:
                    landmarks_list.append(lm)
                
                if len(landmarks_list) >= Config.MAX_FRAMES:
                    break
            
            frame_idx += 1
        
        cap.release()
        del pose_extractor
        
        if len(landmarks_list) < 5:
            return {'status': 'error', 'reason': 'too_short', 'video_id': video_id}
        
        # Preprocess
        raw_landmarks = np.array(landmarks_list, dtype=np.float32)
        smoothed_pos = smooth_features(raw_landmarks, sigma=Config.SMOOTHING_SIGMA)
        velocity = compute_velocity(smoothed_pos)
        velocity = smooth_features(velocity, sigma=Config.SMOOTHING_SIGMA)
        acceleration = compute_acceleration(velocity)
        acceleration = smooth_features(acceleration, sigma=Config.SMOOTHING_SIGMA)
        
        features = np.concatenate([smoothed_pos, velocity, acceleration], axis=1).astype(np.float32)
        
        return {
            'status': 'success',
            'video_id': video_id,
            'text': text,
            'features': features,  # (T, 540)
            'num_frames': features.shape[0]
        }
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e), 'video_id': video_id}


# ============================================================================
# Main - Creates SINGLE consolidated file
# ============================================================================

def main():
    print("="*70)
    print("ISL VIDEO FEATURE EXTRACTION - CONSOLIDATED OUTPUT")
    print("="*70)
    print(f"Output: SINGLE .npy file for easy SCP transfer")
    print("="*70)
    
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    print(f"\nLoading CSV from {Config.CSV_PATH}...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"Total entries: {len(df)}")
    
    # Create task list
    tasks = []
    missing_videos = 0
    
    for _, row in df.iterrows():
        video_id = row['uid']
        text = row['text']
        video_path = Config.VIDEO_DIR / f"{video_id}.mp4"
        
        if video_path.exists():
            tasks.append((video_id, text, video_path))
        else:
            missing_videos += 1
    
    print(f"Found {len(tasks)} videos ({missing_videos} missing)")
    
    # Process in batches to avoid memory issues
    BATCH_SIZE = 5000
    all_results = []
    
    print(f"\nProcessing with {Config.NUM_WORKERS} workers...")
    
    for batch_start in range(0, len(tasks), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]
        
        print(f"\nBatch {batch_start//BATCH_SIZE + 1}: Processing videos {batch_start+1} to {batch_end}")
        
        with mproc.Pool(processes=Config.NUM_WORKERS) as pool:
            batch_results = list(tqdm(
                pool.imap(process_single_video, batch_tasks),
                total=len(batch_tasks),
                desc="Extracting"
            ))
        
        all_results.extend(batch_results)
        gc.collect()
    
    # Filter successful results
    successful = [r for r in all_results if r['status'] == 'success']
    errors = len(all_results) - len(successful)
    
    print(f"\n{'='*70}")
    print(f"Extraction complete: {len(successful)} success, {errors} errors")
    
    # Create splits
    np.random.seed(42)
    n = len(successful)
    indices = np.random.permutation(n)
    
    train_end = int(n * Config.TRAIN_RATIO)
    val_end = int(n * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    
    splits = ['test'] * n
    for i in indices[:train_end]:
        splits[i] = 'train'
    for i in indices[train_end:val_end]:
        splits[i] = 'val'
    
    # Create metadata
    metadata = []
    for i, r in enumerate(successful):
        metadata.append({
            'idx': i,
            'video_id': r['video_id'],
            'text': r['text'],
            'num_frames': r['num_frames'],
            'split': splits[i]
        })
    
    meta_df = pd.DataFrame(metadata)
    
    # Save metadata
    meta_path = Config.OUTPUT_DIR / 'video_metadata.csv'
    meta_df.to_csv(meta_path, index=False)
    
    print(f"\nSplit distribution:")
    print(f"  Train: {(meta_df['split'] == 'train').sum()}")
    print(f"  Val:   {(meta_df['split'] == 'val').sum()}")
    print(f"  Test:  {(meta_df['split'] == 'test').sum()}")
    
    # Save as consolidated numpy arrays (per split for easier handling)
    print(f"\nSaving consolidated arrays...")
    
    for split in ['train', 'val', 'test']:
        split_indices = meta_df[meta_df['split'] == split]['idx'].values
        split_features = [successful[i]['features'] for i in split_indices]
        split_lengths = [f.shape[0] for f in split_features]
        
        # Pad to max length in this split
        max_len = max(split_lengths)
        padded = np.zeros((len(split_features), max_len, Config.OUTPUT_DIM), dtype=np.float32)
        
        for i, feat in enumerate(split_features):
            padded[i, :feat.shape[0]] = feat
        
        # Save
        features_path = Config.OUTPUT_DIR / f'{split}_features.npy'
        lengths_path = Config.OUTPUT_DIR / f'{split}_lengths.npy'
        
        np.save(features_path, padded)
        np.save(lengths_path, np.array(split_lengths, dtype=np.int32))
        
        # Get file size
        size_mb = features_path.stat().st_size / (1024 * 1024)
        print(f"  {split}: {padded.shape} ({size_mb:.1f} MB)")
    
    print(f"\n{'='*70}")
    print("FILES TO TRANSFER VIA SCP:")
    print("="*70)
    print(f"  1. {Config.OUTPUT_DIR / 'train_features.npy'}")
    print(f"  2. {Config.OUTPUT_DIR / 'train_lengths.npy'}")
    print(f"  3. {Config.OUTPUT_DIR / 'val_features.npy'}")
    print(f"  4. {Config.OUTPUT_DIR / 'val_lengths.npy'}")
    print(f"  5. {Config.OUTPUT_DIR / 'test_features.npy'}")
    print(f"  6. {Config.OUTPUT_DIR / 'test_lengths.npy'}")
    print(f"  7. {Config.OUTPUT_DIR / 'video_metadata.csv'}")
    print(f"\nTotal: 7 files (vs 127K individual files!)")
    print("="*70)


if __name__ == "__main__":
    main()
