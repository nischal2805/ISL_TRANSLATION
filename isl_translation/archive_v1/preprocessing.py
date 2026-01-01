"""
ISL Translation System - Preprocessing Pipeline
================================================
MediaPipe landmark extraction and feature engineering.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from config import data_config, landmark_config


class LandmarkExtractor:
    """Extract landmarks from video using MediaPipe Holistic."""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=landmark_config.static_image_mode,
            model_complexity=landmark_config.model_complexity,
            min_detection_confidence=landmark_config.min_detection_confidence,
            min_tracking_confidence=landmark_config.min_tracking_confidence,
            enable_segmentation=False,
            refine_face_landmarks=False  # No face landmarks
        )
        
        # Landmark indices
        self.pose_indices = landmark_config.pose_indices
        self.hand_indices = landmark_config.hand_indices
    
    def extract_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract landmarks from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Landmarks array (T, 46, 3) or None if extraction fails
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        landmarks_list = []
        frame_count = 0
        detection_failures = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.holistic.process(rgb_frame)
            
            # Extract landmarks
            frame_landmarks = self._extract_frame_landmarks(results)
            
            if frame_landmarks is None:
                detection_failures += 1
                # Fill with zeros for failed frames
                frame_landmarks = np.zeros((landmark_config.num_landmarks, 3), dtype=np.float32)
            
            landmarks_list.append(frame_landmarks)
            frame_count += 1
        
        cap.release()
        
        if frame_count == 0:
            return None
        
        # Check detection success rate
        success_rate = (frame_count - detection_failures) / frame_count
        if success_rate < 0.5:
            print(f"Warning: Low detection rate ({success_rate:.1%}) for {video_path}")
        
        # Stack to array (T, 46, 3)
        landmarks = np.stack(landmarks_list, axis=0).astype(np.float32)
        
        return landmarks
    
    def _extract_frame_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract 46 landmarks from MediaPipe results for a single frame."""
        landmarks = []
        
        # Check if any landmarks detected
        has_pose = results.pose_landmarks is not None
        has_left_hand = results.left_hand_landmarks is not None
        has_right_hand = results.right_hand_landmarks is not None
        
        if not (has_pose or has_left_hand or has_right_hand):
            return None
        
        # Pose landmarks (4 points: shoulders + elbows)
        if has_pose:
            for idx in self.pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * len(self.pose_indices))
        
        # Left hand landmarks (21 points)
        if has_left_hand:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        # Right hand landmarks (21 points)
        if has_right_hand:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        return np.array(landmarks, dtype=np.float32)
    
    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()


class FeatureProcessor:
    """Process raw landmarks into normalized features."""
    
    def __init__(self):
        self.sigma = data_config.gaussian_sigma
    
    def process(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline: normalize, compute derivatives, smooth.
        
        Args:
            landmarks: Raw landmarks (T, 46, 3)
            
        Returns:
            Features (T, 414) - position + velocity + acceleration
        """
        # Step 1: Flatten to (T, 138)
        T = landmarks.shape[0]
        positions = landmarks.reshape(T, -1)  # (T, 138)
        
        # Step 2: Normalize
        positions = self._normalize(positions, landmarks)
        
        # Step 3: Smooth positions
        positions = self._smooth(positions)
        
        # Step 4: Compute velocity
        velocity = self._compute_derivative(positions)
        
        # Step 5: Smooth velocity
        velocity = self._smooth(velocity)
        
        # Step 6: Compute acceleration
        acceleration = self._compute_derivative(velocity)
        
        # Step 7: Concatenate (T, 414)
        features = np.concatenate([positions, velocity, acceleration], axis=1)
        
        # Step 8: Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features.astype(np.float32)
    
    def _normalize(self, positions: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks: hand-centered, body-centered, scale-invariant.
        
        Args:
            positions: Flattened positions (T, 138)
            landmarks: Original landmarks (T, 46, 3) for reference
            
        Returns:
            Normalized positions (T, 138)
        """
        T = landmarks.shape[0]
        normalized = np.zeros_like(positions)
        
        for t in range(T):
            frame = landmarks[t]  # (46, 3)
            
            # Pose landmarks (indices 0-3 in our selection)
            # 0: left shoulder, 1: right shoulder, 2: left elbow, 3: right elbow
            pose = frame[:4]  # (4, 3)
            
            # Left hand (indices 4-24)
            left_hand = frame[4:25]  # (21, 3)
            
            # Right hand (indices 25-45)
            right_hand = frame[25:46]  # (21, 3)
            
            # Normalize left hand (center to wrist, scale by hand span)
            left_wrist = left_hand[0]
            left_centered = left_hand - left_wrist
            left_span = np.linalg.norm(left_hand[12] - left_hand[0]) + 1e-6  # middle finger tip to wrist
            left_normalized = left_centered / left_span
            
            # Normalize right hand
            right_wrist = right_hand[0]
            right_centered = right_hand - right_wrist
            right_span = np.linalg.norm(right_hand[12] - right_hand[0]) + 1e-6
            right_normalized = right_centered / right_span
            
            # Normalize body (center to shoulder midpoint, scale by shoulder width)
            shoulder_mid = (pose[0] + pose[1]) / 2
            pose_centered = pose - shoulder_mid
            shoulder_width = np.linalg.norm(pose[1] - pose[0]) + 1e-6
            pose_normalized = pose_centered / shoulder_width
            
            # Reconstruct normalized frame
            norm_frame = np.concatenate([
                pose_normalized,      # (4, 3)
                left_normalized,      # (21, 3)
                right_normalized      # (21, 3)
            ], axis=0)  # (46, 3)
            
            normalized[t] = norm_frame.flatten()
        
        return normalized
    
    def _smooth(self, features: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing along time axis."""
        if self.sigma <= 0:
            return features
        
        smoothed = gaussian_filter1d(features, sigma=self.sigma, axis=0, mode='nearest')
        return smoothed
    
    def _compute_derivative(self, features: np.ndarray) -> np.ndarray:
        """Compute first derivative (velocity or acceleration)."""
        derivative = np.zeros_like(features)
        derivative[1:] = features[1:] - features[:-1]
        derivative[0] = derivative[1]  # Copy first valid derivative
        return derivative


def preprocess_dataset(
    videos_dir: str,
    annotations_file: str,
    output_dir: str,
    subset_ratio: float = 1.0,
    seed: int = 42
) -> Dict[str, str]:
    """
    Preprocess entire dataset: extract landmarks, compute features, save.
    
    Args:
        videos_dir: Directory containing video files
        annotations_file: CSV file with video_id and text columns
        output_dir: Directory to save preprocessed features
        subset_ratio: Fraction of dataset to use (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping video_id to preprocessed file path
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    annotations = pd.read_csv(annotations_file)
    
    # Subsample if needed
    if subset_ratio < 1.0:
        np.random.seed(seed)
        n_samples = int(len(annotations) * subset_ratio)
        indices = np.random.choice(len(annotations), n_samples, replace=False)
        annotations = annotations.iloc[indices].reset_index(drop=True)
        print(f"Using {subset_ratio*100:.0f}% subset: {len(annotations)} samples")
    
    # Split by signer if available, otherwise random
    if 'signer_id' in annotations.columns:
        splits = _stratified_split_by_signer(annotations, seed)
    else:
        splits = _random_split(annotations, seed)
    
    # Initialize extractors
    extractor = LandmarkExtractor()
    processor = FeatureProcessor()
    
    # Process each split
    processed_files = {}
    metadata_list = []
    
    for split_name, split_data in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_data)} videos)...")
        
        for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc=split_name):
            # Support both 'uid' and 'video_id' column names
            video_id = row.get('uid', row.get('video_id', None))
            if video_id is None:
                print(f"Row missing uid/video_id column")
                continue
            text = row['text']
            
            # Find video file
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(videos_dir, video_id)
                if not os.path.exists(video_path):
                    print(f"Video not found: {video_id}")
                    continue
            
            # Extract landmarks
            landmarks = extractor.extract_from_video(video_path)
            if landmarks is None:
                print(f"Failed to extract landmarks: {video_id}")
                continue
            
            # Process features
            features = processor.process(landmarks)
            
            # Validate sequence length
            if features.shape[0] < data_config.min_src_len:
                print(f"Sequence too short: {video_id} ({features.shape[0]} frames)")
                continue
            
            if features.shape[0] > data_config.max_src_len:
                features = features[:data_config.max_src_len]
            
            # Save features
            output_path = os.path.join(output_dir, split_name, f"{video_id}.npy")
            np.save(output_path, features)
            
            processed_files[video_id] = output_path
            metadata_list.append({
                'video_id': video_id,
                'text': text,
                'split': split_name,
                'length': features.shape[0],
                'path': output_path
            })
    
    extractor.close()
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total processed: {len(processed_files)}")
    for split_name in ['train', 'val', 'test']:
        count = len(metadata_df[metadata_df['split'] == split_name])
        print(f"  {split_name}: {count}")
    
    return processed_files


def _stratified_split_by_signer(
    annotations: pd.DataFrame,
    seed: int
) -> Dict[str, pd.DataFrame]:
    """Split dataset ensuring different signers in each split."""
    np.random.seed(seed)
    
    signers = annotations['signer_id'].unique().tolist()
    np.random.shuffle(signers)
    
    n_signers = len(signers)
    n_train = int(n_signers * data_config.train_ratio)
    n_val = int(n_signers * data_config.val_ratio)
    
    train_signers = set(signers[:n_train])
    val_signers = set(signers[n_train:n_train + n_val])
    test_signers = set(signers[n_train + n_val:])
    
    return {
        'train': annotations[annotations['signer_id'].isin(train_signers)],
        'val': annotations[annotations['signer_id'].isin(val_signers)],
        'test': annotations[annotations['signer_id'].isin(test_signers)]
    }


def _random_split(
    annotations: pd.DataFrame,
    seed: int
) -> Dict[str, pd.DataFrame]:
    """Random split without signer stratification."""
    np.random.seed(seed)
    
    indices = np.random.permutation(len(annotations))
    
    n_train = int(len(annotations) * data_config.train_ratio)
    n_val = int(len(annotations) * data_config.val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'train': annotations.iloc[train_idx],
        'val': annotations.iloc[val_idx],
        'test': annotations.iloc[test_idx]
    }


# Utility functions for inference
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks for inference.
    
    Args:
        landmarks: (T, 138) flattened landmarks
        
    Returns:
        (T, 138) normalized landmarks
    """
    processor = FeatureProcessor()
    # Reshape for processing
    T = landmarks.shape[0]
    landmarks_3d = landmarks.reshape(T, 46, 3)
    positions = landmarks.copy()
    return processor._normalize(positions, landmarks_3d)


def compute_motion_features(positions: np.ndarray) -> np.ndarray:
    """
    Compute full feature set from positions.
    
    Args:
        positions: (T, 138) normalized positions
        
    Returns:
        (T, 414) features with velocity and acceleration
    """
    processor = FeatureProcessor()
    
    # Smooth positions
    positions = processor._smooth(positions)
    
    # Compute velocity
    velocity = processor._compute_derivative(positions)
    velocity = processor._smooth(velocity)
    
    # Compute acceleration
    acceleration = processor._compute_derivative(velocity)
    
    # Concatenate
    features = np.concatenate([positions, velocity, acceleration], axis=1)
    
    return features.astype(np.float32)


if __name__ == "__main__":
    # Test preprocessing
    print("ISL Preprocessing Test")
    print("=" * 50)
    
    # Test with dummy data
    print("\nTesting feature processor with dummy data...")
    dummy_landmarks = np.random.randn(100, 46, 3).astype(np.float32)
    
    processor = FeatureProcessor()
    features = processor.process(dummy_landmarks)
    
    print(f"Input shape: {dummy_landmarks.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (100, 414)")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Contains NaN: {np.isnan(features).any()}")
