"""
Extract video frames + pose landmarks for ISL translation.
Processes 127K videos using multicore processing.

Output: .npz files with {frames, landmarks, text, fps}
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mproc
from typing import Dict, List, Tuple
import warnings
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
    TARGET_FPS = 10  # Downsample to 10fps for efficiency
    FRAME_SIZE = (224, 224)  # MobileNet input size
    MAX_FRAMES = 150  # Max 15 seconds at 10fps
    
    # Pose extraction
    EXTRACT_POSE = True
    POSE_CONFIDENCE = 0.5
    
    # Multiprocessing
    NUM_WORKERS = 14  # Adjust based on CPU cores


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
            landmarks: (204,) array or None if detection fails
                      [hands(42*3) + pose(33*3) + face(468*3 downsampled to 40*3)]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.holistic.process(rgb)
        
        # Extract landmarks
        landmarks = []
        
        # Left hand (21 landmarks * 3 coords = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Right hand (21 landmarks * 3 coords = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
        
        # Pose (33 landmarks * 3 coords = 99, but we keep only upper body = 13*3 = 39)
        if results.pose_landmarks:
            # Keep only upper body: 0-10 (face/shoulders), 11-12 (shoulders), 13-16 (elbows/wrists)
            upper_body_indices = list(range(13))
            for idx in upper_body_indices:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 39)
        
        # Face (sample 5 key points: nose, eyes, mouth corners)
        if results.face_landmarks:
            key_face_indices = [1, 33, 263, 61, 291]  # nose, left eye, right eye, left mouth, right mouth
            for idx in key_face_indices:
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 15)
        
        return np.array(landmarks, dtype=np.float32)  # Total: 63+63+39+15 = 180 dims
    
    def __del__(self):
        self.holistic.close()


# ============================================================================
# Video Processor
# ============================================================================

def process_single_video(args: Tuple[str, str, Path]) -> Dict:
    """Process one video file.
    
    Args:
        args: (video_id, text, video_path)
    
    Returns:
        dict with status and output path
    """
    video_id, text, video_path = args
    
    try:
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'status': 'error', 'reason': 'cannot_open', 'video_id': video_id}
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps == 0 or total_frames == 0:
            cap.release()
            return {'status': 'error', 'reason': 'invalid_video', 'video_id': video_id}
        
        # Calculate frame sampling
        frame_skip = max(1, int(original_fps / Config.TARGET_FPS))
        
        # Initialize pose extractor
        pose_extractor = PoseExtractor() if Config.EXTRACT_POSE else None
        
        frames = []
        landmarks_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % frame_skip == 0:
                # Resize frame
                resized = cv2.resize(frame, Config.FRAME_SIZE)
                frames.append(resized)
                
                # Extract pose
                if pose_extractor:
                    lm = pose_extractor.extract(frame)
                    landmarks_list.append(lm)
                
                # Stop if max frames reached
                if len(frames) >= Config.MAX_FRAMES:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        # Check if valid
        if len(frames) < 5:  # Too short
            return {'status': 'error', 'reason': 'too_short', 'video_id': video_id}
        
        # Convert to numpy arrays
        frames_array = np.array(frames, dtype=np.uint8)  # (T, H, W, 3)
        landmarks_array = np.array(landmarks_list, dtype=np.float32) if pose_extractor else None
        
        # Save as .npz
        output_path = Config.OUTPUT_DIR / f"{video_id}.npz"
        
        save_dict = {
            'frames': frames_array,
            'text': text,
            'fps': Config.TARGET_FPS,
            'video_id': video_id
        }
        
        if landmarks_array is not None:
            save_dict['landmarks'] = landmarks_array
        
        np.savez_compressed(output_path, **save_dict)
        
        return {
            'status': 'success',
            'video_id': video_id,
            'num_frames': len(frames),
            'output_path': str(output_path)
        }
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e), 'video_id': video_id}


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("ISL VIDEO FEATURE EXTRACTION")
    print("="*70)
    
    # Create output directory
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
    
    # Process videos
    print(f"\nProcessing with {Config.NUM_WORKERS} workers...")
    print(f"Target: {Config.TARGET_FPS}fps, {Config.FRAME_SIZE}, max {Config.MAX_FRAMES} frames")
    
    # Multiprocessing
    with mproc.Pool(processes=Config.NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, tasks),
            total=len(tasks),
            desc="Extracting"
        ))
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    errors = len(results) - success
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"Success: {success}/{len(tasks)}")
    print(f"Errors: {errors}")
    print(f"Output: {Config.OUTPUT_DIR}")
    
    # Save metadata
    metadata_path = Config.OUTPUT_DIR / "extraction_metadata.csv"
    pd.DataFrame(results).to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
