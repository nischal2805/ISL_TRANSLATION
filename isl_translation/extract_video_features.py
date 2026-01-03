"""
Extract video frames + pose landmarks for ISL translation.
Processes videos using multicore processing.

Output: .npz files with {frames, landmarks, text, fps}

Usage:
    python extract_video_features.py                    # Process ALL videos
    python extract_video_features.py --limit 1         # Process only 1 video
    python extract_video_features.py --limit 10        # Process only 10 videos
    python extract_video_features.py --video-id ABC123 # Process specific video
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mproc
from typing import Dict, List, Tuple, Optional
import argparse
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
    """Process one video file with detailed error handling.
    
    Args:
        args: (video_id, text, video_path)
    
    Returns:
        dict with status, video_id, and either output_path or error reason
    """
    video_id, text, video_path = args
    
    try:
        # ===== VALIDATE INPUT =====
        if not video_path.exists():
            return {'status': 'error', 'reason': 'file_not_found', 'video_id': video_id}
        
        if not text or (isinstance(text, float) and np.isnan(text)):
            return {'status': 'error', 'reason': 'invalid_text', 'video_id': video_id}
        
        # ===== OPEN VIDEO =====
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'status': 'error', 'reason': 'cannot_open_video', 'video_id': video_id}
        
        # ===== GET VIDEO PROPERTIES =====
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if original_fps == 0:
            cap.release()
            return {'status': 'error', 'reason': 'invalid_fps', 'video_id': video_id}
        
        if total_frames == 0:
            cap.release()
            return {'status': 'error', 'reason': 'no_frames', 'video_id': video_id}
        
        if width == 0 or height == 0:
            cap.release()
            return {'status': 'error', 'reason': 'invalid_dimensions', 'video_id': video_id}
        
        # ===== CALCULATE FRAME SAMPLING =====
        frame_skip = max(1, int(original_fps / Config.TARGET_FPS))
        
        # ===== INITIALIZE POSE EXTRACTOR =====
        pose_extractor = None
        if Config.EXTRACT_POSE:
            try:
                pose_extractor = PoseExtractor()
            except Exception as e:
                return {'status': 'error', 'reason': f'pose_init_failed: {str(e)}', 'video_id': video_id}
        
        # ===== PROCESS FRAMES =====
        frames = []
        landmarks_list = []
        frame_idx = 0
        read_errors = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_idx % frame_skip == 0:
                try:
                    # Resize frame
                    resized = cv2.resize(frame, Config.FRAME_SIZE)
                    frames.append(resized)
                    
                    # Extract pose landmarks
                    if pose_extractor:
                        lm = pose_extractor.extract(frame)
                        landmarks_list.append(lm)
                except Exception as e:
                    read_errors += 1
                    if read_errors > 10:
                        break
                
                # Stop if max frames reached
                if len(frames) >= Config.MAX_FRAMES:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        # ===== VALIDATE OUTPUT =====
        if len(frames) < 5:
            return {'status': 'error', 'reason': f'too_short_{len(frames)}_frames', 'video_id': video_id}
        
        # ===== CONVERT TO NUMPY =====
        frames_array = np.array(frames, dtype=np.uint8)  # (T, H, W, 3)
        
        if pose_extractor and landmarks_list:
            landmarks_array = np.array(landmarks_list, dtype=np.float32)
            # Validate landmarks
            if np.isnan(landmarks_array).all():
                return {'status': 'error', 'reason': 'all_landmarks_nan', 'video_id': video_id}
        else:
            landmarks_array = None
        
        # ===== SAVE OUTPUT =====
        output_path = Config.OUTPUT_DIR / f"{video_id}.npz"
        
        save_dict = {
            'frames': frames_array,
            'text': str(text),
            'fps': Config.TARGET_FPS,
            'video_id': str(video_id),
            'original_fps': original_fps,
            'original_frames': total_frames
        }
        
        if landmarks_array is not None:
            save_dict['landmarks'] = landmarks_array
        
        np.savez_compressed(output_path, **save_dict)
        
        return {
            'status': 'success',
            'video_id': video_id,
            'num_frames': len(frames),
            'output_path': str(output_path),
            'has_landmarks': landmarks_array is not None
        }
        
    except Exception as e:
        return {'status': 'error', 'reason': f'exception: {str(e)}', 'video_id': video_id}


# ============================================================================
# Main
# ============================================================================

def main():
    # ===== ARGUMENT PARSING =====
    parser = argparse.ArgumentParser(
        description='Extract video features for ISL translation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_video_features.py                    # Process ALL videos
    python extract_video_features.py --limit 1         # Process only 1 video  
    python extract_video_features.py --limit 10        # Process only 10 videos
    python extract_video_features.py --video-id ABC123 # Process specific video
    python extract_video_features.py --workers 4       # Use 4 workers
        """
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of videos to process (default: all)')
    parser.add_argument('--video-id', type=str, default=None,
                        help='Process only this specific video ID')
    parser.add_argument('--workers', type=int, default=Config.NUM_WORKERS,
                        help=f'Number of worker processes (default: {Config.NUM_WORKERS})')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Override video directory path')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Override CSV file path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory path')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.video_dir:
        Config.VIDEO_DIR = Path(args.video_dir)
    if args.csv_path:
        Config.CSV_PATH = Path(args.csv_path)
    if args.output_dir:
        Config.OUTPUT_DIR = Path(args.output_dir)
    Config.NUM_WORKERS = args.workers
    
    print("=" * 70)
    print("ISL VIDEO FEATURE EXTRACTION")
    print("=" * 70)
    
    # ===== VALIDATE PATHS =====
    print("\n[STEP 1] Validating paths...")
    
    if not Config.VIDEO_DIR.exists():
        print(f"  ❌ ERROR: Video directory not found: {Config.VIDEO_DIR}")
        return
    print(f"  ✓ Video directory: {Config.VIDEO_DIR}")
    
    if not Config.CSV_PATH.exists():
        print(f"  ❌ ERROR: CSV file not found: {Config.CSV_PATH}")
        return
    print(f"  ✓ CSV file: {Config.CSV_PATH}")
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {Config.OUTPUT_DIR}")
    
    # ===== LOAD CSV =====
    print(f"\n[STEP 2] Loading CSV...")
    try:
        df = pd.read_csv(Config.CSV_PATH)
        print(f"  ✓ Loaded {len(df)} entries")
    except Exception as e:
        print(f"  ❌ ERROR loading CSV: {e}")
        return
    
    # Check required columns
    required_cols = ['uid', 'text']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  ❌ ERROR: Missing columns in CSV: {missing_cols}")
        print(f"  Available columns: {list(df.columns)}")
        return
    
    # ===== CREATE TASK LIST =====
    print(f"\n[STEP 3] Creating task list...")
    tasks = []
    missing_videos = 0
    skipped_nan = 0
    
    for _, row in df.iterrows():
        video_id = str(row['uid'])
        text = row['text']
        
        # Skip if text is NaN
        if pd.isna(text):
            skipped_nan += 1
            continue
        
        # Filter by specific video ID if provided
        if args.video_id and video_id != args.video_id:
            continue
        
        video_path = Config.VIDEO_DIR / f"{video_id}.mp4"
        
        if video_path.exists():
            tasks.append((video_id, str(text), video_path))
        else:
            missing_videos += 1
        
        # Apply limit if specified
        if args.limit and len(tasks) >= args.limit:
            break
    
    print(f"  ✓ Found {len(tasks)} videos to process")
    if missing_videos > 0:
        print(f"  ⚠️ {missing_videos} videos not found in directory")
    if skipped_nan > 0:
        print(f"  ⚠️ {skipped_nan} entries skipped (NaN text)")
    
    if len(tasks) == 0:
        print("  ❌ ERROR: No videos to process!")
        if args.video_id:
            print(f"     Video ID '{args.video_id}' not found or missing file")
        return
    
    # ===== PROCESS VIDEOS =====
    print(f"\n[STEP 4] Processing videos...")
    print(f"  Workers: {Config.NUM_WORKERS}")
    print(f"  Target FPS: {Config.TARGET_FPS}")
    print(f"  Frame size: {Config.FRAME_SIZE}")
    print(f"  Max frames: {Config.MAX_FRAMES}")
    print()
    
    if len(tasks) == 1:
        # Single video - process directly (no multiprocessing overhead)
        print("  Processing single video directly...")
        results = [process_single_video(tasks[0])]
    else:
        # Multiple videos - use multiprocessing
        with mproc.Pool(processes=Config.NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(process_single_video, tasks),
                total=len(tasks),
                desc="  Extracting"
            ))
    
    # ===== SUMMARY =====
    success = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"  ✓ Success: {len(success)}/{len(tasks)}")
    if errors:
        print(f"  ❌ Errors: {len(errors)}")
        # Show first few errors
        print("\n  Error details:")
        for err in errors[:5]:
            print(f"    - {err['video_id']}: {err.get('reason', 'unknown')}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    print(f"\n  Output directory: {Config.OUTPUT_DIR}")
    
    # Show output files for successful extractions
    if success:
        print(f"\n  Output files:")
        for s in success[:5]:
            print(f"    - {s['output_path']} ({s['num_frames']} frames)")
        if len(success) > 5:
            print(f"    ... and {len(success) - 5} more files")
    
    # ===== SAVE METADATA =====
    metadata_path = Config.OUTPUT_DIR / "extraction_metadata.csv"
    pd.DataFrame(results).to_csv(metadata_path, index=False)
    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
