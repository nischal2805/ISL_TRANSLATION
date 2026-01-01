"""
Landmark Extraction Script for ISL Translation
===============================================
Extracts MediaPipe landmarks from videos and saves as .npy files.
Output: (T, 138) - positions only (2 hands + shoulders + elbows)

Usage:
    python extract_landmarks.py --video_dir data/videos --output_dir data/landmarks/train
"""

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import os
import argparse
from pathlib import Path


# =====================
# CONFIG
# =====================
MAX_HANDS = 2
FPS_SKIP = 3   # ~8–10 FPS effective (process every 3rd frame)


def setup_mediapipe():
    """Initialize MediaPipe models."""
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,      # lightweight
        smooth_landmarks=False,  # faster
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return hands, pose


def extract_from_video(video_path, hands, pose):
    """
    Extract landmarks from a single video.
    
    Returns:
        np.ndarray: Shape (T, 138) where T is number of frames
                    138 = 2 hands * 21 * 3 + 2 shoulders * 3 + 2 elbows * 3
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FPS_SKIP != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------- Hands --------
        hands_result = hands.process(rgb)

        hand_landmarks = np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
        shoulder_landmarks = np.zeros((2, 3), dtype=np.float32)
        elbow_landmarks = np.zeros((2, 3), dtype=np.float32)

        has_hand = False

        if hands_result.multi_hand_landmarks:
            has_hand = True
            for h_id, hand_lm in enumerate(hands_result.multi_hand_landmarks):
                if h_id >= MAX_HANDS:
                    break
                for lm_id, lm in enumerate(hand_lm.landmark):
                    hand_landmarks[h_id, lm_id] = [lm.x, lm.y, lm.z]

        # -------- Pose (ONLY if hands exist) --------
        if has_hand:
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                lm = pose_result.pose_landmarks.landmark

                # Shoulders
                shoulder_landmarks[0] = [lm[11].x, lm[11].y, lm[11].z]  # Left
                shoulder_landmarks[1] = [lm[12].x, lm[12].y, lm[12].z]  # Right

                # Elbows
                elbow_landmarks[0] = [lm[13].x, lm[13].y, lm[13].z]     # Left
                elbow_landmarks[1] = [lm[14].x, lm[14].y, lm[14].z]     # Right

        # -------- Combine --------
        combined = np.concatenate([
            hand_landmarks.reshape(-1),      # 126
            shoulder_landmarks.reshape(-1),  # 6
            elbow_landmarks.reshape(-1)      # 6
        ])                                   # = 138

        frames.append(combined)

    cap.release()
    return np.asarray(frames, dtype=np.float32)  # (T, 138)


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks from videos')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                        help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='data/landmarks/train',
                        help='Directory to save extracted landmarks')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed videos')
    args = parser.parse_args()
    
    # Setup paths
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video list
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    # Initialize MediaPipe
    hands, pose = setup_mediapipe()
    
    # Process videos
    processed = 0
    skipped = 0
    failed = 0
    
    for vid in tqdm(video_files, desc="Extracting landmarks"):
        out_path = output_dir / vid.replace(".mp4", ".npy")
        
        # Skip if already processed (resume mode)
        if args.resume and out_path.exists():
            skipped += 1
            continue
        
        video_path = video_dir / vid
        data = extract_from_video(str(video_path), hands, pose)
        
        if len(data) == 0:
            print(f"⚠️ No usable frames: {vid}")
            failed += 1
            continue
        
        np.save(out_path, data)
        processed += 1
    
    # Cleanup
    hands.close()
    pose.close()
    
    print(f"\n✅ Landmark extraction completed!")
    print(f"   Processed: {processed}")
    print(f"   Skipped (already exists): {skipped}")
    print(f"   Failed: {failed}")
    print(f"   Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
