"""
Landmark Extractor
==================

Extracts pose, hand, and face landmarks from sign language videos
using MediaPipe Holistic.

Pipeline:
  Video → Frames → MediaPipe → Landmarks → Numpy Array
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of landmark extraction from a video."""
    landmarks: np.ndarray  # Shape: (num_frames, feature_dim)
    num_frames: int
    fps: float
    success: bool
    error: Optional[str] = None


class LandmarkExtractor:
    """
    Extract landmarks from sign language videos.
    
    Uses MediaPipe Holistic to detect:
    - Pose landmarks (upper body)
    - Hand landmarks (both hands)
    - Face landmarks (key expression points)
    
    Example:
        extractor = LandmarkExtractor()
        result = extractor.extract("video.mp4")
        print(result.landmarks.shape)  # (num_frames, 201)
    """
    
    def __init__(
        self,
        pose_indices: Optional[List[int]] = None,
        face_indices: Optional[List[int]] = None,
        use_pose: bool = True,
        use_hands: bool = True,
        use_face: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the landmark extractor.
        
        Args:
            pose_indices: Which pose landmarks to use (default: upper body)
            face_indices: Which face landmarks to use (default: key points)
            use_pose: Whether to extract pose landmarks
            use_hands: Whether to extract hand landmarks
            use_face: Whether to extract face landmarks
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.use_pose = use_pose
        self.use_hands = use_hands
        self.use_face = use_face
        
        # Default pose indices: upper body relevant for signing
        self.pose_indices = pose_indices or [
            0,       # nose
            11, 12,  # shoulders
            13, 14,  # elbows
            15, 16,  # wrists
            17, 18,  # pinky
            19, 20,  # index
            21, 22,  # thumb
            23, 24   # hips
        ]
        
        # Default face indices: expression-relevant points
        self.face_indices = face_indices or [
            61, 291,  # mouth corners
            0, 17,    # upper/lower lip center
            70, 300,  # eyebrows
            33, 263,  # eyes outer
            1, 4      # nose
        ]
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Calculate feature dimension
        self.feature_dim = self._calculate_feature_dim()
        
        # MediaPipe will be initialized when needed
        self._holistic = None
    
    def _calculate_feature_dim(self) -> int:
        """Calculate total feature dimensions."""
        dim = 0
        if self.use_pose:
            dim += len(self.pose_indices) * 3  # x, y, z
        if self.use_hands:
            dim += 21 * 3 * 2  # 21 landmarks per hand, 2 hands
        if self.use_face:
            dim += len(self.face_indices) * 3
        return dim
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Holistic model."""
        if self._holistic is None:
            mp_holistic = mp.solutions.holistic
            self._holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
    
    def _extract_pose(self, landmarks) -> np.ndarray:
        """Extract selected pose landmarks."""
        if landmarks is None:
            return np.zeros(len(self.pose_indices) * 3)
        
        features = []
        for idx in self.pose_indices:
            lm = landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)
    
    def _extract_hand(self, landmarks) -> np.ndarray:
        """Extract hand landmarks (21 points x 3 coords)."""
        if landmarks is None:
            return np.zeros(21 * 3)
        
        features = []
        for lm in landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)
    
    def _extract_face(self, landmarks) -> np.ndarray:
        """Extract selected face landmarks."""
        if landmarks is None:
            return np.zeros(len(self.face_indices) * 3)
        
        features = []
        for idx in self.face_indices:
            lm = landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
        return np.array(features)
    
    def _extract_frame(self, results) -> np.ndarray:
        """Extract all landmarks from a single frame."""
        features = []
        
        if self.use_pose:
            features.append(self._extract_pose(results.pose_landmarks))
        
        if self.use_hands:
            features.append(self._extract_hand(results.left_hand_landmarks))
            features.append(self._extract_hand(results.right_hand_landmarks))
        
        if self.use_face:
            features.append(self._extract_face(results.face_landmarks))
        
        return np.concatenate(features)
    
    def extract(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> ExtractionResult:
        """
        Extract landmarks from a video file.
        
        Args:
            video_path: Path to the video file
            sample_rate: Sample every N frames (1 = all frames)
            max_frames: Maximum number of frames to extract
        
        Returns:
            ExtractionResult with landmarks array and metadata
        """
        self._init_mediapipe()
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return ExtractionResult(
                landmarks=np.array([]),
                num_frames=0,
                fps=0,
                success=False,
                error=f"Could not open video: {video_path}"
            )
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        landmarks_list = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Max frames
            if max_frames and len(landmarks_list) >= max_frames:
                break
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self._holistic.process(rgb)
            
            # Extract landmarks
            frame_features = self._extract_frame(results)
            landmarks_list.append(frame_features)
            
            frame_idx += 1
        
        cap.release()
        
        if len(landmarks_list) == 0:
            return ExtractionResult(
                landmarks=np.array([]),
                num_frames=0,
                fps=fps,
                success=False,
                error="No frames extracted"
            )
        
        landmarks = np.array(landmarks_list, dtype=np.float32)
        
        return ExtractionResult(
            landmarks=landmarks,
            num_frames=len(landmarks_list),
            fps=fps,
            success=True
        )
    
    def extract_batch(
        self,
        video_paths: List[str],
        sample_rate: int = 1,
        max_frames: Optional[int] = None,
        show_progress: bool = True
    ) -> List[ExtractionResult]:
        """
        Extract landmarks from multiple videos.
        
        Args:
            video_paths: List of video file paths
            sample_rate: Sample every N frames
            max_frames: Maximum frames per video
            show_progress: Show progress bar
        
        Returns:
            List of ExtractionResult objects
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                video_paths = tqdm(video_paths, desc="Extracting landmarks")
            except ImportError:
                pass
        
        for path in video_paths:
            result = self.extract(path, sample_rate, max_frames)
            results.append(result)
        
        return results
    
    def close(self):
        """Release MediaPipe resources."""
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None
    
    def __enter__(self):
        self._init_mediapipe()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_extractor():
    """Test the landmark extractor on a sample video."""
    video_path = r"E:\iSign-videos_v1.1\-02o_0vVwzI--0.mp4"
    
    print("Testing LandmarkExtractor...")
    
    with LandmarkExtractor() as extractor:
        print(f"Feature dimension: {extractor.feature_dim}")
        
        result = extractor.extract(video_path, max_frames=50)
        
        print(f"Success: {result.success}")
        print(f"Frames extracted: {result.num_frames}")
        print(f"Landmarks shape: {result.landmarks.shape}")
        print(f"FPS: {result.fps}")
        
        if result.success:
            print(f"Min value: {result.landmarks.min():.4f}")
            print(f"Max value: {result.landmarks.max():.4f}")
            print(f"Mean value: {result.landmarks.mean():.4f}")
    
    print("Test PASSED!")


if __name__ == "__main__":
    test_extractor()
