"""
ISL Translation System - Real-time Demo
========================================
OpenCV-based webcam demo with landmark visualization.
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import time
from collections import deque
from typing import Optional, Tuple

from config import DEVICE, landmark_config, data_config
from vocab import Vocabulary
from model import create_model, ISLTranslationModel
from preprocessing import FeatureProcessor


class RealtimeISLTranslator:
    """Real-time ISL translation with webcam."""
    
    def __init__(
        self,
        model_path: str,
        buffer_size: int = 60,
        prediction_interval: int = 30,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize real-time translator.
        
        Args:
            model_path: Path to trained model checkpoint
            buffer_size: Number of frames to buffer (2 seconds at 30fps)
            prediction_interval: Frames between predictions
            confidence_threshold: Minimum confidence to display prediction
        """
        self.buffer_size = buffer_size
        self.prediction_interval = prediction_interval
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=landmark_config.model_complexity,
            min_detection_confidence=landmark_config.min_detection_confidence,
            min_tracking_confidence=landmark_config.min_tracking_confidence,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        
        # Landmark indices (same as preprocessing)
        self.pose_indices = landmark_config.pose_indices
        
        # Load model
        self.vocab = Vocabulary()
        self.model = self._load_model(model_path)
        
        # Feature processor
        self.processor = FeatureProcessor()
        
        # Frame buffer
        self.landmark_buffer = deque(maxlen=buffer_size)
        
        # State
        self.frame_count = 0
        self.current_prediction = ""
        self.current_confidence = 0.0
        self.prediction_history = deque(maxlen=5)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.time()
    
    def _load_model(self, model_path: str) -> ISLTranslationModel:
        """Load trained model."""
        model = create_model()
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        
        print(f"Model loaded: {model.count_parameters():,} parameters")
        return model
    
    def _extract_landmarks(self, results) -> Optional[np.ndarray]:
        """Extract 46 landmarks from MediaPipe results."""
        landmarks = []
        
        has_pose = results.pose_landmarks is not None
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None
        
        if not (has_pose or has_left or has_right):
            return None
        
        # Pose (4 points)
        if has_pose:
            for idx in self.pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * 4)
        
        # Left hand (21 points)
        if has_left:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        # Right hand (21 points)
        if has_right:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([[0.0, 0.0, 0.0]] * 21)
        
        return np.array(landmarks, dtype=np.float32)
    
    @torch.no_grad()
    def _predict(self) -> Tuple[str, float]:
        """Run prediction on buffered landmarks."""
        if len(self.landmark_buffer) < 10:
            return "", 0.0
        
        # Stack landmarks
        landmarks = np.stack(list(self.landmark_buffer), axis=0)  # (T, 46, 3)
        
        # Process features
        features = self.processor.process(landmarks)  # (T, 414)
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        feature_lengths = torch.tensor([features.size(1)], dtype=torch.long).to(DEVICE)
        
        # Decode
        output_ids, output_probs = self.model.decode(features, feature_lengths)
        
        # Get prediction
        pred_text = self.vocab.decode(output_ids[0].tolist())
        
        # Calculate confidence (mean of token probabilities)
        valid_probs = output_probs[0][output_probs[0] > 0]
        confidence = valid_probs.mean().item() if len(valid_probs) > 0 else 0.0
        
        return pred_text, confidence
    
    def _draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw landmarks on frame."""
        # Draw pose landmarks (only shoulders and elbows)
        if results.pose_landmarks:
            # Custom drawing for selected pose points
            h, w = frame.shape[:2]
            for idx in self.pose_indices:
                lm = results.pose_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (0, 200, 200), 2)
        
        # Draw hand landmarks
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return frame
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Prediction text
        if self.current_prediction:
            text = f"Prediction: {self.current_prediction}"
            cv2.putText(frame, text, (20, h - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Confidence bar
            bar_width = int(300 * self.current_confidence)
            cv2.rectangle(frame, (20, h - 40), (320, h - 20), (100, 100, 100), -1)
            
            color = (0, 255, 0) if self.current_confidence > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), color, -1)
            
            conf_text = f"{self.current_confidence:.0%}"
            cv2.putText(frame, conf_text, (330, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Waiting for signs...", (20, h - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
        
        # FPS display
        if len(self.fps_buffer) > 0:
            fps = np.mean(list(self.fps_buffer))
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Buffer indicator
        buffer_fill = len(self.landmark_buffer) / self.buffer_size
        cv2.rectangle(frame, (w - 120, 50), (w - 20, 60), (100, 100, 100), -1)
        cv2.rectangle(frame, (w - 120, 50), (w - 120 + int(100 * buffer_fill), 60), (0, 255, 0), -1)
        cv2.putText(frame, "Buffer", (w - 120, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'c' to clear buffer", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: BGR frame from webcam
            
        Returns:
            Annotated frame
        """
        # Calculate FPS
        current_time = time.time()
        self.fps_buffer.append(1.0 / (current_time - self.last_time + 1e-6))
        self.last_time = current_time
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.holistic.process(rgb_frame)
        
        # Extract and buffer landmarks
        landmarks = self._extract_landmarks(results)
        if landmarks is not None:
            self.landmark_buffer.append(landmarks)
        
        # Run prediction periodically
        self.frame_count += 1
        if self.frame_count % self.prediction_interval == 0:
            pred_text, confidence = self._predict()
            
            if confidence > self.confidence_threshold:
                self.current_prediction = pred_text
                self.current_confidence = confidence
                self.prediction_history.append(pred_text)
        
        # Draw annotations
        frame = self._draw_landmarks(frame, results)
        frame = self._draw_ui(frame)
        
        return frame
    
    def clear_buffer(self):
        """Clear the landmark buffer."""
        self.landmark_buffer.clear()
        self.current_prediction = ""
        self.current_confidence = 0.0
    
    def run(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        Run real-time translation loop.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("\n" + "=" * 50)
        print("ISL Real-time Translation Demo")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Clear buffer")
        print("  's' - Screenshot")
        print("=" * 50 + "\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('ISL Translation', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_buffer()
                    print("Buffer cleared")
                elif key == ord('s'):
                    cv2.imwrite(f'screenshot_{int(time.time())}.png', annotated_frame)
                    print("Screenshot saved")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
    
    def close(self):
        """Release resources."""
        self.holistic.close()


def demo_without_model():
    """
    Demo mode without trained model (just visualization).
    Useful for testing landmark extraction.
    """
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    pose_indices = [11, 12, 13, 14]  # shoulders + elbows
    
    print("\n" + "=" * 50)
    print("ISL Landmark Visualization Demo")
    print("(No model loaded - visualization only)")
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50 + "\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)
            
            # Draw pose points
            if results.pose_landmarks:
                for idx in pose_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)
            
            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detection status
            status = []
            if results.pose_landmarks:
                status.append("Pose")
            if results.left_hand_landmarks:
                status.append("L-Hand")
            if results.right_hand_landmarks:
                status.append("R-Hand")
            
            status_text = "Detected: " + ", ".join(status) if status else "No landmarks detected"
            cv2.putText(frame, status_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('ISL Landmark Demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ISL Real-time Demo')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--width', type=int, default=1280,
                        help='Frame width')
    parser.add_argument('--height', type=int, default=720,
                        help='Frame height')
    parser.add_argument('--buffer-size', type=int, default=60,
                        help='Frame buffer size')
    parser.add_argument('--interval', type=int, default=30,
                        help='Prediction interval (frames)')
    parser.add_argument('--demo-only', action='store_true',
                        help='Run visualization demo without model')
    
    args = parser.parse_args()
    
    if args.demo_only or args.model is None:
        demo_without_model()
    else:
        translator = RealtimeISLTranslator(
            model_path=args.model,
            buffer_size=args.buffer_size,
            prediction_interval=args.interval
        )
        translator.run(args.camera, args.width, args.height)
