"""
ISL Translation System - GPU-Optimized Preprocessing Pipeline
==============================================================
Optimized for A100 GPUs on Linux with:
- Multiprocessing for parallel landmark extraction
- GPU-accelerated feature processing (CuPy/PyTorch)
- Batch processing for maximum GPU utilization
- Hardware video decoding (NVDEC) support
- Async I/O for file operations

Usage:
    python preprocessing_gpu.py --videos_dir /path/to/videos --output_dir /path/to/output
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Queue, Process, Manager
import multiprocessing as mp
from queue import Empty
import threading
import time
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Try importing GPU-accelerated libraries
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cupy_gaussian_filter1d
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available. Install with: pip install cupy-cuda12x")

import mediapipe as mp_lib
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from tqdm import tqdm

from config import data_config, landmark_config


@dataclass
class GPUConfig:
    """GPU processing configuration."""
    # Device settings
    num_gpus: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    primary_device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Batch processing
    batch_size: int = 64  # Videos to process in parallel on GPU
    feature_batch_size: int = 256  # Features to process per GPU batch
    
    # Multiprocessing
    num_extraction_workers: int = min(16, cpu_count())  # MediaPipe workers
    num_io_workers: int = 8  # File I/O workers
    
    # Memory management
    pin_memory: bool = True
    prefetch_factor: int = 4
    
    # Video decoding
    use_nvdec: bool = True  # Use NVIDIA hardware decoder
    
    # Processing
    use_cupy: bool = CUPY_AVAILABLE
    use_mixed_precision: bool = True  # FP16 for faster processing


gpu_config = GPUConfig()


class GPUFeatureProcessor:
    """GPU-accelerated feature processing using PyTorch/CuPy."""
    
    def __init__(self, device: str = "cuda:0", use_cupy: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cupy = use_cupy and CUPY_AVAILABLE
        self.sigma = data_config.gaussian_sigma
        
        # Pre-compute Gaussian kernel for convolution
        self._setup_gaussian_kernel()
        
        print(f"GPUFeatureProcessor initialized on {self.device}")
        print(f"  CuPy acceleration: {self.use_cupy}")
        print(f"  Mixed precision: {gpu_config.use_mixed_precision}")
    
    def _setup_gaussian_kernel(self):
        """Pre-compute Gaussian kernel for GPU smoothing."""
        if self.sigma <= 0:
            self.gaussian_kernel = None
            return
        
        # Create 1D Gaussian kernel
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / self.sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Shape for 1D conv: (out_channels, in_channels, kernel_size)
        self.gaussian_kernel = kernel.view(1, 1, -1).to(self.device)
        self.kernel_padding = kernel_size // 2
    
    @torch.no_grad()
    def process_batch(self, landmarks_batch: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of landmark sequences on GPU.
        
        Args:
            landmarks_batch: List of (T, 46, 3) landmark arrays
            
        Returns:
            List of (T, 414) feature arrays
        """
        results = []
        
        for landmarks in landmarks_batch:
            features = self.process_single(landmarks)
            results.append(features)
        
        return results
    
    @torch.no_grad()
    def process_single(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Process single landmark sequence on GPU.
        
        Args:
            landmarks: (T, 46, 3) landmark array
            
        Returns:
            (T, 414) feature array
        """
        T = landmarks.shape[0]
        
        # Move to GPU
        if self.use_cupy:
            return self._process_cupy(landmarks)
        else:
            return self._process_torch(landmarks)
    
    def _process_torch(self, landmarks: np.ndarray) -> np.ndarray:
        """Process using PyTorch on GPU."""
        T = landmarks.shape[0]
        
        # Convert to tensor
        landmarks_t = torch.from_numpy(landmarks).to(self.device)
        
        # Use FP16 for faster processing on A100
        if gpu_config.use_mixed_precision:
            landmarks_t = landmarks_t.half()
        
        # Flatten: (T, 46, 3) -> (T, 138)
        positions = landmarks_t.reshape(T, -1)
        
        # Normalize
        positions = self._normalize_torch(positions, landmarks_t)
        
        # Smooth positions
        positions = self._smooth_torch(positions)
        
        # Compute velocity
        velocity = self._compute_derivative_torch(positions)
        velocity = self._smooth_torch(velocity)
        
        # Compute acceleration
        acceleration = self._compute_derivative_torch(velocity)
        
        # Concatenate: (T, 414)
        features = torch.cat([positions, velocity, acceleration], dim=1)
        
        # Handle NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Move back to CPU
        return features.float().cpu().numpy()
    
    def _normalize_torch(self, positions: Tensor, landmarks: Tensor) -> Tensor:
        """Vectorized normalization on GPU."""
        T = landmarks.shape[0]
        normalized = torch.zeros_like(positions)
        
        # Extract components
        pose = landmarks[:, :4, :]          # (T, 4, 3)
        left_hand = landmarks[:, 4:25, :]   # (T, 21, 3)
        right_hand = landmarks[:, 25:46, :] # (T, 21, 3)
        
        # Normalize left hand (center to wrist, scale by hand span)
        left_wrist = left_hand[:, 0:1, :]   # (T, 1, 3)
        left_centered = left_hand - left_wrist
        left_span = torch.norm(left_hand[:, 12, :] - left_hand[:, 0, :], dim=1, keepdim=True) + 1e-6
        left_normalized = left_centered / left_span.unsqueeze(-1)
        
        # Normalize right hand
        right_wrist = right_hand[:, 0:1, :]
        right_centered = right_hand - right_wrist
        right_span = torch.norm(right_hand[:, 12, :] - right_hand[:, 0, :], dim=1, keepdim=True) + 1e-6
        right_normalized = right_centered / right_span.unsqueeze(-1)
        
        # Normalize pose (center to shoulder midpoint, scale by shoulder width)
        shoulder_mid = (pose[:, 0, :] + pose[:, 1, :]) / 2  # (T, 3)
        pose_centered = pose - shoulder_mid.unsqueeze(1)
        shoulder_width = torch.norm(pose[:, 1, :] - pose[:, 0, :], dim=1, keepdim=True) + 1e-6
        pose_normalized = pose_centered / shoulder_width.unsqueeze(-1)
        
        # Reconstruct normalized frame
        norm_landmarks = torch.cat([
            pose_normalized,      # (T, 4, 3)
            left_normalized,      # (T, 21, 3)
            right_normalized      # (T, 21, 3)
        ], dim=1)  # (T, 46, 3)
        
        return norm_landmarks.reshape(T, -1)
    
    def _smooth_torch(self, features: Tensor) -> Tensor:
        """Apply Gaussian smoothing using 1D convolution on GPU."""
        if self.gaussian_kernel is None:
            return features
        
        T, D = features.shape
        
        # Reshape for conv1d: (batch, channels, length) -> (D, 1, T)
        x = features.t().unsqueeze(1)  # (D, 1, T)
        
        # Apply same kernel to all channels
        kernel = self.gaussian_kernel.to(features.dtype)
        smoothed = F.conv1d(x, kernel, padding=self.kernel_padding)
        
        # Reshape back: (D, 1, T) -> (T, D)
        return smoothed.squeeze(1).t()
    
    def _compute_derivative_torch(self, features: Tensor) -> Tensor:
        """Compute first derivative on GPU."""
        derivative = torch.zeros_like(features)
        derivative[1:] = features[1:] - features[:-1]
        derivative[0] = derivative[1]
        return derivative
    
    def _process_cupy(self, landmarks: np.ndarray) -> np.ndarray:
        """Process using CuPy (faster for array operations)."""
        T = landmarks.shape[0]
        
        # Move to GPU
        landmarks_gpu = cp.asarray(landmarks)
        
        # Flatten: (T, 46, 3) -> (T, 138)
        positions = landmarks_gpu.reshape(T, -1)
        
        # Normalize
        positions = self._normalize_cupy(positions, landmarks_gpu)
        
        # Smooth positions
        positions = self._smooth_cupy(positions)
        
        # Compute velocity
        velocity = self._compute_derivative_cupy(positions)
        velocity = self._smooth_cupy(velocity)
        
        # Compute acceleration
        acceleration = self._compute_derivative_cupy(velocity)
        
        # Concatenate: (T, 414)
        features = cp.concatenate([positions, velocity, acceleration], axis=1)
        
        # Handle NaN/Inf
        features = cp.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return cp.asnumpy(features).astype(np.float32)
    
    def _normalize_cupy(self, positions: cp.ndarray, landmarks: cp.ndarray) -> cp.ndarray:
        """Vectorized normalization using CuPy."""
        T = landmarks.shape[0]
        
        # Extract components
        pose = landmarks[:, :4, :]
        left_hand = landmarks[:, 4:25, :]
        right_hand = landmarks[:, 25:46, :]
        
        # Normalize left hand
        left_wrist = left_hand[:, 0:1, :]
        left_centered = left_hand - left_wrist
        left_span = cp.linalg.norm(left_hand[:, 12, :] - left_hand[:, 0, :], axis=1, keepdims=True) + 1e-6
        left_normalized = left_centered / left_span[:, :, cp.newaxis]
        
        # Normalize right hand
        right_wrist = right_hand[:, 0:1, :]
        right_centered = right_hand - right_wrist
        right_span = cp.linalg.norm(right_hand[:, 12, :] - right_hand[:, 0, :], axis=1, keepdims=True) + 1e-6
        right_normalized = right_centered / right_span[:, :, cp.newaxis]
        
        # Normalize pose
        shoulder_mid = (pose[:, 0, :] + pose[:, 1, :]) / 2
        pose_centered = pose - shoulder_mid[:, cp.newaxis, :]
        shoulder_width = cp.linalg.norm(pose[:, 1, :] - pose[:, 0, :], axis=1, keepdims=True) + 1e-6
        pose_normalized = pose_centered / shoulder_width[:, :, cp.newaxis]
        
        # Reconstruct
        norm_landmarks = cp.concatenate([
            pose_normalized,
            left_normalized,
            right_normalized
        ], axis=1)
        
        return norm_landmarks.reshape(T, -1)
    
    def _smooth_cupy(self, features: cp.ndarray) -> cp.ndarray:
        """Apply Gaussian smoothing using CuPy."""
        if self.sigma <= 0:
            return features
        return cupy_gaussian_filter1d(features, sigma=self.sigma, axis=0, mode='nearest')
    
    def _compute_derivative_cupy(self, features: cp.ndarray) -> cp.ndarray:
        """Compute derivative using CuPy."""
        derivative = cp.zeros_like(features)
        derivative[1:] = features[1:] - features[:-1]
        derivative[0] = derivative[1]
        return derivative


def extract_landmarks_worker(args: Tuple[str, str]) -> Optional[Tuple[str, np.ndarray]]:
    """
    Worker function for parallel landmark extraction.
    Runs in separate process to maximize CPU utilization.
    
    Args:
        args: (video_id, video_path)
        
    Returns:
        (video_id, landmarks) or None if extraction fails
    """
    video_id, video_path = args
    
    try:
        # Initialize MediaPipe in worker process
        mp_holistic = mp_lib.solutions.holistic
        holistic = mp_holistic.Holistic(
            static_image_mode=landmark_config.static_image_mode,
            model_complexity=landmark_config.model_complexity,
            min_detection_confidence=landmark_config.min_detection_confidence,
            min_tracking_confidence=landmark_config.min_tracking_confidence,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
        
        # Try hardware-accelerated video decoding
        if gpu_config.use_nvdec:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            # Enable hardware acceleration if available
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        landmarks_list = []
        pose_indices = landmark_config.pose_indices
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic.process(rgb_frame)
            
            # Extract landmarks
            frame_landmarks = _extract_frame_landmarks(results, pose_indices)
            landmarks_list.append(frame_landmarks)
        
        cap.release()
        holistic.close()
        
        if len(landmarks_list) == 0:
            return None
        
        landmarks = np.stack(landmarks_list, axis=0).astype(np.float32)
        return (video_id, landmarks)
        
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        return None


def _extract_frame_landmarks(results, pose_indices: List[int]) -> np.ndarray:
    """Extract 46 landmarks from MediaPipe results."""
    landmarks = []
    
    has_pose = results.pose_landmarks is not None
    has_left_hand = results.left_hand_landmarks is not None
    has_right_hand = results.right_hand_landmarks is not None
    
    # Pose landmarks
    if has_pose:
        for idx in pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([[0.0, 0.0, 0.0]] * len(pose_indices))
    
    # Left hand
    if has_left_hand:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([[0.0, 0.0, 0.0]] * 21)
    
    # Right hand
    if has_right_hand:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([[0.0, 0.0, 0.0]] * 21)
    
    return np.array(landmarks, dtype=np.float32)


class AsyncFileSaver:
    """Async file saver using thread pool."""
    
    def __init__(self, num_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = []
    
    def save_async(self, path: str, data: np.ndarray):
        """Queue file save operation."""
        future = self.executor.submit(np.save, path, data)
        self.futures.append(future)
    
    def wait_all(self):
        """Wait for all save operations to complete."""
        for future in as_completed(self.futures):
            try:
                future.result()
            except Exception as e:
                print(f"Save error: {e}")
        self.futures.clear()
    
    def close(self):
        """Shutdown executor."""
        self.wait_all()
        self.executor.shutdown(wait=True)


def preprocess_dataset_gpu(
    videos_dir: str,
    annotations_file: str,
    output_dir: str,
    subset_ratio: float = 1.0,
    seed: int = 42,
    num_workers: int = None,
    device: str = "cuda:0"
) -> Dict[str, str]:
    """
    GPU-optimized dataset preprocessing.
    
    Pipeline:
    1. Parallel landmark extraction (multi-process, CPU)
    2. Batch feature processing (GPU)
    3. Async file saving (thread pool)
    
    Args:
        videos_dir: Directory containing video files
        annotations_file: CSV file with video_id and text columns
        output_dir: Directory to save preprocessed features
        subset_ratio: Fraction of dataset to use (0.0-1.0)
        seed: Random seed
        num_workers: Number of extraction workers (default: auto)
        device: CUDA device for GPU processing
        
    Returns:
        Dictionary mapping video_id to preprocessed file path
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    if num_workers is None:
        num_workers = gpu_config.num_extraction_workers
    
    # Print configuration
    print("\n" + "=" * 60)
    print("GPU-Optimized ISL Preprocessing")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"GPUs available: {gpu_config.num_gpus}")
    print(f"Extraction workers: {num_workers}")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    print(f"NVDEC enabled: {gpu_config.use_nvdec}")
    print("=" * 60 + "\n")
    
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
    
    # Split dataset
    if 'signer_id' in annotations.columns:
        splits = _stratified_split_by_signer(annotations, seed)
    else:
        splits = _random_split(annotations, seed)
    
    # Initialize GPU processor and file saver
    gpu_processor = GPUFeatureProcessor(device=device, use_cupy=gpu_config.use_cupy)
    file_saver = AsyncFileSaver(num_workers=gpu_config.num_io_workers)
    
    # Process each split
    processed_files = {}
    metadata_list = []
    
    for split_name, split_data in splits.items():
        print(f"\n{'='*40}")
        print(f"Processing {split_name} split ({len(split_data)} videos)")
        print(f"{'='*40}")
        
        # Prepare work items
        work_items = []
        text_map = {}
        
        for idx, row in split_data.iterrows():
            video_id = row.get('uid', row.get('video_id', None))
            if video_id is None:
                continue
            
            video_path = os.path.join(videos_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(videos_dir, video_id)
                if not os.path.exists(video_path):
                    continue
            
            work_items.append((video_id, video_path))
            text_map[video_id] = row['text']
        
        print(f"Found {len(work_items)} valid videos")
        
        # Phase 1: Parallel landmark extraction
        print("\nPhase 1: Extracting landmarks (parallel CPU)...")
        landmarks_results = {}
        
        # Use multiprocessing for landmark extraction
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(extract_landmarks_worker, item): item[0] 
                      for item in work_items}
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Extracting"):
                result = future.result()
                if result is not None:
                    video_id, landmarks = result
                    landmarks_results[video_id] = landmarks
        
        print(f"Successfully extracted landmarks for {len(landmarks_results)} videos")
        
        # Phase 2: GPU feature processing
        print("\nPhase 2: Processing features (GPU)...")
        
        video_ids = list(landmarks_results.keys())
        batch_size = gpu_config.feature_batch_size
        
        for i in tqdm(range(0, len(video_ids), batch_size), desc="GPU Processing"):
            batch_ids = video_ids[i:i + batch_size]
            batch_landmarks = [landmarks_results[vid] for vid in batch_ids]
            
            # Process batch on GPU
            batch_features = gpu_processor.process_batch(batch_landmarks)
            
            # Save results async
            for video_id, features in zip(batch_ids, batch_features):
                # Validate sequence length
                if features.shape[0] < data_config.min_src_len:
                    continue
                
                if features.shape[0] > data_config.max_src_len:
                    features = features[:data_config.max_src_len]
                
                output_path = os.path.join(output_dir, split_name, f"{video_id}.npy")
                file_saver.save_async(output_path, features)
                
                processed_files[video_id] = output_path
                metadata_list.append({
                    'video_id': video_id,
                    'text': text_map[video_id],
                    'split': split_name,
                    'length': features.shape[0],
                    'path': output_path
                })
        
        # Wait for all saves to complete
        file_saver.wait_all()
        print(f"Completed {split_name}: {len([m for m in metadata_list if m['split'] == split_name])} files")
    
    file_saver.close()
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")
    
    # Print statistics
    print("\n" + "=" * 40)
    print("Dataset Statistics")
    print("=" * 40)
    print(f"Total processed: {len(processed_files)}")
    for split_name in ['train', 'val', 'test']:
        count = len(metadata_df[metadata_df['split'] == split_name])
        print(f"  {split_name}: {count}")
    
    return processed_files


def _stratified_split_by_signer(annotations: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    """Split dataset by signer."""
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


def _random_split(annotations: pd.DataFrame, seed: int) -> Dict[str, pd.DataFrame]:
    """Random split."""
    np.random.seed(seed)
    
    indices = np.random.permutation(len(annotations))
    
    n_train = int(len(annotations) * data_config.train_ratio)
    n_val = int(len(annotations) * data_config.val_ratio)
    
    return {
        'train': annotations.iloc[indices[:n_train]],
        'val': annotations.iloc[indices[n_train:n_train + n_val]],
        'test': annotations.iloc[indices[n_train + n_val:]]
    }


# Multi-GPU support
class MultiGPUPreprocessor:
    """
    Distribute preprocessing across multiple GPUs.
    Uses data parallelism for feature processing.
    """
    
    def __init__(self, devices: List[str] = None):
        if devices is None:
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        
        self.devices = devices
        self.processors = [GPUFeatureProcessor(device=d) for d in devices]
        print(f"MultiGPUPreprocessor initialized on {len(devices)} GPUs: {devices}")
    
    def process_batch_multi_gpu(self, landmarks_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process landmarks across multiple GPUs.
        
        Args:
            landmarks_list: List of (T, 46, 3) landmark arrays
            
        Returns:
            List of (T, 414) feature arrays
        """
        n_gpus = len(self.devices)
        n_samples = len(landmarks_list)
        
        # Split work across GPUs
        chunk_size = (n_samples + n_gpus - 1) // n_gpus
        chunks = [landmarks_list[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        # Process in parallel using threads (GPU ops release GIL)
        results = [None] * len(chunks)
        
        def process_chunk(idx, chunk, processor):
            results[idx] = processor.process_batch(chunk)
        
        threads = []
        for i, (chunk, processor) in enumerate(zip(chunks, self.processors)):
            if len(chunk) > 0:
                t = threading.Thread(target=process_chunk, args=(i, chunk, processor))
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()
        
        # Flatten results
        return [item for sublist in results if sublist for item in sublist]


def benchmark_preprocessing(num_samples: int = 100):
    """Benchmark GPU vs CPU preprocessing speed."""
    print("\n" + "=" * 50)
    print("Preprocessing Benchmark")
    print("=" * 50)
    
    # Generate dummy data
    print(f"\nGenerating {num_samples} dummy samples...")
    dummy_landmarks = [np.random.randn(100, 46, 3).astype(np.float32) 
                      for _ in range(num_samples)]
    
    # CPU baseline
    print("\nCPU Processing...")
    from preprocessing import FeatureProcessor as CPUProcessor
    cpu_processor = CPUProcessor()
    
    start = time.time()
    for landmarks in tqdm(dummy_landmarks, desc="CPU"):
        _ = cpu_processor.process(landmarks)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.2f}s ({num_samples/cpu_time:.1f} samples/sec)")
    
    # GPU processing
    if torch.cuda.is_available():
        print("\nGPU Processing (PyTorch)...")
        gpu_processor = GPUFeatureProcessor(device="cuda:0", use_cupy=False)
        
        # Warmup
        _ = gpu_processor.process_batch(dummy_landmarks[:10])
        torch.cuda.synchronize()
        
        start = time.time()
        _ = gpu_processor.process_batch(dummy_landmarks)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU Time: {gpu_time:.2f}s ({num_samples/gpu_time:.1f} samples/sec)")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        # CuPy if available
        if CUPY_AVAILABLE:
            print("\nGPU Processing (CuPy)...")
            cupy_processor = GPUFeatureProcessor(device="cuda:0", use_cupy=True)
            
            # Warmup
            _ = cupy_processor.process_batch(dummy_landmarks[:10])
            cp.cuda.Stream.null.synchronize()
            
            start = time.time()
            _ = cupy_processor.process_batch(dummy_landmarks)
            cp.cuda.Stream.null.synchronize()
            cupy_time = time.time() - start
            print(f"CuPy Time: {cupy_time:.2f}s ({num_samples/cupy_time:.1f} samples/sec)")
            print(f"Speedup vs CPU: {cpu_time/cupy_time:.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Optimized ISL Preprocessing")
    parser.add_argument("--videos_dir", type=str, default=data_config.videos_dir,
                       help="Directory containing video files")
    parser.add_argument("--annotations", type=str, default=data_config.annotations_file,
                       help="CSV file with annotations")
    parser.add_argument("--output_dir", type=str, default=data_config.preprocessed_dir,
                       help="Output directory for preprocessed features")
    parser.add_argument("--subset", type=float, default=1.0,
                       help="Fraction of dataset to process (0.0-1.0)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of extraction workers")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="CUDA device for GPU processing")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark instead of preprocessing")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_preprocessing()
    else:
        # Check GPU availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Running on CPU.")
        else:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Run preprocessing
        preprocess_dataset_gpu(
            videos_dir=args.videos_dir,
            annotations_file=args.annotations,
            output_dir=args.output_dir,
            subset_ratio=args.subset,
            num_workers=args.workers,
            device=args.device
        )
