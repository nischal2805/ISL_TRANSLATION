"""
Run preprocessing for ISL Translation System.
Extracts landmarks from videos and creates preprocessed features.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import preprocess_dataset
from config import training_config, set_gpu_mode


def main():
    # Configuration - relative paths (update as needed)
    VIDEOS_DIR = "./data/videos"
    ANNOTATIONS_FILE = "./data/iSign_v1.1.csv"
    OUTPUT_DIR = "./preprocessed_data"
    
    # GPU mode determines subset ratio
    GPU_MODE = "small"  # "small" = 40% data, "large" = 100% data
    
    # Set GPU mode to get correct subset ratio
    set_gpu_mode(GPU_MODE)
    subset_ratio = training_config.subset_ratio
    
    print("=" * 60)
    print("ISL Translation - Preprocessing")
    print("=" * 60)
    print(f"Videos directory: {VIDEOS_DIR}")
    print(f"Annotations file: {ANNOTATIONS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GPU mode: {GPU_MODE}")
    print(f"Subset ratio: {subset_ratio * 100:.0f}%")
    print("=" * 60)
    
    # Verify paths exist
    if not os.path.exists(VIDEOS_DIR):
        print(f"ERROR: Videos directory not found: {VIDEOS_DIR}")
        return
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"ERROR: Annotations file not found: {ANNOTATIONS_FILE}")
        return
    
    # Count videos
    video_count = len([f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4')])
    print(f"Found {video_count} videos")
    
    # Run preprocessing
    print("\nStarting preprocessing...")
    processed_files = preprocess_dataset(
        videos_dir=VIDEOS_DIR,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR,
        subset_ratio=subset_ratio,
        seed=42
    )
    
    print(f"\nPreprocessing complete!")
    print(f"Processed {len(processed_files)} videos")
    print(f"Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
