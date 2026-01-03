"""
Test video-based approach on a small sample before full extraction.
Extract 100 videos and verify the pipeline works.
"""

import sys
sys.path.insert(0, '.')

from extract_video_features import Config, process_single_video
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Test with 100 videos
TEST_SIZE = 100
TEST_OUTPUT = Path("E:/5thsem el/APPROACH 2/video_features_test")

def main():
    print("="*70)
    print("VIDEO PIPELINE TEST - 100 samples")
    print("="*70)
    
    # Override output dir
    Config.OUTPUT_DIR = TEST_OUTPUT
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(Config.CSV_PATH)
    
    # Get first 100 valid videos
    tasks = []
    for _, row in df.iterrows():
        if len(tasks) >= TEST_SIZE:
            break
        
        video_id = row['uid']
        text = row['text']
        video_path = Config.VIDEO_DIR / f"{video_id}.mp4"
        
        if video_path.exists():
            tasks.append((video_id, text, video_path))
    
    print(f"\nProcessing {len(tasks)} test videos...")
    
    # Process
    results = []
    for task in tqdm(tasks):
        result = process_single_video(task)
        results.append(result)
    
    # Summary
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✓ Success: {success}/{len(tasks)}")
    print(f"Output: {TEST_OUTPUT}")
    
    # Load one sample
    if success > 0:
        import numpy as np
        sample_path = next(TEST_OUTPUT.glob("*.npz"))
        data = np.load(sample_path, allow_pickle=True)
        
        print("\nSample data:")
        print(f"  Frames: {data['frames'].shape} (T, H, W, C)")
        if 'landmarks' in data:
            print(f"  Landmarks: {data['landmarks'].shape}")
        print(f"  Text: {data['text']}")
        print(f"  FPS: {data['fps']}")

if __name__ == "__main__":
    main()
