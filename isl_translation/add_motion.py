"""
Add velocity and acceleration to existing .npy files.
Much faster than re-extracting landmarks from videos.
"""
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import shutil

def add_motion_features(input_dir: str, output_dir: str, sigma: float = 1.0):
    """
    Convert (T, 138) position-only files to (T, 414) with velocity + acceleration.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(input_path.glob("*.npy"))
    print(f"Processing {len(npy_files)} files")
    
    for npy_file in tqdm(npy_files):
        # Load position data (T, 138)
        positions = np.load(npy_file)
        
        if positions.shape[1] == 414:
            shutil.copy(npy_file, output_path / npy_file.name)
            continue
        
        # Compute velocity
        velocity = np.zeros_like(positions)
        velocity[1:] = positions[1:] - positions[:-1]
        
        # Compute acceleration
        acceleration = np.zeros_like(positions)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        # Smooth
        velocity = gaussian_filter1d(velocity, sigma=sigma, axis=0)
        acceleration = gaussian_filter1d(acceleration, sigma=sigma, axis=0)
        
        # Concatenate: (T, 414)
        features = np.concatenate([positions, velocity, acceleration], axis=1)
        
        # Save
        np.save(output_path / npy_file.name, features.astype(np.float32))


if __name__ == "__main__":
    base = "/media/rvcse22/CSERV/kortex_sem5/ramita"
    
    splits = ["train", "val", "test"]
    
    # Step 1: Backup
    print("Creating backup...")
    for split in splits:
        src = Path(f"{base}/outpu_final/{split}")
        dst = Path(f"{base}/outpu_final_backup/{split}")
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            print(f"  ✓ Backed up {split}")
    
    # Step 2: Process (read from backup, write to original location)
    print("\nProcessing...")
    for split in splits:
        input_dir = f"{base}/outpu_final_backup/{split}"
        output_dir = f"{base}/outpu_final/{split}"
        print(f"\n[{split}]")
        add_motion_features(input_dir, output_dir)
    
    # Step 3: Verify
    print("\n" + "="*50)
    print("Verification:")
    for split in splits:
        sample = next(Path(f"{base}/outpu_final/{split}").glob("*.npy"))
        data = np.load(sample)
        status = "✓" if data.shape[1] == 414 else "✗"
        print(f"  {split}: shape={data.shape} {status}")