"""
Script to split existing landmark files into train/val/test directories.
Does NOT generate metadata - only organizes files into split folders.

Usage on GPU server:
    python split_dataset.py --input_dir /media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final

This will create:
    /media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/train/
    /media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/val/
    /media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final/test/
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def get_label_from_filename(filename: str) -> str:
    """
    Extract label (video_id) from filename.
    
    Your format:
        "_z80Fp9SjhE--6.npy" -> "_z80Fp9SjhE"
        "_ZO5pgvr47o--0.npy" -> "_ZO5pgvr47o"
    """
    name = Path(filename).stem
    
    # Pattern: video_id--index (e.g., "_z80Fp9SjhE--6.npy" -> "_z80Fp9SjhE")
    if '--' in name:
        parts = name.rsplit('--', 1)
        return parts[0]
    
    # Fallback: return full name
    return name


def scan_landmark_files(input_dir: str) -> list:
    """
    Scan directory for landmark files (.npy or .npz).
    Returns list of (filepath, label) tuples.
    """
    files = []
    input_path = Path(input_dir)
    
    # Skip train/val/test directories if they exist
    skip_dirs = {'train', 'val', 'test'}
    
    # Check if files are organized in label folders
    has_label_folders = False
    for item in input_path.iterdir():
        if item.is_dir() and item.name not in skip_dirs and not item.name.startswith('.'):
            npy_files = list(item.glob("*.npy")) + list(item.glob("*.npz"))
            if npy_files:
                has_label_folders = True
                break
    
    if has_label_folders:
        print("Detected: Files organized in label folders")
        # Scan label folders
        for label_dir in input_path.iterdir():
            if label_dir.is_dir() and label_dir.name not in skip_dirs and not label_dir.name.startswith('.'):
                label = label_dir.name
                for file in label_dir.glob("*.npy"):
                    files.append((str(file), label))
                for file in label_dir.glob("*.npz"):
                    files.append((str(file), label))
    else:
        print("Detected: Flat file structure (extracting labels from filenames)")
        # Scan flat structure
        for file in input_path.glob("*.npy"):
            label = get_label_from_filename(file.name)
            files.append((str(file), label))
        for file in input_path.glob("*.npz"):
            label = get_label_from_filename(file.name)
            files.append((str(file), label))
    
    return files


def create_random_splits(files: list, train_ratio: float = 0.8, 
                          val_ratio: float = 0.1, test_ratio: float = 0.1,
                          seed: int = 42) -> dict:
    """
    Create random train/val/test splits.
    Each file is randomly assigned to a split regardless of its label/video.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle all files
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    n = len(files_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': files_shuffled[:n_train],
        'val': files_shuffled[n_train:n_train + n_val],
        'test': files_shuffled[n_train + n_val:]
    }
    
    return splits


def organize_files(splits: dict, output_dir: str, use_symlinks: bool = True,
                   use_move: bool = False) -> None:
    """
    Organize files into train/val/test directories.
    
    Args:
        splits: Dictionary with train/val/test file lists
        output_dir: Base output directory
        use_symlinks: Create symlinks instead of copying (saves space)
        use_move: Move files instead of copy/symlink
    """
    for split_name, files in splits.items():
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOrganizing {split_name} split ({len(files)} files)...")
        
        for filepath, _ in tqdm(files, desc=split_name):
            src = Path(filepath)
            dest = split_dir / src.name
            
            # Skip if destination already exists
            if dest.exists():
                continue
            
            if use_move:
                shutil.move(str(src), str(dest))
            elif use_symlinks:
                try:
                    if dest.is_symlink():
                        dest.unlink()
                    os.symlink(str(src.absolute()), str(dest))
                except OSError:
                    # Symlinks not supported, copy instead
                    shutil.copy2(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))


def print_statistics(splits: dict) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 50)
    print("DATASET SPLIT STATISTICS")
    print("=" * 50)
    
    total = sum(len(files) for files in splits.values())
    print(f"\nTotal files: {total}")
    
    for split_name, files in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Files: {len(files)} ({100*len(files)/total:.1f}%)")
    
    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Split landmark files into train/val/test directories'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing landmark .npy files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as input_dir)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of creating symlinks')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying (saves space, modifies original)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    output_dir = args.output_dir or args.input_dir
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: train={args.train_ratio:.0%}, val={args.val_ratio:.0%}, test={args.test_ratio:.0%}")
    
    # Scan for files
    print("\nScanning for landmark files...")
    files = scan_landmark_files(args.input_dir)
    
    if not files:
        print("\nERROR: No .npy or .npz files found!")
        print("Please check your input directory path.")
        return
    
    print(f"Found {len(files)} landmark files")
    
    # Show sample files
    print("\nSample files:")
    for filepath, label in files[:5]:
        print(f"  {Path(filepath).name}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more")
    
    # Create splits
    print("\nCreating random splits...")
    splits = create_random_splits(
        files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Print statistics
    print_statistics(splits)
    
    # Organize files
    print("\nOrganizing files into split directories...")
    organize_files(
        splits,
        output_dir,
        use_symlinks=not args.copy and not args.move,
        use_move=args.move
    )
    
    print("\n" + "=" * 50)
    print("SPLIT COMPLETE!")
    print("=" * 50)
    print(f"\nCreated directories:")
    print(f"  {output_dir}/train/ ({len(splits['train'])} files)")
    print(f"  {output_dir}/val/ ({len(splits['val'])} files)")
    print(f"  {output_dir}/test/ ({len(splits['test'])} files)")


if __name__ == '__main__':
    main()
