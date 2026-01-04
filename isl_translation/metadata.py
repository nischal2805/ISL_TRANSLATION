"""
Generate metadata.csv from existing preprocessed .npy files.
Run this script on your remote server where the preprocessed data is stored.

This script will:
1. Scan directories for .npy files
2. Match them with annotations from iSign_v1.1.csv
3. Generate metadata.csv required for training
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def generate_metadata(
    preprocessed_dir: str,
    annotations_file: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    print("=" * 60)
    print("Generating metadata.csv")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load annotations
    # ------------------------------------------------------------
    print(f"\nLoading annotations from: {annotations_file}")
    annotations = pd.read_csv(annotations_file)

    # Ensure required column exists
    if 'text' not in annotations.columns:
        raise ValueError("Annotations file must contain a 'text' column")

    # Find ID column
    id_column = None
    for col in ['uid', 'video_id', 'id', 'filename']:
        if col in annotations.columns:
            id_column = col
            break

    if id_column is None:
        raise ValueError(
            f"Could not find ID column in annotations. "
            f"Available columns: {list(annotations.columns)}"
        )

    print(f"Using '{id_column}' as video ID column")
    print(f"Total annotations: {len(annotations)}")

    # Create lookup: video_id -> cleaned text
    id_to_text = {
        str(row[id_column]): str(row['text']).strip()
        for _, row in annotations.iterrows()
        if pd.notna(row['text'])
    }
    print(f"Valid annotations (non-empty text): {len(id_to_text)}")

    # ------------------------------------------------------------
    # Scan for .npy files
    # ------------------------------------------------------------
    print(f"\nScanning for .npy files in: {preprocessed_dir}")

    preprocessed_path = Path(preprocessed_dir)
    npy_files = []

    split_dirs = ['train', 'val', 'test']
    has_split_dirs = any((preprocessed_path / split).exists() for split in split_dirs)

    if has_split_dirs:
        print("Found split directories (train/val/test)")
        for split in split_dirs:
            split_path = preprocessed_path / split
            if split_path.exists():
                for npy_file in split_path.glob("*.npy"):
                    npy_files.append((npy_file, split))
    else:
        print("No split directories found, will assign splits automatically")
        for npy_file in preprocessed_path.glob("*.npy"):  # Use glob instead of rglob
            npy_files.append((npy_file, None))

    print(f"Found {len(npy_files)} .npy files")

    if len(npy_files) == 0:
        raise RuntimeError("No .npy files found in preprocessed directory")

    # ------------------------------------------------------------
    # Build metadata
    # ------------------------------------------------------------
    metadata = []
    missing_text = 0
    load_errors = 0

    print("\nProcessing files...")
    for npy_path, split in tqdm(npy_files, desc="Building metadata"):
        filename_stem = npy_path.stem

        # Handle filenames like: _z80Fp9SjhE--6.npy or 1782bea75c7d-1.npy
        if '--' in filename_stem:
            video_id = filename_stem.rsplit('--', 1)[0]
        elif '-' in filename_stem:
            # For format like 1782bea75c7d-1, keep the full name as video_id
            video_id = filename_stem
        else:
            video_id = filename_stem

        text = id_to_text.get(video_id)

        if text is None:
            missing_text += 1
            continue

        try:
            features = np.load(npy_path)
            length = int(features.shape[0])
        except Exception as e:
            load_errors += 1
            continue

        metadata.append({
            "video_id": video_id,
            "text": text,
            "split": split,
            "length": length,
            "path": str(npy_path)
        })

    print(f"\nMatched {len(metadata)} files with annotations")
    if missing_text > 0:
        print(f"WARNING: {missing_text} files had no matching annotation")
    if load_errors > 0:
        print(f"WARNING: {load_errors} files failed to load")

    if len(metadata) == 0:
        raise RuntimeError("No files matched with annotations! Check your file naming.")

    metadata_df = pd.DataFrame(metadata)

    # ------------------------------------------------------------
    # Assign splits if needed
    # ------------------------------------------------------------
    if not has_split_dirs or metadata_df['split'].isna().any():
        print("\nAssigning train/val/test splits (80/10/10)...")

        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

        n_total = len(metadata_df)
        n_train = int(0.80 * n_total)
        n_val = int(0.10 * n_total)

        metadata_df.loc[:n_train - 1, 'split'] = 'train'
        metadata_df.loc[n_train:n_train + n_val - 1, 'split'] = 'val'
        metadata_df.loc[n_train + n_val:, 'split'] = 'test'

    # ------------------------------------------------------------
    # Print total number of rows in metadata
    # ------------------------------------------------------------
    total_rows = len(metadata_df)
    print(f"\nTotal rows in metadata.csv: {total_rows}")

    
    # ------------------------------------------------------------
    # Sort for reproducibility
    # ------------------------------------------------------------
    metadata_df = metadata_df.sort_values(
        by=['split', 'video_id']
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Print statistics
    # ------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Dataset Statistics:")
    print("=" * 40)
    for split in ['train', 'val', 'test']:
        count = (metadata_df['split'] == split).sum()
        print(f"  {split}: {count}")
    print(f"  Total: {len(metadata_df)}")
    
    # Text length stats
    text_lengths = metadata_df['text'].str.len()
    print(f"\nText length: min={text_lengths.min()}, max={text_lengths.max()}, avg={text_lengths.mean():.1f}")
    
    # Sequence length stats
    print(f"Seq length: min={metadata_df['length'].min()}, max={metadata_df['length'].max()}, avg={metadata_df['length'].mean():.1f}")

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------
    if output_path is None:
        output_path = os.path.join(preprocessed_dir, "metadata.csv")

    metadata_df.to_csv(output_path, index=False)
    print(f"\nMetadata saved to: {output_path}")

    return metadata_df


def main():
    PREPROCESSED_DIR = "/media/rvcse22/CSERV/kortex_sem5/ramita/outpu_final"
    ANNOTATIONS_FILE = "/media/rvcse22/CSERV/kortex_sem5/ramita/iSign_v1.1.csv"
    OUTPUT_PATH = None

    if not os.path.exists(PREPROCESSED_DIR):
        raise FileNotFoundError(f"Preprocessed directory not found: {PREPROCESSED_DIR}")

    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")

    generate_metadata(
        preprocessed_dir=PREPROCESSED_DIR,
        annotations_file=ANNOTATIONS_FILE,
        output_path=OUTPUT_PATH
    )

    print("\n" + "=" * 60)
    print("SUCCESS! metadata.csv generated.")
    print("You can now start training with:")
    print("  python train.py --gpu-mode large")
    print("=" * 60)


if __name__ == "__main__":
    main()