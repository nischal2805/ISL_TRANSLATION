"""
Run training for ISL Translation System.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import set_gpu_mode, training_config, DEVICE
from vocab import Vocabulary
from dataset import create_dataloaders
from model import create_model
from train import Trainer


def main():
    # Configuration - UPDATE THESE PATHS
    PREPROCESSED_DIR = r"E:\5thsem el\APPROACH 2\preprocessed_data"
    METADATA_FILE = r"E:\5thsem el\APPROACH 2\preprocessed_data\metadata.csv"
    OUTPUT_DIR = r"E:\5thsem el\APPROACH 2\checkpoints"
    
    # GPU mode: "small" for RTX 4060, "large" for A100
    GPU_MODE = "small"
    
    print("=" * 60)
    print("ISL Translation - Training")
    print("=" * 60)
    
    # Set GPU mode
    set_gpu_mode(GPU_MODE)
    
    print(f"Device: {DEVICE}")
    print(f"Preprocessed data: {PREPROCESSED_DIR}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Verify paths
    if not os.path.exists(METADATA_FILE):
        print(f"ERROR: Metadata file not found: {METADATA_FILE}")
        print("Please run preprocessing first: python run_preprocessing.py")
        return
    
    # Create vocabulary
    print("\nCreating vocabulary...")
    vocab = Vocabulary()
    print(f"Vocabulary size: {vocab.size}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=METADATA_FILE,
        features_dir=PREPROCESSED_DIR,
        vocab=vocab,
        batch_size=training_config.batch_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        output_dir=OUTPUT_DIR,
        gpu_mode=GPU_MODE
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_metrics = trainer.train()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Val Loss: {best_metrics.get('loss', 'N/A'):.4f}")
    print(f"Best Char Accuracy: {best_metrics.get('char_accuracy', 'N/A'):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
