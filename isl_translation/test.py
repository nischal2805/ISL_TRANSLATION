"""
ISL Translation System - Testing/Inference Script
==================================================
Test model on test set and run inference on new videos.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from config import DEVICE, model_config, vocab_config
from vocab import Vocabulary
from dataset import ISLDataset, collate_fn, create_dataloaders
from model import create_model, ISLTranslationModel
from evaluate import compute_metrics, print_evaluation_report
from preprocessing import LandmarkExtractor, FeatureProcessor


def load_model(checkpoint_path: str, device: torch.device = DEVICE) -> ISLTranslationModel:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    model = create_model()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Parameters: {model.count_parameters():,}")
    
    return model


@torch.no_grad()
def test_on_dataset(
    model: ISLTranslationModel,
    test_loader: torch.utils.data.DataLoader,
    vocab: Vocabulary,
    use_ctc: bool = False
) -> Tuple[List[str], List[str], dict]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        vocab: Vocabulary
        use_ctc: Use CTC decoding instead of GRU decoder
        
    Returns:
        predictions: List of predicted texts
        targets: List of target texts
        metrics: Evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_video_ids = []
    
    for batch in tqdm(test_loader, desc="Testing"):
        features = batch['features'].to(DEVICE)
        feature_lengths = batch['feature_lengths'].to(DEVICE)
        
        # Decode
        if use_ctc:
            output_ids, _ = model.decode(features, feature_lengths, use_ctc=True)
            # CTC decode with vocab
            for i in range(features.size(0)):
                pred_text = vocab.ctc_decode(output_ids[i].tolist())
                all_predictions.append(pred_text)
        else:
            output_ids, _ = model.decode(features, feature_lengths, use_ctc=False)
            for i in range(features.size(0)):
                pred_text = vocab.decode(output_ids[i].tolist())
                all_predictions.append(pred_text)
        
        all_targets.extend(batch['texts'])
        all_video_ids.extend(batch['video_ids'])
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    return all_predictions, all_targets, metrics


@torch.no_grad()
def inference_single_video(
    model: ISLTranslationModel,
    video_path: str,
    vocab: Vocabulary,
    use_ctc: bool = False
) -> str:
    """
    Run inference on a single video file.
    
    Args:
        model: Trained model
        video_path: Path to video file
        vocab: Vocabulary
        use_ctc: Use CTC decoding
        
    Returns:
        Predicted text
    """
    # Extract landmarks
    extractor = LandmarkExtractor()
    landmarks = extractor.extract_from_video(video_path)
    extractor.close()
    
    if landmarks is None:
        raise ValueError(f"Failed to extract landmarks from {video_path}")
    
    # Process features
    processor = FeatureProcessor()
    features = processor.process(landmarks)
    
    # Convert to tensor
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    feature_lengths = torch.tensor([features.size(1)], dtype=torch.long).to(DEVICE)
    
    # Decode
    model.eval()
    output_ids, output_probs = model.decode(features, feature_lengths, use_ctc=use_ctc)
    
    if use_ctc:
        pred_text = vocab.ctc_decode(output_ids[0].tolist())
    else:
        pred_text = vocab.decode(output_ids[0].tolist())
    
    return pred_text


@torch.no_grad()
def inference_from_features(
    model: ISLTranslationModel,
    features: np.ndarray,
    vocab: Vocabulary,
    use_ctc: bool = False
) -> Tuple[str, np.ndarray]:
    """
    Run inference on preprocessed features.
    
    Args:
        model: Trained model
        features: (T, 414) preprocessed features
        vocab: Vocabulary
        use_ctc: Use CTC decoding
        
    Returns:
        pred_text: Predicted text
        probs: Token probabilities
    """
    model.eval()
    
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    feature_lengths = torch.tensor([features.size(1)], dtype=torch.long).to(DEVICE)
    
    output_ids, output_probs = model.decode(features, feature_lengths, use_ctc=use_ctc)
    
    if use_ctc:
        pred_text = vocab.ctc_decode(output_ids[0].tolist())
    else:
        pred_text = vocab.decode(output_ids[0].tolist())
    
    return pred_text, output_probs[0].cpu().numpy()


def batch_inference(
    model: ISLTranslationModel,
    feature_paths: List[str],
    vocab: Vocabulary,
    batch_size: int = 16
) -> List[str]:
    """
    Run batch inference on multiple preprocessed feature files.
    
    Args:
        model: Trained model
        feature_paths: List of paths to .npy feature files
        vocab: Vocabulary
        batch_size: Batch size for inference
        
    Returns:
        List of predicted texts
    """
    model.eval()
    predictions = []
    
    for i in tqdm(range(0, len(feature_paths), batch_size), desc="Batch inference"):
        batch_paths = feature_paths[i:i + batch_size]
        
        # Load features
        batch_features = []
        batch_lengths = []
        
        for path in batch_paths:
            features = np.load(path)
            batch_features.append(torch.tensor(features, dtype=torch.float32))
            batch_lengths.append(features.shape[0])
        
        # Pad batch
        max_len = max(batch_lengths)
        padded_features = torch.zeros(len(batch_paths), max_len, model_config.input_dim)
        
        for j, feat in enumerate(batch_features):
            padded_features[j, :feat.size(0)] = feat
        
        padded_features = padded_features.to(DEVICE)
        lengths = torch.tensor(batch_lengths, dtype=torch.long).to(DEVICE)
        
        # Decode
        with torch.no_grad():
            output_ids, _ = model.decode(padded_features, lengths)
        
        # Convert to text
        for j in range(len(batch_paths)):
            pred_text = vocab.decode(output_ids[j].tolist())
            predictions.append(pred_text)
    
    return predictions


def export_results(
    predictions: List[str],
    targets: List[str],
    video_ids: List[str],
    output_path: str
):
    """Export test results to CSV."""
    import pandas as pd
    
    df = pd.DataFrame({
        'video_id': video_ids,
        'target': targets,
        'prediction': predictions,
        'match': [p.lower().strip() == t.lower().strip() for p, t in zip(predictions, targets)]
    })
    
    df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test ISL Translation Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'video', 'batch'],
                        help='Inference mode')
    
    # Test mode arguments
    parser.add_argument('--data-dir', type=str,
                        help='Directory with preprocessed features')
    parser.add_argument('--metadata', type=str,
                        help='Path to metadata.csv')
    
    # Video mode arguments
    parser.add_argument('--video', type=str,
                        help='Path to video file for inference')
    
    # Batch mode arguments
    parser.add_argument('--features-dir', type=str,
                        help='Directory with .npy feature files')
    
    # Common arguments
    parser.add_argument('--output', type=str, default='./results.csv',
                        help='Output path for results')
    parser.add_argument('--use-ctc', action='store_true',
                        help='Use CTC decoding instead of GRU decoder')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint)
    vocab = Vocabulary()
    
    if args.mode == 'test':
        # Test on dataset
        if not args.metadata or not args.data_dir:
            print("Error: --metadata and --data-dir required for test mode")
            sys.exit(1)
        
        _, _, test_loader = create_dataloaders(
            args.metadata, args.data_dir, vocab,
            batch_size=args.batch_size
        )
        
        predictions, targets, metrics = test_on_dataset(
            model, test_loader, vocab, use_ctc=args.use_ctc
        )
        
        print_evaluation_report(predictions, targets, sample_size=10)
        
        # Export results
        export_results(predictions, targets, 
                      list(range(len(predictions))), args.output)
        
    elif args.mode == 'video':
        # Single video inference
        if not args.video:
            print("Error: --video required for video mode")
            sys.exit(1)
        
        print(f"\nProcessing video: {args.video}")
        prediction = inference_single_video(model, args.video, vocab, args.use_ctc)
        print(f"\n📝 Prediction: '{prediction}'")
        
    elif args.mode == 'batch':
        # Batch inference
        if not args.features_dir:
            print("Error: --features-dir required for batch mode")
            sys.exit(1)
        
        feature_paths = list(Path(args.features_dir).glob('*.npy'))
        print(f"\nFound {len(feature_paths)} feature files")
        
        predictions = batch_inference(model, feature_paths, vocab, args.batch_size)
        
        # Save predictions
        with open(args.output, 'w') as f:
            for path, pred in zip(feature_paths, predictions):
                f.write(f"{path.stem}\t{pred}\n")
        
        print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
