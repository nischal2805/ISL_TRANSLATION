"""
Mobile Quantization for ISL Translation Model
=============================================
Converts trained model to quantized version for mobile deployment.

Reduces model size from ~400MB to ~100MB with minimal accuracy loss.
"""

import torch
import torch.quantization as quant
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.translator import ISLTranslator


def quantize_model_dynamic(model, dtype=torch.qint8):
    """
    Dynamic quantization - quantizes weights, activations stay float.
    Best for models with variable input sizes (like videos).
    
    Pros:
    - No calibration data needed
    - Works well with variable sequence lengths
    - Good for LSTM/Transformer models
    
    Size reduction: ~75%
    Speed: 2-3x faster on CPU
    Accuracy loss: <1%
    """
    print("Applying dynamic quantization...")
    
    # Quantize linear layers (most of the model weight)
    quantized_model = quant.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=dtype
    )
    
    return quantized_model


def quantize_model_static(model, calibration_loader, dtype=torch.qint8):
    """
    Static quantization - quantizes weights AND activations.
    Requires calibration data.
    
    Pros:
    - Better compression
    - Faster inference
    
    Cons:
    - Needs calibration data
    - More accuracy loss (1-3%)
    
    Size reduction: ~75%
    Speed: 3-4x faster on CPU
    Accuracy loss: 1-3%
    """
    print("Applying static quantization...")
    
    # Set quantization config
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Fuse modules (conv + bn + relu)
    # For our model, we can fuse some decoder layers
    print("Fusing modules...")
    model = quant.fuse_modules(model, [['decoder.ff.0', 'decoder.ff.1']])  # Linear + GELU
    
    # Prepare for quantization
    model_prepared = quant.prepare(model)
    
    # Calibrate with sample data
    print("Calibrating...")
    model_prepared.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= 100:  # Use 100 batches for calibration
                break
            
            if 'video_frames' in batch:
                features = batch['video_frames']
            else:
                features = batch['features']
            
            feature_lengths = batch['feature_lengths']
            targets = batch['targets']
            target_lengths = batch['target_lengths']
            
            # Forward pass for calibration
            _ = model_prepared(features, feature_lengths, targets, target_lengths)
    
    # Convert to quantized model
    quantized_model = quant.convert(model_prepared)
    
    return quantized_model


def export_for_mobile(model, output_path, example_input):
    """
    Export model to TorchScript for mobile.
    
    This is required for running on Android/iOS.
    """
    print(f"Exporting to TorchScript: {output_path}")
    
    # Trace the model
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    # Save
    traced_model.save(output_path)
    
    print(f"✅ Model exported to {output_path}")


def compare_models(original_model, quantized_model, test_loader, device='cpu'):
    """Compare accuracy of original vs quantized model."""
    print("\nComparing models...")
    
    def evaluate(model, loader):
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                if num_batches >= 50:  # Sample 50 batches
                    break
                
                if 'video_frames' in batch:
                    features = batch['video_frames'].to(device)
                else:
                    features = batch['features'].to(device)
                
                feature_lengths = batch['feature_lengths'].to(device)
                targets = batch['targets'].to(device)
                target_lengths = batch['target_lengths'].to(device)
                
                outputs = model(features, feature_lengths, targets, target_lengths)
                
                # Simple loss calculation
                logits = outputs['logits']
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    original_loss = evaluate(original_model, test_loader)
    quantized_loss = evaluate(quantized_model, test_loader)
    
    print(f"  Original model loss: {original_loss:.4f}")
    print(f"  Quantized model loss: {quantized_loss:.4f}")
    print(f"  Accuracy degradation: {((quantized_loss - original_loss) / original_loss * 100):.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantize ISL Translation Model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='quantized_models',
                        help='Output directory for quantized model')
    parser.add_argument('--method', type=str, choices=['dynamic', 'static'], default='dynamic',
                        help='Quantization method')
    parser.add_argument('--test-data', type=str, default=None,
                        help='Path to test data for comparison')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ISL TRANSLATION - MODEL QUANTIZATION FOR MOBILE")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model
    print("Creating model...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model = ISLTranslator(
        hidden_dim=256,
        vocab_size=tokenizer.vocab_size,
        num_frames=16,
        pretrained='MCG-NJU/videomae-base',
        freeze_epochs=10
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"Original model size: {original_size:.1f} MB")
    
    # Quantize
    if args.method == 'dynamic':
        quantized_model = quantize_model_dynamic(model)
    else:
        # Load test data for calibration
        if not args.test_data:
            raise ValueError("--test-data required for static quantization")
        
        from src.data.video_dataset import create_video_dataloaders
        _, _, test_loader = create_video_dataloaders(
            video_dir=args.test_data,
            csv_path=args.test_data.replace('videos', 'metadata.csv'),
            tokenizer=tokenizer,
            batch_size=8,
            num_workers=0
        )
        quantized_model = quantize_model_static(model, test_loader)
    
    # Get quantized size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1e6
    print(f"Quantized model size: {quantized_size:.1f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    # Save quantized model
    quantized_path = output_dir / 'model_quantized.pt'
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantization_method': args.method,
        'original_checkpoint': args.checkpoint,
    }, quantized_path)
    print(f"\n✅ Quantized model saved to: {quantized_path}")
    
    # Export for mobile (TorchScript)
    print("\nExporting for mobile deployment...")
    mobile_path = output_dir / 'model_mobile.pt'
    
    # Create example input
    example_video = torch.randn(1, 16, 3, 224, 224)
    example_lengths = torch.tensor([16])
    example_targets = torch.randint(0, tokenizer.vocab_size, (1, 20))
    example_target_lengths = torch.tensor([20])
    
    export_for_mobile(
        quantized_model,
        mobile_path,
        (example_video, example_lengths, example_targets, example_target_lengths)
    )
    
    # Compare models if test data provided
    if args.test_data:
        print("\nComparing model accuracy...")
        compare_models(model, quantized_model, test_loader)
    
    print("\n" + "="*70)
    print("QUANTIZATION COMPLETE!")
    print("="*70)
    print(f"\nFiles created:")
    print(f"  1. {quantized_path} - Quantized PyTorch model")
    print(f"  2. {mobile_path} - TorchScript for mobile")
    print(f"\nModel size: {original_size:.1f}MB → {quantized_size:.1f}MB")
    print(f"Ready for mobile deployment! 🚀")
    print("="*70)


if __name__ == '__main__':
    main()
