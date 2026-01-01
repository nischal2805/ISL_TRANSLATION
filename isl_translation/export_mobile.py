"""
Export ISL Translation V2 Model for Flutter/Mobile
==================================================
Exports the streaming encoder to ONNX for real-time mobile inference.

The exported model takes landmark features and outputs CTC log probabilities
that can be decoded on the mobile device.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v2 import ModelConfig, create_model_v2, StreamingEncoder
from tokenizer import BPETokenizer


class ONNXStreamingEncoder(nn.Module):
    """
    ONNX-compatible streaming encoder.
    
    Simplified forward pass for ONNX export.
    """
    
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.ctc_head = model.ctc_head
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, 414) landmark features
            
        Returns:
            ctc_log_probs: (B, T', vocab_size) CTC log probabilities
        """
        # Encode (without lengths for ONNX compatibility)
        x = self.encoder.input_proj(features)
        
        for cnn in self.encoder.cnn_blocks:
            x = cnn(x)
        
        # Subsampling
        x = x.transpose(1, 2)
        x = self.encoder.subsample.conv(x)
        x = x.transpose(1, 2)
        
        # Position encoding
        x = self.encoder.pos_enc(x)
        
        # Conformer layers (no mask for ONNX)
        for layer in self.encoder.layers:
            # Feed-forward 1
            x = layer.ff1(x)
            
            # Self-attention (no mask)
            residual = x
            x = layer.self_attn_norm(x)
            x_attn, _ = layer.self_attn(x, x, x)
            x = residual + layer.self_attn_dropout(x_attn)
            
            # Convolution
            x = layer.conv(x)
            
            # Feed-forward 2
            x = layer.ff2(x)
            x = layer.final_norm(x)
        
        # CTC head
        logits = self.ctc_head.fc(x)
        logits = logits.clamp(min=-100, max=100)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        return log_probs


def export_to_onnx(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_dir: str,
    opset_version: int = 14
):
    """
    Export model to ONNX format.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Output directory for ONNX model and assets
        tokenizer_dir: Directory with tokenizer files
        opset_version: ONNX opset version
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Exporting ISL Translation V2 to ONNX")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ModelConfig(**{k: v for k, v in config_dict.items() if hasattr(ModelConfig, k)})
    else:
        config = ModelConfig()
    
    # Load tokenizer to get vocab size
    tokenizer = BPETokenizer.from_pretrained(tokenizer_dir)
    config.vocab_size = tokenizer.size
    print(f"Vocab size: {config.vocab_size}")
    
    # Create model and load weights
    print("\nCreating model...")
    model = create_model_v2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create ONNX-compatible streaming encoder
    print("\nCreating streaming encoder for ONNX...")
    streaming_model = ONNXStreamingEncoder(model)
    streaming_model.eval()
    
    # Dummy input for tracing
    # Use a reasonable sequence length for mobile
    batch_size = 1
    seq_len = 60  # ~2 seconds at 30fps
    input_dim = config.input_dim
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Export to ONNX
    onnx_path = output_dir / 'isl_streaming_encoder.onnx'
    print(f"\nExporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        streaming_model,
        dummy_input,
        str(onnx_path),
        opset_version=opset_version,
        input_names=['features'],
        output_names=['ctc_log_probs'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'time'},
            'ctc_log_probs': {0: 'batch', 1: 'time'}
        },
        do_constant_folding=True
    )
    
    print(f"ONNX model saved: {onnx_path}")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
    
    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(str(onnx_path))
    test_input = dummy_input.numpy()
    
    ort_outputs = session.run(None, {'features': test_input})
    print(f"ONNX output shape: {ort_outputs[0].shape}")
    
    # Compare with PyTorch output
    with torch.no_grad():
        torch_output = streaming_model(dummy_input)
    
    max_diff = np.abs(ort_outputs[0] - torch_output.numpy()).max()
    print(f"Max difference PyTorch vs ONNX: {max_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✓ ONNX model matches PyTorch output!")
    else:
        print("⚠ Warning: Some numerical differences detected")
    
    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    tokenizer_output_dir = output_dir / 'tokenizer'
    tokenizer_output_dir.mkdir(exist_ok=True)
    
    for file in ['tokenizer.model', 'bpe_tokenizer.model', 'tokenizer_config.json']:
        src = Path(tokenizer_dir) / file
        if src.exists():
            shutil.copy(src, tokenizer_output_dir)
            print(f"  Copied: {file}")
    
    # Save vocabulary for mobile
    print("\nSaving vocabulary...")
    vocab = tokenizer.get_vocab()
    vocab_path = output_dir / 'vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved: {vocab_path}")
    
    # Save model config for mobile
    print("\nSaving model config...")
    mobile_config = {
        'input_dim': config.input_dim,
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'blank_id': config.blank_id,
        'pad_id': config.pad_id,
        'bos_id': config.bos_id,
        'eos_id': config.eos_id,
        'subsampling_factor': 2,  # Temporal subsampling in encoder
        'model_file': 'isl_streaming_encoder.onnx',
        'tokenizer_file': 'tokenizer/bpe_tokenizer.model'
    }
    
    config_path = output_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(mobile_config, f, indent=2)
    print(f"Config saved: {config_path}")
    
    # Get model size
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX model size: {onnx_size:.2f} MB")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nExported files:")
    for f in output_dir.rglob('*'):
        if f.is_file():
            size = os.path.getsize(f) / 1024
            print(f"  {f.relative_to(output_dir)} ({size:.1f} KB)")
    
    print(f"\nTo use in Flutter, copy the '{output_dir}' folder to your assets.")
    print("=" * 60)
    
    return str(onnx_path)


def quantize_onnx(onnx_path: str, output_path: str = None):
    """
    Quantize ONNX model to INT8 for faster mobile inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_quantized.onnx')
    
    print(f"\nQuantizing model to INT8...")
    print(f"Input: {onnx_path}")
    print(f"Output: {output_path}")
    
    quantize_dynamic(
        onnx_path,
        output_path,
        weight_type=QuantType.QInt8
    )
    
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"\nOriginal size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export ISL Translation V2 to ONNX')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--output-dir', type=str, default='mobile_export',
                       help='Output directory')
    parser.add_argument('--tokenizer-dir', type=str,
                       default='tokenizer_model',
                       help='Directory with tokenizer files')
    parser.add_argument('--quantize', action='store_true',
                       help='Also export quantized INT8 model')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    # Export
    onnx_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        opset_version=args.opset
    )
    
    # Quantize if requested
    if args.quantize:
        quantize_onnx(onnx_path)


if __name__ == '__main__':
    main()
