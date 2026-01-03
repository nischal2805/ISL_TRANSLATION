"""
Mobile Optimization and Export for ISL Translation Model
=========================================================

Optimizations for Snapdragon 8 Gen 3:
- INT8 dynamic quantization
- ONNX export with optimization
- TFLite export option
- Knowledge distillation for smaller model
"""

import os
import sys
import argparse
from pathlib import Path
import copy
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.quantization as quant

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def quantize_model_dynamic(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Apply dynamic INT8 quantization (works for any model).
    Best for inference on CPU and mobile.
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    model.eval()
    
    # Dynamic quantization - quantizes weights to INT8
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
        dtype=torch.qint8
    )
    
    return quantized


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_dim: int = 540,
    max_seq_len: int = 300,
    max_output_len: int = 50,
    opset_version: int = 14
):
    """
    Export model to ONNX format for mobile deployment.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy inputs
    batch_size = 1
    dummy_features = torch.randn(batch_size, max_seq_len, input_dim).to(device)
    dummy_lengths = torch.tensor([max_seq_len]).to(device)
    
    # Create export-friendly wrapper
    class ONNXWrapper(nn.Module):
        def __init__(self, model, max_output_len):
            super().__init__()
            self.model = model
            self.max_output_len = max_output_len
        
        def forward(self, features, lengths):
            # Use greedy decoding
            return self.model.translate(features, lengths, max_length=self.max_output_len)
    
    wrapper = ONNXWrapper(model, max_output_len)
    
    print(f"Exporting to ONNX: {output_path}")
    
    # Export
    torch.onnx.export(
        wrapper,
        (dummy_features, dummy_lengths),
        output_path,
        input_names=['features', 'lengths'],
        output_names=['predictions'],
        dynamic_axes={
            'features': {0: 'batch', 1: 'seq_len'},
            'lengths': {0: 'batch'},
            'predictions': {0: 'batch', 1: 'output_len'}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"ONNX export complete: {output_path}")
    
    # Verify export
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("ONNX model verified successfully!")
    except ImportError:
        print("Install onnx package to verify: pip install onnx")


def optimize_onnx(input_path: str, output_path: str):
    """Apply ONNX Runtime optimizations."""
    try:
        from onnxruntime.transformers import optimizer
        
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',  # Similar architecture
            num_heads=6,
            hidden_size=384,
            optimization_options=optimizer.FusionOptions('bert')
        )
        
        # Save with FP16 for mobile
        optimized_model.convert_float_to_float16()
        optimized_model.save_model_to_file(output_path)
        
        print(f"Optimized ONNX saved: {output_path}")
        
    except ImportError:
        print("Install onnxruntime-tools: pip install onnxruntime-tools")


class KnowledgeDistillationTrainer:
    """
    Train smaller student model using larger teacher model.
    Reduces model size while maintaining accuracy.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5  # Weight for distillation loss
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = -100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss combining:
        1. Soft target loss (from teacher)
        2. Hard target loss (from ground truth)
        """
        # Soft target loss (KL divergence)
        soft_targets = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        
        kl_loss = nn.functional.kl_div(
            soft_student, soft_targets, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss (cross entropy)
        ce_loss = nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            target.view(-1),
            ignore_index=ignore_index
        )
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return {
            'loss': total_loss,
            'kl_loss': kl_loss,
            'ce_loss': ce_loss
        }
    
    @torch.no_grad()
    def get_teacher_predictions(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ):
        """Get teacher model predictions."""
        return self.teacher(features, feature_lengths, targets, target_lengths)


class MobileOptimizedModel(nn.Module):
    """
    Lightweight model specifically designed for mobile.
    ~10-15M parameters for Snapdragon 8 Gen 3.
    """
    
    def __init__(
        self,
        input_dim: int = 540,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        vocab_size: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Lightweight Conformer-style encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,  # Smaller FFN
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Lightweight decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers // 2)
        
        # Output
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # CTC head
        self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        
        # Cache for inference
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict:
        """Forward pass for training."""
        # Project features
        x = self.input_proj(features)  # [B, T, H]
        
        # Create attention mask
        B, T, _ = x.shape
        src_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        for i, length in enumerate(feature_lengths):
            src_mask[i, length:] = True
        
        # Encode
        encoder_out = self.encoder(x, src_key_padding_mask=src_mask)
        
        # CTC output
        ctc_logits = self.ctc_head(encoder_out)
        
        # Decoder
        tgt_embed = self.embed(targets)
        
        # Causal mask
        tgt_len = targets.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=x.device
        )
        
        # Target padding mask
        tgt_pad_mask = targets == 1  # Assuming 1 is PAD
        
        decoder_out = self.decoder(
            tgt_embed,
            encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask
        )
        
        # Output logits
        logits = self.output(decoder_out)
        
        return {
            'logits': logits,
            'ctc_logits': ctc_logits,
            'encoder_lengths': feature_lengths
        }
    
    @torch.no_grad()
    def translate(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        max_length: int = 50,
        bos_id: int = 2,
        eos_id: int = 3
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        
        B = features.size(0)
        device = features.device
        
        # Encode
        x = self.input_proj(features)
        T = x.size(1)
        src_mask = torch.zeros(B, T, device=device, dtype=torch.bool)
        for i, length in enumerate(feature_lengths):
            src_mask[i, length:] = True
        
        encoder_out = self.encoder(x, src_key_padding_mask=src_mask)
        
        # Initialize decoder input
        decoder_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_length):
            tgt_embed = self.embed(decoder_input)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.size(1), device=device
            )
            
            decoder_out = self.decoder(
                tgt_embed,
                encoder_out,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask
            )
            
            logits = self.output(decoder_out[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Check if all sequences have EOS
            if (decoder_input == eos_id).any(dim=1).all():
                break
        
        return decoder_input
    
    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def benchmark_inference(
    model: nn.Module,
    input_dim: int = 540,
    seq_len: int = 150,
    num_runs: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Benchmark inference speed."""
    import time
    
    model = model.to(device)
    model.eval()
    
    # Dummy input
    features = torch.randn(1, seq_len, input_dim).to(device)
    lengths = torch.tensor([seq_len]).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model.translate(features, lengths)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            model.translate(features, lengths)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times)
    return {
        'avg_ms': avg_time * 1000,
        'fps': 1.0 / avg_time,
        'device': device
    }


def save_for_mobile(model: nn.Module, output_path: str):
    """Save model in TorchScript format for mobile."""
    model.eval()
    
    # Trace for mobile
    scripted = torch.jit.script(model)
    
    # Optimize for mobile
    optimized = torch.utils.mobile_optimizer.optimize_for_mobile(scripted)
    
    optimized._save_for_lite_interpreter(output_path)
    print(f"Saved mobile model: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Mobile optimization for ISL model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--output-dir', type=str, default='mobile_models')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 quantization')
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export to ONNX')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark inference speed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Determine model type and load
    # For now, create mobile optimized model
    model = MobileOptimizedModel()
    
    # Try to load state dict (will fail if architecture differs)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded model weights")
    except Exception as e:
        print(f"Could not load weights: {e}")
        print("Creating fresh mobile model")
    
    params = model.count_parameters()
    print(f"Parameters: {params['total']/1e6:.1f}M")
    
    # Quantize
    if args.quantize:
        print("\nApplying INT8 quantization...")
        model_q = quantize_model_dynamic(model)
        
        # Save quantized
        quant_path = output_dir / 'model_quantized.pt'
        torch.save(model_q.state_dict(), quant_path)
        print(f"Quantized model saved: {quant_path}")
        
        # Compare sizes
        import os
        orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Original size: {orig_size / 1e6:.1f} MB")
    
    # Export ONNX
    if args.export_onnx:
        print("\nExporting to ONNX...")
        onnx_path = output_dir / 'model.onnx'
        export_to_onnx(model, str(onnx_path))
        
        # Optimize ONNX
        opt_path = output_dir / 'model_optimized.onnx'
        optimize_onnx(str(onnx_path), str(opt_path))
    
    # Benchmark
    if args.benchmark:
        print("\nBenchmarking inference...")
        
        # CPU benchmark
        cpu_results = benchmark_inference(model, device='cpu')
        print(f"CPU: {cpu_results['avg_ms']:.1f}ms per inference")
        
        # GPU benchmark if available
        if torch.cuda.is_available():
            gpu_results = benchmark_inference(model, device='cuda')
            print(f"GPU: {gpu_results['avg_ms']:.1f}ms per inference")
        
        # Quantized benchmark
        if args.quantize:
            quant_results = benchmark_inference(model_q, device='cpu')
            print(f"Quantized CPU: {quant_results['avg_ms']:.1f}ms per inference")


if __name__ == '__main__':
    main()
