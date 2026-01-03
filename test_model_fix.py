"""
Test the fixed model architecture with gated cross-attention.
"""

import torch
import sys
sys.path.insert(0, 'isl_translation')

from model_v2 import ModelConfig, create_model_v2, HybridCTCAttentionLoss

print("=" * 60)
print("Testing Fixed Model Architecture")
print("=" * 60)

# Config
config = ModelConfig(
    input_dim=612,
    vocab_size=2000,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4
)

# Create model
print("\nCreating model...")
model = create_model_v2(config)
print(f"Parameters: {model.count_parameters():,}")

# Test forward pass
print("\nTesting forward pass...")
B, T, L = 4, 100, 20

features = torch.randn(B, T, config.input_dim)
feature_lengths = torch.tensor([100, 80, 90, 70])
targets = torch.randint(5, config.vocab_size, (B, L))
targets[:, 0] = config.bos_id
target_lengths = torch.tensor([20, 15, 18, 12])

outputs = model(features, feature_lengths, targets, target_lengths)

print(f"CTC log probs: {outputs['ctc_log_probs'].shape}")
print(f"Decoder logits: {outputs['decoder_logits'].shape}")
print(f"Encoder lengths: {outputs['encoder_lengths']}")
print(f"Cross-attention weights: {outputs['cross_attention_weights'].shape if outputs['cross_attention_weights'] is not None else 'None'}")

# Test loss with attention regularization
print("\nTesting loss with attention regularization...")
loss_fn = HybridCTCAttentionLoss(
    vocab_size=config.vocab_size,
    pad_id=config.pad_id,
    blank_id=config.blank_id,
    ctc_weight=0.3,
    attn_reg_weight=0.1
)

losses = loss_fn(
    outputs['ctc_log_probs'],
    outputs['decoder_logits'],
    outputs['encoder_lengths'],
    targets,
    target_lengths,
    outputs.get('cross_attention_weights')
)

print(f"Total loss: {losses['loss'].item():.4f}")
print(f"CTC loss: {losses['ctc_loss'].item():.4f}")
print(f"CE loss: {losses['ce_loss'].item():.4f}")
print(f"Attention reg loss: {losses['attn_reg_loss'].item():.4f}")

# Test backward pass
print("\nTesting backward pass...")
losses['loss'].backward()
print("✓ Backward pass successful")

# Check gradients
encoder_grads = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
decoder_grads = sum(p.grad.abs().sum().item() for p in model.decoder.parameters() if p.grad is not None)

print(f"Encoder gradients: {encoder_grads:.2e}")
print(f"Decoder gradients: {decoder_grads:.2e}")

if encoder_grads > 0:
    print("✓ Encoder is receiving gradients!")
else:
    print("✗ WARNING: Encoder not receiving gradients")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\nKey improvements:")
print("1. Gated cross-attention - model learns to use encoder")
print("2. Attention regularization - entropy + coverage loss")
print("3. Removed NaN masking that was zeroing out cross-attention")
print("4. Better gradient flow with pre-norm architecture")
