"""
Model Architecture Verification
================================
Comprehensive verification of encoder-decoder-translator architecture.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

print("="*70)
print("MODEL ARCHITECTURE VERIFICATION")
print("="*70)

# Test 1: Import all components
print("\n1. Component Import Test")
print("-"*70)
try:
    from src.models.encoder import SignEncoder
    from src.models.decoder import TextDecoder
    from src.models.translator import ISLTranslator
    print("✅ All model components imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Encoder Architecture
print("\n2. Encoder Architecture Test")
print("-"*70)
try:
    encoder = SignEncoder(
        output_dim=256,
        pretrained='MCG-NJU/videomae-base',
        freeze_epochs=10,
        dropout=0.1,
        num_frames=16
    )
    
    # Test forward pass
    B, C, T, H, W = 2, 3, 16, 224, 224
    video_frames = torch.randn(B, C, T, H, W)
    feature_lengths = torch.tensor([16, 16])
    
    encoder_out, out_lengths = encoder(video_frames, feature_lengths)
    
    print(f"✅ Input shape: {video_frames.shape}")
    print(f"✅ Encoder output: {encoder_out.shape}")
    print(f"✅ Output lengths: {out_lengths}")
    print(f"✅ Encoder output dim: {encoder_out.size(-1)}")
    
    assert encoder_out.size(0) == B, "Batch size mismatch"
    assert encoder_out.size(-1) == 256, "Output dim mismatch"
    assert encoder._frozen == True, "Encoder should be frozen initially"
    
    # Test freeze/unfreeze
    encoder.unfreeze()
    assert encoder._frozen == False, "Encoder should be unfrozen"
    print("✅ Freeze/unfreeze mechanism works")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"✅ Total params: {total_params:,}")
    print(f"✅ Trainable params: {trainable_params:,}")
    
except Exception as e:
    print(f"❌ Encoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Decoder Architecture
print("\n3. Decoder Architecture Test")
print("-"*70)
try:
    decoder = TextDecoder(
        vocab_size=30522,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        max_len=128,
        pad_id=0
    )
    
    # Test forward pass
    B, T_enc, T_dec = 2, 100, 20
    encoder_out = torch.randn(B, T_enc, 256)
    encoder_lengths = torch.tensor([100, 100])
    targets = torch.randint(0, 30522, (B, T_dec))
    
    logits = decoder(targets, encoder_out, encoder_lengths)
    
    print(f"✅ Encoder output: {encoder_out.shape}")
    print(f"✅ Targets: {targets.shape}")
    print(f"✅ Decoder logits: {logits.shape}")
    print(f"✅ Vocab size: {logits.size(-1)}")
    
    assert logits.shape == (B, T_dec, 30522), "Decoder output shape mismatch"
    
    # Test generation
    generated = decoder.generate(encoder_out, encoder_lengths, max_len=50, bos_id=101, eos_id=102)
    print(f"✅ Generated tokens: {generated.shape}")
    assert generated.size(0) == B, "Generation batch size mismatch"
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"✅ Decoder params: {total_params:,}")
    
except Exception as e:
    print(f"❌ Decoder test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Full Translator Architecture
print("\n4. Full Translator Architecture Test")
print("-"*70)
try:
    model = ISLTranslator(
        hidden_dim=256,
        vocab_size=30522,
        decoder_layers=6,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        pad_id=0,
        use_ctc=True,
        pretrained='MCG-NJU/videomae-base',
        freeze_epochs=10,
        num_frames=16
    )
    
    # Test forward pass
    B, C, T_vid, H, W = 2, 3, 16, 224, 224
    T_text = 20
    
    video_frames = torch.randn(B, C, T_vid, H, W)
    feature_lengths = torch.tensor([16, 16])
    targets = torch.randint(0, 30522, (B, T_text))
    target_lengths = torch.tensor([20, 18])
    
    outputs = model(video_frames, feature_lengths, targets, target_lengths)
    
    print(f"✅ Video input: {video_frames.shape}")
    print(f"✅ Targets: {targets.shape}")
    print(f"✅ Output keys: {list(outputs.keys())}")
    print(f"✅ Logits shape: {outputs['logits'].shape}")
    print(f"✅ CTC logits shape: {outputs['ctc_logits'].shape}")
    print(f"✅ Encoder lengths: {outputs['encoder_lengths']}")
    
    assert 'logits' in outputs, "Missing logits in output"
    assert 'ctc_logits' in outputs, "Missing CTC logits in output"
    assert 'encoder_lengths' in outputs, "Missing encoder lengths"
    assert outputs['logits'].shape == (B, T_text, 30522), "Logits shape mismatch"
    
    # Test translate method
    translated = model.translate(video_frames, feature_lengths, max_len=50, bos_id=101, eos_id=102)
    print(f"✅ Translation output: {translated.shape}")
    assert translated.size(0) == B, "Translation batch size mismatch"
    
    # Test transfer learning callback
    assert model.encoder._frozen == True, "Encoder should be frozen initially"
    model.on_epoch_start(10)
    assert model.encoder._frozen == False, "Encoder should unfreeze at epoch 10"
    print(f"✅ Transfer learning callback works")
    
    # Count parameters
    param_counts = model.count_parameters()
    print(f"✅ Total parameters: {param_counts['total']:,}")
    print(f"✅ Trainable parameters: {param_counts['trainable']:,}")
    
except Exception as e:
    print(f"❌ Translator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Loss Function Integration
print("\n5. Loss Function Integration Test")
print("-"*70)
try:
    from src.training.losses import HybridLoss
    
    loss_fn = HybridLoss(
        vocab_size=30522,
        pad_id=0,
        ctc_weight=0.3,
        label_smoothing=0.1
    )
    
    # Create dummy outputs
    B, T_text, T_enc = 2, 20, 100
    outputs = {
        'logits': torch.randn(B, T_text, 30522),
        'ctc_logits': torch.randn(T_enc, B, 30522),
        'encoder_lengths': torch.tensor([100, 100])
    }
    targets = torch.randint(0, 30522, (B, T_text))
    target_lengths = torch.tensor([20, 18])
    encoder_lengths = torch.tensor([100, 100])
    
    losses = loss_fn(outputs, targets, target_lengths, encoder_lengths)
    
    print(f"✅ Loss keys: {list(losses.keys())}")
    print(f"✅ Total loss: {losses['loss'].item():.4f}")
    print(f"✅ CE loss: {losses['ce_loss'].item():.4f}")
    print(f"✅ CTC loss: {losses['ctc_loss'].item():.4f}")
    
    assert 'loss' in losses, "Missing total loss"
    assert 'ce_loss' in losses, "Missing CE loss"
    assert 'ctc_loss' in losses, "Missing CTC loss"
    assert losses['loss'].requires_grad, "Loss should require grad"
    
except Exception as e:
    print(f"❌ Loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Gradient Flow
print("\n6. Gradient Flow Test")
print("-"*70)
try:
    model = ISLTranslator(
        hidden_dim=256,
        vocab_size=1000,
        decoder_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
        use_ctc=True,
        pretrained='MCG-NJU/videomae-base',
        freeze_epochs=10,
        num_frames=16
    )
    
    # Forward pass
    B = 2
    video_frames = torch.randn(B, 3, 16, 224, 224, requires_grad=True)
    feature_lengths = torch.tensor([16, 16])
    targets = torch.randint(0, 1000, (B, 10))
    target_lengths = torch.tensor([10, 8])
    
    outputs = model(video_frames, feature_lengths, targets, target_lengths)
    
    # Compute loss
    from src.training.losses import HybridLoss
    loss_fn = HybridLoss(vocab_size=1000, pad_id=0, ctc_weight=0.3)
    losses = loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
    
    # Backward pass
    losses['loss'].backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            print(f"✅ {name}: grad norm = {param.grad.norm().item():.6f}")
            break
    
    assert has_grad, "No gradients computed!"
    print(f"✅ Gradient flow working correctly")
    
except Exception as e:
    print(f"❌ Gradient flow test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Data Compatibility
print("\n7. Data Compatibility Test")
print("-"*70)
try:
    from src.data.video_dataset import collate_video_fn
    
    # Simulate batch from dataset
    batch_items = [
        {
            'video_frames': torch.randn(3, 16, 224, 224),  # (C, T, H, W)
            'feature_len': 16,
            'targets': torch.randint(0, 1000, (50,)),
            'target_len': 40,
            'text': 'hello world',
            'video_id': 'video1'
        },
        {
            'video_frames': torch.randn(3, 16, 224, 224),
            'feature_len': 16,
            'targets': torch.randint(0, 1000, (50,)),
            'target_len': 35,
            'text': 'test sentence',
            'video_id': 'video2'
        }
    ]
    
    batch = collate_video_fn(batch_items)
    
    print(f"✅ Batch keys: {list(batch.keys())}")
    print(f"✅ Video frames shape: {batch['video_frames'].shape}")
    print(f"✅ Targets shape: {batch['targets'].shape}")
    print(f"✅ Feature lengths: {batch['feature_lengths']}")
    print(f"✅ Target lengths: {batch['target_lengths']}")
    
    assert batch['video_frames'].shape == (2, 3, 16, 224, 224), "Video batch shape mismatch"
    assert batch['targets'].shape[0] == 2, "Targets batch size mismatch"
    
    # Test model with batch
    model = ISLTranslator(hidden_dim=256, vocab_size=1000, use_ctc=True)
    outputs = model(
        batch['video_frames'],
        batch['feature_lengths'],
        batch['targets'],
        batch['target_lengths']
    )
    
    print(f"✅ Model accepts batch correctly")
    print(f"✅ Output logits shape: {outputs['logits'].shape}")
    
except Exception as e:
    print(f"❌ Data compatibility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ ALL ARCHITECTURE TESTS PASSED!")
print("="*70)
print("\nArchitecture Summary:")
print("  • VideoMAE Encoder: Processes (B, C, T, H, W) video frames")
print("  • Transformer Decoder: Cross-attention to encoder output")
print("  • Translator: Combines encoder + decoder + CTC head")
print("  • Transfer Learning: Freeze encoder → unfreeze at epoch 10")
print("  • Hybrid Loss: 70% CE + 30% CTC")
print("  • Gradient Flow: ✓ Working correctly")
print("  • Data Pipeline: ✓ Compatible")
print("\n🚀 Model architecture is correct and ready for training!")
