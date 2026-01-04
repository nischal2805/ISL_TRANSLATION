"""
Model Architecture Verification - Simple Version
================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch

print("="*70)
print("MODEL ARCHITECTURE VERIFICATION")
print("="*70)

# Test 1: Imports
print("\n1. Component Import Test")
print("-"*70)
try:
    from src.models.encoder import SignEncoder
    from src.models.decoder import TextDecoder
    from src.models.translator import ISLTranslator
    from src.training.losses import HybridLoss
    from src.data.video_dataset import collate_video_fn
    print("[PASS] All components imported")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Full Model
print("\n2. Full Model Test")
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
    
    print(f"[INFO] Model created successfully")
    
    # Test forward pass
    B, T, C, H, W = 2, 16, 3, 224, 224
    T_text = 20
    
    video_frames = torch.randn(B, T, C, H, W)
    feature_lengths = torch.tensor([16, 16])
    targets = torch.randint(0, 30522, (B, T_text))
    target_lengths = torch.tensor([20, 18])
    
    print(f"[INFO] Input video shape: {video_frames.shape}")
    print(f"[INFO] Targets shape: {targets.shape}")
    
    outputs = model(video_frames, feature_lengths, targets, target_lengths)
    
    print(f"[INFO] Output keys: {list(outputs.keys())}")
    print(f"[INFO] Logits shape: {outputs['logits'].shape}")
    print(f"[INFO] CTC logits shape: {outputs['ctc_logits'].shape}")
    
    assert 'logits' in outputs
    assert 'ctc_logits' in outputs
    assert 'encoder_lengths' in outputs
    assert outputs['logits'].shape == (B, T_text, 30522)
    
    print("[PASS] Forward pass works correctly")
    
except Exception as e:
    print(f"[FAIL] Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Transfer Learning
print("\n3. Transfer Learning Test")
print("-"*70)
try:
    assert model.encoder._frozen == True
    print("[INFO] Encoder is frozen initially")
    
    model.on_epoch_start(10)
    assert model.encoder._frozen == False
    print("[INFO] Encoder unfrozen at epoch 10")
    
    print("[PASS] Transfer learning callback works")
    
except Exception as e:
    print(f"[FAIL] Transfer learning test failed: {e}")
    sys.exit(1)

# Test 4: Loss Function
print("\n4. Loss Function Test")
print("-"*70)
try:
    loss_fn = HybridLoss(
        vocab_size=30522,
        pad_id=0,
        ctc_weight=0.3,
        label_smoothing=0.1
    )
    
    losses = loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
    
    print(f"[INFO] Total loss: {losses['loss'].item():.4f}")
    print(f"[INFO] CE loss: {losses['ce_loss'].item():.4f}")
    print(f"[INFO] CTC loss: {losses['ctc_loss'].item():.4f}")
    
    assert 'loss' in losses
    assert losses['loss'].requires_grad
    
    print("[PASS] Loss computation works")
    
except Exception as e:
    print(f"[FAIL] Loss test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Gradient Flow
print("\n5. Gradient Flow Test")
print("-"*70)
try:
    # Backward pass
    losses['loss'].backward()
    
    # Check gradients
    grad_found = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"[INFO] {name}: grad norm = {param.grad.norm().item():.6f}")
            grad_found = True
            break
    
    assert grad_found
    print("[PASS] Gradients computed successfully")
    
except Exception as e:
    print(f"[FAIL] Gradient test failed: {e}")
    sys.exit(1)

# Test 6: Parameter Counts
print("\n6. Parameter Count Test")
print("-"*70)
try:
    param_counts = model.count_parameters()
    print(f"[INFO] Total parameters: {param_counts['total']:,}")
    print(f"[INFO] Trainable parameters: {param_counts['trainable']:,}")
    print(f"[INFO] Frozen parameters: {param_counts['total'] - param_counts['trainable']:,}")
    
    print("[PASS] Parameter counting works")
    
except Exception as e:
    print(f"[FAIL] Parameter count failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("[SUCCESS] ALL TESTS PASSED!")
print("="*70)
print("\nArchitecture Summary:")
print("  - VideoMAE Encoder: (B, T, C, H, W) = (B, 16, 3, 224, 224)")
print("  - Transformer Decoder: 6 layers, 8 heads")
print("  - Hybrid Loss: 70% CE + 30% CTC")
print("  - Transfer Learning: Freeze -> Unfreeze at epoch 10")
print("  - Total params: {:,}".format(param_counts['total']))
print("  - Trainable params: {:,}".format(param_counts['trainable']))
print("\nReady for training!")
