"""
Comprehensive Tests for ISL Translation Pipeline
================================================
Verifies: extraction, model architecture, encoder-decoder connection, gradients
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def test_extraction_config():
    """Test extraction configuration."""
    print("\n" + "="*50)
    print("TEST 1: Extraction Configuration")
    print("="*50)
    
    from multiprocessing import cpu_count
    num_workers = max(1, cpu_count() - 2)
    
    print(f"CPU cores: {cpu_count()}")
    print(f"Workers for extraction: {num_workers}")
    
    # Check landmark dimensions
    POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    FACE_INDICES = [61, 291, 0, 17, 70, 300, 33, 263, 1, 4]
    
    pose_dim = len(POSE_INDICES) * 3
    hands_dim = 21 * 3 * 2
    face_dim = len(FACE_INDICES) * 3
    raw_dim = pose_dim + hands_dim + face_dim
    
    print(f"Pose: {pose_dim}, Hands: {hands_dim}, Face: {face_dim}")
    print(f"Raw features: {raw_dim}")
    print(f"With velocity+acceleration: {raw_dim * 3}")
    
    assert raw_dim == 201, f"Expected 201, got {raw_dim}"
    assert raw_dim * 3 == 603, f"Expected 603, got {raw_dim * 3}"
    
    print("✅ Extraction config OK")
    return True


def test_model_architecture():
    """Test model architecture and shapes."""
    print("\n" + "="*50)
    print("TEST 2: Model Architecture")
    print("="*50)
    
    from src.models.translator import ISLTranslator
    
    # Create model without pretrained weights for faster testing
    model = ISLTranslator(
        input_dim=603,
        hidden_dim=256,
        encoder_hidden=768,
        vocab_size=8000,
        decoder_layers=4,
        pretrained="MCG-NJU/videomae-base"
    )
    
    params = model.count_parameters()
    print(f"Total params: {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")
    print(f"Encoder frozen: {model.encoder._frozen}")
    
    # Test forward pass
    B, T, L = 2, 50, 20
    features = torch.randn(B, T, 603)
    feature_lengths = torch.tensor([50, 40])
    targets = torch.randint(0, 8000, (B, L))
    target_lengths = torch.tensor([20, 15])
    
    outputs = model(features, feature_lengths, targets, target_lengths)
    
    print(f"Encoder output shape: {outputs['encoder_out'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"CTC logits shape: {outputs['ctc_logits'].shape}")
    
    assert outputs['logits'].shape == (B, L, 8000)
    assert outputs['encoder_out'].shape[0] == B
    
    print("✅ Model architecture OK")
    return True


def test_encoder_decoder_connection():
    """Test that encoder actually affects decoder output."""
    print("\n" + "="*50)
    print("TEST 3: Encoder-Decoder Connection")
    print("="*50)
    
    from src.models.translator import ISLTranslator
    
    model = ISLTranslator(input_dim=603, vocab_size=8000)
    model.eval()
    
    B, T, L = 2, 50, 20
    targets = torch.randint(0, 8000, (B, L))
    target_lengths = torch.tensor([20, 15])
    feature_lengths = torch.tensor([50, 50])
    
    # Test with different inputs
    features1 = torch.randn(B, T, 603)
    features2 = torch.randn(B, T, 603) * 2 + 1  # Different input
    
    with torch.no_grad():
        out1 = model(features1, feature_lengths, targets, target_lengths)
        out2 = model(features2, feature_lengths, targets, target_lengths)
    
    # Outputs should be different if encoder is connected
    diff = (out1['logits'] - out2['logits']).abs().mean().item()
    print(f"Output difference with different inputs: {diff:.4f}")
    
    assert diff > 0.01, "Encoder not affecting decoder!"
    
    print("✅ Encoder-decoder connection OK")
    return True


def test_gradient_flow():
    """Test gradients flow from decoder loss to encoder."""
    print("\n" + "="*50)
    print("TEST 4: Gradient Flow")
    print("="*50)
    
    from src.models.translator import ISLTranslator
    from src.training.losses import HybridLoss
    
    model = ISLTranslator(input_dim=603, vocab_size=8000)
    model.encoder.unfreeze()  # Unfreeze for gradient test
    
    loss_fn = HybridLoss(vocab_size=8000)
    
    B, T, L = 2, 50, 20
    features = torch.randn(B, T, 603, requires_grad=True)
    feature_lengths = torch.tensor([50, 40])
    targets = torch.randint(1, 8000, (B, L))
    target_lengths = torch.tensor([20, 15])
    
    # Forward
    outputs = model(features, feature_lengths, targets, target_lengths)
    losses = loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
    
    # Backward
    losses['loss'].backward()
    
    # Check gradients
    encoder_grads = []
    for name, param in model.encoder.named_parameters():
        if param.grad is not None:
            encoder_grads.append(param.grad.abs().mean().item())
    
    decoder_grads = []
    for name, param in model.decoder.named_parameters():
        if param.grad is not None:
            decoder_grads.append(param.grad.abs().mean().item())
    
    avg_encoder_grad = np.mean(encoder_grads) if encoder_grads else 0
    avg_decoder_grad = np.mean(decoder_grads) if decoder_grads else 0
    
    print(f"Avg encoder gradient: {avg_encoder_grad:.6f}")
    print(f"Avg decoder gradient: {avg_decoder_grad:.6f}")
    print(f"Loss: {losses['loss'].item():.4f}")
    
    assert avg_encoder_grad > 0, "No gradients in encoder!"
    assert avg_decoder_grad > 0, "No gradients in decoder!"
    
    print("✅ Gradient flow OK")
    return True


def test_loss_function():
    """Test hybrid CTC-Attention loss."""
    print("\n" + "="*50)
    print("TEST 5: Loss Function")
    print("="*50)
    
    from src.training.losses import HybridLoss
    
    loss_fn = HybridLoss(vocab_size=8000, ctc_weight=0.3)
    
    B, T_enc, T_dec, V = 2, 50, 20, 8000
    
    outputs = {
        'logits': torch.randn(B, T_dec, V),
        'ctc_logits': torch.randn(B, T_enc, V),
        'encoder_lengths': torch.tensor([50, 40])
    }
    targets = torch.randint(1, V, (B, T_dec))
    target_lengths = torch.tensor([20, 15])
    
    losses = loss_fn(outputs, targets, target_lengths, outputs['encoder_lengths'])
    
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"CE loss: {losses['ce_loss'].item():.4f}")
    print(f"CTC loss: {losses['ctc_loss'].item():.4f}")
    
    assert not torch.isnan(losses['loss']), "NaN loss!"
    assert losses['loss'].item() > 0, "Zero loss!"
    
    print("✅ Loss function OK")
    return True


def test_inference():
    """Test model inference (translation)."""
    print("\n" + "="*50)
    print("TEST 6: Inference")
    print("="*50)
    
    from src.models.translator import ISLTranslator
    
    model = ISLTranslator(input_dim=603, vocab_size=8000)
    model.eval()
    
    features = torch.randn(1, 50, 603)
    feature_lengths = torch.tensor([50])
    
    with torch.no_grad():
        output_ids = model.translate(features, feature_lengths, max_len=30)
    
    print(f"Generated shape: {output_ids.shape}")
    print(f"Generated IDs (first 10): {output_ids[0, :10].tolist()}")
    
    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] <= 30
    
    print("✅ Inference OK")
    return True


def test_encoder_unfreezing():
    """Test encoder unfreezing logic."""
    print("\n" + "="*50)
    print("TEST 7: Encoder Unfreezing")
    print("="*50)
    
    from src.models.translator import ISLTranslator
    
    model = ISLTranslator(input_dim=603, vocab_size=8000, freeze_epochs=3)
    
    print(f"Initial: frozen={model.encoder._frozen}")
    assert model.encoder._frozen == True
    
    # Simulate epochs
    for epoch in range(1, 6):
        model.on_epoch_start(epoch)
        frozen = model.encoder._frozen
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Epoch {epoch}: frozen={frozen}, trainable={trainable:,}")
    
    assert model.encoder._frozen == False
    
    print("✅ Encoder unfreezing OK")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("ISL TRANSLATION PIPELINE VERIFICATION")
    print("="*60)
    
    tests = [
        test_extraction_config,
        test_model_architecture,
        test_encoder_decoder_connection,
        test_gradient_flow,
        test_loss_function,
        test_inference,
        test_encoder_unfreezing,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            results.append((test.__name__, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️ SOME TESTS FAILED!")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
