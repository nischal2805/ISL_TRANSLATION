"""
NaN Safety Test Script
======================
Verifies that the model and loss functions handle edge cases without NaN/Inf.
Run this before training to ensure stability.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, model_config, set_gpu_mode
from vocab import Vocabulary
from model import create_model
from train import NaNSafeHybridLoss, NaNSafeLabelSmoothingLoss


def test_ctc_head_clamping():
    """Test that CTCHead clamps log probabilities correctly."""
    print("\n" + "="*60)
    print("TEST 1: CTCHead NaN Safety")
    print("="*60)
    
    vocab = Vocabulary()
    model = create_model().to(DEVICE)  # No argument needed, uses model_config
    
    # Test with normal input
    batch_size = 4
    seq_len = 100
    features = torch.randn(batch_size, seq_len, 414).to(DEVICE)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long).to(DEVICE)
    targets = torch.randint(3, 30, (batch_size, 15)).to(DEVICE)
    targets[:, 0] = 1  # SOS
    targets[:, -1] = 2  # EOS
    
    with torch.no_grad():
        outputs = model(features, lengths, targets)
        ctc_logits = outputs['ctc_logits']
    
    # Check for NaN/Inf
    has_nan = torch.isnan(ctc_logits).any().item()
    has_inf = torch.isinf(ctc_logits).any().item()
    min_val = ctc_logits.min().item()
    max_val = ctc_logits.max().item()
    
    print(f"  CTC logits shape: {ctc_logits.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Min value: {min_val:.4f}")
    print(f"  Max value: {max_val:.4f}")
    print(f"  Values clamped >= -100: {min_val >= -100}")
    
    if not has_nan and not has_inf and min_val >= -100:
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_ctc_length_validation():
    """Test CTC loss with length constraint violations."""
    print("\n" + "="*60)
    print("TEST 2: CTC Length Validation")
    print("="*60)
    
    vocab = Vocabulary()
    loss_fn = NaNSafeHybridLoss(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        blank_id=vocab.pad_id
    )
    
    # Simulate scenario where encoder_length < target_length
    batch_size = 4
    
    # CTC logits: (B, T=10, V)
    ctc_logits = torch.randn(batch_size, 10, vocab.size).to(DEVICE)
    ctc_logits = F.log_softmax(ctc_logits, dim=-1).clamp(min=-100)
    
    # Decoder logits: (B, L=20, V)
    decoder_logits = torch.randn(batch_size, 20, vocab.size).to(DEVICE)
    
    # Targets with length 20 (after removing SOS/EOS, target_length=18)
    targets = torch.randint(3, 30, (batch_size, 20)).to(DEVICE)
    targets[:, 0] = 1  # SOS
    targets[:, -1] = 2  # EOS
    
    # Encoder lengths: some smaller than target lengths (should violate constraint)
    encoder_lengths = torch.tensor([5, 10, 8, 12], dtype=torch.long).to(DEVICE)  # Only [10, 12] might be valid
    target_lengths = torch.tensor([20, 20, 20, 20], dtype=torch.long).to(DEVICE)
    
    loss, loss_dict = loss_fn(
        ctc_logits=ctc_logits,
        decoder_logits=decoder_logits,
        targets=targets,
        encoder_lengths=encoder_lengths,
        target_lengths=target_lengths,
        ctc_weight=0.3
    )
    
    print(f"  Encoder lengths: {encoder_lengths.tolist()}")
    print(f"  Target lengths (raw): {target_lengths.tolist()}")
    print(f"  Target lengths (for CTC): {(target_lengths - 2).clamp(min=1).tolist()}")
    print(f"  CTC valid ratio: {loss_dict['ctc_valid_ratio']:.2%}")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  CTC loss: {loss_dict['ctc_loss']:.4f}")
    print(f"  CE loss: {loss_dict['ce_loss']:.4f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    # Check that loss is finite despite constraint violations
    if torch.isfinite(loss) and loss_dict['ctc_valid_ratio'] < 1.0:
        print("  ✅ PASSED - Handled constraint violations gracefully")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_extreme_logits():
    """Test with extreme logit values that could cause overflow."""
    print("\n" + "="*60)
    print("TEST 3: Extreme Logit Values")
    print("="*60)
    
    vocab = Vocabulary()
    loss_fn = NaNSafeLabelSmoothingLoss(smoothing=0.1, ignore_index=0)
    
    batch_size = 4
    seq_len = 20
    
    # Create extreme logits
    logits = torch.randn(batch_size, seq_len, vocab.size).to(DEVICE)
    logits = logits * 1000  # Scale up dramatically
    
    targets = torch.randint(1, vocab.size, (batch_size, seq_len)).to(DEVICE)
    
    print(f"  Logit range: [{logits.min().item():.1f}, {logits.max().item():.1f}]")
    
    loss = loss_fn(logits, targets)
    
    has_nan = torch.isnan(loss).item()
    has_inf = torch.isinf(loss).item()
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    
    if not has_nan and not has_inf:
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_zero_length_sequences():
    """Test handling of zero-length or very short sequences."""
    print("\n" + "="*60)
    print("TEST 4: Zero/Short Length Sequences")
    print("="*60)
    
    vocab = Vocabulary()
    loss_fn = NaNSafeHybridLoss(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        blank_id=vocab.pad_id
    )
    
    batch_size = 4
    
    # CTC logits
    ctc_logits = torch.randn(batch_size, 50, vocab.size).to(DEVICE)
    ctc_logits = F.log_softmax(ctc_logits, dim=-1).clamp(min=-100)
    
    # Decoder logits with very short sequence
    decoder_logits = torch.randn(batch_size, 5, vocab.size).to(DEVICE)
    
    # Targets
    targets = torch.zeros(batch_size, 5, dtype=torch.long).to(DEVICE)
    targets[:, 0] = 1  # SOS
    targets[:, 1] = 3  # One character
    targets[:, 2] = 2  # EOS
    
    encoder_lengths = torch.tensor([50, 50, 50, 50], dtype=torch.long).to(DEVICE)
    target_lengths = torch.tensor([3, 3, 3, 3], dtype=torch.long).to(DEVICE)  # Very short
    
    loss, loss_dict = loss_fn(
        ctc_logits=ctc_logits,
        decoder_logits=decoder_logits,
        targets=targets,
        encoder_lengths=encoder_lengths,
        target_lengths=target_lengths,
        ctc_weight=0.3
    )
    
    print(f"  Target lengths: {target_lengths.tolist()}")
    print(f"  CTC target lengths (after adjustment): {(target_lengths - 2).clamp(min=1).tolist()}")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    
    if torch.isfinite(loss):
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_all_masked_attention():
    """Test attention with all positions masked (edge case)."""
    print("\n" + "="*60)
    print("TEST 5: Masked Attention Edge Case")
    print("="*60)
    
    vocab = Vocabulary()
    model = create_model().to(DEVICE)
    
    batch_size = 2
    seq_len = 50
    
    # Features with varying lengths
    features = torch.randn(batch_size, seq_len, 414).to(DEVICE)
    
    # Very short lengths (most of sequence will be masked)
    lengths = torch.tensor([5, 10], dtype=torch.long).to(DEVICE)
    
    targets = torch.randint(3, 30, (batch_size, 10)).to(DEVICE)
    targets[:, 0] = 1  # SOS
    targets[:, -1] = 2  # EOS
    
    with torch.no_grad():
        outputs = model(features, lengths, targets)
    
    ctc_logits = outputs['ctc_logits']
    decoder_logits = outputs['decoder_logits']
    
    ctc_nan = torch.isnan(ctc_logits).any().item()
    decoder_nan = torch.isnan(decoder_logits).any().item()
    
    print(f"  Input lengths: {lengths.tolist()}")
    print(f"  CTC logits has NaN: {ctc_nan}")
    print(f"  Decoder logits has NaN: {decoder_nan}")
    
    if not ctc_nan and not decoder_nan:
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_full_forward_backward():
    """Test complete forward and backward pass for NaN/Inf."""
    print("\n" + "="*60)
    print("TEST 6: Full Forward-Backward Pass")
    print("="*60)
    
    vocab = Vocabulary()
    model = create_model().to(DEVICE)
    loss_fn = NaNSafeHybridLoss(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        blank_id=vocab.pad_id
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    batch_size = 8
    seq_len = 100
    target_len = 15
    
    # Create batch
    features = torch.randn(batch_size, seq_len, 414).to(DEVICE)
    lengths = torch.randint(50, seq_len, (batch_size,)).to(DEVICE)
    targets = torch.randint(3, 30, (batch_size, target_len)).to(DEVICE)
    targets[:, 0] = 1  # SOS
    targets[:, -1] = 2  # EOS
    target_lengths = torch.full((batch_size,), target_len, dtype=torch.long).to(DEVICE)
    
    # Forward
    optimizer.zero_grad()
    outputs = model(features, lengths, targets)
    
    # Loss
    loss, loss_dict = loss_fn(
        ctc_logits=outputs['ctc_logits'],
        decoder_logits=outputs['decoder_logits'],
        targets=targets,
        encoder_lengths=outputs['encoder_lengths'],
        target_lengths=target_lengths,
        ctc_weight=0.3
    )
    
    print(f"  Forward pass loss: {loss.item():.4f}")
    print(f"  CTC loss: {loss_dict['ctc_loss']:.4f}")
    print(f"  CE loss: {loss_dict['ce_loss']:.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    grad_norms = []
    has_nan_grad = False
    has_inf_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms.append(norm)
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"  NaN gradient in: {name}")
            if torch.isinf(param.grad).any():
                has_inf_grad = True
                print(f"  Inf gradient in: {name}")
    
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    max_grad_norm = max(grad_norms) if grad_norms else 0
    
    print(f"  Average gradient norm: {avg_grad_norm:.4f}")
    print(f"  Max gradient norm: {max_grad_norm:.4f}")
    print(f"  Has NaN gradients: {has_nan_grad}")
    print(f"  Has Inf gradients: {has_inf_grad}")
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if not has_nan_grad and not has_inf_grad and torch.isfinite(loss):
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def test_multiple_iterations():
    """Test multiple training iterations to catch accumulating issues."""
    print("\n" + "="*60)
    print("TEST 7: Multiple Training Iterations")
    print("="*60)
    
    vocab = Vocabulary()
    model = create_model().to(DEVICE)
    loss_fn = NaNSafeHybridLoss(
        vocab_size=vocab.size,
        pad_id=vocab.pad_id,
        blank_id=vocab.pad_id
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_iterations = 20
    nan_count = 0
    inf_count = 0
    losses = []
    
    for i in range(num_iterations):
        batch_size = 8
        seq_len = torch.randint(50, 150, (1,)).item()
        target_len = torch.randint(5, 20, (1,)).item()
        
        features = torch.randn(batch_size, seq_len, 414).to(DEVICE)
        lengths = torch.randint(30, seq_len, (batch_size,)).to(DEVICE)
        targets = torch.randint(3, 30, (batch_size, target_len)).to(DEVICE)
        targets[:, 0] = 1
        targets[:, -1] = 2
        target_lengths = torch.full((batch_size,), target_len, dtype=torch.long).to(DEVICE)
        
        optimizer.zero_grad()
        
        try:
            outputs = model(features, lengths, targets)
            loss, loss_dict = loss_fn(
                ctc_logits=outputs['ctc_logits'],
                decoder_logits=outputs['decoder_logits'],
                targets=targets,
                encoder_lengths=outputs['encoder_lengths'],
                target_lengths=target_lengths,
                ctc_weight=0.3
            )
            
            if torch.isnan(loss):
                nan_count += 1
                continue
            if torch.isinf(loss):
                inf_count += 1
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        except RuntimeError as e:
            print(f"  Iteration {i}: Error - {e}")
            nan_count += 1
    
    print(f"  Total iterations: {num_iterations}")
    print(f"  Successful: {len(losses)}")
    print(f"  NaN losses: {nan_count}")
    print(f"  Inf losses: {inf_count}")
    
    if len(losses) > 0:
        print(f"  Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
        print(f"  Loss trend: {'decreasing' if losses[-1] < losses[0] else 'increasing/stable'}")
    
    success_rate = len(losses) / num_iterations
    print(f"  Success rate: {success_rate:.2%}")
    
    if success_rate > 0.9:  # Allow up to 10% failures
        print("  ✅ PASSED")
        return True
    else:
        print("  ❌ FAILED")
        return False


def main():
    print("\n" + "="*60)
    print("NaN SAFETY TEST SUITE")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Set GPU mode
    set_gpu_mode('small')
    
    results = []
    
    # Run all tests
    results.append(("CTCHead Clamping", test_ctc_head_clamping()))
    results.append(("CTC Length Validation", test_ctc_length_validation()))
    results.append(("Extreme Logits", test_extreme_logits()))
    results.append(("Zero Length Sequences", test_zero_length_sequences()))
    results.append(("Masked Attention", test_all_masked_attention()))
    results.append(("Forward-Backward", test_full_forward_backward()))
    results.append(("Multiple Iterations", test_multiple_iterations()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All NaN safety tests passed! Ready for training.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Review the issues before training.")
        return 1


if __name__ == '__main__':
    exit(main())
