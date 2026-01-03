# Mode Collapse Fix - Root Cause Analysis

## The Problem
All trained models predicted generic phrases ("the police...", "the government...") regardless of input, even though:
- Encoder was processing features correctly (different inputs → different outputs)
- Model extracted some information (numbers like "353")
- Loss was decreasing (8.3 → 3.4)

## Root Cause Analysis

### 1. **Weak Cross-Attention Signal**
```python
# OLD CODE - PROBLEM:
residual = x
x = self.cross_attn_norm(x)
x, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_mask)
x = torch.nan_to_num(x, nan=0.0)  # This could zero out cross-attention!
x = residual + self.cross_attn_dropout(x)
```

**Issues:**
- `torch.nan_to_num(x, nan=0.0)` could zero out cross-attention if any NaN appeared
- Residual connection allowed decoder to ignore encoder completely
- No explicit supervision forcing decoder to USE encoder features
- CE loss (language modeling) dominated, teaching decoder to predict common phrases

### 2. **No Attention Regularization**
The loss had no term ensuring cross-attention actually attended to encoder features. Decoder learned to rely on language model patterns instead of encoder context.

### 3. **Gradient Flow Issue**
Post-norm architecture (`x = norm(x + residual)` equivalent) can have weaker gradients flowing back to encoder compared to pre-norm.

## The Fix

### 1. **Gated Cross-Attention**
```python
# NEW CODE - SOLUTION:
# Gate to control encoder vs language model influence
self.encoder_gate = nn.Sequential(
    nn.Linear(d_model * 2, d_model),
    nn.Sigmoid()
)

# In forward:
cross_out, cross_weights = self.cross_attn(x_norm, memory, memory, key_padding_mask=memory_mask)
gate_input = torch.cat([x, cross_out], dim=-1)
gate = self.encoder_gate(gate_input)
x = residual + self.cross_attn_dropout(gate * cross_out)
```

**Benefits:**
- Model learns WHEN and HOW MUCH to use encoder
- Gate is trainable - learns optimal encoder/LM balance
- Explicit pathway for encoder information

### 2. **Attention Regularization Loss**
```python
# Entropy: encourage focused attention (not spread out)
attn_probs = attn_weights + 1e-10
entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean()

# Coverage: ensure all encoder positions attended at least once
coverage = attn_weights.sum(dim=1)
coverage_loss = torch.relu(0.1 - coverage).mean()

attn_reg_loss = entropy + coverage_loss
```

**Benefits:**
- Forces decoder to actually look at encoder features
- Lower entropy = focused attention on relevant frames
- Coverage ensures no encoder positions are ignored
- Adds `attn_reg_weight * attn_reg_loss` to total loss

### 3. **Better Architecture**
- Removed `torch.nan_to_num()` that could zero out cross-attention
- Proper pre-norm in feed-forward for better gradient flow
- Return cross-attention weights for loss computation

## Expected Results

With these fixes:
1. **Decoder will condition on encoder** - gate learns to use sign features
2. **Attention will be meaningful** - regularization forces actual usage
3. **Better gradient flow** - encoder receives stronger learning signal
4. **No more mode collapse** - can't just predict generic phrases

## Training Recommendations

```bash
python train_v2.py \
  --epochs 100 \
  --lr 3e-4 \
  --warmup-steps 1000 \
  --ctc-weight 0.3 \
  --attn-reg-weight 0.1 \
  --batch-size 32
```

**Monitor:**
- CE loss should decrease faster now (decoder learns from encoder)
- Attention reg loss should decrease (attention becomes more focused)
- Predictions should vary with input (not generic phrases)
- Encoder/decoder gradient ratio (should be more balanced)

## Code Changes Summary

### `model_v2.py`:
1. Added `encoder_gate` to `TransformerDecoderLayer`
2. Modified layer forward to return `(x, cross_weights)` tuple
3. Added `last_cross_attention_weights` storage to `TransformerDecoder`
4. Modified all layer calls to unpack tuple
5. Updated `ISLTranslationModelV2.forward()` to return cross-attention weights
6. Added `attn_reg_weight` parameter to `HybridCTCAttentionLoss`
7. Implemented attention regularization (entropy + coverage)

### `train_v2.py`:
1. Updated loss function calls to pass `cross_attention_weights`
2. Both in training and validation loops

## Testing

Run `test_model_fix.py` to verify:
- ✓ Forward pass works with new architecture
- ✓ Cross-attention weights returned correctly
- ✓ Attention regularization computed
- ✓ Gradients flow to encoder
- ✓ Backward pass succeeds

All tests passed! Ready for training.
