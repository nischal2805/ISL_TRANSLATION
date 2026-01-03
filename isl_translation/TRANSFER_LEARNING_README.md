# ISL Translation with Transfer Learning

## Overview

This module implements Indian Sign Language (ISL) to English translation using transfer learning from pretrained video understanding models. The system is designed for deployment on mobile devices (specifically Snapdragon 8 Gen 3).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ISL Translation Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Video Frames] → [MediaPipe Landmarks] → [Features 540D]   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                       │
│  │  Input Project  │  540D → 384D                          │
│  └────────┬────────┘                                       │
│           ▼                                                │
│  ┌─────────────────┐                                       │
│  │ VideoMAE Encoder│  Pretrained, frozen initially         │
│  │   (Optional)    │  94.2M params                         │
│  └────────┬────────┘                                       │
│           ▼                                                │
│  ┌─────────────────┐                                       │
│  │    Conformer    │  6 layers, 384D                       │
│  │    Encoder      │  Multi-head self-attention            │
│  └────────┬────────┘                                       │
│           │                                                │
│     ┌─────┴─────┐                                         │
│     ▼           ▼                                         │
│  ┌──────┐  ┌─────────────┐                                │
│  │ CTC  │  │ Transformer │                                 │
│  │ Head │  │  Decoder    │  4 layers, gated attention      │
│  └──────┘  └──────┬──────┘                                │
│     │             │                                       │
│     └──────┬──────┘                                       │
│           ▼                                               │
│    [English Text]                                         │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `model_transfer_learning.py` | Main model with VideoMAE encoder + Conformer |
| `train_transfer_learning.py` | Training script with staged unfreezing |
| `mobile_export.py` | Quantization and ONNX export for mobile |
| `dataset_consolidated.py` | Dataset loader for consolidated .npy files |
| `extract_consolidated.py` | Converts preprocessed data to consolidated format |

## Quick Start

### 1. Prepare Data

```bash
# Convert individual .npy files to consolidated format
python extract_consolidated.py \
    --input-dir preprocessed_v2 \
    --output-dir consolidated_data \
    --metadata-csv preprocessed_v2/metadata.csv
```

### 2. Train with Transfer Learning

```bash
python train_transfer_learning.py \
    --data-dir consolidated_data \
    --encoder MCG-NJU/videomae-base \
    --freeze-epochs 5 \
    --epochs 50 \
    --batch-size 16 \
    --encoder-lr 1e-5 \
    --decoder-lr 3e-4
```

### 3. Export for Mobile

```bash
python mobile_export.py \
    --checkpoint checkpoints_transfer/best_model.pt \
    --output-dir mobile_models \
    --quantize \
    --export-onnx \
    --benchmark
```

## Training Strategy

### Staged Unfreezing

1. **Epochs 0-5**: Encoder frozen, only decoder trains
   - Allows decoder to adapt to encoder representations
   - Higher learning rate for decoder (3e-4)

2. **Epochs 5+**: Encoder unfrozen with lower LR
   - Fine-tune encoder for ISL-specific patterns
   - Lower learning rate for encoder (1e-5)

### Loss Function

Hybrid CTC-Attention loss:
```
L = α * L_ctc + (1-α) * L_ce
```
- `L_ctc`: Connectionist Temporal Classification for alignment
- `L_ce`: Cross-entropy for accurate generation
- `α = 0.3` by default

## Model Variants

### Full Model (~35M params)
- VideoMAE encoder + Conformer + Transformer decoder
- Best accuracy, requires GPU for training
- Use for research and server deployment

### Mobile Model (~10M params)
- Lightweight Conformer encoder
- 4 layers, 256D hidden
- Optimized for Snapdragon 8 Gen 3

### Quantized Model
- INT8 dynamic quantization
- ~4x smaller than full model
- Minimal accuracy loss (<1%)

## Performance Expectations

Based on state-of-the-art sign language translation research:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| BLEU | 15-25 | Sign language SOTA ~25 |
| Word Accuracy | 30-50% | Varies by vocabulary |
| Inference (mobile) | 50-150ms | Per sign sequence |

**Why not 90% accuracy?**
- Sign language translation is extremely difficult
- ISL has limited training data compared to ASL/BSL
- Continuous signing has no clear word boundaries
- Same sign can have different meanings in context

## Mobile Deployment

### For Snapdragon 8 Gen 3

```python
# Load ONNX model in Flutter
import onnxruntime as ort

session = ort.InferenceSession("model_optimized.onnx")
predictions = session.run(None, {
    'features': features,
    'lengths': lengths
})
```

### Expected Performance

| Device | Inference Time | Power |
|--------|---------------|-------|
| SD 8 Gen 3 (GPU) | ~50ms | ~2W |
| SD 8 Gen 3 (CPU) | ~150ms | ~1W |
| SD 8 Gen 3 (NPU) | ~30ms | ~0.5W |

## Recommendations

1. **Start with mobile model** - Easier to train, faster iteration
2. **Use knowledge distillation** - Train mobile model from full model
3. **Focus on top-50 signs first** - Build a focused vocabulary
4. **Collect more ISL data** - More data > better architecture

## Known Limitations

1. Dataset size (~127K samples) is small for sign language
2. ISL has regional variations not captured
3. No fingerspelling support currently
4. Single signer may not generalize

## References

- VideoMAE: Masked Autoencoders for Video Understanding
- ISLTranslate: Dataset for Sign Language Translation
- Progressive Transformers for End-to-End Sign Language Production
