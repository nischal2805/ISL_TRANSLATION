# ISL Translation System

Indian Sign Language (ISL) to Text Translation using Deep Learning.

## 🎯 Project Overview

This project implements an end-to-end ISL translation system that converts sign language videos to text. The system uses:

- **MediaPipe** for landmark extraction (46 keypoints: pose + hands, NO face)
- **Multi-scale CNN + Conformer** encoder
- **Dual decoder**: CTC head + GRU decoder with cross-attention
- **~18M parameters**, optimized for mobile deployment

## 📁 Project Structure

```
isl_translation/
├── config.py           # Configuration (GPU mode switching)
├── vocab.py            # Vocabulary (35 tokens)
├── preprocessing.py    # MediaPipe extraction & feature engineering
├── dataset.py          # PyTorch Dataset & DataLoader
├── model.py            # Model architecture (~18M params)
├── train.py            # Training with hybrid CTC+CE loss
├── evaluate.py         # Evaluation metrics (CER, WER, BLEU)
├── test.py             # Testing & inference
├── realtime_demo.py    # OpenCV webcam demo
└── utils.py            # Helper functions
```

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Create virtual environment (if not exists)
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe opencv-python numpy pandas scipy tqdm tensorboard
```

### 2. Preprocess Dataset

```powershell
# Set Python path
$env:PYTHONPATH = "e:\5thsem el\APPROACH 2\isl_translation"

# Run preprocessing
python -c "
from preprocessing import preprocess_dataset

preprocess_dataset(
    videos_dir='path/to/isign/videos',
    annotations_file='path/to/annotations.csv',
    output_dir='./preprocessed_data',
    subset_ratio=0.4  # Use 40% for small GPU
)
"
```

### 3. Train Model

```powershell
# Small GPU mode (RTX 4060, 40% data, 20 epochs)
python train.py --data-dir ./preprocessed_data --metadata ./preprocessed_data/metadata.csv --output-dir ./checkpoints --gpu-mode small

# Large GPU mode (A100, 100% data, 60 epochs)
python train.py --data-dir ./preprocessed_data --metadata ./preprocessed_data/metadata.csv --output-dir ./checkpoints --gpu-mode large
```

### 4. Test Model

```powershell
# Test on test set
python test.py --checkpoint ./checkpoints/best_model.pt --mode test --metadata ./preprocessed_data/metadata.csv --data-dir ./preprocessed_data

# Inference on single video
python test.py --checkpoint ./checkpoints/best_model.pt --mode video --video path/to/video.mp4
```

### 5. Real-time Demo

```powershell
# With trained model
python realtime_demo.py --model ./checkpoints/best_model.pt

# Visualization only (no model)
python realtime_demo.py --demo-only
```

## ⚙️ GPU Mode Configuration

The system supports two GPU modes controlled by `set_gpu_mode()`:

| Parameter | Small (RTX 4060) | Large (A100) |
|-----------|------------------|--------------|
| Batch Size | 8 | 32 |
| Epochs | 20 | 60 |
| Dataset % | 40% | 100% |
| Mixed Precision | ✓ | ✓ |
| Gradient Checkpointing | ✓ | ✗ |

```python
from config import set_gpu_mode

# Switch to small GPU mode
set_gpu_mode('small')

# Switch to large GPU mode  
set_gpu_mode('large')
```

## 🏗️ Architecture Details

### Encoder
```
Input (B, T, 414)
    ↓
Input Projection (Linear → LayerNorm → Dropout)
    ↓
Multi-scale CNN × 2 (kernels: 3, 5, 7)
    ↓
Temporal Subsampling (2x reduction)
    ↓
Positional Encoding
    ↓
Conformer × 2 (FFN → MHSA → Conv → FFN)
    ↓
Output (B, T/2, 256)
```

### Decoder
```
Encoder Output (B, T/2, 256)
    ↓
┌─────────────────────────────────────┐
│ CTC Head          GRU Decoder       │
│ (Linear→LogSoftmax)  (Embed→GRU×2→CrossAttn→Linear)
│     ↓                    ↓          │
│ CTC Loss (λ=0.3→0.1)  CE Loss       │
└─────────────────────────────────────┘
    ↓
Hybrid Loss = λ × CTC + (1-λ) × CE
```

### Features (414 dimensions)
- **Landmarks**: 46 keypoints × 3 coords = 138
- **Velocity**: 138 dimensions
- **Acceleration**: 138 dimensions
- **Total**: 414 dimensions

### Landmarks (46 keypoints)
- **Pose**: 4 points (shoulders 11,12 + elbows 13,14)
- **Left Hand**: 21 points (0-20)
- **Right Hand**: 21 points (0-20)
- **NO FACE LANDMARKS**

## 📊 Training Details

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| Dropout | 0.3 |
| Weight Decay | 5e-5 |
| Learning Rate | 5e-4 |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |

### Scheduled Values
| Value | Start → End |
|-------|-------------|
| CTC Weight (λ) | 0.3 → 0.1 |
| Teacher Forcing | 0.9 → 0.2 |

### Loss Function
```
L = λ × L_ctc + (1-λ) × L_ce

where:
- L_ctc = CTC loss on encoder outputs
- L_ce = CrossEntropy with label smoothing on decoder outputs
- λ decreases from 0.3 to 0.1 during training
```

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| Character Accuracy | >85% |
| Character Error Rate | <15% |
| Word Error Rate | <25% |
| Model Size (INT8) | ~18 MB |
| Inference Speed | <100ms/video |

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in config.py
training_config.batch_size = 4

# Or use gradient checkpointing
training_config.gradient_checkpointing = True
```

### MediaPipe Detection Issues
- Ensure good lighting
- Face the camera directly
- Keep hands visible in frame
- Check `min_detection_confidence` in config

### Training Not Converging
- Check data preprocessing (normalized landmarks?)
- Reduce learning rate
- Increase warmup steps
- Check for NaN in loss

## 📋 Data Format

### Annotations CSV
```csv
video_id,text,signer_id
video_001,hello world,signer_01
video_002,thank you,signer_01
...
```

### Preprocessed Features
```
preprocessed_data/
├── train/
│   ├── video_001.npy  # (T, 414) float32
│   ├── video_002.npy
│   └── ...
├── val/
├── test/
└── metadata.csv
```

## 🚀 Mobile Deployment

### Export to ONNX
```python
from utils import export_to_onnx
from model import create_model

model = create_model()
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])
export_to_onnx(model, 'model.onnx')
```

### Quantization
```python
from utils import quantize_model_int8

quantized_model = quantize_model_int8(model)
# Save for mobile deployment
torch.save(quantized_model.state_dict(), 'model_int8.pt')
```

## 📚 References

- ISign Dataset (127K videos)
- MediaPipe Holistic
- Conformer: Convolution-augmented Transformer for Speech Recognition
- CTC: Connectionist Temporal Classification

## 📝 License

MIT License
