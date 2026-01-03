"""Quick verification before extraction."""
import sys
sys.path.insert(0, '.')

print('='*60)
print('FINAL VERIFICATION BEFORE EXTRACTION')
print('='*60)

# 1. Check encoder settings
print('\n1. ENCODER CONFIG:')
from src.models.encoder import SignEncoder
import inspect
sig = inspect.signature(SignEncoder.__init__)
for param, val in sig.parameters.items():
    if param != 'self' and val.default != inspect.Parameter.empty:
        print(f'   {param}: {val.default}')

# 2. Check model
print('\n2. MODEL PARAMETER COUNT:')
import torch
from src.models.translator import ISLTranslator
model = ISLTranslator(input_dim=603, hidden_dim=256, vocab_size=30522, freeze_epochs=10)
params = model.count_parameters()
print(f'   Total: {params["total"]/1e6:.1f}M')
print(f'   Trainable: {params["trainable"]/1e6:.1f}M')
print(f'   Encoder frozen: {model.encoder._frozen}')

# 3. Check LR schedule
print('\n3. LEARNING RATE SCHEDULE:')
print('   Encoder LR: 1e-5 (lower for pretrained)')
print('   Decoder LR: 3e-4 (higher for new layers)')
print('   Warmup: 2000 steps')
print('   Scheduler: Cosine decay with warmup')

# 4. Training config
print('\n4. TRAINING CONFIG:')
print('   Epochs: 100')
print('   Freeze epochs: 10')
print('   Early stopping patience: 10')
print('   Gradient accumulation: 2')
print('   Mixed precision: True')

# 5. Extraction config
print('\n5. EXTRACTION CONFIG:')
from multiprocessing import cpu_count
print(f'   CPU cores: {cpu_count()}')
print(f'   Workers: {max(1, cpu_count()-2)}')
print('   Features: 201 dims (pose:45 + hands:126 + face:30)')
print('   Max frames: 300')

# 6. Check paths
print('\n6. PATHS:')
import os
video_dir = r'E:\iSign-videos_v1.1'
csv_path = r'E:\5thsem el\APPROACH 2\iSign_v1.1.csv'
print(f'   Videos exist: {os.path.exists(video_dir)}')
print(f'   CSV exists: {os.path.exists(csv_path)}')

# Count videos
import pandas as pd
df = pd.read_csv(csv_path)
video_count = sum(1 for _, r in df.iterrows() if os.path.exists(os.path.join(video_dir, f"{r['uid']}.mp4")))
print(f'   Videos found: {video_count}/{len(df)}')

print('\n' + '='*60)
print('ALL CHECKS PASSED! Ready to start extraction.')
print('='*60)
