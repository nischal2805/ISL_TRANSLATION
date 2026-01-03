"""
Test if using RAW landmarks (less normalization) preserves variance.
"""
import numpy as np
import os
import random

def simple_normalize(data):
    """Simple center+scale normalization that preserves relative differences."""
    # data: (T, 204) - 68 landmarks × 3 coords
    T, D = data.shape
    
    # Center to mean
    centered = data - data.mean(axis=0, keepdims=True)
    
    # Scale by overall std (not per-dimension)
    overall_std = data.std()
    if overall_std > 0:
        scaled = centered / overall_std
    else:
        scaled = centered
    
    return scaled

print('='*70)
print('TESTING: Simple vs Complex Normalization')
print('='*70)

# Load preprocessed samples
train_dir = 'preprocessed_v2/train'
files = [f for f in os.listdir(train_dir) if f.endswith('.npy')]
sample_files = random.sample(files, 100)

# Current (complex) normalization stats
complex_vars = []
for f in sample_files:
    data = np.load(os.path.join(train_dir, f))
    # Extract just position (first 204 dims)
    pos = data[:, :204]
    complex_vars.append(pos.var(axis=0).mean())

complex_vars = np.array(complex_vars)

print(f'Current normalization variance std: {complex_vars.std():.6f}')
print(f'Current normalization variance range: [{complex_vars.min():.6f}, {complex_vars.max():.6f}]')

print('\n' + '='*70)
print('RECOMMENDATION:')
print('='*70)

if complex_vars.std() < 0.002:
    print('❌ FATAL: Current normalization destroys ALL discriminative features')
    print('')
    print('OPTIONS:')
    print('1. Use RAW pixel coordinates (unnormalized)')
    print('2. Use simple center+scale normalization')
    print('3. Add learned positional embeddings to compensate')
    print('4. Use pretrained visual features (e.g., I3D on video frames)')
    print('')
    print('RECOMMENDED: Try option #2 first - simple normalization')
else:
    print('✓ Normalization preserves some variance - architecture issue')
