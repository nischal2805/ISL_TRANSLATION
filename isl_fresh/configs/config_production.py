"""
Configuration for ISL Translation - A100 GPU Setup
==================================================
Update these paths for your A100 GPU environment.
"""

from pathlib import Path

# ============================================================================
# GPU SERVER PATHS (A100 Server - CONFIGURED)
# ============================================================================

# Video data location on GPU server
GPU_VIDEO_DIR = "/media/rvcse22/CSERV/kortex_sem5/videos/iSign-videos_v1.1"

# CSV metadata location on GPU server
GPU_CSV_PATH = "/media/rvcse22/CSERV/kortex_sem5/nischal/training_by_surya/ISL_TRANSLATION/iSign_v1.1.csv"

# Output directories on GPU server
GPU_OUTPUT_DIR = "/media/rvcse22/CSERV/kortex_sem5/nischal/training_by_surya/isl_fresh"
GPU_CHECKPOINT_DIR = f"{GPU_OUTPUT_DIR}/checkpoints_video"
GPU_LOG_DIR = f"{GPU_OUTPUT_DIR}/logs_video"

# ============================================================================
# LOCAL PATHS (Windows - for development/testing)
# ============================================================================

LOCAL_VIDEO_DIR = r"E:\iSign-videos_v1.1"
LOCAL_CSV_PATH = r"E:\5thsem el\APPROACH 2\iSign_v1.1.csv"
LOCAL_OUTPUT_DIR = r"E:\5thsem el\APPROACH 2\isl_fresh"
LOCAL_CHECKPOINT_DIR = f"{LOCAL_OUTPUT_DIR}\\checkpoints_video"
LOCAL_LOG_DIR = f"{LOCAL_OUTPUT_DIR}\\logs_video"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # VideoMAE Encoder
    'pretrained': 'MCG-NJU/videomae-base',  # 86M params
    'num_frames': 16,  # Standard VideoMAE input
    'freeze_epochs': 10,  # Freeze encoder for first 10 epochs
    
    # Decoder
    'hidden_dim': 256,
    'decoder_layers': 4,
    'num_heads': 4,
    'ff_dim': 1024,
    'dropout': 0.1,
    
    # Vocabulary
    'tokenizer': 'bert-base-uncased',  # ~30K vocab
    
    # Loss
    'ctc_weight': 0.3,  # 30% CTC, 70% cross-entropy
    'label_smoothing': 0.1,
}

# ============================================================================
# TRAINING CONFIGURATION (Optimized for A100 40GB)
# ============================================================================

TRAINING_CONFIG = {
    # Batch size
    'batch_size': 16,  # A100 can handle this with video frames
    'gradient_accumulation': 2,  # Effective batch = 32
    
    # Learning rates
    'encoder_lr': 1e-5,  # Lower for pretrained VideoMAE
    'decoder_lr': 3e-4,  # Higher for randomly initialized decoder
    
    # Optimization
    'warmup_steps': 2000,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Training duration
    'num_epochs': 100,
    'patience': 15,  # Early stopping
    
    # System
    'num_workers': 8,  # A100 server usually has good CPU
    'use_amp': True,  # Mixed precision for A100
    'pin_memory': True,
}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================

VIDEO_CONFIG = {
    'num_frames': 16,  # Frames per video clip
    'image_size': (224, 224),  # VideoMAE input size
    'fps': None,  # Use original FPS, sample uniformly
}

# ============================================================================
# MOBILE DEPLOYMENT (Post-training quantization)
# ============================================================================

QUANTIZATION_CONFIG = {
    'enabled': True,  # Enable for mobile deployment
    'method': 'dynamic',  # 'dynamic' or 'static'
    'dtype': 'qint8',  # 8-bit quantization
    'target_device': 'mobile',  # Optimize for mobile
    
    # Expected size reduction
    # Original: ~100M params × 4 bytes = 400MB
    # Quantized: ~100M params × 1 byte = 100MB (75% reduction)
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_paths(mode='local'):
    """
    Get paths based on execution environment.
    
    Args:
        mode: 'local' or 'gpu'
    
    Returns:
        dict with video_dir, csv_path, checkpoint_dir, log_dir
    """
    if mode == 'gpu':
        return {
            'video_dir': GPU_VIDEO_DIR,
            'csv_path': GPU_CSV_PATH,
            'checkpoint_dir': GPU_CHECKPOINT_DIR,
            'log_dir': GPU_LOG_DIR,
        }
    else:  # local
        return {
            'video_dir': LOCAL_VIDEO_DIR,
            'csv_path': LOCAL_CSV_PATH,
            'checkpoint_dir': LOCAL_CHECKPOINT_DIR,
            'log_dir': LOCAL_LOG_DIR,
        }


def get_config(mode='local'):
    """
    Get complete configuration.
    
    Args:
        mode: 'local' or 'gpu'
    
    Returns:
        dict with all configs merged
    """
    config = {
        **MODEL_CONFIG,
        **TRAINING_CONFIG,
        **VIDEO_CONFIG,
        **get_paths(mode)
    }
    return config


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    import json
    
    print("="*70)
    print("ISL TRANSLATION - CONFIGURATION")
    print("="*70)
    
    # GPU config
    print("\n[GPU Configuration]")
    gpu_config = get_config('gpu')
    print(json.dumps({k: str(v) for k, v in gpu_config.items()}, indent=2))
    
    # Local config
    print("\n[Local Configuration]")
    local_config = get_config('local')
    print(json.dumps({k: str(v) for k, v in local_config.items()}, indent=2))
    
    print("\n" + "="*70)
    print("TODO: Update GPU paths in this file before training!")
    print("="*70)
