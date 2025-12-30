"""
ISL Translation System
======================

Indian Sign Language to Text Translation using Deep Learning.

Modules:
    - config: Configuration settings with GPU mode switching
    - vocab: Character-level vocabulary (35 tokens)
    - preprocessing: MediaPipe landmark extraction and feature engineering
    - dataset: PyTorch Dataset and DataLoader
    - model: ISL Translation Model (Encoder + CTC + GRU Decoder)
    - train: Training loop with hybrid loss
    - evaluate: Evaluation metrics
    - test: Testing and inference
    - realtime_demo: OpenCV webcam demo
    - utils: Helper functions
"""

from .config import (
    data_config,
    landmark_config,
    model_config,
    training_config,
    vocab_config,
    inference_config,
    set_gpu_mode,
    DEVICE
)

from .vocab import Vocabulary

from .model import (
    ISLTranslationModel,
    ISLEncoder,
    CTCHead,
    GRUDecoder,
    create_model
)

__version__ = '1.0.0'
__author__ = 'ISL Translation Team'

__all__ = [
    # Config
    'data_config',
    'landmark_config', 
    'model_config',
    'training_config',
    'vocab_config',
    'inference_config',
    'set_gpu_mode',
    'DEVICE',
    # Vocab
    'Vocabulary',
    # Model
    'ISLTranslationModel',
    'ISLEncoder',
    'CTCHead',
    'GRUDecoder',
    'create_model',
]
