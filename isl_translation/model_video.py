"""
ISL Translation Model - Video-based with MobileNetV3
====================================================
Architecture:
- MobileNetV3-Small encoder (pretrained ImageNet)
- Temporal conv for frame aggregation
- Conformer encoder for sequence modeling
- Transformer decoder with attention
- CTC + CrossEntropy hybrid loss

Mobile-optimized for Flutter deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class VideoModelConfig:
    """Model configuration for video-based ISL translation."""
    
    # Input
    frame_size: int = 224  # MobileNet input
    max_frames: int = 150
    
    # Encoder
    mobilenet_features: int = 576  # MobileNetV3-Small output
    d_model: int = 256
    num_encoder_layers: int = 4
    encoder_heads: int = 4
    encoder_ff_dim: int = 1024
    
    # Decoder
    num_decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_ff_dim: int = 1024
    
    # Vocab
    vocab_size: int = 2000
    max_target_len: int = 100
    
    # Special tokens
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    blank_id: int = 4
    
    # Training
    dropout: float = 0.1
    ctc_weight: float = 0.3
    label_smoothing: float = 0.1


# ============================================================================
# MobileNet Frame Encoder
# ============================================================================

class MobileNetFrameEncoder(nn.Module):
    """Extract features from individual frames using MobileNetV3."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained MobileNetV3-Small
        mobilenet = mobilenet_v3_small(pretrained=pretrained)
        
        # Remove classifier, keep only feature extractor
        # MobileNetV3-Small outputs 576-dim features
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        
        # Freeze early layers (optional - can fine-tune later)
        # for param in list(self.features.parameters())[:10]:
        #     param.requires_grad = False
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) - batch of frame sequences
        
        Returns:
            features: (B, T, 576) - per-frame features
        """
        B, T, C, H, W = frames.shape
        
        # Reshape to process all frames at once
        frames_flat = frames.view(B * T, C, H, W)
        
        # Extract features
        x = self.features(frames_flat)  # (B*T, 576, H', W')
        x = self.avgpool(x)  # (B*T, 576, 1, 1)
        x = x.view(B * T, -1)  # (B*T, 576)
        
        # Reshape back to sequence
        features = x.view(B, T, -1)  # (B, T, 576)
        
        return features


# ============================================================================
# Temporal Encoder
# ============================================================================

class TemporalEncoder(nn.Module):
    """Aggregate temporal information from frame features."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        
        # Project MobileNet features to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(config.mobilenet_features, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # 1D Conv for local temporal context
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Conformer-style encoder (simplified)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
    
    def forward(self, frame_features: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Args:
            frame_features: (B, T, 576)
            lengths: (B,) actual sequence lengths
        
        Returns:
            encoded: (B, T, d_model)
            lengths: (B,) output lengths
        """
        # Project to d_model
        x = self.input_proj(frame_features)  # (B, T, d_model)
        
        # Temporal conv
        x_conv = x.transpose(1, 2)  # (B, d_model, T)
        x_conv = self.temporal_conv(x_conv)
        x = x_conv.transpose(1, 2)  # (B, T, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create mask if lengths provided
        mask = None
        if lengths is not None:
            mask = self._create_padding_mask(x.size(0), x.size(1), lengths, x.device)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return encoded, lengths
    
    def _create_padding_mask(self, B: int, T: int, lengths: torch.Tensor, device) -> torch.Tensor:
        """Create padding mask: True for padding positions."""
        mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]
        return mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# Transformer Decoder (reuse from model_v2.py structure)
# ============================================================================

class VideoTranslationModel(nn.Module):
    """Complete video-based ISL translation model."""
    
    def __init__(self, config: VideoModelConfig):
        super().__init__()
        self.config = config
        
        # Frame encoder
        self.frame_encoder = MobileNetFrameEncoder(pretrained=True)
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(config)
        
        # CTC head
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Decoder
        self.decoder_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)
        
        self.decoder_output = nn.Linear(config.d_model, config.vocab_size)
        
        # Positional encoding for decoder
        self.decoder_pos_enc = PositionalEncoding(config.d_model, config.dropout)
    
    def forward(
        self,
        frames: torch.Tensor,
        frame_lengths: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, 224, 224)
            frame_lengths: (B,)
            targets: (B, L) - target token IDs
            target_lengths: (B,)
        
        Returns:
            dict with 'ctc_log_probs', 'decoder_logits', 'encoder_lengths'
        """
        # Encode frames
        frame_features = self.frame_encoder(frames)  # (B, T, 576)
        
        # Temporal encoding
        encoded, enc_lengths = self.temporal_encoder(frame_features, frame_lengths)  # (B, T, d_model)
        
        # CTC output
        ctc_logits = self.ctc_head(encoded)  # (B, T, vocab)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        
        # Decoder (if targets provided)
        decoder_logits = None
        if targets is not None:
            # Embed targets
            tgt_emb = self.decoder_embedding(targets)  # (B, L, d_model)
            tgt_emb = self.decoder_pos_enc(tgt_emb)
            
            # Create causal mask
            L = targets.size(1)
            tgt_mask = self._generate_square_subsequent_mask(L).to(targets.device)
            
            # Decode
            decoder_out = self.transformer_decoder(
                tgt_emb,
                encoded,
                tgt_mask=tgt_mask
            )  # (B, L, d_model)
            
            decoder_logits = self.decoder_output(decoder_out)  # (B, L, vocab)
        
        return {
            'ctc_log_probs': ctc_log_probs,
            'decoder_logits': decoder_logits,
            'encoder_lengths': enc_lengths
        }
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    @torch.no_grad()
    def decode_greedy(self, frames: torch.Tensor, frame_lengths: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        
        B = frames.size(0)
        device = frames.device
        
        # Encode
        frame_features = self.frame_encoder(frames)
        encoded, _ = self.temporal_encoder(frame_features, frame_lengths)
        
        # Start with BOS
        output_tokens = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            # Decode
            tgt_emb = self.decoder_embedding(output_tokens)
            tgt_emb = self.decoder_pos_enc(tgt_emb)
            
            L = output_tokens.size(1)
            tgt_mask = self._generate_square_subsequent_mask(L).to(device)
            
            decoder_out = self.transformer_decoder(tgt_emb, encoded, tgt_mask=tgt_mask)
            logits = self.decoder_output(decoder_out[:, -1:, :])  # (B, 1, vocab)
            
            # Greedy selection
            next_token = logits.argmax(dim=-1)  # (B, 1)
            output_tokens = torch.cat([output_tokens, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == self.config.eos_id).all():
                break
        
        return output_tokens
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Function (reuse hybrid CTC+Attention from model_v2.py)
# ============================================================================

class VideoHybridLoss(nn.Module):
    """Hybrid CTC + Cross-entropy loss for video model."""
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        blank_id: int = 4,
        ctc_weight: float = 0.3,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
        self.ctc_weight = ctc_weight
        self.vocab_size = vocab_size
    
    def forward(
        self,
        ctc_log_probs: torch.Tensor,
        decoder_logits: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ctc_log_probs: (B, T, vocab)
            decoder_logits: (B, L, vocab)
            encoder_lengths: (B,)
            targets: (B, L)
            target_lengths: (B,)
        """
        # CTC loss
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)  # (T, B, vocab)
        ctc_targets = targets[:, 1:]  # Remove BOS
        ctc_target_lengths = target_lengths - 1
        
        ctc_targets_flat = []
        for i in range(targets.size(0)):
            length = ctc_target_lengths[i].item()
            ctc_targets_flat.extend(ctc_targets[i, :length].tolist())
        ctc_targets_flat = torch.tensor(ctc_targets_flat, dtype=torch.long, device=targets.device)
        
        ctc_loss = self.ctc_loss(ctc_log_probs_t, ctc_targets_flat, encoder_lengths, ctc_target_lengths)
        
        # Cross-entropy loss
        ce_targets = targets[:, 1:].contiguous()
        ce_logits = decoder_logits[:, :-1].contiguous()
        ce_loss = self.ce_loss(ce_logits.view(-1, self.vocab_size), ce_targets.view(-1))
        
        # Combined
        total_loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * ce_loss
        
        return {
            'loss': total_loss,
            'ctc_loss': ctc_loss,
            'ce_loss': ce_loss
        }


# ============================================================================
# Helper
# ============================================================================

def create_video_model(config: VideoModelConfig) -> VideoTranslationModel:
    """Create model instance."""
    model = VideoTranslationModel(config)
    print(f"Created VideoTranslationModel with {model.count_parameters():,} parameters")
    return model


if __name__ == '__main__':
    # Test model
    config = VideoModelConfig()
    model = create_video_model(config)
    
    # Dummy input
    frames = torch.randn(2, 50, 3, 224, 224)  # 2 videos, 50 frames each
    frame_lengths = torch.tensor([50, 40])
    targets = torch.randint(5, config.vocab_size, (2, 20))
    targets[:, 0] = config.bos_id
    target_lengths = torch.tensor([20, 15])
    
    # Forward pass
    outputs = model(frames, frame_lengths, targets, target_lengths)
    print(f"CTC log probs: {outputs['ctc_log_probs'].shape}")
    print(f"Decoder logits: {outputs['decoder_logits'].shape}")
    
    # Test loss
    loss_fn = VideoHybridLoss(config.vocab_size)
    losses = loss_fn(
        outputs['ctc_log_probs'],
        outputs['decoder_logits'],
        outputs['encoder_lengths'],
        targets,
        target_lengths
    )
    print(f"Total loss: {losses['loss'].item():.4f}")
    print("✓ Model test passed!")
