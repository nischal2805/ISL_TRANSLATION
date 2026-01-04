"""Sign Language Encoder with VideoMAE Transfer Learning"""
import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
import warnings


class SignEncoder(nn.Module):
    """VideoMAE-based encoder processing actual video frames."""
    
    def __init__(
        self,
        output_dim: int = 256,
        pretrained: str = "MCG-NJU/videomae-base",
        freeze_epochs: int = 10,
        dropout: float = 0.1,
        num_frames: int = 16  # VideoMAE expects 16 frames per clip
    ):
        super().__init__()
        self.output_dim = output_dim
        self.freeze_epochs = freeze_epochs
        self._frozen = True
        self.num_frames = num_frames
        
        # Load pretrained VideoMAE model
        print(f"Loading pretrained VideoMAE: {pretrained}")
        try:
            self.videomae = VideoMAEModel.from_pretrained(pretrained)
            self.processor = VideoMAEImageProcessor.from_pretrained(pretrained)
            self.hidden_dim = self.videomae.config.hidden_size  # 768 for base
        except Exception as e:
            print(f"Warning: Could not load pretrained: {e}")
            warnings.warn("Using random initialization - training may be slower")
            from transformers import VideoMAEConfig
            config = VideoMAEConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
            self.videomae = VideoMAEModel(config)
            self.hidden_dim = 768
            self._frozen = False
        
        # Temporal pooling to reduce sequence length
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Pool over time
        
        # Output projection to decoder dimension
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        if self._frozen:
            self.freeze()
    
    def freeze(self):
        """Freeze VideoMAE weights for initial training."""
        for p in self.videomae.parameters():
            p.requires_grad = False
        self._frozen = True
        print("✓ VideoMAE encoder frozen")
        
    def unfreeze(self):
        """Unfreeze for fine-tuning."""
        for p in self.videomae.parameters():
            p.requires_grad = True
        self._frozen = False
        print("✓ VideoMAE encoder unfrozen for fine-tuning")
        
    def forward(self, video_frames, lengths=None):
        """
        Args:
            video_frames: (B, num_frames, C, H, W) - preprocessed video frames
            lengths: (B,) - actual number of valid frames (currently unused)
        Returns:
            encoder_out: (B, T_out, output_dim) - encoded features
            out_lengths: (B,) - output sequence lengths
        """
        B = video_frames.size(0)
        
        # VideoMAE forward pass
        # Input: (B, num_frames, C, H, W)
        # Output: (B, seq_len, hidden_dim) where seq_len depends on patch size
        outputs = self.videomae(video_frames)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Project to decoder dimension
        encoded = self.output_proj(hidden_states)  # (B, seq_len, output_dim)
        
        # Calculate output lengths (same as input for now)
        if lengths is None:
            out_lengths = torch.full((B,), encoded.size(1), dtype=torch.long, device=encoded.device)
        else:
            # Scale lengths proportionally to sequence reduction
            out_lengths = lengths  # Simplified for now
        
        return encoded, out_lengths
