"""Sign Language Encoder with VideoMAE Transfer Learning"""
import torch
import torch.nn as nn
from transformers import VideoMAEModel, VideoMAEConfig


class PoseProjection(nn.Module):
    """Project pose landmarks to VideoMAE-compatible features."""
    def __init__(self, input_dim: int = 603, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(input_dim, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(384, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # (B, T, C) -> (B, C, T) -> conv -> (B, H, T) -> (B, T, H)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.dropout(self.norm(x))


class SignEncoder(nn.Module):
    """VideoMAE-based encoder with transfer learning."""
    
    def __init__(
        self,
        input_dim: int = 603,
        hidden_dim: int = 768,
        output_dim: int = 256,
        pretrained: str = "MCG-NJU/videomae-base",
        freeze_epochs: int = 10,  # More epochs frozen for stable transfer
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.freeze_epochs = freeze_epochs
        self._frozen = True
        
        # Pose projection to VideoMAE dimension
        self.pose_proj = PoseProjection(input_dim, hidden_dim, dropout)
        
        # Load pretrained VideoMAE encoder
        print(f"Loading pretrained: {pretrained}")
        try:
            self.transformer = VideoMAEModel.from_pretrained(pretrained).encoder
        except Exception as e:
            print(f"Could not load pretrained: {e}, using random init")
            config = VideoMAEConfig(hidden_size=hidden_dim, num_hidden_layers=12, num_attention_heads=12)
            self.transformer = VideoMAEModel(config).encoder
            self._frozen = False
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        if self._frozen:
            self.freeze()
    
    def freeze(self):
        for p in self.transformer.parameters():
            p.requires_grad = False
        self._frozen = True
        
    def unfreeze(self):
        for p in self.transformer.parameters():
            p.requires_grad = True
        self._frozen = False
        print("Encoder unfrozen")
        
    def forward(self, x, lengths):
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,)
        Returns:
            encoder_out: (B, T, output_dim)
            lengths: (B,)
        """
        # Project to VideoMAE dimension
        x = self.pose_proj(x)
        
        # Pass through VideoMAE encoder
        x = self.transformer(hidden_states=x).last_hidden_state
        
        # Project to decoder dimension
        x = self.output_proj(x)
        
        return x, lengths
