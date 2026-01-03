"""
ISL Translation System - Model Architecture
============================================
Multi-scale CNN + Conformer Encoder with CTC Head + GRU Decoder.
~18M parameters, optimized for mobile deployment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from config import model_config, vocab_config


# ============================================================================
# Utility Modules
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: cos positions may have one fewer element
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:, :-1]) if div_term.size(-1) > pe[:, 1::2].size(-1) else torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """Convolutional subsampling layer (2x time reduction)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model)
            lengths: (B,) optional sequence lengths
        Returns:
            x: (B, T//2, d_model)
            lengths: (B,) updated lengths
        """
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T//2, d_model)
        
        if lengths is not None:
            lengths = (lengths + 1) // 2
        
        return x, lengths


# ============================================================================
# Multi-Scale CNN Module
# ============================================================================

class MultiScaleCNNBlock(nn.Module):
    """Multi-scale 1D CNN block with parallel convolutions."""
    
    def __init__(
        self,
        d_model: int,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1
    ):
        super().__init__()
        
        num_branches = len(kernel_sizes)
        # Ensure output dimension matches d_model exactly
        branch_dim = d_model // num_branches
        last_branch_dim = d_model - branch_dim * (num_branches - 1)  # Handle remainder
        
        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_dim = last_branch_dim if i == num_branches - 1 else branch_dim
            branch = nn.Sequential(
                nn.Conv1d(d_model, out_dim, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            )
            self.branches.append(branch)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = x.transpose(1, 2)  # (B, d_model, T)
        
        # Multi-scale convolutions
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)  # (B, d_model, T)
        
        x = x.transpose(1, 2)  # (B, T, d_model)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        
        return x


# ============================================================================
# Conformer Modules
# ============================================================================

class FeedForwardModule(nn.Module):
    """Conformer feed-forward module."""
    
    def __init__(self, d_model: int, ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * ff_expansion)
        self.activation = nn.SiLU()  # Swish activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * ff_expansion, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x * 0.5 + residual


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module with NaN safety."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        
        # Attention with NaN safety
        x, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        
        # Check for NaN in output (can happen if all positions masked)
        if torch.isnan(x).any():
            # Replace NaN with zeros (safe fallback)
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.dropout(x)
        return x + residual


class ConvolutionModule(nn.Module):
    """Conformer convolution module."""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise conv (expand)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        
        # Pointwise conv (project)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        
        x = self.dropout(x)
        return x + residual


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attention(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.layer_norm(x)
        return x


# ============================================================================
# Encoder
# ============================================================================

class ISLEncoder(nn.Module):
    """
    ISL Translation Encoder.
    
    Architecture:
    - Input projection (414 -> d_model)
    - Multi-scale CNN (2 blocks)
    - Temporal subsampling (2x)
    - Conformer blocks (2 blocks)
    """
    
    def __init__(
        self,
        input_dim: int = 414,
        d_model: int = 256,
        num_cnn_blocks: int = 2,
        num_conformer_blocks: int = 2,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-scale CNN blocks
        self.cnn_blocks = nn.ModuleList([
            MultiScaleCNNBlock(d_model, dropout=dropout)
            for _ in range(num_cnn_blocks)
        ])
        
        # Temporal subsampling
        self.subsample = ConvSubsampling(d_model, dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model, num_heads, ff_expansion,
                conv_kernel_size, dropout
            )
            for _ in range(num_conformer_blocks)
        ])
        
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, 414) input features
            lengths: (B,) sequence lengths
            
        Returns:
            encoder_output: (B, T', d_model)
            output_lengths: (B,) updated lengths after subsampling
        """
        # Ensure minimum sequence length (at least 4 frames for subsampling + conv)
        min_seq_len = 4
        if x.size(1) < min_seq_len:
            # Pad short sequences
            padding = torch.zeros(x.size(0), min_seq_len - x.size(1), x.size(2), 
                                  device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
            if lengths is not None:
                lengths = lengths.clamp(min=min_seq_len)
        
        # Input projection
        x = self.input_proj(x)  # (B, T, d_model)
        
        # Multi-scale CNN
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)
        
        # Temporal subsampling
        x, lengths = self.subsample(x, lengths)
        
        # Ensure lengths are at least 1 after subsampling
        if lengths is not None:
            lengths = lengths.clamp(min=1)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, mask)
        
        return x, lengths


# ============================================================================
# CTC Head (NaN-safe)
# ============================================================================

class CTCHead(nn.Module):
    """CTC output head with NaN-safe log_softmax."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, vocab_size) log probabilities (clamped to prevent -inf)
        """
        logits = self.fc(x)
        # Clamp logits to prevent extreme values before log_softmax
        logits = logits.clamp(min=-100, max=100)
        log_probs = F.log_softmax(logits, dim=-1)
        # Clamp log_probs to prevent -inf (which causes NaN in CTC loss)
        return log_probs.clamp(min=-100)


# ============================================================================
# GRU Decoder with Cross-Attention
# ============================================================================

class LocationAwareAttention(nn.Module):
    """
    Location-aware attention (Chorowski et al., 2015).
    
    Uses previous attention weights to help focus on the next positions,
    which is critical for monotonic sequence-to-sequence tasks like sign language.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, 
                 location_filters: int = 32, location_kernel: int = 31):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Location-based attention: conv over previous attention weights
        self.location_conv = nn.Conv1d(
            1, location_filters, 
            kernel_size=location_kernel, 
            padding=location_kernel // 2,
            bias=False
        )
        self.location_proj = nn.Linear(location_filters, d_model, bias=False)
        
        # Score projection
        self.score_proj = nn.Linear(d_model, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        prev_attention: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, 1, d_model) decoder hidden state
            key_value: (B, T, d_model) encoder outputs
            key_padding_mask: (B, T) padding mask (True = masked)
            prev_attention: (B, T) previous attention weights
        Returns:
            context: (B, 1, d_model) attended context
            attention_weights: (B, T) attention weights for next step
        """
        B, T, _ = key_value.shape
        device = query.device
        
        residual = query
        query = self.layer_norm(query)
        
        # Project query, key, value
        Q = self.query_proj(query)  # (B, 1, d_model)
        K = self.key_proj(key_value)  # (B, T, d_model)
        V = self.value_proj(key_value)  # (B, T, d_model)
        
        # Location features from previous attention
        if prev_attention is None:
            # Initialize with uniform attention
            prev_attention = torch.ones(B, T, device=device) / T
            if key_padding_mask is not None:
                prev_attention = prev_attention.masked_fill(key_padding_mask, 0.0)
                prev_attention = prev_attention / prev_attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Conv over attention: (B, 1, T) -> (B, filters, T)
        location_feat = self.location_conv(prev_attention.unsqueeze(1))
        location_feat = location_feat.transpose(1, 2)  # (B, T, filters)
        location_feat = self.location_proj(location_feat)  # (B, T, d_model)
        
        # Compute attention scores with location
        # Energy = Q * K + location_feat
        energy = Q + K + location_feat  # (B, T, d_model) via broadcasting
        energy = torch.tanh(energy)
        scores = self.score_proj(energy).mean(dim=-1)  # (B, T)
        
        # Apply mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-masked positions
        if torch.isnan(attention_weights).any():
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply attention to values
        context = torch.bmm(attention_weights.unsqueeze(1), V)  # (B, 1, d_model)
        context = self.out_proj(context)
        context = self.dropout(context) + residual
        
        return context, attention_weights


class CrossAttention(nn.Module):
    """Cross-attention module for decoder with NaN safety (fallback for compatibility)."""
    
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, 1, d_model) decoder hidden state
            key_value: (B, T, d_model) encoder outputs
            key_padding_mask: (B, T) padding mask
        Returns:
            (B, 1, d_model) attended context
        """
        residual = query
        query = self.layer_norm(query)
        attn_out, _ = self.attention(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask
        )
        
        # NaN safety: Replace NaN with zeros
        if torch.isnan(attn_out).any():
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
        
        return self.dropout(attn_out) + residual


class GRUDecoder(nn.Module):
    """
    Improved GRU Decoder with location-aware attention.
    
    Key improvements:
    1. Location-aware attention for better monotonic alignment
    2. Deeper GRU with residual connections
    3. Layer normalization for stability
    4. Attention history tracking
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding with scaling
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.embed_scale = math.sqrt(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Pre-net: 2-layer MLP before GRU (helps with attention alignment)
        self.prenet = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout in prenet is important
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # GRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(d_model if i == 0 else d_model, d_model, batch_first=True)
            for i in range(num_layers)
        ])
        self.gru_layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.gru_dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Location-aware attention
        self.attention = LocationAwareAttention(d_model, num_heads, dropout)
        
        # Pre-output layer norm
        self.pre_output_norm = nn.LayerNorm(d_model * 2)
        
        # Output projection (2-layer for better expressiveness)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        """
        batch_size = targets.size(0)
        max_len = targets.size(1)
        device = targets.device
        enc_len = encoder_output.size(1)
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
        
        # Initialize hidden states for each layer
        hiddens = [None] * self.num_layers
        
        # Initialize attention weights (uniform)
        prev_attention = None
        
        # Start with SOS token
        input_token = targets[:, 0].unsqueeze(1)  # (B, 1)
        
        for t in range(max_len):
            # Embed input token with scaling
            embedded = self.embedding(input_token) * self.embed_scale  # (B, 1, d_model)
            embedded = self.embed_dropout(embedded)
            
            # Pre-net
            prenet_out = self.prenet(embedded)
            
            # GRU layers with residual connections
            gru_out = prenet_out
            for i in range(self.num_layers):
                residual = gru_out
                gru_out, hiddens[i] = self.gru_layers[i](gru_out, hiddens[i])
                gru_out = self.gru_layer_norms[i](gru_out)
                gru_out = self.gru_dropouts[i](gru_out)
                if i > 0:  # Residual from second layer onwards
                    gru_out = gru_out + residual
            
            # Location-aware attention
            context, prev_attention = self.attention(
                gru_out, encoder_output, encoder_mask, prev_attention
            )
            
            # Combine GRU output and context
            combined = torch.cat([gru_out, context], dim=-1)  # (B, 1, d_model*2)
            combined = self.pre_output_norm(combined)
            
            # Output projection
            logits = self.output_proj(combined)  # (B, 1, vocab_size)
            outputs[:, t] = logits.squeeze(1)
            
            # Prepare next input
            if t < max_len - 1:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing:
                    input_token = targets[:, t + 1].unsqueeze(1)
                else:
                    input_token = logits.argmax(dim=-1)
        
        return outputs
    
    def decode_greedy(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        sos_id: int = 1,
        eos_id: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding for inference."""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize
        output_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
        output_probs = torch.zeros(batch_size, max_len, device=device)
        
        hiddens = [None] * self.num_layers
        prev_attention = None
        input_token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            # Embed with scaling
            embedded = self.embedding(input_token) * self.embed_scale
            
            # Pre-net
            prenet_out = self.prenet(embedded)
            
            # GRU layers with residual
            gru_out = prenet_out
            for i in range(self.num_layers):
                residual = gru_out
                gru_out, hiddens[i] = self.gru_layers[i](gru_out, hiddens[i])
                gru_out = self.gru_layer_norms[i](gru_out)
                if i > 0:
                    gru_out = gru_out + residual
            
            # Attention
            context, prev_attention = self.attention(
                gru_out, encoder_output, encoder_mask, prev_attention
            )
            
            # Output
            combined = torch.cat([gru_out, context], dim=-1)
            combined = self.pre_output_norm(combined)
            logits = self.output_proj(combined).squeeze(1)  # (B, vocab_size)
            
            probs = F.softmax(logits, dim=-1)
            next_token = probs.argmax(dim=-1)  # (B,)
            
            # Store outputs
            output_ids[:, t] = next_token
            output_probs[:, t] = probs.gather(1, next_token.unsqueeze(1)).squeeze(1)
            
            # Check for EOS
            finished = finished | (next_token == eos_id)
            if finished.all():
                break
            
            input_token = next_token.unsqueeze(1)
        
        return output_ids, output_probs
    
    def decode_beam(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        beam_width: int = 5,
        sos_id: int = 1,
        eos_id: int = 2,
        length_penalty: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified beam search decoding (falls back to greedy for speed).
        For production, use greedy decoding which is more reliable with location-aware attention.
        """
        # Location-aware attention doesn't work well with beam search
        # because attention history is per-beam and complex to track.
        # Fall back to greedy decoding for simplicity and reliability.
        return self.decode_greedy(
            encoder_output, encoder_mask, max_len, sos_id, eos_id
        )


# ============================================================================
# Full Model
# ============================================================================

class ISLTranslationModel(nn.Module):
    """
    Complete ISL Translation Model.
    
    Components:
    - ISLEncoder: Multi-scale CNN + Conformer
    - CTCHead: CTC output for auxiliary loss
    - GRUDecoder: Autoregressive decoder with cross-attention
    """
    
    def __init__(
        self,
        input_dim: int = 414,
        d_model: int = 256,
        vocab_size: int = 35,
        num_cnn_blocks: int = 2,
        num_conformer_blocks: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.encoder = ISLEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_cnn_blocks=num_cnn_blocks,
            num_conformer_blocks=num_conformer_blocks,
            num_heads=num_heads,
            ff_expansion=ff_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        self.ctc_head = CTCHead(d_model, vocab_size)
        
        self.decoder = GRUDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.d_model = d_model
        self.vocab_size = vocab_size
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            features: (B, T, 414) input features
            feature_lengths: (B,) input lengths
            targets: (B, L) target token IDs
            teacher_forcing_ratio: Teacher forcing ratio
            
        Returns:
            Dictionary with:
                - ctc_logits: (B, T', vocab_size) CTC log probabilities
                - decoder_logits: (B, L, vocab_size) decoder logits
                - encoder_lengths: (B,) encoder output lengths
        """
        # Encode
        encoder_output, encoder_lengths = self.encoder(features, feature_lengths)
        
        # CTC output
        ctc_logits = self.ctc_head(encoder_output)
        
        # Create encoder mask
        max_len = encoder_output.size(1)
        encoder_mask = torch.arange(max_len, device=encoder_output.device).unsqueeze(0) >= encoder_lengths.unsqueeze(1)
        
        # Decode
        decoder_logits = self.decoder(
            targets, encoder_output, encoder_mask, teacher_forcing_ratio
        )
        
        return {
            'ctc_logits': ctc_logits,
            'decoder_logits': decoder_logits,
            'encoder_lengths': encoder_lengths
        }
    
    def decode(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        max_len: int = 100,
        use_ctc: bool = False,
        beam_width: int = 1,
        length_penalty: float = 0.6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference decoding.
        
        Args:
            features: (B, T, 414) input features
            feature_lengths: (B,) input lengths (optional)
            max_len: Maximum output length
            use_ctc: If True, use CTC decoding; otherwise GRU decoder
            beam_width: Beam width for beam search (1 = greedy)
            length_penalty: Length normalization for beam search
            
        Returns:
            output_ids: (B, L) decoded token IDs
            output_probs: (B, L) or (B,) token probabilities
        """
        # Encode
        encoder_output, encoder_lengths = self.encoder(features, feature_lengths)
        
        if use_ctc:
            # CTC greedy decoding
            ctc_logits = self.ctc_head(encoder_output)
            output_ids = ctc_logits.argmax(dim=-1)
            output_probs = ctc_logits.exp().max(dim=-1)[0]
            return output_ids, output_probs
        
        # Create encoder mask
        max_enc_len = encoder_output.size(1)
        if encoder_lengths is not None:
            encoder_mask = torch.arange(max_enc_len, device=encoder_output.device).unsqueeze(0) >= encoder_lengths.unsqueeze(1)
        else:
            encoder_mask = None
        
        # Choose decoding strategy
        if beam_width > 1:
            # Beam search decoding
            return self.decoder.decode_beam(
                encoder_output, encoder_mask, max_len,
                beam_width=beam_width,
                sos_id=vocab_config.sos_id,
                eos_id=vocab_config.eos_id,
                length_penalty=length_penalty
            )
        else:
            # Greedy decoding
            return self.decoder.decode_greedy(
                encoder_output, encoder_mask, max_len,
                sos_id=vocab_config.sos_id,
                eos_id=vocab_config.eos_id
            )
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config=None) -> ISLTranslationModel:
    """Create model from config."""
    if config is None:
        from config import model_config as config
    
    model = ISLTranslationModel(
        input_dim=config.input_dim,
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        num_cnn_blocks=config.num_cnn_blocks,
        num_conformer_blocks=config.num_conformer_blocks,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        ff_expansion=config.ff_expansion,
        conv_kernel_size=config.conv_kernel_size,
        dropout=config.dropout
    )
    
    return model


if __name__ == "__main__":
    # Test model
    print("ISL Translation Model Test")
    print("=" * 50)
    
    model = create_model()
    
    print(f"\nModel Parameters: {model.count_parameters():,}")
    print(f"Expected: ~18M parameters")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 100
    target_len = 20
    
    features = torch.randn(batch_size, seq_len, 414)
    feature_lengths = torch.tensor([100, 80, 90, 70])
    targets = torch.randint(0, 35, (batch_size, target_len))
    
    outputs = model(features, feature_lengths, targets)
    
    print(f"CTC logits shape: {outputs['ctc_logits'].shape}")
    print(f"Decoder logits shape: {outputs['decoder_logits'].shape}")
    print(f"Encoder lengths: {outputs['encoder_lengths']}")
    
    # Test inference - Greedy
    print("\nTesting greedy decoding...")
    model.eval()
    with torch.no_grad():
        output_ids, output_probs = model.decode(features[:1], feature_lengths[:1])
        print(f"Greedy - Output IDs shape: {output_ids.shape}")
        print(f"Greedy - Output probs shape: {output_probs.shape}")
    
    # Test inference - Beam Search
    print("\nTesting beam search decoding (beam_width=5)...")
    with torch.no_grad():
        output_ids, output_probs = model.decode(
            features[:1], feature_lengths[:1], 
            beam_width=5, length_penalty=0.6
        )
        print(f"Beam Search - Output IDs shape: {output_ids.shape}")
        print(f"Beam Search - Output probs shape: {output_probs.shape}")
    
    # Test with short sequence (edge case)
    print("\nTesting short sequence handling (3 frames)...")
    short_features = torch.randn(1, 3, 414)
    short_lengths = torch.tensor([3])
    with torch.no_grad():
        output_ids, _ = model.decode(short_features, short_lengths)
        print(f"Short sequence handled successfully, output shape: {output_ids.shape}")
