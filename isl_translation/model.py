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

class CrossAttention(nn.Module):
    """Cross-attention module for decoder with NaN safety."""
    
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
    GRU Decoder with cross-attention.
    
    Supports teacher forcing during training and autoregressive generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # GRU layers
        self.gru = nn.GRU(
            d_model, d_model, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Cross-attention
        self.cross_attention = CrossAttention(d_model, num_heads, dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
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
        
        Args:
            targets: (B, L) target token IDs
            encoder_output: (B, T, d_model) encoder outputs
            encoder_mask: (B, T) padding mask
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            (B, L, vocab_size) logits
        """
        batch_size = targets.size(0)
        max_len = targets.size(1)
        device = targets.device
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, max_len, self.vocab_size, device=device)
        
        # Initialize hidden state
        hidden = None
        
        # Start with SOS token
        input_token = targets[:, 0].unsqueeze(1)  # (B, 1)
        
        for t in range(max_len):
            # Embed input token
            embedded = self.embedding(input_token)  # (B, 1, d_model)
            embedded = self.dropout(embedded)
            
            # GRU step
            gru_out, hidden = self.gru(embedded, hidden)  # gru_out: (B, 1, d_model)
            
            # Cross-attention
            context = self.cross_attention(gru_out, encoder_output, encoder_mask)
            
            # Combine GRU output and context
            combined = torch.cat([gru_out, context], dim=-1)  # (B, 1, d_model*2)
            
            # Output projection
            logits = self.output_proj(combined)  # (B, 1, vocab_size)
            outputs[:, t] = logits.squeeze(1)
            
            # Prepare next input
            if t < max_len - 1:
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher_forcing and t + 1 < targets.size(1):
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
        """
        Greedy decoding for inference.
        
        Args:
            encoder_output: (B, T, d_model)
            encoder_mask: (B, T) padding mask
            max_len: Maximum output length
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            
        Returns:
            output_ids: (B, L) decoded token IDs
            output_probs: (B, L) token probabilities
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize
        output_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
        output_probs = torch.zeros(batch_size, max_len, device=device)
        
        hidden = None
        input_token = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(max_len):
            # Embed
            embedded = self.embedding(input_token)
            
            # GRU step
            gru_out, hidden = self.gru(embedded, hidden)
            
            # Cross-attention
            context = self.cross_attention(gru_out, encoder_output, encoder_mask)
            
            # Output
            combined = torch.cat([gru_out, context], dim=-1)
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
        Beam search decoding for inference.
        
        Args:
            encoder_output: (B, T, d_model) - only B=1 supported for beam search
            encoder_mask: (B, T) padding mask
            max_len: Maximum output length
            beam_width: Number of beams to keep
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            length_penalty: Length normalization factor (alpha)
            
        Returns:
            output_ids: (B, L) decoded token IDs (best beam)
            output_probs: (B,) sequence log probability
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # For simplicity, process one sample at a time
        if batch_size > 1:
            all_ids = []
            all_probs = []
            for b in range(batch_size):
                enc_out = encoder_output[b:b+1]
                enc_mask = encoder_mask[b:b+1] if encoder_mask is not None else None
                ids, probs = self.decode_beam(
                    enc_out, enc_mask, max_len, beam_width, sos_id, eos_id, length_penalty
                )
                all_ids.append(ids)
                all_probs.append(probs)
            
            # Pad to same length
            max_out_len = max(ids.size(1) for ids in all_ids)
            padded_ids = torch.zeros(batch_size, max_out_len, dtype=torch.long, device=device)
            for b, ids in enumerate(all_ids):
                padded_ids[b, :ids.size(1)] = ids[0]
            
            return padded_ids, torch.cat(all_probs)
        
        # Beam search for single sample
        # Each beam: (sequence, hidden_state, log_prob, finished)
        
        # Expand encoder output for beam search
        encoder_output_expanded = encoder_output.repeat(beam_width, 1, 1)  # (beam, T, d)
        encoder_mask_expanded = encoder_mask.repeat(beam_width, 1) if encoder_mask is not None else None
        
        # Initialize beams
        beams = [{
            'tokens': [sos_id],
            'log_prob': 0.0,
            'hidden': None,
            'finished': False
        }]
        
        for t in range(max_len):
            all_candidates = []
            
            for beam in beams:
                if beam['finished']:
                    all_candidates.append(beam)
                    continue
                
                # Get last token
                input_token = torch.tensor([[beam['tokens'][-1]]], dtype=torch.long, device=device)
                
                # Embed
                embedded = self.embedding(input_token)
                
                # GRU step
                gru_out, hidden = self.gru(embedded, beam['hidden'])
                
                # Cross-attention (use first beam's encoder output)
                context = self.cross_attention(gru_out, encoder_output, encoder_mask)
                
                # Output
                combined = torch.cat([gru_out, context], dim=-1)
                logits = self.output_proj(combined).squeeze(1)  # (1, vocab_size)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)
                
                # Get top-k candidates
                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                
                for k in range(beam_width):
                    token_id = topk_ids[k].item()
                    token_log_prob = topk_log_probs[k].item()
                    
                    new_beam = {
                        'tokens': beam['tokens'] + [token_id],
                        'log_prob': beam['log_prob'] + token_log_prob,
                        'hidden': hidden.clone() if hidden is not None else None,  # Clone to prevent interference
                        'finished': token_id == eos_id
                    }
                    all_candidates.append(new_beam)
            
            # Select top beams with length normalization
            def score_beam(b):
                length = len(b['tokens'])
                return b['log_prob'] / (length ** length_penalty)
            
            all_candidates.sort(key=score_beam, reverse=True)
            beams = all_candidates[:beam_width]
            
            # Check if all beams finished
            if all(b['finished'] for b in beams):
                break
        
        # Return best beam
        best_beam = max(beams, key=lambda b: b['log_prob'] / (len(b['tokens']) ** length_penalty))
        
        # Remove SOS token from output
        output_tokens = best_beam['tokens'][1:]  # Remove SOS
        if output_tokens and output_tokens[-1] == eos_id:
            output_tokens = output_tokens[:-1]  # Remove EOS
        
        output_ids = torch.tensor([output_tokens], dtype=torch.long, device=device)
        output_prob = torch.tensor([best_beam['log_prob']], device=device)
        
        return output_ids, output_prob


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
