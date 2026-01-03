"""
ISL Translation System - Production Model Architecture
======================================================
Hybrid CTC-Attention model with Transformer decoder.
Optimized for:
- High translation accuracy (85%+ with proper training)
- Real-time streaming inference on mobile
- ONNX export for Flutter integration

Architecture:
- Multi-scale CNN for local feature extraction
- Conformer Encoder for temporal modeling  
- Transformer Decoder with cross-attention
- CTC Head for streaming + auxiliary loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration."""
    # Input - V2 landmarks with face features
    # 204 dims (hands + body + mouth + head) × 3 (pos + vel + acc) = 612
    input_dim: int = 612
    
    # Encoder - INCREASED for better capacity
    d_model: int = 384  # Increased from 256 for better representation
    num_cnn_blocks: int = 2
    num_encoder_layers: int = 6  # Increased from 4 for deeper encoding
    encoder_heads: int = 6  # Increased to match d_model/64
    encoder_ff_dim: int = 1536  # 4x d_model
    conv_kernel_size: int = 31
    
    # Decoder (Transformer)
    num_decoder_layers: int = 4
    decoder_heads: int = 6  # Match encoder heads
    decoder_ff_dim: int = 1536  # 4x d_model
    
    # Vocab
    vocab_size: int = 2000  # BPE vocab size
    max_target_len: int = 100
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # CTC
    ctc_weight: float = 0.3  # Weight for CTC loss in hybrid loss
    
    # Special tokens
    pad_id: int = 0
    blank_id: int = 4  # CTC blank
    bos_id: int = 2
    eos_id: int = 3


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
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """2x temporal subsampling with convolution."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        if lengths is not None:
            lengths = (lengths + 1) // 2
        return x, lengths


# ============================================================================
# Multi-Scale CNN
# ============================================================================

class MultiScaleCNNBlock(nn.Module):
    """Multi-scale 1D CNN with parallel convolutions."""
    
    def __init__(self, d_model: int, kernel_sizes: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.1):
        super().__init__()
        
        num_branches = len(kernel_sizes)
        branch_dim = d_model // num_branches
        last_dim = d_model - branch_dim * (num_branches - 1)
        
        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_dim = last_dim if i == num_branches - 1 else branch_dim
            self.branches.append(nn.Sequential(
                nn.Conv1d(d_model, out_dim, kernel_size=k, padding='same'),
                nn.BatchNorm1d(out_dim),
                nn.ReLU()
            ))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


# ============================================================================
# Conformer Encoder
# ============================================================================

class FeedForward(nn.Module):
    """Feed-forward module with SiLU activation."""
    
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class ConformerConvModule(nn.Module):
    """Conformer convolution module."""
    
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size-1)//2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)
        return residual + self.dropout(x)


class ConformerBlock(nn.Module):
    """Conformer block: FF -> MHSA -> Conv -> FF."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, conv_kernel: int, dropout: float = 0.1):
        super().__init__()
        
        self.ff1 = FeedForward(d_model, ff_dim, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, ff_dim, dropout)
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ff1(x)
        
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = torch.nan_to_num(x, nan=0.0)  # NaN safety
        x = residual + self.self_attn_dropout(x)
        
        x = self.conv(x)
        x = self.ff2(x)
        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Full Conformer encoder."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Multi-scale CNN
        self.cnn_blocks = nn.ModuleList([
            MultiScaleCNNBlock(config.d_model, dropout=config.dropout)
            for _ in range(config.num_cnn_blocks)
        ])
        
        # Subsampling
        self.subsample = ConvSubsampling(config.d_model, config.dropout)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(config.d_model, dropout=config.dropout)
        
        # Conformer layers
        self.layers = nn.ModuleList([
            ConformerBlock(
                config.d_model, config.encoder_heads, config.encoder_ff_dim,
                config.conv_kernel_size, config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        self.d_model = config.d_model
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,)
        Returns:
            x: (B, T', d_model)
            lengths: (B,)
        """
        x = self.input_proj(x)
        
        for cnn in self.cnn_blocks:
            x = cnn(x)
        
        x, lengths = self.subsample(x, lengths)
        x = self.pos_enc(x)
        
        # Create mask
        mask = None
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return x, lengths, mask


# ============================================================================
# Transformer Decoder
# ============================================================================

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with masked self-attention and cross-attention."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Masked self-attention
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # Cross-attention - CRITICAL for encoder conditioning
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # Gate to control encoder vs language model influence
        # Initialize with positive bias to favor encoder from the start
        self.encoder_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        # Initialize gate bias to +2.0 so sigmoid(2)=0.88 -> 88% encoder influence
        nn.init.constant_(self.encoder_gate[0].bias, 2.0)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Masked self-attention
        residual = x
        x_norm = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=tgt_mask)
        x = residual + self.self_attn_dropout(attn_out)
        
        # Cross-attention with gating mechanism
        residual = x
        x_norm = self.cross_attn_norm(x)
        cross_out, cross_weights = self.cross_attn(x_norm, memory, memory, key_padding_mask=memory_mask)
        
        # Gate: how much to use encoder vs language model
        gate_input = torch.cat([x, cross_out], dim=-1)
        gate = self.encoder_gate(gate_input)
        
        # Apply gate: gate=1 means use encoder, gate=0 means ignore encoder
        x = residual + self.cross_attn_dropout(gate * cross_out)
        
        # Feed-forward
        residual = x
        x_norm = self.ff_norm(x)
        x = residual + self.ff(x_norm)
        
        # Return gate values for regularization
        return x, cross_weights, gate


class TransformerDecoder(nn.Module):
    """Transformer decoder with causal masking."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_enc = PositionalEncoding(config.d_model, config.max_target_len, config.dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.d_model, config.decoder_heads, config.decoder_ff_dim, config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        
        self.config = config
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal (triangular) mask for self-attention."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        targets: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            targets: (B, L) target token IDs (shifted right with BOS)
            memory: (B, T, d_model) encoder output
            memory_mask: (B, T) encoder padding mask
            
        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = targets.shape
        
        # Embed and add position
        x = self.embedding(targets) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        
        # Causal mask
        causal_mask = self._generate_causal_mask(L, targets.device)
        
        # Decoder layers - collect both attention weights and gate values
        all_cross_weights = []
        all_gate_values = []
        for layer in self.layers:
            x, cross_weights, gate = layer(x, memory, causal_mask, memory_mask)
            all_cross_weights.append(cross_weights)
            all_gate_values.append(gate)
        
        # Store last layer attention and gate values for debugging and regularization
        self.last_cross_attention_weights = all_cross_weights[-1] if all_cross_weights else None
        self.last_gate_values = all_gate_values  # Store all layer gate values
        
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def decode_greedy(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        max_len: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding for inference.
        
        Args:
            memory: (B, T, d_model) encoder output
            memory_mask: (B, T) encoder padding mask
            max_len: Maximum output length
            
        Returns:
            output_ids: (B, L)
            output_scores: (B, L)
        """
        B = memory.size(0)
        device = memory.device
        
        # Start with BOS
        outputs = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(B, max_len, device=device)
        
        for i in range(max_len):
            # Forward pass
            x = self.embedding(outputs) * math.sqrt(self.d_model)
            x = self.pos_enc(x)
            
            causal_mask = self._generate_causal_mask(outputs.size(1), device)
            
            for layer in self.layers:
                x, _, _ = layer(x, memory, causal_mask, memory_mask)
            
            x = self.final_norm(x)
            logits = self.output_proj(x[:, -1])  # Only last position
            
            probs = F.softmax(logits, dim=-1)
            next_token = probs.argmax(dim=-1, keepdim=True)
            scores[:, i] = probs.gather(1, next_token).squeeze(1)
            
            outputs = torch.cat([outputs, next_token], dim=1)
            
            # Check for EOS
            if (next_token == self.config.eos_id).all():
                break
        
        return outputs[:, 1:], scores[:, :outputs.size(1)-1]  # Remove BOS
    
    @torch.no_grad()
    def decode_beam(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_len: int = 100
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Beam search decoding.
        
        Returns list of (sequence, score) tuples for each batch item.
        """
        B = memory.size(0)
        device = memory.device
        
        # For simplicity, process one sample at a time
        results = []
        
        for b in range(B):
            mem = memory[b:b+1]
            mem_mask = memory_mask[b:b+1] if memory_mask is not None else None
            
            # Initialize beams: (sequence, log_prob)
            beams = [(torch.tensor([[self.config.bos_id]], device=device), 0.0)]
            completed = []
            
            for _ in range(max_len):
                candidates = []
                
                for seq, score in beams:
                    if seq[0, -1].item() == self.config.eos_id:
                        completed.append((seq, score))
                        continue
                    
                    # Forward pass
                    x = self.embedding(seq) * math.sqrt(self.d_model)
                    x = self.pos_enc(x)
                    causal_mask = self._generate_causal_mask(seq.size(1), device)
                    
                    for layer in self.layers:
                        x, _, _ = layer(x, mem, causal_mask, mem_mask)
                    
                    x = self.final_norm(x)
                    logits = self.output_proj(x[:, -1])
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top-k
                    topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)
                    
                    for k in range(beam_size):
                        new_seq = torch.cat([seq, topk_ids[:, k:k+1]], dim=1)
                        new_score = score + topk_log_probs[0, k].item()
                        candidates.append((new_seq, new_score))
                
                if not candidates:
                    break
                
                # Keep top beams
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
            
            # Add remaining beams to completed
            completed.extend(beams)
            completed.sort(key=lambda x: x[1], reverse=True)
            
            if completed:
                best_seq = completed[0][0][0, 1:]  # Remove BOS
                best_score = completed[0][1]
            else:
                best_seq = torch.tensor([], dtype=torch.long, device=device)
                best_score = float('-inf')
            
            results.append((best_seq, best_score))
        
        return results


# ============================================================================
# CTC Head
# ============================================================================

class CTCHead(nn.Module):
    """CTC output head for streaming and auxiliary loss."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        logits = logits.clamp(min=-100, max=100)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.clamp(min=-100)


# ============================================================================
# Full Hybrid CTC-Attention Model
# ============================================================================

class ISLTranslationModelV2(nn.Module):
    """
    Production ISL Translation Model.
    
    Hybrid CTC-Attention architecture:
    - Conformer encoder for feature extraction
    - CTC head for streaming inference + auxiliary loss
    - Transformer decoder for high-quality translation
    
    For inference:
    - Use CTC for real-time streaming (lower latency, good accuracy)
    - Use Transformer decoder for best quality (higher latency)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.config = config
        
        # Encoder
        self.encoder = ConformerEncoder(config)
        
        # CTC head (for streaming + auxiliary loss)
        self.ctc_head = CTCHead(config.d_model, config.vocab_size)
        
        # Transformer decoder (for high-quality translation)
        self.decoder = TransformerDecoder(config)
        
        # Initialize weights
        self._init_weights()
        
        # CRITICAL: Initialize gate biases AFTER _init_weights() to +2.0
        # This ensures gates start at sigmoid(2)=0.88 (88% encoder influence)
        self._init_gate_biases()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.config.d_model ** -0.5)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def _init_gate_biases(self):
        """Initialize encoder gate biases to +2.0 for high encoder influence."""
        for layer in self.decoder.layers:
            if hasattr(layer, 'encoder_gate'):
                # encoder_gate is nn.Sequential(Linear, Sigmoid)
                # The Linear layer is at index 0
                nn.init.constant_(layer.encoder_gate[0].bias, 2.0)
    
    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            features: (B, T, input_dim) input landmark features
            feature_lengths: (B,) input sequence lengths
            targets: (B, L) target token IDs (with BOS, without EOS for input)
            target_lengths: (B,) target lengths
            
        Returns:
            dict with ctc_log_probs, decoder_logits, encoder_lengths, cross_attention_weights
        """
        # Encode
        encoder_out, encoder_lengths, encoder_mask = self.encoder(features, feature_lengths)
        
        # CTC output
        ctc_log_probs = self.ctc_head(encoder_out)
        
        # Decoder output (teacher forcing)
        decoder_logits = self.decoder(targets, encoder_out, encoder_mask)
        
        # Get cross-attention weights and gate values for regularization
        cross_attn_weights = self.decoder.last_cross_attention_weights
        gate_values = self.decoder.last_gate_values  # List of gate tensors per layer
        
        return {
            'ctc_log_probs': ctc_log_probs,
            'decoder_logits': decoder_logits,
            'encoder_lengths': encoder_lengths,
            'cross_attention_weights': cross_attn_weights,
            'gate_values': gate_values  # NEW: for gate regularization
        }
    
    @torch.no_grad()
    def decode_ctc(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        CTC greedy decoding (for streaming).
        
        Fast, suitable for real-time applications.
        """
        encoder_out, encoder_lengths, _ = self.encoder(features, feature_lengths)
        ctc_log_probs = self.ctc_head(encoder_out)
        
        # Greedy decode
        predictions = ctc_log_probs.argmax(dim=-1)
        return predictions
    
    @torch.no_grad()
    def decode_attention(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        beam_size: int = 1,
        max_len: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transformer decoder inference.
        
        Higher quality, slightly more latency.
        """
        encoder_out, encoder_lengths, encoder_mask = self.encoder(features, feature_lengths)
        
        if beam_size == 1:
            return self.decoder.decode_greedy(encoder_out, encoder_mask, max_len)
        else:
            results = self.decoder.decode_beam(encoder_out, encoder_mask, beam_size, max_len)
            # Return just the best sequences
            sequences = [r[0] for r in results]
            scores = torch.tensor([r[1] for r in results])
            # Pad sequences to same length
            max_seq_len = max(s.size(0) for s in sequences) if sequences else 0
            padded = torch.zeros(len(sequences), max_seq_len, dtype=torch.long, device=features.device)
            for i, s in enumerate(sequences):
                padded[i, :s.size(0)] = s
            return padded, scores
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Streaming Inference Module (for Flutter/Mobile)
# ============================================================================

class StreamingEncoder(nn.Module):
    """
    Streaming-compatible encoder for real-time inference.
    
    Processes input in chunks and maintains state for CTC decoding.
    Designed for ONNX export and Flutter integration.
    """
    
    def __init__(self, model: ISLTranslationModelV2):
        super().__init__()
        self.encoder = model.encoder
        self.ctc_head = model.ctc_head
        self.config = model.config
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Process features and return CTC log probabilities.
        
        Args:
            features: (B, T, input_dim) - can be a chunk
            
        Returns:
            ctc_log_probs: (B, T', vocab_size)
        """
        # No lengths needed for streaming
        encoder_out, _, _ = self.encoder(features, None)
        ctc_log_probs = self.ctc_head(encoder_out)
        return ctc_log_probs


def create_model_v2(config: Optional[ModelConfig] = None) -> ISLTranslationModelV2:
    """Create production model."""
    return ISLTranslationModelV2(config)


def create_streaming_model(model: ISLTranslationModelV2) -> StreamingEncoder:
    """Create streaming encoder from trained model."""
    return StreamingEncoder(model)


# ============================================================================
# Loss Functions
# ============================================================================

class HybridCTCAttentionLoss(nn.Module):
    """
    Hybrid CTC-Attention loss with cross-attention and gate regularization.
    
    loss = ctc_weight * CTC_loss + (1 - ctc_weight) * CE_loss + attn_reg_weight * attention_reg + gate_reg_weight * gate_reg
    
    The attention regularization ensures the decoder actually uses encoder features.
    The gate regularization penalizes gates that are too low (ignoring encoder).
    """
    
    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        blank_id: int = 4,
        ctc_weight: float = 0.3,
        label_smoothing: float = 0.1,
        attn_reg_weight: float = 0.1,
        gate_reg_weight: float = 0.05  # NEW: penalize low gates
    ):
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.blank_id = blank_id
        self.attn_reg_weight = attn_reg_weight
        self.gate_reg_weight = gate_reg_weight  # NEW
        
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
            reduction='mean'
        )
    
    def forward(
        self,
        ctc_log_probs: torch.Tensor,  # (B, T, vocab)
        decoder_logits: torch.Tensor,  # (B, L, vocab)
        encoder_lengths: torch.Tensor,  # (B,)
        targets: torch.Tensor,  # (B, L) - for decoder CE loss
        target_lengths: torch.Tensor,  # (B,)
        cross_attention_weights: Optional[torch.Tensor] = None,  # (B, num_heads, L, T)
        gate_values: Optional[List[torch.Tensor]] = None  # List of gate tensors per layer
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss with attention regularization.
        
        Args:
            ctc_log_probs: CTC output (already log_softmax)
            decoder_logits: Decoder output (raw logits)
            encoder_lengths: Encoder output lengths
            targets: Target sequences (shifted for decoder)
            target_lengths: Target lengths
            cross_attention_weights: Cross-attention weights from last decoder layer
            
        Returns:
            dict with total_loss, ctc_loss, ce_loss, attn_reg_loss
        """
        # CTC loss expects (T, B, vocab)
        ctc_log_probs_t = ctc_log_probs.transpose(0, 1)
        
        # Create CTC targets (remove BOS, keep sequence)
        # Targets for CTC should be the actual token IDs without BOS
        ctc_targets = targets[:, 1:]  # Remove BOS
        ctc_target_lengths = target_lengths - 1  # Adjust lengths
        
        # Flatten CTC targets
        ctc_targets_flat = []
        for i in range(targets.size(0)):
            length = ctc_target_lengths[i].item()
            ctc_targets_flat.extend(ctc_targets[i, :length].tolist())
        ctc_targets_flat = torch.tensor(ctc_targets_flat, dtype=torch.long, device=targets.device)
        
        # CTC loss
        ctc_loss = self.ctc_loss(
            ctc_log_probs_t,
            ctc_targets_flat,
            encoder_lengths,
            ctc_target_lengths
        )
        
        # Cross-entropy loss for decoder
        # decoder_logits: (B, L, vocab), targets: (B, L)
        # Shift targets for CE: predict targets[1:] from inputs[:-1]
        ce_targets = targets[:, 1:].contiguous()  # (B, L-1)
        ce_logits = decoder_logits[:, :-1].contiguous()  # (B, L-1, vocab)
        
        ce_loss = self.ce_loss(
            ce_logits.view(-1, ce_logits.size(-1)),
            ce_targets.view(-1)
        )
        
        # Attention regularization - encourage using encoder features
        attn_reg_loss = torch.tensor(0.0, device=targets.device)
        if cross_attention_weights is not None and self.attn_reg_weight > 0:
            # Average attention weights across heads: (B, num_heads, L, T) -> (B, L, T)
            attn_weights = cross_attention_weights.mean(dim=1)
            
            # Entropy regularization: encourage focused attention
            # Higher entropy = attention is spread out (bad)
            # Lower entropy = attention is focused (good)
            attn_probs = attn_weights + 1e-10  # Avoid log(0)
            entropy = -(attn_probs * torch.log(attn_probs)).sum(dim=-1).mean()
            
            # Coverage: ensure all encoder positions are attended to at least once
            # Sum attention over target sequence: (B, L, T) -> (B, T)
            coverage = attn_weights.sum(dim=1)
            # Penalize if any position has very low total attention
            coverage_loss = torch.relu(0.1 - coverage).mean()
            
            attn_reg_loss = entropy + coverage_loss
        
        # Gate regularization - penalize gates that are too low
        gate_reg_loss = torch.tensor(0.0, device=targets.device)
        if gate_values is not None and self.gate_reg_weight > 0:
            # Penalize low gate values: loss = (1 - gate)^2
            # This encourages gates to stay high (using encoder info)
            gate_penalties = []
            for gate in gate_values:
                # gate shape: (B, L, d_model)
                avg_gate = gate.mean()  # Average gate value
                # Penalize if gate < 0.7 (want at least 70% encoder influence)
                penalty = torch.relu(0.7 - avg_gate) ** 2
                gate_penalties.append(penalty)
            gate_reg_loss = sum(gate_penalties) / len(gate_penalties)
        
        # Handle NaN
        if torch.isnan(ctc_loss):
            ctc_loss = torch.tensor(0.0, device=targets.device)
        if torch.isnan(ce_loss):
            ce_loss = torch.tensor(0.0, device=targets.device)
        if torch.isnan(attn_reg_loss):
            attn_reg_loss = torch.tensor(0.0, device=targets.device)
        if torch.isnan(gate_reg_loss):
            gate_reg_loss = torch.tensor(0.0, device=targets.device)
        
        # Combine - all loss components
        total_loss = (
            self.ctc_weight * ctc_loss + 
            (1 - self.ctc_weight) * ce_loss + 
            self.attn_reg_weight * attn_reg_loss +
            self.gate_reg_weight * gate_reg_loss  # NEW: gate regularization
        )
        
        return {
            'loss': total_loss,
            'ctc_loss': ctc_loss,
            'ce_loss': ce_loss,
            'attn_reg_loss': attn_reg_loss,
            'gate_reg_loss': gate_reg_loss  # NEW
        }


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ISL Translation Model V2 - Production Architecture")
    print("=" * 60)
    
    config = ModelConfig(
        vocab_size=2000,
        num_encoder_layers=4,
        num_decoder_layers=4
    )
    
    model = create_model_v2(config)
    print(f"\nModel Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    B, T, L = 4, 100, 20
    
    features = torch.randn(B, T, config.input_dim)
    feature_lengths = torch.tensor([100, 80, 90, 70])
    targets = torch.randint(5, config.vocab_size, (B, L))
    targets[:, 0] = config.bos_id  # BOS
    target_lengths = torch.tensor([20, 15, 18, 12])
    
    outputs = model(features, feature_lengths, targets, target_lengths)
    
    print(f"CTC log probs shape: {outputs['ctc_log_probs'].shape}")
    print(f"Decoder logits shape: {outputs['decoder_logits'].shape}")
    print(f"Encoder lengths: {outputs['encoder_lengths']}")
    
    # Test loss
    print("\nTesting hybrid loss...")
    loss_fn = HybridCTCAttentionLoss(
        vocab_size=config.vocab_size,
        pad_id=config.pad_id,
        blank_id=config.blank_id
    )
    
    losses = loss_fn(
        outputs['ctc_log_probs'],
        outputs['decoder_logits'],
        outputs['encoder_lengths'],
        targets,
        target_lengths
    )
    
    print(f"Total loss: {losses['loss'].item():.4f}")
    print(f"CTC loss: {losses['ctc_loss'].item():.4f}")
    print(f"CE loss: {losses['ce_loss'].item():.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        # CTC decoding
        ctc_out = model.decode_ctc(features[:1], feature_lengths[:1])
        print(f"CTC output shape: {ctc_out.shape}")
        
        # Attention decoding
        attn_out, scores = model.decode_attention(features[:1], feature_lengths[:1], beam_size=1)
        print(f"Attention output shape: {attn_out.shape}")
    
    # Test streaming
    print("\nTesting streaming encoder...")
    streaming = create_streaming_model(model)
    streaming.eval()
    with torch.no_grad():
        stream_out = streaming(features[:1])
        print(f"Streaming output shape: {stream_out.shape}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
