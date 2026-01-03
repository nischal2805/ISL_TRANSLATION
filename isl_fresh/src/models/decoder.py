"""Text Decoder with Cross-Attention"""
import torch
import torch.nn as nn
import math


class TextDecoder(nn.Module):
    """Transformer decoder for text generation."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 100,
        pad_id: int = 0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id
        
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_enc = nn.Embedding(max_len, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, targets, encoder_out, encoder_lengths):
        """
        Args:
            targets: (B, T_out) - target token ids
            encoder_out: (B, T_enc, hidden_dim)
            encoder_lengths: (B,)
        Returns:
            logits: (B, T_out, vocab_size)
        """
        B, T = targets.size()
        device = targets.device
        
        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed(targets) + self.pos_enc(positions)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        
        # Padding masks
        tgt_pad_mask = targets == self.pad_id
        T_enc = encoder_out.size(1)
        memory_pad_mask = torch.arange(T_enc, device=device).expand(B, T_enc) >= encoder_lengths.unsqueeze(1)
        
        # Decode
        x = self.decoder(
            x, encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )
        
        return self.output(x)
    
    @torch.no_grad()
    def generate(self, encoder_out, encoder_lengths, max_len=100, bos_id=1, eos_id=2):
        """Greedy decoding."""
        B = encoder_out.size(0)
        device = encoder_out.device
        
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.forward(tokens, encoder_out, encoder_lengths)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if (next_token == eos_id).all():
                break
        
        return tokens
