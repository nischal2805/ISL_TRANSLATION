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
    def generate(self, encoder_out, encoder_lengths, max_len=100, bos_id=1, eos_id=2, 
                 temperature=1.0, repetition_penalty=1.2, top_k=50, top_p=0.9):
        """Sampling-based decoding with repetition penalty and nucleus sampling."""
        B = encoder_out.size(0)
        device = encoder_out.device
        
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for step in range(max_len - 1):
            logits = self.forward(tokens, encoder_out, encoder_lengths)
            next_logits = logits[:, -1] / temperature  # (B, vocab)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and step > 0:
                for b in range(B):
                    for token_id in set(tokens[b].tolist()):
                        if token_id in [bos_id, eos_id, 0]:  # Don't penalize special tokens
                            continue
                        # Penalize repeated tokens
                        if next_logits[b, token_id] > 0:
                            next_logits[b, token_id] /= repetition_penalty
                        else:
                            next_logits[b, token_id] *= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least 1 token
                sorted_indices_to_remove[..., 0] = False
                
                for b in range(B):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            if (next_token == eos_id).all():
                break
        
        return tokens
