"""ISL Translator with VideoMAE Transfer Learning"""
import torch
import torch.nn as nn
from .encoder import SignEncoder
from .decoder import TextDecoder


class ISLTranslator(nn.Module):
    """Full ISL to Text translation model with VideoMAE + CTC + Attention."""
    
    def __init__(
        self,
        input_dim: int = 603,
        hidden_dim: int = 256,  # Decoder dim
        encoder_hidden: int = 768,  # VideoMAE dim
        vocab_size: int = 8000,
        decoder_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        use_ctc: bool = True,
        pretrained: str = "MCG-NJU/videomae-base",
        freeze_epochs: int = 5
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        
        self.encoder = SignEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden,
            output_dim=hidden_dim,
            pretrained=pretrained,
            freeze_epochs=freeze_epochs,
            dropout=dropout
        )
        
        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=pad_id
        )
        
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id
    
    def forward(self, features, feature_lengths, targets, target_lengths):
        """
        Args:
            features: (B, T, input_dim)
            feature_lengths: (B,)
            targets: (B, T_out)
            target_lengths: (B,)
        Returns:
            dict with 'logits', 'ctc_logits', 'encoder_lengths'
        """
        # Encode
        encoder_out, enc_lengths = self.encoder(features, feature_lengths)
        
        # Decode
        logits = self.decoder(targets, encoder_out, enc_lengths)
        
        output = {
            'logits': logits,
            'encoder_out': encoder_out,
            'encoder_lengths': enc_lengths
        }
        
        # CTC
        if self.use_ctc:
            output['ctc_logits'] = self.ctc_head(encoder_out)
        
        return output
    
    @torch.no_grad()
    def translate(self, features, feature_lengths, max_len=100, bos_id=1, eos_id=2):
        """Translate sign sequence to text tokens."""
        self.eval()
        encoder_out, enc_lengths = self.encoder(features, feature_lengths)
        return self.decoder.generate(encoder_out, enc_lengths, max_len, bos_id, eos_id)
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def on_epoch_start(self, epoch: int):
        """Unfreeze encoder after warmup epochs."""
        if epoch >= self.freeze_epochs and self.encoder._frozen:
            self.encoder.unfreeze()


def create_model(config=None):
    """Create model with default or custom config."""
    if config is None:
        config = {}
    
    return ISLTranslator(
        input_dim=config.get('input_dim', 603),
        hidden_dim=config.get('hidden_dim', 256),
        vocab_size=config.get('vocab_size', 8000),
        encoder_layers=config.get('encoder_layers', 4),
        decoder_layers=config.get('decoder_layers', 4),
        num_heads=config.get('num_heads', 4),
        ff_dim=config.get('ff_dim', 1024),
        dropout=config.get('dropout', 0.1),
        use_ctc=config.get('use_ctc', True)
    )
