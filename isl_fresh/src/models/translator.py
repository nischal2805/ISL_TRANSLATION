"""ISL Translator with VideoMAE Transfer Learning"""
import torch
import torch.nn as nn
from .encoder import SignEncoder
from .decoder import TextDecoder


class ISLTranslator(nn.Module):
    """Full ISL to Text translation model with VideoMAE + CTC + Attention."""
    
    def __init__(
        self,
        hidden_dim: int = 256,  # Decoder dim
        vocab_size: int = 8000,
        decoder_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
        use_ctc: bool = True,
        pretrained: str = "MCG-NJU/videomae-base",
        freeze_epochs: int = 10,
        num_frames: int = 16  # VideoMAE frame count
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.num_frames = num_frames
        
        # VideoMAE encoder (processes actual video frames)
        self.encoder = SignEncoder(
            output_dim=hidden_dim,
            pretrained=pretrained,
            freeze_epochs=freeze_epochs,
            dropout=dropout,
            num_frames=num_frames
        )
        
        # Text decoder
        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_id=pad_id
        )
        
        # CTC head
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_head = nn.Linear(hidden_dim, vocab_size)
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id
    
    def forward(self, video_frames, feature_lengths, targets, target_lengths):
        """
        Args:
            video_frames: (B, num_frames, 3, H, W) - actual video frames
            feature_lengths: (B,) - frame counts (all same for VideoMAE)
            targets: (B, T_out) - target tokens
            target_lengths: (B,) - target lengths
        Returns:
            dict with 'logits', 'ctc_logits', 'encoder_lengths'
        """
        # Encode video frames
        encoder_out, enc_lengths = self.encoder(video_frames, feature_lengths)
        
        # Decode to text
        logits = self.decoder(targets, encoder_out, enc_lengths)
        
        output = {
            'logits': logits,
            'encoder_out': encoder_out,
            'encoder_lengths': enc_lengths
        }
        
        # CTC logits
        if self.use_ctc:
            output['ctc_logits'] = self.ctc_head(encoder_out)
        
        return output
    
    @torch.no_grad()
    def translate(self, video_frames, feature_lengths, max_len=100, bos_id=1, eos_id=2):
        """Translate sign video to text tokens."""
        self.eval()
        encoder_out, enc_lengths = self.encoder(video_frames, feature_lengths)
        return self.decoder.generate(encoder_out, enc_lengths, max_len, bos_id, eos_id)
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def on_epoch_start(self, epoch: int):
        """Unfreeze encoder after warmup epochs."""
        if epoch >= self.freeze_epochs and self.encoder._frozen:
            self.encoder.unfreeze()
            print(f"✓ Epoch {epoch}: Encoder unfrozen for fine-tuning")


def create_model(config=None):
    """Create model with default or custom config."""
    if config is None:
        config = {}
    
    return ISLTranslator(
        hidden_dim=config.get('hidden_dim', 256),
        vocab_size=config.get('vocab_size', 8000),
        decoder_layers=config.get('decoder_layers', 4),
        num_heads=config.get('num_heads', 4),
        ff_dim=config.get('ff_dim', 1024),
        dropout=config.get('dropout', 0.1),
        use_ctc=config.get('use_ctc', True),
        pretrained=config.get('pretrained', 'MCG-NJU/videomae-base'),
        freeze_epochs=config.get('freeze_epochs', 10),
        num_frames=config.get('num_frames', 16)
    )
