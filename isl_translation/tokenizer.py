"""
BPE Tokenizer for ISL Translation System
=========================================
Uses SentencePiece for subword tokenization.
Much better than character-level for translation quality.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    print("Warning: sentencepiece not installed. Run: pip install sentencepiece")


class BPETokenizer:
    """
    BPE Tokenizer using SentencePiece.
    
    Advantages over character-level:
    - Captures word/subword meanings
    - Much smaller sequence lengths (10 words → 10-15 tokens vs 60 chars)
    - Better generalization to unseen words
    - Standard in modern NLP (BERT, GPT, T5 all use this)
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    BLANK_TOKEN = "<blank>"  # For CTC
    
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    BLANK_ID = 4
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 2000,
        model_type: str = "bpe"
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to trained sentencepiece model (.model file)
            vocab_size: Target vocabulary size (only used during training)
            model_type: "bpe" or "unigram"
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.model_path = model_path
        self.sp_model = None
        
        # Token mappings (will be populated after loading model)
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(
        self,
        texts: List[str],
        output_dir: str,
        model_prefix: str = "bpe_tokenizer"
    ) -> str:
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: List of text strings to train on
            output_dir: Directory to save the model
            model_prefix: Prefix for model files
            
        Returns:
            Path to trained model
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required: pip install sentencepiece")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Write texts to temporary file
        train_file = os.path.join(output_dir, "train_texts.txt")
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                # Clean and normalize text
                text = self._preprocess_text(text)
                if text.strip():
                    f.write(text + '\n')
        
        # Train SentencePiece model
        model_prefix_path = os.path.join(output_dir, model_prefix)
        
        # Reserve IDs 0-4 for special tokens
        spm.SentencePieceTrainer.train(
            input=train_file,
            model_prefix=model_prefix_path,
            vocab_size=self.vocab_size - 5,  # Reserve 5 for special tokens
            model_type=self.model_type,
            pad_id=-1,  # We'll handle special tokens ourselves
            unk_id=0,
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=[],
            character_coverage=0.9995,
            num_threads=os.cpu_count() or 4,
            train_extremely_large_corpus=False,
            max_sentence_length=4192,
            shuffle_input_sentence=True,
        )
        
        self.model_path = model_prefix_path + ".model"
        self.load(self.model_path)
        
        # Save config
        config = {
            'vocab_size': self.size,
            'model_type': self.model_type,
            'special_tokens': {
                'pad': self.PAD_ID,
                'unk': self.UNK_ID,
                'bos': self.BOS_ID,
                'eos': self.EOS_ID,
                'blank': self.BLANK_ID
            }
        }
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Tokenizer trained!")
        print(f"  Vocabulary size: {self.size}")
        print(f"  Model saved to: {self.model_path}")
        print(f"  Config saved to: {config_path}")
        
        # Cleanup temp file
        os.remove(train_file)
        
        return self.model_path
    
    def load(self, model_path: str):
        """Load trained tokenizer."""
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required: pip install sentencepiece")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        self.model_path = model_path
        
        # Build token mappings with special tokens at the start
        self._token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.BLANK_TOKEN: self.BLANK_ID,
        }
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        
        # Add SentencePiece tokens (offset by 5)
        for i in range(self.sp_model.get_piece_size()):
            token = self.sp_model.id_to_piece(i)
            token_id = i + 5  # Offset for special tokens
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Keep basic punctuation but normalize
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        return text
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        text = self._preprocess_text(text)
        
        # Get SentencePiece IDs and offset by 5
        sp_ids = self.sp_model.encode(text)
        token_ids = [idx + 5 for idx in sp_ids]
        
        # Add special tokens
        if add_bos:
            token_ids = [self.BOS_ID] + token_ids
        if add_eos:
            token_ids = token_ids + [self.EOS_ID]
        
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.EOS_ID]
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            special_ids = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID, self.BLANK_ID}
            token_ids = [tid for tid in token_ids if tid not in special_ids]
        
        # Convert back to SentencePiece IDs (offset by -5)
        sp_ids = []
        for tid in token_ids:
            if tid >= 5:  # Not a special token
                sp_ids.append(tid - 5)
        
        # Decode
        text = self.sp_model.decode(sp_ids)
        return text
    
    def decode_ctc(self, token_ids: List[int]) -> str:
        """
        Decode CTC output (handles blanks and repetitions).
        
        CTC outputs may have:
        - Repeated tokens (collapse them)
        - Blank tokens (remove them)
        
        Args:
            token_ids: Raw CTC output token IDs
            
        Returns:
            Decoded text
        """
        # Remove blanks and collapse repetitions
        filtered = []
        prev_id = None
        
        for tid in token_ids:
            if tid == self.BLANK_ID:
                prev_id = None
                continue
            if tid == prev_id:
                continue
            if tid in {self.PAD_ID}:
                continue
            filtered.append(tid)
            prev_id = tid
        
        return self.decode(filtered, skip_special_tokens=True)
    
    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        if self.sp_model is None:
            return 5  # Just special tokens
        return self.sp_model.get_piece_size() + 5
    
    @property
    def pad_id(self) -> int:
        return self.PAD_ID
    
    @property
    def blank_id(self) -> int:
        return self.BLANK_ID
    
    @property
    def bos_id(self) -> int:
        return self.BOS_ID
    
    @property
    def eos_id(self) -> int:
        return self.EOS_ID
    
    def get_vocab(self) -> Dict[str, int]:
        """Get full vocabulary mapping."""
        return self._token_to_id.copy()
    
    def save(self, output_dir: str):
        """Save tokenizer to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy model file
        if self.model_path:
            import shutil
            dest = os.path.join(output_dir, "tokenizer.model")
            shutil.copy(self.model_path, dest)
        
        # Save config
        config = {
            'vocab_size': self.size,
            'model_type': self.model_type,
            'model_file': 'tokenizer.model',
            'special_tokens': {
                'pad': self.PAD_ID,
                'unk': self.UNK_ID,
                'bos': self.BOS_ID,
                'eos': self.EOS_ID,
                'blank': self.BLANK_ID
            }
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_dir: str) -> "BPETokenizer":
        """Load tokenizer from directory."""
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        model_path = os.path.join(model_dir, "tokenizer.model")
        
        if not os.path.exists(model_path):
            # Try alternate naming
            model_path = os.path.join(model_dir, "bpe_tokenizer.model")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        tokenizer = cls(
            model_path=model_path,
            vocab_size=config.get('vocab_size', 2000),
            model_type=config.get('model_type', 'bpe')
        )
        return tokenizer


def train_tokenizer_on_dataset(
    csv_path: str,
    output_dir: str,
    vocab_size: int = 2000,
    text_column: str = 'text'
) -> BPETokenizer:
    """
    Train BPE tokenizer on dataset CSV.
    
    Args:
        csv_path: Path to CSV with text data
        output_dir: Where to save tokenizer
        vocab_size: Target vocabulary size
        text_column: Name of text column in CSV
        
    Returns:
        Trained tokenizer
    """
    import pandas as pd
    
    print(f"Loading texts from {csv_path}...")
    df = pd.read_csv(csv_path)
    texts = df[text_column].dropna().tolist()
    print(f"Loaded {len(texts)} texts")
    
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, output_dir)
    
    # Test encoding/decoding
    test_text = texts[0]
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encode/decode:")
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    print(f"  Token count: {len(encoded)} (vs {len(test_text)} chars)")
    
    return tokenizer


# ============================================================================
# Main - Train tokenizer
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--csv', type=str, 
                       default=r'E:\5thsem el\kortex_5th_sem\data\iSign_v1.1.csv',
                       help='Path to CSV with text data')
    parser.add_argument('--output', type=str,
                       default=r'E:\5thsem el\APPROACH 2\isl_translation\tokenizer_model',
                       help='Output directory for tokenizer')
    parser.add_argument('--vocab-size', type=int, default=2000,
                       help='Vocabulary size')
    
    args = parser.parse_args()
    
    tokenizer = train_tokenizer_on_dataset(
        csv_path=args.csv,
        output_dir=args.output,
        vocab_size=args.vocab_size
    )
