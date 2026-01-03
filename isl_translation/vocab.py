"""
ISL Translation System - Vocabulary
====================================
Character-level vocabulary for ISL to English translation.
"""

import math
from typing import Dict, List, Optional
from config import vocab_config


class Vocabulary:
    """
    Character-level vocabulary for ISL translation.
    
    Total: 35 tokens
    - 5 special tokens: <pad>, <sos>, <eos>, <unk>, space
    - 26 letters: a-z (lowercase)
    - 4 punctuation: . , ! ?
    """
    
    def __init__(self):
        # Build token to ID mapping
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Special tokens
        self.token2id[vocab_config.pad_token] = vocab_config.pad_id
        self.token2id[vocab_config.sos_token] = vocab_config.sos_id
        self.token2id[vocab_config.eos_token] = vocab_config.eos_id
        self.token2id[vocab_config.unk_token] = vocab_config.unk_id
        self.token2id[vocab_config.space_token] = vocab_config.space_id
        
        # Letters a-z (indices 5-30)
        for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            self.token2id[char] = vocab_config.char_offset + i
        
        # Punctuation (indices 31-34)
        for char, idx in vocab_config.punct_map.items():
            self.token2id[char] = idx
        
        # Build reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Special IDs for convenience
        self.pad_id = vocab_config.pad_id
        self.sos_id = vocab_config.sos_id
        self.eos_id = vocab_config.eos_id
        self.unk_id = vocab_config.unk_id
        self.blank_id = vocab_config.pad_id  # CTC blank = pad
    
    def __len__(self) -> int:
        return len(self.token2id)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
    
    @property
    def size(self) -> int:
        """Alias for vocab_size."""
        return len(self.token2id)
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert text to list of token IDs.
        
        Args:
            text: Input text string
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        # Handle NaN/None/float values
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = ""
        text = str(text).lower()
        
        ids = []
        
        if add_sos:
            ids.append(self.sos_id)
        
        for char in text:
            if char in self.token2id:
                ids.append(self.token2id[char])
            else:
                ids.append(self.unk_id)
        
        if add_eos:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            remove_special: Remove special tokens from output
            
        Returns:
            Decoded text string
        """
        special_ids = {self.pad_id, self.sos_id}  # Don't include eos_id here - check it first
        
        chars = []
        for id in ids:
            # Check EOS first before filtering specials (Bug #11 fix)
            if id == self.eos_id:
                break
            if remove_special and id in special_ids:
                continue
            if id in self.id2token:
                chars.append(self.id2token[id])
            else:
                chars.append('?')
        
        return ''.join(chars)
    
    def ctc_decode(self, ids: List[int]) -> str:
        """
        CTC decoding: collapse repeated tokens and remove blanks.
        
        Args:
            ids: List of token IDs from CTC output
            
        Returns:
            Decoded text string
        """
        # Remove consecutive duplicates
        collapsed = []
        prev_id = None
        for id in ids:
            if id != prev_id:
                collapsed.append(id)
                prev_id = id
        
        # Remove blank and special tokens (Bug #12 fix)
        special_for_ctc = {self.blank_id, self.sos_id, self.eos_id, self.pad_id}
        filtered = [id for id in collapsed if id not in special_for_ctc]
        
        # Decode to text
        return self.decode(filtered, remove_special=True)


# Global vocabulary instance
vocab = Vocabulary()


if __name__ == "__main__":
    # Test vocabulary
    print("ISL Vocabulary Test")
    print("=" * 50)
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Token to ID mapping (first 15):")
    for token, id in list(vocab.token2id.items())[:15]:
        print(f"  '{token}': {id}")
    
    # Test encoding/decoding
    test_text = "hello world!"
    encoded = vocab.encode(test_text, add_sos=True, add_eos=True)
    decoded = vocab.decode(encoded)
    
    print(f"\nOriginal: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test CTC decoding
    ctc_output = [0, 12, 12, 12, 9, 0, 0, 16, 16, 16, 16, 0, 12, 20, 0]
    ctc_decoded = vocab.ctc_decode(ctc_output)
    print(f"\nCTC output: {ctc_output}")
    print(f"CTC decoded: '{ctc_decoded}'")
