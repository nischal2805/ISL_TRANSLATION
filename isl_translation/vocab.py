"""
ISL Translation System - Vocabulary
====================================
Word-level vocabulary for ISL to English translation.
Builds vocabulary from the iSign_v1.1.csv dataset.
"""

import csv
import re
import os
from typing import Dict, List, Optional
from pathlib import Path


# Special token constants
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_ID = 0  # Also used as CTC blank
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class Vocabulary:
    """
    Word-level vocabulary for ISL translation.
    
    Builds vocabulary from the dataset CSV file where each unique word
    in the sentences is mapped to a unique ID.
    
    Special tokens:
    - <pad> (0): Padding token, also CTC blank
    - <sos> (1): Start of sequence
    - <eos> (2): End of sequence  
    - <unk> (3): Unknown word
    
    Words start from index 4.
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize vocabulary from CSV file.
        
        Args:
            csv_path: Path to iSign_v1.1.csv. If None, searches common locations.
        """
        # Build token to ID mapping
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        
        # Add special tokens first
        self.token2id[PAD_TOKEN] = PAD_ID
        self.token2id[SOS_TOKEN] = SOS_ID
        self.token2id[EOS_TOKEN] = EOS_ID
        self.token2id[UNK_TOKEN] = UNK_ID
        
        # Special IDs for convenience
        self.pad_id = PAD_ID
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.blank_id = PAD_ID  # CTC blank = pad
        
        # Find and load vocabulary from CSV
        if csv_path is None:
            csv_path = self._find_csv_path()
        
        if csv_path and os.path.exists(csv_path):
            self._build_vocab_from_csv(csv_path)
        else:
            print(f"Warning: CSV file not found. Vocabulary will only have special tokens.")
        
        # Build reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
    
    def _find_csv_path(self) -> Optional[str]:
        """Search for CSV file in common locations."""
        possible_paths = [
            # Relative paths from isl_translation directory
            "../iSign_v1.1.csv",
            "../../iSign_v1.1.csv",
            # Absolute paths for GPU server
            "/media/rvcse22/CSERV/kortex_sem5/ramita/iSign_v1.1.csv",
            # Windows paths
            "G:/kortex 5th sem/03-dec approach/ISL_TRANSLATION/iSign_v1.1.csv",
        ]
        
        # Get script directory for relative paths
        script_dir = Path(__file__).parent
        
        for path in possible_paths:
            if path.startswith("/") or path.startswith("G:"):
                full_path = path
            else:
                full_path = str(script_dir / path)
            
            if os.path.exists(full_path):
                return full_path
        
        return None
    
    def _build_vocab_from_csv(self, csv_path: str):
        """
        Extract unique words from CSV and build vocabulary.
        
        Args:
            csv_path: Path to the CSV file with 'text' column
        """
        words = set()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get('text', '')                # Handle NaN/None values
                if text is None or (isinstance(text, float)) or text == 'nan':
                    continue
                if not isinstance(text, str):
                    text = str(text)                # Extract words (lowercase, only alphabetic)
                extracted = re.findall(r'[a-zA-Z]+', text.lower())
                words.update(extracted)
        
        # Sort words for consistent ordering
        sorted_words = sorted(words)
        
        # Assign IDs starting from 4 (after special tokens)
        word_offset = 4
        for i, word in enumerate(sorted_words):
            self.token2id[word] = word_offset + i
        
        print(f"Vocabulary built: {len(self.token2id)} tokens ({len(sorted_words)} words + 4 special)")
    
    def __len__(self) -> int:
        return len(self.token2id)
    
    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
    
    @property
    def size(self) -> int:
        """Alias for vocab_size."""
        return len(self.token2id)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of word tokens
        """
        # Handle non-string inputs (e.g., NaN values from CSV)
        if not isinstance(text, str):
            if text is None or (isinstance(text, float) and str(text) == 'nan'):
                return []
            text = str(text)
        
        # Extract words (lowercase, only alphabetic)
        words = re.findall(r'[a-zA-Z]+', text.lower())
        return words
    
    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Convert text to list of token IDs (word-level).
        
        Args:
            text: Input text string
            add_sos: Add start-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        ids = []
        
        if add_sos:
            ids.append(self.sos_id)
        
        # Tokenize into words
        words = self.tokenize(text)
        
        for word in words:
            if word in self.token2id:
                ids.append(self.token2id[word])
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
            Decoded text string (words joined by spaces)
        """
        special_ids = {self.pad_id, self.sos_id}
        
        words = []
        for id in ids:
            # Check EOS first
            if id == self.eos_id:
                break
            if remove_special and id in special_ids:
                continue
            if id in self.id2token:
                words.append(self.id2token[id])
            else:
                words.append('<?>') 
        
        return ' '.join(words)
    
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
        
        # Remove blank and special tokens
        special_for_ctc = {self.blank_id, self.sos_id, self.eos_id, self.pad_id}
        filtered = [id for id in collapsed if id not in special_for_ctc]
        
        # Decode to text
        return self.decode(filtered, remove_special=True)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            for token, id in sorted(self.token2id.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{id}\n")
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        vocab = cls.__new__(cls)
        vocab.token2id = {}
        vocab.id2token = {}
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token, id = line.strip().split('\t')
                vocab.token2id[token] = int(id)
        
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab.pad_id = PAD_ID
        vocab.sos_id = SOS_ID
        vocab.eos_id = EOS_ID
        vocab.unk_id = UNK_ID
        vocab.blank_id = PAD_ID
        
        return vocab


# Global vocabulary instance
vocab = Vocabulary()


if __name__ == "__main__":
    # Test vocabulary
    print("ISL Word-Level Vocabulary Test")
    print("=" * 50)
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"\nSpecial tokens:")
    for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
        print(f"  '{token}': {vocab.token2id[token]}")
    
    print(f"\nSample word mappings (first 20 words):")
    word_items = [(k, v) for k, v in vocab.token2id.items() if v >= 4][:20]
    for token, id in word_items:
        print(f"  '{token}': {id}")
    
    # Test encoding/decoding
    test_text = "Hello world! How are you today?"
    encoded = vocab.encode(test_text, add_sos=True, add_eos=True)
    decoded = vocab.decode(encoded)
    
    print(f"\nOriginal: '{test_text}'")
    print(f"Tokenized: {vocab.tokenize(test_text)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Test with sentence from dataset
    test_text2 = "Make it shorter."
    encoded2 = vocab.encode(test_text2, add_sos=True, add_eos=True)
    decoded2 = vocab.decode(encoded2)
    
    print(f"\nDataset sentence: '{test_text2}'")
    print(f"Encoded: {encoded2}")
    print(f"Decoded: '{decoded2}'")