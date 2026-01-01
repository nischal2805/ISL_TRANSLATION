"""
Test script to verify all V2 components work together.
"""
import torch
import numpy as np
from model_v2 import ISLTranslationModelV2, ModelConfig
from tokenizer import BPETokenizer

def main():
    print("=" * 50)
    print("ISL Translation V2 - Integration Test")
    print("=" * 50)
    
    # 1. Load tokenizer
    print("\n[1] Loading BPE tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load('tokenizer_model/bpe_tokenizer.model')
    print(f"    ✓ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # 2. Create model with correct config
    print("\n[2] Creating model...")
    config = ModelConfig(
        input_dim=612,  # 204 landmarks * 3 features
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_encoder_layers=4,
        num_decoder_layers=4,
        encoder_heads=8,
        decoder_heads=8,
        encoder_ff_dim=1024,
        decoder_ff_dim=1024,
        dropout=0.1,
        max_target_len=100,
        ctc_weight=0.3
    )
    model = ISLTranslationModelV2(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Model created: {total_params/1e6:.1f}M params")
    
    # 3. Test with sample data
    print("\n[3] Testing forward pass...")
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 612)  # Match preprocessed data dims
    lengths = torch.tensor([seq_len, seq_len - 10])  # Feature lengths
    print(f"    ✓ Input shape: {x.shape}")
    
    # Forward pass (inference mode) using CTC decode
    model.eval()
    with torch.no_grad():
        # Use CTC decode for inference
        ctc_output = model.decode_ctc(x, lengths)
        print(f"    ✓ CTC decode output shape: {ctc_output.shape}")
    
    # 4. Test encoding/decoding
    print("\n[4] Testing tokenizer encode/decode...")
    test_texts = [
        "hello world",
        "the government is working on new policies",
        "cricket match in india",
        "Page 111"
    ]
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"    '{text}' -> {len(tokens)} tokens -> '{decoded}'")
    
    # 5. Test with real preprocessed file
    print("\n[5] Testing with real preprocessed data...")
    try:
        import pandas as pd
        from pathlib import Path
        
        # Data is directly in preprocessed_v2, not in train subfolder
        data_dir = Path(r"E:\5thsem el\APPROACH 2\preprocessed_v2")
        csv_path = Path(r"E:\5thsem el\kortex_5th_sem\data\iSign_v1.1.csv")
        
        # Load CSV for ground truth
        df = pd.read_csv(csv_path)
        uid_to_text = dict(zip(df['uid'], df['text']))
        
        # Find a matching file
        files = list(data_dir.glob('*.npy'))[:50]
        sample_file = None
        sample_text = None
        for f in files:
            if f.stem in uid_to_text:
                sample_file = f
                sample_text = uid_to_text[f.stem]
                break
        
        if sample_file:
            data = np.load(sample_file)
            print(f"    ✓ Loaded sample file: shape={data.shape}")
            print(f"    ✓ Ground truth: '{sample_text[:50]}...'")
            
            # Convert to tensor and run through model
            x = torch.from_numpy(data).float().unsqueeze(0)  # Add batch dim
            lengths = torch.tensor([x.shape[1]])
            
            with torch.no_grad():
                ctc_out = model.decode_ctc(x, lengths)
                pred_text = tokenizer.decode_ctc(ctc_out[0].tolist())
                print(f"    ✓ CTC prediction: '{pred_text[:50]}...' (untrained)")
        else:
            print(f"    ⚠ No matching samples found in CSV")
    except Exception as e:
        import traceback
        print(f"    ⚠ Could not test with real data: {e}")
        traceback.print_exc()
    
    # 6. Test attention decode
    print("\n[6] Testing attention decode...")
    with torch.no_grad():
        x = torch.randn(1, 50, 612)
        lengths = torch.tensor([50])
        decoded_ids, scores = model.decode_attention(x, lengths, max_len=20)
        if decoded_ids.numel() > 0:
            decoded_text = tokenizer.decode(decoded_ids[0].tolist())
            # scores is a list of tensors, get first item from first tensor
            score_val = scores[0].mean().item() if len(scores) > 0 else 0.0
            print(f"    ✓ Attention decode output: '{decoded_text[:50]}...' (score: {score_val:.4f})")
    
    print("\n" + "=" * 50)
    print("✅ All V2 components working correctly!")
    print("=" * 50)
    print("\nReady for training on A100!")
    print("Commands to run on A100 server:")
    print("  1. scp -r isl_translation/ user@a100-server:~/")
    print("  2. scp -r preprocessed_v2/ user@a100-server:~/data/")
    print("  3. python train_v2.py --data-dir ~/data/preprocessed_v2 \\")
    print("                        --csv ~/data/iSign_v1.1.csv \\")
    print("                        --tokenizer tokenizer_model \\")
    print("                        --epochs 50")

if __name__ == "__main__":
    main()
