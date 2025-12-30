# **🏗️ COMPLETE ISL TRANSLATION SYSTEM ARCHITECTURE**

## **This project is an Indian sign language translator app and the following the architecture for training the model for the same which has to be run on an edge device or basically a mobile device and the following is the training methodology and implementation plan for the model , i am using the isign dataset with around 127k videos and gloss text .**

## **Low-Level Design Document**

---

## **📋 TABLE OF CONTENTS**

1. [System Overview](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#1-system-overview)  
2. [Data Pipeline Architecture](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#2-data-pipeline-architecture)  
3. [Preprocessing Pipeline](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#3-preprocessing-pipeline)  
4. [Model Architecture](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#4-model-architecture)  
5. [Training Strategy](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#5-training-strategy)  
6. [Hyperparameters](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#6-hyperparameters)  
7. [Implementation Roadmap](https://claude.ai/chat/bc7abefb-eb02-4ed9-89e0-e4aa1001b84e#7-implementation-roadmap)

---

## **1\. SYSTEM OVERVIEW**

### **1.1 High-Level Flow**

RAW VIDEO (ISign Dataset)  
    ↓  
\[MediaPipe Landmark Extraction\] → 75 landmarks WITHOUT face (Phase 1)  
    ↓  
\[Select 46 Keypoints\] → Hands (42) + Upper Body (4)  
    ↓  
\[Feature Engineering\] → Position \+ Velocity \+ Acceleration  
    ↓  
\[Normalization & Augmentation\] → Scale-invariant features  
    ↓  
\[ENCODER: Video → Hidden Representation\]  
    ↓  
\[CTC HEAD: Character predictions\] (for streaming)  
    ↓  
\[GRU DECODER: Hidden → English Text\] (final polished output)  
    ↓  
ENGLISH TRANSLATION

NOTE: No gloss annotations - direct video-to-text translation

### **1.2 Two-Stage Architecture Rationale**

**Why two stages (Video→Gloss→Text) vs End-to-End (Video→Text)?**

❌ **Your Original Plan:** Video → Gloss → Text (two separate models) ✅ **Recommended:** Video → Text (single model with intermediate CTC supervision)

**Reasoning:**

* **Single model is better because:**

  * ISign dataset doesn't have gloss annotations (only video \+ text)  
  * Glosses are expensive to annotate manually  
  * Error propagation: mistakes in gloss prediction hurt text output  
  * Simpler deployment: one model instead of two  
* **We still use CTC but differently:**

  * CTC predicts character sequences directly (not glosses)  
  * Acts as auxiliary loss to help encoder learn better features  
  * Provides streaming capability for real-time predictions

---

## **2\. DATA PIPELINE ARCHITECTURE**

### **2.1 ISign Dataset Structure**

**Expected Format:**

iSign\_dataset/  
├── videos/  
│   ├── 00001.mp4  
│   ├── 00002.mp4  
│   └── ...  
├── annotations.csv  
    Columns: video\_id, text, signer\_id, duration

### **2.2 Preprocessing Stages**

#### **Stage 1: Landmark Extraction**

**Input:** Raw video (MP4, 30 FPS, variable resolution) **Output:** Numpy array (T, 543, 3\) where T \= number of frames

**MediaPipe Configuration:**

MediaPipe Settings:  
  Model: holistic  
  Min detection confidence: 0.5  
  Min tracking confidence: 0.5  
    
Landmark Breakdown:  
  \- Pose: 33 landmarks × 3 coords \= 99 features  
  \- Left hand: 21 landmarks × 3 coords \= 63 features  
  \- Right hand: 21 landmarks × 3 coords \= 63 features  
  \- Face: 468 landmarks × 3 coords \= 1404 features (optional)  
    
Total (without face): 33 \+ 21 \+ 21 \= 75 landmarks \= 225 coords  
Total (with face): 33 \+ 21 \+ 21 \+ 468 \= 543 landmarks \= 1629 coords

**CRITICAL DECISION: Do we need face landmarks?**

For ISL (Indian Sign Language):

* **YES if:** Dataset has signs with facial expressions (questions, emotions)  
* **NO if:** Focusing on manual signs only (saves 86% of features\!)

**My Recommendation:** Start WITHOUT face landmarks

* Reduces dimension from 1629 → 225 (7x smaller)  
* Much faster training and inference  
* Add face later if accuracy plateaus

---

## **3\. PREPROCESSING PIPELINE**

### **3.1 Feature Selection**

**Recommended Landmark Subset (138 dimensions):**

Selected Landmarks:  
  1\. Both Hands (CRITICAL): 42 landmarks × 3 \= 126 dims  
     \- All 21 hand landmarks per hand  
     \- Captures finger positions, hand shapes  
    
  2\. Upper Body (CONTEXT): 4 landmarks × 3 \= 12 dims  
     \- Left shoulder, Right shoulder  
     \- Left elbow, Right elbow  
     \- Provides signing space reference  
    
Total: 138 position dimensions

**Why not full pose?**

* Hips, legs, feet: irrelevant for sign language  
* Face: adds 1404 dims but ISL is less face-dependent than ASL  
* Shoulders \+ elbows: sufficient for spatial context

### **3.2 Normalization (Critical for Generalization)**

**Goal:** Make features invariant to:

* Signer distance from camera  
* Camera position  
* Signer body size

**Normalization Steps:**

STEP 1: Hand Normalization (per hand, per frame)  
────────────────────────────────────────────────  
For each hand:  
  1.1. Center to wrist (landmark 0\)  
       hand\_landmarks \= hand\_landmarks \- wrist\_position  
       → Translation invariant  
    
  1.2. Scale by hand span  
       span \= distance(wrist, middle\_finger\_tip)  
       hand\_landmarks \= hand\_landmarks / span  
       → Scale invariant  
    
Result: Each hand in normalized \[-1, 1\]³ space

STEP 2: Body Normalization  
────────────────────────────────────────────────  
  2.1. Center to torso midpoint  
       midpoint \= (left\_shoulder \+ right\_shoulder) / 2  
       shoulders \= shoulders \- midpoint  
       elbows \= elbows \- midpoint  
    
  2.2. Scale by shoulder width  
       width \= distance(left\_shoulder, right\_shoulder)  
       shoulders \= shoulders / width  
       elbows \= elbows / width  
    
Result: Body in normalized space

STEP 3: Concatenate  
────────────────────────────────────────────────  
normalized\_frame \= \[left\_hand, right\_hand, body\]  
                 \= \[63, 63, 12\] \= 138 dimensions

### **3.3 Motion Features (Velocity \+ Acceleration)**

**Why add derivatives?**

* Position alone: "where is the hand?"  
* Velocity: "how fast is the hand moving?"  
* Acceleration: "is the movement sharp or smooth?"

**Computation:**

Given position sequence P \= \[p₀, p₁, p₂, ..., p\_T\]  
Each p\_t is (138,) vector

VELOCITY (First Derivative):  
────────────────────────────────────────────────  
v\_t \= p\_t \- p\_{t-1}  for t \> 0  
v\_0 \= 0  (or v\_1, doesn't matter much)

Output: V \= \[v₀, v₁, ..., v\_T\] shape (T, 138\)

ACCELERATION (Second Derivative):  
────────────────────────────────────────────────  
a\_t \= v\_t \- v\_{t-1}  for t \> 0  
a\_0 \= 0

Output: A \= \[a₀, a₁, ..., a\_T\] shape (T, 138\)

FINAL FEATURES:  
────────────────────────────────────────────────  
features \= concatenate(\[P, V, A\], axis=1)  
         \= (T, 414\)

Interpretation:  
  \- dims 0-137: Hand/body positions  
  \- dims 138-275: Movement speeds  
  \- dims 276-413: Movement sharpness

### **3.4 Smoothing (Reduce MediaPipe Jitter)**

**Problem:** MediaPipe landmarks are noisy frame-to-frame

**Solution:** Gaussian smoothing along time axis

Smoothing Parameters:  
  \- Method: 1D Gaussian filter  
  \- Sigma: 1.0 (frames)  
  \- Mode: 'nearest' (edge handling)  
    
Apply to: Position features BEFORE computing velocity  
Why: Smoothing reduces noise in derivatives

Algorithm:  
  1\. Smooth position: P\_smooth \= gaussian\_filter(P, sigma=1.0)  
  2\. Compute velocity: V \= diff(P\_smooth)  
  3\. Smooth velocity: V\_smooth \= gaussian\_filter(V, sigma=1.0)  
  4\. Compute acceleration: A \= diff(V\_smooth)

### **3.5 Final Preprocessing Output**

**Per Video:**

Input: video.mp4 (3-5 seconds)  
Output: features.npy shape (T, 414\) dtype float32

T \= number of frames (typically 90-150 for 3-5 sec at 30fps)  
414 \= 138 position \+ 138 velocity \+ 138 acceleration

---

## **4\. MODEL ARCHITECTURE**

### **4.1 Complete Architecture Diagram**

INPUT: (batch, time, 414\)  
         ↓  
    ┌────────────────────────────────────┐  
    │   INPUT PROJECTION                 │  
    │   Linear(414 → 384\)                │  
    │   LayerNorm                        │  
    └────────────────────────────────────┘  
         ↓ (batch, time, 384\)  
    ┌────────────────────────────────────┐  
    │   MULTI-SCALE CNN (4 layers)       │  
    │   Captures temporal patterns       │  
    │   \- Layer 1: 384→384, kernel=3,5,7,11 │  
    │   \- Layer 2: 384→384, kernel=3,5,7,11 │  
    │   \- Squeeze-Excitation blocks      │  
    │   \- Residual connections           │  
    └────────────────────────────────────┘  
         ↓ (batch, time, 384\)  
    ┌────────────────────────────────────┐  
    │   TEMPORAL SUBSAMPLING             │  
    │   Stride-2 conv: T → T/2          │  
    │   Reduces sequence length 2x       │  
    └────────────────────────────────────┘  
         ↓ (batch, time/2, 384\)  
    ┌────────────────────────────────────┐  
    │   POSITIONAL ENCODING              │  
    │   Sinusoidal (learnable optional)  │  
    └────────────────────────────────────┘  
         ↓ (batch, time/2, 384\)  
    ┌────────────────────────────────────┐  
    │   CONFORMER ENCODER (2 blocks)     │  
    │   Each block:                      │  
    │   \- Feed-forward (50%)             │  
    │   \- Multi-head self-attention      │  
    │   \- Convolution module             │  
    │   \- Feed-forward (50%)             │  
    │   \- Residual \+ LayerNorm           │  
    └────────────────────────────────────┘  
         ↓ (batch, time/2, 384\)  
         ├──────────────────┬─────────────────┐  
         │                  │                 │  
         ↓                  ↓                 ↓  
    ┌────────┐      ┌───────────┐     ┌──────────┐  
    │  CTC   │      │    GRU    │     │Transform.│  
    │  HEAD  │      │  DECODER  │     │ DECODER  │  
    └────────┘      └───────────┘     └──────────┘  
         ↓                ↓                   ↓  
    Streaming       Fast output        Best accuracy  
    prediction      (\~40 FPS)          (\~25 FPS)

### **4.2 Layer-by-Layer Specification**

#### **LAYER 1: Input Projection**

Purpose: Project 414 dims to model hidden dimension  
Architecture:  
  \- Linear(414, 384\)  
  \- LayerNorm(384)  
  \- Dropout(0.2)

Output: (batch, time, 384\)  
Parameters: 414 × 384 \+ 384 \= 159,480 params

#### **LAYER 2: Multi-Scale CNN**

**Why multi-scale?**

* Different signs have different speeds  
* Kernel 3: Fast, sharp gestures  
* Kernel 5: Medium speed transitions  
* Kernel 7: Slow, smooth movements  
* Kernel 11: Context across multiple gestures

Layer 2.1: Multi-Scale Conv Block 1  
────────────────────────────────────  
Input: (batch, 384, time)  \# Transpose for Conv1d

Four parallel branches:  
  Branch 1: Conv1d(384, 96, kernel=3, padding=1)  
  Branch 2: Conv1d(384, 96, kernel=5, padding=2)  
  Branch 3: Conv1d(384, 96, kernel=7, padding=3)  
  Branch 4: Conv1d(384, 96, kernel=11, padding=5)  
    
Concatenate: 96×4 \= 384 channels  
BatchNorm1d(384)  
ReLU  
Squeeze-Excitation(384, reduction=4)  
  \- Learns which channels are important  
  \- Adds channel attention

Residual: output \= SE(conv(x)) \+ x

Layer 2.2: Multi-Scale Conv Block 2  
────────────────────────────────────  
Same as 2.1 (stacking for more capacity)

Output: (batch, time, 384\)  
Parameters: \~1.5M params per block × 2 \= 3M total

**Squeeze-Excitation Block (Channel Attention):**

Input: (batch, 384, time)  
  ↓  
Global Average Pool: (batch, 384, time) → (batch, 384\)  
  ↓  
FC1: Linear(384, 96\)  \# Reduction by 4  
  ↓  
ReLU  
  ↓  
FC2: Linear(96, 384\)  
  ↓  
Sigmoid → Channel weights: (batch, 384\)  
  ↓  
Multiply: input × weights.unsqueeze(2)

Effect: Model learns which channels matter most  
Example: "hand shape channels" get higher weights than "noise channels"

#### **LAYER 3: Temporal Subsampling**

**Purpose:** Reduce sequence length for computational efficiency

Configuration:  
  Input: (batch, time, 384\)  
    
  Subsample by 2x (recommended):  
    Conv1d(384, 384, kernel=3, stride=2, padding=1)  
    BatchNorm1d(384)  
    ReLU  
    Dropout(0.1)  
    
  Output: (batch, time/2, 384\)  
    
Effect:  
  \- 150 frames → 75 frames (2x faster)  
  \- Receptive field effectively doubles  
  \- Minimal accuracy loss (\<1%)

Alternative (if accuracy critical):  
  \- Subsample by 1x (no subsampling)  
  \- Slower but preserves all temporal detail

#### **LAYER 4: Positional Encoding**

**Why needed?** Attention has no notion of position

Type: Sinusoidal Positional Encoding

Formula for position pos, dimension i:  
  PE(pos, 2i)   \= sin(pos / 10000^(2i/d\_model))  
  PE(pos, 2i+1) \= cos(pos / 10000^(2i/d\_model))

Where:  
  \- pos: frame index (0 to T-1)  
  \- i: dimension index (0 to 191 for d\_model=384)  
  \- d\_model: 384 (hidden dimension)

Implementation:  
  features \= features \+ positional\_encoding\[:time\]  
  features \= Dropout(0.1)(features)

Output: (batch, time/2, 384\)  
Parameters: 0 (fixed, not learned)

Alternative (better for sign language):  
  \- Learnable embeddings: nn.Embedding(max\_len=1000, embedding\_dim=384)  
  \- Relative positional bias (learns that adjacent frames are related)

#### **LAYER 5: Conformer Encoder**

**What is Conformer?**

* Hybrid: Convolution (local patterns) \+ Attention (global context)  
* State-of-the-art for sequence modeling (used in Whisper, Wav2Vec2)  
* Better than pure Transformer or pure RNN for sequences

Conformer Block Architecture:  
────────────────────────────────────────────────

1\. Feed-Forward Module (50% expansion)  
   ├─ LayerNorm  
   ├─ Linear(384, 768\)  \# 2x expansion  
   ├─ Swish activation  
   ├─ Dropout(0.1)  
   ├─ Linear(768, 384\)  
   ├─ Dropout(0.1)  
   └─ Residual: output \= 0.5 \* ff(x) \+ x  \# Half-step residual

2\. Multi-Head Self-Attention Module  
   ├─ LayerNorm  
   ├─ Linear self-attention (O(n) complexity)  
   │   \- 4 heads, 96 dims per head  
   │   \- Query, Key, Value projections  
   │   \- Feature map: ELU(x) \+ 1 (makes attention linear)  
   ├─ Dropout(0.1)  
   └─ Residual: output \= attention(x) \+ x

3\. Convolution Module  
   ├─ LayerNorm  
   ├─ Pointwise Conv: 1×1 conv, 384 → 768  
   ├─ GLU activation (Gated Linear Unit)  
   ├─ Depthwise Conv: kernel=7, groups=384  
   │   \- Each channel processed independently  
   │   \- Captures local temporal patterns  
   ├─ BatchNorm  
   ├─ Swish activation  
   ├─ Pointwise Conv: 1×1 conv, 768 → 384  
   ├─ Dropout(0.1)  
   └─ Residual: output \= conv(x) \+ x

4\. Feed-Forward Module (50% expansion)  
   └─ Same as (1)

Total: 2 Conformer blocks stacked  
Parameters: \~4M per block × 2 \= 8M total

**Why Conformer over BiGRU?**

| Aspect | BiGRU | Conformer |
| ----- | ----- | ----- |
| **Parallelization** | Sequential (slow) | Parallel (fast) |
| **Mobile Latency** | 60ms | 25ms |
| **Long-range context** | Limited | Excellent |
| **Training speed** | Slow | 2-3x faster |
| **Accuracy** | Good | \+2-3% better |

#### **LAYER 6A: CTC Head (Streaming Path)**

Purpose: Fast character predictions for real-time display

Architecture:  
  Input: (batch, time/2, 384\)  
  ↓  
  Linear(384, 384\)  
  ReLU  
  Dropout(0.3)  
  ↓  
  Linear(384, vocab\_size)  \# vocab\_size ≈ 75 (letters \+ special tokens)  
  ↓  
  Log-Softmax  
  ↓  
  Output: (batch, time/2, vocab\_size) log probabilities

Loss: CTC Loss  
  \- Blank token: index 0  
  \- Allows repeated characters: "hello" can be "hheelllloo"  
  \- Alignment-free: model learns when to emit characters

Decoding: Greedy or Beam Search  
  1\. For each frame, take argmax  
  2\. Remove consecutive duplicates  
  3\. Remove blank tokens  
    
Example:  
  Frame probs: \[blank, h, h, h, e, blank, l, l, o\]  
  After collapse: \[h, e, l, o\]

Parameters: \~300K

#### **LAYER 6B: GRU Decoder (Fast Attention Path)**

Purpose: Polished output with cross-attention, faster than Transformer

Architecture:  
────────────────────────────────────────────────

Input Embedding:  
  \- Token IDs → Embedding(vocab\_size, 256\)  
  \- Positional encoding

GRU Decoder Layers (2 layers):

Layer Loop:  
  For each target position t:  
    1\. Embed current token: x\_t (256 dims)  
      
    2\. Cross-Attention to encoder:  
       Query: hidden\_state\_t (384 dims)  
       Key, Value: encoder\_output (batch, time/2, 384\)  
         
       Attention mechanism:  
         \- scores \= Query @ Key.T / sqrt(96)  \# 4 heads, 96 per head  
         \- attention \= softmax(scores)  
         \- context \= attention @ Value  \# (384 dims)  
      
    3\. GRU Cell:  
       Input: concat(\[x\_t, context\]) \= (640 dims)  
       hidden\_t \= GRU(input, hidden\_{t-1})  
         
    4\. Output projection:  
       logits \= Linear(concat(\[hidden\_t, context\]), vocab\_size)

Teacher Forcing Schedule:  
  \- Epoch 0-15: 90% → 20% (use ground truth)  
  \- Epoch 15+: 20% (mostly use model predictions)  
  \- Prevents train/test mismatch

Output: (batch, max\_len, vocab\_size)  
Parameters: \~6M

#### **LAYER 6C: Transformer Decoder (Accuracy Path)**

Purpose: Best accuracy, use when latency is not critical

Architecture:  
────────────────────────────────────────────────

2 Transformer Decoder Layers:

Each layer:  
  1\. Masked Self-Attention  
     \- Causal mask: position t can only see ≤ t  
     \- 4 heads, 96 dims per head  
     \- Prevents looking ahead during training  
    
  2\. Cross-Attention to Encoder  
     \- Query: decoder features  
     \- Key, Value: encoder output  
     \- No masking (can attend to all encoder frames)  
    
  3\. Feed-Forward Network  
     \- 384 → 768 → 384  
     \- GELU activation  
    
  Each sublayer has:  
     \- Pre-LayerNorm  
     \- Residual connection  
     \- Dropout(0.3)

Output Projection:  
  Linear(384, vocab\_size)

Output: (batch, max\_len, vocab\_size)  
Parameters: \~8M

### **4.3 Complete Model Summary**

TOTAL MODEL ARCHITECTURE  
═══════════════════════════════════════════════

Input Projection:          160K params  
Multi-Scale CNN (2 blocks): 3M params  
Temporal Subsampling:      150K params  
Positional Encoding:       0 params  
Conformer Encoder (2 blks): 8M params  
CTC Head:                  300K params  
GRU Decoder:               6M params  
Transformer Decoder:       8M params (optional)  
────────────────────────────────────────────────  
TOTAL (GRU only):          \~18M params (\~72 MB FP32, \~18 MB INT8)  
TOTAL (Dual decoder):      \~26M params (\~104 MB FP32, \~26 MB INT8)

Inference Latency (150 frame video):  
  \- Encoder: 80ms  
  \- CTC decode: 15ms (greedy)  
  \- GRU decode: 35ms  
  \- Transformer decode: 60ms

Total latency:  
  \- CTC streaming: \~95ms  
  \- GRU final: \~115ms  
  \- Transformer final: \~140ms

---

## **5\. TRAINING STRATEGY**

### **5.1 Loss Function**

**Hybrid Loss \= λ × CTC\_Loss \+ (1-λ) × Attention\_Loss**

CTC Loss:  
────────────────────────────────────────────────  
Purpose: Train encoder to produce good character sequences  
Function: nn.CTCLoss(blank=0, reduction='mean', zero\_infinity=True)

Inputs:  
  \- log\_probs: (time, batch, vocab\_size) from CTC head  
  \- targets: (batch, target\_len) character indices  
  \- input\_lengths: (batch,) encoder output lengths  
  \- target\_lengths: (batch,) text lengths

Key property: Alignment-free  
  \- Model learns when to emit characters  
  \- Can handle variable speed signing

Attention Loss (Cross-Entropy):  
────────────────────────────────────────────────  
Purpose: Train decoder to generate fluent text  
Function: LabelSmoothingCrossEntropy(smoothing=0.1)

Inputs:  
  \- predictions: (batch, max\_len-1, vocab\_size)  
  \- targets: (batch, max\_len-1) shifted by 1

Why shifted? Decoder predicts next token:  
  Input:  \[\<SOS\>, 'h', 'e', 'l', 'l'\]  
  Target: \['h', 'e', 'l', 'l', 'o', \<EOS\>\]

Label Smoothing:  
  Instead of: target \= \[0, 0, 1, 0, ...\]  (one-hot)  
  Use:        target \= \[0.02, 0.02, 0.9, 0.02, ...\]  
    
  Effect: Prevents overconfidence, better generalization

Dual Decoder Loss (if using both GRU \+ Transformer):  
────────────────────────────────────────────────  
Total \= λ × CTC \+ (1-λ) × (0.6 × GRU\_CE \+ 0.4 × Transformer\_CE)

Why weighted? GRU is primary for mobile deployment

### **5.2 Loss Weight Schedule (λ decay)**

**Problem:** CTC and Attention losses have different scales **Solution:** Dynamic weighting that changes during training

CTC Weight Schedule:  
────────────────────────────────────────────────  
Epoch 0-30: λ starts at 0.3, decays to 0.1  
Epoch 30+:  λ \= 0.1 (fixed)

Formula:  
  if epoch \< 30:  
    λ \= 0.3 \- (0.3 \- 0.1) × (epoch / 30\)  
  else:  
    λ \= 0.1

Rationale:  
  \- Early: CTC helps encoder learn good features (30% weight)  
  \- Late: Focus on attention for fluent text (10% weight)  
  \- CTC becomes auxiliary supervision

### **5.3 Teacher Forcing Schedule**

**What is teacher forcing?**

* During training, feed ground truth tokens to decoder  
* Prevents error accumulation  
* But creates train/test mismatch\!

Teacher Forcing Schedule:  
────────────────────────────────────────────────  
Epoch 0-15: Ratio decays from 0.9 → 0.2  
Epoch 15+:  Ratio \= 0.2 (fixed)

Formula:  
  if epoch \< 15:  
    tf\_ratio \= 0.9 \- (0.9 \- 0.2) × (epoch / 15\)  
  else:  
    tf\_ratio \= 0.2

At each decoder step with probability tf\_ratio:  
  Use ground truth token  
Else:  
  Use model's previous prediction

Example (tf\_ratio=0.5):  
  Step 1: Input \<SOS\>, predict 'h' → USE 'h' (50% chance)  
  Step 2: Input 'e' (ground truth, not 'h'), predict 'l'  
  Step 3: Input 'l' (model prediction), predict 'l'

### **5.4 Training Phases**

PHASE 1: Encoder Pre-training (Optional, 5-10 epochs)  
────────────────────────────────────────────────  
Train only: Encoder \+ CTC head  
Freeze: Decoders  
Loss: CTC only (λ \= 1.0)

Benefits:  
  \- Encoder learns good features first  
  \- Faster initial convergence  
  \- Better initialization for phase 2

PHASE 2: Full Training (60-80 epochs)  
────────────────────────────────────────────────  
Train: Entire model  
Loss: Hybrid (λ schedule as above)  
Teacher forcing: Schedule as above

Epochs 0-20: Focus on convergence  
Epochs 20-40: Refinement, accuracy improves  
Epochs 40-60: Fine-tuning, diminishing returns  
Epochs 60+: Risk of overfitting

PHASE 3: Fine-tuning (Optional, 5-10 epochs)  
────────────────────────────────────────────────  
Very low learning rate: 1e-6  
Minimal augmentation  
No teacher forcing  
Goal: Polish final performance

### **5.5 Data Augmentation Strategy**

**Purpose:** Prevent overfitting, improve generalization

Temporal Augmentations:  
────────────────────────────────────────────────  
1\. Time Warping (Prob: 0.5)  
   \- Speed range: 0.85x \- 1.15x  
   \- Simulates different signing speeds  
   \- Implementation: Interpolate to new length

2\. Frame Dropout (Prob: 0.3)  
   \- Drop rate: 10% of frames  
   \- Simulates camera drops, occlusion  
   \- Random frames removed

3\. Time Masking (Prob: 0.3)  
   \- Mask length: up to 20 consecutive frames  
   \- Set to zero (SpecAugment style)  
   \- Forces model to use context

Spatial Augmentations:  
────────────────────────────────────────────────  
4\. Gaussian Noise (Prob: 0.5)  
   \- Std: 0.02 (2% of normalized range)  
   \- Simulates MediaPipe jitter

5\. Random Scaling (Prob: 0.3)  
   \- Scale range: 0.9 \- 1.1  
   \- Simulates distance variation

6\. Random Rotation (Prob: 0.2)  
   \- Rotation range: ±10 degrees (in XY plane)  
   \- Simulates camera angle variation

Landmark-Specific Augmentations:  
────────────────────────────────────────────────  
7\. Hand Swap (Prob: 0.1)  
   \- Swap left/right hands  
   \- For symmetric signs only  
   \- Must flip X coordinates\!

8\. Landmark Dropout (Prob: 0.2)  
   \- Drop rate: 5% of landmarks  
   \- Simulates partial occlusion

Augmentation Intensity:  
  \- Light: 50% of augmentation probabilities  
  \- Medium: As above (recommended)  
  \- Heavy: 150% of probabilities (risk of too much noise)

---

## **6\. HYPERPARAMETERS**

### **6.1 Model Hyperparameters**

ARCHITECTURE  
═══════════════════════════════════════════════  
input\_dim: 414  
  \# 138 position \+ 138 velocity \+ 138 acceleration

hidden\_dim: 384  
  \# Balance between accuracy and speed  
  \# Options: 256 (mobile), 384 (balanced), 512 (accuracy)

embedding\_dim: 256  
  \# Decoder token embeddings

encoder\_layers:   
  conformer\_blocks: 2  
  \# More blocks \= better accuracy but slower  
  \# 2 blocks is sweet spot for mobile

decoder\_layers:  
  gru\_layers: 2  
  transformer\_layers: 2  
  \# Both use 2 layers for consistency

num\_attention\_heads: 4  
  \# Hidden\_dim must be divisible by num\_heads  
  \# 384 / 4 \= 96 dims per head

dropout: 0.4  
  \# CRITICAL: ISign is small dataset, need strong regularization  
  \# Higher dropout (0.4) prevents overfitting  
  \# Apply after: embeddings, attention, FF, decoder

subsample\_factor: 2  
  \# Temporal downsampling: T → T/2  
  \# Options: 1 (no subsample), 2 (2x faster), 4 (4x faster)

### **6.2 Training Hyperparameters**

OPTIMIZER  
═══════════════════════════════════════════════  
optimizer: AdamW  
learning\_rate: 5e-4  
  \# REDUCED from typical 1e-3 for stability  
  \# Small dataset needs careful tuning

weight\_decay: 1e-4  
  \# L2 regularization  
  \# INCREASED from 1e-5 to combat overfitting

betas: (0.9, 0.999)  
  \# Adam momentum parameters

min\_lr: 1e-6

# **Minimum for cosine annealing**

LEARNING RATE SCHEDULE ═══════════════════════════════════════════════ Type: Cosine Annealing with Warm Restarts

warmup\_epochs: 5

# **Linear warmup: 0 → 5e-4 over 5 epochs**

# **Prevents early training instability**

T\_0: 10 epochs

# **First restart period**

T\_mult: 2

# **Double period after each restart**

# **Cycles: 10, 20, 40, 80 epochs**

Formula: epoch\_in\_cycle \= epoch % current\_period lr \= min\_lr \+ (max\_lr \- min\_lr) × (1 \+ cos(π × epoch\_in\_cycle / current\_period)) / 2

BATCH SIZE ═══════════════════════════════════════════════ batch\_size: 12

# **With gradient checkpointing on RTX 4060 (8GB)**

# **Without checkpointing: 6**

gradient\_accumulation: 2

# **Effective batch \= 12 × 2 \= 24**

# **Accumulate gradients over 2 mini-batches before update**

Why gradient accumulation?

* Simulates larger batch size  
* More stable gradients  
* Better generalization

GRADIENT CLIPPING ═══════════════════════════════════════════════ max\_grad\_norm: 1.0

# **Clip gradient L2 norm to prevent explosion**

# **Especially important for RNN/GRU layers**

MIXED PRECISION (FP16) ═══════════════════════════════════════════════ use\_amp: True

# **Automatic Mixed Precision**

# **FP16 for speed, FP32 for stability**

# **\~2x faster, \~0.5x memory**

gradient\_scaler: GradScaler(enabled=True)

# **Scales loss to prevent FP16 underflow**

\#\#\# 6.3 Regularization Hyperparameters

\`\`\`yaml  
DROPOUT RATES  
═══════════════════════════════════════════════  
input\_dropout: 0.2  
  \# After input projection

cnn\_dropout: 0.1  
  \# After each CNN block

encoder\_dropout: 0.4  
  \# In Conformer blocks (attention, FF)

decoder\_dropout: 0.3  
  \# In GRU/Transformer decoder

LABEL SMOOTHING  
═══════════════════════════════════════════════  
smoothing: 0.1  
  \# Softens one-hot targets  
  \# Prevents overconfidence

DATA AUGMENTATION  
═══════════════════════════════════════════════  
use\_augmentation: True  
intensity: 'medium'  
  \# See Section 5.5 for details

EARLY STOPPING  
═══════════════════════════════════════════════  
patience: 15 epochs  
  \# Stop if no improvement for 15 epochs

min\_delta: 0.001  
  \# Minimum improvement to reset patience

### **6.4 Dataset Hyperparameters**

SEQUENCE LENGTHS  
═══════════════════════════════════════════════  
max\_src\_len: 500 frames  
  \# Maximum video length  
  \# 500 frames \= 16.7 seconds at 30 FPS  
  \# Longer videos are truncated

max\_tgt\_len: 100 characters  
  \# Maximum text length  
  \# ISL signs are typically 10-30 chars  
  \# Longer texts are truncated

min\_src\_len: 4 frames  
  \# Minimum after augmentation  
  \# Ensures CTC has enough frames

VOCABULARY  
═══════════════════════════════════════════════  
Special tokens:  
  \<pad\>: 0  \# Padding, also CTC blank  
  \<sos\>: 1  \# Start of sequence  
  \<eos\>: 2  \# End of sequence  
  \<unk\>: 3  \# Unknown character  
  ' ': 4    \# Space  
    
Characters: a-z (lowercase)  
  5-30: 26 letters

Punctuation: .,\!?  
  31-34: 4 punctuation marks

Total vocab\_size: \~75 tokens

Why lowercase?  
  \- Reduces vocabulary size  
  \- ISL doesn't distinguish upper/lower in signs  
  \- Can post-process with capitalization rules

TRAIN/VAL SPLIT  
═══════════════════════════════════════════════  
Method: Stratified by signer (if metadata available)  
  \- Ensures different signers in train/val  
  \- Tests generalization to new signers

Ratios:  
  train: 85%  
  val: 15%  
    
If using subset (for GPU constraints):  
  subset\_ratio: 0.4 (40% of full dataset)  
  Then split: 85% train, 15% val of subset

Seed: 42 (for reproducibility)

---

## **7\. IMPLEMENTATION ROADMAP**

### **7.1 Week-by-Week Plan**

WEEK 1-2: Data Preprocessing  
═══════════════════════════════════════════════  
Day 1-2: MediaPipe Extraction  
  \- Extract landmarks from all videos  
  \- Save as .npy fileshttps://docs.google.com/document/d/147KBTmJNt1KL94G96w1lUVWSkrBCiiRQL0oxQ5Gm2Is/edit?usp=sharing

if the above note is not opening

  \- Check for corrupted videos

Day 3-4: Feature Engineering  
  \- Implement normalization  
  \- Compute velocity, acceleration  
  \- Apply smoothing

Day 5-7: Preprocessing Pipeline  
  \- Batch process all videos  
  \- Verify output dimensions (T, 414\)  
  \- Split train/val sets

Deliverable: Preprocessed dataset ready

WEEK 3-4: Model Implementation  
═══════════════════════════════════════════════  
Day 1-3: Encoder  
  \- Input projection  
  \- Multi-scale CNN with SE blocks  
  \- Temporal subsampling  
  \- Conformer encoder

Day 4-5: CTC Head  
  \- Projection layer  
  \- CTC loss implementation  
  \- Test with dummy data

Day 6-7: GRU Decoder  
  \- Cross-attention mechanism  
  \- GRU layers  
  \- Output projection

Deliverable: Complete model (forward pass works)

WEEK 5: Training Setup  
═══════════════════════════════════════════════  
Day 1-2: Dataset & DataLoader  
  \- Implement SignLanguageDataset  
  \- Collate function with padding  
  \- Test data loading

Day 3-4: Loss Functions  
  \- Hybrid loss  
  \- Label smoothing  
  \- Loss weight scheduling

Day 5-7: Training Loop  
  \- Optimizer setup  
  \- Learning rate scheduling  
  \- Checkpointing  
  \- Logging

Deliverable: Training script ready

WEEK 6-8: Training Phase 1 (Baseline)  
═══════════════════════════════════════════════  
Goal: Get a working baseline

Config:  
  \- No augmentation  
  \- Smaller hidden\_dim: 256  
  \- epochs: 40  
  \- Expect: 70-75% accuracy

Monitor:  
  \- Training/val loss curves  
  \- Overfitting (val loss increases)  
  \- Sample predictions

Adjustments:  
  \- If overfitting: increase dropout  
  \- If underfitting: increase capacity

WEEK 9-10: Training Phase 2 (Full Model)  
═══════════════════════════════════════════════  
Goal: Maximize accuracy

Config:  
  \- Full augmentation (medium intensity)  
  \- hidden\_dim: 384  
  \- epochs: 80  
  \- Dual decoder training

Expect: 82-88% accuracy

WEEK 11-12: Optimization & Mobile Export  
═══════════════════════════════════════════════  
1\. Quantization  
   \- INT8 quantization  
   \- Test accuracy drop (should be \<2%)

2\. Export  
   \- Convert to TFLite (Android)  
   \- Convert to CoreML (iOS)

3\. Mobile Testing  
   \- Latency profiling  
   \- Memory usage  
   \- Battery impact

Deliverable: Optimized mobile model

### **7.2 Critical Checkpoints**

Checkpoint 1: Data Ready (End of Week 2\)  
───────────────────────────────────────────────  
Verify:  
  ✓ All videos processed  
  ✓ Feature dimensions correct (T, 414\)  
  ✓ No NaN/Inf values  
  ✓ Reasonable sequence lengths (30-150 frames typical)

Checkpoint 2: Model Forward Pass (End of Week 4\)  
───────────────────────────────────────────────  
Verify:  
  ✓ Input (4, 100, 414\) → CTC output (4, 50, 75\)  
  ✓ Decoder output (4, max\_len, 75\)  
  ✓ No shape mismatches  
  ✓ Parameters count \~18M

Checkpoint 3: Training Converges (Week 6\)  
───────────────────────────────────────────────  
Verify:  
  ✓ Loss decreasing  
  ✓ Validation accuracy \> 50%  
  ✓ No NaN losses  
  ✓ Model saves/loads correctly

Checkpoint 4: Baseline Achieved (Week 8\)  
───────────────────────────────────────────────  
Verify:  
  ✓ Validation accuracy ≥ 70%  
  ✓ Can generate some correct translations  
  ✓ Overfitting under control

Checkpoint 5: Production Ready (Week 12\)  
───────────────────────────────────────────────  
Verify:  
  ✓ Validation accuracy ≥ 82%  
  ✓ Mobile latency \< 200ms  
  ✓ Model size \< 20MB  
  ✓ Ready for app integration

### **7.3 Debugging Strategy**

Common Issues & Solutions:  
═══════════════════════════════════════════════

Issue: NaN Loss  
Solution:  
  \- Check input for NaN/Inf (add assertions)  
  \- Reduce learning rate (try 1e-4)  
  \- Add gradient clipping (max\_norm=0.5)  
  \- Check CTC input/target length constraints

Issue: Overfitting (val loss increases)  
Solution:  
  \- Increase dropout (try 0.5)  
  \- Add more augmentation  
  \- Reduce model size  
  \- Use more training data

Issue: Underfitting (both losses high)  
Solution:  
  \- Increase model capacity (hidden\_dim 384→512)  
  \- Train longer  
  \- Reduce regularization  
  \- Check data quality

Issue: Slow Convergence  
Solution:  
  \- Increase learning rate (try 1e-3)  
  \- Reduce batch size for more updates  
  \- Add learning rate warmup  
  \- Check if gradients are too small

Issue: Out of Memory (OOM)  
Solution:  
  \- Reduce batch\_size  
  \- Enable gradient checkpointing  
  \- Reduce max\_src\_len (500→300)  
  \- Use gradient accumulation

---

## **8\. SUCCESS METRICS**

### **8.1 Training Metrics**

Primary Metrics:  
═══════════════════════════════════════════════  
1\. Character Accuracy  
   Formula: correct\_chars / total\_chars × 100%  
   Target: ≥ 85%

2\. Word Error Rate (WER)  
   Formula: (insertions \+ deletions \+ substitutions) / total\_words  
   Target: ≤ 20%

3\. BLEU Score (if have multiple references)  
   Measures n-gram overlap  
   Target: ≥ 40

Secondary Metrics:  
═══════════════════════════════════════════════  
4\. CTC Loss  
   Should decrease to \~0.5-1.0

5\. Attention Loss  
   Should decrease to \~1.0-1.5

6\. Training Speed  
   Target: \< 5 minutes per epoch (on RTX 4060\)

### **8.2 Inference Metrics**

Latency Targets:  
═══════════════════════════════════════════════  
Encoder: \< 100ms  
CTC decode: \< 20ms  
GRU decode: \< 50ms  
Total: \< 170ms (for real-time feel)

Model Size Targets:  
═══════════════════════════════════════════════  
FP32: \< 75 MB  
INT8: \< 20 MB  
    
Mobile Requirements:  
═══════════════════════════════════════════════  
Frame rate: \> 30 FPS  
Memory: \< 500 MB RAM  
Battery: \< 5% drain per minute

---

## **9\. FINAL ARCHITECTURE SUMMARY**

INPUT VIDEO (3-5 sec, 30 FPS)  
    ↓  
MediaPipe: Extract 75 landmarks → (T, 75, 3\)  
    ↓  
Select hands \+ body: → (T, 46, 3\) \= (T, 138\)  
    ↓  
Normalize (wrist-centered, scale by hand span)  
    ↓  
Compute velocity \+ acceleration → (T, 414\)  
    ↓  
═══════════════════════════════════════════════  
              ENCODER (Shared)  
───────────────────────────────────────────────  
Input Projection: 414 → 384  
    ↓  
Multi-Scale CNN (2 blocks)  
    \- Kernels: 3, 5, 7, 11  
    \- SE attention  
    \- Residual connections  
    ↓  
Temporal Subsample: T → T/2  
    ↓  
Positional Encoding (sinusoidal)  
    ↓  
Conformer Encoder (2 blocks)  
    \- Multi-head self-attention (4 heads)  
    \- Convolution module  
    \- Feed-forward networks  
    ↓  
Hidden: (batch, T/2, 384\)  
═══════════════════════════════════════════════  
              OUTPUT HEADS  
───────────────────────────────────────────────  
Path 1: CTC Head → Streaming predictions  
    \- Linear: 384 → vocab\_size  
    \- Log-softmax  
    \- Latency: \~95ms total

Path 2: GRU Decoder → Fast final output  
    \- Cross-attention to encoder  
    \- 2 GRU layers  
    \- Latency: \~115ms total

Path 3: Transformer Decoder → Best accuracy  
    \- 2 Transformer layers  
    \- Self \+ cross attention  
    \- Latency: \~140ms total  
    ↓  
ENGLISH TEXT OUTPUT

**This is your complete architecture. No code, just pure design. Use this as your blueprint for implementation. Every component has a clear purpose, and all hyperparameters are justified.**

**Happens in FP16**

**Solution: \- Clamp logits: logits.clamp(-100, 100\) \- Add this before all softmax/log\_softmax \- Check in CTC head and decoder output**

**Cause 4: Division by Zero Diagnosis: \- Check for zero-length sequences \- Check for all-padding batches**

**Solution: \- Add epsilon in attention: scores / (sqrt(d) \+ 1e-6) \- Handle empty batches: if total\_chars \== 0: skip \- Validate data: no empty labels**

**ISSUE 2: OVERFITTING ═══════════════════════════════════════════════**

**Symptoms:**

* **Train loss decreases**  
* **Val loss increases or plateaus**  
* **Train acc \>\> Val acc (gap \> 15%)**

**Causes & Solutions: ───────────────────────────────────────────────**

**Cause 1: Insufficient Regularization Solution: ✓ Increase dropout: 0.3 → 0.4 → 0.5 ✓ Increase weight decay: 1e-5 → 1e-4 ✓ Add label smoothing: 0.1 → 0.15**

**Cause 2: No Data Augmentation Solution: ✓ Enable augmentation: use\_augmentation=True ✓ Increase intensity: 'light' → 'medium' ✓ Verify augmentation is active (print sample shapes)**

**Cause 3: Model Too Large Solution: ✓ Reduce hidden\_dim: 384 → 256 ✓ Reduce layers: encoder\_layers=3 → 2 ✓ Use fewer attention heads: 4 → 2**

**Cause 4: Training Too Long Solution: ✓ Use best model (lowest val\_loss), not last ✓ Enable early stopping: patience=15 ✓ Stop when val\_loss increases for 10+ epochs**

**ISSUE 3: SLOW CONVERGENCE ═══════════════════════════════════════════════**

**Symptoms:**

* **Loss decreases very slowly**  
* **Accuracy stuck at low values (\< 50%)**  
* **Takes 50+ epochs to see progress**

**Causes & Solutions: ───────────────────────────────────────────────**

**Cause 1: Learning Rate Too Low Diagnosis: \- Check current LR: optimizer.param\_groups\[0\]\['lr'\] \- If LR \< 1e-5 in early epochs: too low**

**Solution: ✓ Increase initial LR: 5e-4 → 1e-3 ✓ Add warmup: 5 epochs 0 → max\_lr ✓ Use larger batch size (with gradient accumulation)**

**Cause 2: Poor Initialization Solution: ✓ Use Xavier/Kaiming init: already in model ✓ Check weight magnitudes: should be \~0.1-1.0 ✓ Add batch normalization: already in CNN**

**Cause 3: Too Much Regularization Diagnosis: \- Both train and val loss high \- Underfitting**

**Solution: ✓ Reduce dropout: 0.4 → 0.2 ✓ Reduce weight decay: 1e-4 → 1e-5 ✓ Reduce augmentation intensity**

**Cause 4: Bad Data Quality Diagnosis: \- Inspect preprocessed samples \- Check for excessive zeros, NaN**

**Solution: ✓ Fix preprocessing pipeline ✓ Improve landmark extraction quality ✓ Filter corrupt samples**

**ISSUE 4: OUT OF MEMORY ═══════════════════════════════════════════════**

**Symptoms:**

* **CUDA out of memory error**  
* **Training crashes after N batches**

**Solutions (in order of preference): ───────────────────────────────────────────────**

1. **Enable Gradient Checkpointing use\_gradient\_checkpointing=True → Saves \~40% memory, allows batch\_size × 2**

2. **Reduce Batch Size batch\_size: 12 → 6 gradient\_accumulation: 2 → 4 (keeps effective batch \= 24\)**

3. **Reduce Sequence Length max\_src\_len: 500 → 300 → Shorter sequences \= less memory**

4. **Use FP16 Training use\_amp=True → \~50% memory reduction**

5. **Reduce Model Size hidden\_dim: 384 → 256 encoder\_layers: 3 → 2**

6. **Clear Cache Periodically After validation: torch.cuda.empty\_cache()**

**\#\#\# 12.2 Validation Strategies**

**\`\`\`yaml**

**VALIDATING MODEL QUALITY DURING TRAINING**

**═══════════════════════════════════════════════**

**QUANTITATIVE METRICS:**

**───────────────────────────────────────────────**

**1\. Character Accuracy**

   **Formula: correct\_chars / total\_chars**

   

   **Milestones:**

     **Epoch 5:  \> 30% (learning started)**

     **Epoch 20: \> 60% (reasonable model)**

     **Epoch 50: \> 80% (good model)**

     **Epoch 80: \> 85% (production ready)**

**2\. Word Error Rate (WER)**

   **Formula: (insertions \+ deletions \+ substitutions) / num\_words**

   

   **Target: \< 20%**

   

   **Better than character accuracy for practical use**

   **One wrong character ≠ one wrong word**

**3\. Loss Values**

   **CTC Loss:**

     **Should decrease to: 0.5 \- 1.5**

     **If stuck \> 3.0: model not learning CTC**

   

   **Attention Loss:**

     **Should decrease to: 1.0 \- 2.0**

     **If stuck \> 4.0: decoder not learning**

**QUALITATIVE VALIDATION:**

**───────────────────────────────────────────────**

**Every 5-10 epochs, manually inspect predictions:**

**1\. Sample 5-10 validation videos**

**2\. Print:**

   **\- Ground truth text**

   **\- Model CTC prediction**

   **\- Model attention prediction (GRU)**

   **\- Model attention prediction (Transformer)**

**Look for:**

  **✓ Correct words appearing**

  **✓ Sensible mistakes (similar letters)**

  **✓ Improvement over epochs**

**Red flags:**

  **✗ Random garbage output**

  **✗ Repeated characters (aaaaaaa)**

  **✗ Same prediction for all inputs**

  **✗ Empty predictions**

**MONITORING TRAINING HEALTH:**

**───────────────────────────────────────────────**

**Plot curves (use tensorboard or matplotlib):**

**1\. Loss Curves**

   **\- Train loss (smooth decreasing)**

   **\- Val loss (decreasing, may plateau)**

   

   **Healthy: Both decrease, val slightly higher than train**

   

   **Overfitting: Train keeps decreasing, val increases**

   

   **Underfitting: Both high and flat**

**2\. Accuracy Curves**

   **\- Train accuracy (increasing)**

   **\- Val accuracy (increasing)**

   

   **Healthy: Both increase, gap \< 10%**

   

   **Overfitting: Large gap (\> 20%)**

**3\. Learning Rate Schedule**

   **\- Should follow cosine annealing**

   **\- Restarts visible as sudden jumps**

**4\. Gradient Norms**

   **\- Should be stable (0.1 \- 10.0)**

   **\- Spikes \> 100: gradient explosion**

   **\- Consistently \< 0.01: vanishing gradients**

---

**This completes the full Low-Level Design. You now have:**

1. **✅ Complete data preprocessing pipeline**  
2. **✅ Detailed model architecture layer-by-layer**  
3. **✅ Training strategy with all hyperparameters**  
4. **✅ Loss functions and optimization details**  
5. **✅ Debugging guide for common issues**  
6. **✅ Validation and monitoring strategies**

**Next step: Start implementation following Week 1-2 of the roadmap (MediaPipe extraction). Do you want me to elaborate on any specific section?**

