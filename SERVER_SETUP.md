# ISL Translation - Server Setup Guide

## If You Already Have Extracted .npy Files

Your extraction script saved files to `iSign_videos/landmarks/train/`. 

### Quick Start (3 commands):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate metadata (point to your existing .npy folder)
python generate_metadata.py --landmarks_dir iSign_videos/landmarks/train

# 3. Train!
cd isl_translation
python run_training.py
```

That's it! The script will:
- Read your .npy files
- Match them with annotations from iSign_v1.1.csv
- Split into train/val/test (70/15/15)
- Copy files to `data/landmarks/{train,val,test}/`
- Create `metadata.csv`

---

## Folder Structure

```
ISL_TRANSLATION/
├── iSign_videos/             # Your existing extraction output
│   └── landmarks/
│       └── train/            # Your .npy files are here
│           ├── video_001.npy
│           └── ...
│
├── data/                     # Will be created by script
│   ├── landmarks/
│   │   ├── train/            # .npy files copied here
│   │   ├── val/
│   │   ├── test/
│   │   └── metadata.csv      # Generated
│   └── iSign_v1.1.csv        # Annotations
│
├── isl_translation/          # Training code
├── checkpoints/              # Model saves here
├── generate_metadata.py      # Run this first
└── requirements.txt
```

---

## Full Setup from Scratch

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. If extracting from videos
```bash
# Put videos in data/videos/
python extract_landmarks.py --video_dir data/videos --output_dir data/landmarks/train
```

### 3. Generate Metadata
```bash
python generate_metadata.py
```

### 4. Train Model
```bash
cd isl_translation
python run_training.py
```
