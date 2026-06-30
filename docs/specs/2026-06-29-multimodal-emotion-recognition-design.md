# Design Spec: 基于深度学习的多模态情感识别模型

- **Date**: 2026-06-29
- **Status**: Approved
- **Timeline**: 1-3 days (defense: 2026-06-30 ~ 2026-07-02)
- **GPU**: NVIDIA ≥8GB VRAM, CUDA 12.1

---

## 1. Overview

Build an end-to-end multimodal emotion recognition system using PyTorch. Three modalities — **text**, **facial image**, and **speech audio** — are fused via cross-modal attention to classify 6–8 emotion categories. A Streamlit web interface provides real-time inference. This is a university deep-learning course capstone project; everything is built from scratch using `nn.Module` subclasses, following the existing `Trainer` class pattern from prior experiments.

## 2. Architecture

Seven-layer pipeline:

```
L1 Data → L2 Preprocessing → L3 Model → L4 Training → L5 Inference → L6 Frontend → L7 Deploy
```

### 2.1 Layer details

| Layer | Responsibility | Input → Output |
|-------|---------------|-----------------|
| **L1 Data** | Dataset acquisition, indexing, train/val/test split | Raw files → train.csv / val.csv / test.csv |
| **L2 Preprocessing** | Tokenization, face detection+crop, MFCC extraction, augmentation | Raw files → [B, max_len] / [B,1,48,48] / [B,40,T] tensors |
| **L3 Model** | Three modality encoders + cross-modal attention fusion + classifier | Three tensors → emotion logits [B, num_classes] |
| **L4 Training** | Trainer class, AMP, AdamW, early stopping, checkpointing | Model params → best_model.pth |
| **L5 Inference** | Single/batch forward pass, missing-modality fallback | Raw input → emotion label + confidence |
| **L6 Frontend** | Streamlit UI: text input, image upload, audio upload, real-time prediction | User interaction → visualized results |
| **L7 Deploy** | `streamlit run app.py` on localhost:8501 | Browser-accessible demo |

### 2.2 Data flow (training)

```
Raw dataset → Preprocess → save .pt tensors → Dataset.__getitem__ → DataLoader → Model.forward → loss.backward → optimizer.step → checkpoint
```

### 2.3 Data flow (inference)

```
User uploads (text + image + audio) → Preprocess pipeline → Model.eval() forward → Softmax → Top-K predictions → Streamlit UI render
```

## 3. Dataset Strategy

**Plan A (preferred)**: **CMU-MOSEI** — 23,453 video segments, text+face+audio naturally aligned, 6 Ekman emotions + intensity. Download from CMU MultiComp Lab.

**Plan B (fallback, instant download)**: Three independent datasets aligned by shared emotion labels:
- Text: **GoEmotions** (58K Reddit comments, HuggingFace) or **ISEAR** (7.5K, GitHub)
- Image: **FER2013** (35,887 48×48 grayscale faces, Kaggle, ~90 MB)
- Audio: **RAVDESS speech-only** (1,440 clips, Kaggle/Zenodo)

Both plans target 6–8 emotion classes mapped to a common label set: `{happy, sad, angry, neutral, fearful, disgust, surprised}`.

### 3.1 Preprocessing

| Modality | Steps |
|----------|-------|
| Text | Lowercase → tokenize → build vocab (top 5000) → pad/truncate to max_len=64 → embedding lookup |
| Image | MTCNN face detect → crop → resize 48×48 → normalize (μ=0.5, σ=0.5) → RandomHorizontalFlip(p=0.5) |
| Audio | Resample 16kHz mono → 40-dim MFCC (win=25ms, hop=10ms) → normalize → SpecAugment (time/freq mask) |

### 3.2 Split

- Train: 70%, Val: 15%, Test: 15% (stratified by emotion class)

## 4. Model: MultiModalEmotionNet

### 4.1 Encoders

```
TextEncoder:
  Embedding(vocab=5000, dim=256)
  → BiLSTM(input=256, hidden=256, layers=2, dropout=0.3)
  → last hidden concat → h_text [B, 256]

ImageEncoder:
  Conv2d(1, 32, k3, s1, p1) → BN → ReLU → MaxPool(2)
  → Conv2d(32, 64, k3, s1, p1) → BN → ReLU → MaxPool(2)
  → Conv2d(64, 128, k3, s1, p1) → BN → ReLU → MaxPool(2)
  → Conv2d(128, 256, k3, s1, p1) → BN → ReLU → AdaptiveAvgPool(1)
  → h_img [B, 256]

AudioEncoder:
  Conv1d(40, 64, k3) → BN → ReLU → MaxPool(2)
  → Conv1d(64, 128, k3) → BN → ReLU → MaxPool(2)
  → BiLSTM(input=128, hidden=128, layers=1, dropout=0.3)
  → last hidden concat → Linear(256→256) → h_audio [B, 256]
```

### 4.2 Fusion

```
CrossModalAttentionFusion:
  concat([h_text, h_img, h_audio]) → [B, 3, 256]
  → MultiHeadSelfAttention(d_model=256, heads=4) + Residual + LayerNorm
  → Flatten → Linear(768 → 512) → ReLU → Dropout(0.5)
  → h_fused [B, 512]
```

### 4.3 Classifier

```
Linear(512 → 256) → ReLU → Dropout(0.5) → Linear(256 → num_classes) → Softmax
```

Total parameters: ~2–3M (lightweight, fits easily in 8GB VRAM).

### 4.4 Missing modality handling

During training, randomly drop one modality (p=0.15) and replace with zero vector. This enables the model to handle single/dual-modality inference gracefully.

## 5. Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 → 1e-6 | CosineAnnealingWarmRestarts, T_0=10, T_mult=2 |
| Batch size | 32 (effective 64) | Gradient accumulation steps=2 |
| Epochs | 30 max | EarlyStopping patience=8, monitor val_loss |
| Optimizer | AdamW | β₁=0.9, β₂=0.999, weight_decay=1e-4 |
| Dropout | 0.3 (encoders) / 0.5 (fusion) | — |
| Label smoothing | 0.1 | CrossEntropyLoss with smoothing |
| Mixed precision | AMP (FP16) | `torch.cuda.amp.autocast()` |
| Gradient clipping | max_norm=1.0 | — |
| Embedding dim | 256 | Uniform across all encoders |
| Fusion dim | 512 | After attention fusion |
| Vocab size | 5000 | For text tokenizer |
| Image size | 48×48 grayscale | Standard FER2013 size |
| Audio MFCC | 40 dims | 25ms window, 10ms hop |
| Max text len | 64 tokens | Pad/truncate |

## 6. CUDA Environment

```bash
# Setup
conda create -n emotion python=3.10 -y
conda activate emotion
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

### 6.1 Memory optimization

1. **AMP** — autocast to FP16, saves ~40% VRAM
2. **Gradient accumulation** — steps=2, effective batch=64 without extra memory
3. **`torch.cuda.empty_cache()`** — between epochs
4. **`pin_memory=True`**, `num_workers=4` — faster CPU→GPU transfer
5. **`del` unused tensors** + `empty_cache()` — after validation

### 6.2 Overfitting / Underfitting diagnosis

| Symptom | Train Acc | Val Acc | Fix |
|---------|-----------|---------|-----|
| Overfitting | >> Val Acc | Low | ↑ dropout, ↑ weight_decay, ↑ label_smoothing, ↓ model size |
| Underfitting | Low | Low | ↑ epochs, ↑ lr, ↑ model capacity |
| Diverging | Erratic | Erratic | ↓ lr, add gradient clipping, check data |

## 7. Implementation Steps

### Step 1: Data Preparation
1. Download dataset (CMU-MOSEI or fallback sets)
2. Run `preprocess.py`: face detection, MFCC extraction, text tokenization
3. Build vocabulary, split indices
4. Save preprocessed tensors to `data/`

### Step 2: Model Implementation
1. Write `model.py`: TextEncoder, ImageEncoder, AudioEncoder, Fusion, Classifier
2. Write `dataset.py`: MultiModalDataset returning (text, image, audio, label) tuples
3. Write forward-pass test: dummy batch → verify output shape

### Step 3: Training
1. Write `trainer.py`: Trainer class with AMP, gradient accumulation, early stopping
2. Write `train.py`: entry point, config loading, main loop
3. Run training, monitor with tqdm, save best_model.pth and training history

### Step 4: Evaluation
1. Write `evaluate.py`: test set evaluation, confusion matrix, classification report
2. Run ablation: text-only, image-only, audio-only, text+image, text+audio, image+audio, all-three
3. Generate plots: training curves, confusion matrix, ablation bar chart

### Step 5: Deployment
1. Write `inference.py`: inference interface with missing-modality fallback
2. Write `app.py`: Streamlit UI with text input, image upload, audio upload, real-time prediction
3. Write `README.md`, `requirements.txt`
4. Write course design report document

## 8. Innovation Points

1. **Cross-modal attention fusion** — learn modality interaction weights, visualize attention heatmaps
2. **Missing modality robustness** — random modality dropout during training, graceful degradation at inference
3. **Ablation study** — quantitative comparison of single/dual/tri-modal performance
4. **Model lightweighting** — compact CNN encoders (~2-3M params) suitable for deployment
5. **Comprehensive regularization** — label smoothing + dropout + weight decay + early stopping combo
6. **Multi-dimensional visualization** — confusion matrix + t-SNE feature distribution + attention weights + training curves

## 9. Deliverables

| # | Deliverable | Path |
|---|-------------|------|
| 1 | Trained weights | `models/best_model.pth`, `models/final_model.pth` |
| 2 | Complete source code | `src/config.py`, `src/preprocess.py`, `src/dataset.py`, `src/model.py`, `src/trainer.py`, `src/train.py`, `src/inference.py` |
| 3 | Streamlit app | `app.py` |
| 4 | Classification report | `results/classification_report.txt` |
| 5 | Training curves | `results/training_curves.png` |
| 6 | Confusion matrix | `results/confusion_matrix.png` |
| 7 | Ablation study chart | `results/ablation_study.png` |
| 8 | Dependencies | `requirements.txt` |
| 9 | Documentation | `README.md` |
| 10 | Course report | `课程设计说明书.docx` |

## 10. Project Structure

```
Course Project/
├── data/                    # Preprocessed tensors + raw files
├── models/                  # .pth checkpoint files
├── results/                 # Plots and reports
├── src/
│   ├── config.py            # All hyperparameters
│   ├── preprocess.py        # Dataset preprocessing pipeline
│   ├── dataset.py           # torch.utils.data.Dataset
│   ├── model.py             # MultiModalEmotionNet
│   ├── trainer.py           # Trainer class
│   ├── train.py             # Training entry point
│   └── inference.py         # Inference interface
├── app.py                   # Streamlit frontend
├── requirements.txt
├── README.md
└── 课程设计说明书.docx
```
